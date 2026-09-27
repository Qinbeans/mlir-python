"""``Module``, ``Program``, and ``Function``: the user-facing objects of
``mlir_python.lang``.

A ``Module`` is a compilation unit, like a C translation unit or library. Its
functions are ordinary Python objects: import them from another module and
call them from compiled code, and the modules link together when something is
run or built. A ``Program`` is a module that can also have an entry point and
build executables.
"""

from __future__ import annotations

import array
import ast
import atexit
import builtins
import ctypes
import itertools
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import CellType
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from .. import _mlir_python as ir
from .. import codegen, passes, pipelines
from .._passes import PassManager, PipelineElement
from ..dialects import func, llvm
from ._abi import ExternABI, UnsupportedABI
from ._abi import lower as lower_abi
from ._compiler import CompileError, FunctionCompiler, Signature, Source, signature_of
from ._types import ScalarType, i32

type Kind = Literal["function", "extern", "main"]

# Every module shares one MLIR context, so modules can be linked by combining
# their operations directly.
_CONTEXTS: list[ir.Context] = []


def _context() -> ir.Context:
    """The context shared by every module (created on first use)."""
    if not _CONTEXTS:
        _CONTEXTS.append(ir.Context())
    return _CONTEXTS[0]


@atexit.register
def _release_context() -> None:
    # Drop the shared context before the extension module is torn down, so
    # nanobind does not report it as leaked at interpreter exit.
    _CONTEXTS.clear()


_MODULE_IDS = itertools.count()

# Run on every compiled module. The compiler emits branches between blocks
# (cf), which express any control flow; these passes then
# 1. fold the SSA plumbing it emits (one block argument per live variable,
#    sign fixes for constant divisors, ...),
# 2. lift the branches to structured control flow (scf.while, scf.if), with
#    break/continue/early return becoming loop-carried flags, and
# 3. turn counted loops (for i in range(...)) into scf.for,
# so later passes (parallelization, vectorization, bufferization) see loops.
_STRUCTURING: list[PipelineElement] = [
    passes.Canonicalizer(),
    passes.CSE(),
    passes.LiftControlFlowToSCF(),
    passes.Canonicalizer(),  # the uplift matches only the simplified loop form
    passes.CSE(),
    passes.UpliftWhileToFor(),
    passes.Canonicalizer(),
]


if TYPE_CHECKING:
    # To type checkers a compiled function is a function value (``Fn``), so
    # it passes wherever an ``Fn[[...], R]`` or a ``ptr`` is expected.
    from .types import Fn as _FunctionValue
else:

    class _FunctionValue[**P, R]:
        pass


class Function[**P, R](_FunctionValue[P, R]):
    """A function of a ``Module``, created by ``@module.function``,
    ``@module.extern``, or ``@program.main``.

    Call it from Python to run it (its module is compiled and linked on first
    use); call it from a compiled function, in any module, to emit a call.
    """

    def __init__(
        self,
        module: Module,
        python: Callable[P, R],
        kind: Kind,
        symbol: str | None = None,
    ) -> None:
        self.module = module
        """The compilation module this function belongs to."""
        self.python = python
        """The decorated Python function (its source is what gets compiled)."""
        self.kind: Kind = kind
        """``"function"``, ``"extern"`` (defined elsewhere, e.g. libc), or ``"main"``."""
        self.name = symbol or ("main" if kind == "main" else python.__name__)
        """The symbol name in the compiled code."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run the compiled function with Python values.

        Raises:
            CompileError: If a module does not compile.
            TypeError: For wrong arguments, or when calling an extern.
        """
        if kwargs:
            raise TypeError(f"{self.name}() takes positional arguments only")
        if self.kind == "extern":
            raise TypeError(
                f"{self.name} is external; call it from a compiled function instead"
            )
        return cast(R, self.module._run(self, args))

    def __repr__(self) -> str:
        return f"<{self.kind} {self.name} of {self.module!r}>"


class NamedExtern:
    """The decorator returned by ``@module.extern(name=...)``."""

    def __init__(self, module: Module, name: str) -> None:
        self.module = module
        self.name = name

    def __call__[**P, R](self, python: Callable[P, R]) -> Function[P, R]:
        return self.module._register(Function(self.module, python, "extern", self.name))


@dataclass
class _Build:
    mlir: ir.Module
    ops: dict[str, func.FuncOp]
    signatures: dict[str, Signature]
    externs: dict[str, ExternABI]  # C functions, called through llvm.call
    strings: dict[str, llvm.GlobalOp]
    imports: dict[str, Module] = field(default_factory=dict)  # symbol -> module
    # C functions the compiled code itself calls (e.g. to report a failed
    # check), declared once per module.
    runtime: dict[str, llvm.FunctionType] = field(default_factory=dict)


class Module:
    """A compilation unit: functions compiled together, which other modules
    can import and call.

    Example::

        # mathlib.py
        mathlib = Module("mathlib")

        @mathlib.function
        def cube(x: i32) -> i32:
            return x * x * x

        # app.py
        from mathlib import cube

        program = Program()

        @program.main
        def main() -> i32:
            return cube(3)          # linked in automatically

    A module's externs may come from libraries, which it names once::

        raylib = Module("raylib", libraries=[Path("lib/libraylib.a"), "m"])

        @raylib.extern
        def GetRandomValue(lo: i32, hi: i32) -> i32: ...
    """

    def __init__(
        self, name: str | None = None, *, libraries: Sequence[codegen.Library] = ()
    ) -> None:
        """Create an empty module.

        Args:
            name: Used in messages; the defining Python module's name by
                default.
            libraries: Libraries defining this module's externs, in link
                order: names (``"m"`` for libm) or paths to ``.so``/``.a``
                files (see ``codegen.Library``). They are loaded for JIT
                calls and linked into anything built from a module that
                imports this one.
        """
        caller = sys._getframe(1).f_globals.get("__name__", "module")
        self.name = name or caller
        """The module's name (for messages)."""
        self.libraries: tuple[codegen.Library, ...] = tuple(libraries)
        """Libraries defining this module's externs."""
        self._id = next(_MODULE_IDS)
        self._functions: dict[str, Function[..., Any]] = {}
        self._build_state: _Build | None = None
        self._version = 0
        self._linked: tuple[tuple[int, ...], ir.Module] | None = None
        self._compiled: tuple[tuple[int, ...], codegen.CompiledModule] | None = None

    # -- defining functions ---------------------------------------------------

    def function[**P, R](self, python: Callable[P, R]) -> Function[P, R]:
        """Compile ``python`` as a function of this module.

        Parameters and the result are typed by annotations (``i32``, ``f64``,
        ``bool``, ``tuple[i32, i32]``, ...; ``int`` means ``i64`` and
        ``float`` means ``f64``). The body is a subset of Python: arithmetic,
        comparisons, ``if``/``while``/``for i in range(...)``, ``break``,
        ``continue``, ``return``, and calls to functions of any module.
        """
        return self._register(Function(self, python, "function"))

    @overload
    def extern[**P, R](self, python: Callable[P, R], /) -> Function[P, R]: ...
    @overload
    def extern(self, /, *, name: str) -> NamedExtern: ...
    def extern(
        self, python: Callable[..., Any] | None = None, /, *, name: str | None = None
    ) -> Function[..., Any] | NamedExtern:
        """Declare a function defined elsewhere, such as C's ``puts``. Its
        body must be ``...``; its annotations give the C signature
        (``cstr`` for ``const char *``, ``*args`` for a variadic function).

        ``@module.extern(name="printf")`` binds a Python name that differs
        from the C symbol.
        """
        if python is not None:
            return self._register(Function(self, python, "extern"))
        if name is None:
            raise TypeError("use @module.extern or @module.extern(name=...)")
        return NamedExtern(self, name)

    def _register[**P, R](self, function: Function[P, R]) -> Function[P, R]:
        if function.name in self._functions:
            raise ValueError(
                f"{self.name} already has a function named '{function.name}'"
            )
        self._functions[function.name] = function
        self._build_state = None
        self._version += 1
        return function

    @property
    def functions(self) -> list[Function[..., Any]]:
        """The module's functions, in definition order."""
        return list(self._functions.values())

    @property
    def imports(self) -> list[Module]:
        """The modules whose functions this module calls."""
        seen: dict[int, Module] = {}
        for module in self._build().imports.values():
            seen.setdefault(id(module), module)
        return list(seen.values())

    # -- results ----------------------------------------------------------------

    @property
    def mlir(self) -> ir.Module:
        """This module alone as MLIR, imports appearing as declarations
        (like an object file). Compiled on first access."""
        return self._build().mlir

    def __str__(self) -> str:
        """This module as MLIR text."""
        return str(self.mlir)

    def linked(self) -> ir.Module:
        """This module and everything it imports, transitively, linked into
        one MLIR module (what runs and gets built), with frees inserted for
        arrays (buffer deallocation sees the whole program).

        Raises:
            codegen.LinkError: If two modules define the same symbol.
        """
        modules = self._closure()
        key = tuple(m._version for m in modules)
        if self._linked is None or self._linked[0] != key:
            with _context():
                linked = _link(modules)
                PassManager(ir.Module, pipelines.buffer_deallocation()).run(linked)
            self._linked = (key, linked)
        return self._linked[1]

    def llvm_ir(
        self, *, opt_level: codegen.OptLevel = codegen.OptLevel.O2
    ) -> codegen.LLVMModule:
        """The linked module as verified LLVM IR; ``print`` it to read it."""
        return codegen.to_llvm_ir(self.linked(), opt_level=opt_level)

    def build_shared_library(
        self,
        output: str | os.PathLike[str],
        *,
        opt_level: codegen.OptLevel = codegen.OptLevel.O2,
        libraries: Sequence[codegen.Library] = (),
        linker: str | None = None,
    ) -> Path:
        """Build a native shared library exporting the functions of this
        module and the modules it imports, linked with their ``libraries``
        and then ``libraries``."""
        return codegen.build_shared_library(
            self.linked(),
            output,
            opt_level=opt_level,
            libraries=[*self._all_libraries(), *libraries],
            linker=linker,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}: {', '.join(self._functions) or 'empty'})"

    # -- compilation ------------------------------------------------------------

    def _build(self) -> _Build:
        if self._build_state is not None:
            return self._build_state
        try:
            with _context(), ir.Location.unknown():
                return self._compile_all()
        except BaseException:
            self._build_state = None
            raise

    def _compile_all(self) -> _Build:
        mlir = ir.Module()
        sources: dict[str, Source] = {}
        state = _Build(mlir, {}, {}, {}, {})
        with ir.InsertionPoint(mlir.body):
            for name, function in self._functions.items():
                source = Source.of(function.python)
                signature = signature_of(
                    function.python, source, allow_variadic=function.kind == "extern"
                )
                sources[name] = source
                state.signatures[name] = signature
                with source.location(source.tree):
                    if function.kind == "extern":
                        _check_extern(signature, source)
                        state.externs[name] = _declare_extern(
                            function.name, signature, source
                        )
                    else:
                        state.ops[name] = self._declare(function, signature)
        # Set before compiling bodies so calls within and across modules
        # (including cycles) see every signature.
        self._build_state = state
        for name, function in self._functions.items():
            if function.kind != "extern":
                FunctionCompiler(
                    self,
                    function,
                    state.ops[name],
                    state.signatures[name],
                    sources[name],
                ).compile()
        mlir.verify()
        PassManager(ir.Module, _STRUCTURING).run(mlir)
        # The passes rewrote the module, so earlier handles are stale; look the
        # functions up again.
        state.ops = {
            op.sym_name: op
            for op in mlir.body.operations
            if isinstance(op, func.FuncOp)
        }
        state.strings = {}
        return state

    def _declare(
        self, function: Function[..., Any], signature: Signature
    ) -> func.FuncOp:
        inputs = [kind.mlir() for _, kind in signature.params]
        results = [kind.mlir() for kind in signature.results]
        if function.kind == "main":
            if signature.params or signature.results not in ([], [i32]):
                source = Source.of(function.python)
                raise source.error(
                    source.tree, "main takes no parameters and returns i32 or None"
                )
            results = [i32.mlir()]
        return func.FuncOp(function.name, ir.FunctionType(inputs, results))

    def _closure(self) -> list[Module]:
        """This module and every module it imports, transitively."""
        order: list[Module] = []
        pending: list[Module] = [self]
        while pending:
            module = pending.pop()
            if any(m is module for m in order):
                continue
            order.append(module)
            pending.extend(module.imports)
        return order

    def _all_libraries(self) -> list[codegen.Library]:
        """The libraries of this module and its imports, each once."""
        libraries: dict[str, codegen.Library] = {}
        for module in self._closure():
            for library in module.libraries:
                key = (
                    library
                    if isinstance(library, str)
                    else str(Path(library).resolve())
                )
                libraries.setdefault(key, library)
        return list(libraries.values())

    def _run(self, function: Function[..., Any], args: Sequence[object]) -> object:
        state = self._build()
        signature = state.signatures[function.name]
        structs = [k for _, k in signature.params] + signature.results
        if any(kind.kind == "struct" for kind in structs):
            raise TypeError(
                f"{function.name}() passes a @struct, which cannot cross from Python "
                "yet; call it from a compiled function"
            )
        modules = self._closure()
        key = tuple(m._version for m in modules)
        if self._compiled is None or self._compiled[0] != key:
            codegen.load_libraries(self._all_libraries())
            _check_externs_resolve(modules)
            self._compiled = (key, codegen.compile(self.linked()))
        linked = self.linked()
        target = next(
            op
            for op in linked.body.operations
            if isinstance(op, func.FuncOp) and op.sym_name == function.name
        )
        buffers = [
            _to_buffer(arg, kind) if kind.kind == "array" else arg
            for arg, (_, kind) in zip(args, signature.params, strict=False)
        ]
        arguments = cast(Sequence[codegen.Argument], buffers)  # checked by the JIT
        # Arrays a function returns were allocated for its caller (buffer
        # deallocation guarantees it), so they are freed once copied.
        compiled = self._compiled[1].function(target, owned_results=True)
        result = compiled(*arguments)
        # Arrays are mutable, as lists are: write changes back to lists.
        for arg, buffer in zip(args, buffers, strict=False):
            if isinstance(arg, list) and buffer is not arg:
                arg[:] = cast(memoryview, buffer).tolist()
        if function.kind == "main" and not signature.results:
            return None
        if signature.returns_tuple and isinstance(result, tuple):
            return tuple(
                _from_buffer(item, kind)
                for item, kind in zip(result, signature.results, strict=True)
            )
        if len(signature.results) == 1:
            return _from_buffer(result, signature.results[0])
        return result

    # -- used by the function compiler ------------------------------------------

    def declaration(
        self, function: Function[..., Any]
    ) -> tuple[func.FuncOp | None, Signature]:
        """The callee and signature of ``function``, importing it when it
        belongs to another module (no ``FuncOp`` for externs, which are
        called through ``llvm.call``; see ``extern_abi``)."""
        state = self._build_state
        assert state is not None
        if function.module is not self and function.name not in state.imports:
            self._import(function, state)
        return state.ops.get(function.name), state.signatures[function.name]

    def _import(self, function: Function[..., Any], state: _Build) -> None:
        """Declares another module's function here, as an external symbol."""
        if function.name in state.signatures:
            raise CompileError(
                f"'{function.name}' is defined in {self.name} and also imported "
                f"from {function.module.name}"
            )
        signature = function.module._build().signatures[function.name]
        with ir.InsertionPoint(state.mlir.body):
            if function.kind == "extern":
                source = Source.of(function.python)
                with source.location(source.tree):
                    state.externs[function.name] = _declare_extern(
                        function.name, signature, source
                    )
            else:
                state.ops[function.name] = func.declare(
                    function.name,
                    [kind.mlir() for _, kind in signature.params],
                    [kind.mlir() for kind in signature.results]
                    or ([i32.mlir()] if function.kind == "main" else []),
                )
        state.signatures[function.name] = signature
        state.imports[function.name] = function.module

    def extern_abi(self, function: Function[..., Any]) -> ExternABI:
        """How calls to the extern ``function`` pass values (the C ABI)."""
        self.declaration(function)
        state = self._build_state
        assert state is not None
        return state.externs[function.name]

    def signature(self, function: Function[..., Any]) -> Signature:
        return function.module._build().signatures[function.name]

    def string_pointer(self, text: str) -> ir.Value:
        """A pointer to a NUL-terminated constant holding ``text``."""
        state = self._build_state
        assert state is not None
        if text not in state.strings:
            with ir.InsertionPoint(state.mlir.body):
                # Unique across modules, so linked modules cannot collide.
                name = f".str.m{self._id}.{len(state.strings)}"
                state.strings[text] = llvm.string_constant(name, text)
        return llvm.address_of(state.strings[text])

    def runtime_function(self, name: str, function_type: llvm.FunctionType) -> None:
        """Declares the C function ``name``, which compiled code calls on its
        own (``write`` and ``abort`` to report a failed check).

        Raises:
            CompileError: If the module declares ``name`` as an extern of
                another type.
        """
        state = self._build_state
        assert state is not None
        if name in state.runtime:
            return
        extern = state.externs.get(name)
        if extern is not None:
            if extern.function_type != function_type:
                raise CompileError(
                    f"'{name}' is declared as an extern of type {extern.function_type}, "
                    f"but compiled code needs it as {function_type}"
                )
        else:
            with ir.InsertionPoint(state.mlir.body), ir.Location.unknown():
                llvm.LLVMFuncOp(name, function_type)
        state.runtime[name] = function_type

    def resolve_global(
        self,
        function: Function[..., Any],
        name: str,
        error: Callable[[], CompileError],
    ) -> object:
        """What ``name`` means in ``function``'s Python scope: its closure,
        then its module's globals, then builtins, as Python resolves it. (Not
        just names the bytecode uses: local annotations such as ``x: i32``
        are never evaluated, so they are not in it.)"""
        python = function.python
        code = getattr(python, "__code__", None)
        cells = cast(tuple[CellType, ...], getattr(python, "__closure__", None) or ())
        if code is not None and name in code.co_freevars:
            try:
                return cells[code.co_freevars.index(name)].cell_contents
            except ValueError:  # an unfilled cell
                raise error() from None
        scope = getattr(python, "__globals__", {})
        if name in scope:
            return scope[name]
        if hasattr(builtins, name):
            return getattr(builtins, name)
        raise error()


class Program(Module):
    """A ``Module`` that can have an entry point and build executables.

    Example::

        program = Program()

        @program.function
        def add(a: i32, b: i32) -> i32:
            return a + b

        @program.main
        def main() -> i32:
            return add(2, 3)

        add(2, 3)                         # 5, via the JIT
        program.build_executable("app")
    """

    def main[**P, R](self, python: Callable[P, R]) -> Function[P, R]:
        """Compile ``python`` as the program's entry point (the C ``main``),
        whatever its Python name. It takes no parameters and returns ``i32``
        (the exit status) or ``None`` (exit status 0)."""
        if any(f.kind == "main" for f in self._functions.values()):
            raise ValueError(f"{self.name} already has a main function")
        return self._register(Function(self, python, "main"))

    def build_executable(
        self,
        output: str | os.PathLike[str],
        *,
        opt_level: codegen.OptLevel = codegen.OptLevel.O2,
        libraries: Sequence[codegen.Library] = (),
        linker: str | None = None,
    ) -> Path:
        """Build a native executable running the ``@program.main`` function,
        linked with every module it imports, their ``libraries``, and then
        ``libraries``.

        Raises:
            ValueError: If the program has no ``@program.main`` function.
            codegen.LinkError: If linking fails.
        """
        if not any(f.kind == "main" for f in self._functions.values()):
            raise ValueError(
                "an executable needs an entry point: decorate one with @program.main"
            )
        return codegen.build_executable(
            self.linked(),
            output,
            opt_level=opt_level,
            libraries=[*self._all_libraries(), *libraries],
            linker=linker,
        )


def _check_extern(signature: Signature, source: Source) -> None:
    """An extern's body is ``...``, and it returns at most one value."""
    body = [
        s
        for s in source.tree.body
        if not (
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Constant)
            and isinstance(s.value.value, str)
        )
    ]
    if not (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and body[0].value.value is ...
    ):
        raise source.error(source.tree, "an extern function's body must be '...'")
    if signature.returns_tuple:
        raise source.error(
            source.tree.returns or source.tree,
            "a C function returns one value; return a @struct to return several",
        )


def _declare_extern(name: str, signature: Signature, source: Source) -> ExternABI:
    """Declares a C function as an ``llvm.func`` with the C ABI's view of its
    parameters (``func.func`` cannot express variadic functions or ABI
    attributes)."""
    try:
        abi = lower_abi(
            [kind for _, kind in signature.params],
            signature.results[0] if signature.results else None,
            variadic=signature.variadic,
        )
    except UnsupportedABI as error:
        raise source.error(source.tree, str(error)) from None
    llvm.LLVMFuncOp(
        name, abi.function_type, arg_attrs=abi.arg_attrs, res_attrs=abi.res_attrs
    )
    return abi


def _symbol(op: ir.Operation) -> tuple[str, bool] | None:
    """(symbol name, is a definition) of a module-level operation."""
    if isinstance(op, (func.FuncOp, llvm.LLVMFuncOp)):
        return op.sym_name, len(op.body) > 0
    if isinstance(op, llvm.GlobalOp):
        return op.sym_name, True
    return None


def _link(modules: list[Module]) -> ir.Module:
    """Combines modules into one: each definition once, declarations only for
    symbols no module defines."""
    with _context(), ir.Location.unknown():
        linked = ir.Module()
        definitions: dict[str, Module] = {}
        declarations: dict[str, ir.Operation] = {}
        for module in modules:
            for op in module.mlir.body.operations:
                symbol = _symbol(op)
                if symbol is None:
                    continue
                name, is_definition = symbol
                if not is_definition:
                    declarations.setdefault(name, op)
                    continue
                if name in definitions:
                    raise codegen.LinkError(
                        f"'{name}' is defined in both {definitions[name].name} "
                        f"and {module.name}"
                    )
                definitions[name] = module
                linked.body.append(op.clone())
        for name, op in declarations.items():
            if name not in definitions:
                linked.body.append(op.clone())
        linked.verify()
        return linked


def _to_buffer(value: object, kind: ScalarType) -> object:
    """A Python list passed as an ``Array[T]``, as a buffer of ``T`` items
    (other buffers, such as NumPy arrays, pass as they are)."""
    if not isinstance(value, list):
        return value
    element = kind.element
    assert element is not None
    if element.kind == "bool":
        buffer = memoryview(bytearray(len(value))).cast("?")
        for index, item in enumerate(value):
            buffer[index] = bool(item)
        return buffer
    if element.kind == "float":
        code = "f" if element.bits == 32 else "d"
    else:
        code = {8: "b", 16: "h", 32: "i", 64: "q"}[element.bits]
        if element.kind == "uint":
            code = code.upper()
    try:
        return memoryview(array.array(code, value))
    except (TypeError, OverflowError) as error:
        raise type(error)(f"cannot pass the list as {kind.name}: {error}") from None


def _from_buffer(value: object, kind: ScalarType) -> object:
    """An ``Array[T]`` result as a Python list."""
    if kind.kind != "array" or not isinstance(value, memoryview):
        return value
    items = value.tolist()
    element = kind.element
    if element is not None and element.kind == "uint":
        # Compiled integers are signless and come back signed; reinterpret.
        mask = (1 << element.bits) - 1
        return [item & mask for item in items]
    return items


def _check_externs_resolve(modules: list[Module]) -> None:
    """The JIT links externs against this process; name any that are missing
    rather than failing with an opaque linker message."""
    process = ctypes.CDLL(None)
    for module in modules:
        for function in module.functions:
            if function.kind == "extern" and not hasattr(process, function.name):
                raise codegen.LinkError(
                    f"extern '{function.name}' of {module.name} is not defined in this "
                    "process or its libraries; check the C name (@module.extern(name=...)) "
                    f"or name its library: {type(module).__name__}(..., libraries=[...])"
                )
