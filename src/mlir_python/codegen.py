"""From MLIR to machine code.

Run a module's functions in this process with ``compile`` (JIT), or build
native binaries with ``build_executable`` and ``build_shared_library``::

    codegen.compile(module).function(add_op)(2, 3)      # the func.FuncOp you built
    codegen.build_executable(module, "hello")           # needs func.func @main
    codegen.build_shared_library(module, "libkernels.so")

All of them work on a copy, so ``module`` and handles into it stay valid.

External code comes in through ``libraries``: a name the system searches for
(``"m"`` for libm) or a ``Path`` to a specific ``.so`` or ``.a`` file::

    codegen.build_executable(module, "game", libraries=[Path("libraylib.a"), "m"])
    codegen.compile(module, libraries=[Path("libkernels.so")])

Each stage is also available on its own: ``llvm_lowering_pipeline`` /
``lower_to_llvm``, ``to_llvm_ir`` / ``translate_to_llvm_ir`` (returning an
``LLVMModule`` to verify, optimize, or emit as an object file or assembly),
and ``ExecutionEngine``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import tempfile
from collections.abc import Buffer, Sequence
from pathlib import Path

from . import passes
from ._mlir_python import (
    ExecutionEngine,
    FunctionType,
    IntegerType,
    LLVMModule,
    Module,
    OptLevel,
    translate_to_llvm_ir,
)
from ._passes import PassManager, PipelineElement
from .dialects import func, llvm

__all__ = [
    "Argument",
    "CompiledFunction",
    "CompiledModule",
    "ExecutionEngine",
    "LLVMModule",
    "Library",
    "LinkError",
    "OptLevel",
    "Result",
    "Scalar",
    "build_executable",
    "build_shared_library",
    "compile",
    "llvm_lowering_pipeline",
    "load_libraries",
    "lower_to_llvm",
    "to_llvm_ir",
    "translate_to_llvm_ir",
]

type Scalar = int | float | bool
"""A value that crosses between Python and compiled code: ``int`` for integer
and index types (``bool`` for ``i1``) and ``float`` for ``f32``/``f64``."""

type Argument = Scalar | Buffer
"""What a compiled function accepts: a ``Scalar``, or for a memref argument a
writable buffer (a NumPy array, ``array.array``, ``memoryview``, ...) with the
memref's element type, rank, and static sizes. The function reads and writes
the buffer in place."""

type Result = Scalar | tuple[Scalar, ...] | None
"""What a compiled function returns: nothing, one value, or a tuple."""

type Library = str | os.PathLike[str]
"""A library providing external functions, in one of two forms:

- a name (``str``) the system searches for: ``"m"`` is libm (``-lm``), and
  ``"raylib"`` finds ``libraylib.so`` or ``libraylib.a`` on the linker's path;
- a path (``pathlib.Path``) to a specific file: a shared library
  (``libfoo.so``) or a static archive (``libfoo.a``).

A path written as a ``str`` is rejected, since it would be searched for as a
name; wrap it in ``Path``.
"""


def _library(library: Library) -> str | Path:
    """A library's name, or its resolved path.

    Raises:
        ValueError: If a ``str`` looks like a path or a linker flag.
        FileNotFoundError: If a path does not exist.
    """
    if isinstance(library, str):
        if "/" in library or library.endswith((".a", ".so")) or ".so." in library:
            raise ValueError(
                f"library {library!r} looks like a file: pass Path({library!r}) "
                "for a file, or a bare name like 'm' to search for libm"
            )
        if not library or library.startswith(("-", ":")):
            raise ValueError(
                f"library {library!r} is not a name: pass 'm' for -lm, "
                "or a Path to a file"
            )
        return library
    path = Path(library).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"library {path} does not exist")
    return path


def _is_archive(path: Path) -> bool:
    with path.open("rb") as file:
        return file.read(8) == b"!<arch>\n"


# Libraries already loaded into this process, so each is loaded once:
# names by name, files by (path, modification time).
_LOADED: dict[str | tuple[Path, int], ctypes.CDLL] = {}


def load_libraries(libraries: Sequence[Library], *, linker: str | None = None) -> None:
    """Load ``libraries`` into this process so JIT-compiled code can call
    their functions (``compile`` does this for its ``libraries``).

    Names are found the way the dynamic loader finds them. A static archive is
    first linked into a temporary shared library, which needs ``linker`` (a C
    compiler driver; ``$CC``, ``cc``, ``clang``, or ``gcc`` by default) and
    objects compiled as position-independent code (``-fPIC``).

    Libraries are loaded last to first, so list them in link order (a library
    before the libraries it depends on), as for ``build_executable``.
    Symbols become visible to the whole process, and stay loaded.

    Raises:
        ValueError: If a ``str`` looks like a path rather than a name.
        FileNotFoundError: If a path does not exist.
        LinkError: If a library cannot be found, linked, or loaded.
    """
    resolved = [_library(library) for library in libraries]
    for library in reversed(resolved):
        if isinstance(library, str):
            if library in _LOADED:
                continue
            found = ctypes.util.find_library(library)
            if found is None:
                raise LinkError(f"library '{library}' (lib{library}.so) not found")
            _LOADED[library] = _dlopen(found, library)
            continue
        key = (library, library.stat().st_mtime_ns)
        if key in _LOADED:
            continue
        if _is_archive(library):
            _LOADED[key] = _load_archive(library, linker)
        else:
            _LOADED[key] = _dlopen(str(library), str(library))


def _dlopen(target: str, shown: str) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(target, mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        raise LinkError(f"could not load library {shown}: {error}") from None


def _load_archive(archive: Path, linker: str | None) -> ctypes.CDLL:
    """Links every object of ``archive`` into a temporary shared library and
    loads it (the file can go once loaded)."""
    driver = _find_linker(linker)
    with tempfile.TemporaryDirectory() as scratch:
        shared = Path(scratch) / f"{archive.stem}.so"
        command = [
            driver,
            "-shared",
            "-Wl,--whole-archive",
            str(archive),
            "-Wl,--no-whole-archive",
            "-o",
            str(shared),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            hint = (
                "\nThe archive's objects must be position-independent to load into "
                "the JIT: rebuild it with -fPIC (CMake: "
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON), or build a shared library."
                if "-fPIC" in result.stderr or "relocation" in result.stderr
                else ""
            )
            raise LinkError(
                f"could not load static library {archive}: "
                f"{' '.join(command)}\n{result.stderr or result.stdout}{hint}"
            )
        return _dlopen(str(shared), str(archive))


def llvm_lowering_pipeline() -> list[PipelineElement]:
    """The passes ``lower_to_llvm`` runs, to inspect or extend.

    Structured control flow becomes branches, every dialect with an LLVM
    lowering is converted, and leftover conversion casts are removed.
    """
    return [
        passes.SCFToControlFlow(),
        passes.ConvertToLLVM(),
        passes.ReconcileUnrealizedCasts(),
    ]


def lower_to_llvm(module: Module) -> None:
    """Lower ``module`` to the LLVM dialect in place.

    Handles into the module's contents become stale; the module itself stays
    usable.

    Raises:
        MLIRError: If some operation has no lowering to LLVM.
    """
    PassManager(Module, llvm_lowering_pipeline()).run(module)


class CompiledFunction:
    """A JIT-compiled function, called like a Python function.

    Arguments and results are checked against the function's MLIR type; see
    ``Argument`` and ``Result`` for what can cross the boundary.
    """

    def __init__(self, engine: ExecutionEngine, name: str, type: FunctionType) -> None:
        self._engine = engine
        self.name = name
        """The function's symbol name."""
        self.type = type
        """The function's MLIR signature."""

    def __call__(self, *args: Argument) -> Result:
        """Call the function.

        Returns:
            ``None`` without results, the value for one result, else a tuple.

        Raises:
            TypeError: For a wrong argument count or argument types, or a
                buffer whose element type, shape, or strides do not match.
            OverflowError: If an int does not fit its integer type.
        """
        result: Result = self._engine.call(self.name, self.type, list(args))
        return result

    def __repr__(self) -> str:
        return f"CompiledFunction({self.name}: {self.type})"


class CompiledModule:
    """The result of ``compile``: the lowered module, its LLVM IR, and the JIT."""

    def __init__(
        self,
        lowered: Module,
        llvm_ir: LLVMModule,
        engine: ExecutionEngine,
        signatures: dict[str, FunctionType],
    ) -> None:
        self.lowered = lowered
        """The compiled copy of the module, in the LLVM dialect."""
        self.llvm_ir = llvm_ir
        """The module as verified LLVM IR, e.g. to write an object file."""
        self.engine = engine
        """The JIT holding the compiled code."""
        self._signatures = signatures

    @property
    def function_names(self) -> list[str]:
        """Names of the callable functions, in module order."""
        return list(self._signatures)

    def function(self, fn: func.FuncOp) -> CompiledFunction:
        """The compiled version of ``fn``, a function of the module that was
        compiled.

        Raises:
            ValueError: If ``fn`` was not compiled here, or its signature
                changed since.
        """
        name = fn.sym_name
        signature = self._signatures.get(name)
        if signature is None:
            raise ValueError(
                f"@{name} is not a function defined in the compiled module"
            )
        if signature != fn.function_type:
            raise ValueError(
                f"@{name} changed type since it was compiled "
                f"({signature} -> {fn.function_type}); compile again"
            )
        return CompiledFunction(self.engine, name, signature)


def compile(
    module: Module,
    *,
    opt_level: OptLevel = OptLevel.O2,
    libraries: Sequence[Library] = (),
    linker: str | None = None,
) -> CompiledModule:
    """Lower, translate, and JIT-compile a copy of ``module``.

    ``module`` itself is not modified, so handles into it stay valid and its
    ``func.FuncOp`` objects select the functions to call.

    Args:
        module: IR in dialects with LLVM lowerings (func, arith, scf, cf,
            memref, math, ...).
        opt_level: LLVM optimization level.
        libraries: Libraries providing external functions, loaded with
            ``load_libraries``. Functions already in this process (the C
            library, for one) need none.
        linker: C compiler driver used to load static archives.

    Raises:
        ValueError, FileNotFoundError, LinkError: See ``load_libraries``.
        MLIRError: If lowering, LLVM IR verification, or compilation fails.
    """
    load_libraries(libraries, linker=linker)
    signatures = {
        op.sym_name: op.function_type
        for op in module.body.operations
        if isinstance(op, func.FuncOp) and len(op.body) > 0
    }
    lowered = module.clone()
    assert isinstance(lowered, Module)
    lower_to_llvm(lowered)
    llvm_ir = translate_to_llvm_ir(lowered)
    llvm_ir.verify()
    engine = ExecutionEngine(lowered, opt_level=opt_level)
    return CompiledModule(lowered, llvm_ir, engine, signatures)


class LinkError(Exception):
    """The system linker failed or could not be found. The message includes
    the linker command and its output (e.g. undefined symbols)."""


def to_llvm_ir(module: Module, *, opt_level: OptLevel = OptLevel.O2) -> LLVMModule:
    """Lower a copy of ``module``, translate it to LLVM IR, optimize it at
    ``opt_level``, and verify it. ``module`` itself is not modified.

    Raises:
        MLIRError: If lowering, translation, or verification fails.
    """
    lowered = module.clone()
    assert isinstance(lowered, Module)
    lower_to_llvm(lowered)
    llvm_ir = translate_to_llvm_ir(lowered)
    llvm_ir.optimize(opt_level)
    llvm_ir.verify()
    return llvm_ir


def _find_linker(linker: str | None) -> str:
    candidates = [linker] if linker else [os.environ.get("CC"), "cc", "clang", "gcc"]
    for candidate in candidates:
        if candidate and (found := shutil.which(candidate)):
            return found
    raise LinkError(
        f"no linker found (tried {', '.join(c for c in candidates if c)}); "
        "install a C compiler or pass linker="
    )


def _link(
    module: Module,
    output: str | os.PathLike[str],
    *,
    shared: bool,
    opt_level: OptLevel,
    libraries: Sequence[Library],
    linker: str | None,
) -> Path:
    resolved = [_library(library) for library in libraries]
    llvm_ir = to_llvm_ir(module, opt_level=opt_level)
    driver = _find_linker(linker)
    target = Path(output).resolve()
    with tempfile.TemporaryDirectory() as scratch:
        obj = Path(scratch) / "module.o"
        llvm_ir.write_object(str(obj), level=opt_level)
        command = [
            driver,
            *(["-shared"] if shared else []),
            str(obj),
            "-o",
            str(target),
        ]
        # After the object, in the given order, as static archives require.
        command += [
            f"-l{library}" if isinstance(library, str) else str(library)
            for library in resolved
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise LinkError(
            f"linking failed: {' '.join(command)}\n{result.stderr or result.stdout}"
        )
    return target


def _check_main(module: Module) -> None:
    """Raises ValueError unless ``module`` defines a C-compatible ``main``."""
    i32 = IntegerType(32, context=module.context)
    entry = next(
        (
            op
            for op in module.body.operations
            if isinstance(op, func.FuncOp) and op.sym_name == "main"
        ),
        None,
    )
    if entry is None or len(entry.body) == 0:
        raise ValueError(
            "an executable needs `main`: define func.FuncOp('main', "
            "FunctionType([], [IntegerType(32)])) with a body"
        )
    inputs, results = entry.function_type.inputs, entry.function_type.results
    argv = [i32, llvm.PointerType(context=module.context)]
    if results != [i32] or inputs not in ([], argv):
        raise ValueError(
            f"`main` has type {entry.function_type}; an executable's main must be "
            "() -> i32 or (i32, !llvm.ptr) -> i32 (argc, argv), returning the exit status"
        )


def build_executable(
    module: Module,
    output: str | os.PathLike[str],
    *,
    opt_level: OptLevel = OptLevel.O2,
    libraries: Sequence[Library] = (),
    linker: str | None = None,
) -> Path:
    """Compile ``module`` into a native executable for this machine.

    ``module`` must define ``func.func @main`` of type ``() -> i32`` or
    ``(i32, !llvm.ptr) -> i32`` (argc and argv); its result is the exit
    status. ``module`` itself is not modified.

    Args:
        module: IR in dialects with LLVM lowerings.
        output: Path of the executable to write.
        opt_level: LLVM optimization level.
        libraries: Libraries to link, in link order: names (``"m"`` for
            ``-lm``) or paths to ``.so``/``.a`` files; see ``Library``.
        linker: C compiler driver used to link; ``$CC``, ``cc``, ``clang``,
            or ``gcc`` by default.

    Returns:
        The absolute path of the executable.

    Raises:
        ValueError: If ``main`` is missing or has another type, or a library
            ``str`` looks like a path.
        FileNotFoundError: If a library path does not exist.
        MLIRError: If lowering, translation, or verification fails.
        LinkError: If no linker is found or linking fails.
    """
    _check_main(module)
    return _link(
        module,
        output,
        shared=False,
        opt_level=opt_level,
        libraries=libraries,
        linker=linker,
    )


def build_shared_library(
    module: Module,
    output: str | os.PathLike[str],
    *,
    opt_level: OptLevel = OptLevel.O2,
    libraries: Sequence[Library] = (),
    linker: str | None = None,
) -> Path:
    """Compile ``module`` into a native shared library for this machine.

    Every public function becomes an exported C symbol, callable from C or
    through ``ctypes``. ``module`` itself is not modified.

    Args:
        module: IR in dialects with LLVM lowerings.
        output: Path of the library to write, e.g. ``"libkernels.so"``.
        opt_level: LLVM optimization level.
        libraries: Libraries to link, in link order: names (``"m"`` for
            ``-lm``) or paths to ``.so``/``.a`` files; see ``Library``.
        linker: C compiler driver used to link; ``$CC``, ``cc``, ``clang``,
            or ``gcc`` by default.

    Returns:
        The absolute path of the library.

    Raises:
        ValueError: If a library ``str`` looks like a path.
        FileNotFoundError: If a library path does not exist.
        MLIRError: If lowering, translation, or verification fails.
        LinkError: If no linker is found or linking fails.
    """
    return _link(
        module,
        output,
        shared=True,
        opt_level=opt_level,
        libraries=libraries,
        linker=linker,
    )
