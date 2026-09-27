"""Compiles Python function source to MLIR.

1. The function's syntax tree is split into a control-flow graph of basic
   blocks holding simple statements (assignments and expressions). Loops,
   conditionals, ``break``, ``continue``, and early ``return`` become edges;
   ``for ... in range`` is desugared into plain statements first.
2. A definite-assignment analysis finds the variables that hold a value on
   every path into each block. They become the block's arguments, so
   variables can be reassigned across branches and loops (SSA form).
3. Blocks are emitted in reverse postorder as ``func``/``arith``/``cf`` IR,
   with MLIR locations pointing at the Python source.

Errors are ``CompileError``, a ``SyntaxError`` subclass, so Python shows the
file, line, and a caret under the offending code.
"""

from __future__ import annotations

import ast
import builtins
import enum
import inspect
import itertools
import operator
import textwrap
import types as pytypes
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .. import _mlir_python as ir
from ..dialects import arith, cf, func, llvm, math, memref, scf
from ._types import (
    ARRAY_ELEMENT_KINDS,
    Array,
    ArrayType,
    Fn,
    FnType,
    Ptr,
    ScalarType,
    StructType,
    array,
    array_of,
    atomic_add,
    boolean,
    cstr,
    f64,
    function_type,
    i32,
    i64,
    pointer_to,
    ptr,
    scalar_type,
    stack,
)

if TYPE_CHECKING:
    from ._program import Function, Module

ATOMIC_ADD = "__mlir_python_atomic_add_i"
"""Prefix of the placeholder that ``atomic_add`` calls until linking."""


class CompileError(SyntaxError):
    """A Python function could not be compiled. Shown like a syntax error:
    file, line, source, and a caret at the problem."""


# ---------------------------------------------------------------------------
# Source and signatures
# ---------------------------------------------------------------------------


@dataclass
class Source:
    """A function's source, parsed, with positions mapped back to the file."""

    filename: str
    first_line: int  # file line of the source's first line
    indent: int  # columns removed by dedenting
    lines: list[str]
    tree: ast.FunctionDef

    @staticmethod
    def of(fn: Callable[..., object]) -> Source:
        try:
            lines, first_line = inspect.getsourcelines(fn)
            filename = inspect.getsourcefile(fn) or "<unknown>"
        except (OSError, TypeError) as error:
            raise CompileError(
                f"cannot read the source of {fn.__qualname__}; compiled functions must be defined in a file"
            ) from error
        text = "".join(lines)
        dedented = textwrap.dedent(text)
        indent = len(lines[0]) - len(lines[0].lstrip()) if lines else 0
        module = ast.parse(dedented)
        tree = module.body[0]
        if not isinstance(tree, ast.FunctionDef):
            raise CompileError(f"{fn.__qualname__} is not a plain function definition")
        return Source(filename, first_line, indent, lines, tree)

    def error(self, node: ast.AST, message: str) -> CompileError:
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0)
        end_line = getattr(node, "end_lineno", None)
        end_column = getattr(node, "end_col_offset", None)
        text = self.lines[line - 1] if 0 < line <= len(self.lines) else None
        return CompileError(
            message,
            (
                self.filename,
                self.first_line + line - 1,
                column + self.indent + 1,
                text,
                self.first_line + end_line - 1 if end_line else None,
                end_column + self.indent + 1 if end_column is not None else None,
            ),
        )

    def location(self, node: ast.AST) -> ir.Location:
        line = self.first_line + getattr(node, "lineno", 1) - 1
        column = getattr(node, "col_offset", 0) + self.indent + 1
        return ir.Location.file(self.filename, line, column)


@dataclass
class Signature:
    params: list[tuple[str, ScalarType]]
    results: list[ScalarType]
    returns_tuple: bool
    variadic: bool = False  # a C variadic function (``*args``), e.g. printf

    def function_type(self) -> ir.FunctionType:
        return ir.FunctionType(
            [t.mlir() for _, t in self.params], [t.mlir() for t in self.results]
        )


def signature_of(
    fn: Callable[..., object], source: Source, *, allow_variadic: bool = False
) -> Signature:
    try:
        annotations = inspect.get_annotations(fn, eval_str=True)
    except Exception as error:
        raise source.error(
            source.tree, f"cannot evaluate annotations: {error}"
        ) from error
    arguments = source.tree.args
    if arguments.kwarg:
        raise source.error(
            arguments.kwarg,
            "C functions have no keyword arguments; declare a variadic function such as printf with *args",
        )
    if arguments.kwonlyargs or arguments.defaults:
        first = (arguments.kwonlyargs or arguments.defaults)[0]
        raise source.error(
            first,
            "compiled functions take plain positional parameters (no defaults or keyword-only parameters)",
        )
    if arguments.vararg and not allow_variadic:
        raise source.error(
            arguments.vararg,
            "only @program.extern functions can take *args (C variadic functions such as printf)",
        )
    variadic = arguments.vararg is not None
    params = []
    for arg in [*arguments.posonlyargs, *arguments.args]:
        if arg.arg not in annotations:
            raise source.error(
                arg, f"parameter '{arg.arg}' needs a type annotation, e.g. i32"
            )
        kind = scalar_type(annotations[arg.arg])
        if kind is None:
            raise source.error(
                arg,
                f"parameter '{arg.arg}' has unsupported type {annotations[arg.arg]!r}",
            )
        params.append((arg.arg, kind))
    result = annotations.get("return")
    node = source.tree.returns or source.tree
    if result is None or result is type(None):
        return Signature(params, [], False, variadic)
    if isinstance(result, pytypes.GenericAlias) and result.__origin__ is tuple:
        results = [scalar_type(item) for item in result.__args__]
        if any(item is None for item in results):
            raise source.error(node, f"unsupported result type {result!r}")
        return Signature(params, [r for r in results if r is not None], True, variadic)
    kind = scalar_type(result)
    if kind is None:
        raise source.error(node, f"unsupported result type {result!r}")
    return Signature(params, [kind], False, variadic)


# ---------------------------------------------------------------------------
# Control-flow graph
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Block:
    statements: list[ast.stmt] = field(default_factory=list)
    terminator: Jump | Branch | Return | None = None

    def successors(self) -> list[Block]:
        match self.terminator:
            case Jump(target=target):
                return [target]
            case Branch(if_true=t, if_false=f):
                return [t, f]
            case _:
                return []


@dataclass(eq=False)
class Jump:
    target: Block


@dataclass(eq=False)
class Branch:
    test: ast.expr
    if_true: Block
    if_false: Block


@dataclass(eq=False)
class Return:
    value: ast.expr | None
    node: ast.AST
    implicit: bool = False


@dataclass
class Loop:
    exit: Block  # break
    next: Block  # continue


class GraphBuilder:
    """Splits a function body into basic blocks."""

    def __init__(self, source: Source) -> None:
        self.source = source
        self.hidden = itertools.count()

    def build(self) -> Block:
        entry = Block()
        end = self.walk(self.source.tree.body, entry, None)
        if end is not None:
            end.terminator = Return(None, self.source.tree, implicit=True)
        return entry

    def walk(
        self, statements: list[ast.stmt], block: Block | None, loop: Loop | None
    ) -> Block | None:
        for statement in statements:
            if block is None:
                break  # unreachable code after return/break/continue
            block = self.statement(statement, block, loop)
        return block

    def statement(
        self, node: ast.stmt, block: Block, loop: Loop | None
    ) -> Block | None:
        match node:
            case (
                ast.Assign()
                | ast.AugAssign()
                | ast.AnnAssign()
                | ast.Expr()
                | ast.Pass()
            ):
                block.statements.append(node)
                return block
            case ast.Return(value=value):
                block.terminator = Return(value, node)
                return None
            case ast.If(test=test, body=body, orelse=orelse):
                then_block, merge = Block(), Block()
                else_block = Block() if orelse else merge
                block.terminator = Branch(test, then_block, else_block)
                reached = False
                for start, statements in ((then_block, body), (else_block, orelse)):
                    if start is merge:
                        reached = True
                        continue
                    end = self.walk(statements, start, loop)
                    if end is not None:
                        end.terminator = Jump(merge)
                        reached = True
                return merge if reached else None
            case ast.While(test=test, body=body, orelse=orelse):
                if orelse:
                    raise self.source.error(node, "while ... else is not supported")
                header, body_block, exit_block = Block(), Block(), Block()
                block.terminator = Jump(header)
                if isinstance(test, ast.Constant) and test.value is True:
                    # `while True`: the exit is reached only through `break`.
                    header.terminator = Jump(body_block)
                else:
                    header.terminator = Branch(test, body_block, exit_block)
                end = self.walk(body, body_block, Loop(exit_block, header))
                if end is not None:
                    end.terminator = Jump(header)
                return exit_block
            case ast.For():
                return self.for_range(node, block)
            case ast.Break():
                if loop is None:
                    raise self.source.error(node, "'break' outside a loop")
                block.terminator = Jump(loop.exit)
                return None
            case ast.Continue():
                if loop is None:
                    raise self.source.error(node, "'continue' outside a loop")
                block.terminator = Jump(loop.next)
                return None
            case (
                ast.FunctionDef()
                | ast.AsyncFunctionDef()
                | ast.ClassDef()
                | ast.Lambda()
            ):
                raise self.source.error(
                    node, "nested functions and classes are not supported"
                )
            case _:
                name = type(node).__name__.lower()
                raise self.source.error(node, f"'{name}' statements are not supported")

    def for_range(self, node: ast.For, block: Block) -> Block:
        if node.orelse:
            raise self.source.error(node, "for ... else is not supported")
        if not isinstance(node.target, ast.Name):
            raise self.source.error(
                node.target, "the loop variable must be a plain name"
            )
        call = node.iter
        k = next(self.hidden)
        counter, limit, stride = f"__for{k}_i", f"__for{k}_stop", f"__for{k}_step"
        sequence = f"__for{k}_items"
        is_range = (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
            and not call.keywords
            and 1 <= len(call.args) <= 3
        )
        if is_range:
            assert isinstance(call, ast.Call)
            args = call.args
            start = args[0] if len(args) > 1 else ast.Constant(0)
            stop = args[1] if len(args) > 1 else args[0]
            step = args[2] if len(args) == 3 else ast.Constant(1)
        else:
            # `for x in xs`: evaluate xs once, then index it from 0 to len(xs).
            start, step = ast.Constant(0), ast.Constant(1)
            # Every field is given: Python < 3.13 has no defaults for lists.
            stop = ast.Call(
                ast.Name("len", ast.Load()), [ast.Name(sequence, ast.Load())], []
            )

        def located[T: ast.AST](new: T, like: ast.AST = node) -> T:
            ast.copy_location(new, like)
            return ast.fix_missing_locations(new)

        def name(identifier: str, like: ast.AST = node) -> ast.Name:
            return located(ast.Name(identifier, ast.Load()), like)

        def store(identifier: str, value: ast.expr, like: ast.AST = node) -> ast.Assign:
            return located(
                ast.Assign([located(ast.Name(identifier, ast.Store()), like)], value),
                like,
            )

        if not is_range:
            block.statements.append(store(sequence, call, call))
            stop = located(stop, call)
        block.statements += [store(counter, start, call), store(limit, stop, call)]
        literal_step = literal_int(step)
        if literal_step == 0:
            raise self.source.error(step, "range() step must not be zero")

        def compare(op: ast.cmpop, left: ast.expr, right: ast.expr) -> ast.Compare:
            return located(ast.Compare(left, [op], [right]), call)

        if literal_step is not None:
            op: ast.cmpop = ast.Lt() if literal_step > 0 else ast.Gt()
            test: ast.expr = compare(op, name(counter), name(limit))
            increment: ast.expr = located(ast.Constant(literal_step), step)
        else:
            block.statements.append(store(stride, step, step))
            zero = located(ast.Constant(0), step)
            forward = located(
                ast.BoolOp(
                    ast.And(),
                    [
                        compare(ast.Gt(), name(stride), zero),
                        compare(ast.Lt(), name(counter), name(limit)),
                    ],
                )
            )
            backward = located(
                ast.BoolOp(
                    ast.And(),
                    [
                        compare(ast.Lt(), name(stride), zero),
                        compare(ast.Gt(), name(counter), name(limit)),
                    ],
                )
            )
            test = located(ast.BoolOp(ast.Or(), [forward, backward]), call)
            increment = name(stride, step)
        header, body, latch, exit_block = Block(), Block(), Block(), Block()
        block.terminator = Jump(header)
        header.terminator = Branch(test, body, exit_block)
        element: ast.expr = (
            name(counter)
            if is_range
            else located(
                ast.Subscript(name(sequence, call), name(counter), ast.Load()), call
            )
        )
        body.statements.append(store(node.target.id, element, node.target))
        end = self.walk(node.body, body, Loop(exit_block, latch))
        if end is not None:
            end.terminator = Jump(latch)
        latch.statements.append(
            located(
                ast.AugAssign(
                    located(ast.Name(counter, ast.Store())), ast.Add(), increment
                )
            )
        )
        latch.terminator = Jump(header)
        return exit_block


def literal_int(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
    ):
        return -node.operand.value
    return None


def assigned_names(statement: ast.stmt) -> set[str]:
    targets: list[ast.expr] = []
    match statement:
        case ast.Assign(targets=assign_targets):
            targets = list(assign_targets)
        case ast.AugAssign(target=target) | ast.AnnAssign(target=target):
            targets = [target]
    names: set[str] = set()
    for target in targets:
        # Only names are assigned; in `p[i] = v`, `p` and `i` are just read.
        elements = target.elts if isinstance(target, ast.Tuple) else [target]
        names.update(e.id for e in elements if isinstance(e, ast.Name))
    return names


def static_object(node: ast.expr, resolve: Callable[[str], object]) -> object:
    """What a name or dotted name (``stack``, ``lang.stack``, ``math.pi``)
    refers to at compile time, or ``None``."""
    if isinstance(node, ast.Name):
        return resolve(node.id)
    if isinstance(node, ast.Attribute):
        base = static_object(node.value, resolve)
        return getattr(base, node.attr, None) if base is not None else None
    return None


def is_runtime_value(node: ast.expr, is_local: Callable[[str], bool]) -> bool:
    """Whether ``node``'s attributes are struct fields (``c.x``, ``p[0].x``,
    ``make().x``) rather than names in a Python namespace (``math.pi``)."""
    match node:
        case ast.Name(id=name):
            return is_local(name)
        case ast.Attribute(value=base):
            return is_runtime_value(base, is_local)
        case ast.Subscript() | ast.Call():
            return True
        case _:
            return False


def type_from_expression(
    node: ast.expr, resolve: Callable[[str], object]
) -> ScalarType | None:
    """The scalar type a type expression (``i32``, ``lang.Ptr[f64]``) names."""
    if isinstance(node, (ast.Name, ast.Attribute)):
        return scalar_type(static_object(node, resolve))
    if isinstance(node, ast.Subscript) and static_object(node.value, resolve) is Ptr:
        element = type_from_expression(node.slice, resolve)
        return pointer_to(element) if element is not None else None
    if isinstance(node, ast.Subscript) and static_object(node.value, resolve) is Array:
        element = type_from_expression(node.slice, resolve)
        if element is None or element.kind not in ARRAY_ELEMENT_KINDS:
            return None
        return array_of(element)
    if isinstance(node, ast.Subscript) and static_object(node.value, resolve) is Fn:
        match node.slice:
            case ast.Tuple(elts=[ast.List(elts=param_nodes), result_node]):
                params = [type_from_expression(p, resolve) for p in param_nodes]
                if any(p is None for p in params):
                    return None
                if isinstance(result_node, ast.Constant) and result_node.value is None:
                    result = None
                else:
                    result = type_from_expression(result_node, resolve)
                    if result is None:
                        return None
                return function_type([p for p in params if p is not None], result)
    return None


def fits_function_type(given: ScalarType, expected: ScalarType) -> bool:
    """Whether a function of type ``given`` can be called as ``expected``: the
    same shape, where a typed pointer (``Ptr[T]``) may stand for C's opaque
    ``ptr`` (``void *``), so ``Fn[[Ptr[State]], None]`` passes as a callback
    of type ``Fn[[ptr], None]``."""
    if not (isinstance(given, FnType) and isinstance(expected, FnType)):
        return False
    if len(given.params) != len(expected.params):
        return False

    def same(a: ScalarType | None, b: ScalarType | None) -> bool:
        if a == b:
            return True
        return (
            a is not None
            and b is not None
            and a.kind == "ptr"
            and b.kind == "ptr"
            and ptr in (a, b)
        )

    return same(given.result, expected.result) and all(
        same(a, b) for a, b in zip(given.params, expected.params, strict=True)
    )


def value_type(signature: Signature) -> FnType | None:
    """The ``Fn`` type of a function with ``signature``, or ``None`` when no
    function value can have it (several results, or C varargs)."""
    if signature.returns_tuple or signature.variadic:
        return None
    return function_type(
        [kind for _, kind in signature.params],
        signature.results[0] if signature.results else None,
    )


def reverse_postorder(entry: Block) -> list[Block]:
    order: list[Block] = []
    seen: set[int] = set()

    def visit(block: Block) -> None:
        seen.add(id(block))
        for successor in block.successors():
            if id(successor) not in seen:
                visit(successor)
        order.append(block)

    visit(entry)
    return order[::-1]


def definitely_assigned(
    entry: Block, blocks: list[Block], params: set[str]
) -> dict[int, set[str]]:
    """For each block, the variables assigned on every path into it."""
    predecessors: dict[int, list[Block]] = {id(b): [] for b in blocks}
    for block in blocks:
        for successor in block.successors():
            predecessors[id(successor)].append(block)
    assigned = {
        id(b): set().union(*(assigned_names(s) for s in b.statements)) for b in blocks
    }
    everything = params.union(*assigned.values())
    entering = {id(b): set(everything) for b in blocks}
    entering[id(entry)] = set(params)
    leaving = {id(b): entering[id(b)] | assigned[id(b)] for b in blocks}
    changed = True
    while changed:
        changed = False
        for block in blocks[1:]:
            new = set(everything)
            for predecessor in predecessors[id(block)]:
                new &= leaving[id(predecessor)]
            if new != entering[id(block)]:
                entering[id(block)] = new
                leaving[id(block)] = new | assigned[id(block)]
                changed = True
    return entering


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


@dataclass
class Typed:
    value: ir.Value
    type: ScalarType


@dataclass
class Literal:
    value: int | float | bool | str


@dataclass
class TupleValue:
    items: list[Typed | Literal]


type Operand = Typed | Literal


def default_type(literal: Literal) -> ScalarType:
    value = literal.value
    if isinstance(value, bool):
        return boolean
    if isinstance(value, int):
        return i64
    if isinstance(value, float):
        return f64
    return cstr


PYTHON_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}

SYMBOLS: dict[type[ast.AST], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

SIGNED_PREDICATES = {
    ast.Eq: arith.CmpIPredicate.EQ,
    ast.NotEq: arith.CmpIPredicate.NE,
    ast.Lt: arith.CmpIPredicate.SLT,
    ast.LtE: arith.CmpIPredicate.SLE,
    ast.Gt: arith.CmpIPredicate.SGT,
    ast.GtE: arith.CmpIPredicate.SGE,
}
UNSIGNED_PREDICATES = {
    ast.Eq: arith.CmpIPredicate.EQ,
    ast.NotEq: arith.CmpIPredicate.NE,
    ast.Lt: arith.CmpIPredicate.ULT,
    ast.LtE: arith.CmpIPredicate.ULE,
    ast.Gt: arith.CmpIPredicate.UGT,
    ast.GtE: arith.CmpIPredicate.UGE,
}
FLOAT_PREDICATES = {
    ast.Eq: arith.CmpFPredicate.OEQ,
    ast.NotEq: arith.CmpFPredicate.UNE,
    ast.Lt: arith.CmpFPredicate.OLT,
    ast.LtE: arith.CmpFPredicate.OLE,
    ast.Gt: arith.CmpFPredicate.OGT,
    ast.GtE: arith.CmpFPredicate.OGE,
}


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------


class LiteralKind(enum.Enum):
    """The type of an unannotated literal: fits any integer (``INT``) or
    float (``FLOAT``) type, and defaults to i64/f64 if nothing constrains it."""

    INT = "int"
    FLOAT = "float"


@dataclass(frozen=True)
class ArrayFact:
    """An array whose element type is still being inferred: the key of its
    element's type (so ``[0, 0]`` becomes ``Array[i32]`` if used as one)."""

    element: object


type TypeFact = ScalarType | LiteralKind | ArrayFact | None
INT_LITERAL = LiteralKind.INT
FLOAT_LITERAL = LiteralKind.FLOAT


class TypeInference:
    """Decides each variable's type from how the whole function uses it
    (one type per variable), so ``total = 0`` is ``i32`` when ``total`` is
    returned from an ``i32`` function. Conflicts are left for emission to
    report at the offending line."""

    def __init__(self, resolve: Callable[[str], object]) -> None:
        self.resolve = resolve
        self.parent: dict[object, object] = {}
        self.fact: dict[object, TypeFact] = {}
        self.counter = itertools.count()
        self.displays: dict[int, object] = {}  # id of a list display -> its key

    def fresh(self, fact: TypeFact) -> object:
        key = ("expr", next(self.counter))
        self.parent[key] = key
        self.fact[key] = fact
        return key

    def variable(self, name: str) -> object:
        key = ("var", name)
        if key not in self.parent:
            self.parent[key] = key
            self.fact[key] = None
        return key

    def find(self, key: object) -> object:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: object, b: object) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.parent[rb] = ra
        self.fact[ra] = self.merge(self.fact[ra], self.fact[rb])

    def merge(self, a: TypeFact, b: TypeFact) -> TypeFact:
        if a is None:
            return b
        if b is None or a == b:
            return a
        if isinstance(a, ArrayFact) or isinstance(b, ArrayFact):
            return self.merge_arrays(a, b)
        if isinstance(a, ScalarType) and isinstance(b, ScalarType):
            return a  # a conflict; emission reports it where it happens
        if isinstance(a, ScalarType):
            return a  # a literal takes the concrete type (or emission reports a misfit)
        if isinstance(b, ScalarType):
            return b
        return FLOAT_LITERAL  # an int literal meets a float literal

    def merge_arrays(self, a: TypeFact, b: TypeFact) -> TypeFact:
        """Unifies the element types of two arrays; a known array type wins."""
        elements = []
        for fact in (a, b):
            if isinstance(fact, ArrayFact):
                elements.append(fact.element)
            elif isinstance(fact, ArrayType) and fact.element is not None:
                elements.append(self.fresh(fact.element))
            else:
                return a  # a conflict; emission reports it where it happens
        self.union(elements[0], elements[1])
        return b if isinstance(b, ArrayType) else a

    def result(self, key: object) -> ScalarType | None:
        fact = self.fact[self.find(key)]
        if fact == INT_LITERAL:
            return i64
        if fact == FLOAT_LITERAL:
            return f64
        if isinstance(fact, ArrayFact):
            element = self.result(fact.element)
            return array_of(element) if element is not None else None
        return fact if isinstance(fact, ScalarType) else None

    # -- constraints ------------------------------------------------------------

    def expression(self, node: ast.expr) -> object | list[object]:
        match node:
            case ast.Constant(value=bool()):
                return self.fresh(boolean)
            case ast.Constant(value=int()):
                return self.fresh(INT_LITERAL)
            case ast.Constant(value=float()):
                return self.fresh(FLOAT_LITERAL)
            case ast.Constant(value=str()):
                return self.fresh(cstr)
            case ast.Name(id=name) if ("var", name) in self.parent:
                return self.variable(name)
            case ast.Attribute(value=base, attr=attr) if is_runtime_value(
                base, lambda name: ("var", name) in self.parent
            ):
                return self.fresh(self.field_of(self.single(base), attr))
            case ast.Name() | ast.Attribute():
                from ._program import Function

                value = static_object(node, self.resolve)
                if isinstance(value, Function):
                    return self.fresh(value_type(value.module.signature(value)))
                if isinstance(value, bool):
                    return self.fresh(boolean)
                if isinstance(value, int):
                    return self.fresh(INT_LITERAL)
                if isinstance(value, float):
                    return self.fresh(FLOAT_LITERAL)
                return self.fresh(None)
            case ast.Tuple(elts=elements):
                return [self.single(e) for e in elements]
            case ast.BinOp(left=left, op=op, right=right):
                a, b = self.single(left), self.single(right)
                fact = self.fact[self.find(a)]
                if isinstance(fact, ScalarType) and fact.kind == "ptr":
                    return a  # `p + n`: a pointer, moved by a count of any integer type
                self.union(a, b)
                if isinstance(op, ast.Div):
                    fact = self.fact[self.find(a)]
                    is_float = fact == FLOAT_LITERAL or (
                        isinstance(fact, ScalarType) and fact.kind == "float"
                    )
                    return a if is_float else self.fresh(f64)
                return a
            case ast.UnaryOp(op=ast.Not()):
                self.single(node.operand)
                return self.fresh(boolean)
            case ast.UnaryOp(operand=operand):
                return self.single(operand)
            case ast.Compare(left=left, comparators=comparators):
                keys = [self.single(left), *(self.single(c) for c in comparators)]
                for a, b in itertools.pairwise(keys):
                    self.union(a, b)
                return self.fresh(boolean)
            case ast.BoolOp(values=values):
                for value in values:
                    self.single(value)
                return self.fresh(boolean)
            case ast.IfExp(test=test, body=body, orelse=orelse):
                self.single(test)
                a, b = self.single(body), self.single(orelse)
                self.union(a, b)
                return a
            case ast.Call(func=ast.Name() | ast.Attribute() as callee, args=args) if (
                is_runtime_value(callee, lambda name: ("var", name) in self.parent)
            ):
                # A call through a function value: `f(x)`, `handlers.on_key(x)`.
                keys = [self.single(arg) for arg in args]
                kind = self.fact[self.find(self.single(callee))]
                if not isinstance(kind, FnType):
                    return self.fresh(None)
                for key, param in zip(keys, kind.params, strict=False):
                    self.union(key, self.fresh(param))
                return self.fresh(kind.result)
            case ast.Call(
                func=ast.Name() | ast.Attribute() as callee,
                args=args,
                keywords=keywords,
            ):
                return self.call(static_object(callee, self.resolve), args, keywords)
            case ast.Call(func=ast.Subscript() as kind_node):  # Ptr[T](raw)
                return self.fresh(type_from_expression(kind_node, self.resolve))
            case ast.Subscript(value=base, slice=index):
                self.single(index)
                return self.element_of(self.single(base))
            case ast.List(elts=elements):
                element = self.fresh(None)
                for item in elements:
                    self.union(element, self.single(item))
                key = self.fresh(ArrayFact(element))
                self.displays[id(node)] = key
                return key
            case _:
                return self.fresh(None)

    def single(self, node: ast.expr) -> object:
        key = self.expression(node)
        return self.fresh(None) if isinstance(key, list) else key

    def call(
        self, target: object, args: list[ast.expr], keywords: list[ast.keyword]
    ) -> object | list[object]:
        from ._program import Function

        if target is stack and args:
            element = type_from_expression(args[0], self.resolve)
            return self.fresh(pointer_to(element) if element is not None else None)
        if target is array and args:
            element = type_from_expression(args[0], self.resolve)
            for arg in args[1:]:
                self.single(arg)
            return self.fresh(array_of(element) if element is not None else None)
        keys = [self.single(arg) for arg in args]
        if target is builtins.len:
            return self.fresh(i64)
        kind = scalar_type(target)
        if isinstance(kind, StructType):
            named = {k.arg: self.single(k.value) for k in keywords if k.arg}
            for index, (name, field_kind) in enumerate(kind.fields):
                key = keys[index] if index < len(keys) else named.get(name)
                if key is not None:
                    self.union(key, self.fresh(field_kind))
            return self.fresh(kind)
        if isinstance(target, Function):
            signature = target.module.signature(target)
            for key, (_, kind) in zip(keys, signature.params, strict=False):
                self.union(key, self.fresh(kind))
            results = [self.fresh(kind) for kind in signature.results]
            if signature.returns_tuple:
                return results
            return results[0] if results else self.fresh(None)
        kind = scalar_type(target)
        if kind is not None:
            return self.fresh(kind)
        if target is builtins.abs and keys:
            return keys[0]
        if target is atomic_add and len(keys) == 2:
            element = self.element_of(keys[0])
            self.union(element, keys[1])
            return element
        if target in (builtins.min, builtins.max) and keys:
            for key in keys[1:]:
                self.union(keys[0], key)
            return keys[0]
        return self.fresh(None)

    def element_of(self, container: object) -> object:
        """The key of the element type of a pointer or array."""
        fact = self.fact[self.find(container)]
        if isinstance(fact, ArrayFact):
            return fact.element
        if isinstance(fact, ScalarType) and fact.element is not None:
            return self.fresh(fact.element)
        return self.fresh(None)

    def field_of(self, value: object, name: str) -> ScalarType | None:
        fact = self.fact[self.find(value)]
        if isinstance(fact, StructType) and (found := fact.field(name)):
            return found[1]
        return None

    def assign(self, target: ast.expr, value: object | list[object]) -> None:
        if isinstance(target, ast.Attribute) and not isinstance(value, list):
            self.union(value, self.single(target))
            return
        if isinstance(target, ast.Subscript) and not isinstance(value, list):
            self.union(value, self.element_of(self.single(target.value)))
            return
        if isinstance(target, ast.Name):
            if not isinstance(value, list):
                self.union(self.variable(target.id), value)
        elif isinstance(target, ast.Tuple) and isinstance(value, list):
            for element, key in zip(target.elts, value, strict=False):
                self.assign(element, key)

    def statement(self, node: ast.stmt) -> None:
        match node:
            case ast.Assign(targets=targets, value=value):
                key = self.expression(value)
                for target in targets:
                    self.assign(target, key)
            case ast.AugAssign(target=ast.Name(id=name), op=op, value=value):
                key = self.single(value)
                if not isinstance(op, ast.Div):
                    self.union(self.variable(name), key)
            case ast.AugAssign(target=ast.Attribute() as target, op=op, value=value):
                key = self.single(value)
                if not isinstance(op, ast.Div):
                    self.union(self.single(target), key)
            case ast.AnnAssign(
                target=ast.Name(id=name), annotation=annotation, value=value
            ):
                kind = type_from_expression(annotation, self.resolve)
                if kind is not None:
                    # The annotation decides; the value is converted to it
                    # (e.g. a Ptr[i32] stored in a `ptr` variable).
                    self.union(self.variable(name), self.fresh(kind))
                    if value is not None:
                        literal = self.single(value)
                        fact = self.fact[self.find(literal)]
                        if fact in (INT_LITERAL, FLOAT_LITERAL) or isinstance(
                            fact, ArrayFact
                        ):
                            self.union(literal, self.fresh(kind))
                elif value is not None:
                    self.union(self.variable(name), self.single(value))
            case ast.Expr(value=value):
                self.expression(value)

    def run(
        self, blocks: list[Block], signature: Signature, names: set[str]
    ) -> dict[str, ScalarType]:
        for name in names:
            self.variable(name)
        for name, kind in signature.params:
            self.union(self.variable(name), self.fresh(kind))
        for block in blocks:
            for statement in block.statements:
                self.statement(statement)
            match block.terminator:
                case Branch(test=test):
                    self.expression(test)
                case Return(value=value) if value is not None:
                    key = self.expression(value)
                    keys = key if isinstance(key, list) else [key]
                    for item, kind in zip(keys, signature.results, strict=False):
                        self.union(item, self.fresh(kind))
        types = {}
        for name in names:
            kind = self.result(self.variable(name))
            if kind is not None:
                types[name] = kind
        return types

    def display_types(self) -> dict[int, ScalarType]:
        """The inferred type of each list display, by ``id`` of its node."""
        return {
            node: kind
            for node, key in self.displays.items()
            if (kind := self.result(key)) is not None
        }


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


class FunctionCompiler:
    """Emits one function's body into its ``func.FuncOp``."""

    def __init__(
        self,
        program: Module,
        function: Function[..., Any],
        op: func.FuncOp,
        signature: Signature,
        source: Source,
    ) -> None:
        self.program = program
        self.function = function
        self.op = op
        self.signature = signature
        self.source = source
        self.region = op.body
        self.variable_types: dict[str, ScalarType] = dict(signature.params)
        self.display_types: dict[int, ScalarType] = {}
        self.env: dict[str, Typed] = {}
        self.blocks: dict[int, ir.Block] = {}
        self.block_arguments: dict[int, list[str]] = {}
        self.entering: dict[int, set[str]] = {}
        self.all_assigned: set[str] = set()
        self._insertion: ir.InsertionPoint | None = None

    # -- structure ------------------------------------------------------------

    def compile(self) -> None:
        entry = GraphBuilder(self.source).build()
        order = reverse_postorder(entry)
        params = {name for name, _ in self.signature.params}
        self.entering = definitely_assigned(entry, order, params)
        self.all_assigned = params.union(
            *(assigned_names(s) for b in order for s in b.statements)
        )
        inference = TypeInference(self.resolve_quietly)
        inferred = inference.run(order, self.signature, self.all_assigned)
        self.display_types = inference.display_types()
        for name, kind in inferred.items():
            self.variable_types.setdefault(name, kind)
        entry_block = self.op.add_entry_block()
        self.entry_block = entry_block
        self.blocks[id(entry)] = entry_block
        self.block_arguments[id(entry)] = [name for name, _ in self.signature.params]
        try:
            for block in order:
                self.emit_block(block)
        finally:
            self.move_to(None)

    def block_for(self, block: Block) -> ir.Block:
        if id(block) not in self.blocks:
            names = sorted(self.entering[id(block)])
            missing = [n for n in names if n not in self.variable_types]
            assert not missing, f"untyped variables at block entry: {missing}"
            self.blocks[id(block)] = self.region.append_block(
                [self.variable_types[n].mlir() for n in names]
            )
            self.block_arguments[id(block)] = names
        return self.blocks[id(block)]

    def move_to(self, block: ir.Block | None) -> None:
        if self._insertion is not None:
            self._insertion.__exit__(None, None, None)
            self._insertion = None
        if block is not None:
            self._insertion = ir.InsertionPoint(block)
            self._insertion.__enter__()

    @property
    def current(self) -> ir.Block:
        assert self._insertion is not None
        return self._insertion.block

    @contextmanager
    def at(self, node: ast.AST) -> Iterator[None]:
        with self.source.location(node):
            yield

    def emit_block(self, block: Block) -> None:
        mlir_block = self.block_for(block)
        self.env = {
            name: Typed(value, self.variable_types[name])
            for name, value in zip(
                self.block_arguments[id(block)], mlir_block.arguments, strict=True
            )
        }
        self.move_to(mlir_block)
        for statement in block.statements:
            with self.at(statement):
                self.statement(statement)
        match block.terminator:
            case Jump(target=target):
                with self.at(self.source.tree):
                    cf.BranchOp(self.block_for(target), self.pass_to(target))
            case Branch(test=test, if_true=t, if_false=f):
                with self.at(test):
                    condition = self.condition(test)
                    cf.CondBranchOp(
                        condition,
                        self.block_for(t),
                        self.block_for(f),
                        true_dest_operands=self.pass_to(t),
                        false_dest_operands=self.pass_to(f),
                    )
            case Return() as ret:
                with self.at(ret.node):
                    self.emit_return(ret)
            case None:
                raise AssertionError("block without terminator")

    def pass_to(self, target: Block) -> list[ir.Value]:
        self.block_for(target)
        return [self.env[name].value for name in self.block_arguments[id(target)]]

    # -- statements -----------------------------------------------------------

    def statement(self, node: ast.stmt) -> None:
        match node:
            case ast.Pass():
                pass
            case ast.Expr(value=ast.Constant(value=str())):
                pass  # docstring
            case ast.Expr(value=value):
                self.expression(value)
            case ast.Assign(targets=targets, value=value):
                result = self.expression(value)
                for target in targets:
                    self.assign(target, result, value)
            case ast.AugAssign(target=ast.Subscript() as target, op=op, value=value):
                element, load, store = self.element(target)
                current = Typed(load(), element)
                updated = self.binary(op, current, self.operand(value), node)
                store(self.materialize(updated, element, node).value)
            case ast.AugAssign(target=ast.Attribute() as target, op=op, value=value):
                current = self.operand(target)
                updated = self.binary(op, current, self.operand(value), node)
                self.assign_field(target, updated, node)
            case ast.AugAssign(target=target, op=op, value=value):
                if not isinstance(target, ast.Name):
                    raise self.source.error(
                        target, "only names, fields, and p[i] can be updated"
                    )
                current = self.load(target)
                self.assign(
                    target, self.binary(op, current, self.operand(value), node), node
                )
            case ast.AnnAssign(target=target, annotation=annotation, value=value):
                if not isinstance(target, ast.Name):
                    raise self.source.error(target, "only plain names can be declared")
                if value is None:
                    raise self.source.error(
                        node, f"give '{target.id}' a value when declaring it"
                    )
                declared = self.resolve_type(annotation)
                existing = self.variable_types.get(target.id)
                if existing is not None and existing != declared:
                    raise self.source.error(
                        annotation, f"'{target.id}' is already {existing}"
                    )
                self.variable_types[target.id] = declared
                self.assign(target, self.expression(value), value)
            case _:
                raise self.source.error(node, "unsupported statement")

    def assign(
        self, target: ast.expr, result: Operand | TupleValue | None, node: ast.AST
    ) -> None:
        if isinstance(target, ast.Subscript):
            if result is None or isinstance(result, TupleValue):
                raise self.source.error(node, "only a single value can be stored")
            element, _, store = self.element(target)
            store(self.materialize(result, element, node).value)
            return
        if isinstance(target, ast.Tuple):
            if not isinstance(result, TupleValue) or len(result.items) != len(
                target.elts
            ):
                raise self.source.error(
                    node, f"cannot unpack into {len(target.elts)} names"
                )
            for element, item in zip(target.elts, result.items, strict=True):
                self.assign(element, item, node)
            return
        if isinstance(target, ast.Attribute):
            if result is None or isinstance(result, TupleValue):
                raise self.source.error(node, "only a single value can be stored")
            self.assign_field(target, result, node)
            return
        if not isinstance(target, ast.Name):
            raise self.source.error(target, "only plain names can be assigned")
        if result is None:
            raise self.source.error(node, "this expression has no value")
        if isinstance(result, TupleValue):
            raise self.source.error(
                node, f"cannot assign several values to '{target.id}'"
            )
        declared = self.variable_types.get(target.id)
        typed = self.materialize(result, declared or self.natural_type(result), node)
        if declared is not None and typed.type != declared:
            raise self.source.error(
                node,
                f"'{target.id}' is {declared}, cannot assign {typed.type} (convert with {declared.name}(...))",
            )
        self.variable_types[target.id] = typed.type
        self.env[target.id] = typed

    def field_path(
        self, target: ast.Attribute
    ) -> tuple[ast.expr, ScalarType, list[int], ScalarType]:
        """For ``root.a.b``: the root expression, its struct type, the field
        indices of ``a.b``, and the type of ``b``."""
        names: list[ast.Attribute] = []
        root: ast.expr = target
        while isinstance(root, ast.Attribute):
            names.append(root)
            root = root.value
        if isinstance(root, ast.Subscript):
            _, kind = self.element_address(root)
        else:
            base = self.operand(root)
            if not isinstance(base, Typed):
                raise self.source.error(root, "only struct values have fields")
            kind = base.type
        root_kind = kind
        indices: list[int] = []
        for attribute in reversed(names):
            index, kind = self.field(kind, attribute)
            indices.append(index)
        return root, root_kind, indices, kind

    def field(self, kind: ScalarType, node: ast.Attribute) -> tuple[int, ScalarType]:
        if not isinstance(kind, StructType):
            raise self.source.error(
                node, f"{kind.name} has no fields (only @struct values do)"
            )
        found = kind.field(node.attr)
        if found is None:
            names = ", ".join(name for name, _ in kind.fields)
            raise self.source.error(
                node, f"{kind.name} has no field '{node.attr}' (it has {names})"
            )
        return found

    def assign_field(
        self, target: ast.Attribute, value: Operand, node: ast.AST
    ) -> None:
        """``c.x = v`` gives the variable ``c`` a new struct value (structs are
        values, as in C); ``p[i].x = v`` stores into memory."""
        root, root_kind, indices, kind = self.field_path(target)
        stored = self.materialize(value, kind, node).value
        if isinstance(root, ast.Subscript):
            address, _ = self.element_address(root)
            field_address = llvm.GEPOp(
                llvm.PointerType(),
                address,
                ir.DenseI32ArrayAttr([0, *indices]),
                root_kind.mlir(),
                [],
            ).result
            llvm.StoreOp(stored, field_address)
            return
        if not isinstance(root, ast.Name):
            raise self.source.error(
                target, "assign fields of a variable (c.x = ...) or of p[i]"
            )
        current = self.load(root)
        updated = llvm.InsertValueOp(
            current.value, stored, ir.DenseI64ArrayAttr(indices)
        ).result
        self.env[root.id] = Typed(updated, current.type)

    def emit_return(self, ret: Return) -> None:
        expected = self.signature.results
        is_main = self.function.kind == "main" and not expected
        if ret.value is None:
            if expected:
                message = (
                    f"'{self.function.name}' must return {self.describe(expected)}"
                    + (" on every path" if ret.implicit else "")
                )
                raise self.source.error(ret.node, message)
            if is_main:
                zero = arith.ConstantOp(ir.IntegerAttr(0, ir.IntegerType(32))).result
                func.ReturnOp([zero])
            else:
                func.ReturnOp()
            return
        result = self.expression(ret.value)
        items = result.items if isinstance(result, TupleValue) else [result]
        if not expected:
            raise self.source.error(
                ret.value, f"'{self.function.name}' returns nothing"
            )
        if len(items) != len(expected):
            raise self.source.error(
                ret.value,
                f"'{self.function.name}' must return {self.describe(expected)}",
            )
        values = []
        for item, kind in zip(items, expected, strict=True):
            if item is None:
                raise self.source.error(ret.value, "this expression has no value")
            values.append(self.materialize(item, kind, ret.value).value)
        func.ReturnOp(values)

    @staticmethod
    def describe(kinds: list[ScalarType]) -> str:
        return (
            kinds[0].name
            if len(kinds) == 1
            else f"({', '.join(k.name for k in kinds)})"
        )

    # -- names and types ------------------------------------------------------

    def load(self, node: ast.Name) -> Typed:
        if node.id in self.env:
            return self.env[node.id]
        if node.id in self.all_assigned:
            raise self.source.error(
                node,
                f"'{node.id}' may be unassigned here (it is not assigned on every path)",
            )
        raise self.source.error(node, f"name '{node.id}' is not defined")

    def is_local(self, name: str) -> bool:
        return name in self.env or name in self.all_assigned

    def resolve_quietly(self, name: str) -> object:
        try:
            return self.program.resolve_global(
                self.function, name, lambda: CompileError(name)
            )
        except CompileError:
            return None

    def lookup_global(self, node: ast.Name) -> object:
        return self.program.resolve_global(
            self.function,
            node.id,
            lambda: self.source.error(node, f"name '{node.id}' is not defined"),
        )

    def resolve_type(self, annotation: ast.expr) -> ScalarType:
        kind = type_from_expression(annotation, self.resolve_quietly)
        if kind is None:
            raise self.source.error(annotation, "unsupported type annotation")
        return kind

    @staticmethod
    def natural_type(result: Operand) -> ScalarType:
        return result.type if isinstance(result, Typed) else default_type(result)

    def materialize(self, operand: Operand, kind: ScalarType, node: ast.AST) -> Typed:
        if isinstance(operand, Typed):
            if operand.type == kind:
                return operand
            if operand.type.kind in ("ptr", "cstr", "fn") and kind == ptr:
                return Typed(operand.value, kind)  # any pointer passes as a ptr
            if fits_function_type(operand.type, kind):
                return Typed(operand.value, kind)
            if (
                operand.type.kind == "array"
                and kind.kind == "ptr"
                and kind.element in (None, operand.type.element)
            ):
                # Like a C array, an array passes as a pointer to its first
                # element (valid while the array is).
                return Typed(self.array_pointer(operand.value), kind)
            if kind.kind in ("struct", "array") or operand.type.kind in (
                "struct",
                "array",
            ):
                raise self.source.error(
                    node, f"expected {kind.name}, got {operand.type.name}"
                )
            raise self.source.error(
                node,
                f"expected {kind.name}, got {operand.type.name} (convert with {kind.name}(...))",
            )
        value = operand.value
        if isinstance(value, str):
            if kind.kind not in ("cstr", "ptr"):
                raise self.source.error(node, f"expected {kind.name}, got a string")
            return Typed(self.program.string_pointer(value), kind)
        if isinstance(value, bool):
            if kind.kind != "bool":
                raise self.source.error(node, f"expected {kind.name}, got {value}")
            attr = ir.BoolAttr(value)
            return Typed(arith.ConstantOp(attr).result, kind)
        if isinstance(value, float) or kind.kind == "float":
            if kind.kind != "float":
                raise self.source.error(
                    node, f"expected {kind.name}, got the float {value}"
                )
            float_type = kind.mlir()
            assert isinstance(float_type, ir.FloatType)
            attr = ir.FloatAttr(float(value), float_type)
            return Typed(arith.ConstantOp(attr).result, kind)
        if not kind.is_integer:
            raise self.source.error(node, f"expected {kind.name}, got the int {value}")
        low, high = kind.integer_range()
        if not low <= value <= high:
            raise self.source.error(node, f"{value} does not fit {kind.name}")
        attr = ir.IntegerAttr(value, ir.IntegerType(kind.bits))
        return Typed(arith.ConstantOp(attr).result, kind)

    # -- expressions ----------------------------------------------------------

    def operand(self, node: ast.expr) -> Operand:
        result = self.expression(node)
        if result is None:
            raise self.source.error(node, "this expression has no value")
        if isinstance(result, TupleValue):
            raise self.source.error(node, "a tuple cannot be used here")
        return result

    def expression(self, node: ast.expr) -> Operand | TupleValue | None:
        with self.at(node):
            return self._expression(node)

    def _expression(self, node: ast.expr) -> Operand | TupleValue | None:
        match node:
            case ast.Constant(value=value):
                if isinstance(value, (bool, int, float, str)):
                    return Literal(value)
                raise self.source.error(node, f"unsupported constant {value!r}")
            case ast.Name(id=name):
                if name in self.env or name in self.all_assigned:
                    return self.load(node)
                found = self.lookup_global(node)
                if isinstance(found, (bool, int, float)):
                    return Literal(found)
                if (address := self.function_address(found, node)) is not None:
                    return address
                raise self.source.error(
                    node, f"'{name}' cannot be used as a value here"
                )
            case ast.Attribute(value=base) if is_runtime_value(base, self.is_local):
                value = self.operand(base)
                if not isinstance(value, Typed):
                    raise self.source.error(base, "only struct values have fields")
                index, kind = self.field(value.type, node)
                return Typed(
                    llvm.ExtractValueOp(
                        kind.mlir(), value.value, ir.DenseI64ArrayAttr([index])
                    ).result,
                    kind,
                )
            case ast.Attribute():
                found = static_object(node, self.resolve_quietly)
                if isinstance(found, (bool, int, float)):
                    return Literal(found)
                if (address := self.function_address(found, node)) is not None:
                    return address
                raise self.source.error(
                    node, f"'{ast.unparse(node)}' cannot be used as a value here"
                )
            case ast.Tuple(elts=elements):
                return TupleValue([self.operand(e) for e in elements])
            case ast.BinOp(left=left, op=op, right=right):
                return self.binary(op, self.operand(left), self.operand(right), node)
            case ast.UnaryOp(op=op, operand=inner):
                return self.unary(op, self.operand(inner), node)
            case ast.Compare():
                return self.compare_chain(node)
            case ast.BoolOp():
                return Typed(self.boolean_operation(node), boolean)
            case ast.IfExp():
                return self.conditional(node)
            case ast.Call():
                return self.call(node)
            case ast.Subscript():
                element, load, _ = self.element(node)
                return Typed(load(), element)
            case ast.List():
                return self.array_display(node)
            case _:
                raise self.source.error(
                    node, f"unsupported expression ({type(node).__name__})"
                )

    def unify(self, a: Operand, b: Operand, node: ast.AST) -> tuple[Typed, Typed]:
        if isinstance(a, Literal) and isinstance(b, Literal):
            kind = default_type(a)
            if isinstance(a.value, int) and isinstance(b.value, float):
                kind = f64
            return self.materialize(a, kind, node), self.materialize(b, kind, node)
        if isinstance(a, Literal):
            assert isinstance(b, Typed)
            return self.materialize(a, b.type, node), b
        if isinstance(b, Literal):
            return a, self.materialize(b, a.type, node)
        if a.type != b.type:
            raise self.source.error(
                node,
                f"cannot combine {a.type.name} and {b.type.name}; convert one, e.g. {a.type.name}(...)",
            )
        return a, b

    def binary(
        self, op: ast.operator, left: Operand, right: Operand, node: ast.AST
    ) -> Operand:
        symbol = SYMBOLS.get(type(op), type(op).__name__)
        if isinstance(left, Typed) and left.type.kind == "ptr":
            return self.pointer_offset(op, left, right, node)
        for side in (left, right):
            if isinstance(side, Typed) and side.type.kind in ("struct", "array"):
                raise self.source.error(
                    node, f"{side.type.name} values do not support {symbol}"
                )
        if (
            isinstance(left, Literal)
            and isinstance(right, Literal)
            and not isinstance(left.value, str)
            and not isinstance(right.value, str)
        ):
            try:
                folded = PYTHON_OPERATORS[type(op)](left.value, right.value)
            except (ArithmeticError, TypeError, ValueError) as error:
                raise self.source.error(node, str(error)) from error
            assert isinstance(folded, (int, float, bool))
            return Literal(folded)
        a, b = self.unify(left, right, node)
        kind = a.type
        x, y = a.value, b.value
        if kind.is_integer:
            if isinstance(op, ast.Div):
                convert = arith.SIToFPOp if kind.signed else arith.UIToFPOp
                return Typed(
                    arith.DivFOp(
                        convert(f64.mlir(), x).result, convert(f64.mlir(), y).result
                    ).result,
                    f64,
                )
            signed = kind.signed
            ops: dict[
                type[ast.operator], Callable[[ir.Value, ir.Value], ir.Operation]
            ] = {
                ast.Add: arith.AddIOp,
                ast.Sub: arith.SubIOp,
                ast.Mult: arith.MulIOp,
                ast.BitAnd: arith.AndIOp,
                ast.BitOr: arith.OrIOp,
                ast.BitXor: arith.XOrIOp,
                ast.LShift: arith.ShLIOp,
                ast.RShift: arith.ShRSIOp if signed else arith.ShRUIOp,
                ast.FloorDiv: arith.FloorDivSIOp if signed else arith.DivUIOp,
            }
            if type(op) in ops:
                return Typed(ops[type(op)](x, y).result, kind)
            if isinstance(op, ast.Mod):
                if not signed:
                    return Typed(arith.RemUIOp(x, y).result, kind)
                return Typed(
                    self.python_modulo(arith.RemSIOp(x, y).result, y, kind), kind
                )
        elif kind.kind == "float":
            float_ops: dict[
                type[ast.operator], Callable[[ir.Value, ir.Value], ir.Operation]
            ] = {
                ast.Add: arith.AddFOp,
                ast.Sub: arith.SubFOp,
                ast.Mult: arith.MulFOp,
                ast.Div: arith.DivFOp,
                ast.Pow: math.PowFOp,
            }
            if type(op) in float_ops:
                return Typed(float_ops[type(op)](x, y).result, kind)
            if isinstance(op, ast.FloorDiv):
                return Typed(math.FloorOp(arith.DivFOp(x, y).result).result, kind)
            if isinstance(op, ast.Mod):
                return Typed(
                    self.python_modulo(arith.RemFOp(x, y).result, y, kind), kind
                )
        elif kind.kind == "bool":
            bool_ops = {
                ast.BitAnd: arith.AndIOp,
                ast.BitOr: arith.OrIOp,
                ast.BitXor: arith.XOrIOp,
            }
            if type(op) in bool_ops:
                return Typed(bool_ops[type(op)](x, y).result, kind)
        hint = (
            " (integer powers are not supported yet)" if isinstance(op, ast.Pow) else ""
        )
        raise self.source.error(
            node, f"'{symbol}' is not supported for {kind.name}{hint}"
        )

    def python_modulo(
        self, remainder: ir.Value, divisor: ir.Value, kind: ScalarType
    ) -> ir.Value:
        """Adjusts a truncated remainder to Python's sign-of-divisor rule."""
        zero = self.materialize(
            Literal(0.0 if kind.kind == "float" else 0), kind, self.source.tree
        ).value
        if kind.kind == "float":
            nonzero = arith.CmpFOp(arith.CmpFPredicate.UNE, remainder, zero).result
            negative_r = arith.CmpFOp(arith.CmpFPredicate.OLT, remainder, zero).result
            negative_d = arith.CmpFOp(arith.CmpFPredicate.OLT, divisor, zero).result
            adjusted = arith.AddFOp(remainder, divisor).result
        else:
            nonzero = arith.CmpIOp(arith.CmpIPredicate.NE, remainder, zero).result
            negative_r = arith.CmpIOp(arith.CmpIPredicate.SLT, remainder, zero).result
            negative_d = arith.CmpIOp(arith.CmpIPredicate.SLT, divisor, zero).result
            adjusted = arith.AddIOp(remainder, divisor).result
        signs_differ = arith.XOrIOp(negative_r, negative_d).result
        fix = arith.AndIOp(nonzero, signs_differ).result
        return arith.SelectOp(fix, adjusted, remainder).result

    def unary(self, op: ast.unaryop, operand: Operand, node: ast.AST) -> Operand:
        if isinstance(operand, Literal) and not isinstance(operand.value, str):
            value = operand.value
            match op:
                case ast.USub():
                    return Literal(-value)
                case ast.UAdd():
                    return Literal(+value)
                case ast.Not():
                    return Literal(not value)
                case ast.Invert() if isinstance(value, int):
                    return Literal(~value)
        if isinstance(op, ast.Not):
            truth = self.truth(operand, node)
            one = arith.ConstantOp(ir.BoolAttr(True)).result
            return Typed(arith.XOrIOp(truth, one).result, boolean)
        assert isinstance(operand, Typed)
        kind = operand.type
        if isinstance(op, ast.UAdd) and (kind.is_integer or kind.kind == "float"):
            return operand
        if isinstance(op, ast.USub):
            if kind.kind == "float":
                return Typed(arith.NegFOp(operand.value).result, kind)
            if kind.is_integer:
                zero = self.materialize(Literal(0), kind, node).value
                return Typed(arith.SubIOp(zero, operand.value).result, kind)
        if isinstance(op, ast.Invert) and kind.is_integer:
            ones = arith.ConstantOp(
                ir.IntegerAttr(-1, ir.IntegerType(kind.bits))
            ).result
            return Typed(arith.XOrIOp(operand.value, ones).result, kind)
        raise self.source.error(node, f"unsupported operator for {kind.name}")

    def compare(
        self, op: ast.cmpop, left: Operand, right: Operand, node: ast.AST
    ) -> ir.Value:
        if isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn)):
            raise self.source.error(
                node, "only ==, !=, <, <=, >, >= comparisons are supported"
            )
        a, b = self.unify(left, right, node)
        kind = a.type
        if kind.kind == "float":
            return arith.CmpFOp(FLOAT_PREDICATES[type(op)], a.value, b.value).result
        if kind.is_integer or (
            kind.kind == "bool" and isinstance(op, (ast.Eq, ast.NotEq))
        ):
            table = (
                UNSIGNED_PREDICATES
                if kind.kind in ("uint", "bool")
                else SIGNED_PREDICATES
            )
            return arith.CmpIOp(table[type(op)], a.value, b.value).result
        raise self.source.error(
            node, f"{kind.name} values cannot be compared with that operator"
        )

    def compare_chain(self, node: ast.Compare) -> Operand:
        left = self.operand(node.left)
        pairs = list(zip(node.ops, node.comparators, strict=True))
        op, comparator = pairs[0]
        right = self.operand(comparator)
        result = self.compare(op, left, right, node)
        for op, comparator in pairs[1:]:
            previous = right

            def next_comparison(
                op: ast.cmpop = op,
                comparator: ast.expr = comparator,
                previous: Operand = previous,
            ) -> ir.Value:
                nonlocal right
                right = self.operand(comparator)
                return self.compare(op, previous, right, node)

            result = self.short_circuit(result, next_comparison, is_and=True)
        return Typed(result, boolean)

    def truth(self, operand: Operand, node: ast.AST) -> ir.Value:
        if isinstance(operand, Literal):
            if isinstance(operand.value, str):
                raise self.source.error(node, "a string is not a condition")
            return arith.ConstantOp(ir.BoolAttr(bool(operand.value))).result
        kind = operand.type
        if kind.kind == "bool":
            return operand.value
        if kind.kind == "struct":
            raise self.source.error(node, f"a {kind.name} is not a condition")
        if kind.kind == "array":  # like a list: true when not empty
            length = self.array_length(operand.value)
            zero = arith.ConstantOp(ir.IntegerAttr(0, ir.IntegerType(64))).result
            return arith.CmpIOp(arith.CmpIPredicate.NE, length, zero).result
        zero = self.materialize(Literal(0.0 if kind.kind == "float" else 0), kind, node)
        if kind.is_integer:
            return arith.CmpIOp(
                arith.CmpIPredicate.NE, operand.value, zero.value
            ).result
        if kind.kind == "float":
            return arith.CmpFOp(
                arith.CmpFPredicate.UNE, operand.value, zero.value
            ).result
        raise self.source.error(
            node, f"{kind.name} values cannot be used as conditions"
        )

    def condition(self, node: ast.expr) -> ir.Value:
        return self.truth(self.operand(node), node)

    def short_circuit(
        self, left: ir.Value, right: Callable[[], ir.Value], *, is_and: bool
    ) -> ir.Value:
        """``left and right()`` / ``left or right()`` evaluating ``right`` only
        when needed, as Python does."""
        rhs_block = self.region.append_block()
        merge = self.region.append_block([ir.IntegerType(1)])
        if is_and:
            cf.CondBranchOp(left, rhs_block, merge, false_dest_operands=[left])
        else:
            cf.CondBranchOp(left, merge, rhs_block, true_dest_operands=[left])
        self.move_to(rhs_block)
        value = right()
        cf.BranchOp(merge, [value])
        self.move_to(merge)
        return merge.arguments[0]

    def boolean_operation(self, node: ast.BoolOp) -> ir.Value:
        is_and = isinstance(node.op, ast.And)
        result = self.condition(node.values[0])
        for operand in node.values[1:]:
            result = self.short_circuit(
                result, lambda operand=operand: self.condition(operand), is_and=is_and
            )
        return result

    def conditional(self, node: ast.IfExp) -> Operand:
        test = self.condition(node.test)
        then_block, else_block = self.region.append_block(), self.region.append_block()
        cf.CondBranchOp(test, then_block, else_block)
        self.move_to(then_block)
        then_value = self.operand(node.body)
        then_end = self.current
        self.move_to(else_block)
        else_value = self.operand(node.orelse)
        else_end = self.current
        kind = (
            then_value.type
            if isinstance(then_value, Typed)
            else else_value.type
            if isinstance(else_value, Typed)
            else default_type(then_value)
        )
        merge = self.region.append_block([kind.mlir()])
        for end, value, branch in (
            (then_end, then_value, node.body),
            (else_end, else_value, node.orelse),
        ):
            self.move_to(end)
            cf.BranchOp(merge, [self.materialize(value, kind, branch).value])
        self.move_to(merge)
        return Typed(merge.arguments[0], kind)

    # -- calls ----------------------------------------------------------------

    def call(self, node: ast.Call) -> Operand | TupleValue | None:
        if isinstance(node.func, ast.Subscript):  # Ptr[T](raw)
            kind = type_from_expression(node.func, self.resolve_quietly)
            if kind is None or len(node.args) != 1:
                raise self.source.error(node, "use Ptr[T](pointer) to type a pointer")
            return self.convert(self.operand(node.args[0]), kind, node)
        if is_runtime_value(node.func, self.is_local):
            return self.call_value(node)
        if isinstance(node.func, ast.Name):
            name = node.func.id
            target = self.lookup_global(node.func)
        elif isinstance(node.func, ast.Attribute):
            name = ast.unparse(node.func)
            target = static_object(node.func, self.resolve_quietly)
            if target is None:
                raise self.source.error(node.func, f"'{name}' is not defined")
        else:
            raise self.source.error(node.func, "only named functions can be called")
        from ._program import Function

        kind = scalar_type(target)
        if isinstance(kind, StructType):
            return self.construct(kind, node)
        if node.keywords:
            raise self.source.error(
                node.keywords[0], "compiled calls take positional arguments only"
            )
        if isinstance(target, Function):
            return self.call_function(target, node)
        kind = scalar_type(target)
        if kind is not None:
            if len(node.args) != 1:
                raise self.source.error(node, f"{name}() converts exactly one value")
            return self.convert(self.operand(node.args[0]), kind, node)
        if target is stack:
            return self.allocate(node)
        if target is atomic_add:
            return self.atomic_add(node)
        if target is array:
            return self.make_array(node)
        if target is builtins.len:
            return self.length(node)
        if target is builtins.abs:
            return self.absolute(node)
        if target in (builtins.min, builtins.max):
            return self.extreme(node, is_min=target is builtins.min)
        if target is builtins.print:
            raise self.source.error(
                node.func,
                "print() is not available; declare puts or printf with @program.extern",
            )
        if target is builtins.range:
            raise self.source.error(node.func, "range() can only be used in a for loop")
        raise self.source.error(
            node.func, f"'{name}' cannot be called from compiled code"
        )

    def function_address(self, target: object, node: ast.AST) -> Typed | None:
        """``f`` used as a value: a pointer to the function, typed ``Fn[...]``
        (``None`` if ``target`` is not a function)."""
        from ._program import Function

        if not isinstance(target, Function):
            return None
        callee, signature = self.program.declaration(target)
        kind = value_type(signature)
        if kind is None:
            reason = "C varargs" if signature.variadic else "several results"
            raise self.source.error(
                node, f"{target.name} takes {reason}, so it cannot be a function value"
            )
        pointer = llvm.PointerType()
        if target.kind == "extern":
            if any(k.kind == "struct" for k in (*kind.params, kind.result) if k):
                raise self.source.error(
                    node,
                    f"{target.name} passes a @struct by value, which calls through "
                    "a function value cannot do the C way; wrap it in a function",
                )
            self.program.extern_abi(target)
            return Typed(llvm.AddressOfOp(pointer, target.name).result, kind)
        if target.kind == "main":
            raise self.source.error(node, "main cannot be used as a function value")
        assert callee is not None
        constant = func.ConstantOp(callee.function_type, target.name).result
        # Lowering to LLVM turns the constant into the function's address.
        cast = ir.Operation.create(
            "builtin.unrealized_conversion_cast", results=[pointer], operands=[constant]
        )
        return Typed(cast.results[0], kind)

    def call_value(self, node: ast.Call) -> Operand | None:
        """``f(x)`` where ``f`` holds a function value: an indirect call."""
        callee = self.operand(node.func)
        if not (isinstance(callee, Typed) and isinstance(callee.type, FnType)):
            described = callee.type.name if isinstance(callee, Typed) else "a constant"
            raise self.source.error(
                node.func,
                f"'{ast.unparse(node.func)}' is {described}, not a function value",
            )
        if node.keywords:
            raise self.source.error(
                node.keywords[0], "compiled calls take positional arguments only"
            )
        kind = callee.type
        if len(node.args) != len(kind.params):
            raise self.source.error(
                node,
                f"{ast.unparse(node.func)}() takes {len(kind.params)} arguments, "
                f"got {len(node.args)}",
            )
        values = [
            self.materialize(self.operand(arg), param, arg).value
            for arg, param in zip(node.args, kind.params, strict=True)
        ]
        call = llvm.CallOp(
            [callee.value, *values],
            result_type=kind.result.mlir() if kind.result is not None else None,
        )
        return Typed(call.result, kind.result) if kind.result is not None else None

    def construct(self, kind: StructType, node: ast.Call) -> Typed:
        """``Color(r, g, b, a)`` or ``Color(r=..., ...)``: every field, once."""
        fields = kind.fields
        if len(node.args) > len(fields):
            raise self.source.error(
                node, f"{kind.name}() takes {len(fields)} fields, got {len(node.args)}"
            )
        given: dict[str, ast.expr] = {
            name: arg for (name, _), arg in zip(fields, node.args, strict=False)
        }
        for keyword in node.keywords:
            if keyword.arg is None or kind.field(keyword.arg) is None:
                names = ", ".join(name for name, _ in fields)
                raise self.source.error(
                    keyword,
                    f"{kind.name} has no field '{keyword.arg}' (it has {names})",
                )
            if keyword.arg in given:
                raise self.source.error(
                    keyword, f"field '{keyword.arg}' of {kind.name} is given twice"
                )
            given[keyword.arg] = keyword.value
        missing = [name for name, _ in fields if name not in given]
        if missing:
            raise self.source.error(
                node, f"{kind.name}() is missing {', '.join(repr(m) for m in missing)}"
            )
        value = llvm.UndefOp(kind.mlir()).result
        for index, (name, field_kind) in enumerate(fields):
            arg = given[name]
            field_value = self.materialize(self.operand(arg), field_kind, arg).value
            value = llvm.InsertValueOp(
                value, field_value, ir.DenseI64ArrayAttr([index])
            ).result
        return Typed(value, kind)

    def call_function(
        self, target: Function[..., Any], node: ast.Call
    ) -> Operand | TupleValue | None:
        callee, signature = self.program.declaration(target)
        fixed = len(signature.params)
        if len(node.args) != fixed and not (
            signature.variadic and len(node.args) > fixed
        ):
            expected = f"at least {fixed}" if signature.variadic else str(fixed)
            raise self.source.error(
                node,
                f"{target.name}() takes {expected} arguments, got {len(node.args)}",
            )
        values = [
            self.materialize(self.operand(arg), kind, arg).value
            for arg, (_, kind) in zip(node.args, signature.params, strict=False)
        ]
        if target.kind == "extern":
            extra = [self.promote(self.operand(arg), arg) for arg in node.args[fixed:]]
            return self.call_extern(target, values, extra)
        assert callee is not None
        results = func.call(callee, values).results
        typed = [
            Typed(value, kind)
            for value, kind in zip(results, signature.results, strict=True)
        ]
        if not typed:
            return None
        if signature.returns_tuple:
            return TupleValue(list(typed))
        return typed[0]

    def call_extern(
        self, target: Function[..., Any], values: list[ir.Value], extra: list[ir.Value]
    ) -> Typed | None:
        """A call to a C function, passing structs as the C ABI requires (see
        ``_abi``)."""
        abi = self.program.extern_abi(target)
        operands: list[ir.Value] = []
        result_slot = None
        if abi.result is not None and abi.result.passing == "byval":
            result_slot = self.stack_slot(abi.result.kind)
            operands.append(result_slot)
        for lowered, value in zip(abi.params, values, strict=True):
            if lowered.passing == "direct":
                operands.append(value)
                continue
            slot = self.stack_slot(lowered.kind)
            llvm.StoreOp(value, slot)
            if lowered.passing == "pieces":
                operands += [
                    llvm.LoadOp(piece, self.byte_offset(slot, offset)).result
                    for offset, piece in lowered.pieces
                ]
            else:
                operands.append(slot)
        result = abi.result
        result_type: ir.Type | None = None
        if result is not None and result.passing == "direct":
            result_type = result.kind.mlir()
        elif result is not None and result.passing == "pieces":
            pieces = [piece for _, piece in result.pieces]
            result_type = pieces[0] if len(pieces) == 1 else llvm.StructType(pieces)
        empty = ir.DictAttr({})
        call = llvm.CallOp(
            operands + extra,
            var_callee_type=abi.function_type
            if target.module.signature(target).variadic
            else None,
            callee=target.name,
            result_type=result_type,
            arg_attrs=ir.ArrayAttr([*abi.arg_attrs, *(empty for _ in extra)]),
            res_attrs=abi.res_attrs,
        )
        if result is None:
            return None
        kind = result.kind
        if result.passing == "direct":
            return Typed(call.result, kind)
        if result.passing == "byval":
            assert result_slot is not None
            return Typed(llvm.LoadOp(kind.mlir(), result_slot).result, kind)
        slot = self.stack_slot(kind)
        if len(result.pieces) == 1:
            llvm.StoreOp(call.result, slot)
        else:
            for index, (offset, piece) in enumerate(result.pieces):
                part = llvm.ExtractValueOp(
                    piece, call.result, ir.DenseI64ArrayAttr([index])
                ).result
                llvm.StoreOp(part, self.byte_offset(slot, offset))
        return Typed(llvm.LoadOp(kind.mlir(), slot).result, kind)

    def stack_slot(self, kind: ScalarType) -> ir.Value:
        """Stack memory for a ``kind`` value, in whole 8-byte words so that
        register-sized pieces can be loaded from it."""
        words = llvm.ArrayType(ir.IntegerType(64), -(-kind.size // 8))
        with ir.InsertionPoint.at_block_begin(self.entry_block):
            one = arith.ConstantOp(ir.IntegerAttr(1, ir.IntegerType(64))).result
            return llvm.AllocaOp(
                llvm.PointerType(), one, words, alignment=max(8, kind.alignment)
            ).result

    @staticmethod
    def byte_offset(address: ir.Value, offset: int) -> ir.Value:
        if offset == 0:
            return address
        return llvm.GEPOp(
            llvm.PointerType(),
            address,
            ir.DenseI32ArrayAttr([offset]),
            ir.IntegerType(8),
            [],
        ).result

    def promote(self, operand: Operand, node: ast.AST) -> ir.Value:
        """C's default argument promotions for the ``...`` part of a variadic
        call: small integers and bool become int (i32), f32 becomes double."""
        if isinstance(operand, Literal):
            value = operand.value
            if isinstance(value, str):
                return self.materialize(operand, cstr, node).value
            if isinstance(value, float):
                return self.materialize(operand, f64, node).value
            number = int(value)
            low, high = i32.integer_range()
            return self.materialize(
                Literal(number), i32 if low <= number <= high else i64, node
            ).value
        kind = operand.type
        if kind.kind == "struct":
            raise self.source.error(
                node, f"a {kind.name} cannot be passed as a variadic argument"
            )
        if kind.kind == "array":
            return self.array_pointer(operand.value)
        if kind.kind == "bool" or (kind.is_integer and kind.bits < 32):
            return self.convert(operand, i32, node).value
        if kind.kind == "float" and kind.bits < 64:
            return self.convert(operand, f64, node).value
        return operand.value

    def convert(self, operand: Operand, kind: ScalarType, node: ast.AST) -> Typed:
        if isinstance(operand, Literal):
            if isinstance(operand.value, str):
                return self.materialize(operand, kind, node)
            try:
                converted = kind(operand.value)
            except (OverflowError, TypeError, ValueError) as error:
                raise self.source.error(node, str(error)) from error
            assert isinstance(converted, (int, float, bool))
            return self.materialize(Literal(converted), kind, node)
        source = operand.type
        value = operand.value
        if source == kind:
            return operand
        target_type = kind.mlir()
        if kind.kind == "bool":
            return Typed(self.truth(operand, node), kind)
        if source.kind == "bool" and kind.is_integer:
            return Typed(arith.ExtUIOp(target_type, value).result, kind)
        if source.is_integer and kind.is_integer:
            if kind.bits > source.bits:
                extend = arith.ExtSIOp if source.signed else arith.ExtUIOp
                return Typed(extend(target_type, value).result, kind)
            if kind.bits < source.bits:
                return Typed(arith.TruncIOp(target_type, value).result, kind)
            return Typed(value, kind)  # same width, other signedness
        if source.is_integer and kind.kind == "float":
            to_float = arith.SIToFPOp if source.signed else arith.UIToFPOp
            return Typed(to_float(target_type, value).result, kind)
        if source.kind == "float" and kind.is_integer:
            to_int = arith.FPToSIOp if kind.signed else arith.FPToUIOp
            return Typed(to_int(target_type, value).result, kind)
        if source.kind == "float" and kind.kind == "float":
            resize = arith.ExtFOp if kind.bits > source.bits else arith.TruncFOp
            return Typed(resize(target_type, value).result, kind)
        if source.kind in ("ptr", "cstr", "fn") and kind.kind in ("ptr", "fn"):
            return Typed(value, kind)  # pointers are untyped in memory
        if source.kind == "array" and kind.kind == "ptr":
            if kind.element not in (None, source.element):
                raise self.source.error(
                    node, f"cannot point at {source.name}'s items as {kind.name}"
                )
            # ptr(xs) / Ptr[T](xs): the address of the first item, as C
            # functions take arrays (valid while the array is).
            return Typed(self.array_pointer(value), kind)
        raise self.source.error(node, f"cannot convert {source.name} to {kind.name}")

    def allocate(self, node: ast.Call) -> Typed:
        """``stack(T, count=1)``: an ``llvm.alloca``, hoisted to the entry
        block when the count is constant so loops do not grow the stack."""
        if not 1 <= len(node.args) <= 2:
            raise self.source.error(node, "stack(T, count=1) takes a type and a count")
        element = type_from_expression(node.args[0], self.resolve_quietly)
        if element is None:
            raise self.source.error(node.args[0], "stack() needs a type such as i32")
        count = literal_int(node.args[1]) if len(node.args) == 2 else 1
        if count is not None and count < 1:
            raise self.source.error(node.args[1], "stack() needs a positive count")
        pointer = llvm.PointerType()
        if count is not None:
            with ir.InsertionPoint.at_block_begin(self.entry_block):
                size = arith.ConstantOp(
                    ir.IntegerAttr(count, ir.IntegerType(64))
                ).result
                address = llvm.AllocaOp(pointer, size, element.mlir()).result
        else:
            size = self.convert(self.operand(node.args[1]), i64, node.args[1]).value
            address = llvm.AllocaOp(pointer, size, element.mlir()).result
        return Typed(address, pointer_to(element))

    def element(
        self, node: ast.Subscript
    ) -> tuple[ScalarType, Callable[[], ir.Value], Callable[[ir.Value], None]]:
        """``xs[i]`` or ``p[i]``: the element type, and functions loading and
        storing the element."""
        base = self.operand(node.value)
        if isinstance(base, Typed) and base.type.kind == "array":
            array_value = base.value
            element = base.type.element
            assert element is not None
            index = self.array_index(array_value, self.operand(node.slice), node.slice)

            def load() -> ir.Value:
                return memref.LoadOp(array_value, [index]).result

            def store(value: ir.Value) -> None:
                memref.StoreOp(value, array_value, [index])

            return element, load, store
        address, element = self.element_address(node, base)

        def load_pointer() -> ir.Value:
            return llvm.LoadOp(element.mlir(), address).result

        def store_pointer(value: ir.Value) -> None:
            llvm.StoreOp(value, address)

        return element, load_pointer, store_pointer

    def check(self, condition: ir.Value, message: str, node: ast.AST) -> None:
        """Stops the program unless ``condition`` holds: writes
        ``file:line: message`` to stderr (unbuffered, so it is never lost) and
        aborts."""
        line = self.source.first_line + getattr(node, "lineno", 1) - 1
        text = f"{self.source.filename}:{line}: {message}\n"
        true = arith.ConstantOp(ir.BoolAttr(True)).result
        failed = arith.XOrIOp(condition, true).result
        i32_type, i64_type = ir.IntegerType(32), ir.IntegerType(64)
        pointer = llvm.PointerType()
        self.program.runtime_function(
            "write", llvm.FunctionType(i64_type, [i32_type, pointer, i64_type])
        )
        self.program.runtime_function("abort", llvm.FunctionType(llvm.VoidType(), []))
        branch = scf.IfOp(failed)
        with ir.InsertionPoint(branch.then_block):
            stderr = arith.ConstantOp(ir.IntegerAttr(2, i32_type)).result
            size = len(text.encode())
            count = arith.ConstantOp(ir.IntegerAttr(size, i64_type)).result
            message_pointer = self.program.string_pointer(text)
            llvm.CallOp(
                [stderr, message_pointer, count], callee="write", result_type=i64_type
            )
            llvm.CallOp([], callee="abort")

    def array_length(self, array_value: ir.Value) -> ir.Value:
        """``len(xs)`` as an i64."""
        zero = arith.ConstantOp(ir.IntegerAttr(0, ir.IndexType())).result
        size = memref.DimOp(array_value, zero).result
        return arith.IndexCastOp(ir.IntegerType(64), size).result

    def array_index(
        self, array_value: ir.Value, index: Operand, node: ast.AST
    ) -> ir.Value:
        """The memref index of ``xs[index]`` with Python's meaning (negative
        indices count from the end), stopping the program when out of range."""
        if (isinstance(index, Typed) and not index.type.is_integer) or (
            isinstance(index, Literal) and type(index.value) is not int
        ):
            raise self.source.error(node, "array indices must be integers")
        position = self.convert(index, i64, node).value
        length = self.array_length(array_value)
        if not (isinstance(index, Literal) and int(index.value) >= 0):
            zero = arith.ConstantOp(ir.IntegerAttr(0, ir.IntegerType(64))).result
            negative = arith.CmpIOp(arith.CmpIPredicate.SLT, position, zero).result
            from_end = arith.AddIOp(position, length).result
            position = arith.SelectOp(negative, from_end, position).result
        # Unsigned, a negative position is huge, so this checks both ends.
        in_range = arith.CmpIOp(arith.CmpIPredicate.ULT, position, length).result
        self.check(in_range, "index out of range", node)
        return arith.IndexCastOp(ir.IndexType(), position).result

    def new_array(
        self, element: ScalarType, length: ir.Value, *, zero: bool
    ) -> ir.Value:
        """A heap array of ``length`` (an i64) ``element`` values, freed by
        buffer deallocation once unused."""
        count = arith.IndexCastOp(ir.IndexType(), length).result
        array_value = memref.AllocOp(
            ir.MemRefType([None], element.mlir()), [count]
        ).memref
        if zero:
            start = arith.ConstantOp(ir.IntegerAttr(0, ir.IndexType())).result
            one = arith.ConstantOp(ir.IntegerAttr(1, ir.IndexType())).result
            empty = self.materialize(
                Literal(False if element.kind == "bool" else 0),
                element,
                self.source.tree,
            ).value
            loop = scf.ForOp(start, count, one)
            with ir.InsertionPoint(loop.body):
                memref.StoreOp(empty, array_value, [loop.induction_variable])
        return array_value

    def array_display(self, node: ast.List) -> Typed:
        """``[a, b, c]``: a new array of the elements."""
        kind = self.display_types.get(id(node))
        if kind is None or kind.element is None:
            if not node.elts:
                raise self.source.error(
                    node, "give an empty list its type, e.g. xs: Array[i32] = []"
                )
            kind = array_of(self.natural_type(self.operand(node.elts[0])))
        element = kind.element
        assert element is not None
        if element.kind not in ARRAY_ELEMENT_KINDS:
            raise self.source.error(
                node, f"arrays hold integers, floats, or bools, not {element.name}"
            )
        length = arith.ConstantOp(ir.IntegerAttr(len(node.elts), ir.IntegerType(64)))
        array_value = self.new_array(element, length.result, zero=False)
        for position, item in enumerate(node.elts):
            value = self.materialize(self.operand(item), element, item).value
            index = arith.ConstantOp(ir.IntegerAttr(position, ir.IndexType())).result
            memref.StoreOp(value, array_value, [index])
        return Typed(array_value, kind)

    def make_array(self, node: ast.Call) -> Typed:
        """``array(T, length)``: a new array of zeros."""
        if len(node.args) != 2:
            raise self.source.error(node, "array(T, length) takes a type and a length")
        element = type_from_expression(node.args[0], self.resolve_quietly)
        if element is None or element.kind not in ARRAY_ELEMENT_KINDS:
            raise self.source.error(
                node.args[0],
                "array() holds integers, floats, or bools, e.g. array(i32, n)",
            )
        length = self.convert(self.operand(node.args[1]), i64, node.args[1]).value
        zero = arith.ConstantOp(ir.IntegerAttr(0, ir.IntegerType(64))).result
        valid = arith.CmpIOp(arith.CmpIPredicate.SGE, length, zero).result
        self.check(valid, "array length must not be negative", node.args[1])
        return Typed(self.new_array(element, length, zero=True), array_of(element))

    def length(self, node: ast.Call) -> Typed:
        if len(node.args) != 1:
            raise self.source.error(node, "len() takes one argument")
        value = self.operand(node.args[0])
        if not (isinstance(value, Typed) and value.type.kind == "array"):
            raise self.source.error(node.args[0], "len() takes an array")
        return Typed(self.array_length(value.value), i64)

    def array_pointer(self, array_value: ir.Value) -> ir.Value:
        """A pointer to an array's first element, to pass to C."""
        address = memref.ExtractAlignedPointerAsIndexOp(array_value).aligned_pointer
        integer = arith.IndexCastOp(ir.IntegerType(64), address).result
        return llvm.IntToPtrOp(llvm.PointerType(), integer).result

    def element_address(
        self, node: ast.Subscript, base: Operand | None = None
    ) -> tuple[ir.Value, ScalarType]:
        """The address of ``p[i]`` and the element type."""
        if base is None:
            base = self.operand(node.value)
        if not (isinstance(base, Typed) and base.type.kind == "ptr"):
            raise self.source.error(
                node.value, "only arrays and pointers (Ptr[T]) can be indexed"
            )
        element = base.type.element
        if element is None:
            raise self.source.error(
                node.value, "index a typed pointer; convert with Ptr[i32](p) first"
            )
        index = self.operand(node.slice)
        if isinstance(index, Literal) and type(index.value) is int and index.value == 0:
            return base.value, element
        if (isinstance(index, Typed) and not index.type.is_integer) or (
            isinstance(index, Literal) and type(index.value) is not int
        ):
            raise self.source.error(node.slice, "pointer indices must be integers")
        offset = self.convert(index, i64, node.slice).value
        dynamic = -(2**31)  # LLVM::GEPOp::kDynamicIndex
        address = llvm.GEPOp(
            llvm.PointerType(),
            base.value,
            ir.DenseI32ArrayAttr([dynamic]),
            element.mlir(),
            [offset],
        ).result
        return address, element

    def pointer_offset(
        self, op: ast.operator, base: Typed, count: Operand, node: ast.AST
    ) -> Typed:
        """``p + n`` and ``p - n``: the address ``n`` values further or back."""
        if base.type.element is None or not isinstance(op, (ast.Add, ast.Sub)):
            raise self.source.error(
                node, "pointer arithmetic is p + n or p - n on a typed pointer (Ptr[T])"
            )
        if (isinstance(count, Typed) and not count.type.is_integer) or (
            isinstance(count, Literal) and type(count.value) is not int
        ):
            raise self.source.error(node, "a pointer moves by an integer count")
        offset = self.convert(count, i64, node).value
        if isinstance(op, ast.Sub):
            zero = self.materialize(Literal(0), i64, node).value
            offset = arith.SubIOp(zero, offset).result
        dynamic = -(2**31)  # LLVM::GEPOp::kDynamicIndex
        address = llvm.GEPOp(
            llvm.PointerType(),
            base.value,
            ir.DenseI32ArrayAttr([dynamic]),
            base.type.element.mlir(),
            [offset],
        ).result
        return Typed(address, base.type)

    def atomic_add(self, node: ast.Call) -> Operand:
        """``atomic_add(p, delta)``: one indivisible read-add-write; the old value."""
        if len(node.args) != 2:
            raise self.source.error(node, "atomic_add() takes a pointer and an amount")
        pointer = self.operand(node.args[0])
        element = pointer.type.element if isinstance(pointer, Typed) else None
        if element is None or not element.is_integer or pointer.type.kind != "ptr":
            raise self.source.error(
                node.args[0],
                "atomic_add() takes a pointer to an integer, like Ptr[i64]",
            )
        delta = self.materialize(self.operand(node.args[1]), element, node.args[1])
        # A call for now, which buffer deallocation accepts (it rejects
        # llvm.atomicrmw, whose memory effects it cannot see); linking
        # replaces it with the atomic instruction (see _program._lower_atomics).
        integer = element.mlir()
        name = f"{ATOMIC_ADD}{element.bits}"
        self.program.runtime_function(
            name, llvm.FunctionType(integer, [llvm.PointerType(), integer])
        )
        call = llvm.CallOp(
            [pointer.value, delta.value], callee=name, result_type=integer
        )
        return Typed(call.result, element)

    def absolute(self, node: ast.Call) -> Operand:
        if len(node.args) != 1:
            raise self.source.error(node, "abs() takes one argument")
        operand = self.operand(node.args[0])
        if isinstance(operand, Literal) and isinstance(operand.value, (int, float)):
            return Literal(abs(operand.value))
        assert isinstance(operand, Typed)
        if operand.type.kind == "float":
            return Typed(math.AbsFOp(operand.value).result, operand.type)
        if operand.type.kind == "uint":
            return operand
        if operand.type.kind == "int":
            zero = self.materialize(Literal(0), operand.type, node).value
            negative = arith.CmpIOp(arith.CmpIPredicate.SLT, operand.value, zero).result
            negated = arith.SubIOp(zero, operand.value).result
            return Typed(
                arith.SelectOp(negative, negated, operand.value).result, operand.type
            )
        raise self.source.error(node, f"abs() does not accept {operand.type.name}")

    def extreme(self, node: ast.Call, *, is_min: bool) -> Operand:
        if len(node.args) < 2:
            raise self.source.error(node, "min()/max() take two or more arguments")
        best = self.operand(node.args[0])
        for arg in node.args[1:]:
            candidate = self.operand(arg)
            a, b = self.unify(best, candidate, node)
            # Python keeps the earlier value unless the later one is strictly better.
            better = self.compare(ast.Lt() if is_min else ast.Gt(), b, a, node)
            best = Typed(arith.SelectOp(better, b.value, a.value).result, a.type)
        return best
