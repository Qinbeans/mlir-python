"""A module's own passes, run on the program it links (``Module(passes=...)``)."""

import dataclasses

import mlir_python as ir
from mlir_python import passes
from mlir_python.dialects import arith, func
from mlir_python.lang import Module, array, i32, i64


@dataclasses.dataclass(frozen=True, kw_only=True)
class DoublesTriple(passes.PythonPass):
    """Rewrites every ``arith.constant 2 : i32`` as 3."""

    ANCHOR = func.FuncOp

    def run(self, op: ir.Operation) -> None:
        def visit(inner: ir.Operation) -> None:
            if inner.name != "arith.constant" or str(inner.result.type) != "i32":
                return
            if str(inner.attributes["value"]) == "2 : i32":
                with ir.InsertionPoint.before(inner), inner.location:
                    three = arith.ConstantOp(ir.IntegerAttr(3, ir.IntegerType(32)))
                inner.result.replace_all_uses_with(three.result)

        op.walk(visit)


SEEN: list[str] = []


@dataclasses.dataclass(frozen=True, kw_only=True)
class Snapshot(passes.PythonPass):
    """Records the linked program's text, as the module's passes see it."""

    def run(self, op: ir.Operation) -> None:
        SEEN.append(str(op))


def test_a_modules_passes_rewrite_what_runs() -> None:
    m = Module(
        "doubling",
        passes=[ir.Nested(func.FuncOp, [DoublesTriple()]), passes.Canonicalizer()],
    )

    @m.function
    def twice(x: i32) -> i32:
        return x * 2

    assert twice(5) == 15
    assert "arith.constant 3" in str(m.linked())
    assert len(m.passes) == 2


def test_passes_see_the_whole_program_before_arrays_are_freed() -> None:
    helpers = Module("helpers")

    @helpers.function
    def sum_to(n: i64) -> i64:
        xs = array(i64, n)
        for i in range(n):
            xs[i] = i
        total = 0
        for x in xs:
            total += x
        return total

    m = Module("root", passes=[Snapshot()])

    @m.function
    def run(n: i64) -> i64:
        return sum_to(n)

    SEEN.clear()
    assert run(5) == 10
    (seen,) = SEEN
    assert "@sum_to" in seen  # an imported module's function, linked in
    assert "memref.alloc" in seen
    assert "memref.dealloc" not in seen  # frees come after the passes
    assert "scf.for" in seen  # control flow is already structured
    assert "memref.dealloc" in str(m.linked())


def test_added_passes_link_again() -> None:
    m = Module("later")

    @m.function
    def twice(x: i32) -> i32:
        return x * 2

    assert "arith.constant 3" not in str(m.linked())
    m.add_passes(ir.Nested(func.FuncOp, [DoublesTriple()]), passes.Canonicalizer())
    assert "arith.constant 3" in str(m.linked())
    assert twice(5) == 15
