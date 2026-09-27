"""Passes written in Python, running inside MLIR pass pipelines."""

import dataclasses

import pytest

import mlir_python as ir
from mlir_python import codegen, irdl, passes
from mlir_python.dialects import arith, func

FUNCTIONS = """
func.func @f(%x: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %y = arith.addi %x, %zero : i32
  return %y : i32
}
func.func @g(%x: i32) -> i32 {
  return %x : i32
}
"""

SEEN: list[tuple[str, int]] = []


@dataclasses.dataclass(frozen=True, kw_only=True)
class CountOps(passes.PythonPass):
    """Records how many operations each function has."""

    ANCHOR = func.FuncOp

    def run(self, op: ir.Operation) -> None:
        assert isinstance(op, func.FuncOp)
        names: list[str] = []
        op.walk(lambda inner: names.append(inner.name))
        SEEN.append((op.sym_name, len(names)))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Fail(passes.PythonPass):
    """Always fails."""

    message: str = "no"

    def run(self, op: ir.Operation) -> None:
        raise ValueError(self.message)


tripling = irdl.Dialect("tripling")


@tripling.operation
class Triple(irdl.Op):
    value = irdl.Operand(irdl.Is(lambda: ir.IntegerType(32)))
    result = irdl.Result(irdl.Is(lambda: ir.IntegerType(32)))


@dataclasses.dataclass(frozen=True, kw_only=True)
class LowerTriple(passes.PythonPass):
    """Rewrites tripling.triple as arith.muli."""

    def run(self, op: ir.Operation) -> None:
        for triple in Triple.all(op):
            with ir.InsertionPoint.before(triple.operation), triple.operation.location:
                three = arith.ConstantOp(ir.IntegerAttr(3, ir.IntegerType(32)))
                product = arith.MulIOp(triple.value, three.result).result
            triple.replace_with([product])


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def test_python_passes_run_between_mlir_passes() -> None:
    SEEN.clear()
    module = ir.Module.parse(FUNCTIONS)
    count = CountOps()
    ir.PassManager(
        ir.Module,
        [
            ir.Nested(func.FuncOp, [count]),
            passes.Canonicalizer(),
            ir.Nested(func.FuncOp, [count]),
        ],
    ).run(module)
    # Canonicalization folded away the addition of zero in @f.
    assert sorted(SEEN) == [("f", 2), ("f", 4), ("g", 2), ("g", 2)]


def test_python_passes_rewrite_ir() -> None:
    i32 = ir.IntegerType(32)
    module = ir.Module()
    with ir.Location.unknown():
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp("nine_times", ir.FunctionType([i32], [i32]))
        with ir.InsertionPoint(fn.add_entry_block()):
            (x,) = fn.arguments
            once = Triple.create(x)
            func.ReturnOp([Triple.create(once.result).result])
    ir.PassManager(ir.Module, [LowerTriple(), passes.Canonicalizer()]).run(module)
    assert "tripling." not in str(module)
    nine_times = next(o for o in module.body.operations if isinstance(o, func.FuncOp))
    assert codegen.compile(module).function(nine_times)(4) == 36


def test_exceptions_fail_the_pipeline_and_propagate() -> None:
    module = ir.Module.parse(FUNCTIONS)
    first = next(iter(module.body.operations))
    with pytest.raises(ValueError, match="stop here"):
        ir.PassManager(ir.Module, [Fail(message="stop here")]).run(module)
    # The pipeline ran, so handles into the module are stale, as after any run.
    with pytest.raises(ValueError, match="stale"):
        _ = first.name


def test_equal_passes_share_a_registration() -> None:
    assert str(Fail(message="a")) == str(Fail(message="a"))
    assert str(Fail(message="a")) != str(Fail(message="b"))
    assert str(CountOps()).startswith("python-count-ops-")


def test_anchors_are_checked() -> None:
    with pytest.raises(ValueError, match="wrap it in Nested"):
        ir.PassManager(ir.Module, [CountOps()])
