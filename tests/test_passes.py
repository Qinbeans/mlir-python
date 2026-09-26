import dataclasses

import pytest

import mlir_python as ir
from mlir_python import passes
from mlir_python.dialects import func

FOLDABLE = """
func.func @f() -> i32 {
  %a = arith.constant 2 : i32
  %b = arith.constant 3 : i32
  %c = arith.addi %a, %b : i32
  %d = arith.addi %a, %b : i32
  %e = arith.muli %c, %d : i32
  return %e : i32
}
"""

INLINABLE = (
    "func.func private @one() -> i32 {\n"
    "  %0 = arith.constant 1 : i32\n  return %0 : i32\n}\n"
    "func.func @main() -> i32 {\n"
    "  %0 = func.call @one() : () -> i32\n  return %0 : i32\n}"
)


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def test_typed_pipeline_runs() -> None:
    module = ir.Module.parse(FOLDABLE)
    pm = ir.PassManager(ir.Module, [passes.Canonicalizer()])
    assert str(pm) == "builtin.module(canonicalize)"
    pm.run(module)
    assert "arith.constant 25 : i32" in str(module)


def test_nested_pipeline() -> None:
    module = ir.Module.parse(FOLDABLE)
    pm = ir.PassManager(ir.Module, [ir.Nested(func.FuncOp, [passes.CSE()])])
    assert str(pm) == "builtin.module(func.func(cse))"
    pm.run(module)
    assert str(module).count("arith.addi") == 1


def test_options_render_only_when_changed() -> None:
    canonicalize = passes.Canonicalizer(
        max_iterations=3,
        region_simplify=passes.GreedySimplifyRegionLevel.AGGRESSIVE,
        disable_patterns=["a", "b"],
    )
    assert str(canonicalize) == (
        "canonicalize{region-simplify=aggressive max-iterations=3 disable-patterns={a,b}}"
    )
    assert str(passes.Canonicalizer(max_iterations=10)) == "canonicalize"
    ir.PassManager(ir.Module, [canonicalize]).run(ir.Module.parse(FOLDABLE))


def test_passes_are_immutable_values() -> None:
    a = passes.Canonicalizer(disable_patterns=["x"])
    assert a == passes.Canonicalizer(disable_patterns=("x",))
    assert hash(a) == hash(passes.Canonicalizer(disable_patterns=["x"]))
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.max_iterations = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        passes.Canonicalizer(10)  # type: ignore[misc]  # options are keyword-only


def test_pipeline_on_other_anchor() -> None:
    module = ir.Module.parse(FOLDABLE)
    fn = module.body.operations[0]
    assert isinstance(fn, func.FuncOp)
    pm = ir.PassManager(func.FuncOp, [passes.CSE()])
    pm.run(fn)
    with pytest.raises(TypeError, match="runs on FuncOp, not Module"):
        pm.run(module)


def test_any_anchor_accepts_every_operation() -> None:
    module = ir.Module.parse(FOLDABLE)
    ir.PassManager(ir.Operation, [passes.Canonicalizer()]).run(module)
    assert "arith.constant 25" in str(module)


def test_inline_and_symbol_dce() -> None:
    module = ir.Module.parse(INLINABLE)
    ir.PassManager(ir.Module, [passes.Inliner(), passes.SymbolDCE()]).run(module)
    assert "func.call" not in str(module) and "@one" not in str(module)


def test_every_generated_pass_parses() -> None:
    generated = [getattr(passes, name) for name in passes.__all__]
    pass_classes = [
        c
        for c in generated
        if isinstance(c, type) and issubclass(c, passes.Pass) and c is not passes.Pass
    ]
    assert len(pass_classes) >= 20
    for cls in pass_classes:
        anchor = cls.ANCHOR or ir.Operation
        ir.PassManager(anchor, [cls()])  # construction checks placement
        ir.PassManager.parse(f"builtin.module({cls()})")  # MLIR accepts the text


def test_failures_raise_mlir_error() -> None:
    module = ir.Module.parse(FOLDABLE)
    fn = module.body.operations[0]
    assert isinstance(fn, func.FuncOp)
    fn.function_type = ir.FunctionType([], [ir.F32Type()])  # now inconsistent
    with pytest.raises(ir.MLIRError, match="pass pipeline failed"):
        ir.PassManager(ir.Module, [passes.CSE()]).run(module)


def test_verify_each_can_be_disabled() -> None:
    module = ir.Module.parse(FOLDABLE)
    fn = module.body.operations[0]
    assert isinstance(fn, func.FuncOp)
    fn.function_type = ir.FunctionType([], [ir.F32Type()])
    ir.PassManager(ir.Module, [passes.CSE()], verify_each=False).run(module)


def test_textual_fallback() -> None:
    module = ir.Module.parse(FOLDABLE)
    parsed = ir.PassManager.parse("builtin.module(canonicalize, func.func(cse))")
    assert parsed.anchor == "builtin.module"
    parsed.run(module)
    with pytest.raises(ValueError, match="invalid pass pipeline"):
        ir.PassManager.parse("builtin.module(no-such-pass)")


def test_handles_into_rewritten_ir_become_stale() -> None:
    module = ir.Module.parse(FOLDABLE)
    fn = module.body.operations[0]
    assert isinstance(fn, func.FuncOp)
    add = fn.body.blocks[0].operations[2]
    value = add.result
    block = fn.body.blocks[0]
    ir.PassManager(ir.Module, [passes.Canonicalizer()]).run(module)
    for stale in (
        lambda: add.name,
        lambda: value.type,
        lambda: len(block),
        lambda: fn.sym_name,
    ):
        with pytest.raises(ValueError, match="stale"):
            stale()
    # The operation the pass ran on stays valid; navigate again from it.
    fresh = module.body.operations[0]
    assert isinstance(fresh, func.FuncOp) and fresh.sym_name == "f"


def test_ancestors_of_the_pass_root_stay_valid() -> None:
    module = ir.Module.parse(FOLDABLE)
    fn = module.body.operations[0]
    assert isinstance(fn, func.FuncOp)
    inner = fn.body.blocks[0].operations[0]
    ir.PassManager(func.FuncOp, [passes.CSE()]).run(fn)
    assert fn.sym_name == "f" and module.name == "builtin.module"
    with pytest.raises(ValueError, match="stale"):
        _ = inner.name


def test_detached_ir_is_unaffected_and_insertable_after_a_pass() -> None:
    from mlir_python.dialects import arith

    module = ir.Module.parse(FOLDABLE)
    const = arith.ConstantOp(ir.IntegerAttr(1, ir.IntegerType(32)))
    ir.PassManager(ir.Module, [passes.Canonicalizer()]).run(module)
    module.body.append(const)
    assert const.value == ir.IntegerAttr(1, ir.IntegerType(32))
