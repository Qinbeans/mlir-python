import pytest

import mlir_python as ir


def test_context_options() -> None:
    ctx = ir.Context(allow_unregistered_dialects=True, multithreading=False)
    assert ctx.allow_unregistered_dialects
    assert not ctx.multithreading
    ctx.allow_unregistered_dialects = False
    assert not ctx.allow_unregistered_dialects


def test_dialects_load_on_demand() -> None:
    ctx = ir.Context()
    assert {"arith", "func", "scf"} <= set(ctx.available_dialects)
    assert "arith" not in ctx.loaded_dialects
    ctx.load_dialect("arith")
    assert "arith" in ctx.loaded_dialects
    with pytest.raises(ValueError, match="unknown dialect"):
        ctx.load_dialect("nope")


def test_is_registered_operation() -> None:
    ctx = ir.Context()
    assert ctx.is_registered_operation("arith.addi")
    assert not ctx.is_registered_operation("arith.nope")


def test_with_context_sets_default() -> None:
    assert ir.Context.current() is None
    with pytest.raises(ValueError, match="no MLIR context"):
        ir.IndexType()
    with ir.Context() as ctx:
        assert ir.Context.current() == ctx
        assert ir.IndexType().context == ctx
    assert ir.Context.current() is None


def test_context_equality_is_identity() -> None:
    a, b = ir.Context(), ir.Context()
    assert a != b
    assert a == ir.IndexType(context=a).context
    assert a != "not a context"
    assert len({a, ir.IndexType(context=a).context}) == 1


def test_locations() -> None:
    with ir.Context():
        file = ir.Location.file("x.py", 3, 7)
        assert str(file) == 'loc("x.py":3:7)'
        named = ir.Location.name("add", file)
        assert str(named) == 'loc("add"("x.py":3:7))'
        assert str(ir.Location.unknown()) == "loc(unknown)"
        fused = ir.Location.fused([file, named])
        assert "fused" in str(fused)
        assert ir.Location.callsite(file, named) != file
        assert ir.Location.file("x.py", 3, 7) == file


def test_with_location_sets_default() -> None:
    with ir.Context():
        loc = ir.Location.file("a.mlir", 1, 1)
        with loc:
            assert ir.Location.current() == loc
            op = ir.Operation.create("builtin.module", regions=1)
            assert op.location == loc
        assert ir.Location.current() == ir.Location.unknown()


def test_mismatched_exit_is_reported() -> None:
    ctx = ir.Context()
    other = ir.Context()
    ctx.__enter__()
    with pytest.raises(RuntimeError, match="out of order"):
        other.__exit__(None, None, None)
    ctx.__exit__(None, None, None)
