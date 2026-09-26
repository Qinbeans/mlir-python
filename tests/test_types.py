import pytest

import mlir_python as ir


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def test_integer_types() -> None:
    i32 = ir.IntegerType(32)
    assert str(i32) == "i32"
    assert i32.width == 32 and i32.is_signless
    si8 = ir.IntegerType(8, signedness=ir.Signedness.SIGNED)
    assert str(si8) == "si8" and si8.is_signed
    assert si8.signedness is ir.Signedness.SIGNED
    assert str(ir.IntegerType(64, signedness=ir.Signedness.UNSIGNED)) == "ui64"
    assert repr(i32) == "IntegerType(i32)"


def test_invalid_integer_width_raises() -> None:
    with pytest.raises(ir.MLIRError, match="invalid integer type"):
        ir.IntegerType(1 << 25)


def test_float_types() -> None:
    assert str(ir.F32Type()) == "f32"
    assert str(ir.BF16Type()) == "bf16"
    assert str(ir.Float8E4M3FNType()) == "f8E4M3FN"
    f64 = ir.F64Type()
    assert isinstance(f64, ir.FloatType)
    assert f64.width == 64 and f64.mantissa_width == 53


def test_uniquing_equality_and_hash() -> None:
    assert ir.IntegerType(32) == ir.IntegerType(32)
    assert ir.IntegerType(32) != ir.IntegerType(64)
    assert ir.IntegerType(32) != ir.F32Type()
    assert ir.IntegerType(32) != 32
    assert len({ir.IntegerType(32), ir.IntegerType(32), ir.F32Type()}) == 2


def test_parse_returns_most_specific_class() -> None:
    parsed = ir.Type.parse("tensor<4x?xf32>")
    assert isinstance(parsed, ir.RankedTensorType)
    assert parsed.shape == [4, None]
    assert parsed.rank == 2
    assert isinstance(parsed.element_type, ir.F32Type)
    assert isinstance(ir.Type.parse("index"), ir.IndexType)
    assert isinstance(ir.Type.parse("vector<4xi8>"), ir.VectorType)


def test_parse_error_carries_diagnostics() -> None:
    with pytest.raises(ir.MLIRError, match="failed to parse type") as info:
        ir.Type.parse("tensor<4x")
    assert "error:" in str(info.value)


def test_shaped_types() -> None:
    f32 = ir.F32Type()
    tensor = ir.RankedTensorType([2, None], f32)
    assert str(tensor) == "tensor<2x?xf32>"
    assert not tensor.has_static_shape and tensor.num_elements is None
    assert ir.RankedTensorType([2, 3], f32).num_elements == 6
    unranked = ir.UnrankedTensorType(f32)
    assert str(unranked) == "tensor<*xf32>" and not unranked.has_rank
    with pytest.raises(ValueError, match="unranked"):
        _ = unranked.shape
    with pytest.raises(ValueError, match="non-negative"):
        ir.RankedTensorType([-1], f32)


def test_vector_types() -> None:
    vec = ir.VectorType([4, 8], ir.F32Type(), scalable=[False, True])
    assert str(vec) == "vector<4x[8]xf32>"
    assert vec.scalable_dims == [False, True] and vec.is_scalable
    with pytest.raises(ir.MLIRError, match="invalid vector type"):
        ir.VectorType([0], ir.F32Type())
    with pytest.raises(ValueError, match="one entry per dimension"):
        ir.VectorType([4], ir.F32Type(), scalable=[True, False])


def test_memref_types() -> None:
    f32 = ir.F32Type()
    layout = ir.StridedLayoutAttr(None, [None, 1])
    memref = ir.MemRefType(
        [4, None], f32, layout=layout, memory_space=ir.IntegerAttr(1)
    )
    assert str(memref) == "memref<4x?xf32, strided<[?, 1], offset: ?>, 1>"
    assert memref.layout == layout
    assert memref.memory_space == ir.IntegerAttr(1)
    assert ir.MemRefType([2], f32).memory_space is None
    assert str(ir.UnrankedMemRefType(f32)) == "memref<*xf32>"
    with pytest.raises(ValueError, match="layout"):
        ir.MemRefType([2], f32, layout=ir.StringAttr("x"))


def test_function_tuple_complex_opaque() -> None:
    i32, f32 = ir.IntegerType(32), ir.F32Type()
    fn = ir.FunctionType([i32, f32], [i32])
    assert str(fn) == "(i32, f32) -> i32"
    assert fn.inputs == [i32, f32] and fn.results == [i32]
    tup = ir.TupleType([i32, f32])
    assert str(tup) == "tuple<i32, f32>" and len(tup) == 2
    assert str(ir.ComplexType(f32)) == "complex<f32>"
    with pytest.raises(ir.MLIRError):
        ir.ComplexType(ir.IndexType())
    with pytest.raises(ir.MLIRError, match="unregistered dialect"):
        ir.OpaqueType("foo", "bar<1>")
    lenient = ir.Context(allow_unregistered_dialects=True)
    opaque = ir.OpaqueType("foo", "bar<1>", context=lenient)
    assert (opaque.dialect, opaque.data) == ("foo", "bar<1>")
    assert str(opaque) == "!foo.bar<1>"


def test_types_from_different_contexts_do_not_mix() -> None:
    other = ir.Context()
    with pytest.raises(ValueError, match="different Context"):
        ir.FunctionType([ir.IntegerType(32)], [ir.IntegerType(32, context=other)])
