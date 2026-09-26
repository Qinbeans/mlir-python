import pytest

import mlir_python as ir


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def test_integer_attr() -> None:
    attr = ir.IntegerAttr(42)
    assert str(attr) == "42 : i64" and attr.value == 42
    assert attr.type == ir.IntegerType(64)
    assert ir.IntegerAttr(-1, ir.IntegerType(8)).value == -1
    assert ir.IntegerAttr(255, ir.IntegerType(8)).value == -1  # signless reads signed
    unsigned = ir.IntegerType(8, signedness=ir.Signedness.UNSIGNED)
    assert ir.IntegerAttr(255, unsigned).value == 255
    assert ir.IntegerAttr(7, ir.IndexType()).value == 7


def test_integer_attr_arbitrary_precision() -> None:
    big = 2**100 + 3
    assert ir.IntegerAttr(big, ir.IntegerType(128)).value == big
    assert ir.IntegerAttr(-(2**127), ir.IntegerType(128)).value == -(2**127)


def test_integer_attr_overflow() -> None:
    with pytest.raises(OverflowError):
        ir.IntegerAttr(256, ir.IntegerType(8))
    with pytest.raises(OverflowError):
        ir.IntegerAttr(-1, ir.IntegerType(8, signedness=ir.Signedness.UNSIGNED))
    with pytest.raises(OverflowError):
        ir.IntegerAttr(128, ir.IntegerType(8, signedness=ir.Signedness.SIGNED))


def test_bool_float_string_unit_type_attrs() -> None:
    true = ir.BoolAttr(True)
    assert str(true) == "true" and true.value is True and bool(true)
    assert isinstance(true, ir.IntegerAttr)
    assert ir.FloatAttr(1.5).value == 1.5
    assert str(ir.FloatAttr(1.5, ir.F32Type())) == "1.500000e+00 : f32"
    assert ir.StringAttr("hi").value == "hi"
    assert str(ir.UnitAttr()) == "unit"
    assert ir.TypeAttr(ir.F32Type()).value == ir.F32Type()


def test_parse_returns_most_specific_class() -> None:
    assert isinstance(ir.Attribute.parse("true"), ir.BoolAttr)
    assert isinstance(ir.Attribute.parse("1 : i32"), ir.IntegerAttr)
    assert isinstance(ir.Attribute.parse("@a::@b"), ir.SymbolRefAttr)
    assert isinstance(ir.Attribute.parse("@a"), ir.FlatSymbolRefAttr)
    assert isinstance(ir.Attribute.parse("array<i64: 1, 2>"), ir.DenseI64ArrayAttr)
    with pytest.raises(ir.MLIRError):
        ir.Attribute.parse("[1,")


def test_array_attr_sequence_protocol() -> None:
    arr = ir.ArrayAttr([ir.IntegerAttr(1), ir.StringAttr("x")])
    assert len(arr) == 2
    assert arr[0] == ir.IntegerAttr(1)
    assert arr[-1] == ir.StringAttr("x")
    assert list(arr) == arr.elements
    with pytest.raises(IndexError):
        arr[2]
    assert len(ir.ArrayAttr([])) == 0


def test_dict_attr_mapping_protocol() -> None:
    d = ir.DictAttr({"b": ir.IntegerAttr(2), "a": ir.UnitAttr()})
    assert len(d) == 2 and "a" in d and "z" not in d
    assert list(d) == ["a", "b"] and d.keys() == ["a", "b"]
    assert d["b"] == ir.IntegerAttr(2)
    assert d.get("z") is None
    assert d.items() == [("a", ir.UnitAttr()), ("b", ir.IntegerAttr(2))]
    with pytest.raises(KeyError):
        d["z"]


def test_symbol_refs() -> None:
    ref = ir.SymbolRefAttr("outer", ["mid", "leaf"])
    assert str(ref) == "@outer::@mid::@leaf"
    assert (ref.root, ref.leaf, ref.nested) == ("outer", "leaf", ["mid", "leaf"])
    assert ir.FlatSymbolRefAttr("f").value == "f"


def test_dense_elements() -> None:
    i32 = ir.IntegerType(32)
    t = ir.RankedTensorType([2, 2], i32)
    ints = ir.DenseIntElementsAttr([1, 2, 3, -4], t)
    assert str(ints) == "dense<[[1, 2], [3, -4]]> : tensor<2x2xi32>"
    assert ints.values == [1, 2, 3, -4] and len(ints) == 4 and not ints.is_splat
    splat = ir.DenseIntElementsAttr([7], t)
    assert splat.is_splat and splat.values == [7, 7, 7, 7]
    floats = ir.DenseFPElementsAttr([0.5, 1.5], ir.VectorType([2], ir.F32Type()))
    assert floats.values == [0.5, 1.5]
    generic = ir.DenseElementsAttr(
        [ir.IntegerAttr(1, i32), ir.IntegerAttr(2, i32)], ir.RankedTensorType([2], i32)
    )
    assert isinstance(ir.Attribute.parse(str(generic)), ir.DenseIntElementsAttr)


def test_dense_elements_validation() -> None:
    i32 = ir.IntegerType(32)
    with pytest.raises(ValueError, match="expected 4 elements"):
        ir.DenseIntElementsAttr([1, 2], ir.RankedTensorType([4], i32))
    with pytest.raises(ValueError, match="static shape"):
        ir.DenseIntElementsAttr([1], ir.RankedTensorType([None], i32))
    with pytest.raises(ValueError, match="ranked tensor or vector"):
        ir.DenseIntElementsAttr([1], ir.MemRefType([1], i32))
    with pytest.raises(ValueError, match="of type i32"):
        ir.DenseElementsAttr([ir.IntegerAttr(1)], ir.RankedTensorType([1], i32))
    with pytest.raises(OverflowError):
        ir.DenseIntElementsAttr([1 << 40], ir.RankedTensorType([1], i32))


def test_dense_arrays() -> None:
    arr = ir.DenseI64ArrayAttr([1, -2, 3])
    assert str(arr) == "array<i64: 1, -2, 3>"
    assert arr.values == [1, -2, 3] and arr[-1] == 3 and len(arr) == 3
    assert ir.DenseBoolArrayAttr([True, False]).values == [True, False]
    assert ir.DenseF32ArrayAttr([0.5]).values == [0.5]
    with pytest.raises(TypeError):
        ir.DenseI8ArrayAttr([1000])


def test_strided_layout() -> None:
    layout = ir.StridedLayoutAttr(4, [None, 1])
    assert str(layout) == "strided<[?, 1], offset: 4>"
    assert layout.offset == 4 and layout.strides == [None, 1]
