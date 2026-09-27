"""The vector dialect: SIMD code lowered to LLVM vector instructions."""

import array

import pytest

import mlir_python as ir
from mlir_python import codegen, passes
from mlir_python.dialects import arith, func, vector

np = pytest.importorskip("numpy")

# Sums eight lanes at a time; the last, partial chunk is read with a mask.
SUM = """
func.func @sum(%buffer: memref<?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant dense<0.0> : vector<8xf32>
  %pad = arith.constant 0.0 : f32
  %n = memref.dim %buffer, %c0 : memref<?xf32>
  %acc = scf.for %i = %c0 to %n step %c8 iter_args(%a = %zero) -> vector<8xf32> {
    %v = vector.transfer_read %buffer[%i], %pad : memref<?xf32>, vector<8xf32>
    %s = arith.addf %a, %v : vector<8xf32>
    scf.yield %s : vector<8xf32>
  }
  %r = vector.reduction <add>, %acc : vector<8xf32> into f32
  return %r : f32
}
"""

# A two-dimensional transfer (lowered through loops) and a transpose.
TRANSPOSE = """
func.func @transpose(%a: memref<4x8xf32>, %b: memref<8x4xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %v = vector.transfer_read %a[%c0, %c0], %pad : memref<4x8xf32>, vector<4x8xf32>
  %t = vector.transpose %v, [1, 0] : vector<4x8xf32> to vector<8x4xf32>
  vector.transfer_write %t, %b[%c0, %c0] : vector<8x4xf32>, memref<8x4xf32>
  return
}
"""

# Row sums of a 4x8 matrix, reduced across one dimension of a 2-D vector.
ROW_SUMS = """
func.func @row_sums(%a: memref<4x8xi32>, %out: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i32
  %v = vector.transfer_read %a[%c0, %c0], %pad : memref<4x8xi32>, vector<4x8xi32>
  %zero = arith.constant dense<0> : vector<4xi32>
  %sums = vector.multi_reduction <add>, %v, %zero [1] : vector<4x8xi32> to vector<4xi32>
  vector.store %sums, %out[%c0] : memref<4xi32>, vector<4xi32>
  return
}
"""


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def function(module: ir.Module, name: str) -> func.FuncOp:
    return next(
        op
        for op in module.body.operations
        if isinstance(op, func.FuncOp) and op.sym_name == name
    )


@pytest.mark.parametrize("level", [codegen.OptLevel.O0, codegen.OptLevel.O2])
def test_simd_reduction_with_a_masked_tail(level: codegen.OptLevel) -> None:
    module = ir.Module.parse(SUM)
    total = codegen.compile(module, opt_level=level).function(function(module, "sum"))
    for n in (0, 5, 8, 1003):
        assert total(array.array("f", range(n))) == sum(range(n))
    assert "<8 x float>" in str(codegen.to_llvm_ir(module))


def test_multi_dimensional_vectors() -> None:
    module = ir.Module.parse(TRANSPOSE + ROW_SUMS)
    compiled = codegen.compile(module)
    a = np.arange(32, dtype=np.float32).reshape(4, 8)
    b = np.zeros((8, 4), dtype=np.float32)
    compiled.function(function(module, "transpose"))(a, b)
    assert (b == a.T).all()
    m = np.arange(32, dtype=np.int32).reshape(4, 8)
    sums = np.zeros(4, dtype=np.int32)
    compiled.function(function(module, "row_sums"))(m, sums)
    assert list(sums) == list(m.sum(axis=1))


def test_vectors_built_with_the_typed_api() -> None:
    f32 = ir.F32Type()
    lanes = ir.VectorType([8], f32)
    buffer = ir.MemRefType([8], f32)
    module = ir.Module()
    with ir.Location.unknown():
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp("dot", ir.FunctionType([buffer, buffer], [f32]))
        with ir.InsertionPoint(fn.add_entry_block()):
            x, y = fn.arguments
            zero = arith.ConstantOp(ir.IntegerAttr(0, ir.IndexType())).result
            products = arith.MulFOp(
                vector.LoadOp(lanes, x, [zero]).result,
                vector.LoadOp(lanes, y, [zero]).result,
            ).result
            dot = vector.ReductionOp(f32, vector.CombiningKind.ADD, products)
            func.ReturnOp([dot.result])
    module.verify()
    dot = codegen.compile(module).function(fn)
    x = np.arange(8, dtype=np.float32)
    assert dot(x, np.full(8, 2, dtype=np.float32)) == 2 * sum(range(8))


def test_vector_passes_are_available() -> None:
    module = ir.Module.parse(ROW_SUMS)
    ir.PassManager(
        ir.Module, [ir.Nested(func.FuncOp, [passes.LowerVectorMultiReduction()])]
    ).run(module)
    assert "vector.multi_reduction" not in str(module)
