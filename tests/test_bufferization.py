"""Bufferization (tensors to buffers) and automatic buffer deallocation."""

import pytest

import mlir_python as ir
from mlir_python import codegen, pipelines
from mlir_python.dialects import func

SQUARES = """
func.func @squares(%n: index) -> tensor<?xi64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %empty = bufferization.alloc_tensor(%n) : tensor<?xi64>
  %filled = scf.for %i = %c0 to %n step %c1 iter_args(%t = %empty) -> tensor<?xi64> {
    %x = arith.index_cast %i : index to i64
    %square = arith.muli %x, %x : i64
    %next = tensor.insert %square into %t[%i] : tensor<?xi64>
    scf.yield %next : tensor<?xi64>
  }
  return %filled : tensor<?xi64>
}
"""

TEMPORARY = """
func.func @sum_of_squares(%n: index) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i64
  %scratch = memref.alloc(%n) : memref<?xi64>
  scf.for %i = %c0 to %n step %c1 {
    %x = arith.index_cast %i : index to i64
    %square = arith.muli %x, %x : i64
    memref.store %square, %scratch[%i] : memref<?xi64>
  }
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> i64 {
    %v = memref.load %scratch[%i] : memref<?xi64>
    %next = arith.addi %acc, %v : i64
    scf.yield %next : i64
  }
  return %sum : i64
}
"""


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def functions(module: ir.Module) -> dict[str, func.FuncOp]:
    return {
        op.sym_name: op for op in module.body.operations if isinstance(op, func.FuncOp)
    }


def test_tensors_bufferize_and_return_to_python() -> None:
    module = ir.Module.parse(SQUARES)
    ir.PassManager(ir.Module, pipelines.bufferize()).run(module)
    text = str(module)
    assert "tensor<" not in text
    assert "memref<?xi64>" in text
    squares = functions(module)["squares"]
    compiled = codegen.compile(module)
    result = compiled.function(squares, owned_results=True)(6)
    assert isinstance(result, memoryview)
    assert result.tolist() == [0, 1, 4, 9, 16, 25]


def test_temporary_buffers_are_freed() -> None:
    module = ir.Module.parse(TEMPORARY)
    ir.PassManager(ir.Module, pipelines.buffer_deallocation()).run(module)
    assert "memref.dealloc" in str(module)
    compiled = codegen.compile(module)
    total = compiled.function(functions(module)["sum_of_squares"])
    assert total(10) == sum(i * i for i in range(10))


def test_returned_buffers_are_not_freed_by_their_function() -> None:
    module = ir.Module.parse(SQUARES)
    ir.PassManager(ir.Module, pipelines.bufferize()).run(module)
    squares = functions(module)["squares"]
    body = str(squares)
    # The caller owns the result, so nothing may free it inside the function.
    assert "memref.dealloc" not in body
    assert "bufferization.dealloc" not in body


def test_pipelines_are_typed_pass_lists() -> None:
    rendered = str(ir.PassManager(ir.Module, pipelines.buffer_deallocation()))
    assert "ownership-based-buffer-deallocation" in rendered
    assert "bufferization-lower-deallocations" in rendered
    assert str(ir.PassManager(ir.Module, pipelines.bufferize(deallocate=False))) == (
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries=true "
        "function-boundary-type-conversion=identity-layout-map})"
    )
