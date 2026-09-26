"""Helpers that build IR from operation objects instead of symbol names."""

import ctypes

import pytest

import mlir_python as ir
from mlir_python import codegen
from mlir_python.dialects import arith, func, llvm


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def build_hello() -> tuple[ir.Module, func.FuncOp]:
    i32 = ir.IntegerType(32)
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        greeting = llvm.string_constant("greeting", "Hello, world!")
        puts = func.declare("puts", [llvm.PointerType()], [i32])

        @func.define([], [i32])
        def main() -> ir.Value:
            func.call(puts, [llvm.address_of(greeting)])
            return arith.ConstantOp(ir.IntegerAttr(0, i32)).result

    module.verify()
    return module, main


def test_hello_world_with_helpers(capfd: pytest.CaptureFixture[str]) -> None:
    module, main = build_hello()
    assert isinstance(main, func.FuncOp) and main.sym_name == "main"
    assert codegen.compile(module).function(main)() == 0
    ctypes.CDLL(None).fflush(None)
    assert capfd.readouterr().out == "Hello, world!\n"
    llvm_ir = str(codegen.to_llvm_ir(module, opt_level=codegen.OptLevel.O0))
    assert '@greeting = internal constant [14 x i8] c"Hello, world!\\00"' in llvm_ir
    assert "declare i32 @puts(ptr)" in llvm_ir


def test_define_passes_arguments_and_returns_values() -> None:
    i32 = ir.IntegerType(32)
    module = ir.Module()
    with ir.InsertionPoint(module.body):

        @func.define([i32, i32], [i32, i32])
        def swap(a: ir.Value, b: ir.Value) -> list[ir.Value]:
            return [b, a]

        @func.define([], [])
        def nothing() -> None:
            pass

    module.verify()
    assert codegen.compile(module).function(swap)(1, 2) == (2, 1)
    assert isinstance(nothing.body.blocks[0].terminator, func.ReturnOp)


def test_define_checks_the_body_against_the_signature() -> None:
    i32, f32 = ir.IntegerType(32), ir.F32Type()
    with ir.InsertionPoint(ir.Module().body):
        with pytest.raises(
            TypeError, match="@one takes 1 arguments, but its body accepts 0"
        ):

            @func.define([i32], [i32])
            def one() -> None:
                pass

        with pytest.raises(
            TypeError, match=r"@wrong must return \(f32\) but its body returned \(i32\)"
        ):

            @func.define([i32], [f32])
            def wrong(x: ir.Value) -> ir.Value:
                return x

        with pytest.raises(
            TypeError, match=r"@missing must return \(i32\) but its body returned \(\)"
        ):

            @func.define([], [i32])
            def missing() -> None:
                pass


def test_call_checks_arguments_and_takes_result_types_from_the_callee() -> None:
    i32, f32 = ir.IntegerType(32), ir.F32Type()
    with ir.InsertionPoint(ir.Module().body):
        square = func.declare("square", [i32], [i32])

        @func.define([f32], [])
        def caller(x: ir.Value) -> None:
            with pytest.raises(
                TypeError, match="argument 0 of @square must be i32, got f32"
            ):
                func.call(square, [x])
            with pytest.raises(TypeError, match="@square takes 1 arguments, got 0"):
                func.call(square, [])
            one = arith.ConstantOp(ir.IntegerAttr(1, i32)).result
            assert func.call(square, [one]).result.type == i32


def test_string_constants_are_sized_in_bytes() -> None:
    with ir.InsertionPoint(ir.Module().body):
        text = llvm.string_constant("snowman", "☃")  # 3 UTF-8 bytes + NUL
        assert text.global_type == llvm.ArrayType(ir.IntegerType(8), 4)
        assert llvm.address_of(text).type == llvm.PointerType()
