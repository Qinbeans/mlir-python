"""C and C++ source from MLIR, through the EmitC dialect."""

import ctypes
import math
import shutil
import subprocess
from pathlib import Path

import pytest

import mlir_python as ir
from mlir_python import codegen, passes
from mlir_python.dialects import emitc, func

CC = shutil.which("cc")
CXX = shutil.which("c++")
needs_cc = pytest.mark.skipif(CC is None, reason="needs a C compiler")
needs_cxx = pytest.mark.skipif(CXX is None, reason="needs a C++ compiler")

KERNELS = """
func.func @norm(%v: memref<4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %acc = scf.for %i = %c0 to %c4 step %c1 iter_args(%a = %zero) -> f32 {
    %x = memref.load %v[%i] : memref<4xf32>
    %sq = arith.mulf %x, %x : f32
    %n = arith.addf %a, %sq : f32
    scf.yield %n : f32
  }
  %r = math.sqrt %acc : f32
  return %r : f32
}

func.func @collatz(%n: i64) -> i64 {
  %one = arith.constant 1 : i64
  %two = arith.constant 2 : i64
  %three = arith.constant 3 : i64
  %zero = arith.constant 0 : i64
  %r:2 = scf.while (%x = %n, %steps = %zero) : (i64, i64) -> (i64, i64) {
    %more = arith.cmpi ne, %x, %one : i64
    scf.condition(%more) %x, %steps : i64, i64
  } do {
  ^bb0(%x: i64, %steps: i64):
    %rem = arith.remsi %x, %two : i64
    %even = arith.cmpi eq, %rem, %zero : i64
    %next = scf.if %even -> i64 {
      %h = arith.divsi %x, %two : i64
      scf.yield %h : i64
    } else {
      %t = arith.muli %x, %three : i64
      %t1 = arith.addi %t, %one : i64
      scf.yield %t1 : i64
    }
    %s = arith.addi %steps, %one : i64
    scf.yield %next, %s : i64, i64
  }
  return %r#1 : i64
}

func.func @special(%x: f64, %y: f64) -> f64 {
  %a = math.tanh %x : f64
  %b = math.rsqrt %y : f64
  %c = math.fma %a, %b, %x : f64
  %d = math.powf %c, %y : f64
  return %d : f64
}

func.func @ordinary(%x: f32) -> i1 {
  %nan = math.isnan %x : f32
  %inf = math.isinf %x : f32
  %bad = arith.ori %nan, %inf : i1
  %true = arith.constant true
  %ok = arith.xori %bad, %true : i1
  return %ok : i1
}

func.func @branchy(%x: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %neg = arith.cmpi slt, %x, %zero : i32
  cf.cond_br %neg, ^negative, ^done(%x : i32)
^negative:
  %flipped = arith.subi %zero, %x : i32
  cf.br ^done(%flipped : i32)
^done(%r: i32):
  return %r : i32
}
"""


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def compile_library(compiler: str, source: str, path: Path, *flags: str) -> ctypes.CDLL:
    path.with_suffix(".src").write_text(source)
    library = path.with_suffix(".so")
    subprocess.run(
        [
            compiler,
            *flags,
            "-shared",
            "-fPIC",
            "-Werror=implicit-function-declaration",
            "-x",
            "c++" if compiler == CXX else "c",
            str(path.with_suffix(".src")),
            "-o",
            str(library),
            "-lm",
        ],
        check=True,
    )
    return ctypes.CDLL(str(library))


def check_kernels(library: ctypes.CDLL) -> None:
    library.norm.restype = ctypes.c_float
    library.norm.argtypes = [ctypes.POINTER(ctypes.c_float)]
    vector = (ctypes.c_float * 4)(1, 2, 2, 4)
    assert library.norm(vector) == 5.0

    library.collatz.restype = ctypes.c_int64
    library.collatz.argtypes = [ctypes.c_int64]
    assert [library.collatz(n) for n in (1, 6, 27)] == [0, 8, 111]

    library.special.restype = ctypes.c_double
    library.special.argtypes = [ctypes.c_double, ctypes.c_double]
    x, y = 0.5, 4.0
    expected = (math.tanh(x) / math.sqrt(y) + x) ** y
    assert library.special(x, y) == pytest.approx(expected)

    library.ordinary.restype = ctypes.c_bool
    library.ordinary.argtypes = [ctypes.c_float]
    assert [library.ordinary(v) for v in (1.0, math.inf, math.nan)] == [
        True,
        False,
        False,
    ]

    library.branchy.restype = ctypes.c_int32
    library.branchy.argtypes = [ctypes.c_int32]
    assert [library.branchy(v) for v in (-7, 0, 7)] == [7, 0, 7]


@needs_cc
def test_c_source_compiles_and_runs(tmp_path: Path) -> None:
    assert CC is not None
    source = codegen.to_c(ir.Module.parse(KERNELS))
    assert "#include <math.h>" in source and "sqrtf(" in source
    assert "float norm(float v1[4])" in source
    library = compile_library(CC, source, tmp_path / "kernels", "-std=c99", "-pedantic")
    check_kernels(library)


@needs_cxx
def test_cpp_source_compiles_and_runs(tmp_path: Path) -> None:
    assert CXX is not None
    source = codegen.to_cpp(ir.Module.parse(KERNELS))
    assert "#include <cmath>" in source and "std::sqrt(" in source
    # C++ mangles names; give the functions C linkage to find them.
    wrapped = (
        source.replace("#include <cmath>\n", '#include <cmath>\nextern "C" {\n') + "}\n"
    )
    library = compile_library(CXX, wrapped, tmp_path / "kernels", "-std=c++11")
    check_kernels(library)


def test_the_module_is_left_alone() -> None:
    module = ir.Module.parse(KERNELS)
    before = str(module)
    codegen.to_c(module)
    assert str(module) == before


def test_dynamic_memrefs_are_rejected() -> None:
    module = ir.Module.parse(
        """
        func.func @first(%v: memref<?xf32>) -> f32 {
          %c0 = arith.constant 0 : index
          %x = memref.load %v[%c0] : memref<?xf32>
          return %x : f32
        }
        """
    )
    with pytest.raises(ValueError, match="static shapes"):
        codegen.to_c(module)


def test_lowering_in_stages() -> None:
    module = ir.Module.parse(KERNELS)
    codegen.lower_to_emitc(module)
    names = set()
    module.walk(lambda op: names.add(op.name.split(".")[0]))
    assert names <= {"builtin", "emitc", "cf"}
    headers = [
        op.include for op in module.body.operations if isinstance(op, emitc.IncludeOp)
    ]
    assert headers[:3] == ["stdbool.h", "stddef.h", "stdint.h"]
    assert "math.h" in headers
    source = codegen.translate_to_cpp(module, declare_variables_at_top=True)
    assert "int64_t collatz(int64_t v1)" in source
    assert any(
        isinstance(p, passes.ConvertToEmitC) for p in codegen.emitc_lowering_pipeline()
    )


@needs_cc
def test_emitc_built_with_the_typed_api(tmp_path: Path) -> None:
    assert CC is not None
    i32 = ir.IntegerType(32)
    module = ir.Module()
    with ir.Location.unknown():
        with ir.InsertionPoint(module.body):
            emitc.IncludeOp("stdint.h", is_standard_include=True)
            emitc.IncludeOp("stdio.h", is_standard_include=True)
            main = func.FuncOp("main", ir.FunctionType([], [i32]))
        with ir.InsertionPoint(main.add_entry_block()):
            text = emitc.LiteralOp(
                emitc.PointerType(emitc.OpaqueType("const char")), '"%d\\n"'
            ).result
            answer = emitc.ConstantOp(i32, ir.IntegerAttr(42, i32)).result
            emitc.CallOpaqueOp("printf", [text, answer], [i32])
            func.ReturnOp([emitc.ConstantOp(i32, ir.IntegerAttr(0, i32)).result])
    module.verify()
    source = tmp_path / "hello.c"
    source.write_text(codegen.translate_to_cpp(module))
    subprocess.run([CC, str(source), "-o", str(tmp_path / "hello")], check=True)
    run = subprocess.run(
        [tmp_path / "hello"], capture_output=True, text=True, check=True
    )
    assert run.stdout == "42\n"


def test_math_without_a_library_function_is_reported() -> None:
    module = ir.Module.parse(
        """
        func.func @clz(%x: i32) -> i32 {
          %r = math.ctlz %x : i32
          return %r : i32
        }
        """
    )
    with pytest.raises(ir.MLIRError):
        codegen.to_c(module)
