"""End-to-end compilation: lowering, LLVM IR, object code, linking, and JIT."""

import ctypes
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

import mlir_python as ir
from mlir_python import codegen
from mlir_python.dialects import arith, func, scf

CC = shutil.which("cc")
NM = shutil.which("nm")
needs_cc = pytest.mark.skipif(CC is None, reason="needs a C compiler to link")


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def build_kernels() -> tuple[ir.Module, dict[str, func.FuncOp]]:
    """add, sum_below (an scf.for loop), scale, divmod, and is_less."""
    i1, i32, i64, f64, index = (
        ir.IntegerType(1),
        ir.IntegerType(32),
        ir.IntegerType(64),
        ir.F64Type(),
        ir.IndexType(),
    )
    module = ir.Module()
    fns: dict[str, func.FuncOp] = {}

    def define(name: str, inputs: list[ir.Type], results: list[ir.Type]) -> func.FuncOp:
        with ir.InsertionPoint(module.body):
            fns[name] = func.FuncOp(name, ir.FunctionType(inputs, results))
        return fns[name]

    add = define("add", [i32, i32], [i32])
    with ir.InsertionPoint(add.add_entry_block()):
        func.ReturnOp([arith.AddIOp(*add.arguments).result])

    sum_below = define("sum_below", [index], [index])
    with ir.InsertionPoint(sum_below.add_entry_block()):
        zero = arith.ConstantOp(ir.IntegerAttr(0, index)).result
        one = arith.ConstantOp(ir.IntegerAttr(1, index)).result
        loop = scf.ForOp(zero, sum_below.arguments[0], one, init_args=[zero])
        with ir.InsertionPoint(loop.body):
            total = arith.AddIOp(loop.inner_iter_args[0], loop.induction_variable)
            scf.YieldOp([total.result])
        func.ReturnOp(loop.results)

    scale = define("scale", [f64, f64], [f64])
    with ir.InsertionPoint(scale.add_entry_block()):
        func.ReturnOp([arith.MulFOp(*scale.arguments).result])

    divmod_ = define("divmod", [i64, i64], [i64, i64])
    with ir.InsertionPoint(divmod_.add_entry_block()):
        a, b = divmod_.arguments
        func.ReturnOp([arith.DivSIOp(a, b).result, arith.RemSIOp(a, b).result])

    is_less = define("is_less", [i32, i32], [i1])
    with ir.InsertionPoint(is_less.add_entry_block()):
        a, b = is_less.arguments
        func.ReturnOp([arith.CmpIOp(arith.CmpIPredicate.SLT, a, b).result])

    module.verify()
    return module, fns


# -- Stage 1: lowering ---------------------------------------------------------


def test_lowering_leaves_only_llvm_operations() -> None:
    module, _ = build_kernels()
    codegen.lower_to_llvm(module)
    module.verify()
    names: set[str] = set()
    module.walk(lambda op: names.add(op.name))
    assert names - {"builtin.module"} and all(
        n.startswith("llvm.") for n in names - {"builtin.module"}
    ), names


def test_lowering_pipeline_is_typed_and_extensible() -> None:
    pipeline = codegen.llvm_lowering_pipeline()
    assert all(isinstance(p, (ir.Nested, codegen.passes.Pass)) for p in pipeline)
    assert str(ir.PassManager(ir.Module, pipeline)) == (
        "builtin.module(convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts)"
    )


# -- Stage 2: LLVM IR ------------------------------------------------------------


def test_llvm_ir_verifies_and_matches_signatures() -> None:
    module, _ = build_kernels()
    codegen.lower_to_llvm(module)
    llvm_ir = codegen.translate_to_llvm_ir(module)
    llvm_ir.verify()
    text = str(llvm_ir)
    assert "define i32 @add(i32 %0, i32 %1)" in text
    assert "define i64 @sum_below(i64 %0)" in text
    assert "define double @scale(double %0, double %1)" in text
    assert "define i1 @is_less(i32 %0, i32 %1)" in text
    assert llvm_ir.defined_functions == [
        "add",
        "sum_below",
        "scale",
        "divmod",
        "is_less",
    ]
    assert (
        platform.machine().replace("amd64", "x86_64").lower() in llvm_ir.target_triple
    )
    assert llvm_ir.data_layout


def test_optimized_llvm_ir_still_verifies() -> None:
    module, _ = build_kernels()
    codegen.lower_to_llvm(module)
    llvm_ir = codegen.translate_to_llvm_ir(module)
    llvm_ir.optimize(codegen.OptLevel.O3)
    llvm_ir.verify()
    assert "define" in str(llvm_ir)


def test_translation_rejects_unlowered_ir() -> None:
    module, _ = build_kernels()
    with pytest.raises(ir.MLIRError, match="lower the module to the LLVM dialect"):
        codegen.translate_to_llvm_ir(module)


# -- Stage 3: object code --------------------------------------------------------


def compile_object(tmp_path: Path) -> Path:
    module, _ = build_kernels()
    codegen.lower_to_llvm(module)
    obj = tmp_path / "kernels.o"
    codegen.translate_to_llvm_ir(module).write_object(str(obj))
    return obj


def test_object_file_exports_functions(tmp_path: Path) -> None:
    obj = compile_object(tmp_path)
    data = obj.read_bytes()
    if platform.system() == "Linux":
        assert data[:4] == b"\x7fELF"
    if NM is not None:
        symbols = subprocess.run(
            [NM, "--defined-only", str(obj)], check=True, capture_output=True, text=True
        ).stdout
        for name in ("add", "sum_below", "scale", "divmod", "is_less"):
            assert f" T {name}\n" in symbols or f" T _{name}\n" in symbols, symbols


def test_assembly_names_functions() -> None:
    module, _ = build_kernels()
    codegen.lower_to_llvm(module)
    assembly = codegen.translate_to_llvm_ir(module).assembly()
    assert "add:" in assembly and "sum_below:" in assembly


# -- Stage 4: linking ----------------------------------------------------------------

MAIN_C = r"""
#include <stdint.h>
#include <stdio.h>
int32_t add(int32_t, int32_t);
int64_t sum_below(int64_t);
double scale(double, double);
_Bool is_less(int32_t, int32_t);
int main(void) {
  printf("%d %lld %.2f %d\n", add(2, 3), (long long)sum_below(10),
         scale(1.5, 4.0), (int)is_less(1, 2));
  return 0;
}
"""


@needs_cc
def test_links_into_an_executable(tmp_path: Path) -> None:
    assert CC is not None
    obj = compile_object(tmp_path)
    main = tmp_path / "main.c"
    main.write_text(MAIN_C)
    exe = tmp_path / "kernels"
    subprocess.run([CC, str(main), str(obj), "-o", str(exe)], check=True)
    output = subprocess.run(
        [str(exe)], check=True, capture_output=True, text=True
    ).stdout
    assert output.split() == ["5", "45", "6.00", "1"]


@needs_cc
def test_links_into_a_shared_library(tmp_path: Path) -> None:
    assert CC is not None
    obj = compile_object(tmp_path)
    library = tmp_path / "libkernels.so"
    subprocess.run([CC, "-shared", str(obj), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    lib.add.argtypes, lib.add.restype = [ctypes.c_int32, ctypes.c_int32], ctypes.c_int32
    lib.sum_below.argtypes, lib.sum_below.restype = [ctypes.c_int64], ctypes.c_int64
    assert lib.add(40, 2) == 42
    assert lib.sum_below(100) == sum(range(100))


# -- Stage 5: JIT ------------------------------------------------------------------------


def test_jit_calls_functions_by_their_func_op() -> None:
    module, fns = build_kernels()
    compiled = codegen.compile(module)
    assert compiled.function(fns["add"])(2, 3) == 5
    assert compiled.function(fns["add"])(-7, 3) == -4
    assert compiled.function(fns["sum_below"])(10) == 45
    assert compiled.function(fns["scale"])(1.5, 4.0) == 6.0
    assert compiled.function(fns["divmod"])(17, 5) == (3, 2)
    assert compiled.function(fns["is_less"])(1, 2) is True
    assert compiled.function(fns["is_less"])(2, 1) is False


@needs_cc
def test_jit_agrees_with_the_linked_library(tmp_path: Path) -> None:
    assert CC is not None
    module, fns = build_kernels()
    jit_sum = codegen.compile(module).function(fns["sum_below"])
    obj = compile_object(tmp_path)
    library = tmp_path / "libkernels.so"
    subprocess.run([CC, "-shared", str(obj), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    lib.sum_below.argtypes, lib.sum_below.restype = [ctypes.c_int64], ctypes.c_int64
    for n in (0, 1, 7, 1000):
        assert jit_sum(n) == lib.sum_below(n) == sum(range(n))


def test_compile_leaves_the_original_module_untouched() -> None:
    module, fns = build_kernels()
    before = str(module)
    compiled = codegen.compile(module)
    assert str(module) == before
    assert fns["add"].sym_name == "add"  # handles into the original stay valid
    assert compiled.function_names == ["add", "sum_below", "scale", "divmod", "is_less"]
    assert all(op.name.startswith("llvm.") for op in compiled.lowered.body.operations)
    compiled.llvm_ir.verify()


def test_jit_checks_arguments() -> None:
    module, fns = build_kernels()
    add = codegen.compile(module).function(fns["add"])
    with pytest.raises(TypeError, match="takes 2 arguments, got 1"):
        add(1)
    with pytest.raises(OverflowError, match="does not fit i32"):
        add(2**40, 1)
    with pytest.raises(TypeError, match="must be an int"):
        add(1.5, 1)


def test_jit_rejects_types_that_cannot_cross_the_boundary() -> None:
    f16 = ir.F16Type()
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        takes_half = func.FuncOp("takes_half", ir.FunctionType([f16], []))
        gives_buffer = func.FuncOp(
            "gives_buffer", ir.FunctionType([], [ir.MemRefType([4], ir.F32Type())])
        )
    with ir.InsertionPoint(takes_half.add_entry_block()):
        func.ReturnOp()
    with ir.InsertionPoint(gives_buffer.add_entry_block()):
        from mlir_python.dialects import memref

        func.ReturnOp([memref.AllocOp(ir.MemRefType([4], ir.F32Type())).memref])
    compiled = codegen.compile(module)
    with pytest.raises(TypeError, match="argument 0 has type f16"):
        compiled.function(takes_half)(1.0)
    with pytest.raises(TypeError, match="result 0 has type memref<4xf32>"):
        compiled.function(gives_buffer)()


def test_function_must_belong_to_the_compiled_module() -> None:
    module, _ = build_kernels()
    compiled = codegen.compile(module)
    stranger = func.FuncOp("stranger", ir.FunctionType([], []))
    with pytest.raises(
        ValueError, match="not a function defined in the compiled module"
    ):
        compiled.function(stranger)


# -- Buffers as memref arguments --------------------------------------------------


def build_buffer_kernels() -> tuple[ir.Module, dict[str, func.FuncOp]]:
    """sum_vector(memref<?xf32>) -> f32, scale_matrix(memref<?x?xf64>, f64),
    and fill4(memref<4xi32>, i32)."""
    from mlir_python.dialects import memref

    f32, f64, i32, index = (
        ir.F32Type(),
        ir.F64Type(),
        ir.IntegerType(32),
        ir.IndexType(),
    )
    module = ir.Module()
    fns: dict[str, func.FuncOp] = {}

    def define(name: str, inputs: list[ir.Type], results: list[ir.Type]) -> func.FuncOp:
        with ir.InsertionPoint(module.body):
            fns[name] = func.FuncOp(name, ir.FunctionType(inputs, results))
        return fns[name]

    def constant_index(value: int) -> ir.Value:
        return arith.ConstantOp(ir.IntegerAttr(value, index)).result

    vector = define("sum_vector", [ir.MemRefType([None], f32)], [f32])
    with ir.InsertionPoint(vector.add_entry_block()):
        (buffer,) = vector.arguments
        zero, one = constant_index(0), constant_index(1)
        size = memref.DimOp(buffer, zero).result
        init = arith.ConstantOp(ir.FloatAttr(0.0, f32)).result
        loop = scf.ForOp(zero, size, one, init_args=[init])
        with ir.InsertionPoint(loop.body):
            element = memref.LoadOp(buffer, [loop.induction_variable]).result
            scf.YieldOp([arith.AddFOp(loop.inner_iter_args[0], element).result])
        func.ReturnOp(loop.results)

    matrix = define("scale_matrix", [ir.MemRefType([None, None], f64), f64], [])
    with ir.InsertionPoint(matrix.add_entry_block()):
        buffer, factor = matrix.arguments
        zero, one = constant_index(0), constant_index(1)
        rows = memref.DimOp(buffer, zero).result
        cols = memref.DimOp(buffer, one).result
        outer = scf.ForOp(zero, rows, one)
        with ir.InsertionPoint(outer.body):
            inner = scf.ForOp(zero, cols, one)
            with ir.InsertionPoint(inner.body):
                where = [outer.induction_variable, inner.induction_variable]
                value = memref.LoadOp(buffer, where).result
                memref.StoreOp(arith.MulFOp(value, factor).result, buffer, where)
        func.ReturnOp()

    fill = define("fill4", [ir.MemRefType([4], i32), i32], [])
    with ir.InsertionPoint(fill.add_entry_block()):
        buffer, value = fill.arguments
        zero, one, four = constant_index(0), constant_index(1), constant_index(4)
        loop = scf.ForOp(zero, four, one)
        with ir.InsertionPoint(loop.body):
            memref.StoreOp(value, buffer, [loop.induction_variable])
        func.ReturnOp()

    module.verify()
    return module, fns


def test_buffers_are_read_and_written_in_place() -> None:
    import numpy as np

    module, fns = build_buffer_kernels()
    compiled = codegen.compile(module)
    vector = np.arange(10, dtype=np.float32)
    assert compiled.function(fns["sum_vector"])(vector) == 45.0
    matrix = np.arange(6, dtype=np.float64).reshape(2, 3)
    compiled.function(fns["scale_matrix"])(matrix, 2.0)
    assert matrix.tolist() == [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]
    ints = np.zeros(4, dtype=np.int32)
    compiled.function(fns["fill4"])(ints, 7)
    assert ints.tolist() == [7, 7, 7, 7]


def test_strided_views_work_where_the_type_allows() -> None:
    import numpy as np

    module, fns = build_buffer_kernels()
    scale = codegen.compile(module).function(fns["scale_matrix"])
    matrix = np.ones((4, 3))
    scale(matrix[::2], 5.0)  # every other row: dynamic outer stride
    assert matrix[:, 0].tolist() == [5.0, 1.0, 5.0, 1.0]


def test_plain_python_buffers_work_without_numpy() -> None:
    import array

    module, fns = build_buffer_kernels()
    compiled = codegen.compile(module)
    assert compiled.function(fns["sum_vector"])(array.array("f", [1.5, 2.5])) == 4.0
    ints = array.array("i", [0, 0, 0, 0])
    compiled.function(fns["fill4"])(ints, -3)
    assert list(ints) == [-3, -3, -3, -3]


def test_buffers_are_checked_against_the_memref_type() -> None:
    import numpy as np

    module, fns = build_buffer_kernels()
    compiled = codegen.compile(module)
    sum_vector = compiled.function(fns["sum_vector"])
    fill4 = compiled.function(fns["fill4"])
    with pytest.raises(TypeError, match="expected float32 items"):
        sum_vector(np.zeros(3, dtype=np.float64))
    with pytest.raises(TypeError, match="expected 1 dimensions, got 2"):
        sum_vector(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="dimension 0 must be 4, got 5"):
        fill4(np.zeros(5, dtype=np.int32), 1)
    with pytest.raises(TypeError, match="stride of dimension 0 must be 1"):
        fill4(np.zeros(8, dtype=np.int32)[::2], 1)  # static strides are compiled in
    readonly = np.zeros(4, dtype=np.int32)
    readonly.setflags(write=False)
    with pytest.raises(TypeError, match="writable buffer"):
        fill4(readonly, 1)
    with pytest.raises(TypeError, match="writable buffer"):
        sum_vector(1.0)  # type: ignore[arg-type]


# -- Pointers and strings: hello world -------------------------------------------


def build_hello_world() -> tuple[ir.Module, func.FuncOp]:
    """main() calls puts on a pointer to a global string constant."""
    from mlir_python.dialects import llvm

    i8, i32 = ir.IntegerType(8), ir.IntegerType(32)
    message = "Hello, world!\0"  # C strings end in NUL
    pointer = llvm.PointerType()
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        llvm.GlobalOp(
            llvm.ArrayType(i8, len(message)),
            "greeting",
            llvm.Linkage.INTERNAL,
            constant=True,
            value=ir.StringAttr(message),
        )
        func.FuncOp("puts", ir.FunctionType([pointer], [i32]), sym_visibility="private")
        main = func.FuncOp("main", ir.FunctionType([], [i32]))
    with ir.InsertionPoint(main.add_entry_block()):
        greeting = llvm.AddressOfOp(pointer, "greeting").result
        func.CallOp("puts", [greeting], [i32])
        func.ReturnOp([arith.ConstantOp(ir.IntegerAttr(0, i32)).result])
    module.verify()
    return module, main


def test_pointer_types_are_typed() -> None:
    from mlir_python.dialects import llvm

    pointer = llvm.PointerType()
    assert str(pointer) == "!llvm.ptr" and pointer.address_space == 0
    assert str(llvm.PointerType(1)) == "!llvm.ptr<1>"
    assert isinstance(ir.Type.parse("!llvm.ptr"), llvm.PointerType)
    array = llvm.ArrayType(ir.IntegerType(8), 14)
    assert str(array) == "!llvm.array<14 x i8>" and array.size == 14
    struct = llvm.StructType([ir.IntegerType(32), pointer])
    assert str(struct) == "!llvm.struct<(i32, ptr)>" and struct.elements[1] == pointer
    printf = llvm.FunctionType(ir.IntegerType(32), [pointer], variadic=True)
    assert str(printf) == "!llvm.func<i32 (ptr, ...)>" and printf.variadic
    assert str(llvm.VoidType()) == "!llvm.void"


def test_hello_world_runs_in_the_jit(capfd: pytest.CaptureFixture[str]) -> None:
    module, main = build_hello_world()
    assert codegen.compile(module).function(main)() == 0
    ctypes.CDLL(None).fflush(None)  # puts writes through C's buffered stdout
    assert capfd.readouterr().out == "Hello, world!\n"


@needs_cc
def test_hello_world_links_into_an_executable(tmp_path: Path) -> None:
    assert CC is not None
    module, _ = build_hello_world()
    codegen.lower_to_llvm(module)
    obj = tmp_path / "hello.o"
    codegen.translate_to_llvm_ir(module).write_object(str(obj))
    exe = tmp_path / "hello"
    subprocess.run([CC, str(obj), "-o", str(exe)], check=True)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert result.stdout == "Hello, world!\n"


# -- Binaries -------------------------------------------------------------------------


@needs_cc
def test_build_executable(tmp_path: Path) -> None:
    module, _ = build_hello_world()
    before = str(module)
    exe = codegen.build_executable(module, tmp_path / "hello")
    assert exe.is_absolute() and exe.exists()
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert result.stdout == "Hello, world!\n"
    assert str(module) == before  # built from a copy


@needs_cc
def test_executable_exit_status_is_mains_result(tmp_path: Path) -> None:
    i32 = ir.IntegerType(32)
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        main = func.FuncOp("main", ir.FunctionType([], [i32]))
    with ir.InsertionPoint(main.add_entry_block()):
        func.ReturnOp([arith.ConstantOp(ir.IntegerAttr(42, i32)).result])
    exe = codegen.build_executable(module, tmp_path / "answer")
    assert subprocess.run([str(exe)], check=False).returncode == 42


@needs_cc
def test_build_shared_library(tmp_path: Path) -> None:
    module, _ = build_kernels()
    library = codegen.build_shared_library(module, tmp_path / "libkernels.so")
    lib = ctypes.CDLL(str(library))
    lib.add.argtypes, lib.add.restype = [ctypes.c_int32, ctypes.c_int32], ctypes.c_int32
    assert lib.add(20, 22) == 42


def test_executable_needs_a_c_compatible_main(tmp_path: Path) -> None:
    module, _ = build_kernels()  # no main
    with pytest.raises(ValueError, match="an executable needs `main`"):
        codegen.build_executable(module, tmp_path / "nope")
    with ir.InsertionPoint(module.body):
        func.FuncOp("main", ir.FunctionType([ir.F32Type()], [ir.IntegerType(32)]))
    main = module.body.operations[-1]
    assert isinstance(main, func.FuncOp)
    with ir.InsertionPoint(main.add_entry_block()):
        func.ReturnOp([arith.ConstantOp(ir.IntegerAttr(0, ir.IntegerType(32))).result])
    with pytest.raises(ValueError, match=r"must be\s+\(\) -> i32"):
        codegen.build_executable(module, tmp_path / "nope")


def test_link_errors_carry_the_linker_output(tmp_path: Path) -> None:
    module, _ = build_hello_world()
    with pytest.raises(codegen.LinkError, match="no linker found"):
        codegen.build_executable(module, tmp_path / "x", linker="no-such-linker")
    if CC is None:
        return
    i32 = ir.IntegerType(32)
    broken = ir.Module()
    with ir.InsertionPoint(broken.body):
        func.FuncOp("missing", ir.FunctionType([], [i32]), sym_visibility="private")
        main = func.FuncOp("main", ir.FunctionType([], [i32]))
    with ir.InsertionPoint(main.add_entry_block()):
        func.ReturnOp(func.CallOp("missing", [], [i32]).results)
    with pytest.raises(codegen.LinkError, match="missing"):
        codegen.build_executable(broken, tmp_path / "broken")
