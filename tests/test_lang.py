"""mlir_python.lang: Python functions compiled to native code.

Most functions here are checked differentially: compiled results must equal
running the same function as plain Python.
"""

import ctypes
import itertools
import shutil
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from mlir_python import codegen as lang_codegen
from mlir_python import lang
from mlir_python.codegen import OptLevel as lang_opt
from mlir_python.lang import (
    CompileError,
    Program,
    cstr,
    f32,
    f64,
    i8,
    i32,
    i64,
    u8,
    u32,
)

program = Program()
LIMIT = 10  # a global constant, usable from compiled code


@program.function
def arithmetic(a: i32, b: i32) -> i32:
    return a * 3 - b + (a ^ b) - (a & 7) + (b | 1)


@program.function
def floor_div(a: i64, b: i64) -> i64:
    return a // b


@program.function
def modulo(a: i64, b: i64) -> i64:
    return a % b


@program.function
def true_div(a: i32, b: i32) -> f64:
    return a / b


@program.function
def float_ops(x: f64, y: f64) -> f64:
    return (x // y) + (x % y) * 2.0 - x / y + abs(x) ** 2.0


@program.function
def shifts(a: i32, n: i32) -> i32:
    return (a << n) + (a >> n)


@program.function
def unsigned_ops(a: u32, b: u32) -> u32:
    return a // b + a % b + (a >> 1)


@program.function
def compare_chain(a: i32, b: i32, c: i32) -> bool:
    return a < b <= c != a


@program.function
def logic(a: bool, b: bool) -> bool:
    return (a and not b) or (b and not a)


@program.function
def pick(n: i32) -> i32:
    return n * 2 if n > 0 else -n


@program.function
def clamp(x: i32, low: i32, high: i32) -> i32:
    return min(max(x, low), high)


@program.function
def sum_below(n: i64) -> i64:
    total = 0
    for i in range(n):
        total += i
    return total


@program.function
def loops(n: i32) -> i32:
    total = 0
    for i in range(n, 0, -2):
        if i % 3 == 0:
            continue
        total += i
    for j in range(LIMIT):
        if j * j > n:
            break
        total += j
    return total


@program.function
def stepped(start: i32, stop: i32, step: i32) -> i32:
    count = 0
    for _ in range(start, stop, step):
        count += 1
    return count


@program.function
def first_divisor(n: i32) -> i32:
    d = 2
    while True:
        if d * d > n:
            return n
        if n % d == 0:
            return d
        d += 1


@program.function
def classify(n: i32) -> i32:
    if n < 0:
        return -1
    elif n == 0:
        return 0
    elif n < 10:
        return 1
    return 2


@program.function
def fib(n: i32) -> i32:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


@program.function
def divmod_(a: i32, b: i32) -> tuple[i32, i32]:
    return a // b, a % b


@program.function
def swap_sum(a: i32, b: i32) -> i32:
    q, r = divmod_(a, b)
    a, b = b, a
    return a * 100 + b + q + r


@program.function
def conversions(x: f64) -> i32:
    small: i8 = i8(i32(x) % 100)
    return i32(small) + i32(u8(i32(x) & 255)) + i32(f32(x) > 1.5)


@program.function
def is_even(n: i32) -> bool:
    return n == 0 or is_odd(n - 1)


@program.function
def is_odd(n: i32) -> bool:
    return n != 0 and is_even(n - 1)


def check(
    compiled: Callable[..., object],
    python: Callable[..., object],
    inputs: Iterable[tuple[object, ...]],
) -> None:
    for args in inputs:
        assert compiled(*args) == pytest.approx(python(*args)), args


SMALL = range(-7, 8)
NONZERO = [n for n in SMALL if n != 0]


def test_integer_arithmetic_matches_python() -> None:
    check(arithmetic, arithmetic.python, itertools.product(SMALL, SMALL))
    check(floor_div, floor_div.python, itertools.product(SMALL, NONZERO))
    check(modulo, modulo.python, itertools.product(SMALL, NONZERO))
    check(true_div, true_div.python, itertools.product(SMALL, NONZERO))
    check(shifts, shifts.python, itertools.product(SMALL, range(4)))


def test_float_arithmetic_matches_python() -> None:
    values = [-5.5, -2.0, -0.5, 0.25, 1.0, 3.75]
    check(float_ops, float_ops.python, itertools.product(values, values))


def test_unsigned_arithmetic() -> None:
    check(
        unsigned_ops,
        unsigned_ops.python,
        itertools.product(range(0, 50, 7), range(1, 9)),
    )


def test_comparisons_logic_and_conditionals_match_python() -> None:
    check(
        compare_chain, compare_chain.python, itertools.product(range(-2, 3), repeat=3)
    )
    check(logic, logic.python, itertools.product([False, True], repeat=2))
    check(pick, pick.python, [(n,) for n in SMALL])
    check(clamp, clamp.python, [(n, -3, 4) for n in SMALL])


def test_loops_match_python() -> None:
    check(sum_below, sum_below.python, [(n,) for n in (0, 1, 10, 1000)])
    check(loops, loops.python, [(n,) for n in range(0, 40, 3)])
    check(
        stepped,
        stepped.python,
        [
            (a, b, s)
            for a, b, s in itertools.product(range(-6, 7, 3), repeat=3)
            if s != 0
        ],
    )
    check(first_divisor, first_divisor.python, [(n,) for n in range(2, 60)])
    check(classify, classify.python, [(n,) for n in (-5, 0, 3, 42)])


def test_recursion_tuples_and_mutual_calls() -> None:
    assert fib(20) == 6765
    assert divmod_(17, 5) == (3, 2) and divmod_(-17, 5) == (-4, 3)
    check(swap_sum, swap_sum.python, [(7, 3), (-9, 4), (100, 7)])
    check(is_even, is_even.python, [(n,) for n in range(12)])


def test_conversions_match_python_semantics() -> None:
    assert conversions(7.9) == 7 + 7 + 1  # truncation toward zero, like int()
    assert conversions(1.2) == 1 + 1 + 0


def test_literals_take_the_type_of_their_use() -> None:
    # `total = 0` becomes i32 because the function returns it as i32.
    text = str(program)
    assert "func.func @loops(%arg0: i32) -> i32" in text
    assert "i64" not in text.split("func.func @loops")[1].split("func.func")[0]


def test_functions_reject_bad_arguments() -> None:
    with pytest.raises(TypeError, match="takes 2 arguments, got 1"):
        arithmetic(1)  # type: ignore[call-arg]
    with pytest.raises(OverflowError):
        arithmetic(2**40, 1)


# -- Hello world, executables, and libraries ------------------------------------

hello = Program()


@hello.extern
def puts(text: cstr) -> i32: ...


@hello.function
def square(x: i32) -> i32:
    return x * x


@hello.main
def main() -> None:
    puts("Hello, world!")


def test_hello_world_runs(capfd: pytest.CaptureFixture[str]) -> None:
    assert main() is None
    ctypes.CDLL(None).fflush(None)
    assert capfd.readouterr().out == "Hello, world!\n"
    assert "declare i32 @puts(ptr)" in str(hello.llvm_ir(opt_level=lang_opt.O0))


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_hello_world_builds(tmp_path: Path) -> None:
    exe = hello.build_executable(tmp_path / "hello")
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    assert result.stdout == "Hello, world!\n"
    library = hello.build_shared_library(tmp_path / "libhello.so")
    lib = ctypes.CDLL(str(library))
    assert lib.square(12) == 144


def test_externs_are_called_from_compiled_code_only() -> None:
    with pytest.raises(TypeError, match="external"):
        puts("hi")


def test_program_structure() -> None:
    assert [f.name for f in hello.functions] == ["puts", "square", "main"]
    with pytest.raises(ValueError, match="already has a main"):
        hello.main(square.python)
    with pytest.raises(ValueError, match="needs an entry point"):
        program.build_executable("never")


# -- Compile errors point at the source -------------------------------------------


def compile_error(source: str, tmp_path: Path) -> CompileError:
    """Compiles ``source`` (defining functions on ``program``) from a file."""
    path = tmp_path / "snippet.py"
    path.write_text("from mlir_python.lang import *\nprogram = Program()\n" + source)
    namespace: dict[str, object] = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
    built = namespace["program"]
    assert isinstance(built, Program)
    with pytest.raises(CompileError) as info:
        _ = built.mlir
    assert info.value.filename == str(path)
    return info.value


ERRORS = [
    ("def f(a: i32) -> i32:\n    return b\n", 5, "name 'b' is not defined"),
    (
        "def f(a: i32) -> i32:\n    if a:\n        x = 1\n    return x\n",
        7,
        "'x' may be unassigned",
    ),
    (
        "def f(a: i32) -> i32:\n    return a + 1.5\n",
        5,
        "expected i32, got the float 1.5",
    ),
    (
        "def f(a: i32, b: f64) -> i32:\n    return a + b\n",
        5,
        "cannot combine i32 and f64",
    ),
    (
        "def f(a: i32) -> i32:\n    if a:\n        return 1\n",
        4,
        "must return i32 on every path",
    ),
    (
        "def f(a: i32) -> i32:\n    print(a)\n    return a\n",
        5,
        "print() is not available",
    ),
    (
        "def f(a: i32) -> i32:\n    try:\n        pass\n    finally:\n        pass\n    return a\n",
        5,
        "'try' statements are not supported",
    ),
    ("def f(a) -> i32:\n    return a\n", 4, "parameter 'a' needs a type annotation"),
    ("def f(a: i32) -> i32:\n    return f(a, a)\n", 5, "f() takes 1 arguments, got 2"),
    ("def f(a: i8) -> i8:\n    return a + 300\n", 5, "300 does not fit i8"),
]


@pytest.mark.parametrize(("body", "line", "message"), ERRORS)
def test_compile_errors(body: str, line: int, message: str, tmp_path: Path) -> None:
    error = compile_error("@program.function\n" + body, tmp_path)
    assert message in error.msg
    assert error.lineno == line
    assert error.text is not None


def test_extern_and_main_shapes_are_checked(tmp_path: Path) -> None:
    error = compile_error(
        "@program.extern\ndef f(a: i32) -> i32:\n    return a\n", tmp_path
    )
    assert "body must be '...'" in error.msg
    error = compile_error(
        "@program.main\ndef start(a: i32) -> i32:\n    return a\n", tmp_path
    )
    assert "main takes no parameters" in error.msg


def test_the_mlir_layer_stays_available() -> None:
    from mlir_python import Module
    from mlir_python.dialects import func

    assert isinstance(program.mlir, Module)
    names = [
        op.sym_name
        for op in program.mlir.body.operations
        if isinstance(op, func.FuncOp)
    ]
    assert "fib" in names
    assert lang.i32 is i32


# -- C variadic functions ----------------------------------------------------------

variadic = Program()


@variadic.extern
def printf(format: cstr, *args: float | str) -> i32: ...


@variadic.function
def report(small: i8, ratio: f32, flag: bool, big: i64) -> i32:
    answer: i32 = 42  # a type used only in a local annotation
    return printf("%d %d %.2f %d %ld %s %d\n", answer, small, ratio, flag, big, "ok", 7)


def test_printf_with_c_argument_promotions(capfd: pytest.CaptureFixture[str]) -> None:
    written = report(-3, 1.5, True, 2**40)
    ctypes.CDLL(None).fflush(None)
    assert capfd.readouterr().out == f"42 -3 1.50 1 {2**40} ok 7\n"
    assert written == len(f"42 -3 1.50 1 {2**40} ok 7\n")
    assert "llvm.func @printf(!llvm.ptr, ...) -> i32" in str(variadic)


def test_variadic_declaration_errors(tmp_path: Path) -> None:
    error = compile_error(
        "@program.extern\ndef f(text: cstr, **args) -> i32: ...\n", tmp_path
    )
    assert "C functions have no keyword arguments" in error.msg
    assert error.lineno == 4 and error.offset == 21  # at **args
    error = compile_error(
        "@program.function\ndef f(*args: i32) -> i32:\n    return 0\n", tmp_path
    )
    assert "only @program.extern functions can take *args" in error.msg
    error = compile_error(
        "@program.extern\ndef g(text: cstr, *args: int) -> i32: ...\n"
        "@program.function\ndef f() -> i32:\n    return g()\n",
        tmp_path,
    )
    assert "g() takes at least 1 arguments, got 0" in error.msg
    error = compile_error(
        "@program.function\ndef f(a: i32) -> i32:\n    return f(a=a)\n", tmp_path
    )
    assert "positional arguments only" in error.msg


# -- Pointers and stack memory ----------------------------------------------------

memory = Program()


@memory.extern
def scanf(format: cstr, *args: object) -> i32: ...


@memory.extern(name="printf")  # a Python name other than the C symbol
def mprintf(format: cstr, *args: object) -> i32: ...


@memory.function
def total(values: lang.Ptr[i32], count: i32) -> i32:
    result = 0
    for i in range(count):
        result += values[i]
    return result


@memory.function
def squares_sum(n: i32) -> i32:
    values = lang.stack(i32, 16)
    for i in range(n):
        values[i] = i * i
    values[0] += 100
    raw: lang.ptr = values  # any pointer passes as an opaque ptr
    again = lang.Ptr[i32](raw)
    return total(again, n)


@memory.main
def scan_main() -> None:
    value = lang.stack(i32)
    scanf("%d", value)
    mprintf("%d\n", value[0] * 2)


def test_stack_memory_and_pointer_parameters() -> None:
    for n in (1, 5, 16):
        assert squares_sum(n) == 100 + sum(i * i for i in range(n))


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_scanf_writes_through_a_pointer(tmp_path: Path) -> None:
    exe = memory.build_executable(tmp_path / "scan")
    result = subprocess.run(
        [str(exe)], input="21\n", capture_output=True, text=True, check=True
    )
    assert result.stdout == "42\n"


def test_pointer_errors(tmp_path: Path) -> None:
    error = compile_error(
        "@program.function\ndef f(a: i32) -> i32:\n    return a[0]\n", tmp_path
    )
    assert "only arrays and pointers (Ptr[T]) can be indexed" in error.msg
    error = compile_error(
        "@program.function\ndef f(p: ptr) -> i32:\n    return p[0]\n", tmp_path
    )
    assert "convert with Ptr[i32](p)" in error.msg
    error = compile_error(
        "@program.function\ndef f() -> i32:\n    p = stack(i32, 0)\n    return 0\n",
        tmp_path,
    )
    assert "positive count" in error.msg
    error = compile_error(
        "@program.function\ndef f(p: Ptr[i32]) -> i32:\n    p[0] = 1.5\n    return 0\n",
        tmp_path,
    )
    assert "expected i32, got the float 1.5" in error.msg


def test_missing_externs_are_named_before_jit() -> None:
    lonely = Program()

    @lonely.extern
    def no_such_c_function(x: i32) -> i32: ...

    @lonely.function
    def use(x: i32) -> i32:
        return no_such_c_function(x)

    with pytest.raises(
        lang_codegen.LinkError, match="'no_such_c_function' of test_lang is not defined"
    ):
        use(1)
