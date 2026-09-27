"""Function values: ``Fn[[params], result]``, C function pointers.

A function named where a value is expected is its address; a value is called
like a function. They cross into C both ways: C calls compiled functions
(``qsort``'s comparator), and compiled code calls C functions through values.
"""

from pathlib import Path

import pytest

from mlir_python.lang import (
    CompileError,
    Fn,
    Module,
    Program,
    Ptr,
    cstr,
    i32,
    i64,
    ptr,
    stack,
    struct,
)
from mlir_python.lang.types import FnType

program = Program()


@program.extern
def qsort(base: ptr, count: i64, size: i64, compare: Fn[[ptr, ptr], i32]) -> None: ...


@program.extern
def strlen(text: cstr) -> i64: ...


@struct
class Handlers:
    on_value: Fn[[i32], i32]
    count: i32


@program.function
def double(x: i32) -> i32:
    return x * 2


@program.function
def square(x: i32) -> i32:
    return x * x


@program.function
def apply(f: Fn[[i32], i32], x: i32) -> i32:
    return f(x)


@program.function
def twice(f: Fn[[i32], i32], x: i32) -> i32:
    return apply(f, apply(f, x))


@program.function
def choose(x: i32) -> i32:
    g = double
    if x > 10:
        g = square
    return apply(g, x)


@program.function
def through_struct(x: i32) -> i32:
    handlers = Handlers(on_value=square, count=1)
    return handlers.on_value(x) + handlers.count


@program.function
def ascending(a: ptr, b: ptr) -> i32:
    return Ptr[i32](a)[0] - Ptr[i32](b)[0]


@program.function
def sort_four() -> i32:
    values = stack(i32, 4)
    values[0] = 40
    values[1] = 10
    values[2] = 30
    values[3] = 20
    qsort(values, 4, 4, ascending)
    return values[0] * 1000000 + values[1] * 10000 + values[2] * 100 + values[3]


@program.function
def c_through_value() -> i64:
    measure = strlen
    return measure("hello")


@program.function
def untyped_round_trip(x: i32) -> i32:
    raw: ptr = double
    back = Fn[[i32], i32](raw)
    return back(x)


@program.function
def twice_double(x: i32) -> i32:
    return twice(double, x)


def test_function_values_are_called_through() -> None:
    assert choose(3) == 6
    assert choose(12) == 144
    assert twice_double(5) == 20
    assert through_struct(5) == 26


def test_c_calls_compiled_functions() -> None:
    assert sort_four() == 10203040


def test_compiled_code_calls_c_through_values() -> None:
    assert c_through_value() == 5
    assert untyped_round_trip(4) == 8


def test_fn_types() -> None:
    kind = Fn[[i32, ptr], None]
    assert isinstance(kind, FnType)
    assert kind.name == "Fn[[i32, ptr], None]"
    assert kind == Fn[[i32, ptr], None]
    assert kind.size == 8
    with pytest.raises(TypeError, match="Fn\\[\\[parameter types\\], result type\\]"):
        Fn[i32, i32]


def test_function_values_cross_modules() -> None:
    other = Program()

    @other.function
    def use(x: i32) -> i32:
        return apply(double, x) + apply(square, x)

    assert use(3) == 15


def compile_error(source: str, tmp_path: Path) -> CompileError:
    path = tmp_path / "snippet.py"
    path.write_text("from mlir_python.lang import *\nprogram = Program()\n" + source)
    namespace: dict[str, object] = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
    built = namespace["program"]
    assert isinstance(built, Module)
    with pytest.raises(CompileError) as info:
        _ = built.mlir
    return info.value


ERRORS = [
    (
        (
            "@program.function\ndef f(x: i32) -> i32:\n    return x\n"
            "@program.function\ndef g(x: i32) -> i32:\n    h = f\n    return h(x, x)\n"
        ),
        "h() takes 1 arguments, got 2",
    ),
    (
        ("@program.function\ndef g(x: i32) -> i32:\n    return x(1)\n"),
        "'x' is i32, not a function value",
    ),
    (
        (
            "@program.function\ndef f(x: i32) -> tuple[i32, i32]:\n    return x, x\n"
            "@program.function\ndef g(x: i32) -> i32:\n    h = f\n    return 0\n"
        ),
        "f takes several results, so it cannot be a function value",
    ),
    (
        (
            "@program.extern\ndef printf(text: cstr, *args) -> i32: ...\n"
            "@program.function\ndef g(x: i32) -> i32:\n    h = printf\n    return 0\n"
        ),
        "printf takes C varargs",
    ),
    (
        (
            "@program.function\ndef f(x: i32) -> i32:\n    return x\n"
            "@program.function\ndef g(h: Fn[[i32], i32]) -> i32:\n    return h(1)\n"
            "@program.function\ndef k() -> i32:\n    return g(1)\n"
        ),
        "expected Fn[[i32], i32], got the int 1",
    ),
]


@pytest.mark.parametrize(("source", "message"), ERRORS)
def test_errors(source: str, message: str, tmp_path: Path) -> None:
    assert message in compile_error(source, tmp_path).msg


@program.function
def read_state(state: Ptr[i32], x: i32) -> i32:
    return state[0] + x


def test_typed_pointers_stand_for_opaque_ones_in_function_types() -> None:
    # A function taking Ptr[i32] fits a callback type taking ptr (C's void *).
    from mlir_python.lang._compiler import fits_function_type

    assert fits_function_type(Fn[[Ptr[i32], i32], i32], Fn[[ptr, i32], i32])
    assert fits_function_type(Fn[[ptr], None], Fn[[Ptr[i32]], None])
    assert not fits_function_type(Fn[[Ptr[i32]], i32], Fn[[i32], i32])
    assert not fits_function_type(Fn[[Ptr[i32]], i32], Fn[[ptr, ptr], i32])
    assert not fits_function_type(Fn[[Ptr[i32]], i32], Fn[[Ptr[i64]], i32])
    assert "read_state" in str(program.mlir)  # passing it compiles (see below)


@program.function
def pass_typed_callback(x: i32) -> i32:
    slot = stack(i32)
    slot[0] = 40
    return apply_state(read_state, slot, x)


@program.function
def apply_state(handler: Fn[[ptr, i32], i32], state: ptr, x: i32) -> i32:
    return handler(state, x)


def test_passing_a_typed_callback() -> None:
    assert pass_typed_callback(2) == 42
