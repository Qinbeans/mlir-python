"""Pointer arithmetic (``p + n``, ``p - n``) and ``atomic_add``.

``atomic_add`` is checked across real threads: four ``pthread`` workers add to
one counter, and no increment is lost.
"""

from pathlib import Path

import pytest

from mlir_python.lang import (
    CompileError,
    Fn,
    Module,
    Program,
    Ptr,
    atomic_add,
    i32,
    i64,
    ptr,
    stack,
    u8,
    u64,
)

program = Program()

THREADS, INCREMENTS = 4, 100_000


@program.extern
def pthread_create(
    thread: Ptr[u64], attributes: i64, start: Fn[[ptr], ptr], argument: ptr
) -> i32: ...


@program.extern
def pthread_join(thread: u64, result: i64) -> i32: ...


@program.function
def walk() -> i64:
    values = stack(i64, 4)
    for i in range(4):
        values[i] = (i + 1) * 10
    last = values + 3
    second = last - 2
    return last[0] + second[0] + (values + 1)[1]  # 40 + 20 + 30


@program.function
def byte_steps() -> i64:
    data = stack(u8, 8)
    for i in range(8):
        data[i] = u8(i)
    return i64((data + 5)[0]) + i64((data + 7 - 4)[0])  # 5 + 3


@program.function
def count_up() -> tuple[i64, i64]:
    counter = stack(i64)
    counter[0] = 5
    before = atomic_add(counter, 3)
    atomic_add(counter, -1)
    return before, counter[0]


@program.function
def worker(argument: ptr) -> ptr:
    counter = Ptr[i64](argument)
    for _ in range(INCREMENTS):
        atomic_add(counter, 1)
    return argument


@program.function
def count_in_threads() -> i64:
    threads = stack(u64, THREADS)
    counter = stack(i64)
    counter[0] = 0
    for i in range(THREADS):
        pthread_create(threads + i, 0, worker, counter)
    for i in range(THREADS):
        pthread_join(threads[i], 0)
    return counter[0]


def test_pointer_arithmetic() -> None:
    assert walk() == 90
    assert byte_steps() == 8


def test_atomic_add_returns_the_old_value() -> None:
    assert count_up() == (5, 7)


def test_atomic_add_across_threads() -> None:
    assert count_in_threads() == THREADS * INCREMENTS


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


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            "p = stack(i64)\n    return (p * 2)[0]",
            "pointer arithmetic is p + n or p - n on a typed pointer",
        ),
        (
            "p = stack(i64)\n    return (p + 1.5)[0]",
            "a pointer moves by an integer count",
        ),
        (
            "p = stack(i64)\n    return atomic_add(p)",
            "atomic_add() takes a pointer and an amount",
        ),
        (
            "p = stack(f64)\n    return i64(atomic_add(p, 1.0))",
            "atomic_add() takes a pointer to an integer",
        ),
    ],
)
def test_errors(body: str, message: str, tmp_path: Path) -> None:
    error = compile_error(
        f"@program.function\ndef bad() -> i64:\n    {body}\n", tmp_path
    )
    assert message in str(error)
