"""Arrays (Array[T]): heap buffers with a length, freed automatically.

Functions are checked against running them as plain Python, where arrays are
lists.
"""

import resource
import subprocess
import sys
from pathlib import Path

import pytest

from mlir_python.lang import (
    Array,
    CompileError,
    Module,
    array,
    f32,
    f64,
    i8,
    i32,
    i64,
    ptr,
    u8,
    u32,
)

m = Module("arrays")


@m.function
def squares(n: i32) -> Array[i32]:
    xs = array(i32, n)
    for i in range(n):
        xs[i] = i * i
    return xs


@m.function
def total(xs: Array[i64]) -> i64:
    s = 0
    for x in xs:
        s += x
    return s


@m.function
def ends(xs: Array[f64]) -> f64:
    return xs[0] * 100.0 + xs[-1] + f64(len(xs))


@m.function
def display() -> i64:
    ys = [3, 1, 4, 1, 5, 9, 2, 6]
    return total(ys) * 10 + ys[-3]


@m.function
def typed_empty() -> i64:
    xs: Array[i64] = []
    return len(xs)


@m.function
def reverse(xs: Array[i32]) -> Array[i32]:
    n = len(xs)
    out = array(i32, n)
    for i in range(n):
        out[i] = xs[n - 1 - i]
    return out


@m.function
def double_in_place(xs: Array[f32]) -> None:
    for i in range(len(xs)):
        xs[i] *= 2.0


@m.function
def split_evens(xs: Array[i64]) -> tuple[Array[i64], Array[i64]]:
    count = 0
    for x in xs:
        if x % 2 == 0:
            count += 1
    evens = array(i64, count)
    odds = array(i64, len(xs) - count)
    e = 0
    o = 0
    for x in xs:
        if x % 2 == 0:
            evens[e] = x
            e += 1
        else:
            odds[o] = x
            o += 1
    return evens, odds


@m.function
def truthiness(n: i32) -> i32:
    xs = array(u8, n)
    if xs:
        return 1
    return 0


@m.function
def unsigned_and_small(n: i32) -> Array[u32]:
    xs = array(u32, n)
    for i in range(n):
        xs[i] = u32(i) + 4000000000
    return xs


@m.function
def bytes_sum(xs: Array[i8]) -> i32:
    s = 0
    for x in xs:
        s += i32(x)
    return s


@m.function
def flags(n: i32) -> Array[bool]:
    xs = array(bool, n)
    for i in range(n):
        xs[i] = i % 3 == 0
    return xs


CASES = [
    (squares, [(0,), (7,)]),
    (total, [([],), ([1, 2, 3],), ([-5, 10**12],)]),
    (ends, [([1.5],), ([1.0, 2.0, 3.25],)]),
    (display, [()]),
    (typed_empty, [()]),
    (reverse, [([],), ([1, 2, 3, 4],)]),
    (split_evens, [([1, 2, 3, 4, 5, 6, 7],), ([],)]),
    (truthiness, [(0,), (3,)]),
    (unsigned_and_small, [(3,)]),
    (bytes_sum, [([-1, -2, 127],)]),
    (flags, [(7,)]),
]


@pytest.mark.parametrize(
    ("index", "args"),
    [(i, a) for i, (_, calls) in enumerate(CASES) for a in calls],
)
def test_results_match_python(index: int, args: tuple[object, ...]) -> None:
    function = CASES[index][0]
    expected = function.python(*[list(a) if isinstance(a, list) else a for a in args])
    assert function(*args) == expected


def test_arrays_are_mutable_like_lists() -> None:
    values = [1.0, 2.5, -3.0]
    double_in_place(values)
    assert values == [2.0, 5.0, -6.0]


def test_buffers_pass_without_copies() -> None:
    import array as pyarray

    values = pyarray.array("f", [1.0, 2.0])
    double_in_place(values)  # pyright: ignore[reportArgumentType]  # buffers pass too
    assert values.tolist() == [2.0, 4.0]


def test_unused_arrays_are_freed() -> None:
    """3 GB of 1 MB arrays, each dead after one iteration."""
    churn = Module("churn")

    @churn.function
    def allocate_many(rounds: i32) -> i64:
        total = 0
        for r in range(rounds):
            xs = array(i64, 131072)
            xs[r % 131072] = i64(r)
            total += xs[r % 131072]
        return total

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert allocate_many(3000) == sum(range(3000))
    grown_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    assert grown_kb < 200_000, f"memory grew by {grown_kb} KB"
    assert "memref.dealloc" in str(churn.linked())


def test_arrays_pass_to_c_as_pointers() -> None:
    c_implicit = Module("array_c_implicit")

    @c_implicit.extern
    def memset(dest: ptr, value: i32, count: i64) -> ptr: ...

    @c_implicit.function
    def fill_implicitly(n: i32) -> Array[u8]:
        xs = array(u8, n)
        memset(xs, 1, i64(n))  # pyright: ignore[reportArgumentType]  # C-style decay
        return xs

    assert fill_implicitly(2) == [1, 1]

    c = Module("array_c")

    @c.extern(name="memset")
    def set_bytes(dest: ptr, value: i32, count: i64) -> ptr: ...

    @c.function
    def fill_bytes(n: i32) -> Array[u8]:
        xs = array(u8, n)
        set_bytes(ptr(xs), 7, i64(n))
        return xs

    assert fill_bytes(4) == [7, 7, 7, 7]


def test_out_of_range_indices_stop_the_program(tmp_path: Path) -> None:
    script = tmp_path / "oob.py"
    script.write_text(
        "from mlir_python.lang import Array, Module, i32\n"
        "m = Module()\n"
        "@m.function\n"
        "def at(xs: Array[i32], i: i32) -> i32:\n"
        "    return xs[i]\n"
        "print(at([1, 2, 3], -3), flush=True)\n"
        "print(at([1, 2, 3], 3), flush=True)\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert result.stdout.splitlines() == ["1"]
    assert result.returncode != 0
    assert "index out of range" in result.stderr


def compile_error(tmp_path: Path, body: str) -> str:
    path = tmp_path / "snippet.py"
    path.write_text(
        "from mlir_python.lang import *\n"
        "m = Module()\n"
        f"@m.function\ndef bad() -> i32:\n{body}\n    return 0\n"
    )
    namespace: dict[str, object] = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
    module = namespace["m"]
    assert isinstance(module, Module)
    with pytest.raises(CompileError) as caught:
        _ = module.mlir
    return str(caught.value.msg)


def test_array_mistakes_are_compile_errors(tmp_path: Path) -> None:
    assert "give an empty list its type" in compile_error(tmp_path, "    xs = []")
    assert "array indices must be integers" in compile_error(
        tmp_path, "    xs = [1, 2]\n    y = xs[1.5]"
    )
    assert "pointer indices must be integers" in compile_error(
        tmp_path, "    p = stack(i32)\n    y = p[0.5]"
    )
    assert "len() takes an array" in compile_error(tmp_path, "    n = len(3)")
    assert "do not support +" in compile_error(
        tmp_path, "    xs = [1]\n    ys = xs + xs"
    )


def test_arrays_cannot_be_struct_fields() -> None:
    from mlir_python.lang import struct

    with pytest.raises(TypeError, match="arrays cannot be stored in structs"):

        @struct
        class Holder:
            values: Array[i32]

        del Holder
