"""Compiled code is structured control flow (scf), not raw branches (cf).

Each function is also checked against running it as plain Python.
"""

import itertools
import re

import pytest

import mlir_python as ir
from mlir_python.dialects import func
from mlir_python.lang import Function, Module, i32, i64

m = Module("structured")


@m.function
def sum_of_squares(n: i32) -> i32:
    total = 0
    for i in range(n):
        total += i * i
    return total


@m.function
def stepped(start: i32, stop: i32) -> i32:
    total = 0
    for i in range(start, stop, 3):
        total += i
    return total


@m.function
def nested(n: i32) -> i32:
    total = 0
    for i in range(n):
        for j in range(i):
            total += i * j
    return total


@m.function
def first_divisor(n: i32) -> i32:
    for d in range(2, n):
        if n % d == 0:
            return d
    return n


@m.function
def skip_threes(n: i32) -> i32:
    total = 0
    i = 0
    while True:
        i += 1
        if i > n:
            break
        if i % 3 == 0:
            continue
        total += i
    return total


@m.function
def countdown(n: i64) -> i64:
    steps = 0
    for _ in range(n, 0, -2):
        steps += 1
    return steps


@m.function
def collatz(n: i64) -> i64:
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


@m.function
def sign(x: i32) -> i32:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


CASES = [
    (sum_of_squares, [(0,), (1,), (10,)]),
    (stepped, [(0, 10), (5, 5), (-7, 8)]),
    (nested, [(0,), (6,)]),
    (first_divisor, [(2,), (91,), (97,)]),
    (skip_threes, [(0,), (10,)]),
    (countdown, [(0,), (9,), (10,)]),
    (collatz, [(1,), (27,)]),
    (sign, [(-5,), (0,), (3,)]),
]


@pytest.mark.parametrize(
    ("function", "args"),
    [(f, a) for f, calls in CASES for a in calls],
    ids=lambda value: getattr(value, "name", str(value)),
)
def test_results_match_python(
    function: Function[..., int], args: tuple[int, ...]
) -> None:
    assert function(*args) == function.python(*args)


def test_no_branches_remain() -> None:
    text = str(m.mlir)
    assert not re.search(r"\bcf\.", text), text


def test_counted_loops_become_scf_for() -> None:
    functions = {
        op.sym_name: op for op in m.mlir.body.operations if isinstance(op, func.FuncOp)
    }

    def ops(name: str) -> list[str]:
        found: list[str] = []

        def walk(op: ir.Operation) -> None:
            found.append(op.name)
            for region in op.regions:
                for block in region:
                    for inner in block.operations:
                        walk(inner)

        walk(functions[name])
        return found

    assert ops("sum_of_squares").count("scf.for") == 1
    assert ops("stepped").count("scf.for") == 1
    assert ops("nested").count("scf.for") == 2
    # Early exits keep a while loop, with the exit as a loop-carried flag.
    assert "scf.while" in ops("first_divisor")
    assert "scf.while" in ops("skip_threes")
    # scf.for counts up only; counting down stays a while loop.
    assert "scf.while" in ops("countdown")
    assert all(op != "scf.for" for op in ops("collatz"))


def test_structure_holds_for_every_small_input() -> None:
    for n in range(-3, 30):
        assert skip_threes(n) == skip_threes.python(n)
        assert first_divisor(max(n, 2)) == first_divisor.python(max(n, 2))
    for start, stop in itertools.product(range(-4, 5), repeat=2):
        assert stepped(start, stop) == stepped.python(start, stop)
