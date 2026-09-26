"""Hand-written helpers for the ``func`` dialect, exported from
``mlir_python.dialects.func``.

They take and return operation objects rather than symbol names, derive types
from signatures, and check calls and returns when the IR is built::

    with InsertionPoint(module.body):
        puts = func.declare("puts", [pointer], [i32])

        @func.define([], [i32])
        def main() -> Value:
            func.call(puts, [greeting])
            return zero
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .._mlir_python import FunctionType, InsertionPoint, Type, Value
from .._mlir_python.func import CallOp, FuncOp, ReturnOp

__all__ = ["Returned", "call", "declare", "define"]

type Returned = Value | Sequence[Value] | None
"""What a ``define`` body returns: nothing, the one result, or all results."""


def declare(name: str, inputs: Sequence[Type], results: Sequence[Type] = ()) -> FuncOp:
    """Declare a function defined elsewhere, such as ``puts`` from the C
    library, like a C prototype. It is created at the current insertion point.

    Example: ``puts = func.declare("puts", [llvm.PointerType()], [i32])``.
    """
    return FuncOp(name, FunctionType(inputs, results), sym_visibility="private")


def _as_values(returned: Returned) -> list[Value]:
    if returned is None:
        return []
    if isinstance(returned, Value):
        return [returned]
    return list(returned)


def define(
    inputs: Sequence[Type] = (),
    results: Sequence[Type] = (),
    *,
    name: str | None = None,
    private: bool = False,
) -> Callable[[Callable[..., Returned]], FuncOp]:
    """Define a function from a Python function that builds its body.

    The decorated function is called once, inside the new function's entry
    block, with one ``BlockArgument`` per input. Whatever it returns becomes
    the ``func.return``, checked against ``results``. The decorator replaces
    it with the ``func.FuncOp``::

        @func.define([i32, i32], [i32])
        def add(a: Value, b: Value) -> Value:
            return arith.AddIOp(a, b).result

    Args:
        inputs: Argument types.
        results: Result types.
        name: Symbol name; the Python function's name by default.
        private: Make the symbol private to the module.

    Raises:
        TypeError: If the body takes a different number of arguments, or
            returns values whose count or types do not match ``results``.
    """

    def build(body: Callable[..., Returned]) -> FuncOp:
        symbol = name or body.__name__
        fn = FuncOp(
            symbol,
            FunctionType(inputs, results),
            sym_visibility="private" if private else None,
        )
        entry = fn.add_entry_block()
        with InsertionPoint(entry):
            try:
                returned = body(*entry.arguments)
            except TypeError as error:
                if len(entry.arguments) != body.__code__.co_argcount:
                    raise TypeError(
                        f"@{symbol} takes {len(entry.arguments)} arguments, but its "
                        f"body accepts {body.__code__.co_argcount}"
                    ) from error
                raise
            values = _as_values(returned)
            if entry.terminator is not None:
                if values:
                    raise TypeError(
                        f"@{symbol}'s body both added a terminator and returned values"
                    )
                return fn
            actual = [value.type for value in values]
            if actual != list(results):
                raise TypeError(
                    f"@{symbol} must return ({', '.join(map(str, results))}) "
                    f"but its body returned ({', '.join(map(str, actual))})"
                )
            ReturnOp(values)
        return fn

    return build


def call(callee: FuncOp, arguments: Sequence[Value] = ()) -> CallOp:
    """Call ``callee`` with ``arguments``, taking the result types from its
    signature. Returns the ``func.CallOp``; its results are the call's.

    Raises:
        TypeError: If the arguments do not match ``callee``'s inputs.
    """
    signature = callee.function_type
    expected = signature.inputs
    if len(arguments) != len(expected):
        raise TypeError(
            f"@{callee.sym_name} takes {len(expected)} arguments, got {len(arguments)}"
        )
    for index, (argument, type_) in enumerate(zip(arguments, expected, strict=True)):
        if argument.type != type_:
            raise TypeError(
                f"argument {index} of @{callee.sym_name} must be {type_}, "
                f"got {argument.type}"
            )
    return CallOp(callee.sym_name, arguments, signature.results)
