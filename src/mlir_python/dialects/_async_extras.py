"""Hand-written helpers for the ``async`` dialect, exported from
``mlir_python.dialects.async_dialect``.

``execute`` computes the task's types from what it computes, so a task is
written like a function body::

    task = async_dialect.execute(results=[i64])
    with InsertionPoint(task.body):
        async_dialect.YieldOp([arith.MulIOp(x, x).result])
    value = async_dialect.await_value(task.body_results[0])
"""

from __future__ import annotations

from collections.abc import Sequence

from .._mlir_python import InsertionPoint, Location, Type, Value
from .._mlir_python.async_dialect import AwaitOp, ExecuteOp, TokenType, ValueType

__all__ = ["await_value", "execute"]


def execute(
    *,
    results: Sequence[Type] = (),
    dependencies: Sequence[Value] = (),
    operands: Sequence[Value] = (),
    location: Location | None = None,
    ip: InsertionPoint | None = None,
) -> ExecuteOp:
    """Start an asynchronous task (``async.execute``).

    The task waits for the ``dependencies`` tokens, receives ``operands``
    (``!async.value`` operands unwrapped) as its body's arguments, and
    produces one ``!async.value`` per type in ``results``. Its ``token``
    completes when it does. Fill ``body`` and end it with ``async.yield``
    (already there when there are no results).
    """
    token = TokenType(context=results[0].context if results else None)
    return ExecuteOp(
        token,
        list(dependencies),
        list(operands),
        [ValueType(kind) for kind in results],
        location=location,
        ip=ip,
    )


def await_value(
    value: Value, *, location: Location | None = None, ip: InsertionPoint | None = None
) -> Value:
    """Wait for ``value`` (an ``!async.value<T>``) and return the ``T`` it
    holds (``async.await``). For a token, use ``AwaitOp(token)``.

    Raises:
        TypeError: If ``value`` is not an ``!async.value``.
    """
    kind = value.type
    if not isinstance(kind, ValueType):
        raise TypeError(f"await_value() takes an !async.value, got {kind}")
    return AwaitOp(value, result_type=kind.value_type, location=location, ip=ip).result
