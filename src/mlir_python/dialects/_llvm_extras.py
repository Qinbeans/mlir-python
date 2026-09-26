"""Hand-written helpers for the ``llvm`` dialect, exported from
``mlir_python.dialects.llvm``.

They take and return operation objects rather than symbol names and compute
types and sizes, so they cannot disagree with what they refer to::

    with InsertionPoint(module.body):
        greeting = llvm.string_constant("greeting", "Hello, world!")
    ...
    pointer = llvm.address_of(greeting)
"""

from __future__ import annotations

from .._mlir_python import IntegerType, StringAttr, Value
from .._mlir_python.llvm import (
    AddressOfOp,
    ArrayType,
    GlobalOp,
    Linkage,
    PointerType,
)

__all__ = ["address_of", "string_constant"]


def string_constant(
    name: str, text: str, *, linkage: Linkage = Linkage.INTERNAL
) -> GlobalOp:
    """A constant global holding ``text`` as a NUL-terminated UTF-8 C string,
    created at the current insertion point.

    The array size is the byte length, so non-ASCII text is sized correctly;
    the terminating NUL is added for you.

    Args:
        name: Symbol name of the global.
        text: The string, without a trailing NUL.
        linkage: ``INTERNAL`` keeps it private to this module's object file.
    """
    data = text + "\0"
    size = len(data.encode("utf-8"))
    return GlobalOp(
        ArrayType(IntegerType(8), size),
        name,
        linkage,
        constant=True,
        value=StringAttr(data),
    )


def address_of(global_: GlobalOp) -> Value:
    """A pointer to ``global_`` (``!llvm.ptr`` in its address space)."""
    pointer = PointerType(global_.addr_space, context=global_.context)
    return AddressOfOp(pointer, global_.sym_name).result
