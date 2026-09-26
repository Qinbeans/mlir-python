"""Runtime definitions of the scalar types (see ``types.py`` for the public
module and ``types.pyi`` for what type checkers see).

Scalar types for ``mlir_python.lang`` function signatures.

Annotate parameters, results, and variables with these to fix their machine
representation; Python's own ``int``, ``float``, and ``bool`` mean ``i64``,
``f64``, and a 1-bit boolean.

Type checkers see ``i32`` and friends as aliases of ``int`` (``f32``/``f64``
of ``float``, ``cstr`` of ``str``; see ``types.pyi``), so function bodies are
ordinary, fully type-checked Python. Calling a type converts: ``i32(x)``,
``f64(n)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .._mlir_python import F32Type, F64Type, IntegerType, Type
from ..dialects import llvm

type Kind = Literal["int", "uint", "float", "bool", "ptr", "cstr"]


@dataclass(frozen=True)
class ScalarType:
    """A machine scalar type. The public instances are the names exported
    from this module; the compiler reads them from annotations."""

    name: str
    kind: Kind
    bits: int
    element: ScalarType | None = None  # what a typed pointer (Ptr[T]) points to

    def mlir(self) -> Type:
        """The MLIR type representing this type (in the current context)."""
        if self.kind in ("int", "uint"):
            return IntegerType(self.bits)
        if self.kind == "bool":
            return IntegerType(1)
        if self.kind == "float":
            return F32Type() if self.bits == 32 else F64Type()
        return llvm.PointerType()

    @property
    def is_integer(self) -> bool:
        return self.kind in ("int", "uint")

    @property
    def signed(self) -> bool:
        return self.kind == "int"

    def integer_range(self) -> tuple[int, int]:
        if self.kind == "bool":
            return 0, 1
        if self.kind == "uint":
            return 0, 2**self.bits - 1
        return -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1

    def __call__(self, value: object) -> object:
        """Convert a Python value (outside compiled code), checking range."""
        if self.kind == "float":
            return float(value)  # type: ignore[arg-type]
        if self.kind == "bool":
            return bool(value)
        if self.is_integer:
            number = int(value)  # type: ignore[call-overload]
            low, high = self.integer_range()
            if not low <= number <= high:
                raise OverflowError(f"{number} does not fit {self.name}")
            return number
        raise TypeError(f"{self.name} values only exist in compiled code")

    def __repr__(self) -> str:
        return self.name


i8 = ScalarType("i8", "int", 8)
i16 = ScalarType("i16", "int", 16)
i32 = ScalarType("i32", "int", 32)
i64 = ScalarType("i64", "int", 64)
u8 = ScalarType("u8", "uint", 8)
u16 = ScalarType("u16", "uint", 16)
u32 = ScalarType("u32", "uint", 32)
u64 = ScalarType("u64", "uint", 64)
f32 = ScalarType("f32", "float", 32)
f64 = ScalarType("f64", "float", 64)
boolean = ScalarType("bool", "bool", 1)
ptr = ScalarType("ptr", "ptr", 64)
"""An opaque pointer, as returned by C functions such as ``malloc``."""


class Ptr:
    """``Ptr[T]``: a pointer to ``T`` values, e.g. ``Ptr[i32]``. Index it to
    read and write (``p[0]``, ``p[i] = x``); ``Ptr[T](raw)`` types an opaque
    ``ptr``. At runtime ``Ptr[T]`` is a pointer ``ScalarType``."""

    def __class_getitem__(cls, element: object) -> ScalarType:
        from ._compiler import scalar_type

        kind = scalar_type(element)
        if kind is None:
            raise TypeError(f"Ptr[...] needs a scalar type, got {element!r}")
        return pointer_to(kind)


def pointer_to(element: ScalarType) -> ScalarType:
    """The type ``Ptr[element]``."""
    return ScalarType(f"Ptr[{element.name}]", "ptr", 64, element)


def stack(kind: object, count: int = 1) -> object:
    """``stack(T, count=1)``: memory for ``count`` values of type ``T`` on the
    stack of the compiled function, as a ``Ptr[T]`` (for example to pass to
    C's ``scanf``). It lives until the function returns."""
    del kind, count
    raise TypeError("stack() allocates memory in compiled code only")


cstr = ScalarType("cstr", "cstr", 64)
"""A pointer to a NUL-terminated C string; string literals convert to it."""

__all__ = [
    "Ptr",
    "ScalarType",
    "cstr",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "ptr",
    "stack",
    "u8",
    "u16",
    "u32",
    "u64",
]
