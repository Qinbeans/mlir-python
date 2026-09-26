# ruff: noqa: UP040, PYI042
# - UP040: these must be old-style aliases; a PEP 695 `type` alias is not
#   callable, and `i32(x)` conversions must type-check.
# - PYI042: `i32`/`f64` are the conventional names (as in MLIR and NumPy).
# What type checkers see: the scalar types are aliases of Python's own, so
# function bodies are ordinary Python (`a + b`, `return 0`, `i32(x)`). At
# runtime they are `ScalarType` objects that the compiler reads from
# annotations (see types.py).

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from .._mlir_python import Type

type Kind = Literal["int", "uint", "float", "bool", "ptr", "cstr"]

@dataclass(frozen=True)
class ScalarType:
    name: str
    kind: Kind
    bits: int
    element: ScalarType | None = None
    def mlir(self) -> Type: ...
    @property
    def is_integer(self) -> bool: ...
    @property
    def signed(self) -> bool: ...
    def integer_range(self) -> tuple[int, int]: ...

i8: TypeAlias = int
i16: TypeAlias = int
i32: TypeAlias = int
i64: TypeAlias = int
u8: TypeAlias = int
u16: TypeAlias = int
u32: TypeAlias = int
u64: TypeAlias = int
f32: TypeAlias = float
f64: TypeAlias = float
cstr: TypeAlias = str

class ptr:
    """An opaque pointer, as returned by C functions such as ``malloc``."""

class Ptr[T](ptr):
    """A pointer to ``T`` values, e.g. ``Ptr[i32]``. Index it to read and
    write (``p[0]``, ``p[i] = x``); ``Ptr[T](raw)`` types an opaque ``ptr``.
    Any ``Ptr[T]`` passes where a ``ptr`` is expected."""

    def __init__(self, address: ptr | Ptr[Any]) -> None: ...
    def __getitem__(self, index: int) -> T: ...
    def __setitem__(self, index: int, value: T) -> None: ...

def stack[T](kind: type[T], count: int = 1) -> Ptr[T]:
    """Memory for ``count`` values of type ``kind`` on the compiled function's
    stack, e.g. ``value = stack(i32)`` to pass to C's ``scanf``. It lives until
    the function returns."""
