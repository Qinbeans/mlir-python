# ruff: noqa: UP040, PYI042
# - UP040: these must be old-style aliases; a PEP 695 `type` alias is not
#   callable, and `i32(x)` conversions must type-check.
# - PYI042: `i32`/`f64` are the conventional names (as in MLIR and NumPy).
# What type checkers see: the scalar types are aliases of Python's own, so
# function bodies are ordinary Python (`a + b`, `return 0`, `i32(x)`). At
# runtime they are `ScalarType` objects that the compiler reads from
# annotations (see types.py).

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, dataclass_transform

from .._mlir_python import Type

type Kind = Literal[
    "int", "uint", "float", "bool", "ptr", "cstr", "struct", "array", "fn"
]

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
    @property
    def size(self) -> int: ...
    @property
    def alignment(self) -> int: ...

@dataclass(frozen=True)
class StructType(ScalarType):
    fields: tuple[tuple[str, ScalarType], ...] = ()
    python: type | None = None
    def offsets(self) -> list[int]: ...
    def field(self, name: str) -> tuple[int, ScalarType] | None: ...

@dataclass(frozen=True)
class FnType(ScalarType):
    params: tuple[ScalarType, ...] = ()
    result: ScalarType | None = None

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
    """An opaque pointer, as returned by C functions such as ``malloc``.
    ``ptr(xs)`` is the address of an array's first item, to pass to C."""

    def __init__(self, address: ptr | list[Any], /) -> None: ...

type Array[T] = list[T]
"""``Array[T]``: an array of ``T`` values (``Array[i32]``), created with a
list display (``[1, 2, 3]``) or ``array(i32, n)``. Index it like a list
(``xs[i]``, ``xs[-1]``, ``xs[i] = v``; out-of-range indices stop the
program), take ``len(xs)``, iterate it, and pass and return it; it is freed
automatically. To type checkers it is ``list[T]``, so Python lists pass in
and come back out (buffers such as NumPy arrays are also accepted at runtime,
without copying). Pass it to C as ``ptr(xs)`` or ``Ptr[T](xs)``."""

def array[T](kind: type[T], length: int, /) -> list[T]:
    """A new array of ``length`` zeros of type ``kind``: ``array(f64, n)``."""

@dataclass(frozen=True)
class ArrayType(ScalarType):
    def mlir(self) -> Type: ...

class Fn[**P, R](ptr):
    """``Fn[[P1, P2], R]``: a function value (a C function pointer), e.g. a
    compiled or extern function passed as an argument. Call it like a
    function; ``Fn[[P1], R](raw)`` types an opaque ``ptr``. Any ``Fn`` passes
    where a ``ptr`` is expected."""

    def __init__(self, address: ptr, /) -> None: ...
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

class Ptr[T](ptr):
    """A pointer to ``T`` values, e.g. ``Ptr[i32]``. Index it to read and
    write (``p[0]``, ``p[i] = x``); ``p + n`` and ``p - n`` point ``n`` values
    further or back, as in C; ``Ptr[T](raw)`` types an opaque ``ptr``.
    Any ``Ptr[T]`` passes where a ``ptr`` is expected."""

    def __init__(self, address: ptr | Ptr[Any] | list[T], /) -> None: ...
    def __getitem__(self, index: int) -> T: ...
    def __add__(self, count: int, /) -> Ptr[T]: ...
    def __sub__(self, count: int, /) -> Ptr[T]: ...
    def __setitem__(self, index: int, value: T) -> None: ...

def atomic_add[T: int](pointer: Ptr[T], delta: T, /) -> T:
    """Add ``delta`` to the integer ``pointer`` points at as one indivisible
    step (sequentially consistent, safe across threads); return the value it
    held before, e.g. for a reference count."""

def stack[T](kind: type[T], count: int = 1) -> Ptr[T]:
    """Memory for ``count`` values of type ``kind`` on the compiled function's
    stack, e.g. ``value = stack(i32)`` to pass to C's ``scanf``. It lives until
    the function returns."""

@dataclass_transform()
def struct[T](cls: type[T]) -> type[T]:
    """Declare a C struct: a class whose annotated fields are machine types,
    in C's order::

        @struct
        class Color:
            r: u8
            g: u8
            b: u8
            a: u8

    Compiled code constructs it (``Color(245, 245, 245, 255)``, or with field
    names), reads and assigns fields (``c.r``, ``c.r = 0``), passes it to and
    returns it from functions (by value, following the C ABI for externs),
    and points at it (``Ptr[Color]``, ``stack(Color)``). The class is also an
    ordinary dataclass in Python."""
