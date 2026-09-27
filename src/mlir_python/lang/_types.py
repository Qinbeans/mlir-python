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

import dataclasses
import inspect
from dataclasses import dataclass
from typing import Literal

from .._mlir_python import F32Type, F64Type, IntegerType, MemRefType, Type
from ..dialects import llvm

type Kind = Literal[
    "int", "uint", "float", "bool", "ptr", "cstr", "struct", "array", "fn"
]


@dataclass(frozen=True)
class ScalarType:
    """A machine type: a scalar, or a ``StructType``. The public instances are
    the names exported from this module; the compiler reads them from
    annotations."""

    name: str
    kind: Kind
    bits: int
    element: ScalarType | None = None  # what a typed pointer (Ptr[T]) points to

    @property
    def size(self) -> int:
        """Size in bytes, as C lays it out (``sizeof``)."""
        return 1 if self.kind == "bool" else self.bits // 8

    @property
    def alignment(self) -> int:
        """Alignment in bytes, as C lays it out (``_Alignof``)."""
        return self.size

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
    read and write (``p[0]``, ``p[i] = x``); ``p + n`` and ``p - n`` point
    ``n`` values further or back, as in C; ``Ptr[T](raw)`` types an opaque
    ``ptr``. At runtime ``Ptr[T]`` is a pointer ``ScalarType``."""

    def __class_getitem__(cls, element: object) -> ScalarType:
        kind = scalar_type(element)
        if kind is None:
            raise TypeError(f"Ptr[...] needs a scalar type, got {element!r}")
        return pointer_to(kind)


def pointer_to(element: ScalarType) -> ScalarType:
    """The type ``Ptr[element]``."""
    return ScalarType(f"Ptr[{element.name}]", "ptr", 64, element)


@dataclass(frozen=True, repr=False)
class ArrayType(ScalarType):
    """``Array[T]``: a heap array of ``T`` values with a length, like a
    Python list of fixed size. Compiled as a one-dimensional ``memref``;
    freed automatically once nothing uses it."""

    def mlir(self) -> Type:
        assert self.element is not None
        return MemRefType([None], self.element.mlir())

    def __call__(self, value: object) -> object:
        raise TypeError(f"create a {self.name} with [...] or array(T, length)")


ARRAY_ELEMENT_KINDS = ("int", "uint", "float", "bool")


class Array:
    """``Array[T]``: an array of ``T`` values (``Array[i32]``), created with
    a list display (``[1, 2, 3]``) or ``array(i32, n)``. Index it like a
    list (``xs[i]``, ``xs[-1]``, ``xs[i] = v``; out-of-range indices stop the
    program), take ``len(xs)``, iterate it (``for x in xs``), and pass and
    return it. From Python, pass a list (or a buffer such as a NumPy array),
    and get a list back. To type checkers it is ``list[T]``."""

    def __class_getitem__(cls, element: object) -> ScalarType:
        kind = scalar_type(element)
        if kind is None or kind.kind not in ARRAY_ELEMENT_KINDS:
            raise TypeError(
                f"Array[...] holds integers, floats, or bools, got {element!r}"
            )
        return array_of(kind)


def array_of(element: ScalarType) -> ArrayType:
    """The type ``Array[element]``."""
    return ArrayType(f"Array[{element.name}]", "array", 64, element)


def array(kind: object, length: int) -> list[object]:
    """``array(T, length)``: a new array of ``length`` zeros of type ``T``
    (in plain Python, a list of zeros)."""
    element = scalar_type(kind)
    if element is None or element.kind not in ARRAY_ELEMENT_KINDS:
        raise TypeError(f"array() holds integers, floats, or bools, got {kind!r}")
    return [element(0)] * length


def stack(kind: object, count: int = 1) -> object:
    """``stack(T, count=1)``: memory for ``count`` values of type ``T`` on the
    stack of the compiled function, as a ``Ptr[T]`` (for example to pass to
    C's ``scanf``). It lives until the function returns."""
    del kind, count
    raise TypeError("stack() allocates memory in compiled code only")


def atomic_add(pointer: object, delta: object) -> object:
    """``atomic_add(p, delta)``: add ``delta`` to the integer ``p`` points at
    as one indivisible step (sequentially consistent, safe across threads),
    and return the value it held before, e.g. for a reference count."""
    del pointer, delta
    raise TypeError("atomic_add() works on memory in compiled code only")


@dataclass(frozen=True)
class FnType(ScalarType):
    """A pointer to a function taking ``params`` and returning ``result``
    (``None`` for nothing): ``Fn[[i32, i32], i32]``. It is a C function
    pointer, so compiled functions can be passed to C and C callbacks can
    be called."""

    params: tuple[ScalarType, ...] = ()
    result: ScalarType | None = None

    def __call__(self, value: object) -> object:
        raise TypeError(f"{self.name} values only exist in compiled code")


class Fn:
    """``Fn[[P1, P2, ...], R]``: a function value, e.g. ``Fn[[i32], i32]``
    or ``Fn[[ptr], None]``. Name a function where a value is expected to
    take its address (``apply(double, 21)``); call a value like a function
    (``f(x)``). At runtime ``Fn[...]`` is an ``FnType``."""

    def __class_getitem__(cls, item: object) -> FnType:
        if not (
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], list)
        ):
            raise TypeError(
                "write Fn[[parameter types], result type], e.g. Fn[[i32], i32]"
            )
        params_given, result_given = item
        params = [scalar_type(p) for p in params_given]
        if any(p is None for p in params):
            raise TypeError(
                f"Fn[...] parameters must be machine types, got {params_given!r}"
            )
        no_result = result_given is None or result_given is type(None)
        result = None if no_result else scalar_type(result_given)
        if result is None and not no_result:
            raise TypeError(
                f"Fn[...] result must be a machine type or None, got {result_given!r}"
            )
        return function_type([p for p in params if p is not None], result)


def function_type(params: list[ScalarType], result: ScalarType | None) -> FnType:
    """The type ``Fn[params, result]``."""
    name = f"Fn[[{', '.join(p.name for p in params)}], {result.name if result else 'None'}]"
    return FnType(name, "fn", 64, None, tuple(params), result)


cstr = ScalarType("cstr", "cstr", 64)
"""A pointer to a NUL-terminated C string; string literals convert to it."""

BUILTIN_TYPES: dict[object, ScalarType] = {
    int: i64,
    float: f64,
    bool: boolean,
    str: cstr,
}


def scalar_type(annotation: object) -> ScalarType | None:
    """The machine type an annotation names: a ``ScalarType``, one of
    Python's ``int``/``float``/``bool``/``str``, or a ``@struct`` class."""
    if isinstance(annotation, ScalarType):
        return annotation
    if isinstance(annotation, type):
        declared = annotation.__dict__.get("__lang_struct__")
        if isinstance(declared, StructType):
            return declared
    return BUILTIN_TYPES.get(annotation)


@dataclass(frozen=True)
class StructType(ScalarType):
    """A C struct declared with ``@struct``: its fields in order, laid out
    as C lays them out (each field at its natural alignment)."""

    fields: tuple[tuple[str, ScalarType], ...] = ()
    python: type | None = dataclasses.field(default=None, compare=False)

    @property
    def size(self) -> int:
        end = 0
        for _, kind in self.fields:
            end = _align(end, kind.alignment) + kind.size
        return _align(end, self.alignment)

    @property
    def alignment(self) -> int:
        return max((kind.alignment for _, kind in self.fields), default=1)

    def offsets(self) -> list[int]:
        """Each field's byte offset."""
        offsets, end = [], 0
        for _, kind in self.fields:
            start = _align(end, kind.alignment)
            offsets.append(start)
            end = start + kind.size
        return offsets

    def field(self, name: str) -> tuple[int, ScalarType] | None:
        """The index and type of field ``name``."""
        for index, (field_name, kind) in enumerate(self.fields):
            if field_name == name:
                return index, kind
        return None

    def mlir(self) -> Type:
        # LLVM lays out a non-packed struct as C does for these field types.
        return llvm.StructType([kind.mlir() for _, kind in self.fields])

    def __call__(self, value: object) -> object:
        raise TypeError(f"construct {self.name} with {self.name}(...)")


def _align(offset: int, alignment: int) -> int:
    return -(-offset // alignment) * alignment


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
    ordinary dataclass in Python.

    Raises:
        TypeError: If a field is not a machine type or another ``@struct``.
    """
    annotations = inspect.get_annotations(cls, eval_str=True)
    if not annotations:
        raise TypeError(f"@struct class {cls.__name__} declares no fields")
    fields: list[tuple[str, ScalarType]] = []
    for name, annotation in annotations.items():
        kind = scalar_type(annotation)
        if kind is None:
            raise TypeError(
                f"field '{name}' of {cls.__name__} has type {annotation!r}; "
                "use a type such as i32, f32, ptr, or another @struct"
            )
        if kind.kind == "array":
            raise TypeError(
                f"field '{name}' of {cls.__name__} is an array; arrays cannot be "
                "stored in structs yet (their memory is freed automatically, which "
                "a struct field cannot track)"
            )
        fields.append((name, kind))
    declared = dataclasses.dataclass(cls)
    kind = StructType(cls.__name__, "struct", 0, None, tuple(fields), declared)
    kind = dataclasses.replace(kind, bits=kind.size * 8)
    declared.__lang_struct__ = kind  # type: ignore[attr-defined]
    return declared


__all__ = [
    "Array",
    "ArrayType",
    "Fn",
    "FnType",
    "Ptr",
    "ScalarType",
    "StructType",
    "array",
    "atomic_add",
    "cstr",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "ptr",
    "stack",
    "struct",
    "u8",
    "u16",
    "u32",
    "u64",
]
