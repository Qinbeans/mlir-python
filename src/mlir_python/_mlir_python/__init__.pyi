"""Typed bindings over the MLIR C++ API."""

import enum
from collections.abc import Buffer, Callable, Iterator, Mapping, Sequence
from typing import Self

from . import arith as arith
from . import async_dialect as async_dialect
from . import bufferization as bufferization
from . import cf as cf
from . import emitc as emitc
from . import func as func
from . import irdl as irdl
from . import llvm as llvm
from . import math as math
from . import memref as memref
from . import scf as scf
from . import tensor as tensor
from . import vector as vector

class MLIRError(Exception):
    """
    Raised when MLIR reports an error (parsing, verification, a failing pass).

    The message lists every diagnostic MLIR emitted, one per line, as ``location: severity: message``.
    """

class Context:
    """
    Owns an MLIR context: the dialects it has loaded and the uniqued
    types, attributes, and locations created in it.

    Everything derived from a context keeps it alive. A context can be
    entered with ``with`` to make it the default for calls that take an
    optional ``context=`` argument::

        with Context():
            i32 = IntegerType(32)
    """

    def __init__(
        self,
        *,
        allow_unregistered_dialects: bool = False,
        multithreading: bool = True,
        load_all_dialects: bool = False,
    ) -> None:
        """
        Create a context.

        Args:
            allow_unregistered_dialects: Accept operations, types, and
                attributes from dialects this build does not know.
            multithreading: Let MLIR use a thread pool (e.g. for passes).
            load_all_dialects: Load every available dialect up front
                instead of on first use.
        """

    @property
    def allow_unregistered_dialects(self) -> bool:
        """Whether unknown dialects are accepted."""

    @allow_unregistered_dialects.setter
    def allow_unregistered_dialects(self, arg: bool, /) -> None: ...
    @property
    def multithreading(self) -> bool:
        """Whether MLIR may use its thread pool."""

    @multithreading.setter
    def multithreading(self, arg: bool, /) -> None: ...
    @property
    def loaded_dialects(self) -> list[str]:
        """Namespaces of the dialects currently loaded, sorted."""

    @property
    def available_dialects(self) -> list[str]:
        """Namespaces of every dialect that can be loaded, sorted."""

    def load_dialect(self, name: str) -> None:
        """
        Load the dialect with namespace ``name`` (e.g. ``"arith"``).

        Raises:
            ValueError: If no such dialect is available.
        """

    def load_all_available_dialects(self) -> None:
        """Load every dialect in ``available_dialects``."""

    def is_registered_operation(self, name: str) -> bool:
        """
        Whether ``name`` (e.g. ``"arith.addi"``) is a known operation.
        Loads the operation's dialect if needed.
        """

    @staticmethod
    def current() -> Context | None:
        """
        The context of the innermost ``with Context()``,
        ``with Location``, or ``with InsertionPoint`` block, if any.
        """

    def __enter__(self) -> Self:
        """Make this the default context inside a ``with`` block."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
        /,
    ) -> None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class Location:
    """
    A source location attached to operations and diagnostics.

    Enter a location with ``with`` to make it the default for operations
    created inside the block.
    """

    @staticmethod
    def unknown(*, context: Context | None = None) -> Location:
        """A location carrying no information (``loc(unknown)``)."""

    @staticmethod
    def file(
        filename: str, line: int, column: int, *, context: Context | None = None
    ) -> Location:
        """A ``filename:line:column`` location."""

    @staticmethod
    def name(
        name: str, child: Location | None = None, *, context: Context | None = None
    ) -> Location:
        """A named location, optionally wrapping a more precise ``child``."""

    @staticmethod
    def callsite(callee: Location, caller: Location) -> Location:
        """The location of ``callee`` as called from ``caller``."""

    @staticmethod
    def fused(
        locations: Sequence[Location],
        metadata: Attribute | None = None,
        *,
        context: Context | None = None,
    ) -> Location:
        """Several locations merged into one, with optional metadata."""

    @staticmethod
    def current() -> Location | None:
        """
        The default location for new operations: the innermost
        ``with Location`` block, else ``unknown`` in the current context,
        else ``None``.
        """

    @property
    def context(self) -> Context:
        """The context this location belongs to."""

    def __enter__(self) -> Self:
        """Make this the default location inside a ``with`` block."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
        /,
    ) -> None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Type:
    """
    Base class of every MLIR type.

    Types are immutable and uniqued in their context: two types are equal
    exactly when they print the same. Objects returned by the API are
    always the most specific subclass (e.g. ``IntegerType``), so narrow
    with ``isinstance``.
    """

    @staticmethod
    def parse(source: str, *, context: Context | None = None) -> Type:
        """
        Parse a type from its textual form, e.g. ``"tensor<?xf32>"``.

        Raises:
            MLIRError: If ``source`` is not a valid type.
        """

    @property
    def context(self) -> Context:
        """The context this type belongs to."""

    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Signedness(enum.Enum):
    """How an ``IntegerType`` interprets its bits."""

    SIGNLESS = 0
    """No signedness; operations decide (``i32``). The common case."""

    SIGNED = 1
    """Signed (``si32``)."""

    UNSIGNED = 2
    """Unsigned (``ui32``)."""

class IntegerType(Type):
    """A fixed-width integer type such as ``i32``, ``si8``, or ``ui64``."""

    def __init__(
        self,
        width: int,
        *,
        signedness: Signedness = Signedness.SIGNLESS,
        context: Context | None = None,
    ) -> None:
        """
        Create an integer type of ``width`` bits.

        Args:
            width: Bit width, 1 to 16777215.
            signedness: Signless by default.
            context: Context for the type; defaults to the current context.
        """

    @property
    def width(self) -> int:
        """Bit width."""

    @property
    def signedness(self) -> Signedness:
        """Signedness semantics."""

    @property
    def is_signless(self) -> bool:
        """Whether the type is signless (``iN``)."""

    @property
    def is_signed(self) -> bool:
        """Whether the type is signed (``siN``)."""

    @property
    def is_unsigned(self) -> bool:
        """Whether the type is unsigned (``uiN``)."""

class IndexType(Type):
    """The target-width integer type used for sizes and indices (``index``)."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``index``."""

class NoneType(Type):
    """The unit type ``none``."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``none``."""

class FloatType(Type):
    """
    Base class of the floating-point types. Construct a specific
    subclass such as ``F32Type``.
    """

    @property
    def width(self) -> int:
        """Bit width."""

    @property
    def mantissa_width(self) -> int:
        """Width of the significand, including the implicit bit."""

class BF16Type(FloatType):
    """The ``bf16`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``bf16``."""

class F16Type(FloatType):
    """The ``f16`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f16``."""

class TF32Type(FloatType):
    """The ``tf32`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``tf32``."""

class F32Type(FloatType):
    """The ``f32`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f32``."""

class F64Type(FloatType):
    """The ``f64`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f64``."""

class F80Type(FloatType):
    """The ``f80`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f80``."""

class F128Type(FloatType):
    """The ``f128`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f128``."""

class Float8E5M2Type(FloatType):
    """The ``f8E5M2`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f8E5M2``."""

class Float8E4M3Type(FloatType):
    """The ``f8E4M3`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f8E4M3``."""

class Float8E4M3FNType(FloatType):
    """The ``f8E4M3FN`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f8E4M3FN``."""

class Float8E5M2FNUZType(FloatType):
    """The ``f8E5M2FNUZ`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f8E5M2FNUZ``."""

class Float8E4M3FNUZType(FloatType):
    """The ``f8E4M3FNUZ`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f8E4M3FNUZ``."""

class Float8E4M3B11FNUZType(FloatType):
    """The ``f8E4M3B11FNUZ`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f8E4M3B11FNUZ``."""

class Float8E3M4Type(FloatType):
    """The ``f8E3M4`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f8E3M4``."""

class Float4E2M1FNType(FloatType):
    """The ``f4E2M1FN`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f4E2M1FN``."""

class Float6E2M3FNType(FloatType):
    """The ``f6E2M3FN`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f6E2M3FN``."""

class Float6E3M2FNType(FloatType):
    """The ``f6E3M2FN`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f6E3M2FN``."""

class Float8E8M0FNUType(FloatType):
    """The ``f8E8M0FNU`` floating-point type."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``f8E8M0FNU``."""

class ComplexType(Type):
    """A complex number type, e.g. ``complex<f32>``."""

    def __init__(self, element_type: Type) -> None:
        """
        Create ``complex<element_type>``. The element type must be an
        integer or floating-point type.
        """

    @property
    def element_type(self) -> Type:
        """Type of the real and imaginary parts."""

class FunctionType(Type):
    """A function signature, e.g. ``(i32, f32) -> i1``."""

    def __init__(
        self,
        inputs: Sequence[Type],
        results: Sequence[Type],
        *,
        context: Context | None = None,
    ) -> None:
        """Create ``(inputs) -> results``."""

    @property
    def inputs(self) -> list[Type]:
        """Argument types."""

    @property
    def results(self) -> list[Type]:
        """Result types."""

class TupleType(Type):
    """A fixed-size tuple of types, e.g. ``tuple<i32, f32>``."""

    def __init__(
        self, types: Sequence[Type], *, context: Context | None = None
    ) -> None:
        """Create ``tuple<types...>``."""

    @property
    def types(self) -> list[Type]:
        """Element types."""

    def __len__(self) -> int: ...

class OpaqueType(Type):
    """
    A type from an unregistered dialect, kept as uninterpreted text
    (``!dialect.data``).
    """

    def __init__(
        self, dialect: str, data: str, *, context: Context | None = None
    ) -> None:
        """
        Create ``!dialect.data``, e.g. ``OpaqueType("foo", "bar<1>")``
        is ``!foo.bar<1>``.
        """

    @property
    def dialect(self) -> str:
        """The dialect namespace."""

    @property
    def data(self) -> str:
        """The uninterpreted type body."""

class ShapedType(Type):
    """
    Base class of tensor, vector, and memref types.

    Shapes are lists of ``int | None``, where ``None`` is a dynamic
    dimension (``?`` in the textual form).
    """

    @property
    def element_type(self) -> Type:
        """The element type."""

    @property
    def has_rank(self) -> bool:
        """Whether the number of dimensions is known."""

    @property
    def rank(self) -> int:
        """
        Number of dimensions.

        Raises:
            ValueError: If unranked.
        """

    @property
    def shape(self) -> list[int | None]:
        """
        Dimension sizes, ``None`` for dynamic ones.

        Raises:
            ValueError: If unranked.
        """

    @property
    def has_static_shape(self) -> bool:
        """Whether the type is ranked and every dimension is static."""

    @property
    def num_elements(self) -> int | None:
        """Total element count, or ``None`` unless the shape is static."""

class RankedTensorType(ShapedType):
    """A tensor of known rank, e.g. ``tensor<4x?xf32>``."""

    def __init__(
        self,
        shape: Sequence[int | None],
        element_type: Type,
        encoding: Attribute | None = None,
    ) -> None:
        """
        Create ``tensor<shape x element_type>``.

        Args:
            shape: Dimension sizes; ``None`` marks a dynamic dimension.
            element_type: The element type.
            encoding: Optional encoding attribute (e.g. a sparse layout).
        """

    @property
    def encoding(self) -> Attribute | None:
        """The encoding attribute, if any."""

class UnrankedTensorType(ShapedType):
    """A tensor of unknown rank, e.g. ``tensor<*xf32>``."""

    def __init__(self, element_type: Type) -> None:
        """Create ``tensor<*xelement_type>``."""

class VectorType(ShapedType):
    """
    A fixed-shape SIMD vector, e.g. ``vector<4xf32>`` or the scalable
    ``vector<[4]xf32>``.
    """

    def __init__(
        self,
        shape: Sequence[int],
        element_type: Type,
        *,
        scalable: Sequence[bool] | None = None,
    ) -> None:
        """
        Create ``vector<shape x element_type>``.

        Args:
            shape: Static, positive dimension sizes.
            element_type: An integer, index, or float type.
            scalable: Per-dimension scalability flags (all ``False`` by
                default).
        """

    @property
    def scalable_dims(self) -> list[bool]:
        """Per-dimension scalability flags."""

    @property
    def is_scalable(self) -> bool:
        """Whether any dimension is scalable."""

class MemRefType(ShapedType):
    """A ranked reference to a region of memory, e.g. ``memref<4x?xf32>``."""

    def __init__(
        self,
        shape: Sequence[int | None],
        element_type: Type,
        *,
        layout: Attribute | None = None,
        memory_space: Attribute | None = None,
    ) -> None:
        """
        Create ``memref<shape x element_type, layout, memory_space>``.

        Args:
            shape: Dimension sizes; ``None`` marks a dynamic dimension.
            element_type: The element type.
            layout: A layout attribute such as an affine map or strided
                layout; identity when omitted.
            memory_space: Optional memory-space attribute.
        """

    @property
    def layout(self) -> Attribute:
        """The layout attribute."""

    @property
    def memory_space(self) -> Attribute | None:
        """The memory space, or ``None`` for the default one."""

class UnrankedMemRefType(ShapedType):
    """A memref of unknown rank, e.g. ``memref<*xf32>``."""

    def __init__(
        self, element_type: Type, *, memory_space: Attribute | None = None
    ) -> None:
        """Create ``memref<*xelement_type>``."""

    @property
    def memory_space(self) -> Attribute | None:
        """The memory space, or ``None`` for the default one."""

class Attribute:
    """
    Base class of every MLIR attribute: compile-time constant data such
    as numbers, strings, types, and arrays.

    Attributes are immutable and uniqued in their context. Objects
    returned by the API are always the most specific subclass.
    """

    @staticmethod
    def parse(source: str, *, context: Context | None = None) -> Attribute:
        """
        Parse an attribute from its textual form, e.g. ``"42 : i32"``.

        Raises:
            MLIRError: If ``source`` is not a valid attribute.
        """

    @property
    def context(self) -> Context:
        """The context this attribute belongs to."""

    @property
    def type(self) -> Type | None:
        """The attribute's type if it has one (e.g. ``i32`` for ``42 : i32``)."""

    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class IntegerAttr(Attribute):
    """An integer constant with an integer or index type."""

    def __init__(
        self,
        value: int,
        type: IntegerType | IndexType | None = None,
        *,
        context: Context | None = None,
    ) -> None:
        """
        Create ``value : type``.

        Args:
            value: Any Python int that fits in ``type``.
            type: An ``IntegerType`` or ``IndexType``; ``i64`` if omitted.
            context: Used only when ``type`` is omitted.

        Raises:
            OverflowError: If ``value`` does not fit.
        """

    @property
    def value(self) -> int:
        """The value. Signless integers are read as signed."""

class BoolAttr(IntegerAttr):
    """A boolean constant (``true`` / ``false``), typed ``i1``."""

    def __init__(self, value: bool, *, context: Context | None = None) -> None:
        """Create ``true`` or ``false``."""

    @property
    def value(self) -> bool:
        """The value."""

    def __bool__(self) -> bool: ...

class FloatAttr(Attribute):
    """A floating-point constant, e.g. ``1.5 : f32``."""

    def __init__(
        self,
        value: float,
        type: FloatType | None = None,
        *,
        context: Context | None = None,
    ) -> None:
        """
        Create ``value : type``, rounding ``value`` to ``type``.

        Args:
            value: The value.
            type: A ``FloatType``; ``f64`` if omitted.
            context: Used only when ``type`` is omitted.
        """

    @property
    def value(self) -> float:
        """The value, converted to a Python float."""

class StringAttr(Attribute):
    """A string constant, e.g. ``"hello"``."""

    def __init__(self, value: str, *, context: Context | None = None) -> None:
        """Create a string attribute."""

    @property
    def value(self) -> str:
        """The string."""

class UnitAttr(Attribute):
    """A marker attribute with no value; its presence is the information."""

    def __init__(self, *, context: Context | None = None) -> None:
        """Create ``unit``."""

class TypeAttr(Attribute):
    """An attribute holding a type."""

    def __init__(self, value: Type) -> None:
        """Wrap ``value`` as an attribute."""

    @property
    def value(self) -> Type:
        """The wrapped type."""

class ArrayAttr(Attribute):
    """
    An ordered list of attributes, e.g. ``[1 : i32, "x"]``. Supports
    ``len``, indexing, and iteration.
    """

    def __init__(
        self, elements: Sequence[Attribute], *, context: Context | None = None
    ) -> None:
        """
        Create an array. ``context`` is needed only when ``elements`` is
        empty and no context is active.
        """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Attribute: ...
    def __iter__(self) -> Iterator[Attribute]: ...
    @property
    def elements(self) -> list[Attribute]:
        """The elements as a list."""

class DictAttr(Attribute):
    """
    A string-keyed dictionary of attributes, sorted by key. Supports
    ``len``, ``in``, indexing by key, and iteration over keys.
    """

    def __init__(
        self, elements: Mapping[str, Attribute], *, context: Context | None = None
    ) -> None:
        """
        Create a dictionary. ``context`` is needed only when ``elements``
        is empty and no context is active.
        """

    def __len__(self) -> int: ...
    def __contains__(self, key: str) -> bool: ...
    def __getitem__(self, key: str) -> Attribute: ...
    def get(self, key: str) -> Attribute | None:
        """The value for ``key``, or ``None``."""

    def __iter__(self) -> Iterator[str]: ...
    def keys(self) -> list[str]:
        """The keys, sorted."""

    def items(self) -> list[tuple[str, Attribute]]:
        """``(key, value)`` pairs, sorted by key."""

class SymbolRefAttr(Attribute):
    """A reference to a symbol, possibly nested: ``@root::@a::@b``."""

    def __init__(
        self, root: str, nested: Sequence[str] = [], *, context: Context | None = None
    ) -> None:
        """Create ``@root::@nested[0]::...``."""

    @property
    def root(self) -> str:
        """The outermost symbol name."""

    @property
    def leaf(self) -> str:
        """The innermost symbol name."""

    @property
    def nested(self) -> list[str]:
        """Names after the root, outermost first."""

class FlatSymbolRefAttr(SymbolRefAttr):
    """A reference to a single symbol: ``@name``."""

    def __init__(self, name: str, *, context: Context | None = None) -> None:
        """Create ``@name``."""

    @property
    def value(self) -> str:
        """The symbol name."""

class DenseElementsAttr(Attribute):
    """
    A constant tensor or vector with every element stored, e.g.
    ``dense<[1, 2]> : tensor<2xi32>``. Use ``DenseIntElementsAttr`` or
    ``DenseFPElementsAttr`` to build one from Python numbers.
    """

    def __init__(self, elements: Sequence[Attribute], type: ShapedType) -> None:
        """
        Create from element attributes in row-major order.

        Args:
            elements: One attribute per element, or a single one to
                splat. Each must be typed with the element type.
            type: A statically shaped ``RankedTensorType`` or
                ``VectorType``.
        """

    @property
    def is_splat(self) -> bool:
        """Whether every element has the same value."""

    def __len__(self) -> int: ...
    @property
    def elements(self) -> list[Attribute]:
        """Every element as an attribute, row-major."""

class DenseIntElementsAttr(DenseElementsAttr):
    """Dense elements whose element type is an integer or index type."""

    def __init__(self, values: Sequence[int], type: ShapedType) -> None:
        """
        Create from Python ints in row-major order.

        Args:
            values: One int per element, or a single one to splat.
            type: A statically shaped tensor or vector of integers.

        Raises:
            OverflowError: If a value does not fit the element type.
        """

    @property
    def values(self) -> list[int]:
        """Every element as a Python int, row-major."""

class DenseFPElementsAttr(DenseElementsAttr):
    """Dense elements whose element type is a floating-point type."""

    def __init__(self, values: Sequence[float], type: ShapedType) -> None:
        """
        Create from Python floats in row-major order.

        Args:
            values: One float per element, or a single one to splat.
            type: A statically shaped tensor or vector of floats.
        """

    @property
    def values(self) -> list[float]:
        """Every element as a Python float, row-major."""

class DenseArrayAttr(Attribute):
    """
    Base class of the ``array<T: ...>`` attributes: compact 1-D arrays of
    a primitive type, often used for operation properties.
    """

    def __len__(self) -> int: ...

class DenseBoolArrayAttr(DenseArrayAttr):
    """An ``array<i1: ...>`` attribute."""

    def __init__(
        self, values: Sequence[bool], *, context: Context | None = None
    ) -> None:
        """Create from a sequence of bool values."""

    @property
    def values(self) -> list[bool]:
        """The values."""

    def __getitem__(self, index: int) -> bool: ...

class DenseI8ArrayAttr(DenseArrayAttr):
    """An ``array<i8: ...>`` attribute."""

    def __init__(
        self, values: Sequence[int], *, context: Context | None = None
    ) -> None:
        """Create from a sequence of int values."""

    @property
    def values(self) -> list[int]:
        """The values."""

    def __getitem__(self, index: int) -> int: ...

class DenseI16ArrayAttr(DenseArrayAttr):
    """An ``array<i16: ...>`` attribute."""

    def __init__(
        self, values: Sequence[int], *, context: Context | None = None
    ) -> None:
        """Create from a sequence of int values."""

    @property
    def values(self) -> list[int]:
        """The values."""

    def __getitem__(self, index: int) -> int: ...

class DenseI32ArrayAttr(DenseArrayAttr):
    """An ``array<i32: ...>`` attribute."""

    def __init__(
        self, values: Sequence[int], *, context: Context | None = None
    ) -> None:
        """Create from a sequence of int values."""

    @property
    def values(self) -> list[int]:
        """The values."""

    def __getitem__(self, index: int) -> int: ...

class DenseI64ArrayAttr(DenseArrayAttr):
    """An ``array<i64: ...>`` attribute."""

    def __init__(
        self, values: Sequence[int], *, context: Context | None = None
    ) -> None:
        """Create from a sequence of int values."""

    @property
    def values(self) -> list[int]:
        """The values."""

    def __getitem__(self, index: int) -> int: ...

class DenseF32ArrayAttr(DenseArrayAttr):
    """An ``array<f32: ...>`` attribute."""

    def __init__(
        self, values: Sequence[float], *, context: Context | None = None
    ) -> None:
        """Create from a sequence of float values."""

    @property
    def values(self) -> list[float]:
        """The values."""

    def __getitem__(self, index: int) -> float: ...

class DenseF64ArrayAttr(DenseArrayAttr):
    """An ``array<f64: ...>`` attribute."""

    def __init__(
        self, values: Sequence[float], *, context: Context | None = None
    ) -> None:
        """Create from a sequence of float values."""

    @property
    def values(self) -> list[float]:
        """The values."""

    def __getitem__(self, index: int) -> float: ...

class StridedLayoutAttr(Attribute):
    """
    A memref layout given by an offset and per-dimension strides, e.g.
    ``strided<[?, 1], offset: 4>``. ``None`` marks a dynamic value.
    """

    def __init__(
        self,
        offset: int | None,
        strides: Sequence[int | None],
        *,
        context: Context | None = None,
    ) -> None:
        """Create ``strided<strides, offset: offset>``."""

    @property
    def offset(self) -> int | None:
        """The offset, or ``None`` if dynamic."""

    @property
    def strides(self) -> list[int | None]:
        """Strides, ``None`` for dynamic ones."""

class WalkOrder(enum.Enum):
    """Traversal order for ``Operation.walk``."""

    PRE_ORDER = 0
    """Visit an operation before the operations nested in it."""

    POST_ORDER = 1
    """
    Visit an operation after the operations nested in it. Safe for
    erasing the visited operation.
    """

class WalkResult(enum.Enum):
    """What a ``walk`` callback wants to happen next."""

    ADVANCE = 0
    """Keep walking (same as ``None``)."""

    INTERRUPT = 1
    """Stop the walk."""

    SKIP = 2
    """Do not visit this operation's nested operations (pre-order)."""

class Value:
    """
    An SSA value: an operation result (``OpResult``) or a block argument
    (``BlockArgument``).
    """

    @property
    def type(self) -> Type:
        """
        The value's type. Assigning changes it in place without checks;
        verify afterwards.
        """

    @type.setter
    def type(self, arg: Type, /) -> None: ...
    @property
    def context(self) -> Context:
        """The context the value belongs to."""

    @property
    def location(self) -> Location:
        """Where the value is defined."""

    @property
    def defining_op(self) -> Operation | None:
        """
        The operation producing this value, or ``None`` for a block
        argument.
        """

    @property
    def owner(self) -> Operation | Block:
        """The defining operation or, for a block argument, its block."""

    @property
    def uses(self) -> list[OpOperand]:
        """Every operand slot that currently reads this value."""

    @property
    def has_uses(self) -> bool:
        """Whether anything reads this value."""

    def replace_all_uses_with(self, replacement: Value) -> None:
        """Make every user of this value read ``replacement`` instead."""

    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class OpResult(Value):
    """A value produced by an operation."""

    @property
    def owner(self) -> Operation:
        """The operation producing this result."""

    @property
    def result_number(self) -> int:
        """Position among the owner's results."""

class BlockArgument(Value):
    """A value passed into a block."""

    @property
    def owner(self) -> Block:
        """The block this argument belongs to."""

    @property
    def arg_number(self) -> int:
        """Position among the block's arguments."""

class OpOperand:
    """One use of a value: an operand slot of an operation."""

    @property
    def owner(self) -> Operation:
        """The operation this operand belongs to."""

    @property
    def operand_number(self) -> int:
        """Position among the owner's operands."""

    @property
    def value(self) -> Value:
        """The value currently used."""

class Operation:
    """
    A generic MLIR operation.

    An operation created without an insertion point is detached and owned
    by Python; inserting it into a block transfers ownership to that
    block's IR. Handles into IR keep the whole tree alive.

    ``erase()`` destroys an operation immediately; other handles to it or
    to IR nested in it must not be used afterwards.
    """

    @staticmethod
    def create(
        name: str,
        *,
        results: Sequence[Type] | None = None,
        operands: Sequence[Value] = [],
        attributes: Mapping[str, Attribute] | None = None,
        successors: Sequence[Block] = [],
        regions: int = 0,
        location: Location | None = None,
        ip: InsertionPoint | None = None,
        context: Context | None = None,
    ) -> Operation:
        """
        Create an operation by name.

        Args:
            name: Full name, e.g. ``"arith.addi"``. Its dialect is loaded
                on demand.
            results: Result types. When ``None``, they are inferred for
                operations that support it, and empty otherwise.
            operands: Operand values.
            attributes: Attributes by name, inherent or discardable.
            successors: Successor blocks (terminators only).
            regions: Number of empty regions to create.
            location: Defaults to the current ``Location``, else unknown.
            ip: Where to insert; defaults to the current
                ``InsertionPoint``. Without one the result is detached.
            context: Needed only if nothing else determines the context.

        Raises:
            ValueError: If the operation is unknown and the context does
                not allow unregistered dialects.
            MLIRError: If result types cannot be inferred.
        """

    @staticmethod
    def parse(source: str, *, context: Context | None = None) -> Operation:
        """
        Parse a single detached operation from its textual form.

        Raises:
            MLIRError: If ``source`` does not parse.
            ValueError: If it holds more than one top-level operation.
        """

    @property
    def name(self) -> str:
        """Full operation name, e.g. ``"func.func"``."""

    @property
    def is_registered(self) -> bool:
        """Whether the operation's definition is known to this build."""

    @property
    def context(self) -> Context:
        """The context the operation belongs to."""

    @property
    def location(self) -> Location:
        """Source location."""

    @location.setter
    def location(self, arg: Location, /) -> None: ...
    @property
    def attributes(self) -> AttributeMap:
        """Attributes by name (live view)."""

    @property
    def operands(self) -> OperandList:
        """Operand values (live view)."""

    @property
    def results(self) -> list[OpResult]:
        """Result values."""

    @property
    def result(self) -> OpResult:
        """
        The only result.

        Raises:
            ValueError: Unless there is exactly one result.
        """

    @property
    def regions(self) -> list[Region]:
        """Attached regions."""

    @property
    def successors(self) -> list[Block]:
        """Successor blocks of a terminator."""

    @property
    def parent(self) -> Operation | None:
        """The operation whose region contains this one, if any."""

    @property
    def block(self) -> Block | None:
        """The block containing this operation, or ``None`` if detached."""

    def verify(self) -> None:
        """
        Check the operation and everything nested in it.

        Raises:
            MLIRError: With the verifier's diagnostics.
        """

    def get_asm(
        self,
        *,
        generic: bool = False,
        debug_info: bool = False,
        large_elements_limit: int | None = None,
        use_local_scope: bool = False,
        assume_verified: bool = False,
        skip_regions: bool = False,
    ) -> str:
        """
        Print the operation to a string.

        Args:
            generic: Print the generic form ``"dialect.op"(...)``.
            debug_info: Include source locations.
            large_elements_limit: Elide elements attributes with more than
                this many elements.
            use_local_scope: Number SSA values as if this were the top level.
            assume_verified: Skip the verifier the printer otherwise runs
                (and falls back to the generic form on failure).
            skip_regions: Omit region bodies.
        """

    def clone(self) -> Operation:
        """A detached deep copy of this operation."""

    def detach_from_parent(self) -> Operation:
        """
        Remove the operation from its block and return a handle that owns
        it. Keep using the returned handle: older handles do not keep the
        detached operation alive.
        """

    def move_before(self, other: Operation) -> None:
        """Move this operation immediately before ``other``."""

    def move_after(self, other: Operation) -> None:
        """Move this operation immediately after ``other``."""

    def erase(self) -> None:
        """
        Destroy the operation and everything nested in it.

        Raises:
            ValueError: If a result (or a value nested inside) is still
                used elsewhere.
        """

    def walk(
        self,
        callback: Callable[[Operation], WalkResult | None],
        order: WalkOrder = WalkOrder.POST_ORDER,
    ) -> WalkResult:
        """
        Call ``callback`` on this operation and every nested one.

        Args:
            callback: Returns ``None`` (or ``WalkResult.ADVANCE``) to
                continue, ``INTERRUPT`` to stop, or ``SKIP`` (pre-order
                only) to skip nested operations. Exceptions propagate.
            order: Post-order by default.

        Returns:
            ``WalkResult.INTERRUPT`` if the walk stopped early, else
            ``WalkResult.ADVANCE``.
        """

    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class OperandList:
    """
    Live view of an operation's operands. Supports ``len``, indexing,
    assignment, and iteration.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Value: ...
    def __setitem__(self, index: int, value: Value) -> None: ...
    def __iter__(self) -> Iterator[Value]: ...

class AttributeMap:
    """
    Live view of an operation's attributes, including inherent ones
    stored as properties. Supports ``len``, ``in``, indexing, assignment,
    deletion, and iteration over names.
    """

    def __len__(self) -> int: ...
    def __contains__(self, name: str) -> bool: ...
    def __getitem__(self, name: str) -> Attribute: ...
    def __setitem__(self, name: str, attr: Attribute) -> None: ...
    def __delitem__(self, name: str) -> None: ...
    def get(self, name: str) -> Attribute | None:
        """The attribute called ``name``, or ``None``."""

    def __iter__(self) -> Iterator[str]: ...
    def keys(self) -> list[str]:
        """Attribute names, sorted."""

    def items(self) -> list[tuple[str, Attribute]]:
        """``(name, attribute)`` pairs, sorted by name."""

class Block:
    """A list of operations with typed arguments, inside a region."""

    @property
    def arguments(self) -> list[BlockArgument]:
        """The block's arguments."""

    def add_argument(
        self, type: Type, location: Location | None = None
    ) -> BlockArgument:
        """Append an argument of ``type`` and return it."""

    def erase_argument(self, index: int) -> None:
        """Remove an unused argument."""

    @property
    def operations(self) -> list[Operation]:
        """The operations, in order."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Operation]: ...
    @property
    def terminator(self) -> Operation | None:
        """The last operation if it is (or may be) a terminator."""

    @property
    def parent_op(self) -> Operation | None:
        """The operation owning this block's region."""

    @property
    def region(self) -> Region | None:
        """The region containing this block."""

    def append(self, op: Operation) -> None:
        """
        Move ``op`` to the end of this block. A detached operation becomes
        owned by this block's IR.
        """

    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...

class Region:
    """A list of blocks attached to an operation."""

    @property
    def blocks(self) -> list[Block]:
        """The blocks, in order. The first is the entry block."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Block]: ...
    @property
    def entry_block(self) -> Block | None:
        """The first block, or ``None`` if the region is empty."""

    def append_block(
        self,
        arg_types: Sequence[Type] = [],
        arg_locations: Sequence[Location] | None = None,
    ) -> Block:
        """
        Append a new block with the given argument types and return it.

        Args:
            arg_types: Argument types.
            arg_locations: One location per argument; the current
                location by default.
        """

    @property
    def parent_op(self) -> Operation:
        """The operation owning this region."""

    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class InsertionPoint:
    """
    A position in a block where new operations are inserted.

    Enter it with ``with`` to make ``Operation.create`` insert there by
    default.
    """

    def __init__(self, block: Block) -> None:
        """
        Insert at the end of ``block``, but before its terminator if it
        already has one (operations after a terminator are invalid).
        """

    @staticmethod
    def at_block_end(block: Block) -> InsertionPoint:
        """Insert after everything in ``block``, including a terminator."""

    @staticmethod
    def at_block_begin(block: Block) -> InsertionPoint:
        """Insert at the start of ``block``."""

    @staticmethod
    def at_block_terminator(block: Block) -> InsertionPoint:
        """
        Insert just before ``block``'s terminator.

        Raises:
            ValueError: If the block has no terminator.
        """

    @staticmethod
    def before(op: Operation) -> InsertionPoint:
        """Insert just before ``op``."""

    @staticmethod
    def after(op: Operation) -> InsertionPoint:
        """Insert just after ``op``."""

    @staticmethod
    def current() -> InsertionPoint | None:
        """The innermost ``with InsertionPoint`` block's point, if any."""

    @property
    def block(self) -> Block:
        """The block operations are inserted into."""

    @property
    def ref_operation(self) -> Operation | None:
        """The operation new ones go before, or ``None`` for the block end."""

    def insert(self, op: Operation) -> None:
        """
        Move ``op`` to this point. A detached operation becomes owned by
        this block's IR.
        """

    def __enter__(self) -> Self:
        """Make this the default insertion point inside a ``with`` block."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
        /,
    ) -> None: ...

class Module(Operation):
    """
    A ``builtin.module``: the usual top-level container of IR.

    A module is an ``Operation``; navigating to a ``builtin.module``
    always produces a ``Module``.
    """

    def __init__(
        self, location: Location | None = None, *, context: Context | None = None
    ) -> None:
        """Create an empty module."""

    OPERATION_NAME: str = "builtin.module"

    @staticmethod
    def parse(source: str, *, context: Context | None = None) -> Module:
        """
        Parse a module from text. Top-level operations that are not a
        module are wrapped in one.

        Raises:
            MLIRError: With the parser's diagnostics.
        """

    @staticmethod
    def parse_file(path: str, *, context: Context | None = None) -> Module:
        """
        Parse a module from a ``.mlir`` file (text or bytecode).

        Raises:
            MLIRError: If the file cannot be read or parsed.
        """

    @property
    def body(self) -> Block:
        """The module's single block."""

def register_python_pass(argument: str, description: str, run: Callable) -> None:
    """
    Low level: register a pass named ``argument`` that calls ``run`` with
    each operation it runs on (see ``passes.PythonPass``).
    """

def release_python_passes() -> None:
    """
    Low level: release the callables of registered Python passes (run
    at interpreter exit).
    """

class ParsedPassPipeline:
    """
    A pass pipeline parsed from MLIR's textual syntax, ready to run.

    Low level: build pipelines with ``mlir_python.PassManager`` and the
    typed passes in ``mlir_python.passes``, which produce this object.
    """

    @staticmethod
    def parse(pipeline: str, *, context: Context | None = None) -> ParsedPassPipeline:
        """
        Parse an anchored pipeline such as
        ``"builtin.module(canonicalize, cse)"``.

        Raises:
            ValueError: If the pipeline does not parse.
        """

    def enable_verifier(self, enabled: bool = True) -> None:
        """Verify the IR after each pass (on by default)."""

    def run(self, op: Operation) -> None:
        """
        Run the pipeline on ``op`` in place.

        Raises:
            MLIRError: If a pass fails or the result does not verify.
        """

    @property
    def anchor(self) -> str:
        """Name of the operations this manager runs on."""

    @property
    def context(self) -> Context:
        """The context passes run in."""

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class OptLevel(enum.Enum):
    """LLVM optimization level."""

    O0 = 0
    """No optimization."""

    O1 = 1
    """Light optimization."""

    O2 = 2
    """Default optimization."""

    O3 = 3
    """Aggressive optimization."""

class LLVMModule:
    """
    An LLVM IR module translated from LLVM-dialect MLIR, targeting the
    host (its triple and data layout are set).
    """

    def __str__(self) -> str: ...
    def verify(self) -> None:
        """
        Run LLVM's IR verifier.

        Raises:
            MLIRError: With the verifier's findings.
        """

    @property
    def defined_functions(self) -> list[str]:
        """Names of the functions with a body, in module order."""

    @property
    def target_triple(self) -> str:
        """The target triple, e.g. ``x86_64-unknown-linux-gnu``."""

    @property
    def data_layout(self) -> str:
        """The data layout string."""

    def optimize(self, level: OptLevel = OptLevel.O2) -> None:
        """Run LLVM's optimization pipeline in place."""

    def write_object(
        self, path: str, *, level: OptLevel = OptLevel.O2, host_cpu: bool = False
    ) -> None:
        """
        Write a position-independent object file for the host, linkable
        into executables and shared libraries.

        Args:
            path: Output file.
            level: Code generation optimization level.
            host_cpu: Use every feature of this machine's CPU; the object
                may then not run on other machines.
        """

    def assembly(self, *, level: OptLevel = OptLevel.O2, host_cpu: bool = False) -> str:
        """The host assembly for this module."""

def translate_to_llvm_ir(module: Operation) -> LLVMModule:
    """
    Translate an MLIR module in the LLVM dialect to LLVM IR.

    Raises:
        MLIRError: If the module still contains non-LLVM operations.
    """

def translate_to_cpp(
    module: Operation, *, declare_variables_at_top: bool = False
) -> str:
    """
    Translate an MLIR module in the EmitC dialect to C or C++ source.

    With ``declare_variables_at_top``, each function declares all its
    variables before its first statement (needed for C89 and for
    functions with more than one block).

    Raises:
        MLIRError: If the module contains operations EmitC cannot print.
    """

class ExecutionEngine:
    """
    JIT-compiles an MLIR module in the LLVM dialect into this process.

    Prefer ``codegen.compile``, which lowers a copy of a module and calls
    functions by their ``func.FuncOp``.
    """

    def __init__(
        self,
        module: Operation,
        *,
        opt_level: OptLevel = OptLevel.O2,
        shared_libraries: Sequence[str] = [],
    ) -> None:
        """
        JIT-compile ``module``.

        Args:
            module: A module in the LLVM dialect.
            opt_level: LLVM optimization level.
            shared_libraries: Paths of shared libraries to load for external
                symbols (``codegen.compile`` also takes names and archives).

        Raises:
            MLIRError: If translation or compilation fails.
        """

    def call(
        self,
        name: str,
        signature: FunctionType,
        args: Sequence[int | float | bool | Buffer],
        /,
        *,
        owned_results: bool = False,
    ) -> (
        int
        | float
        | bool
        | memoryview
        | tuple[int | float | bool | memoryview, ...]
        | None
    ):
        """
        Low level: call function ``name`` of type ``signature`` with
        ``args``: Python numbers for scalars, writable buffers (such as
        NumPy arrays) for memrefs, which the function reads and writes in
        place. Returns ``None``, one value, or a tuple of values; a memref
        result is copied into a new ``memoryview`` shaped like it. With
        ``owned_results``, memref results are buffers the function
        allocated for the caller, freed once copied.

        Raises:
            TypeError: For a wrong argument count, argument types, or
                types that cannot cross the Python boundary.
            OverflowError: If an int does not fit its integer type.
            MLIRError: If no such function exists.
        """
