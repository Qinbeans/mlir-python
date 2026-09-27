"""Typed operations of the MLIR ``memref`` dialect."""

from collections.abc import Sequence

import mlir_python._mlir_python
import mlir_python._mlir_python.arith

class AssumeAlignmentOp(mlir_python._mlir_python.Operation):
    """
    ``memref.assume_alignment``: assumption that gives alignment information to the input memref.

    The `assume_alignment` operation takes a memref and an integer alignment
    value. It returns a new SSA value of the same memref type, but associated
    with the assumption that the underlying buffer is aligned to the given
    alignment.

    If the buffer isn't aligned to the given alignment, its result is poison.
    This operation doesn't affect the semantics of a program where the
    alignment assumption holds true. It is intended for optimization purposes,
    allowing the compiler to generate more efficient code based on the
    alignment assumption. The optimization is best-effort.
    """

    def __init__(
        self,
        memref: mlir_python._mlir_python.Value,
        alignment: int,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.assume_alignment``: assumption that gives alignment information to the input memref.

        Result types are inferred.

        Args:
            memref: Operand ``memref`` (memref of any type values).
            alignment: Attribute ``alignment`` (32-bit signless integer attribute whose value is positive).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def memref(self) -> mlir_python._mlir_python.Value:
        """Operand ``memref``: memref of any type values."""

    @property
    def alignment(self) -> int:
        """
        Attribute ``alignment``: 32-bit signless integer attribute whose value is positive.
        """

    @alignment.setter
    def alignment(self, arg: int, /) -> None: ...

    OPERATION_NAME: str = "memref.assume_alignment"

class AtomicRMWOp(mlir_python._mlir_python.Operation):
    """
    ``memref.atomic_rmw``: atomic read-modify-write operation.

    The `memref.atomic_rmw` operation provides a way to perform a read-modify-write
    sequence that is free from data races. The kind enumeration specifies the
    modification to perform. The value operand represents the new value to be
    applied during the modification. The memref operand represents the buffer
    that the read and write will be performed against, as accessed by the
    specified indices. The arity of the indices is the rank of the memref. The
    result represents the latest value that was stored.

    Example:

    ```mlir
    %x = memref.atomic_rmw "addf" %value, %I[%i] : (f32, memref<10xf32>) -> f32
    ```
    """

    def __init__(
        self,
        kind: mlir_python._mlir_python.arith.AtomicRMWKind,
        value: mlir_python._mlir_python.Value,
        memref: mlir_python._mlir_python.Value,
        indices: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.atomic_rmw``: atomic read-modify-write operation.

        Result types are inferred.

        Args:
            kind: Attribute ``kind`` (a member of ``AtomicRMWKind``).
            value: Operand ``value`` (signless integer or floating-point).
            memref: Operand ``memref`` (the reference to read from and write to).
            indices: Operand ``indices`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: signless integer or floating-point."""

    @property
    def memref(self) -> mlir_python._mlir_python.Value:
        """Operand ``memref``: the reference to read from and write to."""

    @property
    def indices(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``indices``: index."""

    @property
    def kind(self) -> mlir_python._mlir_python.arith.AtomicRMWKind:
        """Attribute ``kind``: a member of ``AtomicRMWKind``."""

    @kind.setter
    def kind(self, arg: mlir_python._mlir_python.arith.AtomicRMWKind, /) -> None: ...

    OPERATION_NAME: str = "memref.atomic_rmw"

class AtomicYieldOp(mlir_python._mlir_python.Operation):
    """
    ``memref.atomic_yield``: yield operation for GenericAtomicRMWOp.

    "memref.atomic_yield" yields an SSA value from a
    GenericAtomicRMWOp region.
    """

    def __init__(
        self,
        result: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.atomic_yield``: yield operation for GenericAtomicRMWOp.

        Args:
            result: Operand ``result`` (any type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "memref.atomic_yield"

class CopyOp(mlir_python._mlir_python.Operation):
    """
    ``memref.copy``.

    Copies the data from the source to the destination memref.

    Usage:

    ```mlir
    memref.copy %arg0, %arg1 : memref<?xf32> to memref<?xf32>
    ```

    Source and destination are expected to have the same element type and shape.
    Otherwise, the result is undefined. They may have different layouts.
    """

    def __init__(
        self,
        source: mlir_python._mlir_python.Value,
        target: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.copy``.

        Args:
            source: Operand ``source`` (the memref to copy from).
            target: Operand ``target`` (the memref to copy to).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: the memref to copy from."""

    @property
    def target(self) -> mlir_python._mlir_python.Value:
        """Operand ``target``: the memref to copy to."""

    OPERATION_NAME: str = "memref.copy"

class DistinctObjectsOp(mlir_python._mlir_python.Operation):
    """
    ``memref.distinct_objects``: assumption that acesses to specific memrefs will never alias.

    The `distinct_objects` operation takes a list of memrefs and returns the same
    memrefs, with the additional assumption that accesses to them will never
    alias with each other. This means that loads and stores to different
    memrefs in the list can be safely reordered.

    If the memrefs do alias, the load/store behavior is undefined. This
    operation doesn't affect the semantics of a valid program. It is
    intended for optimization purposes, allowing the compiler to generate more
    efficient code based on the non-aliasing assumption. The optimization is
    best-effort.

    Example:

    ```mlir
    %1, %2 = memref.distinct_objects %a, %b : memref<?xf32>, memref<?xf32>
    ```
    """

    def __init__(
        self,
        operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.distinct_objects``: assumption that acesses to specific memrefs will never alias.

        Result types are inferred.

        Args:
            operands: Operand ``operands`` (memref of any type values). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "memref.distinct_objects"

class GenericAtomicRMWOp(mlir_python._mlir_python.Operation):
    """
    ``memref.generic_atomic_rmw``: atomic read-modify-write operation with a region.

    The `memref.generic_atomic_rmw` operation provides a way to perform a
    read-modify-write sequence that is free from data races. The memref operand
    represents the buffer that the read and write will be performed against, as
    accessed by the specified indices. The arity of the indices is the rank of
    the memref. The result represents the latest value that was stored. The
    region contains the code for the modification itself. The entry block has
    a single argument that represents the value stored in `memref[indices]`
    before the write is performed. No side-effecting ops are allowed in the
    body of `GenericAtomicRMWOp`.

    Example:

    ```mlir
    %x = memref.generic_atomic_rmw %I[%i] : memref<10xf32> {
      ^bb0(%current_value : f32):
        %c1 = arith.constant 1.0 : f32
        %inc = arith.addf %c1, %current_value : f32
        memref.atomic_yield %inc : f32
    }
    ```
    """

    def __init__(
        self,
        memref: mlir_python._mlir_python.Value,
        indices: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.generic_atomic_rmw``: atomic read-modify-write operation with a region.

        Result types are inferred.

        Args:
            memref: Operand ``memref`` (the reference to read from and write to).
            indices: Operand ``indices`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def memref(self) -> mlir_python._mlir_python.Value:
        """Operand ``memref``: the reference to read from and write to."""

    @property
    def indices(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``indices``: index."""

    @property
    def atomic_body(self) -> mlir_python._mlir_python.Region:
        """Region ``atomic_body``: any region."""

    OPERATION_NAME: str = "memref.generic_atomic_rmw"

class LoadOp(mlir_python._mlir_python.Operation):
    """
    ``memref.load``: load operation.

    The `load` op reads an element from a memref at the specified indices.

    The number of indices must match the rank of the memref. The indices must
    be in-bounds: `0 <= idx < dim_size`.

    Lowerings of `memref.load` may emit attributes, e.g. `inbouds` + `nuw`
    when converting to LLVM's `llvm.getelementptr`, that would cause undefined
    behavior if indices are out of bounds or if computing the offset in the
    memref would cause signed overflow of the `index` type.

    The single result of `memref.load` is a value with the same type as the
    element type of the memref.

    A set `nontemporal` attribute indicates that this load is not expected to
    be reused in the cache. For details, refer to the
    [LLVM load instruction](https://llvm.org/docs/LangRef.html#load-instruction).

    An optional `alignment` attribute allows to specify the byte alignment of the
    load operation. It must be a positive power of 2. The operation must access
    memory at an address aligned to this boundary. Violations may lead to
    architecture-specific faults or performance penalties.
    A value of 0 indicates no specific alignment requirement.
    Example:

    ```mlir
    %0 = memref.load %A[%a, %b] : memref<8x?xi32, #layout, memspace0>
    ```
    """

    def __init__(
        self,
        memref: mlir_python._mlir_python.Value,
        indices: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        nontemporal: bool = False,
        alignment: int | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.load``: load operation.

        Result types are inferred.

        Args:
            memref: Operand ``memref`` (the reference to load from).
            indices: Operand ``indices`` (index). Empty by default.
            nontemporal: Attribute ``nontemporal`` (bool attribute). Omit for the default.
            alignment: Attribute ``alignment`` (64-bit signless integer attribute whose value is positive and whose value is a power of two > 0). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def memref(self) -> mlir_python._mlir_python.Value:
        """Operand ``memref``: the reference to load from."""

    @property
    def indices(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``indices``: index."""

    @property
    def nontemporal(self) -> bool:
        """Attribute ``nontemporal``: bool attribute."""

    @nontemporal.setter
    def nontemporal(self, arg: bool, /) -> None: ...
    @property
    def alignment(self) -> int | None:
        """
        Attribute ``alignment``: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0.
        """

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...

    OPERATION_NAME: str = "memref.load"

class AllocOp(mlir_python._mlir_python.Operation):
    """
    ``memref.alloc``: memory allocation operation.

    The `alloc` operation allocates a region of memory, as specified by its
    memref type.

    Example:

    ```mlir
    %0 = memref.alloc() : memref<8x64xf32, 1>
    ```

    The optional list of dimension operands are bound to the dynamic dimensions
    specified in its memref type. In the example below, the ssa value '%d' is
    bound to the second dimension of the memref (which is dynamic).

    ```mlir
    %0 = memref.alloc(%d) : memref<8x?xf32, 1>
    ```

    The optional list of symbol operands are bound to the symbols of the
    memrefs affine map. In the example below, the ssa value '%s' is bound to
    the symbol 's0' in the affine map specified in the allocs memref type.

    ```mlir
    %0 = memref.alloc()[%s] : memref<8x64xf32,
                              affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
    ```

    This operation returns a single ssa value of memref type, which can be used
    by subsequent load and store operations.

    The optional `alignment` attribute may be specified to ensure that the
    region of memory that will be indexed is aligned at the specified byte
    boundary.

    ```mlir
    %0 = memref.alloc()[%s] {alignment = 8} :
      memref<8x64xf32, affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>, 1>
    ```
    """

    def __init__(
        self,
        memref_type: mlir_python._mlir_python.Type,
        dynamic_sizes: Sequence[mlir_python._mlir_python.Value] = [],
        symbol_operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        alignment: int | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.alloc``: memory allocation operation.

        Args:
            memref_type: Type of result ``memref`` (memref of any type values).
            dynamic_sizes: Operand ``dynamicSizes`` (index). Empty by default.
            symbol_operands: Operand ``symbolOperands`` (index). Empty by default.
            alignment: Attribute ``alignment`` (64-bit signless integer attribute whose value is positive and whose value is a power of two > 0). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def dynamic_sizes(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``dynamicSizes``: index."""

    @property
    def symbol_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``symbolOperands``: index."""

    @property
    def memref(self) -> mlir_python._mlir_python.OpResult:
        """Result ``memref``: memref of any type values."""

    @property
    def alignment(self) -> int | None:
        """
        Attribute ``alignment``: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0.
        """

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...

    OPERATION_NAME: str = "memref.alloc"

class AllocaOp(mlir_python._mlir_python.Operation):
    """
    ``memref.alloca``: stack memory allocation operation.

    The `alloca` operation allocates memory on the stack, to be automatically
    released when control transfers back from the region of its closest
    surrounding operation with an
    [`AutomaticAllocationScope`](../Traits/#automaticallocationscope) trait.
    The amount of memory allocated is specified by its memref and additional
    operands. For example:

    ```mlir
    %0 = memref.alloca() : memref<8x64xf32>
    ```

    The optional list of dimension operands are bound to the dynamic dimensions
    specified in its memref type. In the example below, the SSA value '%d' is
    bound to the second dimension of the memref (which is dynamic).

    ```mlir
    %0 = memref.alloca(%d) : memref<8x?xf32>
    ```

    The optional list of symbol operands are bound to the symbols of the
    memref's affine map. In the example below, the SSA value '%s' is bound to
    the symbol 's0' in the affine map specified in the allocs memref type.

    ```mlir
    %0 = memref.alloca()[%s] : memref<8x64xf32,
                               affine_map<(d0, d1)[s0] -> ((d0 + s0), d1)>>
    ```

    This operation returns a single SSA value of memref type, which can be used
    by subsequent load and store operations. An optional alignment attribute, if
    specified, guarantees alignment at least to that boundary. If not specified,
    an alignment on any convenient boundary compatible with the type will be
    chosen.
    """

    def __init__(
        self,
        memref_type: mlir_python._mlir_python.Type,
        dynamic_sizes: Sequence[mlir_python._mlir_python.Value] = [],
        symbol_operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        alignment: int | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.alloca``: stack memory allocation operation.

        Args:
            memref_type: Type of result ``memref`` (memref of any type values).
            dynamic_sizes: Operand ``dynamicSizes`` (index). Empty by default.
            symbol_operands: Operand ``symbolOperands`` (index). Empty by default.
            alignment: Attribute ``alignment`` (64-bit signless integer attribute whose value is positive and whose value is a power of two > 0). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def dynamic_sizes(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``dynamicSizes``: index."""

    @property
    def symbol_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``symbolOperands``: index."""

    @property
    def memref(self) -> mlir_python._mlir_python.OpResult:
        """Result ``memref``: memref of any type values."""

    @property
    def alignment(self) -> int | None:
        """
        Attribute ``alignment``: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0.
        """

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...

    OPERATION_NAME: str = "memref.alloca"

class AllocaScopeOp(mlir_python._mlir_python.Operation):
    """
    ``memref.alloca_scope``: explicitly delimited scope for stack allocation.

    The `memref.alloca_scope` operation represents an explicitly-delimited
    scope for the alloca allocations. Any `memref.alloca` operations that are
    used within this scope are going to be cleaned up automatically once
    the control-flow exits the nested region. For example:

    ```mlir
    memref.alloca_scope {
      %myalloca = memref.alloca(): memref<4x3xf32>
      ...
    }
    ```

    Here, `%myalloca` memref is valid within the explicitly delimited scope
    and is automatically deallocated at the end of the given region. Conceptually,
    `memref.alloca_scope` is a passthrough operation with
    `AutomaticAllocationScope` that spans the body of the region within the operation.

    `memref.alloca_scope` may also return results that are defined in the nested
    region. To return a value, one should use `memref.alloca_scope.return`
    operation:

    ```mlir
    %result = memref.alloca_scope -> f32 {
      %value = arith.constant 1.0 : f32
      ...
      memref.alloca_scope.return %value : f32
    }
    ```

    If `memref.alloca_scope` returns no value, the `memref.alloca_scope.return ` can
    be left out, and will be inserted implicitly.
    """

    def __init__(
        self,
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.alloca_scope``: explicitly delimited scope for stack allocation.

        Args:
            result_types: Type of result ``results`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def body_region(self) -> mlir_python._mlir_python.Region:
        """Region ``bodyRegion``: region with 1 blocks."""

    OPERATION_NAME: str = "memref.alloca_scope"

class AllocaScopeReturnOp(mlir_python._mlir_python.Operation):
    """
    ``memref.alloca_scope.return``: terminator for alloca_scope operation.

    `memref.alloca_scope.return` operation returns zero or more SSA values
    from the region within `memref.alloca_scope`. If no values are returned,
    the return operation may be omitted. Otherwise, it has to be present
    to indicate which values are going to be returned. For example:

    ```mlir
    memref.alloca_scope.return %value : f32
    ```
    """

    def __init__(
        self,
        results: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.alloca_scope.return``: terminator for alloca_scope operation.

        Args:
            results: Operand ``results`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "memref.alloca_scope.return"

class CastOp(mlir_python._mlir_python.Operation):
    """
    ``memref.cast``: memref cast operation.

    The `memref.cast` operation converts a memref from one type to an equivalent
    type with a compatible shape. The source and destination types are
    compatible if:

    a. Both are ranked memref types with the same element type, address space,
    and rank and:
      1. Both have the same layout or both have compatible strided layouts.
      2. The individual sizes (resp. offset and strides in the case of strided
         memrefs) may convert constant dimensions to dynamic dimensions and
         vice-versa.

    If the cast converts any dimensions from an unknown to a known size, then it
    acts as an assertion that fails at runtime if the dynamic dimensions
    disagree with resultant destination size.

    Example:

    ```mlir
    // Assert that the input dynamic shape matches the destination static shape.
    %2 = memref.cast %1 : memref<?x?xf32> to memref<4x4xf32>
    // Erase static shape information, replacing it with dynamic information.
    %3 = memref.cast %1 : memref<4xf32> to memref<?xf32>

    // The same holds true for offsets and strides.

    // Assert that the input dynamic shape matches the destination static stride.
    %4 = memref.cast %1 : memref<12x4xf32, strided<[?, ?], offset: ?>> to
                          memref<12x4xf32, strided<[4, 1], offset: 5>>
    // Erase static offset and stride information, replacing it with
    // dynamic information.
    %5 = memref.cast %1 : memref<12x4xf32, strided<[4, 1], offset: 5>> to
                          memref<12x4xf32, strided<[?, ?], offset: ?>>
    ```

    b. Either or both memref types are unranked with the same element type, and
    address space.

    Example:

    ```mlir
    // Cast to concrete shape.
    %4 = memref.cast %1 : memref<*xf32> to memref<4x?xf32>

    // Erase rank information.
    %5 = memref.cast %1 : memref<4x?xf32> to memref<*xf32>
    ```
    """

    def __init__(
        self,
        dest_type: mlir_python._mlir_python.Type,
        source: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.cast``: memref cast operation.

        Args:
            dest_type: Type of result ``dest`` (ranked or unranked memref of any type values).
            source: Operand ``source`` (ranked or unranked memref of any type values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: ranked or unranked memref of any type values."""

    @property
    def dest(self) -> mlir_python._mlir_python.OpResult:
        """Result ``dest``: ranked or unranked memref of any type values."""

    OPERATION_NAME: str = "memref.cast"

class CollapseShapeOp(mlir_python._mlir_python.Operation):
    """
    ``memref.collapse_shape``: operation to produce a memref with a smaller rank..

    The `memref.collapse_shape` op produces a new view with a smaller rank
    whose sizes are a reassociation of the original `view`. The operation is
    limited to such reassociations, where subsequent, contiguous dimensions are
    collapsed into a single dimension. Such reassociations never require
    additional allocs or copies.

    Collapsing non-contiguous dimensions is undefined behavior. When a group of
    dimensions can be statically proven to be non-contiguous, collapses of such
    groups are rejected in the verifier on a best-effort basis. In the general
    case, collapses of dynamically-sized dims with dynamic strides cannot be
    proven to be contiguous or non-contiguous due to limitations in the memref
    type.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an array of DenseI64ArrayAttr attribute.

    Note: Only the dimensions within a reassociation group must be contiguous.
    The remaining dimensions may be non-contiguous.

    The result memref type can be zero-ranked if the source memref type is
    statically shaped with all dimensions being unit extent. In such a case, the
    reassociation indices must be empty.

    Examples:

    ```mlir
    // Dimension collapse (i, j) -> i' and k -> k'
    %1 = memref.collapse_shape %0 [[0, 1], [2]] :
        memref<?x?x?xf32, stride_spec> into memref<?x?xf32, stride_spec_2>
    ```

    For simplicity, this op may not be used to cast dynamicity of dimension
    sizes and/or strides. I.e., a result dimension must be dynamic if and only
    if at least one dimension in the corresponding reassociation group is
    dynamic. Similarly, the stride of a result dimension must be dynamic if and
    only if the corresponding start dimension in the source type is dynamic.

    Note: This op currently assumes that the inner strides are of the
    source/result layout map are the faster-varying ones.
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        src: mlir_python._mlir_python.Value,
        reassociation: mlir_python._mlir_python.ArrayAttr,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.collapse_shape``: operation to produce a memref with a smaller rank..

        Args:
            result_type: Type of result ``result`` (memref of any type values).
            src: Operand ``src`` (memref of any type values).
            reassociation: Attribute ``reassociation`` (Array of 64-bit integer array attributes).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def src(self) -> mlir_python._mlir_python.Value:
        """Operand ``src``: memref of any type values."""

    @property
    def reassociation(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``reassociation``: Array of 64-bit integer array attributes."""

    @reassociation.setter
    def reassociation(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...

    OPERATION_NAME: str = "memref.collapse_shape"

class DeallocOp(mlir_python._mlir_python.Operation):
    """
    ``memref.dealloc``: memory deallocation operation.

    The `dealloc` operation frees the region of memory referenced by a memref
    which was originally created by the `alloc` operation.
    The `dealloc` operation should not be called on memrefs which alias an
    alloc'd memref (e.g. memrefs returned by `view` operations).

    Example:

    ```mlir
    %0 = memref.alloc() : memref<8x64xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>
    memref.dealloc %0 : memref<8x64xf32,  affine_map<(d0, d1) -> (d0, d1)>, 1>
    ```
    """

    def __init__(
        self,
        memref: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.dealloc``: memory deallocation operation.

        Args:
            memref: Operand ``memref`` (ranked or unranked memref of any type values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def memref(self) -> mlir_python._mlir_python.Value:
        """Operand ``memref``: ranked or unranked memref of any type values."""

    OPERATION_NAME: str = "memref.dealloc"

class DimOp(mlir_python._mlir_python.Operation):
    """
    ``memref.dim``: dimension index operation.

    The `dim` operation takes a memref and a dimension operand of type `index`.
    It returns the size of the requested dimension of the given memref.
    If the dimension index is out of bounds the behavior is undefined.

    The specified memref type is that of the first operand.

    Example:

    ```mlir
    // Always returns 4, can be constant folded:
    %c0 = arith.constant 0 : index
    %x = memref.dim %A, %c0 : memref<4 x ? x f32>

    // Returns the dynamic dimension of %A.
    %c1 = arith.constant 1 : index
    %y = memref.dim %A, %c1 : memref<4 x ? x f32>

    // Equivalent generic form:
    %x = "memref.dim"(%A, %c0) : (memref<4 x ? x f32>, index) -> index
    %y = "memref.dim"(%A, %c1) : (memref<4 x ? x f32>, index) -> index
    ```
    """

    def __init__(
        self,
        source: mlir_python._mlir_python.Value,
        index: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.dim``: dimension index operation.

        Result types are inferred.

        Args:
            source: Operand ``source`` (unranked.memref of any type values or non-0-ranked.memref of any type values).
            index: Operand ``index`` (index).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``source``: unranked.memref of any type values or non-0-ranked.memref of any type values.
        """

    @property
    def index(self) -> mlir_python._mlir_python.Value:
        """Operand ``index``: index."""

    OPERATION_NAME: str = "memref.dim"

class DmaStartOp(mlir_python._mlir_python.Operation):
    """
    ``memref.dma_start``: non-blocking DMA operation that starts a transfer.

    Syntax:

    ```
    operation ::= `memref.dma_start` ssa-use`[`ssa-use-list`]` `,`
                   ssa-use`[`ssa-use-list`]` `,` ssa-use `,`
                   ssa-use`[`ssa-use-list`]` (`,` ssa-use `,` ssa-use)?
                  `:` memref-type `,` memref-type `,` memref-type
    ```

    DmaStartOp starts a non-blocking DMA operation that transfers data from a
    source memref to a destination memref. The source and destination memref
    need not be of the same dimensionality, but need to have the same elemental
    type. The operands include the source and destination memref's each followed
    by its indices, size of the data transfer in terms of the number of elements
    (of the elemental type of the memref), a tag memref with its indices, and
    optionally at the end, a stride and a number_of_elements_per_stride
    arguments. The tag location is used by a DmaWaitOp to check for completion.
    The indices of the source memref, destination memref, and the tag memref
    have the same restrictions as any load/store. The optional stride arguments
    should be of 'index' type, and specify a stride for the slower memory space
    (memory space with a lower memory space id), transferring chunks of
    number_of_elements_per_stride every stride until %num_elements are
    transferred. Either both or no stride arguments should be specified. If the
    source and destination locations overlap the behavior of this operation is
    not defined.

    For example, a DmaStartOp operation that transfers 256 elements of a memref
    '%src' in memory space 0 at indices [%i, %j] to memref '%dst' in memory
    space 1 at indices [%k, %l], would be specified as follows:

    ```mlir
    %num_elements = arith.constant 256 : index
    %idx = arith.constant 0 : index
    %tag = memref.alloc() : memref<1 x i32, affine_map<(d0) -> (d0)>, 2>
    memref.dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx] :
      memref<40 x 128 x f32, affine_map<(d0, d1) -> (d0, d1)>, 0>,
      memref<2 x 1024 x f32, affine_map<(d0, d1) -> (d0, d1)>, 1>,
      memref<1 x i32, affine_map<(d0) -> (d0)>, 2>
    ```

    If %stride and %num_elt_per_stride are specified, the DMA is expected to
    transfer %num_elt_per_stride elements every %stride elements apart from
    memory space 0 until %num_elements are transferred.

    ```mlir
    memref.dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx], %stride,
                     %num_elt_per_stride :
    ```

    * TODO: add additional operands to allow source and destination striding, and
    multiple stride levels.
    * TODO: Consider replacing src/dst memref indices with view memrefs.
    """

    def __init__(
        self,
        operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.dma_start``: non-blocking DMA operation that starts a transfer.

        Args:
            operands: Operand ``operands`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "memref.dma_start"

class DmaWaitOp(mlir_python._mlir_python.Operation):
    """
    ``memref.dma_wait``: blocking DMA operation that waits for transfer completion.

    DmaWaitOp blocks until the completion of a DMA operation associated with the
    tag element '%tag[%index]'. %tag is a memref, and %index has to be an index
    with the same restrictions as any load/store index. %num_elements is the
    number of elements associated with the DMA operation.

    Example:

    ```mlir
     memref.dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%index] :
       memref<2048 x f32, affine_map<(d0) -> (d0)>, 0>,
       memref<256 x f32, affine_map<(d0) -> (d0)>, 1>,
       memref<1 x i32, affine_map<(d0) -> (d0)>, 2>
     ...
     ...
     dma_wait %tag[%index], %num_elements : memref<1 x i32, affine_map<(d0) -> (d0)>, 2>
     ```
    """

    def __init__(
        self,
        tag_mem_ref: mlir_python._mlir_python.Value,
        num_elements: mlir_python._mlir_python.Value,
        tag_indices: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.dma_wait``: blocking DMA operation that waits for transfer completion.

        Args:
            tag_mem_ref: Operand ``tagMemRef`` (memref of any type values).
            num_elements: Operand ``numElements`` (index).
            tag_indices: Operand ``tagIndices`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def tag_mem_ref(self) -> mlir_python._mlir_python.Value:
        """Operand ``tagMemRef``: memref of any type values."""

    @property
    def tag_indices(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``tagIndices``: index."""

    @property
    def num_elements(self) -> mlir_python._mlir_python.Value:
        """Operand ``numElements``: index."""

    OPERATION_NAME: str = "memref.dma_wait"

class ExpandShapeOp(mlir_python._mlir_python.Operation):
    """
    ``memref.expand_shape``: operation to produce a memref with a higher rank..

    The `memref.expand_shape` op produces a new view with a higher rank whose
    sizes are a reassociation of the original `view`. The operation is limited
    to such reassociations, where a dimension is expanded into one or multiple
    contiguous dimensions. Such reassociations never require additional allocs
    or copies.

    A reassociation is defined as a grouping of dimensions and is represented
    with an array of DenseI64ArrayAttr attributes.

    Example:

    ```mlir
    %r = memref.expand_shape %0 [[0, 1], [2]] output_shape [%sz0, %sz1, 32]
        : memref<?x32xf32> into memref<?x?x32xf32>
    ```

    If an op can be statically proven to be invalid (e.g, an expansion from
    `memref<10xf32>` to `memref<2x6xf32>`), it is rejected by the verifier. If
    it cannot statically be proven invalid (e.g., the full example above; it is
    unclear whether the first source dimension is divisible by 5), the op is
    accepted by the verifier. However, if the op is in fact invalid at runtime,
    the behavior is undefined.

    The source memref can be zero-ranked. In that case, the reassociation
    indices must be empty and the result shape may only consist of unit
    dimensions.

    For simplicity, this op may not be used to cast dynamicity of dimension
    sizes and/or strides. I.e., if and only if a source dimension is dynamic,
    there must be a dynamic result dimension in the corresponding reassociation
    group. Same for strides.

    The representation for the output shape supports a partially-static
    specification via attributes specified through the `static_output_shape`
    argument.  A special sentinel value `ShapedType::kDynamic` encodes that the
    corresponding entry has a dynamic value.  There must be exactly as many SSA
    inputs in `output_shape` as there are `ShapedType::kDynamic` entries in
    `static_output_shape`.

    Note: This op currently assumes that the inner strides are of the
    source/result layout map are the faster-varying ones.
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        src: mlir_python._mlir_python.Value,
        reassociation: mlir_python._mlir_python.ArrayAttr,
        static_output_shape: mlir_python._mlir_python.Attribute,
        output_shape: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.expand_shape``: operation to produce a memref with a higher rank..

        Args:
            result_type: Type of result ``result`` (memref of any type values).
            src: Operand ``src`` (memref of any type values).
            reassociation: Attribute ``reassociation`` (Array of 64-bit integer array attributes).
            static_output_shape: Attribute ``static_output_shape`` (i64 dense array attribute).
            output_shape: Operand ``output_shape`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def src(self) -> mlir_python._mlir_python.Value:
        """Operand ``src``: memref of any type values."""

    @property
    def output_shape(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``output_shape``: index."""

    @property
    def reassociation(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``reassociation``: Array of 64-bit integer array attributes."""

    @reassociation.setter
    def reassociation(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...
    @property
    def static_output_shape(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``static_output_shape``: i64 dense array attribute."""

    @static_output_shape.setter
    def static_output_shape(
        self, arg: mlir_python._mlir_python.Attribute, /
    ) -> None: ...

    OPERATION_NAME: str = "memref.expand_shape"

class ExtractAlignedPointerAsIndexOp(mlir_python._mlir_python.Operation):
    """
    ``memref.extract_aligned_pointer_as_index``: Extracts a memref's underlying aligned pointer as an index.

    Extracts the underlying aligned pointer as an index.

    This operation is useful for lowering to lower-level dialects while still
    avoiding the need to define a pointer type in higher-level dialects such as
    the memref dialect.

    This operation is intended solely as step during lowering, it has no side
    effects. A reverse operation that creates a memref from an index interpreted
    as a pointer is explicitly discouraged.

    Example:

    ```
      %0 = memref.extract_aligned_pointer_as_index %arg : memref<4x4xf32> -> index
      %1 = arith.index_cast %0 : index to i64
      %2 = llvm.inttoptr %1 : i64 to !llvm.ptr
      call @foo(%2) : (!llvm.ptr) ->()
    ```
    """

    def __init__(
        self,
        source: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.extract_aligned_pointer_as_index``: Extracts a memref's underlying aligned pointer as an index.

        Result types are inferred.

        Args:
            source: Operand ``source`` (ranked or unranked memref of any type values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: ranked or unranked memref of any type values."""

    @property
    def aligned_pointer(self) -> mlir_python._mlir_python.OpResult:
        """Result ``aligned_pointer``: index."""

    OPERATION_NAME: str = "memref.extract_aligned_pointer_as_index"

class ExtractStridedMetadataOp(mlir_python._mlir_python.Operation):
    """
    ``memref.extract_strided_metadata``: Extracts a buffer base with offset and strides.

    Extracts a base buffer, offset and strides. This op allows additional layers
    of transformations and foldings to be added as lowering progresses from
    higher-level dialect to lower-level dialects such as the LLVM dialect.

    The op requires a strided memref source operand. If the source operand is not
    a strided memref, then verification fails.

    This operation is also useful for completeness to the existing memref.dim op.
    While accessing strides, offsets and the base pointer independently is not
    available, this is useful for composing with its natural complement op:
    `memref.reinterpret_cast`.

    Intended Use Cases:

    The main use case is to expose the logic for manipulate memref metadata at a
    higher level than the LLVM dialect.
    This makes lowering more progressive and brings the following benefits:
      - not all users of MLIR want to lower to LLVM and the information to e.g.
        lower to library calls---like libxsmm---or to SPIR-V was not available.
      - foldings and canonicalizations can happen at a higher level in MLIR:
        before this op existed, lowering to LLVM would create large amounts of
        LLVMIR. Even when LLVM does a good job at folding the low-level IR from
        a performance perspective, it is unnecessarily opaque and inefficient to
        send unkempt IR to LLVM.

    Example:

    ```mlir
      %base, %offset, %sizes:2, %strides:2 =
        memref.extract_strided_metadata %memref : memref<10x?xf32>
          -> memref<f32>, index, index, index, index, index

      // After folding, the type of %m2 can be memref<10x?xf32> and further
      // folded to %memref.
      %m2 = memref.reinterpret_cast %base to
          offset: [%offset],
          sizes: [%sizes#0, %sizes#1],
          strides: [%strides#0, %strides#1]
        : memref<f32> to memref<?x?xf32, strided<[?, ?], offset:?>>
    ```
    """

    def __init__(
        self,
        source: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.extract_strided_metadata``: Extracts a buffer base with offset and strides.

        Result types are inferred.

        Args:
            source: Operand ``source`` (memref of any type values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: memref of any type values."""

    @property
    def base_buffer(self) -> mlir_python._mlir_python.OpResult:
        """Result ``base_buffer``: strided memref of any type values of rank 0."""

    @property
    def offset(self) -> mlir_python._mlir_python.OpResult:
        """Result ``offset``: index."""

    @property
    def sizes(self) -> list[mlir_python._mlir_python.OpResult]:
        """Result ``sizes``: index."""

    @property
    def strides(self) -> list[mlir_python._mlir_python.OpResult]:
        """Result ``strides``: index."""

    OPERATION_NAME: str = "memref.extract_strided_metadata"

class GetGlobalOp(mlir_python._mlir_python.Operation):
    """
    ``memref.get_global``: get the memref pointing to a global variable.

    The `memref.get_global` operation retrieves the memref pointing to a
    named global variable. If the global variable is marked constant, writing
    to the result memref (such as through a `memref.store` operation) is
    undefined.

    Example:

    ```mlir
    %x = memref.get_global @foo : memref<2xf32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        name: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.get_global``: get the memref pointing to a global variable.

        Args:
            result_type: Type of result ``result`` (memref of any type values).
            name: Attribute ``name`` (flat symbol reference attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "memref.get_global"

class GlobalOp(mlir_python._mlir_python.Operation):
    """
    ``memref.global``: declare or define a global memref variable.

    The `memref.global` operation declares or defines a named global memref
    variable. The backing memory for the variable is allocated statically and is
    described by the type of the variable (which should be a statically shaped
    memref type). The operation is a declaration if no `initial_value` is
    specified, else it is a definition. The `initial_value` can either be a unit
    attribute to represent a definition of an uninitialized global variable, or
    an elements attribute to represent the definition of a global variable with
    an initial value. The global variable can also be marked constant using the
    `constant` unit attribute. Writing to such constant global variables is
    undefined.

    The global variable can be accessed by using the `memref.get_global` to
    retrieve the memref for the global variable. Note that the memref
    for such global variable itself is immutable (i.e., memref.get_global for a
    given global variable will always return the same memref descriptor).

    Example:

    ```mlir
    // Private variable with an initial value.
    memref.global "private" @x : memref<2xf32> = dense<[0.0, 2.0]>

    // Private variable with an initial value and an alignment (power of 2).
    memref.global "private" @x : memref<2xf32> = dense<[0.0, 2.0]> {alignment = 64}

    // Declaration of an external variable.
    memref.global "private" @y : memref<4xi32>

    // Uninitialized externally visible variable.
    memref.global @z : memref<3xf16> = uninitialized

    // Externally visible constant variable.
    memref.global constant @c : memref<2xi32> = dense<[1, 4]>
    ```
    """

    def __init__(
        self,
        sym_name: str,
        type: mlir_python._mlir_python.MemRefType,
        *,
        sym_visibility: str | None = None,
        initial_value: mlir_python._mlir_python.Attribute | None = None,
        constant: bool = False,
        alignment: int | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.global``: declare or define a global memref variable.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            type: Attribute ``type`` (memref type attribute).
            sym_visibility: Attribute ``sym_visibility`` (string attribute). Optional.
            initial_value: Attribute ``initial_value`` (any attribute). Optional.
            constant: Attribute ``constant`` (unit attribute). Omit for the default.
            alignment: Attribute ``alignment`` (64-bit signless integer attribute whose value is positive and whose value is a power of two > 0). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def sym_visibility(self) -> str | None:
        """Attribute ``sym_visibility``: string attribute."""

    @sym_visibility.setter
    def sym_visibility(self, arg: str | None) -> None: ...
    @property
    def type(self) -> mlir_python._mlir_python.MemRefType:
        """Attribute ``type``: memref type attribute."""

    @type.setter
    def type(self, arg: mlir_python._mlir_python.MemRefType, /) -> None: ...
    @property
    def initial_value(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``initial_value``: any attribute."""

    @initial_value.setter
    def initial_value(self, arg: mlir_python._mlir_python.Attribute | None) -> None: ...
    @property
    def constant(self) -> bool:
        """Attribute ``constant``: unit attribute."""

    @constant.setter
    def constant(self, arg: bool, /) -> None: ...
    @property
    def alignment(self) -> int | None:
        """
        Attribute ``alignment``: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0.
        """

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...

    OPERATION_NAME: str = "memref.global"

class MemorySpaceCastOp(mlir_python._mlir_python.Operation):
    """
    ``memref.memory_space_cast``: memref memory space cast operation.

    This operation casts memref values between memory spaces.
    The input and result will be memrefs of the same types and shape that alias
    the same underlying memory, though, for some casts on some targets,
    the underlying values of the pointer stored in the memref may be affected
    by the cast.

    The input and result must have the same shape, element type, rank, and layout.

    If the source and target address spaces are the same, this operation is a noop.

    Finally, if the target memory-space is the generic/default memory-space,
    then it is assumed this cast can be bubbled down safely. See the docs of
    `MemorySpaceCastOpInterface` interface for more details.

    Example:

    ```mlir
    // Cast a GPU private memory attribution into a generic pointer
    %2 = memref.memory_space_cast %1 : memref<?xf32, 5> to memref<?xf32>
    // Cast a generic pointer to workgroup-local memory
    %4 = memref.memory_space_cast %3 : memref<5x4xi32> to memref<5x34xi32, 3>
    // Cast between two non-default memory spaces
    %6 = memref.memory_space_cast %5
      : memref<*xmemref<?xf32>, 5> to memref<*xmemref<?xf32>, 3>
    ```
    """

    def __init__(
        self,
        dest_type: mlir_python._mlir_python.Type,
        source: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.memory_space_cast``: memref memory space cast operation.

        Args:
            dest_type: Type of result ``dest`` (ranked or unranked memref of any type values).
            source: Operand ``source`` (ranked or unranked memref of any type values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: ranked or unranked memref of any type values."""

    @property
    def dest(self) -> mlir_python._mlir_python.OpResult:
        """Result ``dest``: ranked or unranked memref of any type values."""

    OPERATION_NAME: str = "memref.memory_space_cast"

class PrefetchOp(mlir_python._mlir_python.Operation):
    """
    ``memref.prefetch``: prefetch operation.

    The "prefetch" op prefetches data from a memref location described with
    subscript indices similar to memref.load, and with three attributes: a
    read/write specifier, a locality hint, and a cache type specifier as shown
    below:

    ```mlir
    memref.prefetch %0[%i, %j], read, locality<3>, data : memref<400x400xi32>
    ```

    The read/write specifier is either 'read' or 'write', the locality hint
    ranges from locality<0> (no locality) to locality<3> (extremely local keep
    in cache). The cache type specifier is either 'data' or 'instr'
    and specifies whether the prefetch is performed on data cache or on
    instruction cache.
    """

    def __init__(
        self,
        memref: mlir_python._mlir_python.Value,
        is_write: bool,
        locality_hint: int,
        is_data_cache: bool,
        indices: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.prefetch``: prefetch operation.

        Args:
            memref: Operand ``memref`` (prefetch address).
            is_write: Attribute ``isWrite`` (bool attribute).
            locality_hint: Attribute ``localityHint`` (32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3).
            is_data_cache: Attribute ``isDataCache`` (bool attribute).
            indices: Operand ``indices`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def memref(self) -> mlir_python._mlir_python.Value:
        """Operand ``memref``: prefetch address."""

    @property
    def indices(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``indices``: index."""

    @property
    def is_write(self) -> bool:
        """Attribute ``isWrite``: bool attribute."""

    @is_write.setter
    def is_write(self, arg: bool, /) -> None: ...
    @property
    def locality_hint(self) -> int:
        """
        Attribute ``localityHint``: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3.
        """

    @locality_hint.setter
    def locality_hint(self, arg: int, /) -> None: ...
    @property
    def is_data_cache(self) -> bool:
        """Attribute ``isDataCache``: bool attribute."""

    @is_data_cache.setter
    def is_data_cache(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "memref.prefetch"

class RankOp(mlir_python._mlir_python.Operation):
    """
    ``memref.rank``: rank operation.

    The `memref.rank` operation takes a memref operand and returns its rank.

    Example:

    ```mlir
    %0 = memref.rank %arg0 : memref<*xf32>
    %1 = memref.rank %arg1 : memref<?x?xf32>
    ```
    """

    def __init__(
        self,
        memref: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.rank``: rank operation.

        Result types are inferred.

        Args:
            memref: Operand ``memref`` (ranked or unranked memref of any type values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def memref(self) -> mlir_python._mlir_python.Value:
        """Operand ``memref``: ranked or unranked memref of any type values."""

    OPERATION_NAME: str = "memref.rank"

class ReallocOp(mlir_python._mlir_python.Operation):
    """
    ``memref.realloc``: memory reallocation operation.

    The `realloc` operation changes the size of a memory region. The memory
    region is specified by a 1D source memref and the size of the new memory
    region is specified by a 1D result memref type and an optional dynamic Value
    of `Index` type. The source and the result memref must be in the same memory
    space and have the same element type.

    The operation may move the memory region to a new location. In this case,
    the content of the memory block is preserved up to the lesser of the new
    and old sizes. If the new size if larger, the value of the extended memory
    is undefined. This is consistent with the ISO C realloc.

    The operation returns an SSA value for the memref.

    Example:

    ```mlir
    %0 = memref.realloc %src : memref<64xf32> to memref<124xf32>
    ```

    The source memref may have a dynamic shape, in which case, the compiler will
    generate code to extract its size from the runtime data structure for the
    memref.

    ```mlir
    %1 = memref.realloc %src : memref<?xf32> to memref<124xf32>
    ```

    If the result memref has a dynamic shape, a result dimension operand is
    needed to spefify its dynamic dimension. In the example below, the ssa value
    '%d' specifies the unknown dimension of the result memref.

    ```mlir
    %2 = memref.realloc %src(%d) : memref<?xf32> to memref<?xf32>
    ```

    An optional `alignment` attribute may be specified to ensure that the
    region of memory that will be indexed is aligned at the specified byte
    boundary.  This is consistent with the fact that memref.alloc supports such
    an optional alignment attribute. Note that in ISO C standard, neither alloc
    nor realloc supports alignment, though there is aligned_alloc but not
    aligned_realloc.

    ```mlir
    %3 = memref.realloc %src {alignment = 8} : memref<64xf32> to memref<124xf32>
    ```

    Referencing the memref through the old SSA value after realloc is undefined
    behavior.

    ```mlir
    %new = memref.realloc %old : memref<64xf32> to memref<124xf32>
    %4 = memref.load %new[%index] : memref<124xf32> // ok
    %5 = memref.load %old[%index] : memref<64xf32>  // undefined behavior
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        source: mlir_python._mlir_python.Value,
        *,
        dynamic_result_size: mlir_python._mlir_python.Value | None = None,
        alignment: int | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.realloc``: memory reallocation operation.

        Args:
            result_type: Type of result ``result`` (memref of any type values).
            source: Operand ``source`` (memref of any type values).
            dynamic_result_size: Operand ``dynamicResultSize`` (index). Optional.
            alignment: Attribute ``alignment`` (64-bit signless integer attribute whose value is positive and whose value is a power of two > 0). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: memref of any type values."""

    @property
    def dynamic_result_size(self) -> mlir_python._mlir_python.Value | None:
        """Operand ``dynamicResultSize``: index."""

    @property
    def alignment(self) -> int | None:
        """
        Attribute ``alignment``: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0.
        """

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...

    OPERATION_NAME: str = "memref.realloc"

class ReinterpretCastOp(mlir_python._mlir_python.Operation):
    """
    ``memref.reinterpret_cast``: memref reinterpret cast operation.

    Modify offset, sizes and strides of an unranked/ranked memref.

    Example 1:

    Consecutive `reinterpret_cast` operations on memref's with static
    dimensions.

    We distinguish between *underlying memory* — the sequence of elements as
    they appear in the contiguous memory of the memref — and the
    *strided memref*, which refers to the underlying memory interpreted
    according to specified offsets, sizes, and strides.

    ```mlir
    %result1 = memref.reinterpret_cast %arg0 to
      offset: [9],
      sizes: [4, 4],
      strides: [16, 2]
    : memref<8x8xf32, strided<[8, 1], offset: 0>> to
      memref<4x4xf32, strided<[16, 2], offset: 9>>

    %result2 = memref.reinterpret_cast %result1 to
      offset: [0],
      sizes: [2, 2],
      strides: [4, 2]
    : memref<4x4xf32, strided<[16, 2], offset: 9>> to
      memref<2x2xf32, strided<[4, 2], offset: 0>>
    ```

    The underlying memory of `%arg0` consists of a linear sequence of integers
    from 1 to 64. Its memref has the following 8x8 elements:

    ```mlir
    [[1,  2,  3,  4,  5,  6,  7,  8],
    [9,  10, 11, 12, 13, 14, 15, 16],
    [17, 18, 19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30, 31, 32],
    [33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54, 55, 56],
    [57, 58, 59, 60, 61, 62, 63, 64]]
    ```

    Following the first `reinterpret_cast`, the strided memref elements
    of `%result1` are:

    ```mlir
    [[10, 12, 14, 16],
    [26, 28, 30, 32],
    [42, 44, 46, 48],
    [58, 60, 62, 64]]
    ```

    Note: The offset and strides are relative to the underlying memory of
    `%arg0`.

    The second `reinterpret_cast` results in the following strided memref
    for `%result2`:

    ```mlir
    [[1, 3],
    [5, 7]]
    ```

    Notice that it does not matter if you use %result1 or %arg0 as a source
    for the second `reinterpret_cast` operation. Only the underlying memory
    pointers will be reused.

    The offset and stride are relative to the base underlying memory of the
    memref, starting at 1, not at 10 as seen in the output of `%result1`.
    This behavior contrasts with the `subview` operator, where values are
    relative to the strided memref (refer to `subview` examples).
    Consequently, the second `reinterpret_cast` behaves as if `%arg0` were
    passed directly as its argument.

    Example 2:
    ```mlir
    memref.reinterpret_cast %ranked to
      offset: [0],
      sizes: [%size0, 10],
      strides: [1, %stride1]
    : memref<?x?xf32> to memref<?x10xf32, strided<[1, ?], offset: 0>>

    memref.reinterpret_cast %unranked to
      offset: [%offset],
      sizes: [%size0, %size1],
      strides: [%stride0, %stride1]
    : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    ```

    This operation creates a new memref descriptor using the base of the
    source and applying the input arguments to the other metadata.
    In other words:
    ```mlir
    %dst = memref.reinterpret_cast %src to
      offset: [%offset],
      sizes: [%sizes],
      strides: [%strides] :
      memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    ```
    means that `%dst`'s descriptor will be:
    ```mlir
    %dst.base = %src.base
    %dst.aligned = %src.aligned
    %dst.offset = %offset
    %dst.sizes = %sizes
    %dst.strides = %strides
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        source: mlir_python._mlir_python.Value,
        static_offsets: mlir_python._mlir_python.Attribute,
        static_sizes: mlir_python._mlir_python.Attribute,
        static_strides: mlir_python._mlir_python.Attribute,
        offsets: Sequence[mlir_python._mlir_python.Value] = [],
        sizes: Sequence[mlir_python._mlir_python.Value] = [],
        strides: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.reinterpret_cast``: memref reinterpret cast operation.

        Args:
            result_type: Type of result ``result`` (memref of any type values).
            source: Operand ``source`` (ranked or unranked memref of any type values).
            static_offsets: Attribute ``static_offsets`` (i64 dense array attribute).
            static_sizes: Attribute ``static_sizes`` (i64 dense array attribute).
            static_strides: Attribute ``static_strides`` (i64 dense array attribute).
            offsets: Operand ``offsets`` (index). Empty by default.
            sizes: Operand ``sizes`` (index). Empty by default.
            strides: Operand ``strides`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: ranked or unranked memref of any type values."""

    @property
    def offsets(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``offsets``: index."""

    @property
    def sizes(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``sizes``: index."""

    @property
    def strides(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``strides``: index."""

    @property
    def static_offsets(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``static_offsets``: i64 dense array attribute."""

    @static_offsets.setter
    def static_offsets(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def static_sizes(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``static_sizes``: i64 dense array attribute."""

    @static_sizes.setter
    def static_sizes(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def static_strides(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``static_strides``: i64 dense array attribute."""

    @static_strides.setter
    def static_strides(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "memref.reinterpret_cast"

class ReshapeOp(mlir_python._mlir_python.Operation):
    """
    ``memref.reshape``: memref reshape operation.

    The `reshape` operation converts a memref from one type to an
    equivalent type with a provided shape. The data is never copied or
    modified. The source and destination types are compatible if both have the
    same element type, same number of elements, address space and identity
    layout map. The following combinations are possible:

    a. Source type is ranked or unranked. Shape argument has static size.
    Result type is ranked.

    ```mlir
    // Reshape statically-shaped memref.
    %dst = memref.reshape %src(%shape)
             : (memref<4x1xf32>, memref<1xi32>) -> memref<4xf32>
    %dst0 = memref.reshape %src(%shape0)
             : (memref<4x1xf32>, memref<2xi32>) -> memref<2x2xf32>
    // Flatten unranked memref.
    %dst = memref.reshape %src(%shape)
             : (memref<*xf32>, memref<1xi32>) -> memref<?xf32>
    ```

    b. Source type is ranked or unranked. Shape argument has dynamic size.
    Result type is unranked.

    ```mlir
    // Reshape dynamically-shaped 1D memref.
    %dst = memref.reshape %src(%shape)
             : (memref<?xf32>, memref<?xi32>) -> memref<*xf32>
    // Reshape unranked memref.
    %dst = memref.reshape %src(%shape)
             : (memref<*xf32>, memref<?xi32>) -> memref<*xf32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        source: mlir_python._mlir_python.Value,
        shape: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.reshape``: memref reshape operation.

        Args:
            result_type: Type of result ``result`` (ranked or unranked memref of any type values).
            source: Operand ``source`` (ranked or unranked memref of any type values).
            shape: Operand ``shape`` (dynamically-sized shape).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: ranked or unranked memref of any type values."""

    @property
    def shape(self) -> mlir_python._mlir_python.Value:
        """Operand ``shape``: dynamically-sized shape."""

    OPERATION_NAME: str = "memref.reshape"

class StoreOp(mlir_python._mlir_python.Operation):
    """
    ``memref.store``: store operation.

    The `store` op stores an element into a memref at the specified indices.

    The number of indices must match the rank of the memref. The indices must
    be in-bounds: `0 <= idx < dim_size`.

    Lowerings of `memref.store` may emit attributes, e.g. `inbouds` + `nuw`
    when converting to LLVM's `llvm.getelementptr`, that would cause undefined
    behavior if indices are out of bounds or if computing the offset in the
    memref would cause signed overflow of the `index` type.

    A set `nontemporal` attribute indicates that this store is not expected to
    be reused in the cache. For details, refer to the
    [LLVM store instruction](https://llvm.org/docs/LangRef.html#store-instruction).

    An optional `alignment` attribute allows to specify the byte alignment of the
    store operation. It must be a positive power of 2. The operation must access
    memory at an address aligned to this boundary. Violations may lead to
    architecture-specific faults or performance penalties.
    A value of 0 indicates no specific alignment requirement.
    Example:

    ```mlir
    memref.store %val, %A[%a, %b] : memref<8x?xi32, #layout, memspace0>
    ```
    """

    def __init__(
        self,
        value: mlir_python._mlir_python.Value,
        memref: mlir_python._mlir_python.Value,
        indices: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        nontemporal: bool = False,
        alignment: int | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.store``: store operation.

        Args:
            value: Operand ``value`` (any type).
            memref: Operand ``memref`` (the reference to store to).
            indices: Operand ``indices`` (index). Empty by default.
            nontemporal: Attribute ``nontemporal`` (bool attribute). Omit for the default.
            alignment: Attribute ``alignment`` (64-bit signless integer attribute whose value is positive and whose value is a power of two > 0). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: any type."""

    @property
    def memref(self) -> mlir_python._mlir_python.Value:
        """Operand ``memref``: the reference to store to."""

    @property
    def indices(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``indices``: index."""

    @property
    def nontemporal(self) -> bool:
        """Attribute ``nontemporal``: bool attribute."""

    @nontemporal.setter
    def nontemporal(self, arg: bool, /) -> None: ...
    @property
    def alignment(self) -> int | None:
        """
        Attribute ``alignment``: 64-bit signless integer attribute whose value is positive and whose value is a power of two > 0.
        """

    @alignment.setter
    def alignment(self, arg: int | None) -> None: ...

    OPERATION_NAME: str = "memref.store"

class TransposeOp(mlir_python._mlir_python.Operation):
    """
    ``memref.transpose``: `transpose` produces a new strided memref (metadata-only).

    The `transpose` op produces a strided memref whose sizes and strides
    are a permutation of the original `in` memref. This is purely a metadata
    transformation.

    Example:

    ```mlir
    %1 = memref.transpose %0 (i, j) -> (j, i) : memref<?x?xf32> to memref<?x?xf32, affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        permutation: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.transpose``: `transpose` produces a new strided memref (metadata-only).

        Args:
            result_type: Type of result ``result`` (memref of any type values).
            in_: Operand ``in`` (memref of any type values).
            permutation: Attribute ``permutation`` (AffineMap attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def permutation(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``permutation``: AffineMap attribute."""

    @permutation.setter
    def permutation(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "memref.transpose"

class ViewOp(mlir_python._mlir_python.Operation):
    """
    ``memref.view``: memref view operation.

    The "view" operation extracts an N-D contiguous memref with empty layout map
    with arbitrary element type from a 1-D contiguous memref with empty layout
    map of i8 element  type. The ViewOp supports the following arguments:

    * A single dynamic byte-shift operand must be specified which represents a
      a shift of the base 1-D memref pointer from which to create the resulting
      contiguous memref view with identity layout.
    * A dynamic size operand that must be specified for each dynamic dimension
      in the resulting view memref type.

    The "view" operation gives a structured indexing form to a flat 1-D buffer.
    Unlike "subview" it can perform a type change. The type change behavior
    requires the op to have special semantics because, e.g. a byte shift of 3
    cannot be represented as an offset on f64.
    For now, a "view" op:

    1. Only takes a contiguous source memref with 0 offset and empty layout.
    2. Must specify a byte_shift operand (in the future, a special integer
       attribute may be added to support the folded case).
    3. Returns a contiguous memref with 0 offset and empty layout.

    Example:

    ```mlir
    // Allocate a flat 1D/i8 memref.
    %0 = memref.alloc() : memref<2048xi8>

    // ViewOp with dynamic offset and static sizes.
    %1 = memref.view %0[%offset_1024][] : memref<2048xi8> to memref<64x4xf32>

    // ViewOp with dynamic offset and two dynamic size.
    %2 = memref.view %0[%offset_1024][%size0, %size1] :
      memref<2048xi8> to memref<?x4x?xf32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        source: mlir_python._mlir_python.Value,
        byte_shift: mlir_python._mlir_python.Value,
        sizes: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.view``: memref view operation.

        Args:
            result_type: Type of result ``result`` (memref of any type values).
            source: Operand ``source`` (memref of 8-bit signless integer values).
            byte_shift: Operand ``byte_shift`` (index).
            sizes: Operand ``sizes`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: memref of 8-bit signless integer values."""

    @property
    def byte_shift(self) -> mlir_python._mlir_python.Value:
        """Operand ``byte_shift``: index."""

    @property
    def sizes(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``sizes``: index."""

    OPERATION_NAME: str = "memref.view"

class SubViewOp(mlir_python._mlir_python.Operation):
    """
    ``memref.subview``: memref subview operation.

    The `subview` operation converts a memref type to a memref type which
    represents a reduced-size view of the original memref as specified by the
    operation's offsets, sizes and strides arguments.

    The `subview` operation supports the following arguments:

    * source: the "base" memref on which to create a "view" memref.
    * offsets: memref-rank number of offsets into the "base" memref at which to
               create the "view" memref.
    * sizes: memref-rank number of sizes which specify the sizes of the result
             "view" memref type.
    * strides: memref-rank number of strides that compose multiplicatively with
               the base memref strides in each dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value `ShapedType::kDynamic` encodes that the corresponding entry
    has a dynamic value.

    A `subview` operation may additionally reduce the rank of the resulting
    view by removing dimensions that are statically known to be of size 1.

    In the absence of rank reductions, the resulting memref type is computed
    as follows:
    ```
    result_sizes[i] = size_operands[i]
    result_strides[i] = src_strides[i] * stride_operands[i]
    result_offset = src_offset + dot_product(offset_operands, src_strides)
    ```

    The offset, size and stride operands must be in-bounds with respect to the
    source memref. When possible, the static operation verifier will detect
    out-of-bounds subviews. Subviews that cannot be confirmed to be in-bounds
    or out-of-bounds based on compile-time information are valid. However,
    performing an out-of-bounds subview at runtime is undefined behavior.

    Example 1:

    Consecutive `subview` operations on memref's with static dimensions.

    We distinguish between *underlying memory* — the sequence of elements as
    they appear in the contiguous memory of the memref — and the
    *strided memref*, which refers to the underlying memory interpreted
    according to specified offsets, sizes, and strides.

    ```mlir
    %result1 = memref.subview %arg0[1, 1][4, 4][2, 2]
    : memref<8x8xf32, strided<[8, 1], offset: 0>> to
      memref<4x4xf32, strided<[16, 2], offset: 9>>

    %result2 = memref.subview %result1[1, 1][2, 2][2, 2]
    : memref<4x4xf32, strided<[16, 2], offset: 9>> to
      memref<2x2xf32, strided<[32, 4], offset: 27>>
    ```

    The underlying memory of `%arg0` consists of a linear sequence of integers
    from 1 to 64. Its memref has the following 8x8 elements:

    ```mlir
    [[1,  2,  3,  4,  5,  6,  7,  8],
    [9,  10, 11, 12, 13, 14, 15, 16],
    [17, 18, 19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30, 31, 32],
    [33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54, 55, 56],
    [57, 58, 59, 60, 61, 62, 63, 64]]
    ```

    Following the first `subview`, the strided memref elements of `%result1`
    are:

    ```mlir
    [[10, 12, 14, 16],
    [26, 28, 30, 32],
    [42, 44, 46, 48],
    [58, 60, 62, 64]]
    ```

    Note: The offset and strides are relative to the strided memref of `%arg0`
    (compare to the corresponding `reinterpret_cast` example).

    The second `subview` results in the following strided memref for
    `%result2`:

    ```mlir
    [[28, 32],
    [60, 64]]
    ```

    Unlike the `reinterpret_cast`, the values are relative to the strided
    memref of the input (`%result1` in this case) and not its
    underlying memory.

    Example 2:

    ```mlir
    // Subview of static memref with strided layout at static offsets, sizes
    // and strides.
    %1 = memref.subview %0[4, 2][8, 2][3, 2]
        : memref<64x4xf32, strided<[7, 9], offset: 91>> to
          memref<8x2xf32, strided<[21, 18], offset: 137>>
    ```

    Example 3:

    ```mlir
    // Subview of static memref with identity layout at dynamic offsets, sizes
    // and strides.
    %1 = memref.subview %0[%off0, %off1][%sz0, %sz1][%str0, %str1]
        : memref<64x4xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
    ```

    Example 4:

    ```mlir
    // Subview of dynamic memref with strided layout at dynamic offsets and
    // strides, but static sizes.
    %1 = memref.subview %0[%off0, %off1][4, 4][%str0, %str1]
        : memref<?x?xf32, strided<[?, ?], offset: ?>> to
          memref<4x4xf32, strided<[?, ?], offset: ?>>
    ```

    Example 5:

    ```mlir
    // Rank-reducing subviews.
    %1 = memref.subview %0[0, 0, 0][1, 16, 4][1, 1, 1]
        : memref<8x16x4xf32> to memref<16x4xf32>
    %3 = memref.subview %2[3, 4, 2][1, 6, 3][1, 1, 1]
        : memref<8x16x4xf32> to memref<6x3xf32, strided<[4, 1], offset: 210>>
    ```

    Example 6:

    ```mlir
    // Identity subview. The subview is the full source memref.
    %1 = memref.subview %0[0, 0, 0] [8, 16, 4] [1, 1, 1]
        : memref<8x16x4xf32> to memref<8x16x4xf32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        source: mlir_python._mlir_python.Value,
        static_offsets: mlir_python._mlir_python.Attribute,
        static_sizes: mlir_python._mlir_python.Attribute,
        static_strides: mlir_python._mlir_python.Attribute,
        offsets: Sequence[mlir_python._mlir_python.Value] = [],
        sizes: Sequence[mlir_python._mlir_python.Value] = [],
        strides: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``memref.subview``: memref subview operation.

        Args:
            result_type: Type of result ``result`` (memref of any type values).
            source: Operand ``source`` (memref of any type values).
            static_offsets: Attribute ``static_offsets`` (i64 dense array attribute).
            static_sizes: Attribute ``static_sizes`` (i64 dense array attribute).
            static_strides: Attribute ``static_strides`` (i64 dense array attribute).
            offsets: Operand ``offsets`` (index). Empty by default.
            sizes: Operand ``sizes`` (index). Empty by default.
            strides: Operand ``strides`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: memref of any type values."""

    @property
    def offsets(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``offsets``: index."""

    @property
    def sizes(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``sizes``: index."""

    @property
    def strides(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``strides``: index."""

    @property
    def static_offsets(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``static_offsets``: i64 dense array attribute."""

    @static_offsets.setter
    def static_offsets(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def static_sizes(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``static_sizes``: i64 dense array attribute."""

    @static_sizes.setter
    def static_sizes(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def static_strides(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``static_strides``: i64 dense array attribute."""

    @static_strides.setter
    def static_strides(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "memref.subview"
