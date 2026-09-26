"""Typed operations of the MLIR ``tensor`` dialect."""

from collections.abc import Sequence

import mlir_python._mlir_python

class BitcastOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.bitcast``: tensor bitcast operation.

    Bitcast a tensor from one type to another type of equivalent element width.
    If both are ranked, then the rank should be the same and static dimensions
    should match.

    Example:

    ```mlir
    // Bitcast from unsigned to signed or signless integer.
    %2 = tensor.bitcast %1 : tensor<4xui32> to tensor<4xi32>
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
        Create ``tensor.bitcast``: tensor bitcast operation.

        Args:
            dest_type: Type of result ``dest`` (tensor of signless integer or unsigned integer or signed integer or floating-point values).
            source: Operand ``source`` (tensor of signless integer or unsigned integer or signed integer or floating-point values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``source``: tensor of signless integer or unsigned integer or signed integer or floating-point values.
        """

    @property
    def dest(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``dest``: tensor of signless integer or unsigned integer or signed integer or floating-point values.
        """

    OPERATION_NAME: str = "tensor.bitcast"

class CastOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.cast``: tensor cast operation.

    Convert a tensor from one type to an equivalent type without changing any
    data elements. The source and destination types must both be tensor types
    with the same element type. If both are ranked, then the rank should be the
    same and static dimensions should match. The operation is invalid if
    converting to a mismatching constant dimension.

    Example:

    ```mlir
    // Convert from unknown rank to rank 2 with unknown dimension sizes.
    %2 = tensor.cast %1 : tensor<*xf32> to tensor<?x?xf32>

    // Convert to a type with more known dimensions.
    %3 = tensor.cast %2 : tensor<?x?xf32> to tensor<4x?xf32>

    // Discard static dimension and rank information.
    %4 = tensor.cast %3 : tensor<4x?xf32> to tensor<?x?xf32>
    %5 = tensor.cast %4 : tensor<?x?xf32> to tensor<*xf32>
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
        Create ``tensor.cast``: tensor cast operation.

        Args:
            dest_type: Type of result ``dest`` (tensor of any type values).
            source: Operand ``source`` (tensor of any type values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: tensor of any type values."""

    @property
    def dest(self) -> mlir_python._mlir_python.OpResult:
        """Result ``dest``: tensor of any type values."""

    OPERATION_NAME: str = "tensor.cast"

class CollapseShapeOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.collapse_shape``: operation to produce a tensor with a smaller rank.

    The `tensor.collapse_shape` op produces a new tensor of lower (or equal)
    rank whose dimension sizes are a reassociation of the original `src` dimensions.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented by an array of DenseI64ArrayAttr attribute. The reassociation
    maps are applied to the operand shape to obtain the result shape.


    Example:

    ```mlir
    // Dimension collapse (i, j) -> i' and k -> k'
    %b = tensor.collapse_shape %a [[0, 1], [2]]
        : tensor<?x?x?xf32> into tensor<?x?xf32>
    ```
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
        Create ``tensor.collapse_shape``: operation to produce a tensor with a smaller rank.

        Args:
            result_type: Type of result ``result`` (tensor of any type values).
            src: Operand ``src`` (tensor of any type values).
            reassociation: Attribute ``reassociation`` (Array of 64-bit integer array attributes).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def src(self) -> mlir_python._mlir_python.Value:
        """Operand ``src``: tensor of any type values."""

    @property
    def reassociation(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``reassociation``: Array of 64-bit integer array attributes."""

    @reassociation.setter
    def reassociation(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...

    OPERATION_NAME: str = "tensor.collapse_shape"

class ConcatOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.concat``: tensor concatenation operation.

    The "concat" operation constructs a tensor out of a variadic list of input
    tensors, concatenated along a static dimension number. All inputs and the
    result type must share the same rank.

    `dim` specifies the dimension along which to concatenate. The size of the
    concatenated dimension in the result must be equal to the sum of the sizes
    of the inputs along that dimension. All other dimensions in both the inputs
    and result must be the same size.

    Example:

    ```mlir
    %0 = tensor.concat dim(0) %0, %1, %2 :
        (tensor<3x6xf32>, tensor<3x6xf32>, tensor<1x6xf32) -> tensor<7x6xf32>

    // Dynamic + dynamic -> static
    %0 = tensor.concat dim(1) %0, %1, %2 :
        (tensor<3x?xf32>, tensor<3x2xf32>, tensor<3x?xf32) -> tensor<3x10xf32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        dim: int,
        inputs: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.concat``: tensor concatenation operation.

        Args:
            result_type: Type of result ``result`` (ranked tensor of any type values).
            dim: Attribute ``dim`` (64-bit signless integer attribute).
            inputs: Operand ``inputs`` (ranked tensor of any type values). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def inputs(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``inputs``: ranked tensor of any type values."""

    @property
    def dim(self) -> int:
        """Attribute ``dim``: 64-bit signless integer attribute."""

    @dim.setter
    def dim(self, arg: int, /) -> None: ...

    OPERATION_NAME: str = "tensor.concat"

class DimOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.dim``: dimension index operation.

    The `tensor.dim` operation takes a tensor and a dimension operand of type
    `index`. It returns the size of the requested dimension of the given
    tensor. If the dimension index is out of bounds, the behavior is undefined.

    The specified tensor type is that of the first operand.

    Example:

    ```mlir
    // Always returns 4, can be constant folded:
    %c0 = arith.constant 0 : index
    %x = tensor.dim %A, %c0 : tensor<4x?xf32>

    // Return the dynamic dimension of %A.
    %c1 = arith.constant 1 : index
    %y = tensor.dim %A, %c1 : tensor<4x?xf32>

    // Equivalent generic form:
    %x = "tensor.dim"(%A, %c0) : (tensor<4x?xf32>, index) -> index
    %y = "tensor.dim"(%A, %c1) : (tensor<4x?xf32>, index) -> index
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
        Create ``tensor.dim``: dimension index operation.

        Result types are inferred.

        Args:
            source: Operand ``source`` (non-0-ranked or unranked tensor).
            index: Operand ``index`` (index).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: non-0-ranked or unranked tensor."""

    @property
    def index(self) -> mlir_python._mlir_python.Value:
        """Operand ``index``: index."""

    OPERATION_NAME: str = "tensor.dim"

class EmptyOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.empty``: empty tensor operation.

    `tensor.empty` is an operation that defines a tensor of a particular shape.
    The shape could be dynamic or static. The contents of the tensor are
    unspecified and the only purpose of the op result is to materialize the
    specified shape in IR and make it available to other transformations.

    `tensor.empty` is useful in transformations that expect destination style
    ops. I.e., ops that implement `DestinationStyleOpInterface`. Ops that are
    not in destination style can be made compatible with such transformations
    with a `tensor.empty` destination.

    Note: This op can be lowered to a `bufferization.alloc_tensor`, at which
    point it turns into an explicit buffer allocation.
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        dynamic_sizes: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.empty``: empty tensor operation.

        Args:
            result_type: Type of result ``result`` (ranked tensor of any type values).
            dynamic_sizes: Operand ``dynamicSizes`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def dynamic_sizes(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``dynamicSizes``: index."""

    OPERATION_NAME: str = "tensor.empty"

class ExpandShapeOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.expand_shape``: operation to produce a tensor with a higher rank.

    The `tensor.expand_shape` op produces a tensor of higher (or equal)
    rank than the operand `src` whose dimension sizes are a reassociation of
    `src`.

    A reassociation is defined as a continuous grouping of dimensions and is
    represented with an array of DenseI64ArrayAttr attribute.  The reassociation
    maps applied to the result tensor with the higher rank must result in the
    operand tensor with the smaller rank.

    The representation for the output shape supports a partially-static
    specification via attributes specified through the `static_output_shape`
    argument.  A special sentinel value `ShapedType::kDynamic` encodes that the
    corresponding entry has a dynamic value.  There must be exactly as many SSA
    inputs in `output_shape` as there are `ShapedType::kDynamic` entries in
    `static_output_shape`.

    Example:

    ```mlir
    // Dimension expansion i -> (i', j') and (k) -> (k')
    %b = tensor.expand_shape %a [[0, 1], [2]] output_shape [%sz0, %sz1, 32]
        : tensor<?x32xf32> into tensor<?x?x32xf32>
    ```
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
        Create ``tensor.expand_shape``: operation to produce a tensor with a higher rank.

        Args:
            result_type: Type of result ``result`` (tensor of any type values).
            src: Operand ``src`` (tensor of any type values).
            reassociation: Attribute ``reassociation`` (Array of 64-bit integer array attributes).
            static_output_shape: Attribute ``static_output_shape`` (i64 dense array attribute).
            output_shape: Operand ``output_shape`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def src(self) -> mlir_python._mlir_python.Value:
        """Operand ``src``: tensor of any type values."""

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

    OPERATION_NAME: str = "tensor.expand_shape"

class ExtractOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.extract``: element extraction operation.

    The `tensor.extract` op reads a ranked tensor and returns one element as
    specified by the given indices. The result of the op is a value with the
    same type as the elements of the tensor. The arity of indices must match
    the rank of the accessed value. All indices should all be of `index` type.

    Example:

    ```mlir
    %4 = tensor.extract %t[%1, %2] : tensor<4x4xi32>
    %5 = tensor.extract %rt[%1, %2] : tensor<?x?xi32>
    ```
    """

    def __init__(
        self,
        tensor: mlir_python._mlir_python.Value,
        indices: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.extract``: element extraction operation.

        Result types are inferred.

        Args:
            tensor: Operand ``tensor`` (ranked tensor of any type values).
            indices: Operand ``indices`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def tensor(self) -> mlir_python._mlir_python.Value:
        """Operand ``tensor``: ranked tensor of any type values."""

    @property
    def indices(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``indices``: index."""

    OPERATION_NAME: str = "tensor.extract"

class ExtractSliceOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.extract_slice``: extract slice operation.

    The "extract_slice" operation extract a tensor from another tensor as
    specified by the operation's offsets, sizes and strides arguments.

    The extract_slice operation supports the following arguments:

    * source: the "base" tensor from which to extract a slice.
    * offsets: tensor-rank number of offsets into the "base" tensor from which
               to extract the slice.
    * sizes: tensor-rank number of sizes which specify the sizes of the result
             tensor type.
    * strides: tensor-rank number of strides specifying subsampling in each
               dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value ShapedType::kDynamic encodes that the corresponding entry has
    a dynamic value.

    After buffer allocation, the "extract_slice" op is expected to lower into a
    memref.subview op.

    An extract_slice operation may additionally reduce the rank of the resulting
    tensor by removing dimensions that are statically known to be of size 1.
    This rank-reduction behavior is not required by the op semantics: this
    flexibility allows to progressively drop unit dimensions while lowering
    between different flavors of ops on that operate on tensors.

    #### Verification vs Inference in the rank-reduced case

    Note that there may be multiple ways to infer a resulting rank-reduced type.
      e.g. 1x6x1 could potentially rank-reduce to either 1x6 or 6x1 2-D shapes.

    To disambiguate, the inference helpers `inferCanonicalRankReducedResultType`
    only drop the first unit dimensions, in order:
      e.g. 1x6x1 rank-reduced to 2-D will infer the 6x1 2-D shape, but not 1x6.

    Verification however has access to result type and does not need to infer.
    The verifier calls `isRankReducedType(getSource(), getResult())` to
    determine whether the result type is rank-reduced from the source type.
    This computes a so-called rank-reduction mask, consisting of dropped unit
    dims, to map the rank-reduced type to the source type by dropping ones:
      e.g. 1x6 is a rank-reduced version of 1x6x1 by mask {2}
           6x1 is a rank-reduced version of 1x6x1 by mask {0}
           1x2x1x4 is a rank-reduced version of 1x1x2x1x1x4x1 by mask {1, 4, 6}
             (remaining common 1 dimensions are matched eagerly)

    Example:

    ```mlir
    // Rank-reducing extract_slice.
    %1 = tensor.extract_slice %0[0, 0, 0][1, 16, 4][1, 1, 1] :
      tensor<8x16x4xf32> to tensor<16x4xf32>
    %3 = tensor.extract_slice %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1] :
      tensor<8x16x4xf32> to tensor<1x?xf32>
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
        Create ``tensor.extract_slice``: extract slice operation.

        Args:
            result_type: Type of result ``result`` (ranked tensor of any type values).
            source: Operand ``source`` (ranked tensor of any type values).
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
        """Operand ``source``: ranked tensor of any type values."""

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

    OPERATION_NAME: str = "tensor.extract_slice"

class FromElementsOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.from_elements``: tensor from elements operation..

    Create a N-D tensor from a range of same-type arguments. The number of
    provided `elements` should equal to the number of the elements in the
    result type. The `elements` correspond to a flattened tensor.

    Example:

    ```mlir
    tensor.from_elements %a, %b, %c, %d, %e, %f :  tensor<2x3xindex>
    ```

    will result in a tensor

    [[%a, %b, %c]
     [%d, %e, %f]]
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        elements: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.from_elements``: tensor from elements operation..

        Args:
            result_type: Type of result ``result`` (statically shaped tensor of any type values).
            elements: Operand ``elements`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def elements(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``elements``: any type."""

    OPERATION_NAME: str = "tensor.from_elements"

class GatherOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.gather``: gather a subset of a tensor at specified indices.

    The `gather` operation extracts a subset of the elements from a `source`
    tensor at the given indices.

    In its most general form, the tensor of indices specifies all the coordinates
    of every element to extract (i.e. COO format, without the payload).
    The indices are expected to be confined to coordinate values that fit the
    range of the `source` tensor, otherwise the behavior is undefined.

    The leading dimensions of the index tensor give the result tensor its leading
    dimensions. The trailing dimensions of the result tensor are obtained from
    the source tensor by omitting the dimensions specified in `gather_dims`
    (rank-reducing semantics) or setting them to `1` (rank-preserving semantics)
    (see examples).
    The trailing dimension of the index tensor contains the coordinates and is
    expected to have its size equal to the number of dimensions being gathered.
    This convention allows an idiomatic specification and lowering of "gathering
    multiple N-D slices from the source tensor".

    Note: in the examples below, we separate out the indexing part of the tensor
    type by a whitespace for readability purposes.

    Example:

    ```mlir
        // For each 1x2 triple of coordinates in %indices, extract the
        // element (i.e. 0-D subset) at the coordinates triple in %source.
        //
        %out = tensor.gather %source[%indices] gather_dims([0, 1, 2]) :
          (tensor<4x4x4xf32>, tensor<1x2x 3xindex>) -> tensor<1x2x 1x1x1xf32>

        // Note: result type may be further rank-reduced to tensor<1x2x f32>.
    ```

    A slice variant is provided to allow specifying whole slices of the source
    tensor.

    Example:

    ```mlir
        // For each 5x6 singleton of coordinates in %indices, extract the 2-D
        // slice %source[*, %indices[...]:%indices[...] + 1, *] with the indices
        // corresponding to the `gather_dims` attribute specified by %indices.
        //
        %out = tensor.gather %source[%indices] gather_dims([1]) :
          (tensor<3x4x5xf32>, tensor<6x7x 1xindex>) -> tensor<6x7x 3x1x5xf32>

        // Note: result type may be further rank-reduced to tensor<6x7x 3x5xf32>.
    ```

    The dimensions specified in the gather_dims attribute are ones for which the
    result tensor has size `1`.
    I.e. if the source type is `axbxcxd` and the coordinates are [1, 3], then
    the shape suffix is `ax1xcx1`.
    Gather also allows rank-reducing semantics where the shape `ax1xcx1` can be
    further simplified to `axc`.

    The elemental type of the indices tensor can be any integer type.
    In the absence of target-specific or problem specific information the default
    type one should use is `index`.

    This operation does not support unranked tensors.

    An optional `unique` unit attribute may be specified to indicate that the
    coordinates in `indices` are statically guaranteed to be unique at runtime.
    Incorrectly setting the `unique` attribute when the coordinates are not truly
    unique is undefined behavior.

    Only full slices are meant to be supported by this op, if one desires
    partial slices (e.g. strided windows) one should compose this op with other
    tensor ops (e.g. tensor.extract_slice). This is to avoid a slippery slope of
    complexity that would make the op unusable in practice.

    At the tensor-level, the index tensor is specified in an AoS form (i.e.
    coordinate tuple is the most minor). It is the responsibility of further
    lowerings and bufferization to implement various concrete layouts.

    Note: As currently specified, the operation must lower to an abstraction that
    performs copies to the output tensor. This is because the buffer type system
    is currently not rich enough to allow multiple non-contiguous views in the
    same type. This is visible more clearly in a notional buffer version of the
    op:

    ```mlir
        // memref<?x4x1xf32> is a contiguous buffer of ?x4x1 elements.
        // gather from random source slices must copy to the contiguous output.
        %out = memref.gather %source[%indices] gather_dims([1]) :
          (memref<4x4xf32>, memref<?x 1xindex>) -> memref<?x 4x1xf32>

        // Nested buffer support would allow gather to directly index into the
        // source buffer (i.e. represent a jagged view into the source).
        %out = memref.gather %source[%indices] gather_dims([1]) :
          (memref<4x4xf32>, memref<?x 1xindex>) -> memref<? x memref<4x1xf32>>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        source: mlir_python._mlir_python.Value,
        indices: mlir_python._mlir_python.Value,
        gather_dims: mlir_python._mlir_python.Attribute,
        *,
        unique: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.gather``: gather a subset of a tensor at specified indices.

        Args:
            result_type: Type of result ``result`` (ranked tensor of any type values).
            source: Operand ``source`` (ranked tensor of any type values).
            indices: Operand ``indices`` (ranked tensor of signless integer or index values).
            gather_dims: Attribute ``gather_dims`` (i64 dense array attribute).
            unique: Attribute ``unique`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: ranked tensor of any type values."""

    @property
    def indices(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``indices``: ranked tensor of signless integer or index values.
        """

    @property
    def gather_dims(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``gather_dims``: i64 dense array attribute."""

    @gather_dims.setter
    def gather_dims(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def unique(self) -> bool:
        """Attribute ``unique``: unit attribute."""

    @unique.setter
    def unique(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "tensor.gather"

class GenerateOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.generate``: Creates a dynamically sized tensor from elements.

    This operation creates a dynamically sized tensor with elements of any type.
    It expects one index operand per dynamic extent of the result tensor.

    The body region defines the tensor's elements. It takes index operands as
    its region arguments that span the index space. The element at the given
    position is yielded with the `yield` operation (see `YieldOp`). There is
    no defined ordering to the invocations of the body. It is conceptually
    a "parallel map" operation.

    Example:

    ```mlir
      %tnsr = tensor.generate %m, %n {
      ^bb0(%i : index, %j : index, %k : index):
        ...
        yield %elem : f32
      } : tensor<?x3x?f32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        dynamic_extents: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.generate``: Creates a dynamically sized tensor from elements.

        Args:
            result_type: Type of result ``result`` (ranked tensor of any type values).
            dynamic_extents: Operand ``dynamicExtents`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def dynamic_extents(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``dynamicExtents``: index."""

    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: region with 1 blocks."""

    OPERATION_NAME: str = "tensor.generate"

class InsertOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.insert``: element insertion operation.

    The `tensor.insert` op inserts a scalar into a ranked tensor `dest` as
    specified by the operation's indices.

    It returns a copy of `dest` with the indexed position updated to the value
    of `scalar`.

    The arity of `indices `must match the rank of the tensor `dest`. All
    indices should be of `index` type.

    Example:

    ```mlir
    %4 = tensor.insert %t into %dest[%1, %2] : tensor<4x4xi32>
    %5 = tensor.insert %rt into %dest[%1, %2] : tensor<?x?xi32>
    ```
    """

    def __init__(
        self,
        scalar: mlir_python._mlir_python.Value,
        dest: mlir_python._mlir_python.Value,
        indices: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.insert``: element insertion operation.

        Result types are inferred.

        Args:
            scalar: Operand ``scalar`` (any type).
            dest: Operand ``dest`` (ranked tensor of any type values).
            indices: Operand ``indices`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def scalar(self) -> mlir_python._mlir_python.Value:
        """Operand ``scalar``: any type."""

    @property
    def dest(self) -> mlir_python._mlir_python.Value:
        """Operand ``dest``: ranked tensor of any type values."""

    @property
    def indices(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``indices``: index."""

    OPERATION_NAME: str = "tensor.insert"

class InsertSliceOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.insert_slice``: insert_slice operation.

    The "insert_slice" operation insert a tensor `source` into another
    tensor `dest` as specified by the operation's offsets, sizes and strides
    arguments.

    It returns a copy of `dest` with the proper slice updated with the value
    of `source`.

    The insert_slice operation supports the following arguments:

    * source: the tensor that is inserted.
    * dest: the tensor into which the source tensor is inserted.
    * offsets: tensor-rank number of offsets into the `dest` tensor into which
               the slice is inserted.
    * sizes: tensor-rank number of sizes which specify the sizes of the source
             tensor type.
    * strides: tensor-rank number of strides that specify subsampling in each
               dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value ShapedType::kDynamic encodes that the corresponding entry has
    a dynamic value.

    After buffer allocation, the "insert_slice" op is expected to lower into a
    memref.subview op.

    An insert_slice operation may additionally specify insertion into a tensor
    of higher rank than the source tensor, along dimensions that are statically
    known to be of size 1.
    This rank-altering behavior is not required by the op semantics: this
    flexibility allows to progressively drop unit dimensions while lowering
    between different flavors of ops on that operate on tensors.
    The rank-altering behavior of tensor.insert_slice matches the rank-reducing
    behavior of tensor.extract_slice.

    #### Verification in the rank-reduced case

    The same verification discussion and mechanisms apply as for ExtractSliceOp.
    Unlike ExtractSliceOp however, there is no need for a specific inference.

    Example:

    ```mlir
    // Rank-altering insert_slice.
    %1 = tensor.insert_slice %t into %0[0, 0, 0][1, 16, 4][1, 1, 1] :
      tensor<16x4xf32> into tensor<8x16x4xf32>
    %3 = tensor.insert_slice %tt into %2[%o0, 4, %o2][1, %sz1, 1][1, %st1, 1] :
      tensor<1x?xf32> into tensor<8x16x4xf32>
    ```
    """

    def __init__(
        self,
        source: mlir_python._mlir_python.Value,
        dest: mlir_python._mlir_python.Value,
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
        Create ``tensor.insert_slice``: insert_slice operation.

        Result types are inferred.

        Args:
            source: Operand ``source`` (ranked tensor of any type values).
            dest: Operand ``dest`` (ranked tensor of any type values).
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
        """Operand ``source``: ranked tensor of any type values."""

    @property
    def dest(self) -> mlir_python._mlir_python.Value:
        """Operand ``dest``: ranked tensor of any type values."""

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

    OPERATION_NAME: str = "tensor.insert_slice"

class PadOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.pad``: tensor pad operation.

    `tensor.pad` is an operation that pads the `source` tensor
    with given `low` and `high` padding config.

    The PadOp operation supports the following arguments:

    * source: the "base" tensor on which to pad.
    * low: A list contains the padding along the start of each
           dimension, i.e., how many padded values are prepended
           to the beginning of the tensor in each dimension.
    * high: A list contains the padding along the end of each
            dimension, i.e., how many padded values are appended
            to the end of the tensor in each dimension.
    * nofold: indicates that the operation should not be folded when source and
              result types are equal.

    The result tensor dimensions are `low[i]` + `dim[i]` + `high[i]` for each
    dimension `i`. The number of elements of `low` and `high` must match the
    rank of the input tensor. They can be either a constant or a dynamic value.

    The region of the `tensor.pad` operation returns the value to use
    for the padding. The arguments of the region represent the index
    of the source being accessed. There should be as many arguments as
    the rank of the `source` tensor. The value `yield`-ed by the
    region is used as the value of the view at the given position.

    If `nofold` is set, the padding operation will not be folded away even
    if the source type and the padded type have the same static shape. This can
    be used, e.g., for packing or promotion to faster memory.

    Example 1: add 3 zeros to the beginning and 5 zeros to the end of a 1D
    tensor.

    ```mlir
      %arg0 = ... : tensor<10xi32>
      %c0_i32 = arith.constant 0 : i32
      %padded = tensor.pad %arg0 low[3] high[5] {
      ^bb0(%arg1: index):
        tensor.yield %c0_i32 : i32
      } : tensor<10xi32> to tensor<18xi32>
    ```

    Example 2: add 1 value to the beginning of dimension 0, 2 values to the end
    of dimension 0, 2 values to the start of dimension 1, and 3 values to the
    end of dimension 1.

    ```mlir
      %pad_value = ... : f32
      %0 = tensor.pad %0 low[1, 2] high[2, 3] {
      ^bb0(%arg0 : index, %arg1 : index):
        tensor.yield %pad_value : f32
      } : tensor<?x?xf32> to tensor<?x?xf32>
    ```

    Example 3:

    ```mlir
      %pad_value = ... : f32
      %0 = tensor.pad %arg0 low[2, %arg1, 3, 3] high[3, 3, %arg1, 2] {
      ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
          tensor.yield %pad_value : f32
      } : tensor<1x2x2x?xf32> to tensor<6x?x?x?xf32>
    ```

    Example 4:

    ```mlir
      %pad_value = ... : f32
      %0 = tensor.pad %arg0 low[0, 0] high[%ub0, %ub1] {
      ^bb0(%arg1: index, %arg2: index):
        tensor.yield %pad_value : f32
      } : tensor<2x3xf32> to tensor<?x?xf32>
    ```

    Example 5: Force a padded value to be always exist with `nofold`, even
    though the padding config specifies that no new elements will be added to
    the tensor.

    ```mlir
      %pad_value = ... : f32
      %0 = tensor.pad %arg0 nofold low[0, 0] high[0, 0] {
      ^bb0(%arg1: index, %arg2: index):
        tensor.yield %pad_value : f32
      } : tensor<2x3xf32> to tensor<2x3xf32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        source: mlir_python._mlir_python.Value,
        static_low: mlir_python._mlir_python.Attribute,
        static_high: mlir_python._mlir_python.Attribute,
        low: Sequence[mlir_python._mlir_python.Value] = [],
        high: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        nofold: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.pad``: tensor pad operation.

        Args:
            result_type: Type of result ``result`` (ranked tensor of any type values).
            source: Operand ``source`` (ranked tensor of any type values).
            static_low: Attribute ``static_low`` (i64 dense array attribute).
            static_high: Attribute ``static_high`` (i64 dense array attribute).
            low: Operand ``low`` (index). Empty by default.
            high: Operand ``high`` (index). Empty by default.
            nofold: Attribute ``nofold`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: ranked tensor of any type values."""

    @property
    def low(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``low``: index."""

    @property
    def high(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``high``: index."""

    @property
    def static_low(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``static_low``: i64 dense array attribute."""

    @static_low.setter
    def static_low(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def static_high(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``static_high``: i64 dense array attribute."""

    @static_high.setter
    def static_high(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def nofold(self) -> bool:
        """Attribute ``nofold``: unit attribute."""

    @nofold.setter
    def nofold(self, arg: bool, /) -> None: ...
    @property
    def region(self) -> mlir_python._mlir_python.Region:
        """Region ``region``: region with 1 blocks."""

    OPERATION_NAME: str = "tensor.pad"

class ParallelInsertSliceOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.parallel_insert_slice``:
        Specify the tensor slice update of a single thread of a parent
        InParallelOpInterface op.
      .

    The `parallel_insert_slice` yields a subset tensor value to its parent
    InParallelOpInterface. These subset tensor values are aggregated to
    in some unspecified order into a full tensor value returned by the parent
    parallel iterating op.
    The `parallel_insert_slice` is one such op allowed in the
    InParallelOpInterface op.

    Conflicting writes result in undefined semantics, in that the indices written
    to by multiple parallel updates might contain data from any of the updates,
    or even a malformed bit pattern.

    If an index is updated exactly once, the value contained at that index
    in the resulting tensor will be equal to the value at a corresponding index
    of a slice that was used for the updated. If an index is not updated at all,
    its value will be equal to the one in the original tensor.

    This op does not create a new value, which allows maintaining a clean
    separation between the subset and full tensor.

    Note that we cannot mark this operation as pure (Pures), even
    though it has no side effects, because it will get DCEd during
    canonicalization.

    The parallel_insert_slice operation supports the following arguments:

    * source: the tensor that is inserted.
    * dest: the tensor into which the source tensor is inserted.
    * offsets: tensor-rank number of offsets into the `dest` tensor into which
               the slice is inserted.
    * sizes: tensor-rank number of sizes which specify the sizes of the source
             tensor type.
    * strides: tensor-rank number of strides that specify subsampling in each
               dimension.

    The representation based on offsets, sizes and strides support a
    partially-static specification via attributes specified through the
    `static_offsets`, `static_sizes` and `static_strides` arguments. A special
    sentinel value ShapedType::kDynamic encodes that the corresponding entry has
    a dynamic value.

    After buffer allocation, the "parallel_insert_slice" op is expected to lower
    into a memref.subview op.

    A parallel_insert_slice operation may additionally specify insertion into a
    tensor of higher rank than the source tensor, along dimensions that are
    statically known to be of size 1.
    This rank-altering behavior is not required by the op semantics: this
    flexibility allows to progressively drop unit dimensions while lowering
    between different flavors of ops on that operate on tensors.
    The rank-altering behavior of tensor.parallel_insert_slice matches the
    rank-reducing behavior of tensor.insert_slice and tensor.extract_slice.

    #### Verification in the rank-reduced case

    The same verification discussion and mechanisms apply as for ExtractSliceOp.
    Unlike ExtractSliceOp however, there is no need for a specific inference.
    """

    def __init__(
        self,
        source: mlir_python._mlir_python.Value,
        dest: mlir_python._mlir_python.Value,
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
        Create ``tensor.parallel_insert_slice``:
            Specify the tensor slice update of a single thread of a parent
            InParallelOpInterface op.
          .

        Args:
            source: Operand ``source`` (ranked tensor of any type values).
            dest: Operand ``dest`` (ranked tensor of any type values).
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
        """Operand ``source``: ranked tensor of any type values."""

    @property
    def dest(self) -> mlir_python._mlir_python.Value:
        """Operand ``dest``: ranked tensor of any type values."""

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

    OPERATION_NAME: str = "tensor.parallel_insert_slice"

class RankOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.rank``: rank operation.

    The `tensor.rank` operation takes a tensor operand and returns its rank.

    Example:

    ```mlir
    %0 = tensor.rank %arg0 : tensor<*xf32>
    %1 = tensor.rank %arg1 : tensor<?x?xf32>
    ```
    """

    def __init__(
        self,
        tensor: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.rank``: rank operation.

        Result types are inferred.

        Args:
            tensor: Operand ``tensor`` (tensor of any type values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def tensor(self) -> mlir_python._mlir_python.Value:
        """Operand ``tensor``: tensor of any type values."""

    OPERATION_NAME: str = "tensor.rank"

class ReshapeOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.reshape``: tensor reshape operation.

    The `reshape` operation converts a tensor from one type to an equivalent
    type with a provided shape. The source and destination types are compatible
    if both have the same element type, same number of elements. The following
    combinations are possible:

    a. Source type is ranked or unranked. Shape argument has static size.
    Result type is ranked.

    ```mlir
    // Reshape statically-shaped tensor.
    %dst = tensor.reshape %src(%shape)
             : (tensor<4x1xf32>, tensor<1xi32>) -> tensor<4xf32>
    %dst0 = tensor.reshape %src(%shape0)
             : (tensor<4x1xf32>, tensor<2xi32>) -> tensor<2x2xf32>
    // Flatten unranked tensor.
    %dst = tensor.reshape %src(%shape)
             : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
    ```

    b. Source type is ranked or unranked. Shape argument has dynamic size.
    Result type is unranked.

    ```mlir
    // Reshape dynamically-shaped 1D tensor.
    %dst = tensor.reshape %src(%shape)
             : (tensor<?xf32>, tensor<?xi32>) -> tensor<*xf32>
    // Reshape unranked tensor.
    %dst = tensor.reshape %src(%shape)
             : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
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
        Create ``tensor.reshape``: tensor reshape operation.

        Args:
            result_type: Type of result ``result`` (tensor of any type values).
            source: Operand ``source`` (tensor of any type values).
            shape: Operand ``shape`` (1D tensor of signless integer or index values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: tensor of any type values."""

    @property
    def shape(self) -> mlir_python._mlir_python.Value:
        """Operand ``shape``: 1D tensor of signless integer or index values."""

    OPERATION_NAME: str = "tensor.reshape"

class ScatterOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.scatter``: scatter a tensor into a destination tensor at specified indices.

    The `scatter` operation inserts a `source` tensor into a `dest` tensor at
    the given indices.

    In its most general form, the tensor of indices specifies all the coordinates
    of every element to insert (i.e. COO format, without the payload).
    The indices are expected to be confined to coordinate values that fit the
    range of the `dest` tensor, otherwise the behavior is undefined.

    The leading dimensions of the index tensor must match that of the dest
    tensor. The trailing dimensions of the dest tensor must match those of the
    source tensor by omitting the dimensions specified in scatter_dims
    (rank-reducing semantics) or setting them to `1` (rank-preserving semantics)
    (see examples).
    This convention allows an idiomatic specification and lowering of
    "scattering multiple N-D slices into the dest tensor".
    The result type must match the type of the dest tensor.

    Note: in the examples below, we separate out the indexing part of the tensor
    type by a whitespace for readability purposes.

    Example:

    ```mlir
        // For each 1x2 triple of coordinates in %indices, insert the
        // element (i.e. 0-D subset) at the coordinates triple in %dest.
        //
        %out = tensor.scatter %source into %dest[%indices]
            scatter_dims([0, 1, 2]) unique :
          (tensor<1x2x 1x1x1xf32>, tensor<4x4x4xf32>, tensor<1x2x 3xindex>)
            -> tensor<4x4x4xf32>

        // Note: source type may be further rank-reduced to tensor<1x2x f32>.
    ```

    A slice variant is provided to allow specifying insertion of whole tensor
    slices into the `dest` tensor.

    Example:

    ```mlir
        // For each 3 singleton of coordinates in %indices, insert the 2-D
        // slice into %dest[*, %indices[...]:%indices[...] + 1, *] with the
        // indices corresponding to the scatter_dims attribute specified by
        // %indices.
        //
        %out = tensor.scatter %source into %dest[%indices] scatter_dims([1]) unique :
          (tensor<3x 4x1x6xf32>, tensor<4x5x6xf32>, tensor<3x 1xindex>)
            -> tensor<4x5x6xf32>
    ```

    The dimensions specified in the scatter_dims attribute are ones for which the
    source tensor has size `1`.
    I.e. if the dest type is `axbxcxd` and the coordinates are [1, 3], then
    the source type suffix is `ax1xcx1`.
    Scatter also allows rank-reducing semantics where the shape `ax1xcx1` can be
    further simplified to `axc`.

    The elemental type of the indices tensor can be any integer type.
    In the absence of target-specific or problem specific information the default
    type one should use is `index`.

    This operation does not support unranked tensors.

    A `unique` unit attribute must be be specified to indicate that the
    coordinates are statically guaranteed to be unique at runtime. If coordinates
    are not truly unique at runtime, the behavior is undefined.

    Only full slices are meant to be supported by this op, if one desires
    partial slices (e.g. strided windows) one should compose this op with other
    tensor ops (e.g. tensor.insert_slice). This is to avoid a slippery slope of
    complexity that would make the op unusable in practice.

    At the tensor-level, the index tensor is specified in an AoS form (i.e.
    coordinate tuple is the most minor). It is the responsibility of further
    lowerings and bufferization to implement various concrete layouts.

    Note: As currently specified, the operation must lower to an abstraction that
    performs copies to the output tensor. This is because the buffer type system
    is currently not rich enough to allow multiple non-contiguous views in the
    same type. This is visible more clearly in a notional buffer version of the
    op:

    ```mlir
        // memref<?x 4xf32> is a contiguous buffer of ?x4 elements, scatter into
        // random dest slices must copy to the contiguous dest.
        //
        some_side_effecting_op_writing_into %source, ...: memref<3x 4xf32>
        memref.scatter %source into %dest[%indices] scatter_dims([1]) unique :
          (memref<3x 4xf32>, memref<?x 4xf32>, memref<?x 1xindex>)

        // Nested buffer support in the producing op would allow writing directly
        // into the dest buffer.
        %v = some_nested_buffer_view_op %dest[%indices] scatter_dims([1]) unique :
          memref<? x memref<4xf32>>
        some_side_effecting_op_writing_into %v, ...: memref<? x memref<4xf32>>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        source: mlir_python._mlir_python.Value,
        dest: mlir_python._mlir_python.Value,
        indices: mlir_python._mlir_python.Value,
        scatter_dims: mlir_python._mlir_python.Attribute,
        *,
        unique: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.scatter``: scatter a tensor into a destination tensor at specified indices.

        Args:
            result_type: Type of result ``result`` (ranked tensor of any type values).
            source: Operand ``source`` (ranked tensor of any type values).
            dest: Operand ``dest`` (ranked tensor of any type values).
            indices: Operand ``indices`` (ranked tensor of signless integer or index values).
            scatter_dims: Attribute ``scatter_dims`` (i64 dense array attribute).
            unique: Attribute ``unique`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: ranked tensor of any type values."""

    @property
    def dest(self) -> mlir_python._mlir_python.Value:
        """Operand ``dest``: ranked tensor of any type values."""

    @property
    def indices(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``indices``: ranked tensor of signless integer or index values.
        """

    @property
    def scatter_dims(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``scatter_dims``: i64 dense array attribute."""

    @scatter_dims.setter
    def scatter_dims(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def unique(self) -> bool:
        """Attribute ``unique``: unit attribute."""

    @unique.setter
    def unique(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "tensor.scatter"

class SplatOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.splat``: tensor splat or broadcast operation.

    Broadcast the operand to all elements of the result tensor.

    An additional argument of type `index` must be provided for each dynamic
    dimension present in the result type.

    Example for a statically shaped tensor:

    ```mlir
    %s = arith.constant 1.0 : f32
    %t = tensor.splat %s : tensor<8x16xf32>
    ```

    Example for a tensor containing dynamic dimensions:

    ```mlir
    // Broadcasts %s to a 3D dynamically shaped tensor, with %m and %n binding
    // to dimensions 0 and 2 of the resulting tensor, respectively.
    %m = arith.constant 10 : index
    %n = arith.constant 30 : index
    %t = tensor.splat %s[%m, %n] : tensor<?x20x?xf32>
    ```
    """

    def __init__(
        self,
        aggregate_type: mlir_python._mlir_python.Type,
        input: mlir_python._mlir_python.Value,
        dynamic_sizes: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.splat``: tensor splat or broadcast operation.

        Args:
            aggregate_type: Type of result ``aggregate`` (ranked tensor of any type values).
            input: Operand ``input`` (any type).
            dynamic_sizes: Operand ``dynamicSizes`` (index). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def input(self) -> mlir_python._mlir_python.Value:
        """Operand ``input``: any type."""

    @property
    def dynamic_sizes(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``dynamicSizes``: index."""

    @property
    def aggregate(self) -> mlir_python._mlir_python.OpResult:
        """Result ``aggregate``: ranked tensor of any type values."""

    OPERATION_NAME: str = "tensor.splat"

class YieldOp(mlir_python._mlir_python.Operation):
    """
    ``tensor.yield``: Yield a value from a region.

    This operation is used to yield a single value from a within a region. It
    is used to create dynamically sized tensors
    (see `tensor.generate` and `tensor.pad` ops).
    """

    def __init__(
        self,
        value: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``tensor.yield``: Yield a value from a region.

        Args:
            value: Operand ``value`` (any type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: any type."""

    OPERATION_NAME: str = "tensor.yield"
