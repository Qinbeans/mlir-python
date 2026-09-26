"""Typed operations of the MLIR ``arith`` dialect."""

import enum

import mlir_python._mlir_python

class FastMathFlags(enum.Flag):
    """Floating point fast math flags"""

    NONE = 0
    """``none`` in MLIR text."""

    REASSOC = 1
    """``reassoc`` in MLIR text."""

    NNAN = 2
    """``nnan`` in MLIR text."""

    NINF = 4
    """``ninf`` in MLIR text."""

    NSZ = 8
    """``nsz`` in MLIR text."""

    ARCP = 16
    """``arcp`` in MLIR text."""

    CONTRACT = 32
    """``contract`` in MLIR text."""

    AFN = 64
    """``afn`` in MLIR text."""

    FAST = 127
    """``fast`` in MLIR text."""

class IntegerOverflowFlags(enum.Flag):
    """Integer overflow arith flags"""

    NONE = 0
    """``none`` in MLIR text."""

    NSW = 1
    """``nsw`` in MLIR text."""

    NUW = 2
    """``nuw`` in MLIR text."""

class CmpFPredicate(enum.Enum):
    """
    allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    """

    ALWAYS_FALSE = 0
    """``false`` in MLIR text."""

    OEQ = 1
    """``oeq`` in MLIR text."""

    OGT = 2
    """``ogt`` in MLIR text."""

    OGE = 3
    """``oge`` in MLIR text."""

    OLT = 4
    """``olt`` in MLIR text."""

    OLE = 5
    """``ole`` in MLIR text."""

    ONE = 6
    """``one`` in MLIR text."""

    ORD = 7
    """``ord`` in MLIR text."""

    UEQ = 8
    """``ueq`` in MLIR text."""

    UGT = 9
    """``ugt`` in MLIR text."""

    UGE = 10
    """``uge`` in MLIR text."""

    ULT = 11
    """``ult`` in MLIR text."""

    ULE = 12
    """``ule`` in MLIR text."""

    UNE = 13
    """``une`` in MLIR text."""

    UNO = 14
    """``uno`` in MLIR text."""

    ALWAYS_TRUE = 15
    """``true`` in MLIR text."""

class CmpIPredicate(enum.Enum):
    """allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9"""

    EQ = 0
    """``eq`` in MLIR text."""

    NE = 1
    """``ne`` in MLIR text."""

    SLT = 2
    """``slt`` in MLIR text."""

    SLE = 3
    """``sle`` in MLIR text."""

    SGT = 4
    """``sgt`` in MLIR text."""

    SGE = 5
    """``sge`` in MLIR text."""

    ULT = 6
    """``ult`` in MLIR text."""

    ULE = 7
    """``ule`` in MLIR text."""

    UGT = 8
    """``ugt`` in MLIR text."""

    UGE = 9
    """``uge`` in MLIR text."""

class RoundingMode(enum.Enum):
    """Floating point rounding mode"""

    TO_NEAREST_EVEN = 0
    """``to_nearest_even`` in MLIR text."""

    DOWNWARD = 1
    """``downward`` in MLIR text."""

    UPWARD = 2
    """``upward`` in MLIR text."""

    TOWARD_ZERO = 3
    """``toward_zero`` in MLIR text."""

    TO_NEAREST_AWAY = 4
    """``to_nearest_away`` in MLIR text."""

class AtomicRMWKind(enum.Enum):
    """
    allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    """

    ADDF = 0
    """``addf`` in MLIR text."""

    ADDI = 1
    """``addi`` in MLIR text."""

    ANDI = 2
    """``andi`` in MLIR text."""

    ASSIGN = 3
    """``assign`` in MLIR text."""

    MAXIMUMF = 4
    """``maximumf`` in MLIR text."""

    MAXNUMF = 5
    """``maxnumf`` in MLIR text."""

    MAXS = 6
    """``maxs`` in MLIR text."""

    MAXU = 7
    """``maxu`` in MLIR text."""

    MINIMUMF = 8
    """``minimumf`` in MLIR text."""

    MINNUMF = 9
    """``minnumf`` in MLIR text."""

    MINS = 10
    """``mins`` in MLIR text."""

    MINU = 11
    """``minu`` in MLIR text."""

    MULF = 12
    """``mulf`` in MLIR text."""

    MULI = 13
    """``muli`` in MLIR text."""

    ORI = 14
    """``ori`` in MLIR text."""

    XORI = 15
    """``xori`` in MLIR text."""

class AddFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.addf``: floating point addition operation.

    The `addf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar addition.
    %a = arith.addf %b, %c : f64

    // SIMD vector addition, e.g. for Intel SSE.
    %f = arith.addf %g, %h : vector<4xf32>

    // Tensor addition.
    %x = arith.addf %y, %z : tensor<4x?xbf16>
    ```

    TODO: In the distant future, this will accept optional attributes for fast
    math, contraction, rounding mode, and other controls.
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.addf``: floating point addition operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.addf"

class AddIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.addi``: integer addition operation.

    Performs N-bit addition on the operands. The operands are interpreted as
    unsigned bitvectors. The result is represented by a bitvector containing the
    mathematical value of the addition modulo 2^n, where `n` is the bitwidth.
    Because `arith` integers use a two's complement representation, this operation
    is applicable on both signed and unsigned integer operands.

    The `addi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type,
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar addition.
    %a = arith.addi %b, %c : i64

    // Scalar addition with overflow flags.
    %a = arith.addi %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise addition.
    %f = arith.addi %g, %h : vector<4xi32>

    // Tensor element-wise addition.
    %x = arith.addi %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        overflow_flags: IntegerOverflowFlags = IntegerOverflowFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.addi``: integer addition operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            overflow_flags: Attribute ``overflowFlags`` (flags of ``IntegerOverflowFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def overflow_flags(self) -> IntegerOverflowFlags:
        """Attribute ``overflowFlags``: flags of ``IntegerOverflowFlags``."""

    @overflow_flags.setter
    def overflow_flags(self, arg: IntegerOverflowFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.addi"

class AddUIExtendedOp(mlir_python._mlir_python.Operation):
    """
    ``arith.addui_extended``:
        extended unsigned integer addition operation returning sum and overflow bit
      .

    Performs (N+1)-bit addition on zero-extended operands. Returns two results:
    the N-bit sum (same type as both operands), and the overflow bit
    (boolean-like), where `1` indicates unsigned addition overflow, while `0`
    indicates no overflow.

    Example:

    ```mlir
    // Scalar addition.
    %sum, %overflow = arith.addui_extended %b, %c : i64, i1

    // Vector element-wise addition.
    %d:2 = arith.addui_extended %e, %f : vector<4xi32>, vector<4xi1>

    // Tensor element-wise addition.
    %x:2 = arith.addui_extended %y, %z : tensor<4x?xi8>, tensor<4x?xi1>
    ```
    """

    def __init__(
        self,
        sum_type: mlir_python._mlir_python.Type,
        overflow_type: mlir_python._mlir_python.Type,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.addui_extended``:
            extended unsigned integer addition operation returning sum and overflow bit
          .

        Args:
            sum_type: Type of result ``sum`` (signless-integer-like).
            overflow_type: Type of result ``overflow`` (bool-like).
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def sum(self) -> mlir_python._mlir_python.OpResult:
        """Result ``sum``: signless-integer-like."""

    @property
    def overflow(self) -> mlir_python._mlir_python.OpResult:
        """Result ``overflow``: bool-like."""

    OPERATION_NAME: str = "arith.addui_extended"

class AndIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.andi``: integer binary and.

    The `andi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar
    type, a vector whose element type is integer, or a tensor of integers. It
    has no standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise and.
    %a = arith.andi %b, %c : i64

    // SIMD vector element-wise bitwise integer and.
    %f = arith.andi %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer and.
    %x = arith.andi %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.andi``: integer binary and.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.andi"

class BitcastOp(mlir_python._mlir_python.Operation):
    """
    ``arith.bitcast``: bitcast between values of equal bit width.

    Bitcast an integer or floating point value to an integer or floating point
    value of equal bit width. When operating on vectors, casts elementwise.

    Note that this implements a logical bitcast independent of target
    endianness. This allows constant folding without target information and is
    consitent with the bitcast constant folders in LLVM (see
    https://github.com/llvm/llvm-project/blob/18c19414eb/llvm/lib/IR/ConstantFold.cpp#L168)
    For targets where the source and target type have the same endianness (which
    is the standard), this cast will also change no bits at runtime, but it may
    still require an operation, for example if the machine has different
    floating point and integer register files. For targets that have a different
    endianness for the source and target types (e.g. float is big-endian and
    integer is little-endian) a proper lowering would add operations to swap the
    order of words in addition to the bitcast.
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.bitcast``: bitcast between values of equal bit width.

        Args:
            out_type: Type of result ``out`` (signless-integer-or-float-like or memref of signless-integer or float).
            in_: Operand ``in`` (signless-integer-or-float-like or memref of signless-integer or float).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """
        Result ``out``: signless-integer-or-float-like or memref of signless-integer or float.
        """

    OPERATION_NAME: str = "arith.bitcast"

class CeilDivSIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.ceildivsi``: signed ceil integer division operation.

    Signed integer division. Rounds towards positive infinity, i.e. `7 / -2 = -3`.

    Divison by zero, or signed division overflow (minimum value divided by -1)
    is undefined behavior. When applied to `vector` and `tensor` values, the
    behavior is undefined if _any_ of its elements are divided by zero or has a
    signed division overflow.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = arith.ceildivsi %b, %c : i64
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.ceildivsi``: signed ceil integer division operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.ceildivsi"

class CeilDivUIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.ceildivui``: unsigned ceil integer division operation.

    Unsigned integer division. Rounds towards positive infinity. Treats the
    leading bit as the most significant, i.e. for `i16` given two's complement
    representation, `6 / -2 = 6 / (2^16 - 2) = 1`.

    Division by zero is undefined behavior. When applied to `vector` and
    `tensor` values, the behavior is undefined if _any_ elements are divided by
    zero.

    Example:

    ```mlir
    // Scalar unsigned integer division.
    %a = arith.ceildivui %b, %c : i64
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.ceildivui``: unsigned ceil integer division operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.ceildivui"

class CmpFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.cmpf``: floating-point comparison operation.

    The `cmpf` operation compares its two operands according to the float
    comparison rules and the predicate specified by the respective attribute.
    The predicate defines the type of comparison: (un)orderedness, (in)equality
    and signed less/greater than (or equal to) as well as predicates that are
    always true or false.  The operands must have the same type, and this type
    must be a float type, or a vector or tensor thereof.  The result is an i1,
    or a vector/tensor thereof having the same shape as the inputs. Unlike cmpi,
    the operands are always treated as signed. The u prefix indicates
    *unordered* comparison, not unsigned comparison, so "une" means unordered or
    not equal. For the sake of readability by humans, custom assembly form for
    the operation uses a string-typed attribute for the predicate.  The value of
    this attribute corresponds to lower-cased name of the predicate constant,
    e.g., "one" means "ordered not equal".  The string representation of the
    attribute is merely a syntactic sugar and is converted to an integer
    attribute by the parser.

    Example:

    ```mlir
    %r1 = arith.cmpf oeq, %0, %1 : f32
    %r2 = arith.cmpf ult, %0, %1 : tensor<42x42xf64>
    %r3 = "arith.cmpf"(%0, %1) {predicate: 0} : (f8, f8) -> i1
    ```
    """

    def __init__(
        self,
        predicate: CmpFPredicate,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.cmpf``: floating-point comparison operation.

        Result types are inferred.

        Args:
            predicate: Attribute ``predicate`` (a member of ``CmpFPredicate``).
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: floating-point-like."""

    @property
    def predicate(self) -> CmpFPredicate:
        """Attribute ``predicate``: a member of ``CmpFPredicate``."""

    @predicate.setter
    def predicate(self, arg: CmpFPredicate, /) -> None: ...
    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.cmpf"

class CmpIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.cmpi``: integer comparison operation.

    The `cmpi` operation is a generic comparison for integer-like types. Its two
    arguments can be integers, vectors or tensors thereof as long as their types
    match. The operation produces an i1 for the former case, a vector or a
    tensor of i1 with the same shape as inputs in the other cases.

    Its first argument is an attribute that defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   signed less than (mnemonic: `"slt"`; integer value: `2`)
    -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
    -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
    -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
    -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)

    The result is `1` if the comparison is true and `0` otherwise. For vector or
    tensor operands, the comparison is performed elementwise and the element of
    the result indicates whether the comparison is true for the operand elements
    with the same indices as those of the result.

    Note: while the custom assembly form uses strings, the actual underlying
    attribute has integer type (or rather enum class in C++ code) as seen from
    the generic assembly form. String literals are used to improve readability
    of the IR by humans.

    This operation only applies to integer-like operands, but not floats. The
    main reason being that comparison operations have diverging sets of
    attributes: integers require sign specification while floats require various
    floating point-related particularities, e.g., `-ffast-math` behavior,
    IEEE754 compliance, etc
    ([rationale](../Rationale/Rationale.md#splitting-floating-point-vs-integer-operations)).
    The type of comparison is specified as attribute to avoid introducing ten
    similar operations, taking into account that they are often implemented
    using the same operation downstream
    ([rationale](../Rationale/Rationale.md#specifying-comparison-kind-as-attribute)). The
    separation between signed and unsigned order comparisons is necessary
    because of integers being signless. The comparison operation must know how
    to interpret values with the foremost bit being set: negatives in two's
    complement or large positives
    ([rationale](../Rationale/Rationale.md#specifying-sign-in-integer-comparison-operations)).

    Example:

    ```mlir
    // Custom form of scalar "signed less than" comparison.
    %x = arith.cmpi slt, %lhs, %rhs : i32

    // Generic form of the same operation.
    %x = "arith.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

    // Custom form of vector equality comparison.
    %x = arith.cmpi eq, %lhs, %rhs : vector<4xi64>

    // Generic form of the same operation.
    %x = "arith.cmpi"(%lhs, %rhs) {predicate = 0 : i64}
        : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
    ```
    """

    def __init__(
        self,
        predicate: CmpIPredicate,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.cmpi``: integer comparison operation.

        Result types are inferred.

        Args:
            predicate: Attribute ``predicate`` (a member of ``CmpIPredicate``).
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def predicate(self) -> CmpIPredicate:
        """Attribute ``predicate``: a member of ``CmpIPredicate``."""

    @predicate.setter
    def predicate(self, arg: CmpIPredicate, /) -> None: ...

    OPERATION_NAME: str = "arith.cmpi"

class ConstantOp(mlir_python._mlir_python.Operation):
    """
    ``arith.constant``: integer or floating point constant.

    The `constant` operation produces an SSA value equal to some integer or
    floating-point constant specified by an attribute. This is the way MLIR
    forms simple integer and floating point constants.

    Example:

    ```
    // Integer constant
    %1 = arith.constant 42 : i32

    // Equivalent generic form
    %1 = "arith.constant"() {value = 42 : i32} : () -> i32
    ```
    """

    def __init__(
        self,
        value: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.constant``: integer or floating point constant.

        Result types are inferred.

        Args:
            value: Attribute ``value`` (TypedAttr instance).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``value``: TypedAttr instance."""

    @value.setter
    def value(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "arith.constant"

class DivFOp(mlir_python._mlir_python.Operation):
    """``arith.divf``: floating point division operation."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.divf``: floating point division operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.divf"

class DivSIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.divsi``: signed integer division operation.

    Signed integer division. Rounds towards zero. Treats the leading bit as
    sign, i.e. `6 / -2 = -3`.

    Divison by zero, or signed division overflow (minimum value divided by -1)
    is undefined behavior. When applied to `vector` and `tensor` values, the
    behavior is undefined if _any_ of its elements are divided by zero or has a
    signed division overflow.

    If the `exact` attribute is present, the result value is poison if `lhs` is
    not a multiple of `rhs`.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = arith.divsi %b, %c : i64

    // Scalar signed integer division where %b is known to be a multiple of %c.
    %a = arith.divsi %b, %c exact : i64

    // SIMD vector element-wise division.
    %f = arith.divsi %g, %h : vector<4xi32>

    // Tensor element-wise integer division.
    %x = arith.divsi %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        is_exact: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.divsi``: signed integer division operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            is_exact: Attribute ``isExact`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def is_exact(self) -> bool:
        """Attribute ``isExact``: unit attribute."""

    @is_exact.setter
    def is_exact(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "arith.divsi"

class DivUIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.divui``: unsigned integer division operation.

    Unsigned integer division. Rounds towards zero. Treats the leading bit as
    the most significant, i.e. for `i16` given two's complement representation,
    `6 / -2 = 6 / (2^16 - 2) = 0`.

    Division by zero is undefined behavior. When applied to `vector` and
    `tensor` values, the behavior is undefined if _any_ elements are divided by
    zero.

    If the `exact` attribute is present, the result value is poison if `lhs` is
    not a multiple of `rhs`.

    Example:

    ```mlir
    // Scalar unsigned integer division.
    %a = arith.divui %b, %c : i64

    // Scalar unsigned integer division where %b is known to be a multiple of %c.
    %a = arith.divui %b, %c exact : i64

    // SIMD vector element-wise division.
    %f = arith.divui %g, %h : vector<4xi32>

    // Tensor element-wise integer division.
    %x = arith.divui %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        is_exact: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.divui``: unsigned integer division operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            is_exact: Attribute ``isExact`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def is_exact(self) -> bool:
        """Attribute ``isExact``: unit attribute."""

    @is_exact.setter
    def is_exact(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "arith.divui"

class ExtFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.extf``: cast from floating-point to wider floating-point.

    Cast a floating-point value to a larger floating-point-typed value.
    The destination type must to be strictly wider than the source type.
    When operating on vectors, casts elementwise.
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.extf``: cast from floating-point to wider floating-point.

        Args:
            out_type: Type of result ``out`` (floating-point-like).
            in_: Operand ``in`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags | None:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags | None) -> None: ...

    OPERATION_NAME: str = "arith.extf"

class ExtSIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.extsi``: integer sign extension operation.

    The integer sign extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with copies
    of the most-significant bit of the input.

    Example:

    ```mlir
    %1 = arith.constant 5 : i3      // %1 is 0b101
    %2 = arith.extsi %1 : i3 to i6  // %2 is 0b111101
    %3 = arith.constant 2 : i3      // %3 is 0b010
    %4 = arith.extsi %3 : i3 to i6  // %4 is 0b000010

    %5 = arith.extsi %0 : vector<2 x i32> to vector<2 x i64>
    ```
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.extsi``: integer sign extension operation.

        Args:
            out_type: Type of result ``out`` (signless-fixed-width-integer-like).
            in_: Operand ``in`` (signless-fixed-width-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: signless-fixed-width-integer-like."""

    OPERATION_NAME: str = "arith.extsi"

class ExtUIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.extui``: integer zero extension operation.

    The integer zero extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with zeros.

    Example:

    ```mlir
      %1 = arith.constant 5 : i3      // %1 is 0b101
      %2 = arith.extui %1 : i3 to i6  // %2 is 0b000101
      %3 = arith.constant 2 : i3      // %3 is 0b010
      %4 = arith.extui %3 : i3 to i6  // %4 is 0b000010

      %5 = arith.extui %0 : vector<2 x i32> to vector<2 x i64>
    ```
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.extui``: integer zero extension operation.

        Args:
            out_type: Type of result ``out`` (signless-fixed-width-integer-like).
            in_: Operand ``in`` (signless-fixed-width-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: signless-fixed-width-integer-like."""

    OPERATION_NAME: str = "arith.extui"

class FPToSIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.fptosi``: cast from floating-point type to integer type.

    Cast from a value interpreted as floating-point to the nearest (rounding
    towards zero) signed integer value. When operating on vectors, casts
    elementwise.
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.fptosi``: cast from floating-point type to integer type.

        Args:
            out_type: Type of result ``out`` (signless-fixed-width-integer-like).
            in_: Operand ``in`` (floating-point-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: signless-fixed-width-integer-like."""

    OPERATION_NAME: str = "arith.fptosi"

class FPToUIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.fptoui``: cast from floating-point type to integer type.

    Cast from a value interpreted as floating-point to the nearest (rounding
    towards zero) unsigned integer value. When operating on vectors, casts
    elementwise.
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.fptoui``: cast from floating-point type to integer type.

        Args:
            out_type: Type of result ``out`` (signless-fixed-width-integer-like).
            in_: Operand ``in`` (floating-point-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: signless-fixed-width-integer-like."""

    OPERATION_NAME: str = "arith.fptoui"

class FloorDivSIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.floordivsi``: signed floor integer division operation.

    Signed integer division. Rounds towards negative infinity, i.e. `5 / -2 = -3`.

    Divison by zero, or signed division overflow (minimum value divided by -1)
    is undefined behavior. When applied to `vector` and `tensor` values, the
    behavior is undefined if _any_ of its elements are divided by zero or has a
    signed division overflow.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = arith.floordivsi %b, %c : i64

    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.floordivsi``: signed floor integer division operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.floordivsi"

class IndexCastOp(mlir_python._mlir_python.Operation):
    """
    ``arith.index_cast``: cast between index and integer types.

    Casts between scalar or vector integers and corresponding 'index' scalar or
    vectors. Index is an integer of platform-specific bit width. If casting to
    a wider integer, the value is sign-extended. If casting to a narrower
    integer, the value is truncated.
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.index_cast``: cast between index and integer types.

        Args:
            out_type: Type of result ``out`` (signless-integer-like or memref of signless-integer).
            in_: Operand ``in`` (signless-integer-like or memref of signless-integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: signless-integer-like or memref of signless-integer."""

    OPERATION_NAME: str = "arith.index_cast"

class IndexCastUIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.index_castui``: unsigned cast between index and integer types.

    Casts between scalar or vector integers and corresponding 'index' scalar or
    vectors. Index is an integer of platform-specific bit width. If casting to
    a wider integer, the value is zero-extended. If casting to a narrower
    integer, the value is truncated.
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.index_castui``: unsigned cast between index and integer types.

        Args:
            out_type: Type of result ``out`` (signless-integer-like or memref of signless-integer).
            in_: Operand ``in`` (signless-integer-like or memref of signless-integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: signless-integer-like or memref of signless-integer."""

    OPERATION_NAME: str = "arith.index_castui"

class MaxNumFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.maxnumf``: floating-point maximum operation.

    Returns the maximum of the two arguments.
    If the arguments are -0.0 and +0.0, then the result is either of them.
    If one of the arguments is NaN, then the result is the other argument.

    Example:

    ```mlir
    // Scalar floating-point maximum.
    %a = arith.maxnumf %b, %c : f64
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.maxnumf``: floating-point maximum operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.maxnumf"

class MaxSIOp(mlir_python._mlir_python.Operation):
    """``arith.maxsi``: signed integer maximum operation."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.maxsi``: signed integer maximum operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.maxsi"

class MaxUIOp(mlir_python._mlir_python.Operation):
    """``arith.maxui``: unsigned integer maximum operation."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.maxui``: unsigned integer maximum operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.maxui"

class MaximumFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.maximumf``: floating-point maximum operation.

    Returns the maximum of the two arguments, treating -0.0 as less than +0.0.
    If one of the arguments is NaN, then the result is also NaN.

    Example:

    ```mlir
    // Scalar floating-point maximum.
    %a = arith.maximumf %b, %c : f64
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.maximumf``: floating-point maximum operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.maximumf"

class MinNumFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.minnumf``: floating-point minimum operation.

    Returns the minimum of the two arguments.
    If the arguments are -0.0 and +0.0, then the result is either of them.
    If one of the arguments is NaN, then the result is the other argument.

    Example:

    ```mlir
    // Scalar floating-point minimum.
    %a = arith.minnumf %b, %c : f64
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.minnumf``: floating-point minimum operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.minnumf"

class MinSIOp(mlir_python._mlir_python.Operation):
    """``arith.minsi``: signed integer minimum operation."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.minsi``: signed integer minimum operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.minsi"

class MinUIOp(mlir_python._mlir_python.Operation):
    """``arith.minui``: unsigned integer minimum operation."""

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.minui``: unsigned integer minimum operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.minui"

class MinimumFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.minimumf``: floating-point minimum operation.

    Returns the minimum of the two arguments, treating -0.0 as less than +0.0.
    If one of the arguments is NaN, then the result is also NaN.

    Example:

    ```mlir
    // Scalar floating-point minimum.
    %a = arith.minimumf %b, %c : f64
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.minimumf``: floating-point minimum operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.minimumf"

class MulFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.mulf``: floating point multiplication operation.

    The `mulf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar multiplication.
    %a = arith.mulf %b, %c : f64

    // SIMD pointwise vector multiplication, e.g. for Intel SSE.
    %f = arith.mulf %g, %h : vector<4xf32>

    // Tensor pointwise multiplication.
    %x = arith.mulf %y, %z : tensor<4x?xbf16>
    ```

    TODO: In the distant future, this will accept optional attributes for fast
    math, contraction, rounding mode, and other controls.
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.mulf``: floating point multiplication operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.mulf"

class MulIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.muli``:
        Integer multiplication operation.
      .

    Performs N-bit multiplication on the operands. The operands are interpreted as
    unsigned bitvectors. The result is represented by a bitvector containing the
    mathematical value of the multiplication modulo 2^n, where `n` is the bitwidth.
    Because `arith` integers use a two's complement representation, this operation is
    applicable on both signed and unsigned integer operands.

    The `muli` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type,
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar multiplication.
    %a = arith.muli %b, %c : i64

    // Scalar multiplication with overflow flags.
    %a = arith.muli %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise multiplication.
    %f = arith.muli %g, %h : vector<4xi32>

    // Tensor element-wise multiplication.
    %x = arith.muli %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        overflow_flags: IntegerOverflowFlags = IntegerOverflowFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.muli``:
            Integer multiplication operation.
          .

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            overflow_flags: Attribute ``overflowFlags`` (flags of ``IntegerOverflowFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def overflow_flags(self) -> IntegerOverflowFlags:
        """Attribute ``overflowFlags``: flags of ``IntegerOverflowFlags``."""

    @overflow_flags.setter
    def overflow_flags(self, arg: IntegerOverflowFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.muli"

class MulSIExtendedOp(mlir_python._mlir_python.Operation):
    """
    ``arith.mulsi_extended``:
        extended signed integer multiplication operation
      .

    Performs (2*N)-bit multiplication on sign-extended operands. Returns two
    N-bit results: the low and the high halves of the product. The low half has
    the same value as the result of regular multiplication `arith.muli` with
    the same operands.

    Example:

    ```mlir
    // Scalar multiplication.
    %low, %high = arith.mulsi_extended %a, %b : i32

    // Vector element-wise multiplication.
    %c:2 = arith.mulsi_extended %d, %e : vector<4xi32>

    // Tensor element-wise multiplication.
    %x:2 = arith.mulsi_extended %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.mulsi_extended``:
            extended signed integer multiplication operation
          .

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def low(self) -> mlir_python._mlir_python.OpResult:
        """Result ``low``: signless-integer-like."""

    @property
    def high(self) -> mlir_python._mlir_python.OpResult:
        """Result ``high``: signless-integer-like."""

    OPERATION_NAME: str = "arith.mulsi_extended"

class MulUIExtendedOp(mlir_python._mlir_python.Operation):
    """
    ``arith.mului_extended``:
        extended unsigned integer multiplication operation
      .

    Performs (2*N)-bit multiplication on zero-extended operands. Returns two
    N-bit results: the low and the high halves of the product. The low half has
    the same value as the result of regular multiplication `arith.muli` with
    the same operands.

    Example:

    ```mlir
    // Scalar multiplication.
    %low, %high = arith.mului_extended %a, %b : i32

    // Vector element-wise multiplication.
    %c:2 = arith.mului_extended %d, %e : vector<4xi32>

    // Tensor element-wise multiplication.
    %x:2 = arith.mului_extended %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.mului_extended``:
            extended unsigned integer multiplication operation
          .

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def low(self) -> mlir_python._mlir_python.OpResult:
        """Result ``low``: signless-integer-like."""

    @property
    def high(self) -> mlir_python._mlir_python.OpResult:
        """Result ``high``: signless-integer-like."""

    OPERATION_NAME: str = "arith.mului_extended"

class NegFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.negf``: floating point negation.

    The `negf` operation computes the negation of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar negation value.
    %a = arith.negf %b : f64

    // SIMD vector element-wise negation value.
    %f = arith.negf %g : vector<4xf32>

    // Tensor element-wise negation value.
    %x = arith.negf %y : tensor<4x?xf8>
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.negf``: floating point negation.

        Result types are inferred.

        Args:
            operand: Operand ``operand`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.negf"

class OrIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.ori``: integer binary or.

    The `ori` operation takes two operands and returns one result, each of these
    is required to be the same type. This type may be an integer scalar type, a
    vector whose element type is integer, or a tensor of integers. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise or.
    %a = arith.ori %b, %c : i64

    // SIMD vector element-wise bitwise integer or.
    %f = arith.ori %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer or.
    %x = arith.ori %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.ori``: integer binary or.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.ori"

class RemFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.remf``: floating point division remainder operation.

    Returns the floating point division remainder.
    The remainder has the same sign as the dividend (lhs operand).
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.remf``: floating point division remainder operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.remf"

class RemSIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.remsi``: signed integer division remainder operation.

    Signed integer division remainder. Treats the leading bit as sign, i.e. `6 %
    -2 = 0`.

    Division by zero is undefined behavior. When applied to `vector` and
    `tensor` values, the behavior is undefined if _any_ elements are divided by
    zero.

    Example:

    ```mlir
    // Scalar signed integer division remainder.
    %a = arith.remsi %b, %c : i64

    // SIMD vector element-wise division remainder.
    %f = arith.remsi %g, %h : vector<4xi32>

    // Tensor element-wise integer division remainder.
    %x = arith.remsi %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.remsi``: signed integer division remainder operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.remsi"

class RemUIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.remui``: unsigned integer division remainder operation.

    Unsigned integer division remainder. Treats the leading bit as the most
    significant, i.e. for `i16`, `6 % -2 = 6 % (2^16 - 2) = 6`.

    Division by zero is undefined behavior. When applied to `vector` and
    `tensor` values, the behavior is undefined if _any_ elements are divided by
    zero.

    Example:

    ```mlir
    // Scalar unsigned integer division remainder.
    %a = arith.remui %b, %c : i64

    // SIMD vector element-wise division remainder.
    %f = arith.remui %g, %h : vector<4xi32>

    // Tensor element-wise integer division remainder.
    %x = arith.remui %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.remui``: unsigned integer division remainder operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.remui"

class SIToFPOp(mlir_python._mlir_python.Operation):
    """
    ``arith.sitofp``: cast from integer type to floating-point.

    Cast from a value interpreted as a signed integer to the corresponding
    floating-point value. If the value cannot be exactly represented, it is
    rounded using the default rounding mode. When operating on vectors, casts
    elementwise.
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.sitofp``: cast from integer type to floating-point.

        Args:
            out_type: Type of result ``out`` (floating-point-like).
            in_: Operand ``in`` (signless-fixed-width-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: floating-point-like."""

    OPERATION_NAME: str = "arith.sitofp"

class ScalingExtFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.scaling_extf``: Upcasts input floats using provided scales values following OCP MXFP Spec.

    This operation upcasts input floating-point values using provided scale
    values. It expects both scales and the input operand to be of the same shape,
    making the operation elementwise. Scales are usually calculated per block
    following the OCP MXFP spec as described in https://arxiv.org/abs/2310.10537.

    If scales are calculated per block where blockSize != 1, then scales may
    require broadcasting to make this operation elementwise. For example, let's
    say the input is of shape `<dim1 x dim2 x ... dimN>`. Given blockSize != 1 and
    assuming quantization happens on the last axis, the input can be reshaped to
    `<dim1 x dim2 x ... (dimN/blockSize) x blockSize>`. Scales will be calculated
    per block on the last axis. Therefore, scales will be of shape
    `<dim1 x dim2 x ... (dimN/blockSize) x 1>`. Scales could also be of some other
    shape as long as it is broadcast compatible with the input, e.g.,
    `<1 x 1 x ... (dimN/blockSize) x 1>`.

    In this example, before calling into `arith.scaling_extf`, scales must be
    broadcasted to `<dim1 x dim2 x dim3 ... (dimN/blockSize) x blockSize>`. Note
    that there could be multiple quantization axes. Internally,
    `arith.scaling_extf` would perform the following:

    ```mlir
    // Cast scale to result type.
    %0 = arith.truncf %1 : f32 to f8E8M0FNU
    %1 = arith.extf %0 : f8E8M0FNU to f16

    // Cast input to result type.
    %2 = arith.extf %3 : f4E2M1FN to f16

    // Perform scaling
    %3 = arith.mulf %2, %1 : f16
    ```
    It propagates NaN values. Therefore, if either scale or the input element
    contains NaN, then the output element value will also be a NaN.

    Example:

    ```mlir
    // Upcast from f4E2M1FN to f32.
    %a = arith.scaling_extf %b, %c : f4E2M1FN, f8E8M0FNU to f32

    // Element-wise upcast with broadcast (blockSize = 32).
    %f = vector.broadcast %g : vector<1xf8E8M0FNU> to vector<32xf8E8M0FNU>
    %h = arith.scaling_extf %i, %f : vector<32xf4E2M1FN>, vector<32xf8E8M0FNU> to vector<32xbf16>
    ```
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        scale: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.scaling_extf``: Upcasts input floats using provided scales values following OCP MXFP Spec.

        Args:
            out_type: Type of result ``out`` (floating-point-like).
            in_: Operand ``in`` (floating-point-like).
            scale: Operand ``scale`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def scale(self) -> mlir_python._mlir_python.Value:
        """Operand ``scale``: floating-point-like."""

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags | None:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags | None) -> None: ...

    OPERATION_NAME: str = "arith.scaling_extf"

class ScalingTruncFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.scaling_truncf``: Downcasts input floating point values using provided scales values following OCP MXFP Spec.

    This operation downcasts input using the provided scale values. It expects
    both scales and the input operand to be of the same shape and, therefore,
    makes the operation elementwise. Scales are usually calculated per block
    following the OCP MXFP spec as described in https://arxiv.org/abs/2310.10537.
    Users are required to normalize and clamp the scales as necessary before calling
    passing them to this operation.  OCP MXFP spec also does the flushing of denorms
    on the input operand, which should be handled during lowering by passing appropriate
    fastMath flag to this operation.

    If scales are calculated per block where blockSize != 1, scales may require
    broadcasting to make this operation elementwise. For example, let's say the
    input is of shape `<dim1 x dim2 x ... dimN>`. Given blockSize != 1 and
    assuming quantization happens on the last axis, the input can be reshaped to
    `<dim1 x dim2 x ... (dimN/blockSize) x blockSize>`. Scales will be calculated
    per block on the last axis. Therefore, scales will be of shape
    `<dim1 x dim2 x ... (dimN/blockSize) x 1>`. Scales could also be of some other
    shape as long as it is broadcast compatible with the input, e.g.,
    `<1 x 1 x ... (dimN/blockSize) x 1>`.

    In this example, before calling into `arith.scaling_truncf`, scales must be
    broadcasted to `<dim1 x dim2 x dim3 ... (dimN/blockSize) x blockSize>`. Note
    that there could be multiple quantization axes. Internally,
    `arith.scaling_truncf` would perform the following:

    ```mlir
    // Cast scale to input type.
    %0 = arith.truncf %1 : f32 to f8E8M0FNU
    %1 = arith.extf %0 : f8E8M0FNU to f16

    // Perform scaling.
    %3 = arith.divf %2, %1 : f16

    // Cast to result type.
    %4 = arith.truncf %3 : f16 to f4E2M1FN
    ```

    Example:

    ```mlir
    // Downcast from f32 to f4E2M1FN.
    %a = arith.scaling_truncf %b, %c : f32, f8E8M0FNU to f4E2M1FN

    // Element-wise downcast with broadcast (blockSize = 32).
    %f = vector.broadcast %g : vector<1xf8E8M0FNU> to vector<32xf8E8M0FNU>
    %h = arith.scaling_truncf %i, %f : vector<32xbf16>, vector<32xf8E8M0FNU> to vector<32xf4E2M1FN>
    ```
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        scale: mlir_python._mlir_python.Value,
        *,
        roundingmode: RoundingMode | None = None,
        fastmath: FastMathFlags | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.scaling_truncf``: Downcasts input floating point values using provided scales values following OCP MXFP Spec.

        Args:
            out_type: Type of result ``out`` (floating-point-like).
            in_: Operand ``in`` (floating-point-like).
            scale: Operand ``scale`` (floating-point-like).
            roundingmode: Attribute ``roundingmode`` (a member of ``RoundingMode``). Optional.
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def scale(self) -> mlir_python._mlir_python.Value:
        """Operand ``scale``: floating-point-like."""

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: floating-point-like."""

    @property
    def roundingmode(self) -> RoundingMode | None:
        """Attribute ``roundingmode``: a member of ``RoundingMode``."""

    @roundingmode.setter
    def roundingmode(self, arg: RoundingMode | None) -> None: ...
    @property
    def fastmath(self) -> FastMathFlags | None:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags | None) -> None: ...

    OPERATION_NAME: str = "arith.scaling_truncf"

class ShLIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.shli``: integer left-shift.

    The `shli` operation shifts the integer value of the first operand to the left
    by the integer value of the second operand. The second operand is interpreted as
    unsigned. The low order bits are filled with zeros. If the value of the second
    operand is greater or equal than the bitwidth of the first operand, then the
    operation returns poison.

    This op supports `nuw`/`nsw` overflow flags which stands for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    %1 = arith.constant 5 : i8  // %1 is 0b00000101
    %2 = arith.constant 3 : i8
    %3 = arith.shli %1, %2 : i8 // %3 is 0b00101000
    %4 = arith.shli %1, %2 overflow<nsw, nuw> : i8
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        overflow_flags: IntegerOverflowFlags = IntegerOverflowFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.shli``: integer left-shift.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            overflow_flags: Attribute ``overflowFlags`` (flags of ``IntegerOverflowFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def overflow_flags(self) -> IntegerOverflowFlags:
        """Attribute ``overflowFlags``: flags of ``IntegerOverflowFlags``."""

    @overflow_flags.setter
    def overflow_flags(self, arg: IntegerOverflowFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.shli"

class ShRSIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.shrsi``: signed integer right-shift.

    The `shrsi` operation shifts an integer value of the first operand to the right
    by the value of the second operand. The first operand is interpreted as signed,
    and the second operand is interpreter as unsigned. The high order bits in the
    output are filled with copies of the most-significant bit of the shifted value
    (which means that the sign of the value is preserved). If the value of the second
    operand is greater or equal than bitwidth of the first operand, then the operation
    returns poison.

    If the `exact` attribute is present, the result value of shrsi is a poison
    value if any of the bits shifted out are non-zero.

    Example:

    ```mlir
    %1 = arith.constant 160 : i8         // %1 is 0b10100000
    %2 = arith.constant 3 : i8
    %3 = arith.shrsi %1, %2 exact : i8   // %3 is 0b11110100
    %4 = arith.constant 98 : i8          // %4 is 0b01100010
    %5 = arith.shrsi %4, %2 : i8         // %5 is 0b00001100
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        is_exact: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.shrsi``: signed integer right-shift.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            is_exact: Attribute ``isExact`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def is_exact(self) -> bool:
        """Attribute ``isExact``: unit attribute."""

    @is_exact.setter
    def is_exact(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "arith.shrsi"

class ShRUIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.shrui``: unsigned integer right-shift.

    The `shrui` operation shifts an integer value of the first operand to the right
    by the value of the second operand. The first operand is interpreted as unsigned,
    and the second operand is interpreted as unsigned. The high order bits are always
    filled with zeros. If the value of the second operand is greater or equal than the
    bitwidth of the first operand, then the operation returns poison.

    If the `exact` attribute is present, the result value of shrui is a poison
    value if any of the bits shifted out are non-zero.

    Example:

    ```mlir
    %1 = arith.constant 160 : i8        // %1 is 0b10100000
    %2 = arith.constant 3 : i8
    %3 = arith.constant 6 : i8
    %4 = arith.shrui %1, %2 exact : i8  // %4 is 0b00010100
    %5 = arith.shrui %1, %3 : i8        // %3 is 0b00000010
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        is_exact: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.shrui``: unsigned integer right-shift.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            is_exact: Attribute ``isExact`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def is_exact(self) -> bool:
        """Attribute ``isExact``: unit attribute."""

    @is_exact.setter
    def is_exact(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "arith.shrui"

class SubFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.subf``: floating point subtraction operation.

    The `subf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar subtraction.
    %a = arith.subf %b, %c : f64

    // SIMD vector subtraction, e.g. for Intel SSE.
    %f = arith.subf %g, %h : vector<4xf32>

    // Tensor subtraction.
    %x = arith.subf %y, %z : tensor<4x?xbf16>
    ```

    TODO: In the distant future, this will accept optional attributes for fast
    math, contraction, rounding mode, and other controls.
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: FastMathFlags = FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.subf``: floating point subtraction operation.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: floating-point-like."""

    @property
    def fastmath(self) -> FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.subf"

class SubIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.subi``:
        Integer subtraction operation.
      .

    Performs N-bit subtraction on the operands. The operands are interpreted as unsigned
    bitvectors. The result is represented by a bitvector containing the mathematical
    value of the subtraction modulo 2^n, where `n` is the bitwidth. Because `arith`
    integers use a two's complement representation, this operation is applicable on
    both signed and unsigned integer operands.

    The `subi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type,
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar subtraction.
    %a = arith.subi %b, %c : i64

    // Scalar subtraction with overflow flags.
    %a = arith.subi %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise subtraction.
    %f = arith.subi %g, %h : vector<4xi32>

    // Tensor element-wise subtraction.
    %x = arith.subi %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        overflow_flags: IntegerOverflowFlags = IntegerOverflowFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.subi``:
            Integer subtraction operation.
          .

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            overflow_flags: Attribute ``overflowFlags`` (flags of ``IntegerOverflowFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def overflow_flags(self) -> IntegerOverflowFlags:
        """Attribute ``overflowFlags``: flags of ``IntegerOverflowFlags``."""

    @overflow_flags.setter
    def overflow_flags(self, arg: IntegerOverflowFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.subi"

class TruncFOp(mlir_python._mlir_python.Operation):
    """
    ``arith.truncf``: cast from floating-point to narrower floating-point.

    Truncate a floating-point value to a smaller floating-point-typed value.
    The destination type must be strictly narrower than the source type.
    If the value cannot be exactly represented, it is rounded using the
    provided rounding mode or the default one if no rounding mode is provided.
    When operating on vectors, casts elementwise.
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        roundingmode: RoundingMode | None = None,
        fastmath: FastMathFlags | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.truncf``: cast from floating-point to narrower floating-point.

        Args:
            out_type: Type of result ``out`` (floating-point-like).
            in_: Operand ``in`` (floating-point-like).
            roundingmode: Attribute ``roundingmode`` (a member of ``RoundingMode``). Optional.
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: floating-point-like."""

    @property
    def roundingmode(self) -> RoundingMode | None:
        """Attribute ``roundingmode``: a member of ``RoundingMode``."""

    @roundingmode.setter
    def roundingmode(self, arg: RoundingMode | None) -> None: ...
    @property
    def fastmath(self) -> FastMathFlags | None:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(self, arg: FastMathFlags | None) -> None: ...

    OPERATION_NAME: str = "arith.truncf"

class TruncIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.trunci``: integer truncation operation.

    The integer truncation operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be smaller than the input bit-width (N < M).
    The top-most (N - M) bits of the input are discarded.

    This op supports `nuw`/`nsw` overflow flags which stands for "No Unsigned
    Wrap" and "No Signed Wrap", respectively. If the nuw keyword is present,
    and any of the truncated bits are non-zero, the result is a poison value.
    If the nsw keyword is present, and any of the truncated bits are not the
    same as the top bit of the truncation result, the result is a poison value.

    Example:

    ```mlir
      // Scalar truncation.
      %1 = arith.constant 21 : i5     // %1 is 0b10101
      %2 = arith.trunci %1 : i5 to i4 // %2 is 0b0101
      %3 = arith.trunci %1 : i5 to i3 // %3 is 0b101

      // Vector truncation.
      %4 = arith.trunci %0 : vector<2 x i32> to vector<2 x i16>

      // Scalar truncation with overflow flags.
      %5 = arith.trunci %a overflow<nsw, nuw> : i32 to i16
    ```
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        overflow_flags: IntegerOverflowFlags = IntegerOverflowFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.trunci``: integer truncation operation.

        Args:
            out_type: Type of result ``out`` (signless-fixed-width-integer-like).
            in_: Operand ``in`` (signless-fixed-width-integer-like).
            overflow_flags: Attribute ``overflowFlags`` (flags of ``IntegerOverflowFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: signless-fixed-width-integer-like."""

    @property
    def overflow_flags(self) -> IntegerOverflowFlags:
        """Attribute ``overflowFlags``: flags of ``IntegerOverflowFlags``."""

    @overflow_flags.setter
    def overflow_flags(self, arg: IntegerOverflowFlags, /) -> None: ...

    OPERATION_NAME: str = "arith.trunci"

class UIToFPOp(mlir_python._mlir_python.Operation):
    """
    ``arith.uitofp``: cast from unsigned integer type to floating-point.

    Cast from a value interpreted as unsigned integer to the corresponding
    floating-point value. If the value cannot be exactly represented, it is
    rounded using the default rounding mode. When operating on vectors, casts
    elementwise.
    """

    def __init__(
        self,
        out_type: mlir_python._mlir_python.Type,
        in_: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.uitofp``: cast from unsigned integer type to floating-point.

        Args:
            out_type: Type of result ``out`` (floating-point-like).
            in_: Operand ``in`` (signless-fixed-width-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def out(self) -> mlir_python._mlir_python.OpResult:
        """Result ``out``: floating-point-like."""

    OPERATION_NAME: str = "arith.uitofp"

class XOrIOp(mlir_python._mlir_python.Operation):
    """
    ``arith.xori``: integer binary xor.

    The `xori` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar
    type, a vector whose element type is integer, or a tensor of integers. It
    has no standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise xor.
    %a = arith.xori %b, %c : i64

    // SIMD vector element-wise bitwise integer xor.
    %f = arith.xori %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer xor.
    %x = arith.xori %y, %z : tensor<4x?xi8>
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.xori``: integer binary xor.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (signless-integer-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: signless-integer-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    OPERATION_NAME: str = "arith.xori"

class SelectOp(mlir_python._mlir_python.Operation):
    """
    ``arith.select``: select operation.

    The `arith.select` operation chooses one value based on a binary condition
    supplied as its first operand.

    If the value of the first operand (the condition) is `1`, then the second
    operand is returned, and the third operand is ignored, even if it was poison.

    If the value of the first operand (the condition) is `0`, then the third
    operand is returned, and the second operand is ignored, even if it was poison.

    If the value of the first operand (the condition) is poison, then the
    operation returns poison.

    The operation applies to vectors and tensors elementwise given the _shape_
    of all operands is identical. The choice is made for each element
    individually based on the value at the same position as the element in the
    condition operand. If an i1 is provided as the condition, the entire vector
    or tensor is chosen.

    Example:

    ```mlir
    // Custom form of scalar selection.
    %x = arith.select %cond, %true, %false : i32

    // Generic form of the same operation.
    %x = "arith.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

    // Element-wise vector selection.
    %vx = arith.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

    // Full vector selection.
    %vx = arith.select %cond, %vtrue, %vfalse : vector<42xf32>
    ```
    """

    def __init__(
        self,
        condition: mlir_python._mlir_python.Value,
        true_value: mlir_python._mlir_python.Value,
        false_value: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``arith.select``: select operation.

        Result types are inferred.

        Args:
            condition: Operand ``condition`` (bool-like).
            true_value: Operand ``true_value`` (any type).
            false_value: Operand ``false_value`` (any type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def condition(self) -> mlir_python._mlir_python.Value:
        """Operand ``condition``: bool-like."""

    @property
    def true_value(self) -> mlir_python._mlir_python.Value:
        """Operand ``true_value``: any type."""

    @property
    def false_value(self) -> mlir_python._mlir_python.Value:
        """Operand ``false_value``: any type."""

    OPERATION_NAME: str = "arith.select"
