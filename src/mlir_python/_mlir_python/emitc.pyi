"""Typed operations of the MLIR ``emitc`` dialect."""

import enum
from collections.abc import Sequence

import mlir_python._mlir_python

class CmpPredicate(enum.Enum):
    """allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6"""

    EQ = 0
    """``eq`` in MLIR text."""

    NE = 1
    """``ne`` in MLIR text."""

    LT = 2
    """``lt`` in MLIR text."""

    LE = 3
    """``le`` in MLIR text."""

    GT = 4
    """``gt`` in MLIR text."""

    GE = 5
    """``ge`` in MLIR text."""

    THREE_WAY = 6
    """``three_way`` in MLIR text."""

class AddOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.add``: Addition operation.

    With the `emitc.add` operation the arithmetic operator + (addition) can
    be applied.

    Example:

    ```mlir
    // Custom form of the addition operation.
    %0 = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.add %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 + v2;
    float* v6 = v3 + v4;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.add``: Addition operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            lhs: Operand ``lhs`` (type supported by EmitC).
            rhs: Operand ``rhs`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: type supported by EmitC."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.add"

class AddressOfOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.address_of``: Address operation.

    This operation models the C & (address of) operator for a single operand,
    which must be an emitc.lvalue, and returns an emitc pointer to its location.

    Example:

    ```mlir
    // Custom form of applying the & operator.
    %0 = emitc.address_of %arg0 : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        reference: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.address_of``: Address operation.

        Args:
            result_type: Type of result ``result`` (EmitC pointer type).
            reference: Operand ``reference`` (EmitC lvalue type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def reference(self) -> mlir_python._mlir_python.Value:
        """Operand ``reference``: EmitC lvalue type."""

    OPERATION_NAME: str = "emitc.address_of"

class ApplyOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.apply``: Deprecated (use address_of/dereference).

    With the `emitc.apply` operation the operators & (address of) and * (contents of)
    can be applied to a single operand.

    Example:

    ```mlir
    // Custom form of applying the & operator.
    %0 = emitc.apply "&"(%arg0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

    // Generic form of the same operation.
    %0 = "emitc.apply"(%arg0) {applicableOperator = "&"}
        : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        applicable_operator: str,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.apply``: Deprecated (use address_of/dereference).

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            applicable_operator: Attribute ``applicableOperator`` (the operator to apply).
            operand: Operand ``operand`` (type supported by EmitC or EmitC lvalue type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: type supported by EmitC or EmitC lvalue type."""

    @property
    def applicable_operator(self) -> str:
        """Attribute ``applicableOperator``: the operator to apply."""

    @applicable_operator.setter
    def applicable_operator(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "emitc.apply"

class AssignOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.assign``: Assign operation.

    The `emitc.assign` operation stores an SSA value to the location designated by an
    EmitC variable. This operation doesn't return any value. The assigned value
    must be of the same type as the variable being assigned. The operation is
    emitted as a C/C++ '=' operator.

    Example:

    ```mlir
    // Integer variable
    %0 = "emitc.variable"(){value = 42 : i32} : () -> !emitc.lvalue<i32>
    %1 = emitc.call_opaque "foo"() : () -> (i32)

    // Assign emitted as `... = ...;`
    "emitc.assign"(%0, %1) : (!emitc.lvalue<i32>, i32) -> ()
    ```
    """

    def __init__(
        self,
        var: mlir_python._mlir_python.Value,
        value: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.assign``: Assign operation.

        Args:
            var: Operand ``var`` (EmitC lvalue type).
            value: Operand ``value`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def var(self) -> mlir_python._mlir_python.Value:
        """Operand ``var``: EmitC lvalue type."""

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.assign"

class BitwiseAndOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.bitwise_and``: Bitwise and operation.

    With the `emitc.bitwise_and` operation the bitwise operator & (and) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_and %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 & v2;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.bitwise_and``: Bitwise and operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            lhs: Operand ``lhs`` (type supported by EmitC).
            rhs: Operand ``rhs`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: type supported by EmitC."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.bitwise_and"

class BitwiseLeftShiftOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.bitwise_left_shift``: Bitwise left shift operation.

    With the `emitc.bitwise_left_shift` operation the bitwise operator <<
    (left shift) can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_left_shift %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 << v2;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.bitwise_left_shift``: Bitwise left shift operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            lhs: Operand ``lhs`` (type supported by EmitC).
            rhs: Operand ``rhs`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: type supported by EmitC."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.bitwise_left_shift"

class BitwiseNotOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.bitwise_not``: Bitwise not operation.

    With the `emitc.bitwise_not` operation the bitwise operator ~ (not) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_not %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = ~v1;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        operand_0: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.bitwise_not``: Bitwise not operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            operand_0: Operand ``operand_0`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand_0(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand_0``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.bitwise_not"

class BitwiseOrOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.bitwise_or``: Bitwise or operation.

    With the `emitc.bitwise_or` operation the bitwise operator | (or)
    can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_or %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 | v2;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.bitwise_or``: Bitwise or operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            lhs: Operand ``lhs`` (type supported by EmitC).
            rhs: Operand ``rhs`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: type supported by EmitC."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.bitwise_or"

class BitwiseRightShiftOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.bitwise_right_shift``: Bitwise right shift operation.

    With the `emitc.bitwise_right_shift` operation the bitwise operator >>
    (right shift) can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_right_shift %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 >> v2;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.bitwise_right_shift``: Bitwise right shift operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            lhs: Operand ``lhs`` (type supported by EmitC).
            rhs: Operand ``rhs`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: type supported by EmitC."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.bitwise_right_shift"

class BitwiseXorOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.bitwise_xor``: Bitwise xor operation.

    With the `emitc.bitwise_xor` operation the bitwise operator ^ (xor)
    can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_xor %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 ^ v2;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.bitwise_xor``: Bitwise xor operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            lhs: Operand ``lhs`` (type supported by EmitC).
            rhs: Operand ``rhs`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: type supported by EmitC."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.bitwise_xor"

class CallOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.call``: Call operation.

    The `emitc.call` operation represents a direct call to an `emitc.func`
    that is within the same symbol scope as the call. The operands and result type
    of the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = emitc.call @my_add(%0, %1) : (f32, f32) -> f32
    ```
    """

    def __init__(
        self,
        callee: str,
        operands: Sequence[mlir_python._mlir_python.Value] = [],
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        arg_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.call``: Call operation.

        Args:
            callee: Attribute ``callee`` (flat symbol reference attribute).
            operands: Operand ``operands`` (type supported by EmitC). Empty by default.
            result_types: Type of result ``result`` (type supported by EmitC). Empty by default.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def callee(self) -> str:
        """Attribute ``callee``: flat symbol reference attribute."""

    @callee.setter
    def callee(self, arg: str, /) -> None: ...
    @property
    def arg_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``arg_attrs``: Array of dictionary attributes."""

    @arg_attrs.setter
    def arg_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def res_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``res_attrs``: Array of dictionary attributes."""

    @res_attrs.setter
    def res_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...

    OPERATION_NAME: str = "emitc.call"

class CallOpaqueOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.call_opaque``: Opaque call operation.

    The `emitc.call_opaque` operation represents a C++ function call. The callee
    can be an arbitrary non-empty string. The call allows specifying order
    of operands and attributes in the call as follows:

    - integer value of index type refers to an operand;
    - attribute which will get lowered to constant value in call;

    Example:

    ```mlir
    // Custom form defining a call to `foo()`.
    %0 = emitc.call_opaque "foo" () : () -> i32

    // Generic form of the same operation.
    %0 = "emitc.call_opaque"() {callee = "foo"} : () -> i32
    ```
    """

    def __init__(
        self,
        callee: str,
        operands: Sequence[mlir_python._mlir_python.Value] = [],
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        args: mlir_python._mlir_python.ArrayAttr | None = None,
        template_args: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.call_opaque``: Opaque call operation.

        Args:
            callee: Attribute ``callee`` (the C++ function to call).
            operands: Operand ``operands`` (type supported by EmitC). Empty by default.
            result_types: Type of result ``result`` (type supported by EmitC). Empty by default.
            args: Attribute ``args`` (the order of operands and further attributes). Optional.
            template_args: Attribute ``template_args`` (template arguments). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def callee(self) -> str:
        """Attribute ``callee``: the C++ function to call."""

    @callee.setter
    def callee(self, arg: str, /) -> None: ...
    @property
    def args(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``args``: the order of operands and further attributes."""

    @args.setter
    def args(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def template_args(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``template_args``: template arguments."""

    @template_args.setter
    def template_args(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...

    OPERATION_NAME: str = "emitc.call_opaque"

class CastOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.cast``: Cast operation.

    The `emitc.cast` operation performs an explicit type conversion and is emitted
    as a C-style cast expression. It can be applied to integer, float, index
    and EmitC types.

    Example:

    ```mlir
    // Cast from `int32_t` to `float`
    %0 = emitc.cast %arg0: i32 to f32

    // Cast from `void` to `int32_t` pointer
    %1 = emitc.cast %arg1 :
        !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
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
        Create ``emitc.cast``: Cast operation.

        Args:
            dest_type: Type of result ``dest`` (type supported by EmitC).
            source: Operand ``source`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def source(self) -> mlir_python._mlir_python.Value:
        """Operand ``source``: type supported by EmitC."""

    @property
    def dest(self) -> mlir_python._mlir_python.OpResult:
        """Result ``dest``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.cast"

class ClassOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.class``: Represents a C++ class definition, encapsulating fields and methods..

    The `emitc.class` operation defines a C++ class, acting as a container
    for its data fields (`emitc.field`) and methods (`emitc.func`).
    It creates a distinct scope, isolating its contents from the surrounding
    MLIR region, similar to how C++ classes encapsulate their internals.

    Example:

    ```mlir
    emitc.class @modelClass {
      emitc.field @fieldName0 : !emitc.array<1xf32> = {emitc.opaque = "input_tensor"}
      emitc.func @execute() {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = get_field @fieldName0 : !emitc.array<1xf32>
        %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
        return
      }
    }
    // Class with a final specifer
    emitc.class final @modelClass {
      emitc.field @fieldName0 : !emitc.array<1xf32> = {emitc.opaque = "input_tensor"}
      emitc.func @execute() {
        %0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
        %1 = get_field @fieldName0 : !emitc.array<1xf32>
        %2 = subscript %1[%0] : (!emitc.array<1xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
        return
      }
    }
    ```
    """

    def __init__(
        self,
        sym_name: str,
        *,
        final_specifier: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.class``: Represents a C++ class definition, encapsulating fields and methods..

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            final_specifier: Attribute ``final_specifier`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def final_specifier(self) -> bool:
        """Attribute ``final_specifier``: unit attribute."""

    @final_specifier.setter
    def final_specifier(self, arg: bool, /) -> None: ...
    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: any region."""

    OPERATION_NAME: str = "emitc.class"

class CmpOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.cmp``: Comparison operation.

    With the `emitc.cmp` operation the comparison operators ==, !=, <, <=, >, >=, <=>
    can be applied.

    Its first argument is an attribute that defines the comparison operator:

    - equal to (mnemonic: `"eq"`; integer value: `0`)
    - not equal to (mnemonic: `"ne"`; integer value: `1`)
    - less than (mnemonic: `"lt"`; integer value: `2`)
    - less than or equal to (mnemonic: `"le"`; integer value: `3`)
    - greater than (mnemonic: `"gt"`; integer value: `4`)
    - greater than or equal to (mnemonic: `"ge"`; integer value: `5`)
    - three-way-comparison (mnemonic: `"three_way"`; integer value: `6`)

    Example:
    ```mlir
    // Custom form of the cmp operation.
    %0 = emitc.cmp eq, %arg0, %arg1 : (i32, i32) -> i1
    %1 = emitc.cmp lt, %arg2, %arg3 :
        (
          !emitc.opaque<"std::valarray<float>">,
          !emitc.opaque<"std::valarray<float>">
        ) -> !emitc.opaque<"std::valarray<bool>">
    ```
    ```c++
    // Code emitted for the operations above.
    bool v5 = v1 == v2;
    std::valarray<bool> v6 = v3 < v4;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        predicate: CmpPredicate,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.cmp``: Comparison operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            predicate: Attribute ``predicate`` (a member of ``CmpPredicate``).
            lhs: Operand ``lhs`` (type supported by EmitC).
            rhs: Operand ``rhs`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: type supported by EmitC."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: type supported by EmitC."""

    @property
    def predicate(self) -> CmpPredicate:
        """Attribute ``predicate``: a member of ``CmpPredicate``."""

    @predicate.setter
    def predicate(self, arg: CmpPredicate, /) -> None: ...

    OPERATION_NAME: str = "emitc.cmp"

class ConditionalOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.conditional``: Conditional (ternary) operation.

    With the `emitc.conditional` operation the ternary conditional operator can
    be applied.

    Example:

    ```mlir
    %0 = emitc.cmp gt, %arg0, %arg1 : (i32, i32) -> i1

    %c0 = "emitc.constant"() {value = 10 : i32} : () -> i32
    %c1 = "emitc.constant"() {value = 11 : i32} : () -> i32

    %1 = emitc.conditional %0, %c0, %c1 : i32
    ```
    ```c++
    // Code emitted for the operations above.
    bool v3 = v1 > v2;
    int32_t v4 = 10;
    int32_t v5 = 11;
    int32_t v6 = v3 ? v4 : v5;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        condition: mlir_python._mlir_python.Value,
        true_value: mlir_python._mlir_python.Value,
        false_value: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.conditional``: Conditional (ternary) operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            condition: Operand ``condition`` (1-bit signless integer).
            true_value: Operand ``true_value`` (type supported by EmitC).
            false_value: Operand ``false_value`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def condition(self) -> mlir_python._mlir_python.Value:
        """Operand ``condition``: 1-bit signless integer."""

    @property
    def true_value(self) -> mlir_python._mlir_python.Value:
        """Operand ``true_value``: type supported by EmitC."""

    @property
    def false_value(self) -> mlir_python._mlir_python.Value:
        """Operand ``false_value``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.conditional"

class ConstantOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.constant``: Constant operation.

    The `emitc.constant` operation produces an SSA value equal to some constant
    specified by an attribute. This can be used to form simple integer and
    floating point constants, as well as more exotic things like tensor
    constants. The `emitc.constant` operation also supports the EmitC opaque
    attribute and the EmitC opaque type. Since folding is supported,
    it should not be used with pointers.

    Example:

    ```mlir
    // Integer constant
    %0 = "emitc.constant"(){value = 42 : i32} : () -> i32

    // Constant emitted as `char = CHAR_MIN;`
    %1 = "emitc.constant"() {value = #emitc.opaque<"CHAR_MIN">}
      : () -> !emitc.opaque<"char">
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        value: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.constant``: Constant operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            value: Attribute ``value`` (An opaque attribute or TypedAttr instance).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``value``: An opaque attribute or TypedAttr instance."""

    @value.setter
    def value(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "emitc.constant"

class DeclareFuncOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.declare_func``: An operation to declare a function.

    The `emitc.declare_func` operation allows to insert a function declaration for an
    `emitc.func` at a specific position. The operation only requires the "callee"
    of the `emitc.func` to be specified as an attribute.

    Example:

    ```mlir
    emitc.declare_func @bar
    emitc.func @foo(%arg0: i32) -> i32 {
      %0 = emitc.call @bar(%arg0) : (i32) -> (i32)
      emitc.return %0 : i32
    }

    emitc.func @bar(%arg0: i32) -> i32 {
      emitc.return %arg0 : i32
    }
    ```

    ```c++
    // Code emitted for the operations above.
    int32_t bar(int32_t v1);
    int32_t foo(int32_t v1) {
      int32_t v2 = bar(v1);
      return v2;
    }

    int32_t bar(int32_t v1) {
      return v1;
    }
    ```
    """

    def __init__(
        self,
        sym_name: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.declare_func``: An operation to declare a function.

        Args:
            sym_name: Attribute ``sym_name`` (flat symbol reference attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: flat symbol reference attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "emitc.declare_func"

class DereferenceOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.dereference``: Dereference operation.

    This operation models the C * (dereference) operator, which must be of
    !emitc.ptr<> type, returning an !emitc.lvalue<> the value pointed to by the
    pointer.

    Example:

    ```mlir
    // Custom form of the dereference operator.
    %0 = emitc.dereference %arg0 : (!emitc.ptr<i32>) -> !emitc.lvalue<i32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        pointer: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.dereference``: Dereference operation.

        Args:
            result_type: Type of result ``result`` (EmitC lvalue type).
            pointer: Operand ``pointer`` (EmitC pointer type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def pointer(self) -> mlir_python._mlir_python.Value:
        """Operand ``pointer``: EmitC pointer type."""

    OPERATION_NAME: str = "emitc.dereference"

class DivOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.div``: Division operation.

    With the `emitc.div` operation the arithmetic operator / (division) can
    be applied.

    Example:

    ```mlir
    // Custom form of the division operation.
    %0 = emitc.div %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.div %arg2, %arg3 : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 / v2;
    float v6 = v3 / v4;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        operand_0: mlir_python._mlir_python.Value,
        operand_1: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.div``: Division operation.

        Args:
            result_type: Type of result ``result`` (floating-point type supported by EmitC or integer, index or opaque type supported by EmitC).
            operand_0: Operand ``operand_0`` (floating-point type supported by EmitC or integer, index or opaque type supported by EmitC).
            operand_1: Operand ``operand_1`` (floating-point type supported by EmitC or integer, index or opaque type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand_0(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand_0``: floating-point type supported by EmitC or integer, index or opaque type supported by EmitC.
        """

    @property
    def operand_1(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand_1``: floating-point type supported by EmitC or integer, index or opaque type supported by EmitC.
        """

    OPERATION_NAME: str = "emitc.div"

class DoOp(mlir_python._mlir_python.Operation):
    r"""
    ``emitc.do``: Do-while operation.

    The `emitc.do` operation represents a C/C++ do-while loop construct that
    repeatedly executes a body region as long as a condition region evaluates to
    true. The operation has two regions:

    1. A body region that contains the loop body
    2. A condition region that must yield a boolean value (i1)

    The condition is evaluated before each iteration as follows:
    - The condition region must contain exactly one block with:
      1. An `emitc.expression` operation producing an i1 value
      2. An `emitc.yield` passing through the expression result
    - The expression's body contains the actual condition logic

    The body region is executed before the first evaluation of the
    condition. Thus, there is a guarantee that the loop will be executed
    at least once. The loop terminates when the condition yields false.

    The canonical structure of `emitc.do` is:

    ```mlir
    emitc.do {
      // Body region (no terminator required).
      // Loop body operations...
    } while {
      // Condition region (must yield i1)
      %condition = emitc.expression : () -> i1 {
        // Condition computation...
        %result = ... : i1  // Last operation must produce i1
        emitc.yield %result : i1
      }
      // Forward expression result
      emitc.yield %condition : i1
    }
    ```

    Example:

    ```mlir
    emitc.func @do_example() {
      %counter = "emitc.variable"() <{value = 0 : i32}> : () -> !emitc.lvalue<i32>
      %end = emitc.literal "10" : i32
      %step = emitc.literal "1" : i32

      emitc.do {
        // Print current value
        %val = emitc.load %counter : !emitc.lvalue<i32>
        emitc.verbatim "printf(\"%d\\n\", {});" args %val : i32

        // Increment counter
        %new_val = emitc.add %val, %step : (i32, i32) -> i32
        "emitc.assign"(%counter, %new_val) : (!emitc.lvalue<i32>, i32) -> ()
      } while {
        %condition = emitc.expression %counter, %end : (!emitc.lvalue<i32>, i32) -> i1 {
          %current = emitc.load %counter : !emitc.lvalue<i32>
          %cmp_res = emitc.cmp lt, %current, %end : (i32, i32) -> i1
          emitc.yield %cmp_res : i1
        }
        emitc.yield %condition : i1
      }
      return
    }
    ```
    ```c++
    // Code emitted for the operation above.
    void do_example() {
      int32_t v1 = 0;
      do {
        int32_t v2 = v1;
        printf("%d\n", v2);
        int32_t v3 = v2 + 1;
        v1 = v3;
      } while (v1 < 10);
      return;
    }
    ```
    """

    def __init__(
        self,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """Create ``emitc.do``: Do-while operation."""

    @property
    def body_region(self) -> mlir_python._mlir_python.Region:
        """Region ``bodyRegion``: region with 1 blocks."""

    @property
    def condition_region(self) -> mlir_python._mlir_python.Region:
        """Region ``conditionRegion``: region with 1 blocks."""

    OPERATION_NAME: str = "emitc.do"

class ExpressionOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.expression``: Expression operation.

    The `emitc.expression` operation returns a single SSA value which is yielded by
    its single-basic-block region. The operation takes zero or more input operands
    that are passed as block arguments to the region.

    As the operation is to be emitted as a C expression, the operations within
    its body must form a single Def-Use tree, or a DAG trivially expandable to
    one, i.e. a DAG where each operation with side effects is only reachable
    once from the expression root.

    Input operands can be of both value types (`EmitCType`) and lvalue types
    (`EmitC_LValueType`).

    Example:
    ```mlir
    %r = emitc.expression %a, %b, %c : (i32, i32, i32) -> i32 {
      %0 = emitc.call_opaque "foo"(%a) : (i32) -> i32
      %1 = emitc.add %b, %c : (i32, i32) -> i32
      %2 = emitc.mul %0, %1 : (i32, i32) -> i32
      emitc.yield %2 : i32
    }
    ```

    May be emitted as:
    ```c++
    int32_t v4 = foo(v1) * (v2 + v3);
    ```

    When specified, the optional `noinline` indicates that the expression is
    to be emitted as seen above, i.e. as the rhs of an EmitC SSA value
    definition. Otherwise, the expression may be emitted inline, i.e. directly
    at its use.
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        defs: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        do_not_inline: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.expression``: Expression operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            defs: Operand ``defs`` (type supported by EmitC or EmitC lvalue type). Empty by default.
            do_not_inline: Attribute ``do_not_inline`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def defs(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``defs``: type supported by EmitC or EmitC lvalue type."""

    @property
    def do_not_inline(self) -> bool:
        """Attribute ``do_not_inline``: unit attribute."""

    @do_not_inline.setter
    def do_not_inline(self, arg: bool, /) -> None: ...
    @property
    def region(self) -> mlir_python._mlir_python.Region:
        """Region ``region``: region with 1 blocks."""

    OPERATION_NAME: str = "emitc.expression"

class FieldOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.field``: A field within a class.

    The `emitc.field` operation declares a named field within an `emitc.class`
    operation. The field's type must be an EmitC type.

    Example:

    ```mlir
    // Example with an attribute:
    emitc.field @fieldName0 : !emitc.array<1xf32>  {emitc.opaque = "another_feature"}
    // Example with no attribute:
    emitc.field @fieldName0 : !emitc.array<1xf32>
    // Example with an initial value:
    emitc.field @fieldName0 : !emitc.array<1xf32> = dense<0.0>
    // Example with an initial value and attributes:
    emitc.field @fieldName0 : !emitc.array<1xf32> = dense<0.0> {
      emitc.opaque = "input_tensor"}
    ```
    """

    def __init__(
        self,
        sym_name: str,
        type: mlir_python._mlir_python.Type,
        *,
        initial_value: mlir_python._mlir_python.Attribute | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.field``: A field within a class.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            type: Attribute ``type`` (any type attribute).
            initial_value: Attribute ``initial_value`` (An opaque attribute or TypedAttr instance). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def type(self) -> mlir_python._mlir_python.Type:
        """Attribute ``type``: any type attribute."""

    @type.setter
    def type(self, arg: mlir_python._mlir_python.Type, /) -> None: ...
    @property
    def initial_value(self) -> mlir_python._mlir_python.Attribute | None:
        """
        Attribute ``initial_value``: An opaque attribute or TypedAttr instance.
        """

    @initial_value.setter
    def initial_value(self, arg: mlir_python._mlir_python.Attribute | None) -> None: ...

    OPERATION_NAME: str = "emitc.field"

class FileOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.file``: A file container operation.

    A `file` represents a single C/C++ file.

    `mlir-translate` ignores the body of all `emitc.file` ops
    unless the `-file-id=id` flag is used. With that flag, all `emitc.file` ops
    with matching id are emitted.

    Example:

    ```mlir
    emitc.file "main" {
      emitc.func @func_one() {
        emitc.return
      }
    }
    ```
    """

    def __init__(
        self,
        id: mlir_python._mlir_python.StringAttr,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.file``: A file container operation.

        Args:
            id: Attribute ``id`` (An Attribute containing a string).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def id(self) -> mlir_python._mlir_python.StringAttr:
        """Attribute ``id``: An Attribute containing a string."""

    @id.setter
    def id(self, arg: mlir_python._mlir_python.StringAttr, /) -> None: ...
    @property
    def body_region(self) -> mlir_python._mlir_python.Region:
        """Region ``bodyRegion``: region with 1 blocks."""

    OPERATION_NAME: str = "emitc.file"

class ForOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.for``: For operation.

    The `emitc.for` operation represents a C loop of the following form:

    ```c++
    for (T i = lb; i < ub; i += step) { /* ... */ } // where T is typeof(lb)
    ```

    The operation takes 3 SSA values as operands that represent the lower bound,
    upper bound and step respectively, and defines an SSA value for its
    induction variable. It has one region capturing the loop body. The induction
    variable is represented as an argument of this region. This SSA value is a
    signless integer, or an index. The step is a value of same type.

    This operation has no result. The body region must contain exactly one block
    that terminates with `emitc.yield`. Calling ForOp::build will create such a
    region and insert the terminator implicitly if none is defined, so will the
    parsing even in cases when it is absent from the custom format. For example:

    ```mlir
    // Index case.
    emitc.for %iv = %lb to %ub step %step {
      ... // body
    }
    ...
    // Integer case.
    emitc.for %iv_32 = %lb_32 to %ub_32 step %step_32 : i32 {
      ... // body
    }
    ```
    """

    def __init__(
        self,
        lower_bound: mlir_python._mlir_python.Value,
        upper_bound: mlir_python._mlir_python.Value,
        step: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.for``: For operation.

        Args:
            lower_bound: Operand ``lowerBound`` (integer, index or opaque type supported by EmitC).
            upper_bound: Operand ``upperBound`` (integer, index or opaque type supported by EmitC).
            step: Operand ``step`` (integer, index or opaque type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lower_bound(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``lowerBound``: integer, index or opaque type supported by EmitC.
        """

    @property
    def upper_bound(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``upperBound``: integer, index or opaque type supported by EmitC.
        """

    @property
    def step(self) -> mlir_python._mlir_python.Value:
        """Operand ``step``: integer, index or opaque type supported by EmitC."""

    @property
    def region(self) -> mlir_python._mlir_python.Region:
        """Region ``region``: region with 1 blocks."""

    OPERATION_NAME: str = "emitc.for"

class FuncOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.func``: An operation with a name containing a single `SSACFG` region.

    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). While the MLIR textual form provides a nice
    inline syntax for function arguments, they are internally represented as
    “block arguments” to the first block in the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

    Example:

    ```mlir
    // A function with no results:
    emitc.func @foo(%arg0 : i32) {
      emitc.call_opaque "bar" (%arg0) : (i32) -> ()
      emitc.return
    }

    // A function with its argument as single result:
    emitc.func @foo(%arg0 : i32) -> i32 {
      emitc.return %arg0 : i32
    }

    // A function with specifiers attribute:
    emitc.func @example_specifiers_fn_attr() -> i32
                attributes {specifiers = ["static","inline"]} {
      %0 = emitc.call_opaque "foo" (): () -> i32
      emitc.return %0 : i32
    }

    // An external function definition:
    emitc.func private @extern_func(i32)
                        attributes {specifiers = ["extern"]}
    ```
    """

    def __init__(
        self,
        sym_name: str,
        function_type: mlir_python._mlir_python.FunctionType,
        *,
        specifiers: mlir_python._mlir_python.ArrayAttr | None = None,
        arg_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.func``: An operation with a name containing a single `SSACFG` region.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            function_type: Attribute ``function_type`` (type attribute of function type).
            specifiers: Attribute ``specifiers`` (string array attribute). Optional.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def function_type(self) -> mlir_python._mlir_python.FunctionType:
        """Attribute ``function_type``: type attribute of function type."""

    @function_type.setter
    def function_type(self, arg: mlir_python._mlir_python.FunctionType, /) -> None: ...
    @property
    def specifiers(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``specifiers``: string array attribute."""

    @specifiers.setter
    def specifiers(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def arg_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``arg_attrs``: Array of dictionary attributes."""

    @arg_attrs.setter
    def arg_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def res_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``res_attrs``: Array of dictionary attributes."""

    @res_attrs.setter
    def res_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: any region."""

    OPERATION_NAME: str = "emitc.func"

class GetFieldOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.get_field``: Obtain access to a field within a class instance.

    The `emitc.get_field` operation retrieves the lvalue of a
    named field from a given class instance.

    Example:

    ```mlir
    %0 = get_field @fieldName0 : !emitc.array<1xf32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        field_name: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.get_field``: Obtain access to a field within a class instance.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            field_name: Attribute ``field_name`` (flat symbol reference attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def field_name(self) -> str:
        """Attribute ``field_name``: flat symbol reference attribute."""

    @field_name.setter
    def field_name(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "emitc.get_field"

class GetGlobalOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.get_global``: Obtain access to a global variable.

    The `emitc.get_global` operation retrieves the lvalue of a
    named global variable. If the global variable is marked constant, assigning
    to that lvalue is undefined.

    Example:

    ```mlir
    %x = emitc.get_global @foo : !emitc.array<2xf32>
    %y = emitc.get_global @bar : !emitc.lvalue<i32>
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
        Create ``emitc.get_global``: Obtain access to a global variable.

        Args:
            result_type: Type of result ``result`` (EmitC array type or EmitC lvalue type).
            name: Attribute ``name`` (flat symbol reference attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "emitc.get_global"

class GlobalOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.global``: A global variable.

    The `emitc.global` operation declares or defines a named global variable.
    The backing memory for the variable is allocated statically and described by
    the variable's type, which must be an EmitC type.
    Optionally, an `initial_value` can be provided.
    Internal linkage can be specified using the `static_specifier` unit attribute
    and external linkage can be specified using the `extern_specifier` unit attribute.
    Note that the default linkage without those two keywords depends on whether
    the target is C or C++ and whether the global variable is `const`.
    The global variable can also be marked constant using the `const_specifier`
    unit attribute. Writing to such constant global variables is
    undefined.

    The global variable can be accessed by using the `emitc.get_global` to
    retrieve the value for the global variable.

    Example:

    ```mlir
    // Global variable with an initial value.
    emitc.global @x : !emitc.array<2xf32> = dense<0.0>
    // Global variable with an initial values.
    emitc.global @x : !emitc.array<3xi32> = dense<[0, 1, 2]>
    // Global variable with an opaque initial value.
    emitc.global @x : !emitc.opaque<"char"> = #emitc.opaque<"CHAR_MIN">
    // External global variable
    emitc.global extern @x : !emitc.array<2xf32>
    // Constant global variable with internal linkage
    emitc.global static const @x : i32 = 0
    ```
    """

    def __init__(
        self,
        sym_name: str,
        type: mlir_python._mlir_python.Type,
        *,
        initial_value: mlir_python._mlir_python.Attribute | None = None,
        extern_specifier: bool = False,
        static_specifier: bool = False,
        const_specifier: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.global``: A global variable.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            type: Attribute ``type`` (any type attribute).
            initial_value: Attribute ``initial_value`` (An opaque attribute or TypedAttr instance). Optional.
            extern_specifier: Attribute ``extern_specifier`` (unit attribute). Omit for the default.
            static_specifier: Attribute ``static_specifier`` (unit attribute). Omit for the default.
            const_specifier: Attribute ``const_specifier`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def type(self) -> mlir_python._mlir_python.Type:
        """Attribute ``type``: any type attribute."""

    @type.setter
    def type(self, arg: mlir_python._mlir_python.Type, /) -> None: ...
    @property
    def initial_value(self) -> mlir_python._mlir_python.Attribute | None:
        """
        Attribute ``initial_value``: An opaque attribute or TypedAttr instance.
        """

    @initial_value.setter
    def initial_value(self, arg: mlir_python._mlir_python.Attribute | None) -> None: ...
    @property
    def extern_specifier(self) -> bool:
        """Attribute ``extern_specifier``: unit attribute."""

    @extern_specifier.setter
    def extern_specifier(self, arg: bool, /) -> None: ...
    @property
    def static_specifier(self) -> bool:
        """Attribute ``static_specifier``: unit attribute."""

    @static_specifier.setter
    def static_specifier(self, arg: bool, /) -> None: ...
    @property
    def const_specifier(self) -> bool:
        """Attribute ``const_specifier``: unit attribute."""

    @const_specifier.setter
    def const_specifier(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "emitc.global"

class IfOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.if``: If-then-else operation.

    The `emitc.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    emitc.if %b  {
      ...
    } else {
      ...
    }
    ```

    The "then" region has exactly 1 block. The "else" region may have 0 or 1
    blocks. The blocks are always terminated with `emitc.yield`, which can be
    left out to be inserted implicitly. This operation doesn't produce any
    results.
    """

    def __init__(
        self,
        condition: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.if``: If-then-else operation.

        Args:
            condition: Operand ``condition`` (1-bit signless integer).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def condition(self) -> mlir_python._mlir_python.Value:
        """Operand ``condition``: 1-bit signless integer."""

    @property
    def then_region(self) -> mlir_python._mlir_python.Region:
        """Region ``thenRegion``: region with 1 blocks."""

    @property
    def else_region(self) -> mlir_python._mlir_python.Region:
        """Region ``elseRegion``: region with at most 1 blocks."""

    OPERATION_NAME: str = "emitc.if"

class IncludeOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.include``: Include operation.

    The `emitc.include` operation allows to define a source file inclusion via the
    `#include` directive.

    Example:

    ```mlir
    // Custom form defining the inclusion of `<myheader>`.
    emitc.include <"myheader.h">

    // Generic form of the same operation.
    "emitc.include" (){include = "myheader.h", is_standard_include} : () -> ()

    // Custom form defining the inclusion of `"myheader"`.
    emitc.include "myheader.h"

    // Generic form of the same operation.
    "emitc.include" (){include = "myheader.h"} : () -> ()
    ```
    """

    def __init__(
        self,
        include: str,
        *,
        is_standard_include: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.include``: Include operation.

        Args:
            include: Attribute ``include`` (source file to include).
            is_standard_include: Attribute ``is_standard_include`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def include(self) -> str:
        """Attribute ``include``: source file to include."""

    @include.setter
    def include(self, arg: str, /) -> None: ...
    @property
    def is_standard_include(self) -> bool:
        """Attribute ``is_standard_include``: unit attribute."""

    @is_standard_include.setter
    def is_standard_include(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "emitc.include"

class LiteralOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.literal``: Literal operation.

    The `emitc.literal` operation produces an SSA value equal to some constant
    specified by an attribute.

    Example:

    ```mlir
    %p0 = emitc.literal "M_PI" : f32
    %1 = "emitc.add" (%arg0, %p0) : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operation above.
    float v2 = v1 + M_PI;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        value: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.literal``: Literal operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            value: Attribute ``value`` (string attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> str:
        """Attribute ``value``: string attribute."""

    @value.setter
    def value(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "emitc.literal"

class LoadOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.load``: Load an lvalue into an SSA value..

    This operation loads the content of a modifiable lvalue into an SSA value.
    Modifications of the lvalue executed after the load are not observable on
    the produced value.

    Example:

    ```mlir
    %1 = emitc.load %0 : !emitc.lvalue<i32>
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = v1;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.load``: Load an lvalue into an SSA value..

        Args:
            result_type: Type of result ``result`` (any type).
            operand: Operand ``operand`` (EmitC lvalue type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: EmitC lvalue type."""

    OPERATION_NAME: str = "emitc.load"

class LogicalAndOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.logical_and``: Logical and operation.

    With the `emitc.logical_and` operation the logical operator && (and) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.logical_and %arg0, %arg1 : i32, i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v3 = v1 && v2;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.logical_and``: Logical and operation.

        Args:
            result_type: Type of result ``result`` (1-bit signless integer).
            lhs: Operand ``lhs`` (type supported by EmitC).
            rhs: Operand ``rhs`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: type supported by EmitC."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.logical_and"

class LogicalNotOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.logical_not``: Logical not operation.

    With the `emitc.logical_not` operation the logical operator ! (negation) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.logical_not %arg0 : i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v2 = !v1;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        operand_0: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.logical_not``: Logical not operation.

        Args:
            result_type: Type of result ``result`` (1-bit signless integer).
            operand_0: Operand ``operand_0`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand_0(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand_0``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.logical_not"

class LogicalOrOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.logical_or``: Logical or operation.

    With the `emitc.logical_or` operation the logical operator || (inclusive or)
    can be applied.

    Example:

    ```mlir
    %0 = emitc.logical_or %arg0, %arg1 : i32, i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v3 = v1 || v2;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.logical_or``: Logical or operation.

        Args:
            result_type: Type of result ``result`` (1-bit signless integer).
            lhs: Operand ``lhs`` (type supported by EmitC).
            rhs: Operand ``rhs`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: type supported by EmitC."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.logical_or"

class MemberOfPtrOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.member_of_ptr``: Member of pointer operation.

    With the `emitc.member_of_ptr` operation the member access operator `->`
    can be applied.

    Example:

    ```mlir
    %0 = "emitc.member_of_ptr" (%arg0) {member = "a"}
        : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>)
        -> !emitc.lvalue<i32>
    %1 = "emitc.member_of_ptr" (%arg0) {member = "b"}
        : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>)
        -> !emitc.array<2xi32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        member: str,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.member_of_ptr``: Member of pointer operation.

        Args:
            result_type: Type of result ``result`` (EmitC array type or EmitC lvalue type).
            member: Attribute ``member`` (the member to access).
            operand: Operand ``operand`` (emitc.lvalue of EmitC opaque type or EmitC pointer type values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand``: emitc.lvalue of EmitC opaque type or EmitC pointer type values.
        """

    @property
    def member(self) -> str:
        """Attribute ``member``: the member to access."""

    @member.setter
    def member(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "emitc.member_of_ptr"

class MemberOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.member``: Member operation.

    With the `emitc.member` operation the member access operator `.` can be
    applied.

    Example:

    ```mlir
    %0 = "emitc.member" (%arg0) {member = "a"}
        : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.lvalue<i32>
    %1 = "emitc.member" (%arg0) {member = "b"}
        : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.array<2xi32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        member: str,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.member``: Member operation.

        Args:
            result_type: Type of result ``result`` (EmitC array type or EmitC lvalue type).
            member: Attribute ``member`` (the member to access).
            operand: Operand ``operand`` (emitc.lvalue of EmitC opaque type values).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: emitc.lvalue of EmitC opaque type values."""

    @property
    def member(self) -> str:
        """Attribute ``member``: the member to access."""

    @member.setter
    def member(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "emitc.member"

class MulOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.mul``: Multiplication operation.

    With the `emitc.mul` operation the arithmetic operator * (multiplication) can
    be applied.

    Example:

    ```mlir
    // Custom form of the multiplication operation.
    %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.mul %arg2, %arg3 : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 * v2;
    float v6 = v3 * v4;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        operand_0: mlir_python._mlir_python.Value,
        operand_1: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.mul``: Multiplication operation.

        Args:
            result_type: Type of result ``result`` (floating-point type supported by EmitC or integer, index or opaque type supported by EmitC).
            operand_0: Operand ``operand_0`` (floating-point type supported by EmitC or integer, index or opaque type supported by EmitC).
            operand_1: Operand ``operand_1`` (floating-point type supported by EmitC or integer, index or opaque type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand_0(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand_0``: floating-point type supported by EmitC or integer, index or opaque type supported by EmitC.
        """

    @property
    def operand_1(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand_1``: floating-point type supported by EmitC or integer, index or opaque type supported by EmitC.
        """

    OPERATION_NAME: str = "emitc.mul"

class RemOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.rem``: Remainder operation.

    With the `emitc.rem` operation the arithmetic operator % (remainder) can
    be applied.

    Example:

    ```mlir
    // Custom form of the remainder operation.
    %0 = emitc.rem %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v5 = v1 % v2;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        operand_0: mlir_python._mlir_python.Value,
        operand_1: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.rem``: Remainder operation.

        Args:
            result_type: Type of result ``result`` (integer, index or opaque type supported by EmitC).
            operand_0: Operand ``operand_0`` (integer, index or opaque type supported by EmitC).
            operand_1: Operand ``operand_1`` (integer, index or opaque type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand_0(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand_0``: integer, index or opaque type supported by EmitC.
        """

    @property
    def operand_1(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand_1``: integer, index or opaque type supported by EmitC.
        """

    OPERATION_NAME: str = "emitc.rem"

class ReturnOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.return``: Function return operation.

    The `emitc.return` operation represents a return operation within a function.
    The operation takes zero or exactly one operand and produces no results.
    The operand number and type must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    emitc.func @foo() -> (i32) {
      ...
      emitc.return %0 : i32
    }
    ```
    """

    def __init__(
        self,
        *,
        operand: mlir_python._mlir_python.Value | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.return``: Function return operation.

        Args:
            operand: Operand ``operand`` (type supported by EmitC). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value | None:
        """Operand ``operand``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.return"

class SubOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.sub``: Subtraction operation.

    With the `emitc.sub` operation the arithmetic operator - (subtraction) can
    be applied.

    Example:

    ```mlir
    // Custom form of the substraction operation.
    %0 = emitc.sub %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.sub %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    %2 = emitc.sub %arg4, %arg5 : (!emitc.ptr<i32>, !emitc.ptr<i32>)
        -> !emitc.ptrdiff_t
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v7 = v1 - v2;
    float* v8 = v3 - v4;
    ptrdiff_t v9 = v5 - v6;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.sub``: Subtraction operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            lhs: Operand ``lhs`` (type supported by EmitC).
            rhs: Operand ``rhs`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: type supported by EmitC."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.sub"

class SubscriptOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.subscript``: Subscript operation.

    With the `emitc.subscript` operation the subscript operator `[]` can be applied
    to variables or arguments of array, pointer and opaque type.

    Example:

    ```mlir
    %i = index.constant 1
    %j = index.constant 7
    %0 = emitc.subscript %arg0[%i, %j] : (!emitc.array<4x8xf32>, index, index)
           -> !emitc.lvalue<f32>
    %1 = emitc.subscript %arg1[%i] : (!emitc.ptr<i32>, index)
           -> !emitc.lvalue<i32>
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        value: mlir_python._mlir_python.Value,
        indices: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.subscript``: Subscript operation.

        Args:
            result_type: Type of result ``result`` (EmitC lvalue type).
            value: Operand ``value`` (the value to subscript).
            indices: Operand ``indices`` (type supported by EmitC). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: the value to subscript."""

    @property
    def indices(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``indices``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.subscript"

class SwitchOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.switch``: Switch operation.

    The `emitc.switch` is a control-flow operation that branches to one of
    the given regions based on the values of the argument and the cases.
    The operand to a switch operation is a opaque, integral or pointer
    wide types.

    The operation always has a "default" region and any number of case regions
    denoted by integer constants. Control-flow transfers to the case region
    whose constant value equals the value of the argument. If the argument does
    not equal any of the case values, control-flow transfer to the "default"
    region.

    The operation does not return any value. Moreover, case regions must be
    explicitly terminated using the `emitc.yield` operation. Default region is
    yielded implicitly.

    Example:

    ```mlir
    // Example:
    emitc.switch %0 : i32
    case 2 {
      %1 = emitc.call_opaque "func_b" () : () -> i32
      emitc.yield
    }
    case 5 {
      %2 = emitc.call_opaque "func_a" () : () -> i32
      emitc.yield
    }
    default {
      %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
      emitc.call_opaque "func2" (%3) : (f32) -> ()
    }
    ```
    ```c++
    // Code emitted for the operations above.
    switch (v1) {
    case 2: {
      int32_t v2 = func_b();
      break;
    }
    case 5: {
      int32_t v3 = func_a();
      break;
    }
    default: {
      float v4 = 4.200000000e+01f;
      func2(v4);
      break;
    }
    }
    ```
    """

    def __init__(
        self,
        arg: mlir_python._mlir_python.Value,
        cases: mlir_python._mlir_python.Attribute,
        *,
        num_case_regions: int = 0,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.switch``: Switch operation.

        Args:
            arg: Operand ``arg`` (integer, index or opaque type supported by EmitC).
            cases: Attribute ``cases`` (i64 dense array attribute).
            num_case_regions: Number of ``caseRegions`` regions.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """Operand ``arg``: integer, index or opaque type supported by EmitC."""

    @property
    def cases(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``cases``: i64 dense array attribute."""

    @cases.setter
    def cases(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def default_region(self) -> mlir_python._mlir_python.Region:
        """Region ``defaultRegion``: region with 1 blocks."""

    @property
    def case_regions(self) -> list[mlir_python._mlir_python.Region]:
        """Region ``caseRegions``: region with 1 blocks."""

    OPERATION_NAME: str = "emitc.switch"

class UnaryMinusOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.unary_minus``: Unary minus operation.

    With the `emitc.unary_minus` operation the unary operator - (minus) can be
    applied.

    Example:

    ```mlir
    %0 = emitc.unary_minus %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = -v1;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        operand_0: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.unary_minus``: Unary minus operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            operand_0: Operand ``operand_0`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand_0(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand_0``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.unary_minus"

class UnaryPlusOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.unary_plus``: Unary plus operation.

    With the `emitc.unary_plus` operation the unary operator + (plus) can be
    applied.

    Example:

    ```mlir
    %0 = emitc.unary_plus %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = +v1;
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        operand_0: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.unary_plus``: Unary plus operation.

        Args:
            result_type: Type of result ``result`` (type supported by EmitC).
            operand_0: Operand ``operand_0`` (type supported by EmitC).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand_0(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand_0``: type supported by EmitC."""

    OPERATION_NAME: str = "emitc.unary_plus"

class VariableOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.variable``: Variable operation.

    The `emitc.variable` operation produces an SSA value equal to some value
    specified by an attribute. This can be used to form simple integer and
    floating point variables, as well as more exotic things like tensor
    variables. The `emitc.variable` operation also supports the EmitC opaque
    attribute and the EmitC opaque type. If further supports the EmitC
    pointer type, whereas folding is not supported.
    The `emitc.variable` is emitted as a C/C++ local variable.

    Example:

    ```mlir
    // Integer variable
    %0 = "emitc.variable"(){value = 42 : i32} : () -> !emitc.lvalue<i32>

    // Variable emitted as `int32_t* = NULL;`
    %1 = "emitc.variable"() {value = #emitc.opaque<"NULL">}
      : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>
    ```

    Since folding is not supported, it can be used with pointers.
    As an example, it is valid to create pointers to `variable` operations
    by using `apply` operations and pass these to a `call` operation.
    ```mlir
    %0 = "emitc.variable"() {value = 0 : i32} : () -> !emitc.lvalue<i32>
    %1 = "emitc.variable"() {value = 0 : i32} : () -> !emitc.lvalue<i32>
    %2 = emitc.apply "&"(%0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    %3 = emitc.apply "&"(%1) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    emitc.call_opaque "write"(%2, %3)
      : (!emitc.ptr<i32>, !emitc.ptr<i32>) -> ()
    ```
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        value: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.variable``: Variable operation.

        Args:
            result_type: Type of result ``result`` (EmitC array type or EmitC lvalue type).
            value: Attribute ``value`` (An opaque attribute or TypedAttr instance).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``value``: An opaque attribute or TypedAttr instance."""

    @value.setter
    def value(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "emitc.variable"

class VerbatimOp(mlir_python._mlir_python.Operation):
    r"""
    ``emitc.verbatim``: Verbatim operation.

    The `emitc.verbatim` operation produces no results and the value is emitted as is
    followed by a line break  ('\n' character) during translation.

    Note: Use with caution. This operation can have arbitrary effects on the
    semantics of the emitted code. Use semantically more meaningful operations
    whenever possible. Additionally this op is *NOT* intended to be used to
    inject large snippets of code.

    This operation can be used in situations where a more suitable operation is
    not yet implemented in the dialect or where preprocessor directives
    interfere with the structure of the code. One example of this is to declare
    the linkage of external symbols to make the generated code usable in both C
    and C++ contexts:

    ```c++
    #ifdef __cplusplus
    extern "C" {
    #endif

    ...

    #ifdef __cplusplus
    }
    #endif
    ```

    If the `emitc.verbatim` op has operands, then the `value` is interpreted as
    format string, where `{}` is a placeholder for an operand in their order.
    For example, `emitc.verbatim "#pragma my src={} dst={}" %src, %dest : i32, i32`
    would be emitted as `#pragma my src=a dst=b` if `%src` became `a` and
    `%dest` became `b` in the C code.
    `{{` in the format string is interpreted as a single `{` and doesn't introduce
    a placeholder.

    Example:

    ```mlir
    emitc.verbatim "typedef float f32;"
    emitc.verbatim "#pragma my var={} property" args %arg : f32
    ```
    ```c++
    // Code emitted for the operation above.
    typedef float f32;
    #pragma my var=v1 property
    ```
    """

    def __init__(
        self,
        value: str,
        fmt_args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.verbatim``: Verbatim operation.

        Args:
            value: Attribute ``value`` (string attribute).
            fmt_args: Operand ``fmtArgs`` (type supported by EmitC or EmitC lvalue type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def fmt_args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``fmtArgs``: type supported by EmitC or EmitC lvalue type."""

    @property
    def value(self) -> str:
        """Attribute ``value``: string attribute."""

    @value.setter
    def value(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "emitc.verbatim"

class YieldOp(mlir_python._mlir_python.Operation):
    """
    ``emitc.yield``: Block termination operation.

    The `emitc.yield` terminates its parent EmitC op's region, optionally yielding
    an SSA value. The semantics of how the values are yielded is defined by the
    parent operation.
    If `emitc.yield` has an operand, the operand must match the parent operation's
    result. If the parent operation defines no values, then the `emitc.yield`
    may be left out in the custom syntax and the builders will insert one
    implicitly. Otherwise, it has to be present in the syntax to indicate which
    value is yielded.
    """

    def __init__(
        self,
        *,
        result: mlir_python._mlir_python.Value | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``emitc.yield``: Block termination operation.

        Args:
            result: Operand ``result`` (type supported by EmitC). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "emitc.yield"

class OpaqueType(mlir_python._mlir_python.Type):
    """
    ``!emitc.opaque<"T">``: a C or C++ type spelled out as text, e.g.
    ``OpaqueType("FILE")`` or ``OpaqueType("std::vector<int>")``.
    Pointers are ``PointerType``s: ``PointerType(OpaqueType("FILE"))``.
    """

    def __init__(
        self, value: str, *, context: mlir_python._mlir_python.Context | None = None
    ) -> None:
        """Create ``!emitc.opaque<"value">``."""

    @property
    def value(self) -> str:
        """The type as written in C or C++."""

class PointerType(mlir_python._mlir_python.Type):
    """``!emitc.ptr<T>``: a C pointer to ``T``."""

    def __init__(self, pointee: mlir_python._mlir_python.Type) -> None:
        """Create ``!emitc.ptr<pointee>``."""

    @property
    def pointee(self) -> mlir_python._mlir_python.Type:
        """The type pointed to."""

class ArrayType(mlir_python._mlir_python.ShapedType):
    """
    ``!emitc.array<NxMxT>``: a C array with static sizes, e.g. what a
    ``memref<4xf32>`` becomes.
    """

    def __init__(
        self, shape: Sequence[int], element_type: mlir_python._mlir_python.Type
    ) -> None:
        """Create ``!emitc.array<shape x element_type>``."""

class LValueType(mlir_python._mlir_python.Type):
    """
    ``!emitc.lvalue<T>``: an assignable ``T`` (a variable), read with
    ``emitc.load`` and written with ``emitc.assign``.
    """

    def __init__(self, value_type: mlir_python._mlir_python.Type) -> None:
        """Create ``!emitc.lvalue<value_type>``."""

    @property
    def value_type(self) -> mlir_python._mlir_python.Type:
        """The type of the value held."""

class SizeTType(mlir_python._mlir_python.Type):
    """``!emitc.size_t``: C's ``size_t``."""

    def __init__(
        self, *, context: mlir_python._mlir_python.Context | None = None
    ) -> None:
        """Create ``!emitc.size_t``."""

class SignedSizeTType(mlir_python._mlir_python.Type):
    """``!emitc.ssize_t``: POSIX's ``ssize_t`` (not C99)."""

    def __init__(
        self, *, context: mlir_python._mlir_python.Context | None = None
    ) -> None:
        """Create ``!emitc.ssize_t``."""

class PtrDiffTType(mlir_python._mlir_python.Type):
    """``!emitc.ptrdiff_t``: C's ``ptrdiff_t``."""

    def __init__(
        self, *, context: mlir_python._mlir_python.Context | None = None
    ) -> None:
        """Create ``!emitc.ptrdiff_t``."""
