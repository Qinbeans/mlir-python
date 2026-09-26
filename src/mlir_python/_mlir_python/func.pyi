"""Typed operations of the MLIR ``func`` dialect."""

from collections.abc import Sequence

import mlir_python._mlir_python

class CallIndirectOp(mlir_python._mlir_python.Operation):
    """
    ``func.call_indirect``: indirect call operation.

    The `func.call_indirect` operation represents an indirect call to a value
    of function type. The operands and result types of the call must match the
    specified function type.

    Function values can be created with the
    [`func.constant` operation](#funcconstant-constantop).

    Example:

    ```mlir
    %func = func.constant @my_func : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    %result = func.call_indirect %func(%0, %1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    ```
    """

    def __init__(
        self,
        callee: mlir_python._mlir_python.Value,
        callee_operands: Sequence[mlir_python._mlir_python.Value] = [],
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        arg_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``func.call_indirect``: indirect call operation.

        Args:
            callee: Operand ``callee`` (function type).
            callee_operands: Operand ``callee_operands`` (any type). Empty by default.
            result_types: Type of result ``results`` (any type). Empty by default.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def callee(self) -> mlir_python._mlir_python.Value:
        """Operand ``callee``: function type."""

    @property
    def callee_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``callee_operands``: any type."""

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

    OPERATION_NAME: str = "func.call_indirect"

class CallOp(mlir_python._mlir_python.Operation):
    """
    ``func.call``: call operation.

    The `func.call` operation represents a direct call to a function that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = func.call @my_add(%0, %1) : (f32, f32) -> f32
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
        no_inline: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``func.call``: call operation.

        Args:
            callee: Attribute ``callee`` (flat symbol reference attribute).
            operands: Operand ``operands`` (any type). Empty by default.
            result_types: Type of result ``result`` (any type). Empty by default.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            no_inline: Attribute ``no_inline`` (unit attribute). Omit for the default.
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
    @property
    def no_inline(self) -> bool:
        """Attribute ``no_inline``: unit attribute."""

    @no_inline.setter
    def no_inline(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "func.call"

class ConstantOp(mlir_python._mlir_python.Operation):
    """
    ``func.constant``: constant.

    The `func.constant` operation produces an SSA value from a symbol reference
    to a `func.func` operation

    Example:

    ```mlir
    // Reference to function @myfn.
    %2 = func.constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>

    // Equivalent generic forms
    %2 = "func.constant"() { value = @myfn } : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
    ```

    MLIR does not allow direct references to functions in SSA operands because
    the compiler is multithreaded, and disallowing SSA values to directly
    reference a function simplifies this
    ([rationale](../Rationale/Rationale.md#multithreading-the-compiler)).
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
        Create ``func.constant``: constant.

        Args:
            result_type: Type of result ``result`` (any type).
            value: Attribute ``value`` (flat symbol reference attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> str:
        """Attribute ``value``: flat symbol reference attribute."""

    @value.setter
    def value(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "func.constant"

class FuncOp(mlir_python._mlir_python.Operation):
    """
    ``func.func``: An operation with a name containing a single `SSACFG` region.

    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external function declaration (used when
    referring to a function declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for function arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

    Example:

    ```mlir
    // External function definitions.
    func.func private @abort()
    func.func private @scribble(i32, i64, memref<? x 128 x f32, #layout_map0>) -> f64

    // A function that returns its argument twice:
    func.func @count(%x: i64) -> (i64, i64)
      attributes {fruit = "banana"} {
      return %x, %x: i64, i64
    }

    // A function with an argument attribute
    func.func private @example_fn_arg(%x: i32 {swift.self = unit})

    // A function with a result attribute
    func.func private @example_fn_result() -> (f64 {dialectName.attrName = 0 : i64})

    // A function with an attribute
    func.func private @example_fn_attr() attributes {dialectName.attrName = false}
    ```
    """

    def __init__(
        self,
        sym_name: str,
        function_type: mlir_python._mlir_python.FunctionType,
        *,
        sym_visibility: str | None = None,
        arg_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        no_inline: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``func.func``: An operation with a name containing a single `SSACFG` region.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            function_type: Attribute ``function_type`` (type attribute of function type).
            sym_visibility: Attribute ``sym_visibility`` (string attribute). Optional.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            no_inline: Attribute ``no_inline`` (unit attribute). Omit for the default.
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
    def sym_visibility(self) -> str | None:
        """Attribute ``sym_visibility``: string attribute."""

    @sym_visibility.setter
    def sym_visibility(self, arg: str | None) -> None: ...
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
    def no_inline(self) -> bool:
        """Attribute ``no_inline``: unit attribute."""

    @no_inline.setter
    def no_inline(self, arg: bool, /) -> None: ...
    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: any region."""

    OPERATION_NAME: str = "func.func"

    def add_entry_block(self) -> mlir_python._mlir_python.Block:
        """
        Give this declaration a body: an entry block whose arguments match
        ``function_type.inputs``. Returns the block.

        Raises:
            ValueError: If the function already has a body.
        """

    @property
    def arguments(self) -> list[mlir_python._mlir_python.BlockArgument]:
        """
        The entry block's arguments.

        Raises:
            ValueError: If the function has no body.
        """

class ReturnOp(mlir_python._mlir_python.Operation):
    """
    ``func.return``: Function return operation.

    The `func.return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    func.func @foo() -> (i32, f8) {
      ...
      return %0, %1 : i32, f8
    }
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
        Create ``func.return``: Function return operation.

        Args:
            operands: Operand ``operands`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "func.return"
