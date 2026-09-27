"""Typed operations of the MLIR ``irdl`` dialect."""

import enum
from collections.abc import Sequence

import mlir_python._mlir_python

class Variadicity(enum.Enum):
    """variadicity kind"""

    SINGLE = 0
    """``single`` in MLIR text."""

    OPTIONAL = 1
    """``optional`` in MLIR text."""

    VARIADIC = 2
    """``variadic`` in MLIR text."""

class AllOfOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.all_of``: Constraints to the intersection of the provided constraints.

    `irdl.all_of` defines a constraint that accepts any type or attribute that
    satisfies all of its provided constraints.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex_f32 {
        %0 = irdl.is i32
        %1 = irdl.is f32
        %2 = irdl.any_of(%0, %1) // is 32-bit

        %3 = irdl.is f32
        %4 = irdl.is f64
        %5 = irdl.any_of(%3, %4) // is a float

        %6 = irdl.all_of(%2, %5) // is a 32-bit float
        irdl.parameters(%6)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath` that
    has one parameter that must be 32-bit long and a float (in other
    words, that must be `f32`).
    """

    def __init__(
        self,
        output_type: mlir_python._mlir_python.Type,
        args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.all_of``: Constraints to the intersection of the provided constraints.

        Args:
            output_type: Type of result ``output`` (IRDL handle to an `mlir::Attribute`).
            args: Operand ``args`` (IRDL handle to an `mlir::Attribute`). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``args``: IRDL handle to an `mlir::Attribute`."""

    @property
    def output(self) -> mlir_python._mlir_python.OpResult:
        """Result ``output``: IRDL handle to an `mlir::Attribute`."""

    OPERATION_NAME: str = "irdl.all_of"

class AnyOfOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.any_of``: Constraints to the union of the provided constraints.

    `irdl.any_of` defines a constraint that accepts any type or attribute that
    satisfies at least one of its provided type constraints.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex {
        %0 = irdl.is i32
        %1 = irdl.is i64
        %2 = irdl.is f32
        %3 = irdl.is f64
        %4 = irdl.any_of(%0, %1, %2, %3)
        irdl.parameters(%4)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath` that
    has a single type parameter that can be either `i32`, `i64`, `f32` or
    `f64`.
    """

    def __init__(
        self,
        output_type: mlir_python._mlir_python.Type,
        args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.any_of``: Constraints to the union of the provided constraints.

        Args:
            output_type: Type of result ``output`` (IRDL handle to an `mlir::Attribute`).
            args: Operand ``args`` (IRDL handle to an `mlir::Attribute`). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``args``: IRDL handle to an `mlir::Attribute`."""

    @property
    def output(self) -> mlir_python._mlir_python.OpResult:
        """Result ``output``: IRDL handle to an `mlir::Attribute`."""

    OPERATION_NAME: str = "irdl.any_of"

class AnyOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.any``: Accept any type or attribute.

    `irdl.any` defines a constraint that accepts any type or attribute.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex_flexible {
        %0 = irdl.any
        irdl.parameters(%0)
      }
    }
    ```

    The above program defines a type `complex_flexible` inside the dialect
    `cmath` that has a single parameter that can be any attribute.
    """

    def __init__(
        self,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.any``: Accept any type or attribute.

        Result types are inferred.
        """

    @property
    def output(self) -> mlir_python._mlir_python.OpResult:
        """Result ``output``: IRDL handle to an `mlir::Attribute`."""

    OPERATION_NAME: str = "irdl.any"

class AttributeOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.attribute``: Define a new attribute.

    `irdl.attribute` defines a new attribute belonging to the `irdl.dialect`
    parent.

    The attribute parameters can be defined with an `irdl.parameters` operation
    in the optional region.

    Example:

    ```mlir
    irdl.dialect @testd {
      irdl.attribute @enum_attr {
        %0 = irdl.is "foo"
        %1 = irdl.is "bar"
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
      }
    }
    ```

    The above program defines an `enum_attr` attribute inside the `testd`
    dialect. The attribute has one `StringAttr` parameter that should be
    either a `"foo"` or a `"bar"`.
    """

    def __init__(
        self,
        sym_name: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.attribute``: Define a new attribute.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: region with 1 blocks."""

    OPERATION_NAME: str = "irdl.attribute"

class AttributesOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.attributes``: Define the attributes of an operation.

    `irdl.attributes` defines the attributes of the `irdl.operation` parent
    operation definition.

    In the following example, `irdl.attributes` defines the attributes of the
    `attr_op` operation:

    ```mlir
    irdl.dialect @example {

      irdl.operation @attr_op {
        %0 = irdl.any
        %1 = irdl.is i64
        irdl.attibutes {
          "attr1" = %0,
          "attr2" = %1
        }
      }
    }
    ```

    The operation will expect an arbitrary attribute "attr1" and an
    attribute "attr2" with value `i64`.
    """

    def __init__(
        self,
        attribute_value_names: mlir_python._mlir_python.ArrayAttr,
        attribute_values: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.attributes``: Define the attributes of an operation.

        Args:
            attribute_value_names: Attribute ``attributeValueNames`` (string array attribute).
            attribute_values: Operand ``attributeValues`` (IRDL handle to an `mlir::Attribute`). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def attribute_values(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``attributeValues``: IRDL handle to an `mlir::Attribute`."""

    @property
    def attribute_value_names(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``attributeValueNames``: string array attribute."""

    @attribute_value_names.setter
    def attribute_value_names(
        self, arg: mlir_python._mlir_python.ArrayAttr, /
    ) -> None: ...

    OPERATION_NAME: str = "irdl.attributes"

class BaseOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.base``: Constraints an attribute/type base.

    `irdl.base` defines a constraint that only accepts a single type
    or attribute base, e.g. an `IntegerType`. The attribute base is defined
    either by a symbolic reference to the corresponding IRDL definition,
    or by the name of the base. Named bases are prefixed with `!` or `#`
    respectively for types and attributes.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex {
        %0 = irdl.base "!builtin.integer"
        irdl.parameters(%0)
      }

      irdl.type @complex_wrapper {
        %0 = irdl.base @cmath::@complex
        irdl.parameters(%0)
      }
    }
    ```

    The above program defines a `cmath.complex` type that expects a single
    parameter, which is a type with base name `builtin.integer`, which is the
    name of an `IntegerType` type.
    It also defines a `cmath.complex_wrapper` type that expects a single
    parameter, which is a type of base type `cmath.complex`.
    """

    def __init__(
        self,
        *,
        base_ref: mlir_python._mlir_python.SymbolRefAttr | None = None,
        base_name: str | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.base``: Constraints an attribute/type base.

        Result types are inferred.

        Args:
            base_ref: Attribute ``base_ref`` (symbol reference attribute). Optional.
            base_name: Attribute ``base_name`` (string attribute). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def output(self) -> mlir_python._mlir_python.OpResult:
        """Result ``output``: IRDL handle to an `mlir::Attribute`."""

    @property
    def base_ref(self) -> mlir_python._mlir_python.SymbolRefAttr | None:
        """Attribute ``base_ref``: symbol reference attribute."""

    @base_ref.setter
    def base_ref(self, arg: mlir_python._mlir_python.SymbolRefAttr | None) -> None: ...
    @property
    def base_name(self) -> str | None:
        """Attribute ``base_name``: string attribute."""

    @base_name.setter
    def base_name(self, arg: str | None) -> None: ...

    OPERATION_NAME: str = "irdl.base"

class CPredOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.c_pred``: Constraints an attribute using a C++ predicate.

    `irdl.c_pred` defines a constraint that is written in C++.

    Dialects using this operation cannot be registered at runtime, as it relies
    on C++ code.

    Special placeholders can be used to refer to entities in the context where
    this predicate is used. They serve as "hooks" to the enclosing environment.
    The following special placeholders are supported in constraints for an op:

    * `$_builder` will be replaced by a mlir::Builder instance.
    * `$_op` will be replaced by the current operation.
    * `$_self` will be replaced with the entity this predicate is attached to.
       Compared to ODS, `$_self` is always of type `mlir::Attribute`, and types
       are manipulated as `TypeAttr` attributes.

    Example:
    ```mlir
    irdl.type @op_with_attr {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerAttr>($_self)"
      irdl.parameters(%0)
    }
    ```

    In this example, @op_with_attr is defined as a type with a single
    parameter, which is an `IntegerAttr`, as constrained by the C++ predicate.
    """

    def __init__(
        self,
        pred: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.c_pred``: Constraints an attribute using a C++ predicate.

        Result types are inferred.

        Args:
            pred: Attribute ``pred`` (string attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def output(self) -> mlir_python._mlir_python.OpResult:
        """Result ``output``: IRDL handle to an `mlir::Attribute`."""

    @property
    def pred(self) -> str:
        """Attribute ``pred``: string attribute."""

    @pred.setter
    def pred(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "irdl.c_pred"

class DialectOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.dialect``: Define a new dialect.

    The `irdl.dialect` operation defines a dialect. All operations, attributes,
    and types defined inside its region will be part of the dialect.

    Example:

    ```mlir
    irdl.dialect @cmath {
      ...
    }
    ```

    The above program defines a `cmath` dialect.
    """

    def __init__(
        self,
        sym_name: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.dialect``: Define a new dialect.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: region with 1 blocks."""

    OPERATION_NAME: str = "irdl.dialect"

class IsOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.is``: Constraints an attribute/type to be a specific attribute instance.

    `irdl.is` defines a constraint that only accepts a specific instance of a
    type or attribute.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex_i32 {
        %0 = irdl.is i32
        irdl.parameters(%0)
      }
    }
    ```

    The above program defines a `complex_i32` type inside the dialect `cmath`
    that can only have a `i32` as its parameter.
    """

    def __init__(
        self,
        expected: mlir_python._mlir_python.Attribute,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.is``: Constraints an attribute/type to be a specific attribute instance.

        Result types are inferred.

        Args:
            expected: Attribute ``expected`` (any attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def output(self) -> mlir_python._mlir_python.OpResult:
        """Result ``output``: IRDL handle to an `mlir::Attribute`."""

    @property
    def expected(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``expected``: any attribute."""

    @expected.setter
    def expected(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "irdl.is"

class OperandsOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.operands``: Define the operands of an operation.

    `irdl.operands` define the operands of the `irdl.operation` parent operation
    definition. Each operand is named after an identifier.

    In the following example, `irdl.operands` defines the operands of the
    `mul` operation:

    ```mlir
    irdl.dialect @cmath {

      irdl.type @complex { /* ... */ }

      irdl.operation @mul {
        %0 = irdl.any
        %1 = irdl.parametric @cmath::@complex<%0>
        irdl.results(res: %1)
        irdl.operands(lhs: %1, rhs: %1)
      }
    }
    ```

    The `mul` operation will expect two operands of type `cmath.complex`, that
    have the same type, and return a result of the same type.

    The operands can also be marked as variadic or optional:
    ```mlir
    irdl.operands(foo: %0, bar: single %1, baz: optional %2, qux: variadic %3)
    ```

    Here, foo and bar are required single operands, baz is an optional operand,
    and qux is a variadic operand.

    When more than one operand is marked as optional or variadic, the operation
    will expect a 'operandSegmentSizes' attribute that defines the number of
    operands in each segment.
    """

    def __init__(
        self,
        names: mlir_python._mlir_python.ArrayAttr,
        variadicity: mlir_python._mlir_python.Attribute,
        args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.operands``: Define the operands of an operation.

        Args:
            names: Attribute ``names`` (string array attribute).
            variadicity: Attribute ``variadicity``.
            args: Operand ``args`` (IRDL handle to an `mlir::Attribute`). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``args``: IRDL handle to an `mlir::Attribute`."""

    @property
    def names(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``names``: string array attribute."""

    @names.setter
    def names(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...
    @property
    def variadicity(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``variadicity``."""

    @variadicity.setter
    def variadicity(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "irdl.operands"

class OperationOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.operation``: Define a new operation.

    `irdl.operation` defines a new operation belonging to the `irdl.dialect`
    parent.

    Operations can define constraints on their operands and results with the
    `irdl.results` and `irdl.operands` operations. If these operations are not
    present in the region, the results or operands are expected to be empty.

    Example:

    ```mlir
    irdl.dialect @cmath {

      irdl.type @complex { /* ... */ }

      irdl.operation @norm {
        %0 = irdl.any
        %1 = irdl.parametric @cmath::@complex<%0>
        irdl.results(%0)
        irdl.operands(%1)
      }
    }
    ```

    The above program defines an operation `norm` inside the dialect `cmath`.
    The operation expects a single operand of base type `cmath.complex`, and
    returns a single result of the element type of the operand.
    """

    def __init__(
        self,
        sym_name: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.operation``: Define a new operation.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: region with 1 blocks."""

    OPERATION_NAME: str = "irdl.operation"

class ParametersOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.parameters``: Define the constraints on parameters of a type/attribute definition.

    `irdl.parameters` defines the constraints on parameters of a type or
    attribute definition. Each parameter is named after an identifier.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex {
        %0 = irdl.is i32
        %1 = irdl.is i64
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(elem: %2)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath`. The
    type has a single parameter `elem` that should be either `i32` or `i64`.
    """

    def __init__(
        self,
        names: mlir_python._mlir_python.ArrayAttr,
        args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.parameters``: Define the constraints on parameters of a type/attribute definition.

        Args:
            names: Attribute ``names`` (string array attribute).
            args: Operand ``args`` (IRDL handle to an `mlir::Attribute`). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``args``: IRDL handle to an `mlir::Attribute`."""

    @property
    def names(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``names``: string array attribute."""

    @names.setter
    def names(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...

    OPERATION_NAME: str = "irdl.parameters"

class ParametricOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.parametric``: Constraints an attribute/type base and its parameters.

    `irdl.parametric` defines a constraint that accepts only a single type
    or attribute base. The attribute base is defined by a symbolic reference
    to the corresponding definition. It will additionally constraint the
    parameters of the type/attribute.

    Example:

    ```mlir
    irdl.dialect @cmath {

      irdl.type @complex { /* ... */ }

      irdl.operation @norm {
        %0 = irdl.any
        %1 = irdl.parametric @cmath::@complex<%0>
        irdl.operands(%1)
        irdl.results(%0)
      }
    }
    ```

    The above program defines an operation `norm` inside the dialect `cmath` that
    for any `T` takes a `cmath.complex` with parameter `T` and returns a `T`.
    """

    def __init__(
        self,
        base_type: mlir_python._mlir_python.SymbolRefAttr,
        args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.parametric``: Constraints an attribute/type base and its parameters.

        Result types are inferred.

        Args:
            base_type: Attribute ``base_type`` (symbol reference attribute).
            args: Operand ``args`` (IRDL handle to an `mlir::Attribute`). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``args``: IRDL handle to an `mlir::Attribute`."""

    @property
    def output(self) -> mlir_python._mlir_python.OpResult:
        """Result ``output``: IRDL handle to an `mlir::Attribute`."""

    @property
    def base_type(self) -> mlir_python._mlir_python.SymbolRefAttr:
        """Attribute ``base_type``: symbol reference attribute."""

    @base_type.setter
    def base_type(self, arg: mlir_python._mlir_python.SymbolRefAttr, /) -> None: ...

    OPERATION_NAME: str = "irdl.parametric"

class RegionOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.region``: Define a region of an operation.

    The irdl.region construct defines a set of characteristics
    that a region of an operation should satify. Each region is named after
    an identifier.

    These characteristics include constraints for the entry block arguments
    of the region and the total number of blocks it contains.
    The number of blocks must be a non-zero and non-negative integer,
    and it is optional by default.
    The set of constraints for the entry block arguments may be optional or
    empty. If no parentheses are provided, the set is assumed to be optional,
    and the arguments are not constrained in any way. If parentheses are
    provided with no arguments, it means that the region must have
    no entry block arguments


    Example:

    ```mlir
    irdl.dialect @example {
      irdl.operation @op_with_regions {
          %r0 = irdl.region
          %r1 = irdl.region()
          %v0 = irdl.is i32
          %v1 = irdl.is i64
          %r2 = irdl.region(%v0, %v1)
          %r3 = irdl.region with size 3

          irdl.regions(foo: %r0, bar: %r1, baz: %r2, qux: %r3)
      }
    }
    ```

    The above snippet demonstrates an operation named `@op_with_regions`,
    which is constrained to have four regions.

    * Region `foo` doesn't have any constraints on the arguments
      or the number of blocks.
    * Region `bar` should have an empty set of arguments.
    * Region `baz` should have two arguments of types `i32` and `i64`.
    * Region `qux` should contain exactly three blocks.
    """

    def __init__(
        self,
        entry_block_args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        number_of_blocks: int | None = None,
        constrained_arguments: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.region``: Define a region of an operation.

        Result types are inferred.

        Args:
            entry_block_args: Operand ``entryBlockArgs`` (IRDL handle to an `mlir::Attribute`). Empty by default.
            number_of_blocks: Attribute ``numberOfBlocks`` (32-bit signless integer attribute). Optional.
            constrained_arguments: Attribute ``constrainedArguments`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def entry_block_args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``entryBlockArgs``: IRDL handle to an `mlir::Attribute`."""

    @property
    def output(self) -> mlir_python._mlir_python.OpResult:
        """Result ``output``: IRDL handle to a region definition."""

    @property
    def number_of_blocks(self) -> int | None:
        """Attribute ``numberOfBlocks``: 32-bit signless integer attribute."""

    @number_of_blocks.setter
    def number_of_blocks(self, arg: int | None) -> None: ...
    @property
    def constrained_arguments(self) -> bool:
        """Attribute ``constrainedArguments``: unit attribute."""

    @constrained_arguments.setter
    def constrained_arguments(self, arg: bool, /) -> None: ...

    OPERATION_NAME: str = "irdl.region"

class RegionsOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.regions``: Define the regions of an operation.

    `irdl.regions` defines the regions of an operation by accepting
    values produced by `irdl.region` operation as arguments. Each
    region has an identifier as name.

    Example:

    ```mlir
    irdl.dialect @example {
      irdl.operation @op_with_regions {
        %r1 = irdl.region with size 3
        %0 = irdl.any
        %r2 = irdl.region(%0)
        irdl.regions(foo: %r1, bar: %r2)
      }
    }
    ```

    In the snippet above the operation is constrained to have two regions.
    The first region (`foo`) should contain three blocks.
    The second region (`bar`) should have one region with one argument.
    """

    def __init__(
        self,
        names: mlir_python._mlir_python.ArrayAttr,
        args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.regions``: Define the regions of an operation.

        Args:
            names: Attribute ``names`` (string array attribute).
            args: Operand ``args`` (IRDL handle to a region definition). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``args``: IRDL handle to a region definition."""

    @property
    def names(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``names``: string array attribute."""

    @names.setter
    def names(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...

    OPERATION_NAME: str = "irdl.regions"

class ResultsOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.results``: Define the results of an operation.

    `irdl.results` define the results of the `irdl.operation` parent operation
    definition. Each result is named after an identifier.

    In the following example, `irdl.results` defines the results of the
    `get_values` operation:

    ```mlir
    irdl.dialect @cmath {

      irdl.type @complex { /* ... */ }

      /// Returns the real and imaginary parts of a complex number.
      irdl.operation @get_values {
        %0 = irdl.any
        %1 = irdl.parametric @cmath::@complex<%0>
        irdl.results(re: %0, im: %0)
        irdl.operands(complex: %1)
      }
    }
    ```

    The operation will expect one operand of the `cmath.complex` type, and two
    results that have the underlying type of the `cmath.complex`.

    The results can also be marked as variadic or optional:
    ```mlir
    irdl.results(foo: %0, bar: single %1, baz: optional %2, qux: variadic %3)
    ```

    Here, foo and bar are required single results, baz is an optional result,
    and qux is a variadic result.

    When more than one result is marked as optional or variadic, the operation
    will expect a 'resultSegmentSizes' attribute that defines the number of
    results in each segment.
    """

    def __init__(
        self,
        names: mlir_python._mlir_python.ArrayAttr,
        variadicity: mlir_python._mlir_python.Attribute,
        args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.results``: Define the results of an operation.

        Args:
            names: Attribute ``names`` (string array attribute).
            variadicity: Attribute ``variadicity``.
            args: Operand ``args`` (IRDL handle to an `mlir::Attribute`). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``args``: IRDL handle to an `mlir::Attribute`."""

    @property
    def names(self) -> mlir_python._mlir_python.ArrayAttr:
        """Attribute ``names``: string array attribute."""

    @names.setter
    def names(self, arg: mlir_python._mlir_python.ArrayAttr, /) -> None: ...
    @property
    def variadicity(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``variadicity``."""

    @variadicity.setter
    def variadicity(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...

    OPERATION_NAME: str = "irdl.results"

class TypeOp(mlir_python._mlir_python.Operation):
    """
    ``irdl.type``: Define a new type.

    `irdl.type` defines a new type belonging to the `irdl.dialect` parent.

    The type parameters can be defined with an `irdl.parameters` operation in
    the optional region.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex {
        %0 = irdl.is i32
        %1 = irdl.is i64
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath`. The
    type has a single parameter that should be either `i32` or `i64`.
    """

    def __init__(
        self,
        sym_name: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``irdl.type``: Define a new type.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: region with 1 blocks."""

    OPERATION_NAME: str = "irdl.type"

def load_dialects(module: mlir_python._mlir_python.Module) -> None:
    """
    Define the dialects that ``module`` describes with ``irdl.dialect``
    operations, in ``module``'s context. Their operations, types, and
    attributes can then be created (``Operation.create``, ``Type.parse``)
    and are verified against the IRDL constraints. See
    ``mlir_python.irdl`` for declaring dialects as Python classes.

    Raises:
        MLIRError: If a definition is invalid or a dialect of that name
            is already loaded.
    """
