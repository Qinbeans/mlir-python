"""Typed operations of the MLIR ``cf`` dialect."""

from collections.abc import Sequence

import mlir_python._mlir_python

class AssertOp(mlir_python._mlir_python.Operation):
    """
    ``cf.assert``: Assert operation with message attribute.

    Assert operation at runtime with single boolean operand and an error
    message attribute.
    If the argument is `true` this operation has no effect. Otherwise, the
    program execution will abort. The provided error message may be used by a
    runtime to propagate the error to the user.

    Example:

    ```mlir
    cf.assert %b, "Expected ... to be true"
    ```
    """

    def __init__(
        self,
        arg: mlir_python._mlir_python.Value,
        msg: str,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``cf.assert``: Assert operation with message attribute.

        Args:
            arg: Operand ``arg`` (1-bit signless integer).
            msg: Attribute ``msg`` (string attribute).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """Operand ``arg``: 1-bit signless integer."""

    @property
    def msg(self) -> str:
        """Attribute ``msg``: string attribute."""

    @msg.setter
    def msg(self, arg: str, /) -> None: ...

    OPERATION_NAME: str = "cf.assert"

class BranchOp(mlir_python._mlir_python.Operation):
    """
    ``cf.br``: Branch operation.

    The `cf.br` operation represents a direct branch operation to a given
    block. The operands of this operation are forwarded to the successor block,
    and the number and type of the operands must match the arguments of the
    target block.

    Example:

    ```mlir
    ^bb2:
      %2 = call @someFn()
      cf.br ^bb3(%2 : tensor<*xf32>)
    ^bb3(%3: tensor<*xf32>):
    ```
    """

    def __init__(
        self,
        dest: mlir_python._mlir_python.Block,
        dest_operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``cf.br``: Branch operation.

        Args:
            dest: Successor ``dest`` (any successor).
            dest_operands: Operand ``destOperands`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def dest_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``destOperands``: any type."""

    @property
    def dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``dest``."""

    OPERATION_NAME: str = "cf.br"

class CondBranchOp(mlir_python._mlir_python.Operation):
    """
    ``cf.cond_br``: Conditional branch operation.

    The `cf.cond_br` terminator operation represents a conditional branch on a
    boolean (1-bit integer) value. If the bit is set, then the first destination
    is jumped to; if it is false, the second destination is chosen. The count
    and types of operands must align with the arguments in the corresponding
    target blocks.

    The MLIR conditional branch operation is not allowed to target the entry
    block for a region. The two destinations of the conditional branch operation
    are allowed to be the same.

    The following example illustrates a function with a conditional branch
    operation that targets the same block.

    Example:

    ```mlir
    func.func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
      // Both targets are the same, operands differ
      cf.cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

    ^bb1(%x : i32) :
      return %x : i32
    }
    ```
    """

    def __init__(
        self,
        condition: mlir_python._mlir_python.Value,
        true_dest: mlir_python._mlir_python.Block,
        false_dest: mlir_python._mlir_python.Block,
        true_dest_operands: Sequence[mlir_python._mlir_python.Value] = [],
        false_dest_operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        branch_weights: mlir_python._mlir_python.Attribute | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``cf.cond_br``: Conditional branch operation.

        Args:
            condition: Operand ``condition`` (1-bit signless integer).
            true_dest: Successor ``trueDest`` (any successor).
            false_dest: Successor ``falseDest`` (any successor).
            true_dest_operands: Operand ``trueDestOperands`` (any type). Empty by default.
            false_dest_operands: Operand ``falseDestOperands`` (any type). Empty by default.
            branch_weights: Attribute ``branch_weights`` (i32 dense array attribute). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def condition(self) -> mlir_python._mlir_python.Value:
        """Operand ``condition``: 1-bit signless integer."""

    @property
    def true_dest_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``trueDestOperands``: any type."""

    @property
    def false_dest_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``falseDestOperands``: any type."""

    @property
    def branch_weights(self) -> mlir_python._mlir_python.Attribute | None:
        """Attribute ``branch_weights``: i32 dense array attribute."""

    @branch_weights.setter
    def branch_weights(
        self, arg: mlir_python._mlir_python.Attribute | None
    ) -> None: ...
    @property
    def true_dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``trueDest``."""

    @property
    def false_dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``falseDest``."""

    OPERATION_NAME: str = "cf.cond_br"

class SwitchOp(mlir_python._mlir_python.Operation):
    """
    ``cf.switch``: Switch operation.

    The `cf.switch` terminator operation represents a switch on a signless integer
    value. If the flag matches one of the specified cases, then the
    corresponding destination is jumped to. If the flag does not match any of
    the cases, the default destination is jumped to. The count and types of
    operands must align with the arguments in the corresponding target blocks.

    Example:

    ```mlir
    cf.switch %flag : i32, [
      default: ^bb1(%a : i32),
      42: ^bb1(%b : i32),
      43: ^bb3(%c : i32)
    ]
    ```
    """

    def __init__(
        self,
        flag: mlir_python._mlir_python.Value,
        default_destination: mlir_python._mlir_python.Block,
        case_destinations: Sequence[mlir_python._mlir_python.Block],
        default_operands: Sequence[mlir_python._mlir_python.Value] = [],
        case_operands: Sequence[Sequence[mlir_python._mlir_python.Value]] = [],
        *,
        case_values: mlir_python._mlir_python.DenseIntElementsAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``cf.switch``: Switch operation.

        Args:
            flag: Operand ``flag`` (integer).
            default_destination: Successor ``defaultDestination`` (any successor).
            case_destinations: Successor ``caseDestinations`` (any successor).
            default_operands: Operand ``defaultOperands`` (any type). Empty by default.
            case_operands: Operand ``caseOperands`` (any type). Empty by default.
            case_values: Attribute ``case_values`` (integer elements attribute). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def flag(self) -> mlir_python._mlir_python.Value:
        """Operand ``flag``: integer."""

    @property
    def default_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``defaultOperands``: any type."""

    @property
    def case_operands(self) -> list[list[mlir_python._mlir_python.Value]]:
        """Operand ``caseOperands``: any type."""

    @property
    def case_values(self) -> mlir_python._mlir_python.DenseIntElementsAttr | None:
        """Attribute ``case_values``: integer elements attribute."""

    @case_values.setter
    def case_values(
        self, arg: mlir_python._mlir_python.DenseIntElementsAttr | None
    ) -> None: ...
    @property
    def default_destination(self) -> mlir_python._mlir_python.Block:
        """Successor ``defaultDestination``."""

    @property
    def case_destinations(self) -> list[mlir_python._mlir_python.Block]:
        """Successor ``caseDestinations``."""

    OPERATION_NAME: str = "cf.switch"
