"""Typed operations of the MLIR ``math`` dialect."""

import mlir_python._mlir_python
import mlir_python._mlir_python.arith

class AbsFOp(mlir_python._mlir_python.Operation):
    """
    ``math.absf``: floating point absolute-value operation.

    The `absf` operation computes the absolute value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result
    of the same type.

    Example:

    ```mlir
    // Scalar absolute value.
    %a = math.absf %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.absf``: floating point absolute-value operation.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.absf"

class AbsIOp(mlir_python._mlir_python.Operation):
    """
    ``math.absi``: integer absolute-value operation.

    The `absi` operation computes the absolute value. It takes one operand of
    integer type (i.e., scalar, tensor or vector) and returns one result of the
    same type.

    Example:

    ```mlir
    // Scalar absolute value.
    %a = math.absi %b : i64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.absi``: integer absolute-value operation.

        Result types are inferred.

        Args:
            operand: Operand ``operand`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: signless-integer-like."""

    OPERATION_NAME: str = "math.absi"

class AcosOp(mlir_python._mlir_python.Operation):
    """
    ``math.acos``: arcus cosine of the specified value.

    The `acos` operation computes the arcus cosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar arcus cosine value.
    %a = math.acos %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.acos``: arcus cosine of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.acos"

class AcoshOp(mlir_python._mlir_python.Operation):
    """
    ``math.acosh``: Hyperbolic arcus cosine of the given value.

    Syntax:

    ```
    operation ::= ssa-id `=` `math.acosh` ssa-use `:` type
    ```

    The `acosh` operation computes the arcus cosine of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Hyperbolic arcus cosine of scalar value.
    %a = math.acosh %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.acosh``: Hyperbolic arcus cosine of the given value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.acosh"

class AsinOp(mlir_python._mlir_python.Operation):
    """
    ``math.asin``: arcus sine of the given value.

    Syntax:

    ```
    operation ::= ssa-id `=` `math.asin` ssa-use `:` type
    ```

    The `asin` operation computes the arcus sine of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Arcus sine of scalar value.
    %a = math.asin %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.asin``: arcus sine of the given value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.asin"

class AsinhOp(mlir_python._mlir_python.Operation):
    """
    ``math.asinh``: hyperbolic arcus sine of the given value.

    Syntax:

    ```
    operation ::= ssa-id `=` `math.asinh` ssa-use `:` type
    ```

    The `asinh` operation computes the hyperbolic arcus sine of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Hyperbolic arcus sine of scalar value.
    %a = math.asinh %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.asinh``: hyperbolic arcus sine of the given value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.asinh"

class Atan2Op(mlir_python._mlir_python.Operation):
    """
    ``math.atan2``: 2-argument arcus tangent of the given values.

    The `atan2` operation takes two operands and returns one result, all of
    which must be of the same type.  The operands must be of floating point type
    (i.e., scalar, tensor or vector).

    The 2-argument arcus tangent `atan2(y, x)` returns the angle in the
    Euclidian plane between the positive x-axis and the ray through the point
    (x, y).  It is a generalization of the 1-argument arcus tangent which
    returns the angle on the basis of the ratio y/x.

    See also https://en.wikipedia.org/wiki/Atan2

    Example:

    ```mlir
    // Scalar variant.
    %a = math.atan2 %b, %c : f32
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.atan2``: 2-argument arcus tangent of the given values.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.atan2"

class AtanOp(mlir_python._mlir_python.Operation):
    """
    ``math.atan``: arcus tangent of the given value.

    The `atan` operation computes the arcus tangent of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Arcus tangent of scalar value.
    %a = math.atan %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.atan``: arcus tangent of the given value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.atan"

class AtanhOp(mlir_python._mlir_python.Operation):
    """
    ``math.atanh``: hyperbolic arcus tangent of the given value.

    Syntax:

    ```
    operation ::= ssa-id `=` `math.atanh` ssa-use `:` type
    ```

    The `atanh` operation computes the hyperbolic arcus tangent of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Hyperbolic arcus tangent of scalar value.
    %a = math.atanh %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.atanh``: hyperbolic arcus tangent of the given value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.atanh"

class CbrtOp(mlir_python._mlir_python.Operation):
    """
    ``math.cbrt``: cube root of the specified value.

    The `cbrt` operation computes the cube root. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result
    of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar cube root value.
    %a = math.cbrt %b : f64
    ```

    Note: This op is not equivalent to powf(..., 1/3.0).
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.cbrt``: cube root of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.cbrt"

class CeilOp(mlir_python._mlir_python.Operation):
    """
    ``math.ceil``: ceiling of the specified value.

    The `ceil` operation computes the ceiling of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar ceiling value.
    %a = math.ceil %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.ceil``: ceiling of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.ceil"

class ClampFOp(mlir_python._mlir_python.Operation):
    """
    ``math.clampf``: floating point clamping operation.

    The `clampf` operation takes three operands and returns one result, each of
    these is required to be the same type. Operands must be of floating point type
    (i.e., scalar, tensor or vector).

    The semantics of the operation are described by:
    ```
      clampf(value, min, max) = maxf(minf(value, min), max)
    ```

    Example:

    ```mlir
    %d = math.clampf %value to [%min, %max] : f64
    ```
    """

    def __init__(
        self,
        value: mlir_python._mlir_python.Value,
        min: mlir_python._mlir_python.Value,
        max: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.clampf``: floating point clamping operation.

        Result types are inferred.

        Args:
            value: Operand ``value`` (floating-point-like).
            min: Operand ``min`` (floating-point-like).
            max: Operand ``max`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: floating-point-like."""

    @property
    def min(self) -> mlir_python._mlir_python.Value:
        """Operand ``min``: floating-point-like."""

    @property
    def max(self) -> mlir_python._mlir_python.Value:
        """Operand ``max``: floating-point-like."""

    @property
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.clampf"

class CopySignOp(mlir_python._mlir_python.Operation):
    """
    ``math.copysign``: A copysign operation.

    The `copysign` returns a value with the magnitude of the first operand and
    the sign of the second operand. It takes two operands and returns one result of
    the same type. The operands must be of floating point type (i.e., scalar,
    tensor or vector). It has no standard attributes.

    Example:

    ```mlir
    // Scalar copysign value.
    %a = math.copysign %b, %c : f64
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.copysign``: A copysign operation.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.copysign"

class CosOp(mlir_python._mlir_python.Operation):
    """
    ``math.cos``: cosine of the specified value.

    The `cos` operation computes the cosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar cosine value.
    %a = math.cos %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.cos``: cosine of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.cos"

class CoshOp(mlir_python._mlir_python.Operation):
    """
    ``math.cosh``: hyperbolic cosine of the specified value.

    The `cosh` operation computes the hyperbolic cosine. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar hyperbolic cosine value.
    %a = math.cosh %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.cosh``: hyperbolic cosine of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.cosh"

class CountLeadingZerosOp(mlir_python._mlir_python.Operation):
    """
    ``math.ctlz``: counts the leading zeros an integer value.

    The `ctlz` operation computes the number of leading zeros of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    ```mlir
    // Scalar ctlz function value.
    %a = math.ctlz %b : i32
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.ctlz``: counts the leading zeros an integer value.

        Result types are inferred.

        Args:
            operand: Operand ``operand`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: signless-integer-like."""

    OPERATION_NAME: str = "math.ctlz"

class CountTrailingZerosOp(mlir_python._mlir_python.Operation):
    """
    ``math.cttz``: counts the trailing zeros an integer value.

    The `cttz` operation computes the number of trailing zeros of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    ```mlir
    // Scalar cttz function value.
    %a = math.cttz %b : i32
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.cttz``: counts the trailing zeros an integer value.

        Result types are inferred.

        Args:
            operand: Operand ``operand`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: signless-integer-like."""

    OPERATION_NAME: str = "math.cttz"

class CtPopOp(mlir_python._mlir_python.Operation):
    """
    ``math.ctpop``: counts the number of set bits of an integer value.

    The `ctpop` operation computes the number of set bits of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    ```mlir
    // Scalar ctpop function value.
    %a = math.ctpop %b : i32
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.ctpop``: counts the number of set bits of an integer value.

        Result types are inferred.

        Args:
            operand: Operand ``operand`` (signless-integer-like).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: signless-integer-like."""

    OPERATION_NAME: str = "math.ctpop"

class ErfOp(mlir_python._mlir_python.Operation):
    """
    ``math.erf``: error function of the specified value.

    The `erf` operation computes the error function. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar error function value.
    %a = math.erf %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.erf``: error function of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.erf"

class ErfcOp(mlir_python._mlir_python.Operation):
    """
    ``math.erfc``: complementary error function of the specified value.

    The `erfc` operation computes the complementary error function, defined as
    1-erf(x). This function is part of libm and is needed for accuracy, since
    simply calculating 1-erf(x) when x is close to 1 will give inaccurate results.
    It takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar error function value.
    %a = math.erfc %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.erfc``: complementary error function of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.erfc"

class Exp2Op(mlir_python._mlir_python.Operation):
    """
    ``math.exp2``: base-2 exponential of the specified value.

    The `exp` operation takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no standard
    attributes.

    Example:

    ```mlir
    // Scalar natural exponential.
    %a = math.exp2 %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.exp2``: base-2 exponential of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.exp2"

class ExpM1Op(mlir_python._mlir_python.Operation):
    """
    ``math.expm1``: base-e exponential of the specified value minus 1.

    expm1(x) := exp(x) - 1

    The `expm1` operation takes one operand of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar natural exponential minus 1.
    %a = math.expm1 %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.expm1``: base-e exponential of the specified value minus 1.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.expm1"

class ExpOp(mlir_python._mlir_python.Operation):
    """
    ``math.exp``: base-e exponential of the specified value.

    The `exp` operation takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no standard
    attributes.

    Example:

    ```mlir
    // Scalar natural exponential.
    %a = math.exp %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.exp``: base-e exponential of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.exp"

class FPowIOp(mlir_python._mlir_python.Operation):
    """
    ``math.fpowi``: floating point raised to the signed integer power.

    The `fpowi` operation takes a `base` operand of floating point type
    (i.e. scalar, tensor or vector) and a `power` operand of integer type
    (also scalar, tensor or vector) and returns one result of the same type
    as `base`. The result is `base` raised to the power of `power`.
    The operation is elementwise for non-scalars, e.g.:

    ```mlir
    %v = math.fpowi %base, %power : vector<2xf32>, vector<2xi32
    ```

    The result is a vector of:

    ```
    [<math.fpowi %base[0], %power[0]>, <math.fpowi %base[1], %power[1]>]
    ```

    Example:

    ```mlir
    // Scalar exponentiation.
    %a = math.fpowi %base, %power : f64, i32
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.fpowi``: floating point raised to the signed integer power.

        Result types are inferred.

        Args:
            lhs: Operand ``lhs`` (floating-point-like).
            rhs: Operand ``rhs`` (signless-integer-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``lhs``: floating-point-like."""

    @property
    def rhs(self) -> mlir_python._mlir_python.Value:
        """Operand ``rhs``: signless-integer-like."""

    @property
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.fpowi"

class FloorOp(mlir_python._mlir_python.Operation):
    """
    ``math.floor``: floor of the specified value.

    The `floor` operation computes the floor of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar floor value.
    %a = math.floor %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.floor``: floor of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.floor"

class FmaOp(mlir_python._mlir_python.Operation):
    """
    ``math.fma``: floating point fused multipy-add operation.

    The `fma` operation takes three operands and returns one result, each of
    these is required to be the same type. Operands must be of floating point type
    (i.e., scalar, tensor or vector).

    Example:

    ```mlir
    // Scalar fused multiply-add: d = a*b + c
    %d = math.fma %a, %b, %c : f64
    ```

    The semantics of the operation correspond to those of the `llvm.fma`
    [intrinsic](https://llvm.org/docs/LangRef.html#llvm-fma-intrinsic). In the
    particular case of lowering to LLVM, this is guaranteed to lower
    to the `llvm.fma.*` intrinsic.
    """

    def __init__(
        self,
        a: mlir_python._mlir_python.Value,
        b: mlir_python._mlir_python.Value,
        c: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.fma``: floating point fused multipy-add operation.

        Result types are inferred.

        Args:
            a: Operand ``a`` (floating-point-like).
            b: Operand ``b`` (floating-point-like).
            c: Operand ``c`` (floating-point-like).
            fastmath: Attribute ``fastmath`` (flags of ``FastMathFlags``). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def a(self) -> mlir_python._mlir_python.Value:
        """Operand ``a``: floating-point-like."""

    @property
    def b(self) -> mlir_python._mlir_python.Value:
        """Operand ``b``: floating-point-like."""

    @property
    def c(self) -> mlir_python._mlir_python.Value:
        """Operand ``c``: floating-point-like."""

    @property
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.fma"

class IPowIOp(mlir_python._mlir_python.Operation):
    """
    ``math.ipowi``: signed integer raised to the power of operation.

    The `ipowi` operation takes two operands of integer type (i.e., scalar,
    tensor or vector) and returns one result of the same type. Operands
    must have the same type.

    Example:

    ```mlir
    // Scalar signed integer exponentiation.
    %a = math.ipowi %b, %c : i32
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
        Create ``math.ipowi``: signed integer raised to the power of operation.

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

    OPERATION_NAME: str = "math.ipowi"

class IsFiniteOp(mlir_python._mlir_python.Operation):
    """
    ``math.isfinite``: returns true if the operand classifies as finite.

    Determines if the given floating-point number has finite value i.e. it
    is normal, subnormal or zero, but not infinite or NaN.

    Example:

    ```mlir
    %f = math.isfinite %a : f32
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.isfinite``: returns true if the operand classifies as finite.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.isfinite"

class IsInfOp(mlir_python._mlir_python.Operation):
    """
    ``math.isinf``: returns true if the operand classifies as infinite.

    Determines if the given floating-point number is positive or negative
    infinity.

    Example:

    ```mlir
    %f = math.isinf %a : f32
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.isinf``: returns true if the operand classifies as infinite.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.isinf"

class IsNaNOp(mlir_python._mlir_python.Operation):
    """
    ``math.isnan``: returns true if the operand classifies as NaN.

    Determines if the given floating-point number is a not-a-number (NaN)
    value.

    Example:

    ```mlir
    %f = math.isnan %a : f32
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.isnan``: returns true if the operand classifies as NaN.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.isnan"

class IsNormalOp(mlir_python._mlir_python.Operation):
    """
    ``math.isnormal``: returns true if the operand classifies as normal.

    Determines if the given floating-point number is normal, i.e. is neither
    zero, subnormal, infinite, nor NaN.

    Example:

    ```mlir
    %f = math.isnormal %a : f32
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.isnormal``: returns true if the operand classifies as normal.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.isnormal"

class Log10Op(mlir_python._mlir_python.Operation):
    """
    ``math.log10``: base-10 logarithm of the specified value.

    Computes the base-10 logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    ```mlir
    // Scalar log10 operation.
    %y = math.log10 %x : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.log10``: base-10 logarithm of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.log10"

class Log1pOp(mlir_python._mlir_python.Operation):
    """
    ``math.log1p``: Computes the natural logarithm of one plus the given value.

    Computes the base-e logarithm of one plus the given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.

    log1p(x) := log(1 + x)

    Example:

    ```mlir
    // Scalar log1p operation.
    %y = math.log1p %x : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.log1p``: Computes the natural logarithm of one plus the given value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.log1p"

class Log2Op(mlir_python._mlir_python.Operation):
    """
    ``math.log2``: base-2 logarithm of the specified value.

    Computes the base-2 logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    ```mlir
    // Scalar log2 operation.
    %y = math.log2 %x : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.log2``: base-2 logarithm of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.log2"

class LogOp(mlir_python._mlir_python.Operation):
    """
    ``math.log``: base-e logarithm of the specified value.

    Computes the base-e logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    ```mlir
    // Scalar log operation.
    %y = math.log %x : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.log``: base-e logarithm of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.log"

class PowFOp(mlir_python._mlir_python.Operation):
    """
    ``math.powf``: floating point raised to the power of operation.

    The `powf` operation takes two operands of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. Operands
    must have the same type.

    Example:

    ```mlir
    // Scalar exponentiation.
    %a = math.powf %b, %c : f64
    ```
    """

    def __init__(
        self,
        lhs: mlir_python._mlir_python.Value,
        rhs: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.powf``: floating point raised to the power of operation.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.powf"

class RoundEvenOp(mlir_python._mlir_python.Operation):
    """
    ``math.roundeven``: round of the specified value with halfway cases to even.

    The `roundeven` operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.  The
    operation rounds the argument to the nearest integer value in floating-point
    format, rounding halfway cases to even, regardless of the current
    rounding direction.

    Example:

    ```mlir
    // Scalar round operation.
    %a = math.roundeven %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.roundeven``: round of the specified value with halfway cases to even.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.roundeven"

class RoundOp(mlir_python._mlir_python.Operation):
    """
    ``math.round``: round of the specified value.

    The `round` operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.  The
    operation rounds the argument to the nearest integer value in floating-point
    format, rounding halfway cases away from zero, regardless of the current
    rounding direction.

    Example:

    ```mlir
    // Scalar round operation.
    %a = math.round %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.round``: round of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.round"

class RsqrtOp(mlir_python._mlir_python.Operation):
    """
    ``math.rsqrt``: reciprocal of sqrt (1 / sqrt of the specified value).

    The `rsqrt` operation computes the reciprocal of the square root. It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar reciprocal square root value.
    %a = math.rsqrt %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.rsqrt``: reciprocal of sqrt (1 / sqrt of the specified value).

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.rsqrt"

class SinOp(mlir_python._mlir_python.Operation):
    """
    ``math.sin``: sine of the specified value.

    The `sin` operation computes the sine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    // Scalar sine value.
    %a = math.sin %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.sin``: sine of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.sin"

class SincosOp(mlir_python._mlir_python.Operation):
    """
    ``math.sincos``: sine and cosine of the specified value.

    The `sincos` operation computes both the sine and cosine of a given value
    simultaneously. It takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns two results of the same type. This operation
    can be more efficient than computing sine and cosine separately when both
    values are needed.

    Example:

    ```mlir
    // Scalar sine and cosine values.
    %sin, %cos = math.sincos %input : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.sincos``: sine and cosine of the specified value.

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
    def sin(self) -> mlir_python._mlir_python.OpResult:
        """Result ``sin``: floating-point-like."""

    @property
    def cos(self) -> mlir_python._mlir_python.OpResult:
        """Result ``cos``: floating-point-like."""

    @property
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.sincos"

class SinhOp(mlir_python._mlir_python.Operation):
    """
    ``math.sinh``: hyperbolic sine of the specified value.

    The `sinh` operation computes the hyperbolic sine. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar hyperbolic sine value.
    %a = math.sinh %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.sinh``: hyperbolic sine of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.sinh"

class SqrtOp(mlir_python._mlir_python.Operation):
    """
    ``math.sqrt``: sqrt of the specified value.

    The `sqrt` operation computes the square root. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar square root value.
    %a = math.sqrt %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.sqrt``: sqrt of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.sqrt"

class TanOp(mlir_python._mlir_python.Operation):
    """
    ``math.tan``: tangent of the specified value.

    The `tan` operation computes the tangent. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar tangent value.
    %a = math.tan %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.tan``: tangent of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.tan"

class TanhOp(mlir_python._mlir_python.Operation):
    """
    ``math.tanh``: hyperbolic tangent of the specified value.

    The `tanh` operation computes the hyperbolic tangent. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    // Scalar hyperbolic tangent value.
    %a = math.tanh %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.tanh``: hyperbolic tangent of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.tanh"

class TruncOp(mlir_python._mlir_python.Operation):
    """
    ``math.trunc``: trunc of the specified value.

    The `trunc` operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.
    The operation always rounds to the nearest integer not larger in magnitude
    than the operand, regardless of the current rounding direction.

    Example:

    ```mlir
    // Scalar trunc operation.
    %a = math.trunc %b : f64
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        fastmath: mlir_python._mlir_python.arith.FastMathFlags = mlir_python._mlir_python.arith.FastMathFlags.NONE,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``math.trunc``: trunc of the specified value.

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
    def fastmath(self) -> mlir_python._mlir_python.arith.FastMathFlags:
        """Attribute ``fastmath``: flags of ``FastMathFlags``."""

    @fastmath.setter
    def fastmath(
        self, arg: mlir_python._mlir_python.arith.FastMathFlags, /
    ) -> None: ...

    OPERATION_NAME: str = "math.trunc"
