"""Define MLIR dialects at runtime, as Python classes (IRDL, no C++).

Declare a dialect's types and operations, then create, inspect, verify,
and rewrite them like any other operation::

    from mlir_python import irdl

    bifrost = irdl.Dialect("bifrost")

    Mutex = bifrost.type("mutex")                    # !bifrost.mutex

    @bifrost.operation
    class Guard(irdl.Op):
        \"\"\"Locks ``mutex``; ``value`` is readable until the lock ends.\"\"\"

        mutex = irdl.Operand(irdl.BaseOf(Mutex))
        value = irdl.Result(irdl.Any())

    with ir.Context(), ir.Location.unknown(), ir.InsertionPoint(block):
        guard = Guard.create(mutex, value=ir.IntegerType(32))
        guard.value                                   # the result
        guard.operation.verify()                      # checks the constraints

Operations are verified against the declared constraints (operand, result,
and attribute types, counts, regions). The dialect is loaded into a context
the first time it is needed there (``Dialect.load`` does it explicitly).
``Op.all(root)`` finds a dialect's operations, e.g. to check or lower them in
a ``passes.PythonPass``.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import ClassVar, Self, overload

from . import _mlir_python as ir
from .dialects import irdl as irdl_ops

__all__ = [
    "AllOf",
    "Any",
    "AnyOf",
    "Attr",
    "BaseOf",
    "Constraint",
    "Dialect",
    "Is",
    "Op",
    "Operand",
    "OptionalOperand",
    "OptionalResult",
    "Parametric",
    "Region",
    "Result",
    "TypeDef",
    "VariadicOperand",
    "VariadicResult",
]


# -- constraints -------------------------------------------------------------------


class Constraint:
    """What a type or attribute must be: ``Any()``, ``Is(t)``, ``BaseOf(C)``,
    ``AnyOf(...)``, ``AllOf(...)``, or ``Parametric(T, ...)``."""

    def _emit(self, writer: _Writer) -> str:
        raise NotImplementedError


class Any(Constraint):
    """Anything."""

    def _emit(self, writer: _Writer) -> str:
        return writer.define("irdl.any")


class Is(Constraint):
    """Exactly ``value``, a type or attribute (or a function returning one,
    to create it only once a context exists): ``Is(ir.IntegerType(64))``."""

    def __init__(self, value: ir.Type | ir.Attribute | Callable[[], object]) -> None:
        self.value = value

    def resolve(self) -> ir.Type | ir.Attribute:
        value = self.value() if callable(self.value) else self.value
        if not isinstance(value, (ir.Type, ir.Attribute)):
            raise TypeError(f"Is() needs a type or attribute, got {value!r}")
        return value

    def _emit(self, writer: _Writer) -> str:
        return writer.define(f"irdl.is {self.resolve()}")


def _builtin_bases() -> dict[type, str]:
    names = {
        "IntegerType": "!builtin.integer",
        "IndexType": "!builtin.index",
        "F16Type": "!builtin.f16",
        "BF16Type": "!builtin.bf16",
        "F32Type": "!builtin.f32",
        "F64Type": "!builtin.f64",
        "ComplexType": "!builtin.complex",
        "MemRefType": "!builtin.memref",
        "RankedTensorType": "!builtin.tensor",
        "UnrankedTensorType": "!builtin.unranked_tensor",
        "VectorType": "!builtin.vector",
        "FunctionType": "!builtin.function",
        "TupleType": "!builtin.tuple",
        "NoneType": "!builtin.none",
        "IntegerAttr": "#builtin.integer",
        "FloatAttr": "#builtin.float",
        "StringAttr": "#builtin.string",
        "ArrayAttr": "#builtin.array",
        "DictAttr": "#builtin.dictionary",
        "TypeAttr": "#builtin.type",
        "UnitAttr": "#builtin.unit",
        "SymbolRefAttr": "#builtin.symbol_ref",
    }
    return {getattr(ir, cls): name for cls, name in names.items() if hasattr(ir, cls)}


BUILTIN_BASES: dict[type, str] = _builtin_bases()


class BaseOf(Constraint):
    """Any instance of ``kind``: a builtin type or attribute class
    (``BaseOf(ir.IntegerType)`` for any integer type), or a type of a
    runtime dialect (``BaseOf(Mutex)``, whatever its parameters)."""

    def __init__(self, kind: type | TypeDef) -> None:
        if isinstance(kind, type) and kind not in BUILTIN_BASES:
            known = ", ".join(sorted(c.__name__ for c in BUILTIN_BASES))
            raise TypeError(f"BaseOf() takes a TypeDef or one of {known}")
        self.kind = kind

    def _emit(self, writer: _Writer) -> str:
        if isinstance(self.kind, TypeDef):
            return writer.define(f"irdl.base {self.kind.symbol}")
        return writer.define(f'irdl.base "{BUILTIN_BASES[self.kind]}"')


class AnyOf(Constraint):
    """Satisfies at least one of ``constraints``."""

    def __init__(self, *constraints: Constraint) -> None:
        self.constraints = constraints

    def _emit(self, writer: _Writer) -> str:
        names = [c._emit(writer) for c in self.constraints]
        return writer.define(f"irdl.any_of({', '.join(names)})")


class AllOf(Constraint):
    """Satisfies every one of ``constraints``."""

    def __init__(self, *constraints: Constraint) -> None:
        self.constraints = constraints

    def _emit(self, writer: _Writer) -> str:
        names = [c._emit(writer) for c in self.constraints]
        return writer.define(f"irdl.all_of({', '.join(names)})")


class Parametric(Constraint):
    """A ``kind`` (a runtime dialect type) whose parameters satisfy
    ``parameters``: ``Parametric(Box, Is(ir.IntegerType(32)))``."""

    def __init__(self, kind: TypeDef, *parameters: Constraint) -> None:
        self.kind = kind
        self.parameters = parameters

    def _emit(self, writer: _Writer) -> str:
        names = [c._emit(writer) for c in self.parameters]
        return writer.define(f"irdl.parametric {self.kind.symbol}<{', '.join(names)}>")


class _Writer:
    """Renders constraint definitions into one IRDL body."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.counter = itertools.count()

    def define(self, text: str) -> str:
        name = f"%c{next(self.counter)}"
        self.lines.append(f"{name} = {text}")
        return name


# -- operation declarations --------------------------------------------------------


class _Field:
    """A declared part of an operation, named by its attribute name."""

    name = ""
    order = itertools.count()

    def __init__(self) -> None:
        self._order = next(_Field.order)

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name


class _ValueField(_Field):
    VARIADICITY = "single"

    def __init__(self, constraint: Constraint | None = None) -> None:
        super().__init__()
        self.constraint = constraint if constraint is not None else Any()


class Operand(_ValueField):
    """One operand satisfying ``constraint`` (``Any()`` by default)."""

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...
    @overload
    def __get__(self, instance: Op, owner: type) -> ir.Value: ...
    def __get__(self, instance: Op | None, owner: type) -> Self | ir.Value:
        if instance is None:
            return self
        (value,) = instance._operand_group(self.name)
        return value


class OptionalOperand(_ValueField):
    """An operand that may be left out (``None``)."""

    VARIADICITY = "optional"

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...
    @overload
    def __get__(self, instance: Op, owner: type) -> ir.Value | None: ...
    def __get__(self, instance: Op | None, owner: type) -> Self | ir.Value | None:
        if instance is None:
            return self
        group = instance._operand_group(self.name)
        return group[0] if group else None


class VariadicOperand(_ValueField):
    """Any number of operands, each satisfying ``constraint``."""

    VARIADICITY = "variadic"

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...
    @overload
    def __get__(self, instance: Op, owner: type) -> list[ir.Value]: ...
    def __get__(self, instance: Op | None, owner: type) -> Self | list[ir.Value]:
        if instance is None:
            return self
        return instance._operand_group(self.name)


class Result(_ValueField):
    """One result satisfying ``constraint``. When the constraint is
    ``Is(type)``, ``create`` knows the result type; otherwise pass it."""

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...
    @overload
    def __get__(self, instance: Op, owner: type) -> ir.OpResult: ...
    def __get__(self, instance: Op | None, owner: type) -> Self | ir.OpResult:
        if instance is None:
            return self
        (value,) = instance._result_group(self.name)
        return value


class OptionalResult(_ValueField):
    """A result that may be left out (pass its type as ``None``)."""

    VARIADICITY = "optional"

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...
    @overload
    def __get__(self, instance: Op, owner: type) -> ir.OpResult | None: ...
    def __get__(self, instance: Op | None, owner: type) -> Self | ir.OpResult | None:
        if instance is None:
            return self
        group = instance._result_group(self.name)
        return group[0] if group else None


class VariadicResult(_ValueField):
    """Any number of results (pass a list of types)."""

    VARIADICITY = "variadic"

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...
    @overload
    def __get__(self, instance: Op, owner: type) -> list[ir.OpResult]: ...
    def __get__(self, instance: Op | None, owner: type) -> Self | list[ir.OpResult]:
        if instance is None:
            return self
        return instance._result_group(self.name)


class Attr(_Field):
    """A required attribute satisfying ``constraint``."""

    def __init__(self, constraint: Constraint | None = None) -> None:
        super().__init__()
        self.constraint = constraint if constraint is not None else Any()

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...
    @overload
    def __get__(self, instance: Op, owner: type) -> ir.Attribute: ...
    def __get__(self, instance: Op | None, owner: type) -> Self | ir.Attribute:
        if instance is None:
            return self
        return instance.operation.attributes[self.name]


class Region(_Field):
    """A region (created empty; add blocks to it)."""

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...
    @overload
    def __get__(self, instance: Op, owner: type) -> ir.Region: ...
    def __get__(self, instance: Op | None, owner: type) -> Self | ir.Region:
        if instance is None:
            return self
        index = [r.name for r in type(instance)._regions()].index(self.name)
        return instance.operation.regions[index]


class Op:
    """Base class of runtime dialect operations; declare fields with
    ``Operand``, ``Result`` (and their ``Optional``/``Variadic`` forms),
    ``Attr``, and ``Region``, and register the class with
    ``@dialect.operation``. An instance is a typed view of an operation."""

    NAME: ClassVar[str] = ""
    """The operation's full name, e.g. ``"bifrost.guard"``."""
    DIALECT: ClassVar[Dialect | None] = None

    def __init__(self, operation: ir.Operation) -> None:
        """View ``operation``, which must be one of this class.

        Raises:
            TypeError: If it is another operation.
        """
        if operation.name != self.NAME:
            raise TypeError(f"expected {self.NAME}, got {operation.name}")
        self.operation = operation
        """The viewed operation."""

    # -- declared fields ----------------------------------------------------------

    @classmethod
    def _fields[F: _Field](cls, kinds: tuple[type[F], ...]) -> list[F]:
        found: dict[str, F] = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, kinds):
                    found[name] = value
        return sorted(found.values(), key=lambda f: f._order)

    @classmethod
    def _operands(cls) -> list[_ValueField]:
        return cls._fields((Operand, OptionalOperand, VariadicOperand))

    @classmethod
    def _results(cls) -> list[_ValueField]:
        return cls._fields((Result, OptionalResult, VariadicResult))

    @classmethod
    def _attributes(cls) -> list[Attr]:
        return cls._fields((Attr,))

    @classmethod
    def _regions(cls) -> list[Region]:
        return cls._fields((Region,))

    @staticmethod
    def _segmented(fields: list[_ValueField]) -> bool:
        return sum(f.VARIADICITY != "single" for f in fields) > 1

    def _groups[V](
        self, fields: list[_ValueField], values: list[V], attribute: str
    ) -> dict[str, list[V]]:
        if self._segmented(fields):
            sizes_attr = self.operation.attributes[attribute]
            if not isinstance(sizes_attr, ir.DenseI32ArrayAttr):
                raise TypeError(f"{self.NAME}: '{attribute}' must be array<i32: ...>")
            sizes = sizes_attr.values
        else:
            variable = len(values) - sum(f.VARIADICITY == "single" for f in fields)
            sizes = [1 if f.VARIADICITY == "single" else variable for f in fields]
        groups: dict[str, list[V]] = {}
        start = 0
        for field, size in zip(fields, sizes, strict=True):
            groups[field.name] = values[start : start + size]
            start += size
        return groups

    def _operand_group(self, name: str) -> list[ir.Value]:
        operands = list(self.operation.operands)
        return self._groups(type(self)._operands(), operands, "operandSegmentSizes")[
            name
        ]

    def _result_group(self, name: str) -> list[ir.OpResult]:
        results = list(self.operation.results)
        return self._groups(type(self)._results(), results, "resultSegmentSizes")[name]

    # -- creating, finding ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        *operands: ir.Value | Sequence[ir.Value] | None,
        location: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
        **fields: ir.Type | Sequence[ir.Type] | ir.Attribute | None,
    ) -> Self:
        """Create the operation. Operands are positional, in declaration
        order (a list for a variadic operand, ``None`` to leave out an
        optional one); result types and attributes are keywords, named like
        their fields (a result typed ``Is(t)`` can be left out).

        Raises:
            TypeError: For missing, extra, or misplaced operands or fields.
            MLIRError: If the operation does not satisfy its constraints.
        """
        declared_operands = cls._operands()
        if len(operands) != len(declared_operands):
            names = ", ".join(f.name for f in declared_operands) or "none"
            raise TypeError(
                f"{cls.__name__}.create() takes {len(declared_operands)} operands "
                f"({names}), got {len(operands)}"
            )
        operand_values: list[ir.Value] = []
        operand_sizes: list[int] = []
        for field, given in zip(declared_operands, operands, strict=True):
            group = _group(field, given, f"operand '{field.name}'", ir.Value)
            operand_values += group
            operand_sizes.append(len(group))
        result_types: list[ir.Type] = []
        result_sizes: list[int] = []
        for field in cls._results():
            if field.name in fields:
                given_type = fields.pop(field.name)
            elif field.VARIADICITY == "single" and isinstance(field.constraint, Is):
                given_type = field.constraint.resolve()
            else:
                raise TypeError(
                    f"{cls.__name__}.create() needs the type of '{field.name}'"
                )
            group = _group(field, given_type, f"result '{field.name}'", ir.Type)
            result_types += group
            result_sizes.append(len(group))
        attributes: dict[str, ir.Attribute] = {}
        for field in cls._attributes():
            value = fields.pop(field.name, None)
            if not isinstance(value, ir.Attribute):
                raise TypeError(
                    f"{cls.__name__}.create() needs the attribute '{field.name}'"
                )
            attributes[field.name] = value
        if fields:
            raise TypeError(
                f"{cls.__name__}.create() got unknown fields: {', '.join(fields)}"
            )
        if cls._segmented(cls._operands()):
            attributes["operandSegmentSizes"] = ir.DenseI32ArrayAttr(operand_sizes)
        if cls._segmented(cls._results()):
            attributes["resultSegmentSizes"] = ir.DenseI32ArrayAttr(result_sizes)
        context = ir.Context.current()
        if context is None and operand_values:
            context = operand_values[0].type.context
        if context is None:
            raise TypeError(f"{cls.__name__}.create() needs an active ir.Context")
        dialect = cls.DIALECT
        assert dialect is not None, f"{cls.__name__} is not in a dialect"
        dialect.load(context)
        operation = ir.Operation.create(
            cls.NAME,
            results=result_types,
            operands=operand_values,
            attributes=attributes,
            regions=len(cls._regions()),
            location=location,
            ip=ip,
            context=context,
        )
        return cls(operation)

    @classmethod
    def matches(cls, operation: ir.Operation) -> bool:
        """Whether ``operation`` is one of this class."""
        return operation.name == cls.NAME

    @classmethod
    def all(cls, root: ir.Operation) -> list[Self]:
        """Every operation of this class nested in ``root`` (and ``root``
        itself), in program order."""
        found: list[Self] = []

        def visit(operation: ir.Operation) -> None:
            if operation.name == cls.NAME:
                found.append(cls(operation))

        root.walk(visit, ir.WalkOrder.PRE_ORDER)
        return found

    def replace_with(self, values: Sequence[ir.Value]) -> None:
        """Replace every use of the results with ``values`` and erase the
        operation (for lowering it to other operations)."""
        results = list(self.operation.results)
        if len(values) != len(results):
            raise ValueError(
                f"{self.NAME} has {len(results)} results, got {len(values)}"
            )
        for result, value in zip(results, values, strict=True):
            result.replace_all_uses_with(value)
        self.operation.erase()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.operation})"


def _group[V](field: _ValueField, given: object, what: str, kind: type[V]) -> list[V]:
    """The values of one operand or result group, checked for its variadicity."""
    if field.VARIADICITY == "variadic":
        if not isinstance(given, Sequence) or isinstance(given, str):
            raise TypeError(f"{what} is variadic; pass a list")
        items = list(given)
    elif given is None:
        if field.VARIADICITY != "optional":
            raise TypeError(f"{what} is required")
        items = []
    else:
        items = [given]
    for item in items:
        if not isinstance(item, kind):
            raise TypeError(f"{what} must be {kind.__name__}, got {item!r}")
    return items  # type: ignore[return-value]


# -- types and dialects --------------------------------------------------------------


class TypeDef:
    """A type of a runtime dialect, created by ``Dialect.type``. Call it with
    the parameters to get the type: ``Box(ir.IntegerType(32))``."""

    def __init__(
        self, dialect: Dialect, name: str, parameters: Mapping[str, Constraint]
    ) -> None:
        self.dialect = dialect
        self.name = name
        self.parameters = dict(parameters)

    @property
    def symbol(self) -> str:
        """The IRDL symbol, ``@dialect::@name``."""
        return f"@{self.dialect.name}::@{self.name}"

    def __call__(
        self, *parameters: ir.Type | ir.Attribute, context: ir.Context | None = None
    ) -> ir.Type:
        """The type with ``parameters`` (types or attributes, in order).

        Raises:
            TypeError: For the wrong number of parameters.
            ValueError: If they do not satisfy the parameter constraints.
        """
        if len(parameters) != len(self.parameters):
            raise TypeError(
                f"!{self.dialect.name}.{self.name} takes {len(self.parameters)} "
                f"parameters, got {len(parameters)}"
            )
        context = context or ir.Context.current()
        if context is None:
            raise TypeError("creating a type needs an active ir.Context")
        self.dialect.load(context)
        text = f"!{self.dialect.name}.{self.name}"
        if parameters:
            text += "<" + ", ".join(str(p) for p in parameters) + ">"
        return ir.Type.parse(text, context=context)

    def matches(self, value: ir.Type) -> bool:
        """Whether ``value`` is this type (with any parameters)."""
        return (
            re.fullmatch(
                rf"!{re.escape(self.dialect.name)}\.{re.escape(self.name)}(<.*>)?",
                str(value),
            )
            is not None
        )

    def __repr__(self) -> str:
        return f"TypeDef(!{self.dialect.name}.{self.name})"


class _Register:
    """Registers an ``Op`` subclass with a dialect (``@dialect.operation``)."""

    def __init__(self, dialect: Dialect, name: str | None) -> None:
        self.dialect = dialect
        self.name = name

    def __call__[O: Op](self, op_class: type[O]) -> type[O]:
        dialect = self.dialect
        op_name = (
            self.name or re.sub(r"(?<!^)(?=[A-Z])", "_", op_class.__name__).lower()
        )
        dialect._check_open(op_name)
        if not issubclass(op_class, Op):
            raise TypeError(f"{op_class.__name__} must subclass irdl.Op")
        op_class.NAME = f"{dialect.name}.{op_name}"
        op_class.DIALECT = dialect
        dialect.operations[op_name] = op_class
        return op_class


class Dialect:
    """A dialect defined at runtime. Declare its types with ``type`` and its
    operations with ``@operation``, then use them; the dialect is loaded into
    each context on first use."""

    def __init__(self, name: str) -> None:
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
            raise ValueError(f"dialect names are lowercase identifiers, got {name!r}")
        self.name = name
        self.types: dict[str, TypeDef] = {}
        self.operations: dict[str, type[Op]] = {}
        self._frozen = False

    def type(self, name: str, **parameters: Constraint) -> TypeDef:
        """Declare the type ``!dialect.name``, with parameters named and
        constrained by ``parameters`` (in order)."""
        self._check_open(name)
        definition = TypeDef(self, name, parameters)
        self.types[name] = definition
        return definition

    @overload
    def operation[O: Op](self, cls: type[O], /) -> type[O]: ...
    @overload
    def operation(self, /, *, name: str) -> _Register: ...
    def operation[O: Op](
        self, cls: type[O] | None = None, /, *, name: str | None = None
    ) -> type[O] | _Register:
        """Declare an operation from an ``Op`` subclass, named
        ``dialect.snake_case_class_name`` unless ``name`` is given
        (``@dialect.operation(name="double")``)."""
        register = _Register(self, name)
        return register(cls) if cls is not None else register

    def _check_open(self, name: str) -> None:
        if self._frozen:
            raise RuntimeError(
                f"dialect {self.name} is already in use; declare '{name}' before "
                "its first use"
            )
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
            raise ValueError(f"names are lowercase identifiers, got {name!r}")
        if name in self.types or name in self.operations:
            raise ValueError(f"{self.name} already defines '{name}'")

    def definition(self) -> str:
        """The dialect as IRDL (MLIR's dialect definition language)."""
        lines = [f"irdl.dialect @{self.name} {{"]
        for type_def in self.types.values():
            writer = _Writer()
            names = {n: c._emit(writer) for n, c in type_def.parameters.items()}
            if names:
                writer.lines.append(
                    "irdl.parameters("
                    + ", ".join(f"{n}: {v}" for n, v in names.items())
                    + ")"
                )
            lines += [f"  irdl.type @{type_def.name} {{"]
            lines += [f"    {line}" for line in writer.lines]
            lines += ["  }"]
        for op_name, op_class in self.operations.items():
            writer = _Writer()
            parts: list[str] = []
            for keyword, fields in (
                ("operands", op_class._operands()),
                ("results", op_class._results()),
            ):
                if fields:
                    entries = [
                        f"{f.name}: {f.VARIADICITY} {f.constraint._emit(writer)}"
                        for f in fields
                    ]
                    parts.append(f"irdl.{keyword}({', '.join(entries)})")
            attributes = op_class._attributes()
            if attributes:
                entries = [
                    f'"{a.name}" = {a.constraint._emit(writer)}' for a in attributes
                ]
                parts.append(f"irdl.attributes {{{', '.join(entries)}}}")
            regions = op_class._regions()
            if regions:
                names = [writer.define("irdl.region") for _ in regions]
                entries = [
                    f"{r.name}: {n}" for r, n in zip(regions, names, strict=True)
                ]
                parts.append(f"irdl.regions({', '.join(entries)})")
            lines += [f"  irdl.operation @{op_name} {{"]
            lines += [f"    {line}" for line in writer.lines + parts]
            lines += ["  }"]
        lines.append("}")
        return "\n".join(lines)

    def load(self, context: ir.Context | None = None) -> None:
        """Load the dialect into ``context`` (the current one by default);
        loading it again is a no-op. Declarations are closed after the first
        load.

        Raises:
            MLIRError: If the definition is invalid, or another dialect of
                this name is loaded.
        """
        context = context or ir.Context.current()
        if context is None:
            raise TypeError("loading a dialect needs an ir.Context")
        self._frozen = True
        if self.name in context.loaded_dialects:
            return
        with context:
            definitions = ir.Module.parse(self.definition())
            irdl_ops.load_dialects(definitions)

    def __iter__(self) -> Iterator[type[Op]]:
        return iter(self.operations.values())

    def __repr__(self) -> str:
        return f"Dialect({self.name}: {', '.join([*self.types, *self.operations])})"
