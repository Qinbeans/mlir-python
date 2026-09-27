"""Generates typed nanobind bindings for one MLIR dialect.

Inputs:
  * the dialect's ODS records, dumped with ``llvm-tblgen --dump-json``;
  * facts about the compiled operations from ``mlir-python-introspect``
    (whether result types are inferred, which ODS adds implicitly).

Outputs:
  * a C++ file defining ``bindDialect_<name>``, with one ``Operation``
    subclass per op (typed constructor, typed accessors, ODS docs) and one
    Python enum per ODS enum the dialect owns;
  * a Python module re-exporting those names as ``mlir_python.dialects.<name>``.

Attribute conversions use the C++ snippets ODS itself defines for every
attribute constraint (``constBuilderCall``, ``convertFromStorage``,
``returnType``), so parameters get precise Python types without any text
parsing at runtime.
"""

from __future__ import annotations

import argparse
import json
import keyword
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from pyformat import format_python

# Members of the Python ``Operation`` class; generated accessors must not
# shadow them (a clash gets a trailing underscore).
OPERATION_MEMBERS = {
    "attributes",
    "block",
    "clone",
    "context",
    "create",
    "detach_from_parent",
    "erase",
    "get_asm",
    "is_registered",
    "location",
    "move_after",
    "move_before",
    "name",
    "operands",
    "parent",
    "parse",
    "regions",
    "result",
    "results",
    "successors",
    "verify",
    "walk",
}
# Keyword-only parameters every generated constructor takes.
RESERVED_PARAMETERS = {"location", "ip"}

# Operations whose result-type inference reads their regions, which are still
# empty when the constructor runs; their result types become parameters.
INFER_FROM_REGIONS = {"scf.if"}
# Operations whose result types always equal the types of one variadic operand
# group (ODS name); the constructor derives them instead of asking twice.
RESULT_TYPES_FROM_OPERANDS = {"scf.for": "initArgs"}
# Variadic-of-variadic operands that need one group per block of a successor
# list (ODS does not record the pairing). Omitted groups default to one empty
# group per successor; a count mismatch raises instead of reaching MLIR's
# verifier, which asserts on it.
GROUPS_PER_SUCCESSOR = {
    ("cf.switch", "caseOperands"): "caseDestinations",
    ("llvm.switch", "caseOperands"): "caseDestinations",
}

# Classes defined by hand in src/bindings/DialectExtras.cpp, exported from the
# dialect's Python module alongside the generated ones.
EXTRA_NAMES = {
    "async": ["GroupType", "TokenType", "ValueType"],
    "emitc": [
        "ArrayType",
        "LValueType",
        "OpaqueType",
        "PointerType",
        "PtrDiffTType",
        "SignedSizeTType",
        "SizeTType",
    ],
    "irdl": ["load_dialects"],
    "llvm": ["ArrayType", "FunctionType", "PointerType", "StructType", "VoidType"],
}
# Hand-written Python helpers in src/mlir_python/dialects/_<dialect>_extras.py,
# also exported from the dialect's module.
PYTHON_EXTRAS = {
    "async": ["await_value", "execute"],
    "func": ["Returned", "call", "declare", "define"],
    "llvm": ["address_of", "string_constant"],
}

# Builtin C++ classes with a dedicated Python class (see Core.h).
TYPE_CLASSES = {
    "::mlir::Type": "PyType",
    "::mlir::IntegerType": "PyIntegerType",
    "::mlir::IndexType": "PyIndexType",
    "::mlir::FloatType": "PyFloatType",
    "::mlir::NoneType": "PyNoneType",
    "::mlir::ComplexType": "PyComplexType",
    "::mlir::FunctionType": "PyFunctionType",
    "::mlir::TupleType": "PyTupleType",
    "::mlir::ShapedType": "PyShapedType",
    "::mlir::RankedTensorType": "PyRankedTensorType",
    "::mlir::UnrankedTensorType": "PyUnrankedTensorType",
    "::mlir::VectorType": "PyVectorType",
    "::mlir::MemRefType": "PyMemRefType",
    "::mlir::UnrankedMemRefType": "PyUnrankedMemRefType",
}
ATTRIBUTE_CLASSES = {
    "::mlir::Attribute": "PyAttribute",
    "::mlir::IntegerAttr": "PyIntegerAttr",
    "::mlir::BoolAttr": "PyBoolAttr",
    "::mlir::FloatAttr": "PyFloatAttr",
    "::mlir::StringAttr": "PyStringAttr",
    "::mlir::UnitAttr": "PyUnitAttr",
    "::mlir::TypeAttr": "PyTypeAttr",
    "::mlir::ArrayAttr": "PyArrayAttr",
    "::mlir::DictionaryAttr": "PyDictAttr",
    "::mlir::SymbolRefAttr": "PySymbolRefAttr",
    "::mlir::FlatSymbolRefAttr": "PyFlatSymbolRefAttr",
    "::mlir::DenseElementsAttr": "PyDenseElementsAttr",
    "::mlir::DenseIntElementsAttr": "PyDenseIntElementsAttr",
    "::mlir::DenseFPElementsAttr": "PyDenseFPElementsAttr",
    "::mlir::StridedLayoutAttr": "PyStridedLayoutAttr",
}
INTEGER_TYPES = {
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int",
    "unsigned",
    "unsignedint",
    "size_t",
}
ARRAY_ELEMENTS = {"int8_t", "int16_t", "int32_t", "int64_t", "bool", "float", "double"}


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def snake_case(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def python_identifier(name: str, taken: set[str] = frozenset()) -> str:
    result = snake_case(name)
    if not result.isidentifier() or result[0].isdigit():
        result = "_" + result
    while keyword.iskeyword(result) or result in taken:
        result += "_"
    return result


def enum_member_name(symbol: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", symbol).upper()
    name = re.sub(r"\W", "_", name)
    if not name or name[0].isdigit():
        name = "_" + name
    return name


def normalize_cpp(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([<>,:])\s*", r"\1", text)
    if (
        re.match(r"^[A-Za-z_]", text)
        and "::" in text
        and not text.startswith("std::")
        and not text.startswith("llvm::")
    ):
        text = "::" + text
    return text


def cpp_value(cpp_type: str, literal: str) -> str:
    """A C++ expression converting ODS default ``literal`` to ``cpp_type``."""
    literal = literal.strip()
    if literal.startswith("{"):  # braced initializer, e.g. `{}`
        return f"{cpp_type}{literal}"
    return f"static_cast<{cpp_type}>({literal})"


def cpp_string(text: str) -> str:
    """A C++ raw string literal holding ``text``."""
    return f'R"mlirpy({text})mlirpy"'


def dedent_doc(text: str | None) -> str:
    if not text:
        return ""
    return textwrap.dedent(text).strip("\n")


# ---------------------------------------------------------------------------
# ODS records
# ---------------------------------------------------------------------------


class Records:
    def __init__(self, path: Path):
        self.data = json.loads(path.read_text())

    def __getitem__(self, name: str) -> dict:
        return self.data[name]

    def get(self, name: str) -> dict | None:
        return self.data.get(name)

    def instances(self, cls: str) -> list[str]:
        return self.data.get("!instanceof", {}).get(cls, [])

    @staticmethod
    def supers(record: dict) -> list[str]:
        return record.get("!superclasses", [])


@dataclass
class EnumInfo:
    cpp_type: str
    python_name: str
    summary: str
    is_flag: bool
    cases: list[tuple[str, str, str]]  # (python member, C++ enumerator, text)
    owned: bool  # defined by this dialect (else only referenced)


@dataclass
class AttributeCodec:
    """How one attribute constraint maps to a Python parameter and back."""

    # Templates use @V@ (the parameter value), @B@ (an mlir::Builder) and
    # @CTX@ (the ContextHandle the operation lives in).
    param_type: str  # C++ parameter type (drives the stub type)
    to_attribute: str  # C++ expr producing the attribute from @V@
    from_attribute: str  # C++ expr converting `attr` to the accessor value
    result_type: str  # C++ accessor return type (drives the stub type)
    default_literal: str | None = None  # C++ literal default usable in Python
    hint: str | None = None  # C++ expr giving a ContextHandle for @V@

    def build(self, value: str, builder: str, context: str) -> str:
        return (
            self.to_attribute.replace("@V@", value)
            .replace("@B@", builder)
            .replace("@CTX@", context)
        )


@dataclass
class Argument:
    kind: str  # operand | attribute | region | successor | result
    ods_name: str
    python_name: str
    record: dict
    variadic: bool = False
    optional: bool = False
    summary: str = ""
    codec: AttributeCodec | None = None
    default_cpp: str | None = None  # attribute default (C++ expression)
    # For a variadic-of-variadic operand: the attribute holding group sizes.
    groups_attribute: str | None = None
    # For an enum property: the enum (set through the op's Properties).
    enum: EnumInfo | None = None

    @property
    def keyword_only(self) -> bool:
        if self.kind == "property":
            return True
        if self.kind == "attribute":
            return self.optional or self.default_cpp is not None
        if self.kind in ("operand", "result"):
            return self.optional
        return self.kind == "region" and self.variadic


@dataclass
class OpInfo:
    ods_name: str
    cpp_class: str
    python_class: str
    summary: str
    description: str
    infers_result_types: bool
    result_types_from: str | None  # ODS operand group the result types mirror
    operand_segments: bool
    result_segments: bool
    results: list[Argument] = field(default_factory=list)
    operands: list[Argument] = field(default_factory=list)
    attributes: list[Argument] = field(default_factory=list)
    arguments: list[Argument] = field(default_factory=list)  # ODS order
    regions: list[Argument] = field(default_factory=list)
    successors: list[Argument] = field(default_factory=list)
    properties: list[Argument] = field(default_factory=list)


class Unsupported(Exception):
    pass


class DialectGenerator:
    def __init__(
        self,
        name: str,
        records: Records,
        introspection: dict,
        namespaces: dict[str, str],
    ):
        self.name = name
        # The Python module name: a keyword such as `async` becomes
        # `async_dialect`, as in MLIR's own Python bindings (a trailing
        # underscore would make nanobind's stub generator skip the module).
        self.python_name = f"{name}_dialect" if keyword.iskeyword(name) else name
        self.records = records
        self.introspection = introspection
        # dialect name -> C++ namespace, for every dialect in the build.
        self.namespaces = namespaces
        self.enums: dict[str, EnumInfo] = {}
        self.skipped: list[tuple[str, str]] = []

    # -- enums --------------------------------------------------------------

    def enum_for(self, attr: dict) -> EnumInfo | None:
        record: dict | None = attr
        while record is not None:
            if "enumerants" in record:
                return self.register_enum(record)
            if record.get("enum"):
                return self.register_enum(self.records[record["enum"]["def"]])
            base = record.get("baseAttr")
            record = self.records[base["def"]] if base else None
        return None

    def register_enum(self, info: dict) -> EnumInfo:
        namespace = normalize_cpp(info["cppNamespace"])
        cpp_type = f"{namespace}::{info['className']}"
        if cpp_type in self.enums:
            return self.enums[cpp_type]
        owner = next((d for d, ns in self.namespaces.items() if ns == namespace), None)
        cases = []
        for case in info["enumerants"]:
            case_record = self.records[case["def"]]
            cases.append(
                (
                    enum_member_name(case_record["symbol"]),
                    case_record["symbol"],
                    case_record.get("str", ""),
                )
            )
        enum = EnumInfo(
            cpp_type=cpp_type,
            python_name=info["className"],
            summary=info.get("summary") or "",
            is_flag="BitEnumBase" in Records.supers(info),
            cases=cases,
            owned=owner in (None, self.name),
        )
        self.enums[cpp_type] = enum
        return enum

    # -- attributes ---------------------------------------------------------

    def enum_named(self, cpp_type: str, scope: str) -> EnumInfo | None:
        """The enum a C++ type name refers to from namespace ``scope``."""
        if not hasattr(self, "_enum_records"):
            self._enum_records: dict[str, dict] = {}
            for name in self.records.instances("EnumInfo"):
                record = self.records[name]
                if "enumerants" in record:
                    namespace = normalize_cpp(record["cppNamespace"])
                    self._enum_records[f"{namespace}::{record['className']}"] = record
        parts = [p for p in scope.split("::") if p]
        for depth in range(len(parts), -1, -1):
            candidate = "::" + "::".join([*parts[:depth], cpp_type.strip(":")])
            if candidate in self._enum_records:
                return self.register_enum(self._enum_records[candidate])
        return None

    def attr_defs(self) -> dict[str, dict]:
        """Attribute definitions by their C++ storage class."""
        if not hasattr(self, "_attr_defs"):
            self._attr_defs: dict[str, dict] = {}
            for name in self.records.instances("AttrDef"):
                record = self.records[name]
                storage = normalize_cpp(record.get("storageType"))
                if storage:
                    self._attr_defs[storage] = record
        return self._attr_defs

    def wrapped_enum(self, record: dict) -> tuple[EnumInfo, str] | None:
        """For an attribute defined as a single enum parameter (such as
        ``LLVM::LinkageAttr``), the enum and the parameter's C++ getter."""
        if "AttrDef" not in Records.supers(record):
            # Constraints such as LLVM's `Linkage` wrap the definition; find it
            # by the storage class they share.
            record = self.attr_defs().get(normalize_cpp(record.get("storageType")))
            if record is None:
                return None
        params = (record.get("parameters") or {}).get("args", [])
        if len(params) != 1 or not isinstance(params[0][0], str):
            return None
        cpp_type, name = params[0]
        storage = normalize_cpp(record.get("storageType"))
        enum = self.enum_named(cpp_type, storage.rsplit("::", 1)[0])
        if enum is None:
            return None
        getter = "get" + "".join(
            part[:1].upper() + part[1:] for part in name.split("_")
        )
        return enum, getter

    def attribute_codec(self, record: dict, python_name: str) -> AttributeCodec:
        return_type = normalize_cpp(record.get("returnType"))
        storage = normalize_cpp(record.get("storageType"))
        build = record.get("constBuilderCall") or ""
        read = record.get("convertFromStorage") or "$_self"
        cast_attr = f"llvm::cast<{storage}>(attr)" if storage else "attr"

        def builder(value: str) -> str:
            code = build.replace("$_builder", "@B@").replace(
                "$_ctxt", "@B@.getContext()"
            )
            return code.replace("$0", f"({value})")

        def reader() -> str:
            return f"({read.replace('$_self', cast_attr)})"

        def passthrough() -> AttributeCodec:
            py = ATTRIBUTE_CLASSES.get(storage)
            summary = record.get("summary") or storage or "an attribute"
            if py and py != "PyAttribute":
                return AttributeCodec(
                    param_type=f"const {py} &",
                    to_attribute=f'expectAttribute<{storage}>(@CTX@, @V@, "{python_name}", {cpp_string(summary)})',
                    from_attribute=f"makeAttributeHandle<{py}>(self.tree->context(), attr)",
                    result_type=py,
                    hint="@V@.context",
                )
            check = storage if storage.startswith("::mlir::") else "::mlir::Attribute"
            return AttributeCodec(
                param_type="const PyAttribute &",
                to_attribute=f"mlir::Attribute(expectAttribute<{check}>(@CTX@, "
                f'@V@, "{python_name}", {cpp_string(summary)}))',
                from_attribute="wrapAttribute(self.tree->context(), attr)",
                result_type="AttributeObject",
                hint="@V@.context",
            )

        enum = self.enum_for(record)
        if enum and build:
            return AttributeCodec(
                enum.cpp_type, builder("@V@"), reader(), enum.cpp_type
            )
        wrapped = self.wrapped_enum(record)
        if wrapped:
            enum, getter = wrapped
            return AttributeCodec(
                enum.cpp_type,
                f"{storage}::get(@B@.getContext(), @V@)",
                f"llvm::cast<{storage}>(attr).{getter}()",
                enum.cpp_type,
            )
        if return_type == storage or read.strip() == "$_self":
            return passthrough()
        if not build:
            return passthrough()
        if return_type in INTEGER_TYPES:
            return AttributeCodec(return_type, builder("@V@"), reader(), return_type)
        if return_type == "bool":
            return AttributeCodec("bool", builder("@V@"), reader(), "bool")
        if return_type == "::llvm::StringRef":
            return AttributeCodec(
                "std::string",
                builder("llvm::StringRef(@V@)"),
                f"{reader()}.str()",
                "std::string",
            )
        if return_type == "::llvm::APInt":
            return AttributeCodec(
                "int64_t",
                builder(
                    "llvm::APInt(64, static_cast<uint64_t>(@V@), /*isSigned=*/true)"
                ),
                f"fromAPInt({reader()}, /*isSigned=*/true)",
                "nb::int_",
            )
        if return_type == "::llvm::APFloat":
            match = re.search(r"getFloatAttr\((.*),\s*\$0\)", build)
            if not match:
                return passthrough()
            type_expr = match.group(1).replace("$_builder", "@B@")
            return AttributeCodec(
                "double",
                builder(f"toAPFloat(@V@, {type_expr})"),
                f"toDouble({reader()})",
                "double",
            )
        if return_type in TYPE_CLASSES or (
            return_type.startswith("::mlir::") and return_type.endswith("Type")
        ):
            py = TYPE_CLASSES.get(return_type, "PyType")
            summary = record.get("summary") or "a type"
            value = f'expectType<{return_type}>(@CTX@, @V@, "{python_name}", {cpp_string(summary)})'
            result = (
                f"makeTypeHandle<{py}>(self.tree->context(), {reader()})"
                if py != "PyType"
                else f"wrapType(self.tree->context(), {reader()})"
            )
            return AttributeCodec(
                f"const {py} &",
                builder(value),
                result,
                py if py != "PyType" else "TypeObject",
                hint="@V@.context",
            )
        match = re.fullmatch(r"::llvm::ArrayRef<(\w+)>", return_type)
        if match and match.group(1) in ARRAY_ELEMENTS:
            element = match.group(1)
            return AttributeCodec(
                f"const std::vector<{element}> &",
                builder(f"llvm::SmallVector<{element}>(@V@.begin(), @V@.end())"),
                f"toVector<{element}>({reader()})",
                f"std::vector<{element}>",
            )
        return passthrough()

    # -- operations ---------------------------------------------------------

    def unwrap(self, record: dict) -> tuple[dict, str]:
        """The constraint an ``Arg<...>``/``Res<...>`` wraps (with the summary
        it gives the argument), or ``record`` itself."""
        summary = ""
        while "OpVariable" in Records.supers(record):
            summary = summary or record.get("summary") or ""
            record = self.records[record["constraint"]["def"]]
        return record, summary

    def argument(
        self, kind: str, ods_name: str, record: dict, taken: set[str]
    ) -> Argument:
        supers = Records.supers(record)
        python_name = python_identifier(ods_name, taken)
        arg = Argument(
            kind=kind, ods_name=ods_name, python_name=python_name, record=record
        )
        if kind in ("operand", "result"):
            if "VariadicOfVariadic" in supers:
                if kind != "operand":
                    raise Unsupported(f"{kind} '{ods_name}' is a variadic of variadics")
                arg.groups_attribute = record["segmentAttrName"]
            arg.variadic = "Variadic" in supers
            arg.optional = "Optional" in supers
            base = record.get("baseType")
            base_record = self.records[base["def"]] if base else record
            arg.summary = base_record.get("summary") or record.get("summary") or ""
        elif kind == "attribute":
            outer = record
            wrappers = {
                "OptionalAttr",
                "DefaultValuedAttr",
                "DefaultValuedOptionalAttr",
            }
            codec_record = record
            if wrappers & set(supers) and record.get("baseAttr"):
                codec_record = self.records[record["baseAttr"]["def"]]
            arg.optional = bool(outer.get("isOptional"))
            default = outer.get("defaultValue")
            arg.summary = outer.get("summary") or codec_record.get("summary") or ""
            arg.codec = self.attribute_codec(codec_record, python_name)
            enum = self.enum_for(codec_record)
            if enum is None and (wrapped := self.wrapped_enum(codec_record)):
                enum = wrapped[0]
            if enum is not None:
                kind = "flags" if enum.is_flag else "a member"
                arg.summary = f"{kind} of ``{enum.python_name}``"
            if default:
                if enum is not None and re.fullmatch(r"[\w:]+::\w+", default.strip()):
                    # ODS may leave enum defaults unqualified; qualify them.
                    default = f"{enum.cpp_type}::{default.strip().rsplit('::', 1)[1]}"
                arg.default_cpp = default
                if "$_" not in default and not arg.codec.param_type.startswith(
                    "const "
                ):
                    arg.codec.default_literal = default
            if (
                "UnitAttr" in supers
                or normalize_cpp(codec_record.get("storageType")) == "::mlir::UnitAttr"
            ):
                arg.codec.default_literal = "false"
        elif kind == "region":
            arg.variadic = "VariadicRegion" in supers
            arg.summary = record.get("summary") or ""
        elif kind == "successor":
            arg.variadic = "VariadicSuccessor" in supers
            arg.summary = record.get("summary") or ""
        return arg

    def property(
        self, name: str, record: dict, taken: set[str], namespace: str
    ) -> Argument:
        """An enum property, exposed as a keyword argument of its enum type."""
        if "EnumProp" not in Records.supers(record):
            raise Unsupported(f"argument '{name}' is a non-enum property")
        enum = self.enum_named(record.get("interfaceType") or "", namespace)
        default = (record.get("defaultValue") or "").strip()
        if enum is None or not default:
            raise Unsupported(f"property '{name}' has an unknown enum type")
        member = default.rsplit("::", 1)[1]
        arg = Argument(
            kind="property",
            ods_name=name,
            python_name=python_identifier(name, taken),
            record=record,
            enum=enum,
            default_cpp=f"{enum.cpp_type}::{member}",
            summary=f"{'flags' if enum.is_flag else 'a member'} of ``{enum.python_name}``",
        )
        return arg

    def op_info(self, def_name: str) -> OpInfo:
        record = self.records[def_name]
        dialect = self.records[record["opDialect"]["def"]]
        full_name = f"{dialect['name']}.{record['opName']}"
        facts = self.introspection.get(full_name)
        if facts is None:
            raise Unsupported(f"'{full_name}' is not registered")
        prefix, _, cls = def_name.partition("_")
        cpp_class = cls or prefix
        namespace = normalize_cpp(record.get("cppNamespace") or dialect["cppNamespace"])
        op = OpInfo(
            ods_name=full_name,
            cpp_class=f"{namespace}::{cpp_class}",
            python_class=cpp_class,
            summary=record.get("summary") or "",
            description=dedent_doc(record.get("description")),
            infers_result_types=(
                facts["infers_result_types"] and full_name not in INFER_FROM_REGIONS
            ),
            result_types_from=RESULT_TYPES_FROM_OPERANDS.get(full_name),
            operand_segments=facts["attr_sized_operand_segments"],
            result_segments=facts["attr_sized_result_segments"],
        )
        taken = set(OPERATION_MEMBERS) | RESERVED_PARAMETERS
        # ODS allows unnamed arguments; they get positional names.
        for index, (definition, name) in enumerate(record["arguments"]["args"]):
            arg_record, summary = self.unwrap(self.records[definition["def"]])
            supers = Records.supers(arg_record)
            name = name or f"operand_{index}"
            if "Property" in supers:
                op.properties.append(self.property(name, arg_record, taken, namespace))
                taken.add(op.properties[-1].python_name)
                continue
            kind = (
                "attribute"
                if "Attr" in supers or "AttrConstraint" in supers
                else "operand"
            )
            arg = self.argument(kind, name, arg_record, taken)
            arg.summary = summary or arg.summary
            taken.add(arg.python_name)
            op.arguments.append(arg)
            (op.attributes if kind == "attribute" else op.operands).append(arg)
        # The group sizes of variadic-of-variadic operands are computed from
        # the groups, so their attributes are not parameters.
        derived = {a.groups_attribute for a in op.operands if a.groups_attribute}
        op.arguments = [
            a
            for a in op.arguments
            if not (a.kind == "attribute" and a.ods_name in derived)
        ]
        op.attributes = [a for a in op.attributes if a.ods_name not in derived]
        results = record["results"]["args"]
        for index, (definition, name) in enumerate(results):
            if name is None:  # ODS allows unnamed results.
                name = "result" if len(results) == 1 else f"result_{index}"
            result_record, summary = self.unwrap(self.records[definition["def"]])
            arg = self.argument("result", name, result_record, taken - {"result"})
            arg.summary = summary or arg.summary
            taken.add(arg.python_name)
            op.results.append(arg)
        for index, (definition, name) in enumerate(record["regions"]["args"]):
            name = name or f"region_{index}"
            arg = self.argument("region", name, self.records[definition["def"]], taken)
            taken.add(arg.python_name)
            op.regions.append(arg)
        for index, (definition, name) in enumerate(record["successors"]["args"]):
            name = name or f"successor_{index}"
            arg = self.argument(
                "successor", name, self.records[definition["def"]], taken
            )
            taken.add(arg.python_name)
            op.successors.append(arg)
        return op

    def own_enums(self) -> None:
        """Registers every enum defined in this dialect's namespace, including
        ones only other dialects' operations use (e.g. arith.AtomicRMWKind)."""
        namespace = self.namespaces.get(self.name)
        for name in sorted(self.records.instances("EnumInfo")):
            record = self.records[name]
            if (
                normalize_cpp(record.get("cppNamespace")) == namespace
                and "enumerants" in record
            ):
                self.register_enum(record)

    def ops(self) -> list[OpInfo]:
        result = []
        for def_name in sorted(self.records.instances("Op")):
            record = self.records[def_name]
            if self.records[record["opDialect"]["def"]]["name"] != self.name:
                continue
            try:
                result.append(self.op_info(def_name))
            except Unsupported as reason:
                self.skipped.append((def_name, str(reason)))
        return result

    # -- emission -----------------------------------------------------------

    def struct_name(self, op: OpInfo) -> str:
        return f"Py_{self.name}_{op.python_class}"

    def emit_enum(self, enum: EnumInfo) -> str:
        options = ", nb::is_flag()" if enum.is_flag else ""
        lines = [
            f"  if (!nb::type<{enum.cpp_type}>().is_valid())",
            f'    nb::enum_<{enum.cpp_type}>(m, "{enum.python_name}", '  # noqa: ISC004
            f"{cpp_string(enum.summary or enum.python_name)}{options})",
        ]
        for member, enumerator, text in enum.cases:
            lines.append(
                f'        .value("{member}", {enum.cpp_type}::{enumerator}, '
                f"{cpp_string(f'``{text}`` in MLIR text.' if text else member)})"
            )
        lines[-1] += ";"
        return "\n".join(lines)

    def constructor_params(self, op: OpInfo) -> tuple[list[Argument], list[Argument]]:
        """(positional, keyword-only) constructor parameters.

        Required parameters come first in ODS order, then variadic operands
        (defaulting to empty), then keyword-only optional parameters.
        """
        required, variadic, keyword_only = [], [], []
        explicit_results = not (op.infers_result_types or op.result_types_from)
        results = op.results if explicit_results else []
        for arg in [
            *op.arguments,
            *op.successors,
            *op.regions,
            *results,
            *op.properties,
        ]:
            if arg.kind == "region" and not arg.variadic:
                continue
            if arg.keyword_only:
                keyword_only.append(arg)
            elif arg.kind in ("operand", "result") and arg.variadic:
                variadic.append(arg)
            elif arg.kind == "result":
                required.insert(sum(1 for a in required if a.kind == "result"), arg)
            else:
                required.append(arg)
        return required + variadic, keyword_only

    @staticmethod
    def param_name(arg: Argument) -> str:
        if arg.kind == "result":
            base = snake_case(arg.ods_name)
            if base == "results":
                base = "result"
            return f"{base}_types" if arg.variadic else f"{base}_type"
        if arg.kind == "region":
            return f"num_{snake_case(arg.ods_name)}"
        # Parameters only need to avoid Python keywords and the shared
        # keyword-only parameters; accessor names also avoid Operation members.
        return python_identifier(arg.ods_name, RESERVED_PARAMETERS)

    @classmethod
    def cpp_name(cls, arg: Argument) -> str:
        """C++ variable for a parameter; prefixed so it cannot clash with C++
        keywords or the generated locals."""
        return f"arg_{cls.param_name(arg)}"

    def param_decl(self, arg: Argument) -> tuple[str, str]:
        """(C++ lambda parameter, nanobind argument annotation)."""
        name = self.cpp_name(arg)
        py = self.param_name(arg)
        if arg.kind == "property":
            assert arg.enum is not None
            return f"{arg.enum.cpp_type} {name}", f'"{py}"_a = {arg.default_cpp}'
        if arg.kind == "result":
            base = "PyType"
            if arg.variadic:
                return (
                    f"const std::vector<{base}> &{name}",
                    f'"{py}"_a = std::vector<{base}>{{}}',
                )
            if arg.optional:
                return f"std::optional<{base}> {name}", f'"{py}"_a = nb::none()'
            return f"const {base} &{name}", f'"{py}"_a'
        if arg.kind == "operand":
            if arg.groups_attribute:
                return (
                    f"const std::vector<std::vector<PyValue>> &{name}",
                    f'"{py}"_a = std::vector<std::vector<PyValue>>{{}}',
                )
            if arg.variadic:
                return (
                    f"const std::vector<PyValue> &{name}",
                    f'"{py}"_a = std::vector<PyValue>{{}}',
                )
            if arg.optional:
                return f"std::optional<PyValue> {name}", f'"{py}"_a = nb::none()'
            return f"const PyValue &{name}", f'"{py}"_a'
        if arg.kind == "successor":
            if arg.variadic:
                return f"const std::vector<PyBlock> &{name}", f'"{py}"_a'
            return f"const PyBlock &{name}", f'"{py}"_a'
        if arg.kind == "region":
            return f"unsigned {name}", f'"{py}"_a = 0'
        codec = arg.codec
        assert codec is not None
        value_type = codec.param_type
        if arg.default_cpp is not None and codec.default_literal is not None:
            plain = value_type.removeprefix("const ").removesuffix(" &")
            return (
                f"{value_type} {name}",
                f'"{py}"_a = {cpp_value(plain, codec.default_literal)}',
            )
        if arg.optional or arg.default_cpp is not None:
            plain = value_type.removeprefix("const ").removesuffix(" &")
            return f"std::optional<{plain}> {name}", f'"{py}"_a = nb::none()'
        return f"{value_type} {name}", f'"{py}"_a'

    def emit_constructor(self, op: OpInfo) -> str:
        positional, keyword_only = self.constructor_params(op)
        params, annotations = [], []
        for arg in [*positional, *keyword_only]:
            param, annotation = self.param_decl(arg)
            params.append(param)
            annotations.append(annotation)
        params += [
            "std::optional<PyLocation> location",
            "std::optional<PyInsertionPoint> ip",
        ]
        if keyword_only:
            annotations.insert(len(positional), "nb::kw_only()")
        else:
            annotations.append("nb::kw_only()")
        annotations += ['"location"_a = nb::none()', '"ip"_a = nb::none()']

        body = ["std::vector<ContextHandle> hints;"]
        for arg in [*positional, *keyword_only]:
            n = self.cpp_name(arg)
            if arg.kind in ("operand", "successor"):
                if arg.groups_attribute:
                    body.append(
                        f"for (const auto &group : {n}) if (!group.empty()) "
                        f"{{ hints.push_back(group.front().tree->context()); break; }}"
                    )
                elif arg.variadic:
                    body.append(
                        f"if (!{n}.empty()) hints.push_back({n}.front().tree->context());"
                    )
                elif arg.optional:
                    body.append(f"if ({n}) hints.push_back({n}->tree->context());")
                else:
                    body.append(f"hints.push_back({n}.tree->context());")
            elif arg.kind == "result":
                if arg.variadic:
                    body.append(
                        f"if (!{n}.empty()) hints.push_back({n}.front().context);"
                    )
                elif arg.optional:
                    body.append(f"if ({n}) hints.push_back({n}->context);")
                else:
                    body.append(f"hints.push_back({n}.context);")
            elif arg.kind == "attribute" and arg.codec and arg.codec.hint:
                target = (
                    f"(*{n})"
                    if (arg.optional or arg.default_cpp is not None)
                    and arg.codec.default_literal is None
                    else n
                )
                expr = arg.codec.hint.replace("@V@", target)
                if target != n:
                    body.append(f"if ({n}) hints.push_back({expr});")
                else:
                    body.append(f"hints.push_back({expr});")
        body.append(f'GeneratedOpState state("{op.ods_name}", location, ip, hints);')
        body.append("mlir::Builder &b = state.builder();")
        body.append("(void)b;")
        if op.result_types_from:
            source = next(a for a in op.operands if a.ods_name == op.result_types_from)
            body.append("std::vector<PyType> derived_types;")
            body.append(f"for (const PyValue &value : {self.cpp_name(source)})")
            body.append(
                "  derived_types.push_back(PyType{value.tree->context(), get(value).getType()});"
            )
            body.append("state.addVariadicResultTypes(derived_types);")
        elif not op.infers_result_types:
            for arg in op.results:
                n = self.cpp_name(arg)
                call = (
                    "addVariadicResultTypes"
                    if arg.variadic
                    else "addOptionalResultType"
                    if arg.optional
                    else "addResultType"
                )
                body.append(f"state.{call}({n});")
        for arg in op.arguments:
            n = self.cpp_name(arg)
            if arg.kind == "operand" and arg.groups_attribute:
                paired = GROUPS_PER_SUCCESSOR.get((op.ods_name, arg.ods_name))
                if paired:
                    successor = next(a for a in op.successors if a.ods_name == paired)
                    blocks = self.cpp_name(successor)
                    python = self.param_name(arg)
                    body.append(f"std::vector<std::vector<PyValue>> groups_{n} = {n};")
                    body.append(
                        f"if (groups_{n}.empty()) groups_{n}.resize({blocks}.size());"
                    )
                    body.append(
                        f"if (groups_{n}.size() != {blocks}.size()) throw std::invalid_argument("
                        f'"{python} needs one group of operands per entry of '
                        f'{self.param_name(successor)}");'
                    )
                    body.append(
                        f'state.addOperandGroups(groups_{n}, "{arg.groups_attribute}");'
                    )
                else:
                    body.append(
                        f'state.addOperandGroups({n}, "{arg.groups_attribute}");'
                    )
                continue
            if arg.kind == "operand":
                call = (
                    "addVariadicOperands"
                    if arg.variadic
                    else "addOptionalOperand"
                    if arg.optional
                    else "addOperand"
                )
                body.append(f"state.{call}({n});")
                continue
            codec = arg.codec
            assert codec is not None
            if arg.default_cpp is not None and codec.default_literal is not None:
                plain = codec.param_type.removeprefix("const ").removesuffix(" &")
                value = codec.build(n, "b", "state.context()")
                body.append(f"if (!({n} == {cpp_value(plain, codec.default_literal)}))")
                body.append(
                    f'  state.setAttribute("{arg.ods_name}", mlir::Attribute({value}));'
                )
            elif arg.optional or arg.default_cpp is not None:
                value = codec.build(f"(*{n})", "b", "state.context()")
                body.append(f"if ({n})")
                body.append(
                    f'  state.setAttribute("{arg.ods_name}", mlir::Attribute({value}));'
                )
            else:
                value = codec.build(n, "b", "state.context()")
                body.append(
                    f'state.setAttribute("{arg.ods_name}", mlir::Attribute({value}));'
                )
        fixed_regions = sum(1 for r in op.regions if not r.variadic)
        if fixed_regions:
            body.append(f"state.addRegions({fixed_regions});")
        for arg in op.regions:
            if arg.variadic:
                body.append(f"state.addRegions({self.cpp_name(arg)});")
        for arg in op.successors:
            call = "addSuccessors" if arg.variadic else "addSuccessor"
            body.append(f"state.{call}({self.cpp_name(arg)});")
        infer = "true" if op.infers_result_types else "false"
        body.append(
            f"auto created = state.create<{self.struct_name(op)}>({infer}, "
            f"{'true' if op.operand_segments else 'false'}, "
            f"{'true' if op.result_segments else 'false'});"
        )
        for prop in op.properties:
            n = self.cpp_name(prop)
            setter = "set" + prop.ods_name[:1].upper() + prop.ods_name[1:]
            body.append(
                f"llvm::cast<{op.cpp_class}>(created.op).getProperties().{setter}({n});"
            )
        body.append("return created;")

        doc = self.constructor_doc(op, positional, keyword_only)
        joined_params = ",\n                       ".join(params)
        joined_body = "\n             ".join(body)
        joined_annotations = ", ".join(annotations)
        return (
            f"      .def(nb::new_([]({joined_params}) {{\n"
            f"             {joined_body}\n"
            f"           }}),\n"
            f"           {joined_annotations},\n"
            f"           {cpp_string(doc)})"
        )

    def constructor_doc(
        self, op: OpInfo, positional: list[Argument], keyword_only: list[Argument]
    ) -> str:
        lines = [
            f"Create ``{op.ods_name}``" + (f": {op.summary}." if op.summary else "."),
            "",
        ]
        if op.infers_result_types and op.results:
            lines += ["Result types are inferred.", ""]
        if op.result_types_from:
            source = snake_case(op.result_types_from)
            lines += [f"Result types match the types of ``{source}``.", ""]
        documented = [*positional, *keyword_only]
        if documented:
            lines.append("Args:")
            for arg in documented:
                if arg.kind == "region":
                    lines.append(
                        f"    {self.param_name(arg)}: Number of ``{arg.ods_name}`` regions."
                    )
                    continue
                what = {
                    "result": "Type of result",
                    "operand": "Operand",
                    "attribute": "Attribute",
                    "property": "Property",
                    "successor": "Successor",
                }[arg.kind]
                detail = f" ({arg.summary})" if arg.summary else ""
                extra = ""
                if arg.kind == "attribute" and arg.default_cpp is not None:
                    extra = " Omit for the default."
                elif arg.optional:
                    extra = " Optional."
                elif arg.kind in ("operand", "result") and arg.variadic:
                    extra = " Empty by default."
                lines.append(
                    f"    {self.param_name(arg)}: {what} ``{arg.ods_name}``{detail}.{extra}"
                )
            lines.append("    location: Defaults to the current ``Location``.")
            lines.append(
                "    ip: Defaults to the current ``InsertionPoint``; detached without one."
            )
        return "\n".join(lines)

    def emit_accessors(self, op: OpInfo) -> list[str]:
        cast = f"llvm::cast<{op.cpp_class}>(checkedOp(self))"
        struct = self.struct_name(op)
        out = []

        def prop(
            name: str, returns: str, body: str, doc: str, setter: str | None = None
        ):
            if setter:
                out.append(
                    f'      .def_prop_rw("{name}",\n'
                    f"          [](const {struct} &self) -> {returns} {{ {body} }},\n"
                    f"          {setter},\n"
                    f"          {cpp_string(doc)})"
                )
            else:
                out.append(
                    f'      .def_prop_ro("{name}",\n'
                    f"          [](const {struct} &self) -> {returns} {{ {body} }},\n"
                    f"          {cpp_string(doc)})"
                )

        for index, arg in enumerate(op.operands):
            values = f"{cast}.getODSOperands({index})"
            doc = f"Operand ``{arg.ods_name}``" + (
                f": {arg.summary}." if arg.summary else "."
            )
            if arg.groups_attribute:
                prop(
                    arg.python_name,
                    "std::vector<std::vector<ValueObject>>",
                    f'return valueGroups(self, {values}, "{arg.groups_attribute}");',
                    doc,
                )
            elif arg.variadic:
                prop(
                    arg.python_name,
                    "std::vector<ValueObject>",
                    f"return valueList(self, {values});",
                    doc,
                )
            elif arg.optional:
                prop(
                    arg.python_name,
                    "nb::typed<nb::object, std::optional<PyValue>>",
                    f"return optionalValue(self, {values});",
                    doc,
                )
            else:
                prop(
                    arg.python_name,
                    "ValueObject",
                    f"return singleValue(self, {values});",
                    doc,
                )
        for index, arg in enumerate(op.results):
            name = arg.python_name.rstrip("_")
            if len(op.results) == 1 and name in ("result", "results"):
                continue  # Operation.result / Operation.results already cover it.
            name = arg.python_name
            values = f"{cast}.getODSResults({index})"
            doc = f"Result ``{arg.ods_name}``" + (
                f": {arg.summary}." if arg.summary else "."
            )
            if arg.variadic:
                prop(
                    name,
                    "std::vector<PyOpResult>",
                    f"return resultList(self, {values});",
                    doc,
                )
            elif arg.optional:
                prop(
                    name,
                    "std::optional<PyOpResult>",
                    f"return optionalResult(self, {values});",
                    doc,
                )
            else:
                prop(name, "PyOpResult", f"return singleResult(self, {values});", doc)
        for arg in op.attributes:
            codec = arg.codec
            assert codec is not None
            doc = f"Attribute ``{arg.ods_name}``" + (
                f": {arg.summary}." if arg.summary else "."
            )
            readable_default = (
                arg.default_cpp is not None and codec.default_literal is not None
            )
            nullable = (
                arg.optional or arg.default_cpp is not None
            ) and not readable_default
            returns = (
                f"nb::typed<nb::object, std::optional<{codec.result_type}>>"
                if nullable
                else codec.result_type
            )
            missing = (
                f"return {cpp_value(codec.result_type, codec.default_literal)};"
                if readable_default
                else "return nb::none();"
                if nullable
                else f"throw std::invalid_argument(\"'{arg.ods_name}' is missing\");"
            )
            convert = codec.from_attribute
            value = f"nb::cast({convert})" if nullable else convert
            body = f'mlir::Attribute attr = inherentAttr(self, "{arg.ods_name}"); if (!attr) {missing} return {value};'
            setter = None
            plain = codec.param_type.removeprefix("const ").removesuffix(" &")
            if "@V@" in codec.to_attribute:
                param = f"std::optional<{plain}>" if nullable else f"{codec.param_type}"
                value_expr = codec.build(
                    "(*value)" if nullable else "value",
                    "builder",
                    "self.tree->context()",
                )
                prefix = (
                    f'if (!value) {{ setInherentAttr(self, "{arg.ods_name}", {{}}); return; }} '
                    if nullable
                    else ""
                )
                setter = (
                    f"[]({struct} &self, {param} value) {{ {prefix}"
                    f"mlir::Builder builder(&self.tree->context()->context); (void)builder; "
                    f'setInherentAttr(self, "{arg.ods_name}", mlir::Attribute({value_expr})); }}'
                )
            prop(arg.python_name, returns, body, doc, setter)
        for property_ in op.properties:
            assert property_.enum is not None
            suffix = property_.ods_name[:1].upper() + property_.ods_name[1:]
            prop(
                property_.python_name,
                property_.enum.cpp_type,
                f"return {cast}.getProperties().get{suffix}();",
                f"Property ``{property_.ods_name}``: {property_.summary}.",
                f"[]({self.struct_name(op)} &self, {property_.enum.cpp_type} value) "
                f"{{ llvm::cast<{op.cpp_class}>(checkedOp(self)).getProperties()"
                f".set{suffix}(value); }}",
            )
        for index, arg in enumerate(op.regions):
            doc = f"Region ``{arg.ods_name}``" + (
                f": {arg.summary}." if arg.summary else "."
            )
            if arg.variadic:
                prop(
                    arg.python_name,
                    "std::vector<PyRegion>",
                    f"return regionsFrom(self, {index});",
                    doc,
                )
            else:
                prop(
                    arg.python_name, "PyRegion", f"return regionAt(self, {index});", doc
                )
        for index, arg in enumerate(op.successors):
            doc = f"Successor ``{arg.ods_name}``."
            if arg.variadic:
                prop(
                    arg.python_name,
                    "std::vector<PyBlock>",
                    f"return successorsFrom(self, {index});",
                    doc,
                )
            else:
                prop(
                    arg.python_name,
                    "PyBlock",
                    f"return successorAt(self, {index});",
                    doc,
                )
        return out

    def class_doc(self, op: OpInfo) -> str:
        head = f"``{op.ods_name}``" + (f": {op.summary}." if op.summary else ".")
        return head + ("\n\n" + op.description if op.description else "")

    def emit_cpp(self, ops: list[OpInfo], include: str, dialect_doc: str) -> str:
        structs = "\n".join(
            f"struct {self.struct_name(op)} : PyOperation {{}};" for op in ops
        )
        body = [
            f'  nb::module_ m = defineDialectModule(parent, "{self.python_name}", {cpp_string(dialect_doc)});'
        ]
        for enum in self.enums.values():
            if enum.owned:
                body.append(self.emit_enum(enum))
        for op in ops:
            chain = [
                f'  nb::class_<{self.struct_name(op)}, PyOperation>(m, "{op.python_class}", '  # noqa: ISC004
                f"{cpp_string(self.class_doc(op))})",
                self.emit_constructor(op),
                *self.emit_accessors(op),
            ]
            chain[-1] += ";"
            body.append("\n".join(chain))
            body.append(
                f'  m.attr("{op.python_class}").attr("OPERATION_NAME") = "{op.ods_name}";'
            )
            body.append(
                f"  registerOperationClass<{self.struct_name(op)}, {op.cpp_class}>();"
            )
        skipped = "".join(
            f"// Skipped {name}: {reason}\n" for name, reason in self.skipped
        )
        return f"""// Generated by tools/gen_dialects.py for the '{self.name}' dialect. Do not edit.
#include "DialectSupport.h"

#include <{include}>

{skipped}
namespace mlir_python {{

using namespace nb::literals;

namespace {{
{structs}
}} // namespace

void bindDialect_{self.name}(nb::module_ &parent) {{
{chr(10).join(body)}
}}

}} // namespace mlir_python
"""

    def emit_python(self, ops: list[OpInfo]) -> str:
        names = sorted(
            EXTRA_NAMES.get(self.name, [])
            + [op.python_class for op in ops]
            + [e.python_name for e in self.enums.values() if e.owned]
        )
        helpers = PYTHON_EXTRAS.get(self.name, [])
        imports = "".join(f"    {n} as {n},\n" for n in names)
        helper_imports = (
            f"from ._{self.name}_extras import (\n"
            + "".join(f"    {n} as {n},\n" for n in helpers)
            + ")\n"
            if helpers
            else ""
        )
        exported = "".join(f'    "{n}",\n' for n in sorted(names + helpers))
        return (
            f'"""Typed operations of the MLIR ``{self.name}`` dialect.\n\n'
            f"Generated by tools/gen_dialects.py. Do not edit.\n"
            f'"""\n\nfrom .._mlir_python.{self.python_name} import (\n{imports})\n'
            f"{helper_imports}\n"
            f"__all__ = [\n{exported}]\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="dialect namespace, e.g. arith")
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--introspection", required=True, type=Path)
    parser.add_argument("--include", required=True, help="C++ header declaring the ops")
    parser.add_argument(
        "--namespaces",
        required=True,
        help="comma-separated dialect=::cpp::namespace for every bound dialect",
    )
    parser.add_argument("--output-cpp", required=True, type=Path)
    parser.add_argument("--output-py", required=True, type=Path)
    parser.add_argument(
        "--ruff", help="ruff executable used to format the Python output"
    )
    parser.add_argument(
        "--output-classes",
        required=True,
        type=Path,
        help="JSON map from C++ op class to its Python class, used by tools/gen_passes.py",
    )
    args = parser.parse_args()

    namespaces = dict(item.split("=", 1) for item in args.namespaces.split(","))
    generator = DialectGenerator(
        args.name,
        Records(args.records),
        json.loads(args.introspection.read_text())["operations"],
        namespaces,
    )
    ops = generator.ops()
    generator.own_enums()
    doc = f"Typed operations of the MLIR ``{args.name}`` dialect."
    args.output_cpp.parent.mkdir(parents=True, exist_ok=True)
    args.output_py.parent.mkdir(parents=True, exist_ok=True)
    cpp = generator.emit_cpp(ops, args.include, doc)
    if not args.output_cpp.exists() or args.output_cpp.read_text() != cpp:
        args.output_cpp.write_text(cpp)
    classes = {
        op.cpp_class: {
            "module": f"mlir_python.dialects.{generator.python_name}",
            "class": op.python_class,
            "name": op.ods_name,
        }
        for op in ops
    }
    args.output_classes.write_text(json.dumps(classes, indent=1, sort_keys=True))
    py = format_python(generator.emit_python(ops), args.output_py, args.ruff)
    if not args.output_py.exists() or args.output_py.read_text() != py:
        args.output_py.write_text(py)


if __name__ == "__main__":
    main()
