"""Generates ``mlir_python.passes``: one typed class per linked MLIR pass.

Inputs are pass definitions dumped with ``llvm-tblgen --dump-json`` and the
C++-to-Python operation class maps written by ``tools/gen_dialects.py`` (to
turn anchors such as ``OperationPass<ModuleOp>`` into Python classes).

Each pass becomes a frozen dataclass whose fields are the pass options, with
MLIR's defaults and documentation; option values listed with ``cl::values``
become Python enums.
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

INTEGER_TYPES = {
    "int",
    "unsigned",
    "int64_t",
    "uint64_t",
    "int32_t",
    "uint32_t",
    "size_t",
}
FLOAT_TYPES = {"double", "float"}
BUILTIN_CLASSES = {"::mlir::ModuleOp": ("mlir_python", "Module")}


def snake_case(name: str) -> str:
    name = re.sub(r"[-.]", "_", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
    if keyword.iskeyword(name) or not name.isidentifier():
        name += "_"
    return name


def enum_member(symbol: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", symbol)
    name = re.sub(r"\W", "_", name).upper()
    return name if name and not name[0].isdigit() else "_" + name


def docstring(text: str, indent: str) -> str:
    text = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"').strip()
    if "\n" not in text:
        return f'{indent}"""{text}"""\n'
    body = textwrap.indent(text, indent)
    return f'{indent}"""{body.lstrip()}\n{indent}"""\n'


@dataclass
class OptionEnum:
    cpp_type: str
    name: str
    description: str
    members: list[
        tuple[str, str, str, str]
    ]  # (python, text value, C++ enumerator, doc)

    def member_for(self, cpp_value: str) -> str | None:
        short = cpp_value.split("::")[-1]
        for python, _, enumerator, _ in self.members:
            if enumerator.split("::")[-1] == short:
                return python
        return None


@dataclass
class PassOption:
    python_name: str
    argument: str
    annotation: str
    default: str  # Python expression
    description: str


@dataclass
class PassClass:
    python_name: str
    argument: str
    summary: str
    description: str
    anchor: str | None  # Python expression for the anchor class
    anchor_interface: str | None
    options: list[PassOption] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


class PassGenerator:
    def __init__(self, classes: dict[str, dict]):
        self.classes = classes
        self.enums: dict[str, OptionEnum] = {}
        self.imports: set[tuple[str, str]] = set()
        self.skipped: list[tuple[str, str]] = []

    def option_enum(self, record: dict) -> OptionEnum | None:
        values = re.findall(
            r'clEnumValN\(\s*([\w:]+)\s*,\s*"([^"]*)"\s*,\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)',
            record.get("additionalOptFlags") or "",
        )
        if not values:
            return None
        cpp_type = record["type"].lstrip(":")
        if cpp_type not in self.enums:
            name = cpp_type.split("::")[-1]
            members = []
            for enumerator, text, doc in values:
                doc = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', doc))
                members.append((enum_member(text), text, enumerator, doc))
            self.enums[cpp_type] = OptionEnum(
                cpp_type, name, record.get("description") or "", members
            )
        return self.enums[cpp_type]

    def option(self, record: dict, supers: list[str]) -> PassOption | None:
        cpp_type = record["type"].strip()
        is_list = "ListOption" in supers
        default_text = (record.get("defaultValue") or "").strip()
        python_type, default = None, None
        enum = self.option_enum(record)
        if enum is not None:
            python_type = enum.name
            member = enum.member_for(default_text) if default_text else None
            default = f"{enum.name}.{member}" if member else None
        elif cpp_type == "bool":
            python_type = "bool"
            default = {"true": "True", "false": "False"}.get(default_text)
        elif cpp_type in INTEGER_TYPES:
            python_type = "int"
            match = re.fullmatch(r"(-?\d+)[uUlL]*", default_text)
            default = match.group(1) if match else None
        elif cpp_type in FLOAT_TYPES:
            python_type = "float"
            try:
                default = repr(float(default_text.rstrip("fF")))
            except ValueError:
                default = None
        elif cpp_type == "std::string":
            python_type = "str"
            if default_text == "":
                default = '""'
            elif re.fullmatch(r'"(?:[^"\\]|\\.)*"', default_text):
                default = default_text
        elif cpp_type == "OpPassManager" and is_list:
            python_type = "Nested"
        if python_type is None:
            return None
        if is_list:
            return PassOption(
                snake_case(record["argument"]),
                record["argument"],
                f"Sequence[{python_type}]",
                "()",
                record.get("description") or "",
            )
        annotation = python_type if default is not None else f"{python_type} | None"
        return PassOption(
            snake_case(record["argument"]),
            record["argument"],
            annotation,
            default if default is not None else "None",
            record.get("description") or "",
        )

    def anchor(self, base_class: str) -> tuple[str | None, str | None] | None:
        """(anchor class expression, interface name), or None if unavailable."""
        match = re.fullmatch(
            r"::mlir::(OperationPass|InterfacePass)<(.*)>", base_class.strip()
        )
        if not match:
            return None
        kind, argument = match.groups()
        if kind == "InterfacePass":
            return None, argument.split("::")[-1]
        if not argument:
            return None, None
        cpp = argument if argument.startswith("::") else f"::mlir::{argument}"
        if cpp in BUILTIN_CLASSES:
            module, cls = BUILTIN_CLASSES[cpp]
        elif cpp in self.classes:
            module, cls = self.classes[cpp]["module"], self.classes[cpp]["class"]
        else:
            return None
        self.imports.add((module, cls))
        alias = cls if module == "mlir_python" else f"{module.split('.')[-1]}.{cls}"
        if module != "mlir_python":
            self.imports.add((module, ""))
        return alias, None

    def pass_class(self, records: dict, name: str) -> PassClass | None:
        record = records[name]
        anchor = self.anchor(record.get("baseClass") or "")
        if anchor is None:
            self.skipped.append(
                (name, f"anchor {record.get('baseClass')} is not linked")
            )
            return None
        python_name = name.removesuffix("Pass") or name
        result = PassClass(
            python_name=python_name,
            argument=record["argument"],
            summary=(record.get("summary") or "").strip(),
            description=textwrap.dedent(record.get("description") or "").strip(),
            anchor=anchor[0],
            anchor_interface=anchor[1],
        )
        for option in record.get("options", []):
            option_record = records[option["def"]]
            converted = self.option(
                option_record, option_record.get("!superclasses", [])
            )
            if converted is None:
                result.unsupported.append(option_record["argument"])
            else:
                result.options.append(converted)
        return result

    def emit(self, passes: list[PassClass]) -> str:
        out = [
            '"""Typed MLIR passes, one class per pass linked into this build.\n\n'  # noqa: ISC004
            "Build pipelines with ``PassManager`` and ``Nested``::\n\n"
            "    pm = PassManager(Module, [Canonicalizer(), Nested(func.FuncOp, [CSE()])])\n"
            "    pm.run(module)\n\n"
            "Each class's fields are the pass's options, with MLIR's defaults.\n\n"
            'Generated by tools/gen_passes.py. Do not edit.\n"""\n\n',
            "from __future__ import annotations\n\n",
            "import dataclasses\nimport enum\nfrom collections.abc import Sequence\n",
            "from typing import ClassVar\n\n",
            "from ._mlir_python import Operation\n",
            "from ._passes import Nested, Pass, PassManager, PipelineElement\n",
        ]
        builtin = sorted(
            cls for module, cls in self.imports if module == "mlir_python" and cls
        )
        if builtin:
            out.append(f"from ._mlir_python import {', '.join(builtin)}\n")
        for module in sorted(
            {m for m, c in self.imports if m != "mlir_python" and not c}
        ):
            out.append(f"from .dialects import {module.split('.')[-1]}\n")
        out.append("\n")
        names = ["Nested", "Pass", "PassManager", "PipelineElement"]
        for enum in sorted(self.enums.values(), key=lambda e: e.name):
            names.append(enum.name)
            out.append(f"\nclass {enum.name}(enum.Enum):\n")
            out.append(docstring(f"Values of the option: {enum.description}.", "    "))
            for python, text, _, doc in enum.members:
                out.append(f"\n    {python} = {text!r}\n")
                if doc:
                    out.append(docstring(doc, "    "))
            out.append("\n")
        for p in passes:
            names.append(p.python_name)
            doc = f"``{p.argument}``: {p.summary}" if p.summary else f"``{p.argument}``"
            if p.description:
                doc += "\n\n" + p.description
            if p.unsupported:
                doc += "\n\nOptions not exposed here: " + ", ".join(p.unsupported) + "."
            out.append("\n@dataclasses.dataclass(frozen=True, kw_only=True)\n")
            out.append(f"class {p.python_name}(Pass):\n")
            out.append(docstring(doc, "    "))
            out.append(f"\n    ARGUMENT: ClassVar[str] = {p.argument!r}\n")
            if p.anchor:
                out.append(
                    f"    ANCHOR: ClassVar[type[Operation] | None] = {p.anchor}\n"
                )
            if p.anchor_interface:
                out.append(
                    f"    ANCHOR_INTERFACE: ClassVar[str | None] = {p.anchor_interface!r}\n"
                )
            for option in p.options:
                out.append(
                    f"\n    {option.python_name}: {option.annotation} = dataclasses.field(\n"
                    f"        default={option.default}, "
                    f"metadata={{'argument': {option.argument!r}}})\n"
                )
                if option.description:
                    out.append(docstring(option.description, "    "))
            out.append("\n")
        out.append(
            "\n__all__ = [\n" + "".join(f'    "{n}",\n' for n in sorted(names)) + "]\n"
        )
        return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records", required=True, type=Path, nargs="+")
    parser.add_argument("--classes", required=True, type=Path, nargs="+")
    parser.add_argument(
        "--introspection",
        required=True,
        type=Path,
        help="output of mlir-python-introspect; only registered passes get classes",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ruff", help="ruff executable used to format the output")
    args = parser.parse_args()

    classes: dict[str, dict] = {}
    for path in args.classes:
        classes.update(json.loads(path.read_text()))
    generator = PassGenerator(classes)
    registered = set(json.loads(args.introspection.read_text())["passes"])
    passes, seen = [], set()
    for path in args.records:
        records = json.loads(path.read_text())
        for name in sorted(records.get("!instanceof", {}).get("PassBase", [])):
            if (
                records[name]["argument"] in seen
                or records[name]["argument"] not in registered
            ):
                continue
            seen.add(records[name]["argument"])
            generated = generator.pass_class(records, name)
            if generated is not None:
                passes.append(generated)
    text = format_python(
        generator.emit(sorted(passes, key=lambda p: p.python_name)),
        args.output,
        args.ruff,
    )
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
