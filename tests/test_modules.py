"""Compilation modules split across Python files, imported and linked."""

import importlib
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from mlir_python import codegen

FILES = {
    "__init__.py": "",
    "mathlib.py": """
        from mlir_python.lang import Module, i32

        mathlib = Module("mathlib")


        @mathlib.function
        def square(x: i32) -> i32:
            return x * x


        @mathlib.function
        def cube(x: i32) -> i32:
            return square(x) * x
    """,
    "text.py": """
        from mlir_python.lang import Module, cstr, i32

        text = Module("text")


        @text.extern
        def puts(s: cstr) -> i32: ...


        @text.function
        def greet() -> None:
            puts("hello from text")
    """,
    "parity.py": """
        from mlir_python.lang import Module, i32

        from . import parity_odd

        parity = Module("parity")


        @parity.function
        def is_even(n: i32) -> bool:
            return n == 0 or parity_odd.is_odd(n - 1)
    """,
    "parity_odd.py": """
        from mlir_python.lang import Module, i32

        from . import parity

        odd = Module("parity_odd")


        @odd.function
        def is_odd(n: i32) -> bool:
            return n != 0 and parity.is_even(n - 1)
    """,
    "clash.py": """
        from mlir_python.lang import Module, i32

        from .mathlib import cube

        clash = Module("clash")


        @clash.function
        def square(x: i32) -> i32:  # mathlib defines square too
            return x


        @clash.function
        def both(x: i32) -> i32:
            return square(x) + cube(x)
    """,
    "app.py": """
        from mlir_python.lang import Program, cstr, i32

        from .mathlib import cube
        from .text import greet

        program = Program("app")


        @program.extern
        def puts(s: cstr) -> i32: ...


        @program.function
        def volume(x: i32) -> i32:
            return cube(x) + 1


        @program.main
        def main() -> i32:
            greet()
            puts("hello from app")
            return volume(2)
    """,
}


@pytest.fixture
def package(tmp_path: Path) -> Iterator[str]:
    name = f"modpkg_{tmp_path.name.replace('-', '_')}"
    root = tmp_path / name
    root.mkdir()
    for file, text in FILES.items():
        (root / file).write_text(textwrap.dedent(text))
    sys.path.insert(0, str(tmp_path))
    try:
        yield name
    finally:
        sys.path.remove(str(tmp_path))
        for module in [m for m in sys.modules if m.startswith(name)]:
            del sys.modules[module]


def load(package: str, module: str) -> ModuleType:
    return importlib.import_module(f"{package}.{module}")


def test_imported_functions_link_and_run(package: str) -> None:
    app = load(package, "app")
    assert app.volume(2) == 9  # cube comes from mathlib, linked in
    mathlib = load(package, "mathlib")
    assert mathlib.cube(3) == 27  # and mathlib still runs on its own


def test_each_module_compiles_alone_with_import_declarations(package: str) -> None:
    app = load(package, "app")
    text = str(app.program)
    assert "func.func private @cube(i32) -> i32" in text
    assert "func.func @cube" not in text  # defined in mathlib, not here
    linked = str(app.program.linked())
    assert "func.func @cube(" in linked and "func.func @square(" in linked
    names = {m.name for m in app.program.imports}
    assert names == {"mathlib", "text"}


def test_circular_imports(package: str) -> None:
    parity = load(package, "parity")
    assert [parity.is_even(n) for n in range(6)] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_executables_link_imported_modules(package: str, tmp_path: Path) -> None:
    app = load(package, "app")
    exe = app.program.build_executable(tmp_path / "app")
    result = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    assert result.stdout == "hello from text\nhello from app\n"
    assert result.returncode == 9


def test_duplicate_definitions_are_link_errors(package: str) -> None:
    clash = load(package, "clash")
    with pytest.raises(codegen.LinkError, match="'square' is defined in both"):
        clash.both(2)
