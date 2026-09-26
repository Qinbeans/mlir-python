"""External libraries: by name or by path, shared or static, JIT and linked.

Loaded libraries stay in the process with global symbols, so each test's C
functions have their own names.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from mlir_python import codegen
from mlir_python.lang import Module, Program, f64, i32

CC = shutil.which("cc")
pytestmark = pytest.mark.skipif(CC is None, reason="needs a C compiler")


def compile_c(tmp_path: Path, source: str, *flags: str) -> Path:
    """An object file compiled from ``source``."""
    assert CC is not None
    c = tmp_path / "lib.c"
    c.write_text(source)
    obj = tmp_path / "lib.o"
    subprocess.run([CC, "-c", *flags, str(c), "-o", str(obj)], check=True)
    return obj


def archive(tmp_path: Path, source: str, *flags: str) -> Path:
    obj = compile_c(tmp_path, source, *flags)
    path = tmp_path / "libcode.a"
    subprocess.run(["ar", "rcs", str(path), str(obj)], check=True)
    return path


def shared(tmp_path: Path, source: str) -> Path:
    assert CC is not None
    obj = compile_c(tmp_path, source, "-fPIC")
    path = tmp_path / "libcode.so"
    subprocess.run([CC, "-shared", str(obj), "-o", str(path)], check=True)
    return path


def run(executable: Path, **env: str) -> int:
    return subprocess.run(
        [executable], env={**os.environ, **env}, check=False
    ).returncode


def test_static_archive_links_into_an_executable(tmp_path: Path) -> None:
    lib = archive(tmp_path, "int static_triple(int x) { return 3 * x; }")
    program = Program(libraries=[lib])

    @program.extern
    def static_triple(x: i32) -> i32: ...

    @program.main
    def main() -> i32:
        return static_triple(4)

    assert run(program.build_executable(tmp_path / "app")) == 12


def test_static_archive_through_the_jit(tmp_path: Path) -> None:
    lib = archive(tmp_path, "int pic_triple(int x) { return 3 * x; }", "-fPIC")
    module = Module("uses_archive", libraries=[lib])

    @module.extern
    def pic_triple(x: i32) -> i32: ...

    @module.function
    def triple_plus_one(x: i32) -> i32:
        return pic_triple(x) + 1

    assert triple_plus_one(5) == 16


def test_shared_library_through_the_jit_and_linked(tmp_path: Path) -> None:
    lib = shared(tmp_path, "int shared_double(int x) { return 2 * x; }")
    program = Program(libraries=[lib])

    @program.extern
    def shared_double(x: i32) -> i32: ...

    @program.function
    def quadruple(x: i32) -> i32:
        return shared_double(shared_double(x))

    @program.main
    def main() -> i32:
        return quadruple(5)

    assert quadruple(3) == 12
    executable = program.build_executable(tmp_path / "app")
    assert run(executable, LD_LIBRARY_PATH=str(tmp_path)) == 20


def test_library_by_name() -> None:
    module = Module("uses_libm", libraries=["m"])

    @module.extern
    def cbrt(x: f64) -> f64: ...

    @module.function
    def cube_root(x: f64) -> f64:
        return cbrt(x)

    assert cube_root(27.0) == pytest.approx(3.0)


def test_libraries_follow_imports(tmp_path: Path) -> None:
    lib = archive(tmp_path, "int imported_square(int x) { return x * x; }")
    wrapper = Module("wrapper", libraries=[lib])

    @wrapper.extern
    def imported_square(x: i32) -> i32: ...

    @wrapper.function
    def square(x: i32) -> i32:
        return imported_square(x)

    program = Program()

    @program.main
    def main() -> i32:
        return square(7)

    assert run(program.build_executable(tmp_path / "app")) == 49


def test_a_path_given_as_a_string_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"pass Path\('/opt/libfoo.a'\)"):
        codegen.load_libraries(["/opt/libfoo.a"])
    with pytest.raises(ValueError, match="not a name"):
        codegen.load_libraries(["-lm"])


def test_missing_libraries_are_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="libmissing.so"):
        codegen.load_libraries([tmp_path / "libmissing.so"])
    with pytest.raises(codegen.LinkError, match="'no_such_library_xyz'"):
        codegen.load_libraries(["no_such_library_xyz"])


@pytest.mark.skipif(platform.machine() != "x86_64", reason="x86-64 relocations")
def test_an_archive_without_pic_explains_how_to_fix_it(tmp_path: Path) -> None:
    source = "static int table[4] = {1, 2, 3, 4}; int nopic(int i) { return table[i]; }"
    lib = archive(tmp_path, source, "-fno-pic", "-mcmodel=small")
    with pytest.raises(codegen.LinkError, match="-fPIC"):
        codegen.load_libraries([lib])
