"""@struct values and the C calling convention for externs.

The C side is compiled with the system C compiler, so each struct shape is
checked against how C itself passes it: small integer structs, float structs
(SSE registers / AArch64 HFAs), mixed ones, structs too large for registers
(byval / pointer copies, sret results), and register exhaustion.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from mlir_python.lang import (
    CompileError,
    Module,
    Program,
    Ptr,
    cstr,
    f32,
    f64,
    i8,
    i32,
    i64,
    ptr,
    stack,
    struct,
    u8,
    u16,
    u32,
)
from mlir_python.lang._types import ScalarType

CC = shutil.which("cc")
needs_cc = pytest.mark.skipif(CC is None, reason="needs a C compiler")

C_SOURCE = r"""
#include <stdbool.h>
#include <stdint.h>

typedef struct { uint8_t r, g, b, a; } Color;
typedef struct { float x, y; } Vector2;
typedef struct { float x, y, z; } Vector3;
typedef struct { float x, y, width, height; } Rectangle;
typedef struct { int32_t count; float scale; } Mixed;
typedef struct { double d; int32_t i; } DoubleInt;
typedef struct { int64_t x, y; } Pair;
typedef struct { Vector2 offset, target; float rotation, zoom; } Camera2D;
typedef struct { Vector2 position; Color tint; } Sprite;

uint32_t abi_color_bits(Color c) {
    return c.r | (c.g << 8) | ((uint32_t)c.b << 16) | ((uint32_t)c.a << 24);
}
Color abi_color_make(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return (Color){r, g, b, a};
}
Vector2 abi_v2_add(Vector2 a, Vector2 b) { return (Vector2){a.x + b.x, a.y + b.y}; }
Vector3 abi_v3_scale(Vector3 v, float s) { return (Vector3){v.x * s, v.y * s, v.z * s}; }
float abi_rect_area(Rectangle r) { return r.width * r.height + r.x - r.y; }
Rectangle abi_rect_grow(Rectangle r, float by) {
    return (Rectangle){r.x - by, r.y - by, r.width + 2 * by, r.height + 2 * by};
}
float abi_mixed(Mixed m) { return m.count * m.scale; }
double abi_double_int(DoubleInt v) { return v.d * v.i; }
DoubleInt abi_double_int_make(double d, int32_t i) { return (DoubleInt){d, i}; }
int64_t abi_pairs(Pair a, Pair b, Pair c, Pair d, int64_t e) {
    return a.x - a.y + 10 * (b.x - b.y) + 100 * (c.x - c.y) + 1000 * (d.x - d.y) + 10000 * e;
}
float abi_camera(Camera2D c) {
    return c.offset.x + 2 * c.offset.y + 3 * c.target.x + 4 * c.target.y
        + 5 * c.rotation + 6 * c.zoom;
}
Camera2D abi_camera_make(float z) {
    return (Camera2D){{1, 2}, {3, 4}, 5, z};
}
uint32_t abi_sprite(Sprite s) {
    return (uint32_t)(s.position.x * 10 + s.position.y) * 1000 + abi_color_bits(s.tint) % 1000;
}
bool abi_is_positive(int32_t x) { return x > 0; }
int32_t abi_bool_byte(bool b) { return *(unsigned char *)&b; }
int32_t abi_widen(int8_t small, uint16_t medium) { return small * 100000 + medium; }
"""


@struct
class Color:
    r: u8
    g: u8
    b: u8
    a: u8


@struct
class Vector2:
    x: f32
    y: f32


@struct
class Vector3:
    x: f32
    y: f32
    z: f32


@struct
class Rectangle:
    x: f32
    y: f32
    width: f32
    height: f32


@struct
class Mixed:
    count: i32
    scale: f32


@struct
class DoubleInt:
    d: f64
    i: i32


@struct
class Pair:
    x: i64
    y: i64


@struct
class Camera2D:
    offset: Vector2
    target: Vector2
    rotation: f32
    zoom: f32


@struct
class Sprite:
    position: Vector2
    tint: Color


@pytest.fixture(scope="module")
def clib(tmp_path_factory: pytest.TempPathFactory) -> Path:
    assert CC is not None
    directory = tmp_path_factory.mktemp("clib")
    source = directory / "abi.c"
    source.write_text(C_SOURCE)
    library = directory / "libabi.so"
    subprocess.run(
        [CC, "-shared", "-fPIC", "-O1", str(source), "-o", str(library)], check=True
    )
    return library


def test_layout_matches_c() -> None:
    kind = Camera2D.__lang_struct__  # type: ignore[attr-defined]
    assert (kind.size, kind.alignment, kind.offsets()) == (24, 4, [0, 8, 16, 20])
    kind = DoubleInt.__lang_struct__  # type: ignore[attr-defined]
    assert (kind.size, kind.alignment, kind.offsets()) == (16, 8, [0, 8])


def test_a_struct_is_a_python_dataclass() -> None:
    assert Color(1, 2, 3, 4) == Color(r=1, g=2, b=3, a=4)
    assert Vector2(1.0, 2.0).y == 2.0


@needs_cc
def test_structs_cross_into_c_by_value(clib: Path) -> None:
    c = Module("abi_calls", libraries=[clib])

    @c.extern
    def abi_color_bits(color: Color) -> u32: ...
    @c.extern
    def abi_color_make(r: u8, g: u8, b: u8, a: u8) -> Color: ...
    @c.extern
    def abi_v2_add(a: Vector2, b: Vector2) -> Vector2: ...
    @c.extern
    def abi_v3_scale(v: Vector3, s: f32) -> Vector3: ...
    @c.extern
    def abi_rect_area(r: Rectangle) -> f32: ...
    @c.extern
    def abi_rect_grow(r: Rectangle, by: f32) -> Rectangle: ...
    @c.extern
    def abi_mixed(m: Mixed) -> f32: ...
    @c.extern
    def abi_double_int(v: DoubleInt) -> f64: ...
    @c.extern
    def abi_double_int_make(d: f64, i: i32) -> DoubleInt: ...
    @c.extern
    def abi_pairs(a: Pair, b: Pair, c: Pair, d: Pair, e: i64) -> i64: ...
    @c.extern
    def abi_camera(camera: Camera2D) -> f32: ...
    @c.extern
    def abi_camera_make(zoom: f32) -> Camera2D: ...
    @c.extern
    def abi_sprite(s: Sprite) -> u32: ...

    @c.function
    def color() -> u32:
        return abi_color_bits(Color(1, 2, 3, 4))

    @c.function
    def color_round_trip() -> u32:
        made = abi_color_make(10, 20, 30, 40)
        return abi_color_bits(made) + u32(made.g)

    @c.function
    def vectors() -> f32:
        v = abi_v2_add(Vector2(1.5, 2.0), Vector2(x=0.25, y=-4.0))
        w = abi_v3_scale(Vector3(1.0, 2.0, 3.0), 2.0)
        return v.x * 100.0 + v.y + w.x * 1000.0 + w.y * 10000.0 + w.z * 100000.0

    @c.function
    def rectangles() -> f32:
        grown = abi_rect_grow(Rectangle(1.0, 2.0, 3.0, 4.0), 0.5)
        return abi_rect_area(grown) + grown.x * 1000.0

    @c.function
    def mixed() -> f64:
        made = abi_double_int_make(1.25, 3)
        return (
            f64(abi_mixed(Mixed(3, 1.5))) + abi_double_int(made) * 100.0 + f64(made.i)
        )

    @c.function
    def pairs() -> i64:
        return abi_pairs(Pair(5, 1), Pair(9, 2), Pair(8, 1), Pair(4, 2), 3)

    @c.function
    def cameras() -> f32:
        passed = abi_camera(Camera2D(Vector2(1.0, 2.0), Vector2(3.0, 4.0), 5.0, 6.0))
        made = abi_camera_make(0.5)
        return passed * 1000.0 + made.target.y * 10.0 + made.zoom

    @c.function
    def sprites() -> u32:
        return abi_sprite(Sprite(Vector2(1.0, 2.0), Color(7, 0, 0, 0)))

    assert color() == 0x04030201
    assert color_round_trip() == (10 | 20 << 8 | 30 << 16 | 40 << 24) + 20
    # v = (1.75, -2.0); w = (2, 4, 6)
    assert vectors() == pytest.approx(175.0 - 2.0 + 2000.0 + 40000.0 + 600000.0)
    # grown = (0.5, 1.5, 4, 5): area 20 + 0.5 - 1.5
    assert rectangles() == pytest.approx(19.0 + 500.0)
    assert mixed() == pytest.approx(4.5 + 375.0 + 3.0)
    assert pairs() == 4 + 70 + 700 + 2000 + 30000
    assert cameras() == pytest.approx(91.0 * 1000.0 + 40.0 + 0.5)
    assert sprites() == 12 * 1000 + 7


@needs_cc
def test_bool_and_small_integers_follow_c(clib: Path) -> None:
    c = Module("abi_scalars", libraries=[clib])

    @c.extern
    def abi_is_positive(x: i32) -> bool: ...
    @c.extern
    def abi_bool_byte(b: bool) -> i32: ...
    @c.extern
    def abi_widen(small: i8, medium: u16) -> i32: ...

    @c.function
    def classify(x: i32) -> i32:
        if abi_is_positive(x):
            return 1
        return 0

    @c.function
    def bool_byte(x: i32) -> i32:
        return abi_bool_byte(x > 5)

    @c.function
    def widen() -> i32:
        return abi_widen(-3, 65000)

    assert [classify(-2), classify(0), classify(9)] == [0, 0, 1]
    assert [bool_byte(9), bool_byte(1)] == [1, 0]
    assert widen() == -300000 + 65000


@needs_cc
def test_structs_link_into_executables(clib: Path, tmp_path: Path) -> None:
    program = Program(libraries=[clib])

    @program.extern
    def abi_camera_make(zoom: f32) -> Camera2D: ...
    @program.extern
    def abi_color_bits(color: Color) -> u32: ...

    @program.main
    def main() -> i32:
        camera = abi_camera_make(2.0)
        tint = Color(1, 0, 0, 0)
        tint.g = u8(camera.zoom)
        bits = abi_color_bits(tint)  # 1 | 2 << 8; exit statuses are 8 bits
        return i32(bits // 256 * 10 + bits % 256)

    executable = program.build_executable(tmp_path / "app")
    env = {"LD_LIBRARY_PATH": str(clib.parent)}
    result = subprocess.run([executable], env=env, check=False)
    assert result.returncode == 21


def test_fields_are_values_updated_in_place() -> None:
    m = Module("struct_values")

    @m.function
    def walk(steps: i32) -> f32:
        position = Vector2(0.0, 0.0)
        for i in range(steps):
            position.x += 1.0
            if i % 2 == 0:
                position.y = position.y - 0.5
        return position.x * 10.0 + position.y

    @m.function
    def nested() -> f32:
        camera = Camera2D(Vector2(0.0, 0.0), Vector2(1.0, 2.0), 0.0, 1.0)
        camera.target.y = 7.0
        copy = camera
        copy.zoom = 3.0
        return camera.target.y * 10.0 + camera.zoom + copy.zoom * 100.0

    assert walk(4) == pytest.approx(40.0 - 1.0)
    assert nested() == pytest.approx(70.0 + 1.0 + 300.0)


def test_structs_between_compiled_functions() -> None:
    m = Module("struct_functions")

    @m.function
    def scaled(v: Vector2, by: f32) -> Vector2:
        return Vector2(v.x * by, v.y * by)

    @m.function
    def length_squared(v: Vector2) -> f32:
        return v.x * v.x + v.y * v.y

    @m.function
    def run() -> f32:
        return length_squared(scaled(Vector2(3.0, 4.0), 2.0))

    assert run() == pytest.approx(100.0)


def test_structs_in_memory() -> None:
    m = Module("struct_memory")

    @m.function
    def total(count: i32) -> i32:
        pixels: Ptr[Color] = stack(Color, 8)
        for i in range(count):
            pixels[i] = Color(u8(i), 0, 0, 255)
            pixels[i].g = u8(i * 2)
        sum = 0
        for i in range(count):
            sum += i32(pixels[i].r) + i32(pixels[i].g)
        return sum

    assert total(5) == 3 * (0 + 1 + 2 + 3 + 4)


def test_struct_mistakes_are_compile_errors(tmp_path: Path) -> None:
    def fails(body: str) -> str:
        path = tmp_path / "snippet.py"
        path.write_text(
            "from mlir_python.lang import *\n"
            "@struct\nclass Vector2:\n    x: f32\n    y: f32\n"
            "m = Module()\n"
            f"@m.function\ndef bad() -> i32:\n{body}\n    return 0\n"
        )
        namespace: dict[str, object] = {}
        exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
        module = namespace["m"]
        assert isinstance(module, Module)
        with pytest.raises(CompileError) as caught:
            _ = module.mlir
        return str(caught.value.msg)

    assert "no field 'w'" in fails("    Vector2(1.0, 2.0).w")
    assert "missing 'y'" in fails("    Vector2(1.0)")
    assert "given twice" in fails("    Vector2(1.0, x=2.0)")
    assert "do not support +" in fails("    Vector2(1.0, 2.0) + Vector2(1.0, 2.0)")
    assert "not a condition" in fails("    if Vector2(1.0, 2.0):\n        pass")
    assert "i64 has no fields" in fails("    x = 1\n    x.y = 2")


def test_python_cannot_pass_structs_yet() -> None:
    m = Module("struct_python")

    @m.function
    def length(v: Vector2) -> f32:
        return v.x

    with pytest.raises(TypeError, match="cannot cross from Python"):
        length(Vector2(1.0, 2.0))


def test_a_named_string_type_passes_as_a_string() -> None:
    # A compiler built on this names its own string types (one its holder
    # owns, say); they are the same pointer to text as cstr, both ways.
    owned = ScalarType("Owned", "cstr", 64)

    @struct
    class Named:
        text: owned
        count: i32

    m = Module("named_strings")

    @m.extern(name="strlen")
    def c_strlen(text: cstr) -> i64: ...

    @m.extern(name="strdup")
    def c_strdup(text: cstr) -> ptr: ...

    @m.extern(name="free")
    def c_free(pointer: ptr) -> None: ...

    @m.function
    def run() -> i64:
        named = Named(text="hello", count=2)
        copied = cstr(c_strdup(named.text))  # memory read as text, as C's (char *)p
        length = c_strlen(copied) * i64(named.count)
        c_free(copied)
        return length

    assert run() == 10
