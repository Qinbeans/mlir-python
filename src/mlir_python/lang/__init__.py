"""Write compiled programs as ordinary Python functions.

Example::

    from mlir_python.lang import Program, cstr, i32

    program = Program()

    @program.extern
    def puts(text: cstr) -> i32: ...

    @program.function
    def add(a: i32, b: i32) -> i32:
        return a + b

    @program.main
    def main() -> None:
        puts("Hello, world!")

    add(2, 3)                            # 5, run through the JIT
    main()                               # prints "Hello, world!"
    program.build_executable("hello")    # a native binary
    print(program)                       # the MLIR; program.llvm_ir() for LLVM IR

Function bodies are type-checked Python: the scalar types (``i32``, ``f64``,
``cstr``, ...) are aliases of ``int``, ``float``, and ``str`` to type
checkers, and the compiler takes the exact widths from the annotations.

Supported: arithmetic with Python semantics (``//`` floors, ``%`` follows the
divisor's sign, ``/`` on integers gives ``f64``), comparisons (chained too),
``and``/``or``/``not``, conditional expressions, ``if``/``elif``/``else``,
``while``, ``for i in range(...)``, ``break``, ``continue``, ``return``
(tuples for several results), conversions such as ``i32(x)``, ``abs``,
``min``, ``max``, and calls between the program's functions. Mistakes raise
``CompileError``, shown like a ``SyntaxError`` at the offending line.

Group values with ``@struct`` classes (C structs, passed to C functions by
value as C does)::

    @struct
    class Color:
        r: u8
        g: u8
        b: u8
        a: u8

    @program.extern
    def ClearBackground(color: Color) -> None: ...

Functions are values of type ``Fn[[params], result]`` (C function
pointers): name one where a value is expected, call a value like a
function, and pass them to C as callbacks::

    @program.extern
    def qsort(base: ptr, count: i64, size: i64, compare: Fn[[ptr, ptr], i32]) -> None: ...

    @program.function
    def apply(f: Fn[[i32], i32], x: i32) -> i32:
        return f(x)                      # apply(double, 21)

Split code across files with ``Module``: a compilation unit whose functions
other modules import (``from mathlib import cube``) and call from compiled
code; everything a program imports is linked in automatically. A ``Program``
is a ``Module`` that can also have ``@program.main`` and build executables.

The MLIR layer underneath stays available: ``module.mlir`` is an ordinary
``mlir_python.Module`` (``module.linked()`` includes the imports).
"""

from ._compiler import CompileError
from ._program import Function, Module, Program
from .types import (
    Array,
    Fn,
    Ptr,
    array,
    cstr,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    ptr,
    stack,
    struct,
    u8,
    u16,
    u32,
    u64,
)

__all__ = [
    "Array",
    "CompileError",
    "Fn",
    "Function",
    "Module",
    "Program",
    "Ptr",
    "array",
    "cstr",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "ptr",
    "stack",
    "struct",
    "u8",
    "u16",
    "u32",
    "u64",
]
