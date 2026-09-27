# mlir-python

Typed Python bindings over the raw MLIR C++ API, built with CMake, vcpkg, and
nanobind. This project intentionally does not use MLIR's C API.

MLIR is a shallow Git submodule pinned to LLVM 22.1.8; initialize it after a
fresh clone with:

```sh
git submodule update --init --depth 1
```

## Prerequisites

- Python and [uv](https://docs.astral.sh/uv/)
- CMake 3.24 or newer, Ninja, and a C++17 compiler
- [vcpkg](https://learn.microsoft.com/vcpkg/get_started/get-started)

Set `VCPKG_ROOT` to the vcpkg checkout. The manifest installs nanobind; MLIR is
built from `third_party/llvm-project` by the top-level CMake project.

On Fedora, install the host tools required by vcpkg's OpenSSL and libb2 builds:

```sh
sudo dnf install perl-core perl-IPC-Cmd autoconf autoconf-archive automake libtool
```

## Development

```sh
export VCPKG_ROOT=/path/to/vcpkg
uv sync                       # Python tooling (pytest); does not build the extension
cmake --preset dev
cmake --build build/dev       # builds the extension and regenerates its stub
uv run pytest
```

The `dev` preset sets `MLIR_PYTHON_INPLACE`, which copies the built module into
`src/mlir_python/` (gitignored), registers `src/` in `.venv` (so `uv run
script.py` imports it), and regenerates the stubs in `src/mlir_python/_mlir_python/`
with nanobind's stubgen. Commit the regenerated stub with binding changes. clangd
reads `build/dev/compile_commands.json` through `.clangd`.

## Wheels

A wheel is self-contained: LLVM and MLIR are built from the submodule and linked
statically into the extension, so `pip install` needs nothing else (compiling to
executables still calls the system `cc`). MLIR's async runtime ships beside the
extension as `libmlir_async_runtime.so`. Wheels use CPython's stable ABI, so one
wheel per platform serves Python 3.12 and later (`cp312-abi3`).

```sh
uv build --wheel                  # dist/mlir_python-*-cp312-abi3-linux_x86_64.whl
uv run --isolated --with dist/*.whl python -c 'import mlir_python as ir; print(ir.IndexType(context=ir.Context()))'
```

The `Wheels` GitHub Action (`.github/workflows/wheels.yml`) builds portable
`manylinux_2_28` wheels for x86_64 and aarch64 with cibuildwheel, runs the test
suite against each installed wheel, and uploads them as workflow artifacts. To
reproduce it locally:

```sh
CIBW_CONTAINER_ENGINE=podman uvx cibuildwheel --platform linux   # or docker
```

### Releasing and installing elsewhere

To release, bump `version` in `pyproject.toml` and push to main. When the
version has no `v<version>` tag yet, the Wheels workflow builds and tests the
wheels, then creates release `v<version>` (and its tag) at that commit with the
wheels attached. Pushing a `v*` tag by hand also releases, as long as it matches
the version. `.github/workflows/index.yml` then republishes a flat package index
of every release's wheels to GitHub Pages. It also reruns when a release is
edited or deleted.

Other projects install from that index. With uv:

```toml
[project]
dependencies = ["mlir-python"]

[[tool.uv.index]]
name = "mlir-python"
url = "https://qinbeans.github.io/mlir-python/"
format = "flat"
explicit = true            # only mlir-python comes from here

[tool.uv.sources]
mlir-python = { index = "mlir-python" }
```

With pip: `pip install mlir-python --find-links https://qinbeans.github.io/mlir-python/`.
Each link carries the wheel's SHA-256, so installers verify downloads and uv
records the hash in the consumer's lockfile.

## Writing programs

`mlir_python.lang` compiles ordinary, type-checked Python functions to native
code. MLIR concepts (regions, blocks, insertion points) stay out of sight:

```python
from mlir_python.lang import Program, cstr, i32

program = Program()

@program.extern
def puts(text: cstr) -> i32: ...          # defined in the C library

@program.function
def collatz_steps(n: i32) -> i32:
    steps = 0                             # inferred as i32 from its use
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

@program.main                             # the entry point, explicitly
def main() -> None:
    puts("Hello, world!")

collatz_steps(27)                         # 111: compiled and run through the JIT
program.build_executable("hello")         # a native binary running main
print(program)                            # the MLIR; program.llvm_ir() for LLVM IR
```

- Types come from annotations: `i8`-`i64`, `u8`-`u64`, `f32`, `f64`, `bool`,
  `cstr`, `ptr`, and `tuple[...]` for several results (`int`/`float` mean
  `i64`/`f64`). Type checkers see them as `int`/`float`/`str`, so bodies are
  plain Python; the compiler infers unannotated variables' types from use.
- Bodies support arithmetic with Python semantics (`//` floors, `%` follows
  the divisor, `/` on integers gives a float), comparisons, `and`/`or`/`not`,
  conditional expressions, `if`/`elif`/`else`, `while`, `for i in range(...)`,
  `break`, `continue`, `return`, conversions like `i32(x)`, `abs`, `min`,
  `max`, and calls between the program's functions (recursion included).
- Arrays: `Array[T]` (e.g. `Array[i32]`) is created with a list display
  (`[1, 2, 3]`) or `array(f64, n)`, indexed like a list (`xs[i]`, `xs[-1]`,
  `xs[i] = v`), measured with `len`, iterated with `for x in xs`, and passed
  and returned. Out-of-range indices stop the program with a message, and
  arrays are freed automatically. To type checkers an `Array[T]` is a
  `list[T]`, so Python lists pass in and come back out through the JIT;
  NumPy arrays and other buffers are accepted without copying. `ptr(xs)`
  hands one to C.
- C functions are declared with `@program.extern` and a `...` body; `*args`
  declares a variadic one such as `printf(format: cstr, *args: int | float | str)`,
  and calls apply C's argument promotions (`f32` to `double`, small integers
  and `bool` to `int`).
- Memory: `stack(T, count=1)` reserves stack memory and returns a `Ptr[T]`;
  `p[i]` reads and `p[i] = x` writes. Pass pointers to C, e.g.
  `value = stack(i32); scanf("%d", value); value[0]`. `Ptr[T]` is also a
  parameter type, any `Ptr[T]` passes as an opaque `ptr`, and `Ptr[T](raw)`
  types one. `@program.extern(name="printf")` separates the Python name from
  the C symbol.
- Structs: a `@struct` class declares a C struct (and is an ordinary
  dataclass to Python and type checkers). Compiled code constructs it
  (`Color(245, 245, 245, 255)`, or by field name), reads and assigns fields
  (`c.r`, `v.x += 1.0`, `camera.target.y = 0.0`; structs are values, as in C),
  stores it in memory (`Ptr[Color]`, `stack(Color, n)`, `p[i].r = 0`), and
  passes it by value. Calls to externs follow the C calling convention of
  x86-64 and AArch64, including `bool` and small-integer extension, so C
  libraries take and return structs as C code would:

  ```python
  @struct
  class Color:
      r: u8
      g: u8
      b: u8
      a: u8

  @raylib.extern
  def ClearBackground(color: Color) -> None: ...
  @raylib.extern
  def WindowShouldClose() -> bool: ...
  ```

  Structs cannot yet be passed between Python and a JIT-compiled function,
  and functions you compile pass them in MLIR's own convention, so C code
  calling a `build_shared_library` export cannot take structs by value.
- Libraries: `Module(..., libraries=[...])` names the libraries defining its
  externs, in link order: a name the system searches for (`"m"` for libm) or
  a `Path` to a `.so` or `.a` file. They are loaded for JIT calls and linked
  into every executable or shared library that imports the module; a static
  archive loaded into the JIT must be compiled with `-fPIC`.

  ```python
  raylib = Module("raylib", libraries=[Path("lib/libraylib.a"), "m"])
  ```
- Mistakes raise `CompileError`, shown like a `SyntaxError` with the file,
  line, and a caret; reading a variable not assigned on every path is an error.
- Split code across files with `Module`: a compilation unit whose functions
  other modules import and call (`from mathlib import cube`). Whatever a module
  imports, transitively, is linked in when it runs or is built; a symbol
  defined twice is a link error. `Program` is a `Module` that can also have
  `@program.main` and `build_executable`.
- `module.mlir` is one module's MLIR (imports appear as declarations) and
  `module.linked()` the linked whole, for dropping down a level. Loops come
  out as structured control flow (`scf.for` for counted loops, `scf.while`
  and `scf.if` otherwise), ready for further passes.

## Building IR directly

```python
import mlir_python as ir
from mlir_python import passes
from mlir_python.dialects import arith, func, memref, scf

with ir.Context():
    f32, index = ir.F32Type(), ir.IndexType()
    buffer_type = ir.MemRefType([16], f32)
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        kernel = func.FuncOp("sum", ir.FunctionType([buffer_type], [f32]))
    with ir.InsertionPoint(kernel.add_entry_block()):
        (buffer,) = kernel.arguments
        lb, ub, step = (arith.ConstantOp(ir.IntegerAttr(v, index)).result for v in (0, 16, 1))
        zero = arith.ConstantOp(ir.FloatAttr(0.0, f32)).result
        loop = scf.ForOp(lb, ub, step, init_args=[zero])  # body block created for you
        with ir.InsertionPoint(loop.body):
            element = memref.LoadOp(buffer, [loop.induction_variable])
            total = arith.AddFOp(loop.inner_iter_args[0], element.result)
            scf.YieldOp([total.result])
        func.ReturnOp(loop.results)

    module.verify()
    ir.PassManager(ir.Module, [passes.Canonicalizer(), ir.Nested(func.FuncOp, [passes.CSE()])]).run(module)
    print(module)
```

Every signature and docstring is in `src/mlir_python/_mlir_python.pyi`; `help()`
shows the same text.

- **Context and defaults:** `Context`, `Location`, and `InsertionPoint` work as
  `with` blocks. Calls that take `context=`, `location=`, or `ip=` fall back to
  the innermost one.
- **Types and attributes:** every builtin type and the common builtin attributes
  are classes with typed constructors and properties, e.g. `RankedTensorType([4, None], F32Type())`,
  `DenseIntElementsAttr([1, 2], type)`. Objects returned by the API are always
  the most specific class, so narrow with `isinstance`. Dynamic sizes are `None`.
- **Dialects:** `mlir_python.dialects.<name>` has one class per operation,
  generated from MLIR's operation definitions: named, typed constructor
  parameters (enums such as `arith.CmpIPredicate` instead of strings or
  integers), typed accessors, and MLIR's documentation as docstrings. IR
  navigation returns these classes, so `isinstance` narrows operations.
- **IR:** `Operation` (and `Module`, which is one), `Block`, `Region`, `Value`
  (`OpResult`, `BlockArgument`), `walk`, `clone`, `detach_from_parent`, and
  `move_before`/`move_after`. `Operation.create(name, ...)` is the untyped
  fallback for operations without generated classes.
- **Passes:** `mlir_python.passes` has one class per linked pass, with typed,
  documented options and MLIR's defaults. Pipelines are built from them:
  `PassManager(Module, [passes.Canonicalizer(max_iterations=3), Nested(func.FuncOp, [passes.CSE()])])`.
  Placement is checked when the pipeline is built. `PassManager.parse(text)`
  accepts MLIR's textual syntax as a fallback.
- **Errors:** MLIR failures raise `MLIRError` carrying the emitted diagnostics;
  invalid arguments raise `ValueError`, `TypeError`, or `OverflowError` instead
  of aborting the process.

Linked dialects: arith, async, bufferization, cf, emitc, func, irdl, llvm,
math, memref, scf, tensor, and vector. A dialect named like a Python keyword
gets a `_dialect` suffix: `mlir_python.dialects.async_dialect`. Also linked:
MLIR's core transform passes (canonicalize, cse, inline, symbol-dce, ...), the
bufferization, async, vector, and EmitC passes, the conversions to LLVM and
to EmitC, and mlir-python's own passes (`scf-uplift-while-to-for`,
`convert-math-to-emitc-libm`). Other dialects parse only with
`allow_unregistered_dialects=True`.

### Compiling

`mlir_python.codegen` takes a module from the dialects above to machine code:

```python
from mlir_python import codegen

compiled = codegen.compile(module)            # lowers a copy; `module` is unchanged
total = compiled.function(kernel)             # the func.FuncOp you built
compiled.llvm_ir.verify()                     # LLVM's own verifier
compiled.llvm_ir.write_object("kernels.o")    # PIC object for the host; link with cc

codegen.build_executable(module, "app")       # native binary; needs func.func @main
codegen.build_shared_library(module, "libkernels.so")
```

- `build_executable` and `build_shared_library` lower, optimize, verify, and
  link a copy of the module with the system C compiler (`$CC`, `cc`, `clang`,
  or `gcc`; or `linker=`). An executable's `main` must be `() -> i32` or
  `(i32, !llvm.ptr) -> i32`, returning the exit status. Linker failures raise
  `codegen.LinkError` with the linker's output.

- `compiled.function(fn)(*args)` calls the JIT-compiled function. Integer,
  index, `f32`, and `f64` arguments and results cross the boundary, checked
  against the function's type; several results come back as a tuple.
- Memref arguments take writable buffers (NumPy arrays, `array.array`,
  `memoryview`), which the function reads and writes in place. Element type,
  rank, static sizes, and static strides are checked; strided views work where
  the memref type's strides are dynamic.
- Memref results come back as a `memoryview` copy, shaped like the memref.
  When the function allocated the buffer for its caller (after
  `pipelines.buffer_deallocation`, below), `compiled.function(fn,
  owned_results=True)` frees it once copied.
- `libraries=[...]` (names like `"m"`, or `Path`s to `.so` and `.a` files)
  loads external code for the JIT and links it into built binaries.
- Each stage is available separately: `llvm_lowering_pipeline()` (typed passes
  to inspect or extend), `lower_to_llvm`, `translate_to_llvm_ir` (an
  `LLVMModule` to `verify`, `optimize`, `write_object`, or print as IR or
  `assembly`), and `ExecutionEngine`.
- Code generation targets the build machine (`LLVM_TARGETS_TO_BUILD=Native`).

### Tensors and buffers

`mlir_python.pipelines` has typed versions of MLIR's named pipelines.
`bufferize()` turns value-semantic tensors into memrefs (one-shot
bufferization, across function boundaries), and `buffer_deallocation()` frees
every heap buffer once nothing uses it, tracking ownership through branches,
loops, and calls:

```python
from mlir_python import pipelines

ir.PassManager(ir.Module, pipelines.bufferize()).run(module)  # includes deallocation
compiled = codegen.compile(module)
scaled = compiled.function(scale, owned_results=True)(numpy_array)  # a memoryview
```

### Vectors

The vector dialect's SIMD operations lower to LLVM vector instructions for
the host. Multi-dimensional transfers are unrolled into loops, and
`vector.multi_reduction` is reduced to one-dimensional reductions, so vector
code runs in the JIT and in built binaries like any other code:

```mlir
%v = vector.transfer_read %buffer[%i], %pad : memref<?xf32>, vector<8xf32>  // masked past the end
%s = vector.reduction <add>, %v : vector<8xf32> into f32
```

`mlir_python.dialects.vector` builds the same operations (`vector.LoadOp`,
`vector.ReductionOp(f32, vector.CombiningKind.ADD, v)`, ...), and the vector
passes (`passes.LowerVectorMask`, `passes.LowerVectorMultiReduction`, ...)
are available for custom pipelines.

### Async

`async.execute` starts a task, and `async.await` waits for its value. Tasks
lower to LLVM coroutines that call an async runtime. mlir-python bundles MLIR's
reference runtime, a thread pool (`codegen.async_runtime()`), and loads it
automatically for JIT code. It also links the runtime into binaries that use
async:

```python
from mlir_python.dialects import async_dialect

task = async_dialect.execute(results=[i64])        # body block ready to fill
with ir.InsertionPoint(task.body):
    async_dialect.YieldOp([arith.MulIOp(x, x).result])
value = async_dialect.await_value(task.body_results[0])
```

`async_runtime=Path("libmyruntime.so")` on `compile`, `build_executable`,
or `build_shared_library` swaps in another implementation of the same C API
(`mlirAsyncRuntimeExecute`, ...), for example one driven by an event loop.
The JIT's runtime is process-wide: after the first `compile` that uses
async, a different runtime is refused with a `ValueError`.

### C and C++ source

`codegen.to_c(module)` and `codegen.to_cpp(module)` translate a copy of a
module into readable source, via the EmitC dialect, for toolchains MLIR
does not target. Functions keep their names. `memref<4xf32>` becomes
`float v[4]`, and math operations call `<math.h>` (or `<cmath>`):

```python
source = codegen.to_c(module)      # compile with any C99 compiler
```

Memrefs must have static shapes. The stages are also available separately:
`emitc_lowering_pipeline`, `lower_to_emitc`, and `translate_to_cpp`.
`mlir_python.dialects.emitc` and its types (`OpaqueType`, `PointerType`,
`ArrayType`, ...) build EmitC directly.

### Dialects defined in Python

`mlir_python.irdl` declares new dialects at runtime, through IRDL, with no C++
or rebuild. Their operations are created, verified against the declared
constraints, inspected, and rewritten like built-in ones:

```python
from mlir_python import irdl

bifrost = irdl.Dialect("bifrost")
Mutex = bifrost.type("mutex")                          # !bifrost.mutex

@bifrost.operation
class Guard(irdl.Op):
    """Locks ``mutex``; ``value`` is readable while it is held."""

    mutex = irdl.Operand(irdl.BaseOf(Mutex))
    value = irdl.Result(irdl.Any())

guard = Guard.create(mutex, value=ir.IntegerType(32))
guard.value, Guard.all(module)                         # typed accessors; every guard
```

`dialects.irdl.load_dialects(module)` loads dialects written directly in the
IRDL dialect.

### Passes written in Python

Subclass `passes.PythonPass` and implement `run`. The result goes anywhere in a
pipeline, between MLIR's own passes, and dataclass fields become its
options:

```python
@dataclasses.dataclass(frozen=True, kw_only=True)
class LowerGuards(passes.PythonPass):
    """Rewrites bifrost.guard into runtime calls."""

    def run(self, op: ir.Operation) -> None:
        for guard in Guard.all(op):
            ...                                         # build replacements, then
            guard.replace_with([replacement])

ir.PassManager(ir.Module, [LowerGuards(), passes.Canonicalizer()]).run(module)
```

`ANCHOR = func.FuncOp` runs the pass on each function (inside `Nested`). An
exception raised in `run` fails the pipeline and is re-raised from
`PassManager.run`.

### Ownership

Every object keeps its `Context` alive, and every IR handle keeps the IR tree it
points into alive. A detached operation (created without an insertion point) is
owned by Python until it is inserted into a block, and definitions stay alive
while anything uses them.

Running a pass makes handles into the rewritten IR stale: using one raises
`ValueError` rather than touching IR the pass may have freed. The operation the
pass ran on, and its ancestors, stay valid; navigate again from there.
`Operation.erase()` destroys the operation immediately: don't use other handles
to it or to IR nested in it afterwards.
