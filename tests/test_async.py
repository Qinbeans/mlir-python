"""The async dialect: tasks lowered to LLVM coroutines and run by an async
runtime, the bundled one or a replacement."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import mlir_python as ir
from mlir_python import codegen
from mlir_python.dialects import func

CC = shutil.which("cc")
needs_cc = pytest.mark.skipif(CC is None, reason="needs a C compiler")

SQUARE = """
func.func @square(%x: i64) -> i64 {
  %token, %result = async.execute -> !async.value<i64> {
    %y = arith.muli %x, %x : i64
    async.yield %y : i64
  }
  %value = async.await %result : !async.value<i64>
  return %value : i64
}
"""

# Fills buffer[i] = i * i in n concurrent tasks, then sums the buffer.
PARALLEL = """
func.func @sum_of_squares(%buffer: memref<?xi64>) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i64
  %n = memref.dim %buffer, %c0 : memref<?xi64>
  %group = async.create_group %n : !async.group
  scf.for %i = %c0 to %n step %c1 {
    %token = async.execute {
      %x = arith.index_cast %i : index to i64
      %square = arith.muli %x, %x : i64
      memref.store %square, %buffer[%i] : memref<?xi64>
      async.yield
    }
    %rank = async.add_to_group %token, %group : !async.token
  }
  async.await_all %group
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> i64 {
    %v = memref.load %buffer[%i] : memref<?xi64>
    %next = arith.addi %acc, %v : i64
    scf.yield %next : i64
  }
  return %sum : i64
}
"""

MAIN = """
func.func @main() -> i32 {
  %token, %result = async.execute -> !async.value<i32> {
    %seven = arith.constant 7 : i32
    %six = arith.constant 6 : i32
    %product = arith.muli %seven, %six : i32
    async.yield %product : i32
  }
  %value = async.await %result : !async.value<i32>
  return %value : i32
}
"""

# A replacement runtime: single-threaded, running each task as soon as it is
# scheduled, and counting how many it ran.
INLINE_RUNTIME = r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef void (*Resume)(void *);
typedef struct Waiter { void *handle; Resume resume; struct Waiter *next; } Waiter;
typedef struct Object {
  int64_t refs;
  bool ready, error;
  int64_t pending;            /* groups: tokens not yet ready */
  struct Object *groups[8];   /* tokens: the groups they were added to */
  int groupCount;
  Waiter *waiters;
  char *storage;              /* values */
} Object;

static int64_t executed = 0;
int64_t inline_runtime_executed(void) { return executed; }

static Object *create(void) { Object *o = calloc(1, sizeof(Object)); o->refs = 1; return o; }
static void wake(Object *o) {
  Waiter *w = o->waiters;
  o->waiters = NULL;
  while (w) { Waiter *next = w->next; w->resume(w->handle); free(w); w = next; }
}
static void await(Object *o, void *handle, Resume resume) {
  if (o->ready) { resume(handle); return; }
  Waiter *w = malloc(sizeof(Waiter));
  w->handle = handle; w->resume = resume; w->next = o->waiters; o->waiters = w;
}
static void complete(Object *o, bool error) {
  o->ready = true; o->error = error;
  for (int i = 0; i < o->groupCount; ++i) {
    Object *g = o->groups[i];
    g->error |= error;
    if (--g->pending == 0) { g->ready = true; wake(g); }
  }
  wake(o);
}

void mlirAsyncRuntimeAddRef(void *p, int64_t n) { ((Object *)p)->refs += n; }
void mlirAsyncRuntimeDropRef(void *p, int64_t n) {
  Object *o = p;
  if ((o->refs -= n) == 0) { free(o->storage); free(o); }
}
void *mlirAsyncRuntimeCreateToken(void) { return create(); }
void *mlirAsyncRuntimeCreateValue(int64_t size) {
  Object *o = create(); o->storage = calloc(1, size > 0 ? size : 1); return o;
}
void *mlirAsyncRuntimeCreateGroup(int64_t size) {
  (void)size; Object *g = create(); g->ready = true; return g;
}
int64_t mlirAsyncRuntimeAddTokenToGroup(void *token, void *group) {
  Object *t = token, *g = group;
  if (!t->ready) { g->ready = false; g->pending++; t->groups[t->groupCount++] = g; }
  g->error |= t->error;
  return 0;
}
void mlirAsyncRuntimeEmplaceToken(void *t) { complete(t, false); }
void mlirAsyncRuntimeEmplaceValue(void *v) { complete(v, false); }
void mlirAsyncRuntimeSetTokenError(void *t) { complete(t, true); }
void mlirAsyncRuntimeSetValueError(void *v) { complete(v, true); }
bool mlirAsyncRuntimeIsTokenError(void *t) { return ((Object *)t)->error; }
bool mlirAsyncRuntimeIsValueError(void *v) { return ((Object *)v)->error; }
bool mlirAsyncRuntimeIsGroupError(void *g) { return ((Object *)g)->error; }
/* Everything runs inline, so whatever is awaited is already done. */
void mlirAsyncRuntimeAwaitToken(void *t) { if (!((Object *)t)->ready) abort(); }
void mlirAsyncRuntimeAwaitValue(void *v) { if (!((Object *)v)->ready) abort(); }
void mlirAsyncRuntimeAwaitAllInGroup(void *g) { if (!((Object *)g)->ready) abort(); }
char *mlirAsyncRuntimeGetValueStorage(void *v) { return ((Object *)v)->storage; }
void mlirAsyncRuntimeExecute(void *handle, Resume resume) { executed++; resume(handle); }
void mlirAsyncRuntimeAwaitTokenAndExecute(void *t, void *h, Resume r) { await(t, h, r); }
void mlirAsyncRuntimeAwaitValueAndExecute(void *v, void *h, Resume r) { await(v, h, r); }
void mlirAsyncRuntimeAwaitAllInGroupAndExecute(void *g, void *h, Resume r) { await(g, h, r); }
int64_t mlirAsyncRuntimGetNumWorkerThreads(void) { return 1; }
void mlirAsyncRuntimePrintCurrentThreadId(void) {}
"""


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def function(module: ir.Module, name: str) -> func.FuncOp:
    return next(
        op
        for op in module.body.operations
        if isinstance(op, func.FuncOp) and op.sym_name == name
    )


def test_bundled_runtime_ships_with_the_package() -> None:
    runtime = codegen.async_runtime()
    assert runtime.is_file()
    assert runtime.parent == Path(codegen.__file__).resolve().parent


@pytest.mark.parametrize("level", [codegen.OptLevel.O0, codegen.OptLevel.O2])
def test_async_tasks_run_in_the_jit(level: codegen.OptLevel) -> None:
    module = ir.Module.parse(SQUARE)
    compiled = codegen.compile(module, opt_level=level)
    square = compiled.function(function(module, "square"))
    assert [square(n) for n in (0, 3, -12)] == [0, 9, 144]


def test_concurrent_tasks_share_a_buffer() -> None:
    import array

    module = ir.Module.parse(PARALLEL)
    compiled = codegen.compile(module)
    total = compiled.function(function(module, "sum_of_squares"))
    buffer = array.array("q", [0] * 1000)
    assert total(buffer) == sum(i * i for i in range(1000))
    assert buffer[999] == 999 * 999


@needs_cc
def test_async_executables_link_the_runtime(tmp_path: Path) -> None:
    module = ir.Module.parse(MAIN)
    executable = codegen.build_executable(module, tmp_path / "app")
    assert subprocess.run([executable], check=False).returncode == 42


REPLACED_JIT = """
import sys
from pathlib import Path
import mlir_python as ir
from mlir_python import codegen
from mlir_python.dialects import func

source, library = sys.argv[1], Path(sys.argv[2])
with ir.Context():
    module = ir.Module.parse(source)
    compiled = codegen.compile(module, async_runtime=library)
    ops = {op.sym_name: op for op in module.body.operations if isinstance(op, func.FuncOp)}
    square = compiled.function(ops["square"])
    executed = compiled.function(ops["executed"])
    print(square(9), square(4), executed())
    try:
        codegen.compile(module)  # the bundled runtime, after another one
    except ValueError as error:
        print("refused:", "process-wide" in str(error))
"""


@needs_cc
def test_the_runtime_can_be_replaced(tmp_path: Path) -> None:
    assert CC is not None
    source = tmp_path / "inline_runtime.c"
    source.write_text(INLINE_RUNTIME)
    library = tmp_path / "libinline_runtime.so"
    subprocess.run(
        [CC, "-shared", "-fPIC", "-O1", str(source), "-o", str(library)], check=True
    )
    counted = (
        SQUARE
        + """
func.func private @inline_runtime_executed() -> i64
func.func @executed() -> i64 {
  %n = func.call @inline_runtime_executed() : () -> i64
  return %n : i64
}
"""
    )
    # The JIT's runtime is process-wide, so this runs in a fresh process.
    result = subprocess.run(
        [sys.executable, "-c", REPLACED_JIT, counted, str(library)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # Both tasks ran on the replacement runtime; switching runtimes is refused.
    assert result.stdout.splitlines() == ["81 16 2", "refused: True"]

    program = ir.Module.parse(MAIN)
    app = codegen.build_executable(program, tmp_path / "app", async_runtime=library)
    assert subprocess.run([app], check=False).returncode == 42
    needed = subprocess.run(
        ["readelf", "-d", str(app)], capture_output=True, text=True, check=False
    ).stdout
    assert "libinline_runtime.so" in needed


def test_code_without_async_needs_no_runtime() -> None:
    module = ir.Module.parse("func.func @f(%x: i64) -> i64 { return %x : i64 }")
    compiled = codegen.compile(module, async_runtime=Path("/nonexistent/runtime.so"))
    assert compiled.function(function(module, "f"))(5) == 5


def test_tasks_built_with_the_typed_api() -> None:
    from mlir_python.dialects import arith, async_dialect

    i64 = ir.IntegerType(64)
    module = ir.Module()
    with ir.Location.unknown():
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp("cube_later", ir.FunctionType([i64], [i64]))
        with ir.InsertionPoint(fn.add_entry_block()):
            (x,) = fn.arguments
            square = async_dialect.execute(results=[i64])
            with ir.InsertionPoint(square.body):
                async_dialect.YieldOp([arith.MulIOp(x, x).result])
            # A second task receives the first one's value as an argument.
            cube = async_dialect.execute(results=[i64], operands=square.body_results)
            with ir.InsertionPoint(cube.body):
                (squared,) = cube.body.arguments
                async_dialect.YieldOp([arith.MulIOp(squared, x).result])
            done = async_dialect.execute(dependencies=[cube.token])  # no results
            async_dialect.AwaitOp(done.token)
            func.ReturnOp([async_dialect.await_value(cube.body_results[0])])
    module.verify()
    assert isinstance(cube.body_results[0].type, async_dialect.ValueType)
    assert cube.body_results[0].type.value_type == i64
    compiled = codegen.compile(module)
    assert compiled.function(fn)(5) == 125
