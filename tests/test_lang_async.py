"""``async def`` in compiled code: ``await`` for async functions and ``Token``s.

In the JIT the bundled runtime runs the tasks. The executable test brings an
event-loop runtime in C, like a server's: it calls async handlers through
function values, and completes the tokens they wait on out of order.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from mlir_python.lang import CompileError, Fn, Module, Program, Token, cstr, i32, i64

CC = shutil.which("cc")
needs_cc = pytest.mark.skipif(CC is None, reason="needs a C compiler")

program = Program()


@program.function
async def square(x: i64) -> i64:
    return x * x


@program.function
async def sum_of_squares(n: i64) -> i64:
    total = 0
    for i in range(n):
        if i % 2 == 0:
            total = total + await square(i)
        else:
            total = total + i * i
    return total


@program.function
def blocking(n: i64) -> i64:
    return sum_of_squares(n)  # a plain function waits until it is done


def test_async_functions_run_in_the_jit() -> None:
    assert blocking(10) == sum(i * i for i in range(10))


def test_python_cannot_call_an_async_function_directly() -> None:
    with pytest.raises(TypeError, match="is async; call it from a compiled function"):
        square(3)


# A single-threaded runtime on a small event loop. `later(n)` returns a token
# completed n turns later; `serve` starts a handler per request, through its
# function value, then turns the loop until every handler is done.
LOOP_RUNTIME = r"""
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef void (*Resume)(void *);
typedef struct Waiter { void *handle; Resume resume; struct Waiter *next; } Waiter;
typedef struct Object { int64_t refs; bool ready, error; Waiter *waiters; char *storage; } Object;

static struct { void *handle; Resume resume; } ready[256];
static int head, tail;
static struct { Object *token; int64_t turns; } timers[64];
static int timerCount;

static void schedule(void *h, Resume r) { ready[tail % 256].handle = h; ready[tail++ % 256].resume = r; }
static Object *create(void) { Object *o = calloc(1, sizeof(Object)); o->refs = 1; return o; }
static void wake(Object *o) {
  Waiter *w = o->waiters; o->waiters = NULL;
  while (w) { Waiter *n = w->next; schedule(w->handle, w->resume); free(w); w = n; }
}
static void complete(Object *o, bool error) { o->ready = true; o->error = error; wake(o); }
void mlirAsyncRuntimeDropRef(void *p, int64_t n);
static void turn(void) {
  while (head != tail) { int i = head++ % 256; ready[i].resume(ready[i].handle); }
  for (int i = 0; i < timerCount; ++i)
    if (--timers[i].turns == 0) {
      Object *o = timers[i].token;
      timers[i--] = timers[--timerCount];
      complete(o, false);
      mlirAsyncRuntimeDropRef(o, 1);  /* the timer's reference */
    }
}

void *later(int64_t turns) {
  Object *o = create(); o->refs++;
  timers[timerCount].token = o; timers[timerCount++].turns = turns;
  return o;
}

static int64_t order;
void record(int64_t request) { order = order * 10 + request; }

int64_t serve(void *(*handler)(int64_t), int64_t requests) {
  Object *tokens[16];
  for (int64_t i = 0; i < requests; ++i) tokens[i] = handler(i + 1);
  for (bool done = false; !done;) {
    turn();
    done = true;
    for (int64_t i = 0; i < requests; ++i) done &= tokens[i]->ready;
  }
  for (int64_t i = 0; i < requests; ++i) mlirAsyncRuntimeDropRef(tokens[i], 1);
  return order;
}

void mlirAsyncRuntimeAddRef(void *p, int64_t n) { ((Object *)p)->refs += n; }
void mlirAsyncRuntimeDropRef(void *p, int64_t n) {
  Object *o = p; if ((o->refs -= n) == 0) { free(o->storage); free(o); }
}
void *mlirAsyncRuntimeCreateToken(void) { return create(); }
void *mlirAsyncRuntimeCreateValue(int64_t size) { Object *o = create(); o->storage = calloc(1, size > 0 ? size : 1); return o; }
void *mlirAsyncRuntimeCreateGroup(int64_t size) { (void)size; Object *g = create(); g->ready = true; return g; }
int64_t mlirAsyncRuntimeAddTokenToGroup(void *t, void *g) { (void)t; (void)g; abort(); }
void mlirAsyncRuntimeEmplaceToken(void *t) { complete(t, false); }
void mlirAsyncRuntimeEmplaceValue(void *v) { complete(v, false); }
void mlirAsyncRuntimeSetTokenError(void *t) { complete(t, true); }
void mlirAsyncRuntimeSetValueError(void *v) { complete(v, true); }
bool mlirAsyncRuntimeIsTokenError(void *t) { return ((Object *)t)->error; }
bool mlirAsyncRuntimeIsValueError(void *v) { return ((Object *)v)->error; }
bool mlirAsyncRuntimeIsGroupError(void *g) { return ((Object *)g)->error; }
void mlirAsyncRuntimeAwaitToken(void *t) { while (!((Object *)t)->ready) turn(); }
void mlirAsyncRuntimeAwaitValue(void *v) { while (!((Object *)v)->ready) turn(); }
void mlirAsyncRuntimeAwaitAllInGroup(void *g) { while (!((Object *)g)->ready) turn(); }
char *mlirAsyncRuntimeGetValueStorage(void *v) { return ((Object *)v)->storage; }
void mlirAsyncRuntimeExecute(void *h, Resume r) { schedule(h, r); }
static void await(Object *o, void *h, Resume r) {
  if (o->ready) { schedule(h, r); return; }
  Waiter *w = malloc(sizeof(Waiter)); w->handle = h; w->resume = r; w->next = o->waiters; o->waiters = w;
}
void mlirAsyncRuntimeAwaitTokenAndExecute(void *t, void *h, Resume r) { await(t, h, r); }
void mlirAsyncRuntimeAwaitValueAndExecute(void *v, void *h, Resume r) { await(v, h, r); }
void mlirAsyncRuntimeAwaitAllInGroupAndExecute(void *g, void *h, Resume r) { await(g, h, r); }
int64_t mlirAsyncRuntimGetNumWorkerThreads(void) { return 1; }
void mlirAsyncRuntimePrintCurrentThreadId(void) {}
"""

server = Program()


@server.extern
def later(turns: i64) -> Token: ...


@server.extern
def record(request: i64) -> None: ...


@server.extern
def serve(handler: Fn[[i64], Token], requests: i64) -> i64: ...


@server.function
async def pause(turns: i64) -> None:
    await later(turns)


@server.function
async def handle(request: i64) -> None:
    await pause(4 - request)  # the first request waits longest
    record(request)


@server.main
def main() -> i32:
    return i32(serve(handle, 3) % 256)


@needs_cc
def test_an_event_loop_runs_async_handlers(tmp_path: Path) -> None:
    assert CC is not None
    source = tmp_path / "loop.c"
    source.write_text(LOOP_RUNTIME)
    runtime = tmp_path / "libloop.so"
    subprocess.run(
        [CC, "-shared", "-fPIC", "-O1", str(source), "-o", str(runtime)], check=True
    )
    app = server.build_executable(
        tmp_path / "app", libraries=[runtime], async_runtime=runtime
    )
    # Handlers finish in the order their waits end: 3, 2, 1 (recorded as 321).
    assert subprocess.run([app], check=False).returncode == 321 % 256


def compile_error(source: str, tmp_path: Path) -> CompileError:
    path = tmp_path / "snippet.py"
    path.write_text("from mlir_python.lang import *\nprogram = Program()\n" + source)
    namespace: dict[str, object] = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
    built = namespace["program"]
    assert isinstance(built, Module)
    with pytest.raises(CompileError) as info:
        _ = built.mlir
    return info.value


ERRORS = [
    (
        (
            "@program.function\nasync def f() -> i64:\n    return 1\n"
            "@program.function\nasync def g() -> i64:\n    return f()\n"
        ),
        "f is async; wait for it with `await f()`",
    ),
    (
        (
            "@program.function\ndef f() -> i64:\n    return 1\n"
            "@program.function\nasync def g() -> i64:\n    return await f()\n"
        ),
        "f is not async, so there is nothing to await",
    ),
    (
        "@program.function\nasync def g(x: i64) -> None:\n    await x\n",
        "await an async function's call, or a Token",
    ),
    (
        "@program.function\nasync def f() -> tuple[i64, i64]:\n    return 1, 2\n",
        "an async function returns one value, not a tuple",
    ),
    (
        "@program.extern\nasync def f() -> None: ...\n",
        "an extern cannot be async",
    ),
    (
        (
            "@program.function\nasync def f() -> i64:\n    return 1\n"
            "@program.function\ndef g() -> Fn[[], i64]:\n    return f\n"
        ),
        "only an async function that returns nothing can be a function value",
    ),
]


@pytest.mark.parametrize(("source", "message"), ERRORS)
def test_errors(source: str, message: str, tmp_path: Path) -> None:
    assert message in str(compile_error(source, tmp_path))


blocking_program = Program()


@blocking_program.extern
def puts(text: cstr) -> i32: ...


@blocking_program.function
async def cube(x: i64) -> i64:
    return x * x * x


@blocking_program.main
def blocking_main() -> i32:
    puts("waiting")
    return i32(cube(3))  # blocks, without the `cf.assert` that clashes with this `puts`


@needs_cc
def test_blocking_works_beside_the_programs_own_puts(tmp_path: Path) -> None:
    app = blocking_program.build_executable(tmp_path / "app")
    result = subprocess.run([app], capture_output=True, text=True, check=False)
    assert (result.returncode, result.stdout) == (27, "waiting\n")
