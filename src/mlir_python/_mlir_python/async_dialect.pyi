"""Typed operations of the MLIR ``async`` dialect."""

from collections.abc import Sequence

import mlir_python._mlir_python

class AddToGroupOp(mlir_python._mlir_python.Operation):
    """
    ``async.add_to_group``: adds an async token or value to the group.

    The `async.add_to_group` adds an async token or value to the async group.
    Returns the rank of the added element in the group. This rank is fixed
    for the group lifetime.

    Example:

    ```mlir
    %0 = async.create_group %size : !async.group
    %1 = ... : !async.token
    %2 = async.add_to_group %1, %0 : !async.token
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        group: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.add_to_group``: adds an async token or value to the group.

        Result types are inferred.

        Args:
            operand: Operand ``operand`` (async value type or async token type).
            group: Operand ``group`` (async group type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: async value type or async token type."""

    @property
    def group(self) -> mlir_python._mlir_python.Value:
        """Operand ``group``: async group type."""

    @property
    def rank(self) -> mlir_python._mlir_python.OpResult:
        """Result ``rank``: index."""

    OPERATION_NAME: str = "async.add_to_group"

class AwaitAllOp(mlir_python._mlir_python.Operation):
    """
    ``async.await_all``: waits for the all async tokens or values in the group to become ready.

    The `async.await_all` operation waits until all the tokens or values in the
    group become ready.

    Example:

    ```mlir
    %0 = async.create_group %size : !async.group

    %1 = ... : !async.token
    %2 = async.add_to_group %1, %0 : !async.token

    %3 = ... : !async.token
    %4 = async.add_to_group %2, %0 : !async.token

    async.await_all %0
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.await_all``: waits for the all async tokens or values in the group to become ready.

        Args:
            operand: Operand ``operand`` (async group type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: async group type."""

    OPERATION_NAME: str = "async.await_all"

class AwaitOp(mlir_python._mlir_python.Operation):
    """
    ``async.await``: waits for the argument to become ready.

    The `async.await` operation waits until the argument becomes ready, and for
    the `async.value` arguments it unwraps the underlying value

    Example:

    ```mlir
    %0 = ... : !async.token
    async.await %0 : !async.token

    %1 = ... : !async.value<f32>
    %2 = async.await %1 : !async.value<f32>
    ```
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        result_type: mlir_python._mlir_python.Type | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.await``: waits for the argument to become ready.

        Args:
            operand: Operand ``operand`` (async value type or async token type).
            result_type: Type of result ``result`` (any type). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: async value type or async token type."""

    OPERATION_NAME: str = "async.await"

class CallOp(mlir_python._mlir_python.Operation):
    """
    ``async.call``: async call operation.

    The `async.call` operation represents a direct call to an async function
    that is within the same symbol scope as the call. The operands and result
    types of the call must match the specified async function type. The callee
    is encoded as a symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = async.call @my_add(%0, %1) : (f32, f32) -> !async.value<f32>
    ```
    """

    def __init__(
        self,
        callee: str,
        operands: Sequence[mlir_python._mlir_python.Value] = [],
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        arg_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.call``: async call operation.

        Args:
            callee: Attribute ``callee`` (flat symbol reference attribute).
            operands: Operand ``operands`` (any type). Empty by default.
            result_types: Type of result ``result`` (async value type or async token type). Empty by default.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def callee(self) -> str:
        """Attribute ``callee``: flat symbol reference attribute."""

    @callee.setter
    def callee(self, arg: str, /) -> None: ...
    @property
    def arg_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``arg_attrs``: Array of dictionary attributes."""

    @arg_attrs.setter
    def arg_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def res_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``res_attrs``: Array of dictionary attributes."""

    @res_attrs.setter
    def res_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...

    OPERATION_NAME: str = "async.call"

class CoroBeginOp(mlir_python._mlir_python.Operation):
    """
    ``async.coro.begin``: returns a handle to the coroutine.

    The `async.coro.begin` allocates a coroutine frame and returns a handle to
    the coroutine.
    """

    def __init__(
        self,
        id: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.coro.begin``: returns a handle to the coroutine.

        Result types are inferred.

        Args:
            id: Operand ``id`` (switched-resume coroutine identifier).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def id(self) -> mlir_python._mlir_python.Value:
        """Operand ``id``: switched-resume coroutine identifier."""

    @property
    def handle(self) -> mlir_python._mlir_python.OpResult:
        """Result ``handle``: coroutine handle."""

    OPERATION_NAME: str = "async.coro.begin"

class CoroEndOp(mlir_python._mlir_python.Operation):
    """
    ``async.coro.end``: marks the end of the coroutine in the suspend block.

    The `async.coro.end` marks the point where a coroutine needs to return
    control back to the caller if it is not an initial invocation of the
    coroutine. It the start part of the coroutine is is no-op.
    """

    def __init__(
        self,
        handle: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.coro.end``: marks the end of the coroutine in the suspend block.

        Args:
            handle: Operand ``handle`` (coroutine handle).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def handle(self) -> mlir_python._mlir_python.Value:
        """Operand ``handle``: coroutine handle."""

    OPERATION_NAME: str = "async.coro.end"

class CoroFreeOp(mlir_python._mlir_python.Operation):
    """
    ``async.coro.free``: deallocates the coroutine frame.

    The `async.coro.free` deallocates the coroutine frame created by the
    async.coro.begin operation.
    """

    def __init__(
        self,
        id: mlir_python._mlir_python.Value,
        handle: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.coro.free``: deallocates the coroutine frame.

        Args:
            id: Operand ``id`` (switched-resume coroutine identifier).
            handle: Operand ``handle`` (coroutine handle).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def id(self) -> mlir_python._mlir_python.Value:
        """Operand ``id``: switched-resume coroutine identifier."""

    @property
    def handle(self) -> mlir_python._mlir_python.Value:
        """Operand ``handle``: coroutine handle."""

    OPERATION_NAME: str = "async.coro.free"

class CoroIdOp(mlir_python._mlir_python.Operation):
    """
    ``async.coro.id``: returns a switched-resume coroutine identifier.

    The `async.coro.id` returns a switched-resume coroutine identifier.
    """

    def __init__(
        self,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.coro.id``: returns a switched-resume coroutine identifier.

        Result types are inferred.
        """

    @property
    def id(self) -> mlir_python._mlir_python.OpResult:
        """Result ``id``: switched-resume coroutine identifier."""

    OPERATION_NAME: str = "async.coro.id"

class CoroSaveOp(mlir_python._mlir_python.Operation):
    """
    ``async.coro.save``: saves the coroutine state.

    The `async.coro.saves` saves the coroutine state.
    """

    def __init__(
        self,
        handle: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.coro.save``: saves the coroutine state.

        Result types are inferred.

        Args:
            handle: Operand ``handle`` (coroutine handle).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def handle(self) -> mlir_python._mlir_python.Value:
        """Operand ``handle``: coroutine handle."""

    @property
    def state(self) -> mlir_python._mlir_python.OpResult:
        """Result ``state``: saved coroutine state."""

    OPERATION_NAME: str = "async.coro.save"

class CoroSuspendOp(mlir_python._mlir_python.Operation):
    """
    ``async.coro.suspend``: suspends the coroutine.

    The `async.coro.suspend` suspends the coroutine and transfers control to the
    `suspend` successor. If suspended coroutine later resumed it will transfer
    control to the `resume` successor. If it is destroyed it will transfer
    control to the the `cleanup` successor.

    In switched-resume lowering coroutine can be already in resumed state when
    suspend operation is called, in this case control will be transferred to the
    `resume` successor skipping the `suspend` successor.
    """

    def __init__(
        self,
        state: mlir_python._mlir_python.Value,
        suspend_dest: mlir_python._mlir_python.Block,
        resume_dest: mlir_python._mlir_python.Block,
        cleanup_dest: mlir_python._mlir_python.Block,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.coro.suspend``: suspends the coroutine.

        Args:
            state: Operand ``state`` (saved coroutine state).
            suspend_dest: Successor ``suspendDest`` (any successor).
            resume_dest: Successor ``resumeDest`` (any successor).
            cleanup_dest: Successor ``cleanupDest`` (any successor).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def state(self) -> mlir_python._mlir_python.Value:
        """Operand ``state``: saved coroutine state."""

    @property
    def suspend_dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``suspendDest``."""

    @property
    def resume_dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``resumeDest``."""

    @property
    def cleanup_dest(self) -> mlir_python._mlir_python.Block:
        """Successor ``cleanupDest``."""

    OPERATION_NAME: str = "async.coro.suspend"

class CreateGroupOp(mlir_python._mlir_python.Operation):
    """
    ``async.create_group``: creates an empty async group.

    The `async.create_group` allocates an empty async group. Async tokens or
    values can be added to this group later. The size of the group must be
    specified at construction time, and `await_all` operation will first
    wait until the number of added tokens or values reaches the group size.

    Example:

    ```mlir
    %size = ... : index
    %group = async.create_group %size : !async.group
    ...
    async.await_all %group
    ```
    """

    def __init__(
        self,
        size: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.create_group``: creates an empty async group.

        Result types are inferred.

        Args:
            size: Operand ``size`` (index).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def size(self) -> mlir_python._mlir_python.Value:
        """Operand ``size``: index."""

    OPERATION_NAME: str = "async.create_group"

class ExecuteOp(mlir_python._mlir_python.Operation):
    """
    ``async.execute``: Asynchronous execute operation.

     The `body` region attached to the `async.execute` operation semantically
     can be executed concurrently with the successor operation. In the followup
     example "compute0" can be executed concurrently with "compute1".

     The actual concurrency semantics depends on the dialect lowering to the
     executable format. Fully sequential execution ("compute0" completes before
     "compute1" starts) is a completely legal execution.

     Because concurrent execution is not guaranteed, it is illegal to create an
     implicit dependency from "compute1" to "compute0" (e.g. via shared global
     state). All dependencies must be made explicit with async execute arguments
     (`async.token` or `async.value`).

    `async.execute` operation takes `async.token` dependencies and `async.value`
     operands separately, and starts execution of the attached body region only
     when all tokens and values become ready.

     Example:

     ```mlir
     %dependency = ... : !async.token
     %value = ... : !async.value<f32>

     %token, %results =
       async.execute [%dependency](%value as %unwrapped: !async.value<f32>)
                  -> !async.value<!some.type>
       {
         %0 = "compute0"(%unwrapped): (f32) -> !some.type
         async.yield %0 : !some.type
       }

     %1 = "compute1"(...) : !some.type
     ```

     In the example above asynchronous execution starts only after dependency
     token and value argument become ready. Unwrapped value passed to the
     attached body region as an %unwrapped value of f32 type.
    """

    def __init__(
        self,
        token_type: mlir_python._mlir_python.Type,
        dependencies: Sequence[mlir_python._mlir_python.Value] = [],
        body_operands: Sequence[mlir_python._mlir_python.Value] = [],
        body_results_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.execute``: Asynchronous execute operation.

        Args:
            token_type: Type of result ``token`` (async token type).
            dependencies: Operand ``dependencies`` (async token type). Empty by default.
            body_operands: Operand ``bodyOperands`` (async value type or async token type). Empty by default.
            body_results_types: Type of result ``bodyResults`` (async value type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def dependencies(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``dependencies``: async token type."""

    @property
    def body_operands(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``bodyOperands``: async value type or async token type."""

    @property
    def token(self) -> mlir_python._mlir_python.OpResult:
        """Result ``token``: async token type."""

    @property
    def body_results(self) -> list[mlir_python._mlir_python.OpResult]:
        """Result ``bodyResults``: async value type."""

    @property
    def body_region(self) -> mlir_python._mlir_python.Region:
        """Region ``bodyRegion``: region with 1 blocks."""

    OPERATION_NAME: str = "async.execute"

    @property
    def body(self) -> mlir_python._mlir_python.Block:
        """
        The task body. Without results it already ends in ``async.yield``,
        and ``InsertionPoint(body)`` inserts before it.
        """

class FuncOp(mlir_python._mlir_python.Operation):
    """
    ``async.func``: async function operation.

    An async function is like a normal function, but supports non-blocking
    await. Internally, async function is lowered to the LLVM coroutinue with
    async runtime intrinsic. It can return an async token and/or async values.
    The token represents the execution state of async function and can be used
    when users want to express dependencies on some side effects, e.g.,
    the token becomes available once every thing in the func body is executed.

    Example:

    ```mlir
    // Async function can't return void, it always must be some async thing.
    async.func @async.0() -> !async.token {
      return
    }

    // Function returns only async value.
    async.func @async.1() -> !async.value<i32> {
      %0 = arith.constant 42 : i32
      return %0 : i32
    }

    // Implicit token can be added to return types.
    async.func @async.2() -> !async.token, !async.value<i32> {
      %0 = arith.constant 42 : i32
      return %0 : i32
    }
    ```
    """

    def __init__(
        self,
        sym_name: str,
        function_type: mlir_python._mlir_python.FunctionType,
        *,
        sym_visibility: str | None = None,
        arg_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        res_attrs: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.func``: async function operation.

        Args:
            sym_name: Attribute ``sym_name`` (string attribute).
            function_type: Attribute ``function_type`` (type attribute of function type).
            sym_visibility: Attribute ``sym_visibility`` (string attribute). Optional.
            arg_attrs: Attribute ``arg_attrs`` (Array of dictionary attributes). Optional.
            res_attrs: Attribute ``res_attrs`` (Array of dictionary attributes). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def sym_name(self) -> str:
        """Attribute ``sym_name``: string attribute."""

    @sym_name.setter
    def sym_name(self, arg: str, /) -> None: ...
    @property
    def function_type(self) -> mlir_python._mlir_python.FunctionType:
        """Attribute ``function_type``: type attribute of function type."""

    @function_type.setter
    def function_type(self, arg: mlir_python._mlir_python.FunctionType, /) -> None: ...
    @property
    def sym_visibility(self) -> str | None:
        """Attribute ``sym_visibility``: string attribute."""

    @sym_visibility.setter
    def sym_visibility(self, arg: str | None) -> None: ...
    @property
    def arg_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``arg_attrs``: Array of dictionary attributes."""

    @arg_attrs.setter
    def arg_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def res_attrs(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``res_attrs``: Array of dictionary attributes."""

    @res_attrs.setter
    def res_attrs(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def body(self) -> mlir_python._mlir_python.Region:
        """Region ``body``: any region."""

    OPERATION_NAME: str = "async.func"

class ReturnOp(mlir_python._mlir_python.Operation):
    """
    ``async.return``: Async function return operation.

    The `async.return` is a special terminator operation for Async function.

    Example:

    ```mlir
    async.func @foo() : !async.token {
      return
    }
    ```
    """

    def __init__(
        self,
        operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.return``: Async function return operation.

        Args:
            operands: Operand ``operands`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "async.return"

class RuntimeAddRefOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.add_ref``: adds a reference to async value.

    The `async.runtime.add_ref` operation adds a reference(s) to async value
    (token, value or group).
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        count: int,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.add_ref``: adds a reference to async value.

        Args:
            operand: Operand ``operand`` (async value type or async token type or async group type).
            count: Attribute ``count`` (64-bit signless integer attribute whose value is positive).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand``: async value type or async token type or async group type.
        """

    @property
    def count(self) -> int:
        """
        Attribute ``count``: 64-bit signless integer attribute whose value is positive.
        """

    @count.setter
    def count(self, arg: int, /) -> None: ...

    OPERATION_NAME: str = "async.runtime.add_ref"

class RuntimeAddToGroupOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.add_to_group``: adds an async token or value to the group.

    The `async.runtime.add_to_group` adds an async token or value to the async
    group. Returns the rank of the added element in the group.
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        group: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.add_to_group``: adds an async token or value to the group.

        Result types are inferred.

        Args:
            operand: Operand ``operand`` (async value type or async token type).
            group: Operand ``group`` (async group type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: async value type or async token type."""

    @property
    def group(self) -> mlir_python._mlir_python.Value:
        """Operand ``group``: async group type."""

    @property
    def rank(self) -> mlir_python._mlir_python.OpResult:
        """Result ``rank``: index."""

    OPERATION_NAME: str = "async.runtime.add_to_group"

class RuntimeAwaitAndResumeOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.await_and_resume``: awaits the async operand and resumes the coroutine.

    The `async.runtime.await_and_resume` operation awaits for the operand to
    become available or error and resumes the coroutine on a thread managed by
    the runtime.
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        handle: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.await_and_resume``: awaits the async operand and resumes the coroutine.

        Args:
            operand: Operand ``operand`` (async value type or async token type or async group type).
            handle: Operand ``handle`` (coroutine handle).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand``: async value type or async token type or async group type.
        """

    @property
    def handle(self) -> mlir_python._mlir_python.Value:
        """Operand ``handle``: coroutine handle."""

    OPERATION_NAME: str = "async.runtime.await_and_resume"

class RuntimeAwaitOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.await``: blocks the caller thread until the operand becomes available.

    The `async.runtime.await` operation blocks the caller thread until the
    operand becomes available or error.
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.await``: blocks the caller thread until the operand becomes available.

        Args:
            operand: Operand ``operand`` (async value type or async token type or async group type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand``: async value type or async token type or async group type.
        """

    OPERATION_NAME: str = "async.runtime.await"

class RuntimeCreateGroupOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.create_group``: creates an async runtime group.

    The `async.runtime.create_group` operation creates an async dialect group
    of the given size. Group created in the empty state.
    """

    def __init__(
        self,
        size: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.create_group``: creates an async runtime group.

        Result types are inferred.

        Args:
            size: Operand ``size`` (index).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def size(self) -> mlir_python._mlir_python.Value:
        """Operand ``size``: index."""

    OPERATION_NAME: str = "async.runtime.create_group"

class RuntimeCreateOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.create``: creates an async runtime token or value.

    The `async.runtime.create` operation creates an async dialect token or
    value. Tokens and values are created in the non-ready state.
    """

    def __init__(
        self,
        result_type: mlir_python._mlir_python.Type,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.create``: creates an async runtime token or value.

        Args:
            result_type: Type of result ``result`` (async value type or async token type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "async.runtime.create"

class RuntimeDropRefOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.drop_ref``: drops a reference to async value.

    The `async.runtime.drop_ref` operation drops a reference(s) to async value
    (token, value or group).
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        count: int,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.drop_ref``: drops a reference to async value.

        Args:
            operand: Operand ``operand`` (async value type or async token type or async group type).
            count: Attribute ``count`` (64-bit signless integer attribute whose value is positive).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand``: async value type or async token type or async group type.
        """

    @property
    def count(self) -> int:
        """
        Attribute ``count``: 64-bit signless integer attribute whose value is positive.
        """

    @count.setter
    def count(self, arg: int, /) -> None: ...

    OPERATION_NAME: str = "async.runtime.drop_ref"

class RuntimeIsErrorOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.is_error``: returns true if token, value or group is in error state.

    The `async.runtime.is_error` operation returns true if the token, value or
    group (any of the async runtime values) is in the error state. It is the
    caller responsibility to check error state after the call to `await` or
    resuming after `await_and_resume`.
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.is_error``: returns true if token, value or group is in error state.

        Result types are inferred.

        Args:
            operand: Operand ``operand`` (async value type or async token type or async group type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """
        Operand ``operand``: async value type or async token type or async group type.
        """

    @property
    def is_error(self) -> mlir_python._mlir_python.OpResult:
        """Result ``is_error``: 1-bit signless integer."""

    OPERATION_NAME: str = "async.runtime.is_error"

class RuntimeLoadOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.load``: loads the value from the runtime async.value.

    The `async.runtime.load` operation loads the value from the runtime
    async.value storage.
    """

    def __init__(
        self,
        storage: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.load``: loads the value from the runtime async.value.

        Result types are inferred.

        Args:
            storage: Operand ``storage`` (async value type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def storage(self) -> mlir_python._mlir_python.Value:
        """Operand ``storage``: async value type."""

    OPERATION_NAME: str = "async.runtime.load"

class RuntimeNumWorkerThreadsOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.num_worker_threads``: gets the number of threads in the threadpool from the runtime.

    The `async.runtime.num_worker_threads` operation gets the number of threads
    in the threadpool from the runtime.
    """

    def __init__(
        self,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.num_worker_threads``: gets the number of threads in the threadpool from the runtime.

        Result types are inferred.
        """

    OPERATION_NAME: str = "async.runtime.num_worker_threads"

class RuntimeResumeOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.resume``: resumes the coroutine on a thread managed by the runtime.

    The `async.runtime.resume` operation resumes the coroutine on a thread
    managed by the runtime.
    """

    def __init__(
        self,
        handle: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.resume``: resumes the coroutine on a thread managed by the runtime.

        Args:
            handle: Operand ``handle`` (coroutine handle).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def handle(self) -> mlir_python._mlir_python.Value:
        """Operand ``handle``: coroutine handle."""

    OPERATION_NAME: str = "async.runtime.resume"

class RuntimeSetAvailableOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.set_available``: switches token or value to available state.

    The `async.runtime.set_available` operation switches async token or value
    state to available.
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.set_available``: switches token or value to available state.

        Args:
            operand: Operand ``operand`` (async value type or async token type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: async value type or async token type."""

    OPERATION_NAME: str = "async.runtime.set_available"

class RuntimeSetErrorOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.set_error``: switches token or value to error state.

    The `async.runtime.set_error` operation switches async token or value
    state to error.
    """

    def __init__(
        self,
        operand: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.set_error``: switches token or value to error state.

        Args:
            operand: Operand ``operand`` (async value type or async token type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def operand(self) -> mlir_python._mlir_python.Value:
        """Operand ``operand``: async value type or async token type."""

    OPERATION_NAME: str = "async.runtime.set_error"

class RuntimeStoreOp(mlir_python._mlir_python.Operation):
    """
    ``async.runtime.store``: stores the value into the runtime async.value.

    The `async.runtime.store` operation stores the value into the runtime
    async.value storage.
    """

    def __init__(
        self,
        value: mlir_python._mlir_python.Value,
        storage: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.runtime.store``: stores the value into the runtime async.value.

        Args:
            value: Operand ``value`` (any type).
            storage: Operand ``storage`` (async value type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def value(self) -> mlir_python._mlir_python.Value:
        """Operand ``value``: any type."""

    @property
    def storage(self) -> mlir_python._mlir_python.Value:
        """Operand ``storage``: async value type."""

    OPERATION_NAME: str = "async.runtime.store"

class YieldOp(mlir_python._mlir_python.Operation):
    """
    ``async.yield``: terminator for Async execute operation.

    The `async.yield` is a special terminator operation for the block inside
    `async.execute` operation.
    """

    def __init__(
        self,
        operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``async.yield``: terminator for Async execute operation.

        Args:
            operands: Operand ``operands`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "async.yield"

class TokenType(mlir_python._mlir_python.Type):
    """``!async.token``: completion of an asynchronous task, without a value."""

    def __init__(
        self, *, context: mlir_python._mlir_python.Context | None = None
    ) -> None:
        """Create ``!async.token``."""

class ValueType(mlir_python._mlir_python.Type):
    """``!async.value<T>``: a ``T`` an asynchronous task will produce."""

    def __init__(self, value_type: mlir_python._mlir_python.Type) -> None:
        """Create ``!async.value<value_type>``."""

    @property
    def value_type(self) -> mlir_python._mlir_python.Type:
        """The type of the value."""

class GroupType(mlir_python._mlir_python.Type):
    """``!async.group``: a set of tokens to await together."""

    def __init__(
        self, *, context: mlir_python._mlir_python.Context | None = None
    ) -> None:
        """Create ``!async.group``."""
