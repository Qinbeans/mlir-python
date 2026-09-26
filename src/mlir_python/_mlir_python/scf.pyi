"""Typed operations of the MLIR ``scf`` dialect."""

from collections.abc import Sequence

import mlir_python._mlir_python

class ConditionOp(mlir_python._mlir_python.Operation):
    """
    ``scf.condition``: loop continuation condition.

    This operation accepts the continuation (i.e., inverse of exit) condition
    of the `scf.while` construct. If its first argument is true, the "after"
    region of `scf.while` is executed, with the remaining arguments forwarded
    to the entry block of the region. Otherwise, the loop terminates.
    """

    def __init__(
        self,
        condition: mlir_python._mlir_python.Value,
        args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.condition``: loop continuation condition.

        Args:
            condition: Operand ``condition`` (1-bit signless integer).
            args: Operand ``args`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def condition(self) -> mlir_python._mlir_python.Value:
        """Operand ``condition``: 1-bit signless integer."""

    @property
    def args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``args``: any type."""

    OPERATION_NAME: str = "scf.condition"

class ExecuteRegionOp(mlir_python._mlir_python.Operation):
    """
    ``scf.execute_region``: operation that executes its region exactly once.

    The `scf.execute_region` operation is used to allow multiple blocks within SCF
    and other operations which can hold only one block.  The `scf.execute_region`
    operation executes the region held exactly once and cannot have any operands.
    As such, its region has no arguments. All SSA values that dominate the op can
    be accessed inside the op. The op's region can have multiple blocks and the
    blocks can have multiple distinct terminators. Values returned from this op's
    region define the op's results.
    The optional 'no_inline' flag can be set to request the ExecuteRegionOp to be
    preserved as much as possible and not being inlined in the parent block until
    an explicit lowering step.

    Example:

    ```mlir
    scf.for %i = 0 to 128 step %c1 {
      %y = scf.execute_region -> i32 {
        %x = load %A[%i] : memref<128xi32>
        scf.yield %x : i32
      }
    }

    // the same as above but with no_inline attribute
    scf.for %i = 0 to 128 step %c1 {
      %y = scf.execute_region -> i32 no_inline {
        %x = load %A[%i] : memref<128xi32>
        scf.yield %x : i32
      }
    }

    affine.for %i = 0 to 100 {
      "foo"() : () -> ()
      %v = scf.execute_region -> i64 {
        cf.cond_br %cond, ^bb1, ^bb2

      ^bb1:
        %c1 = arith.constant 1 : i64
        cf.br ^bb3(%c1 : i64)

      ^bb2:
        %c2 = arith.constant 2 : i64
        cf.br ^bb3(%c2 : i64)

      ^bb3(%x : i64):
        scf.yield %x : i64
      }
      "bar"(%v) : (i64) -> ()
    }
    ```
    """

    def __init__(
        self,
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        no_inline: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.execute_region``: operation that executes its region exactly once.

        Args:
            result_types: Type of result ``result`` (any type). Empty by default.
            no_inline: Attribute ``no_inline`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def no_inline(self) -> bool:
        """Attribute ``no_inline``: unit attribute."""

    @no_inline.setter
    def no_inline(self, arg: bool, /) -> None: ...
    @property
    def region(self) -> mlir_python._mlir_python.Region:
        """Region ``region``: any region."""

    OPERATION_NAME: str = "scf.execute_region"

class ForOp(mlir_python._mlir_python.Operation):
    """
    ``scf.for``: for operation.

    The `scf.for` operation represents a loop taking 3 SSA value as operands
    that represent the lower bound, upper bound and step respectively. The
    operation defines an SSA value for its induction variable. It has one
    region capturing the loop body. The induction variable is represented as an
    argument of this region. This SSA value is a signless integer or index.
    The step is a value of same type but required to be positive, the lower and
    upper bounds can be also negative or zero. The lower and upper bounds
    specify a half-open range: the iteration is executed iff the comparison of
    induction variable value is less than the upper bound and bigger or equal
    to the lower bound.

    By default, the integer comparison is signed. If the `unsignedCmp` unit
    attribute is specified, the integer comparison is unsigned.

    The body region must contain exactly one block that terminates with
    `scf.yield`. Calling ForOp::build will create such a region and insert
    the terminator implicitly if none is defined, so will the parsing even in
    cases when it is absent from the custom format. For example:

    ```mlir
    // Index case.
    scf.for %iv = %lb to %ub step %step {
      ... // body
    }
    ...
    // Unsigned integer case.
    scf.for unsigned %iv_32 = %lb_32 to %ub_32 step %step_32 : i32 {
      ... // body
    }
    ```

    `scf.for` can also operate on loop-carried variables and returns the final
    values after loop termination. The initial values of the variables are
    passed as additional SSA operands to the `scf.for` following the 3 loop
    control SSA values mentioned above (lower bound, upper bound and step). The
    operation region has an argument for the induction variable, followed by
    one argument for each loop-carried variable, representing the value of the
    variable at the current iteration.

    The region must terminate with a `scf.yield` that passes the current
    values of all loop-carried variables to the next iteration, or to the
    `scf.for` result, if at the last iteration. The static type of a
    loop-carried variable may not change with iterations; its runtime type is
    allowed to change. Note, that when the loop-carried variables are present,
    calling ForOp::build will not insert the terminator implicitly. The caller
    must insert `scf.yield` in that case.

    `scf.for` results hold the final values after the last iteration.
    For example, to sum-reduce a memref:

    ```mlir
    func.func @reduce(%buffer: memref<1024xf32>, %lb: index,
                      %ub: index, %step: index) -> (f32) {
      // Initial sum set to 0.
      %sum_0 = arith.constant 0.0 : f32
      // iter_args binds initial values to the loop's region arguments.
      %sum = scf.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = load %buffer[%iv] : memref<1024xf32>
        %sum_next = arith.addf %sum_iter, %t : f32
        // Yield current iteration sum to next iteration %sum_iter or to %sum
        // if final iteration.
        scf.yield %sum_next : f32
      }
      return %sum : f32
    }
    ```

    If the `scf.for` defines any values, a yield must be explicitly present.
    The number and types of the `scf.for` results must match the initial
    values in the `iter_args` binding and the yield operands.

    Another example with a nested `scf.if` (see `scf.if` for details) to
    perform conditional reduction:

    ```mlir
    func.func @conditional_reduce(%buffer: memref<1024xf32>, %lb: index,
                                  %ub: index, %step: index) -> (f32) {
      %sum_0 = arith.constant 0.0 : f32
      %c0 = arith.constant 0.0 : f32
      %sum = scf.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = load %buffer[%iv] : memref<1024xf32>
        %cond = arith.cmpf "ugt", %t, %c0 : f32
        %sum_next = scf.if %cond -> (f32) {
          %new_sum = arith.addf %sum_iter, %t : f32
          scf.yield %new_sum : f32
        } else {
          scf.yield %sum_iter : f32
        }
        scf.yield %sum_next : f32
      }
      return %sum : f32
    }
    ```
    """

    def __init__(
        self,
        lower_bound: mlir_python._mlir_python.Value,
        upper_bound: mlir_python._mlir_python.Value,
        step: mlir_python._mlir_python.Value,
        init_args: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        unsigned_cmp: bool = False,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.for``: for operation.

        Result types match the types of ``init_args``.

        Args:
            lower_bound: Operand ``lowerBound`` (signless integer or index).
            upper_bound: Operand ``upperBound`` (signless integer or index).
            step: Operand ``step`` (signless integer or index).
            init_args: Operand ``initArgs`` (any type). Empty by default.
            unsigned_cmp: Attribute ``unsignedCmp`` (unit attribute). Omit for the default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lower_bound(self) -> mlir_python._mlir_python.Value:
        """Operand ``lowerBound``: signless integer or index."""

    @property
    def upper_bound(self) -> mlir_python._mlir_python.Value:
        """Operand ``upperBound``: signless integer or index."""

    @property
    def step(self) -> mlir_python._mlir_python.Value:
        """Operand ``step``: signless integer or index."""

    @property
    def init_args(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``initArgs``: any type."""

    @property
    def unsigned_cmp(self) -> bool:
        """Attribute ``unsignedCmp``: unit attribute."""

    @unsigned_cmp.setter
    def unsigned_cmp(self, arg: bool, /) -> None: ...
    @property
    def region(self) -> mlir_python._mlir_python.Region:
        """Region ``region``: region with 1 blocks."""

    OPERATION_NAME: str = "scf.for"

    @property
    def body(self) -> mlir_python._mlir_python.Block:
        """
        The loop body. Without init values it already ends in ``scf.yield``,
        and ``InsertionPoint(body)`` inserts before it.
        """

    @property
    def induction_variable(self) -> mlir_python._mlir_python.BlockArgument:
        """The loop counter, the body's first argument."""

    @property
    def inner_iter_args(self) -> list[mlir_python._mlir_python.BlockArgument]:
        """
        Body arguments carrying the loop-carried values; yield their next
        values with ``scf.YieldOp``.
        """

class ForallOp(mlir_python._mlir_python.Operation):
    """
    ``scf.forall``: evaluate a block multiple times in parallel.

    `scf.forall` is a target-independent multi-dimensional parallel
    region application operation. It has exactly one block that represents the
    parallel body and it takes index operands that specify lower bounds, upper
    bounds and steps.

    The op also takes a variadic number of tensor operands (`shared_outs`).
    The future buffers corresponding to these tensors are shared among all
    threads. Shared tensors should be accessed via their corresponding block
    arguments. If multiple threads write to a shared buffer in a racy
    fashion, these writes will execute in some unspecified order. Tensors that
    are not shared can be used inside the body (i.e., the op is not isolated
    from above); however, if a use of such a tensor bufferizes to a memory
    write, the tensor is privatized, i.e., a thread-local copy of the tensor is
    used. This ensures that memory side effects of a thread are not visible to
    other threads (or in the parent body), apart from explicitly shared tensors.

    The name "thread" conveys the fact that the parallel execution is mapped
    (i.e. distributed) to a set of virtual threads of execution, one function
    application per thread. Further lowerings are responsible for specifying
    how this is materialized on concrete hardware resources.

    An optional `mapping` is an attribute array that specifies processing units
    with their dimension, how it remaps 1-1 to a set of concrete processing
    element resources (e.g. a CUDA grid dimension or a level of concrete nested
    async parallelism). It is expressed via any attribute that implements the
    device mapping interface. It is the reponsibility of the lowering mechanism
    to interpret the `mapping` attributes in the context of the concrete target
    the op is lowered to, or to ignore it when the specification is ill-formed
    or unsupported for a particular target.

    The only allowed terminator is `scf.forall.in_parallel`.
    `scf.forall` returns one value per `shared_out` operand. The
    actions of the `scf.forall.in_parallel` terminators specify how to combine the
    partial results of all parallel invocations into a full value, in some
    unspecified order. The "destination" of each such op must be a `shared_out`
    block argument of the `scf.forall` op.

    The actions involved in constructing the return values are further described
    by `tensor.parallel_insert_slice`.

    `scf.forall` acts as an implicit synchronization point.

    When the parallel function body has side effects, their order is unspecified
    across threads.

    `scf.forall` can be printed in two different ways depending on
    whether the loop is normalized or not. The loop is 'normalized' when all
    lower bounds are equal to zero and steps are equal to one. In that case,
    `lowerBound` and `step` operands will be omitted during printing.

    Normalized loop example:

    ```mlir
    //
    // Sequential context.
    //
    %matmul_and_pointwise:2 = scf.forall (%thread_id_1, %thread_id_2) in
        (%num_threads_1, %numthread_id_2) shared_outs(%o1 = %C, %o2 = %pointwise)
      -> (tensor<?x?xT>, tensor<?xT>) {
      //
      // Parallel context, each thread with id = (%thread_id_1, %thread_id_2)
      // runs its version of the code.
      //
      %sA = tensor.extract_slice %A[f((%thread_id_1, %thread_id_2))]:
        tensor<?x?xT> to tensor<?x?xT>
      %sB = tensor.extract_slice %B[g((%thread_id_1, %thread_id_2))]:
        tensor<?x?xT> to tensor<?x?xT>
      %sC = tensor.extract_slice %o1[h((%thread_id_1, %thread_id_2))]:
        tensor<?x?xT> to tensor<?x?xT>
      %sD = linalg.matmul
        ins(%sA, %sB : tensor<?x?xT>, tensor<?x?xT>)
        outs(%sC : tensor<?x?xT>)

      %spointwise = subtensor %o2[i((%thread_id_1, %thread_id_2))]:
        tensor<?xT> to tensor<?xT>
      %sE = linalg.add ins(%spointwise : tensor<?xT>) outs(%sD : tensor<?xT>)

      scf.forall.in_parallel {
        tensor.parallel_insert_slice %sD into %o1[h((%thread_id_1, %thread_id_2))]:
          tensor<?x?xT> into tensor<?x?xT>

        tensor.parallel_insert_slice %spointwise into %o2[i((%thread_id_1, %thread_id_2))]:
          tensor<?xT> into tensor<?xT>
      }
    }
    // Implicit synchronization point.
    // Sequential context.
    //
    ```

    Loop with loop bounds example:

    ```mlir
    //
    // Sequential context.
    //
    %pointwise = scf.forall (%i, %j) = (0, 0) to (%dim1, %dim2)
      step (%tileSize1, %tileSize2) shared_outs(%o1 = %out)
      -> (tensor<?x?xT>, tensor<?xT>) {
      //
      // Parallel context.
      //
      %sA = tensor.extract_slice %A[%i, %j][%tileSize1, %tileSize2][1, 1]
        : tensor<?x?xT> to tensor<?x?xT>
      %sB = tensor.extract_slice %B[%i, %j][%tileSize1, %tileSize2][1, 1]
        : tensor<?x?xT> to tensor<?x?xT>
      %sC = tensor.extract_slice %o[%i, %j][%tileSize1, %tileSize2][1, 1]
        : tensor<?x?xT> to tensor<?x?xT>

      %add = linalg.map {"arith.addf"}
        ins(%sA, %sB : tensor<?x?xT>, tensor<?x?xT>)
        outs(%sC : tensor<?x?xT>)

      scf.forall.in_parallel {
        tensor.parallel_insert_slice %add into
          %o[%i, %j][%tileSize1, %tileSize2][1, 1]
          : tensor<?x?xT> into tensor<?x?xT>
      }
    }
    // Implicit synchronization point.
    // Sequential context.
    //
    ```

    Example with mapping attribute:

    ```mlir
    //
    // Sequential context. Here `mapping` is expressed as GPU thread mapping
    // attributes
    //
    %matmul_and_pointwise:2 = scf.forall (%thread_id_1, %thread_id_2) in
        (%num_threads_1, %numthread_id_2) shared_outs(...)
      -> (tensor<?x?xT>, tensor<?xT>) {
      //
      // Parallel context, each thread with id = **(%thread_id_2, %thread_id_1)**
      // runs its version of the code.
      //
       scf.forall.in_parallel {
         ...
      }
    } { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
    // Implicit synchronization point.
    // Sequential context.
    //
    ```

    Example with privatized tensors:

    ```mlir
    %t0 = ...
    %t1 = ...
    %r = scf.forall ... shared_outs(%o = t0) -> tensor<?xf32> {
      // %t0 and %t1 are privatized. %t0 is definitely copied for each thread
      // because the scf.forall op's %t0 use bufferizes to a memory
      // write. In the absence of other conflicts, %t1 is copied only if there
      // are uses of %t1 in the body that bufferize to a memory read and to a
      // memory write.
      "some_use"(%t0)
      "some_use"(%t1)
    }
    ```
    """

    def __init__(
        self,
        static_lower_bound: mlir_python._mlir_python.Attribute,
        static_upper_bound: mlir_python._mlir_python.Attribute,
        static_step: mlir_python._mlir_python.Attribute,
        dynamic_lower_bound: Sequence[mlir_python._mlir_python.Value] = [],
        dynamic_upper_bound: Sequence[mlir_python._mlir_python.Value] = [],
        dynamic_step: Sequence[mlir_python._mlir_python.Value] = [],
        outputs: Sequence[mlir_python._mlir_python.Value] = [],
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        mapping: mlir_python._mlir_python.ArrayAttr | None = None,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.forall``: evaluate a block multiple times in parallel.

        Args:
            static_lower_bound: Attribute ``staticLowerBound`` (i64 dense array attribute).
            static_upper_bound: Attribute ``staticUpperBound`` (i64 dense array attribute).
            static_step: Attribute ``staticStep`` (i64 dense array attribute).
            dynamic_lower_bound: Operand ``dynamicLowerBound`` (index). Empty by default.
            dynamic_upper_bound: Operand ``dynamicUpperBound`` (index). Empty by default.
            dynamic_step: Operand ``dynamicStep`` (index). Empty by default.
            outputs: Operand ``outputs`` (ranked tensor of any type values). Empty by default.
            result_types: Type of result ``results`` (any type). Empty by default.
            mapping: Attribute ``mapping`` (Device Mapping array attribute). Optional.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def dynamic_lower_bound(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``dynamicLowerBound``: index."""

    @property
    def dynamic_upper_bound(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``dynamicUpperBound``: index."""

    @property
    def dynamic_step(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``dynamicStep``: index."""

    @property
    def outputs(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``outputs``: ranked tensor of any type values."""

    @property
    def static_lower_bound(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``staticLowerBound``: i64 dense array attribute."""

    @static_lower_bound.setter
    def static_lower_bound(
        self, arg: mlir_python._mlir_python.Attribute, /
    ) -> None: ...
    @property
    def static_upper_bound(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``staticUpperBound``: i64 dense array attribute."""

    @static_upper_bound.setter
    def static_upper_bound(
        self, arg: mlir_python._mlir_python.Attribute, /
    ) -> None: ...
    @property
    def static_step(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``staticStep``: i64 dense array attribute."""

    @static_step.setter
    def static_step(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def mapping(self) -> mlir_python._mlir_python.ArrayAttr | None:
        """Attribute ``mapping``: Device Mapping array attribute."""

    @mapping.setter
    def mapping(self, arg: mlir_python._mlir_python.ArrayAttr | None) -> None: ...
    @property
    def region(self) -> mlir_python._mlir_python.Region:
        """Region ``region``: region with 1 blocks."""

    OPERATION_NAME: str = "scf.forall"

class IfOp(mlir_python._mlir_python.Operation):
    """
    ``scf.if``: if-then-else operation.

    The `scf.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    scf.if %b  {
      ...
    } else {
      ...
    }
    ```

    `scf.if` may also produce results. Which values are returned depends on
    which execution path is taken.

    Example:

    ```mlir
    %x, %y = scf.if %b -> (f32, f32) {
      %x_true = ...
      %y_true = ...
      scf.yield %x_true, %y_true : f32, f32
    } else {
      %x_false = ...
      %y_false = ...
      scf.yield %x_false, %y_false : f32, f32
    }
    ```

    The "then" region has exactly 1 block. The "else" region may have 0 or 1
    block. In case the `scf.if` produces results, the "else" region must also
    have exactly 1 block.

    The blocks are always terminated with `scf.yield`. If `scf.if` defines no
    values, the `scf.yield` can be left out, and will be inserted implicitly.
    Otherwise, it must be explicit.

    Example:

    ```mlir
    scf.if %b  {
      ...
    }
    ```

    The types of the yielded values must match the result types of the
    `scf.if`.
    """

    def __init__(
        self,
        condition: mlir_python._mlir_python.Value,
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.if``: if-then-else operation.

        Args:
            condition: Operand ``condition`` (1-bit signless integer).
            result_types: Type of result ``results`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def condition(self) -> mlir_python._mlir_python.Value:
        """Operand ``condition``: 1-bit signless integer."""

    @property
    def then_region(self) -> mlir_python._mlir_python.Region:
        """Region ``thenRegion``: region with 1 blocks."""

    @property
    def else_region(self) -> mlir_python._mlir_python.Region:
        """Region ``elseRegion``: region with at most 1 blocks."""

    OPERATION_NAME: str = "scf.if"

    @property
    def then_block(self) -> mlir_python._mlir_python.Block:
        """The block run when the condition holds."""

    @property
    def else_block(self) -> mlir_python._mlir_python.Block | None:
        """The block run otherwise, or ``None`` (see ``add_else_block``)."""

    def add_else_block(self) -> mlir_python._mlir_python.Block:
        """
        Add the else block (ending in ``scf.yield`` when the if has no
        results) and return it. An if with results already has one.

        Raises:
            ValueError: If the else block exists.
        """

class InParallelOp(mlir_python._mlir_python.Operation):
    """
    ``scf.forall.in_parallel``: terminates a `forall` block.

    The `scf.forall.in_parallel` is a designated terminator for
    the `scf.forall` operation.

    It has a single region with a single block that contains a flat list of ops.
    Each such op participates in the aggregate formation of a single result of
    the enclosing `scf.forall`.
    The result number corresponds to the position of the op in the terminator.
    """

    def __init__(
        self,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """Create ``scf.forall.in_parallel``: terminates a `forall` block."""

    @property
    def region(self) -> mlir_python._mlir_python.Region:
        """Region ``region``: region with 1 blocks."""

    OPERATION_NAME: str = "scf.forall.in_parallel"

class IndexSwitchOp(mlir_python._mlir_python.Operation):
    """
    ``scf.index_switch``: switch-case operation on an index argument.

    The `scf.index_switch` is a control-flow operation that branches to one of
    the given regions based on the values of the argument and the cases. The
    argument is always of type `index`.

    The operation always has a "default" region and any number of case regions
    denoted by integer constants. Control-flow transfers to the case region
    whose constant value equals the value of the argument. If the argument does
    not equal any of the case values, control-flow transfer to the "default"
    region.

    Example:

    ```mlir
    %0 = scf.index_switch %arg0 : index -> i32
    case 2 {
      %1 = arith.constant 10 : i32
      scf.yield %1 : i32
    }
    case 5 {
      %2 = arith.constant 20 : i32
      scf.yield %2 : i32
    }
    default {
      %3 = arith.constant 30 : i32
      scf.yield %3 : i32
    }
    ```
    """

    def __init__(
        self,
        arg: mlir_python._mlir_python.Value,
        cases: mlir_python._mlir_python.Attribute,
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        num_case_regions: int = 0,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.index_switch``: switch-case operation on an index argument.

        Args:
            arg: Operand ``arg`` (index).
            cases: Attribute ``cases`` (i64 dense array attribute).
            result_types: Type of result ``results`` (any type). Empty by default.
            num_case_regions: Number of ``caseRegions`` regions.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def arg(self) -> mlir_python._mlir_python.Value:
        """Operand ``arg``: index."""

    @property
    def cases(self) -> mlir_python._mlir_python.Attribute:
        """Attribute ``cases``: i64 dense array attribute."""

    @cases.setter
    def cases(self, arg: mlir_python._mlir_python.Attribute, /) -> None: ...
    @property
    def default_region(self) -> mlir_python._mlir_python.Region:
        """Region ``defaultRegion``: region with 1 blocks."""

    @property
    def case_regions(self) -> list[mlir_python._mlir_python.Region]:
        """Region ``caseRegions``: region with 1 blocks."""

    OPERATION_NAME: str = "scf.index_switch"

class ParallelOp(mlir_python._mlir_python.Operation):
    """
    ``scf.parallel``: parallel for operation.

    The `scf.parallel` operation represents a loop nest taking 4 groups of SSA
    values as operands that represent the lower bounds, upper bounds, steps and
    initial values, respectively. The operation defines a variadic number of
    SSA values for its induction variables. It has one region capturing the
    loop body. The induction variables are represented as an argument of this
    region. These SSA values always have type index, which is the size of the
    machine word. The steps are values of type index, required to be positive.
    The lower and upper bounds specify a half-open range: the range includes
    the lower bound but does not include the upper bound. The initial values
    have the same types as results of `scf.parallel`. If there are no results,
    the keyword `init` can be omitted.

    Semantically we require that the iteration space can be iterated in any
    order, and the loop body can be executed in parallel. If there are data
    races, the behavior is undefined.

    The parallel loop operation supports reduction of values produced by
    individual iterations into a single result. This is modeled using the
    `scf.reduce` terminator operation (see `scf.reduce` for details). The i-th
    result of an `scf.parallel` operation is associated with the i-th initial
    value operand, the i-th operand of the `scf.reduce` operation (the value to
    be reduced) and the i-th region of the `scf.reduce` operation (the reduction
    function). Consequently, we require that the number of results of an
    `scf.parallel` op matches the number of initial values and the the number of
    reductions in the `scf.reduce` terminator.

    The body region must contain exactly one block that terminates with a
    `scf.reduce` operation. If an `scf.parallel` op has no reductions, the
    terminator has no operands and no regions. The `scf.parallel` parser will
    automatically insert the terminator for ops that have no reductions if it is
    absent.

    Example:

    ```mlir
    %init = arith.constant 0.0 : f32
    %r:2 = scf.parallel (%iv) = (%lb) to (%ub) step (%step) init (%init, %init)
        -> f32, f32 {
      %elem_to_reduce1 = load %buffer1[%iv] : memref<100xf32>
      %elem_to_reduce2 = load %buffer2[%iv] : memref<100xf32>
      scf.reduce(%elem_to_reduce1, %elem_to_reduce2 : f32, f32) {
        ^bb0(%lhs : f32, %rhs: f32):
          %res = arith.addf %lhs, %rhs : f32
          scf.reduce.return %res : f32
      }, {
        ^bb0(%lhs : f32, %rhs: f32):
          %res = arith.mulf %lhs, %rhs : f32
          scf.reduce.return %res : f32
      }
    }
    ```
    """

    def __init__(
        self,
        lower_bound: Sequence[mlir_python._mlir_python.Value] = [],
        upper_bound: Sequence[mlir_python._mlir_python.Value] = [],
        step: Sequence[mlir_python._mlir_python.Value] = [],
        init_vals: Sequence[mlir_python._mlir_python.Value] = [],
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.parallel``: parallel for operation.

        Args:
            lower_bound: Operand ``lowerBound`` (index). Empty by default.
            upper_bound: Operand ``upperBound`` (index). Empty by default.
            step: Operand ``step`` (index). Empty by default.
            init_vals: Operand ``initVals`` (any type). Empty by default.
            result_types: Type of result ``results`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def lower_bound(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``lowerBound``: index."""

    @property
    def upper_bound(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``upperBound``: index."""

    @property
    def step(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``step``: index."""

    @property
    def init_vals(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``initVals``: any type."""

    @property
    def region(self) -> mlir_python._mlir_python.Region:
        """Region ``region``: region with 1 blocks."""

    OPERATION_NAME: str = "scf.parallel"

class ReduceOp(mlir_python._mlir_python.Operation):
    """
    ``scf.reduce``: reduce operation for scf.parallel.

    The `scf.reduce` operation is the terminator for `scf.parallel` operations. It can model
    an arbitrary number of reductions. It has one region per reduction. Each
    region has one block with two arguments which have the same type as the
    corresponding operand of `scf.reduce`. The operands of the op are the values
    that should be reduce; one value per reduction.

    The i-th reduction (i.e., the i-th region and the i-th operand) corresponds
    the i-th initial value and the i-th result of the enclosing `scf.parallel`
    op.

    The `scf.reduce` operation contains regions whose entry blocks expect two
    arguments of the same type as the corresponding operand. As the iteration
    order of the enclosing parallel loop and hence reduction order is
    unspecified, the results of the reductions may be non-deterministic unless
    the reductions are associative and commutative.

    The result of a reduction region (`scf.reduce.return` operand) must have the
    same type as the corresponding `scf.reduce` operand and the corresponding
    `scf.parallel` initial value.

    Example:

    ```mlir
    %operand = arith.constant 1.0 : f32
    scf.reduce(%operand : f32) {
      ^bb0(%lhs : f32, %rhs: f32):
        %res = arith.addf %lhs, %rhs : f32
        scf.reduce.return %res : f32
    }
    ```
    """

    def __init__(
        self,
        operands: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        num_reductions: int = 0,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.reduce``: reduce operation for scf.parallel.

        Args:
            operands: Operand ``operands`` (any type). Empty by default.
            num_reductions: Number of ``reductions`` regions.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def reductions(self) -> list[mlir_python._mlir_python.Region]:
        """Region ``reductions``: region with 1 blocks."""

    OPERATION_NAME: str = "scf.reduce"

class ReduceReturnOp(mlir_python._mlir_python.Operation):
    """
    ``scf.reduce.return``: terminator for reduce operation.

    The `scf.reduce.return` operation is a special terminator operation for the block inside
    `scf.reduce` regions. It terminates the region. It should have the same
    operand type as the corresponding operand of the enclosing `scf.reduce` op.

    Example:

    ```mlir
    scf.reduce.return %res : f32
    ```
    """

    def __init__(
        self,
        result: mlir_python._mlir_python.Value,
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.reduce.return``: terminator for reduce operation.

        Args:
            result: Operand ``result`` (any type).
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "scf.reduce.return"

class WhileOp(mlir_python._mlir_python.Operation):
    """
    ``scf.while``: a generic 'while' loop.

    This operation represents a generic "while"/"do-while" loop that keeps
    iterating as long as a condition is satisfied. There is no restriction on
    the complexity of the condition. It consists of two regions (with single
    block each): "before" region and "after" region. The names of regions
    indicates whether they execute before or after the condition check.
    Therefore, if the main loop payload is located in the "before" region, the
    operation is a "do-while" loop. Otherwise, it is a "while" loop.

    The "before" region terminates with a special operation, `scf.condition`,
    that accepts as its first operand an `i1` value indicating whether to
    proceed to the "after" region (value is `true`) or not. The two regions
    communicate by means of region arguments. Initially, the "before" region
    accepts as arguments the operands of the `scf.while` operation and uses them
    to evaluate the condition. It forwards the trailing, non-condition operands
    of the `scf.condition` terminator either to the "after" region if the
    control flow is transferred there or to results of the `scf.while` operation
    otherwise. The "after" region takes as arguments the values produced by the
    "before" region and uses `scf.yield` to supply new arguments for the
    "before" region, into which it transfers the control flow unconditionally.

    A simple "while" loop can be represented as follows.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> f32 {
      // "Before" region.
      // In a "while" loop, this region computes the condition.
      %condition = call @evaluate_condition(%arg1) : (f32) -> i1

      // Forward the argument (as result or "after" region argument).
      scf.condition(%condition) %arg1 : f32

    } do {
    ^bb0(%arg2: f32):
      // "After" region.
      // In a "while" loop, this region is the loop body.
      %next = call @payload(%arg2) : (f32) -> f32

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %next : f32
    }
    ```

    A simple "do-while" loop can be represented by reducing the "after" block
    to a simple forwarder.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> f32 {
      // "Before" region.
      // In a "do-while" loop, this region contains the loop body.
      %next = call @payload(%arg1) : (f32) -> f32

      // And also evaluates the condition.
      %condition = call @evaluate_condition(%arg1) : (f32) -> i1

      // Loop through the "after" region.
      scf.condition(%condition) %next : f32

    } do {
    ^bb0(%arg2: f32):
      // "After" region.
      // Forwards the values back to "before" region unmodified.
      scf.yield %arg2 : f32
    }
    ```

    Note that the types of region arguments need not to match with each other.
    The op expects the operand types to match with argument types of the
    "before" region; the result types to match with the trailing operand types
    of the terminator of the "before" region, and with the argument types of the
    "after" region. The following scheme can be used to share the results of
    some operations executed in the "before" region with the "after" region,
    avoiding the need to recompute them.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> i64 {
      // One can perform some computations, e.g., necessary to evaluate the
      // condition, in the "before" region and forward their results to the
      // "after" region.
      %shared = call @shared_compute(%arg1) : (f32) -> i64

      // Evaluate the condition.
      %condition = call @evaluate_condition(%arg1, %shared) : (f32, i64) -> i1

      // Forward the result of the shared computation to the "after" region.
      // The types must match the arguments of the "after" region as well as
      // those of the `scf.while` results.
      scf.condition(%condition) %shared : i64

    } do {
    ^bb0(%arg2: i64) {
      // Use the partial result to compute the rest of the payload in the
      // "after" region.
      %res = call @payload(%arg2) : (i64) -> f32

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %res : f32
    }
    ```

    The custom syntax for this operation is as follows.

    ```
    op ::= `scf.while` assignments `:` function-type region `do` region
           `attributes` attribute-dict
    initializer ::= /* empty */ | `(` assignment-list `)`
    assignment-list ::= assignment | assignment `,` assignment-list
    assignment ::= ssa-value `=` ssa-value
    ```
    """

    def __init__(
        self,
        inits: Sequence[mlir_python._mlir_python.Value] = [],
        result_types: Sequence[mlir_python._mlir_python.Type] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.while``: a generic 'while' loop.

        Args:
            inits: Operand ``inits`` (any type). Empty by default.
            result_types: Type of result ``results`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    @property
    def inits(self) -> list[mlir_python._mlir_python.Value]:
        """Operand ``inits``: any type."""

    @property
    def before(self) -> mlir_python._mlir_python.Region:
        """Region ``before``: region with 1 blocks."""

    @property
    def after(self) -> mlir_python._mlir_python.Region:
        """Region ``after``: region with 1 blocks."""

    OPERATION_NAME: str = "scf.while"

    @property
    def before_block(self) -> mlir_python._mlir_python.Block:
        """Computes the condition; ends in ``scf.ConditionOp``."""

    @property
    def after_block(self) -> mlir_python._mlir_python.Block:
        """The loop body; ends in ``scf.YieldOp``."""

class YieldOp(mlir_python._mlir_python.Operation):
    """
    ``scf.yield``: loop yield and termination operation.

    The `scf.yield` operation yields an SSA value from the SCF dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If `scf.yield` has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the `scf.yield` may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
    """

    def __init__(
        self,
        results: Sequence[mlir_python._mlir_python.Value] = [],
        *,
        location: mlir_python._mlir_python.Location | None = None,
        ip: mlir_python._mlir_python.InsertionPoint | None = None,
    ) -> None:
        """
        Create ``scf.yield``: loop yield and termination operation.

        Args:
            results: Operand ``results`` (any type). Empty by default.
            location: Defaults to the current ``Location``.
            ip: Defaults to the current ``InsertionPoint``; detached without one.
        """

    OPERATION_NAME: str = "scf.yield"
