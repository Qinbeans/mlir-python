import pytest

import mlir_python as ir
from mlir_python.dialects import arith, cf, func, memref, scf


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def test_build_function_with_typed_ops() -> None:
    i32 = ir.IntegerType(32)
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        add = func.FuncOp("add", ir.FunctionType([i32, i32], [i32]))
    entry = add.body.append_block([i32, i32])
    with ir.InsertionPoint(entry):
        a, b = entry.arguments
        total = arith.AddIOp(a, b)
        func.ReturnOp([total.result])
    module.verify()
    assert str(module) == str(
        ir.Module.parse(
            "func.func @add(%a: i32, %b: i32) -> i32 {\n"
            "  %0 = arith.addi %a, %b : i32\n  return %0 : i32\n}"
        )
    )


def test_navigation_returns_typed_classes() -> None:
    module = ir.Module.parse(
        "func.func @f(%a: i32) -> i1 {\n"
        "  %0 = arith.cmpi slt, %a, %a : i32\n  return %0 : i1\n}"
    )
    fn = module.body.operations[0]
    assert isinstance(fn, func.FuncOp)
    assert fn.sym_name == "f"
    assert fn.function_type == ir.FunctionType(
        [ir.IntegerType(32)], [ir.IntegerType(1)]
    )
    cmp, ret = fn.body.blocks[0].operations
    assert isinstance(cmp, arith.CmpIOp) and isinstance(ret, func.ReturnOp)
    assert cmp.predicate is arith.CmpIPredicate.SLT
    assert cmp.lhs == fn.body.blocks[0].arguments[0]
    assert isinstance(cmp.parent, func.FuncOp)
    assert isinstance(fn.parent, ir.Module)
    assert isinstance(
        ir.Operation.create("arith.constant", attributes={"value": ir.IntegerAttr(1)}),
        arith.ConstantOp,
    )


def test_result_types_are_inferred() -> None:
    i32 = ir.IntegerType(32)
    c = arith.ConstantOp(ir.IntegerAttr(7, i32))
    assert c.result.type == i32
    assert c.value == ir.IntegerAttr(7, i32)
    cmp = arith.CmpIOp(arith.CmpIPredicate.EQ, c.result, c.result)
    assert cmp.result.type == ir.IntegerType(1)


def test_enum_attributes_read_and_write() -> None:
    c = arith.ConstantOp(ir.IntegerAttr(1, ir.IntegerType(8)))
    cmp = arith.CmpIOp(arith.CmpIPredicate.ULT, c.result, c.result)
    assert "arith.cmpi ult" in str(cmp)
    cmp.predicate = arith.CmpIPredicate.NE
    assert cmp.predicate is arith.CmpIPredicate.NE
    assert "arith.cmpi ne" in str(cmp)


def test_defaulted_flag_enum_attribute() -> None:
    x = arith.ConstantOp(ir.FloatAttr(1.0, ir.F32Type())).result
    plain = arith.AddFOp(x, x)
    assert plain.fastmath == arith.FastMathFlags.NONE
    assert "fastmath" not in str(plain)
    fast = arith.AddFOp(
        x, x, fastmath=arith.FastMathFlags.NNAN | arith.FastMathFlags.NINF
    )
    assert "fastmath<nnan,ninf>" in str(fast)
    assert fast.fastmath & arith.FastMathFlags.NNAN


def test_optional_and_unit_attributes() -> None:
    fn = func.FuncOp(
        "decl", ir.FunctionType([], []), sym_visibility="private", no_inline=True
    )
    assert fn.sym_visibility == "private" and fn.no_inline
    assert str(fn).startswith("func.func private @decl() attributes {no_inline}")
    fn.sym_visibility = None
    fn.no_inline = False
    assert fn.sym_visibility is None and not fn.no_inline
    assert "sym_visibility" not in fn.attributes and "no_inline" not in fn.attributes


def test_operand_segments_and_named_results() -> None:
    index = ir.IndexType()
    n = arith.ConstantOp(ir.IntegerAttr(4, index)).result
    buffer = memref.AllocOp(ir.MemRefType([None, 8], ir.F32Type()), [n], alignment=64)
    buffer.verify()
    assert buffer.dynamic_sizes == [n] and buffer.symbol_operands == []
    assert buffer.alignment == 64
    assert buffer.memref.type == ir.MemRefType([None, 8], ir.F32Type())
    assert "alignment = 64" in str(buffer)


def test_successors_and_variadic_operands() -> None:
    i1, i32 = ir.IntegerType(1), ir.IntegerType(32)
    fn = func.FuncOp("branch", ir.FunctionType([i1, i32], []))
    entry = fn.body.append_block([i1, i32])
    then_block = fn.body.append_block([i32])
    else_block = fn.body.append_block()
    cond, value = entry.arguments
    with ir.InsertionPoint(entry):
        br = cf.CondBranchOp(cond, then_block, else_block, true_dest_operands=[value])
    for block in (then_block, else_block):
        with ir.InsertionPoint(block):
            func.ReturnOp()
    fn.verify()
    assert br.true_dest == then_block and br.false_dest == else_block
    assert br.true_dest_operands == [value] and br.false_dest_operands == []


def test_function_entry_block_matches_signature() -> None:
    i32 = ir.IntegerType(32)
    fn = func.FuncOp("id", ir.FunctionType([i32], [i32]))
    with pytest.raises(ValueError, match="declaration"):
        _ = fn.arguments
    entry = fn.add_entry_block()
    assert [a.type for a in fn.arguments] == [i32] and entry.arguments == fn.arguments
    with ir.InsertionPoint(entry):
        func.ReturnOp(fn.arguments)
    fn.verify()
    with pytest.raises(ValueError, match="already has a body"):
        fn.add_entry_block()


def test_for_loop_without_carried_values() -> None:
    index = ir.IndexType()
    lb, ub, step = (
        arith.ConstantOp(ir.IntegerAttr(v, index)).result for v in (0, 8, 1)
    )
    loop = scf.ForOp(lb, ub, step)
    assert loop.results == [] and loop.induction_variable.type == index
    terminator = loop.body.terminator
    assert isinstance(terminator, scf.YieldOp)  # implicit, like MLIR's builder
    with ir.InsertionPoint(loop.body):  # inserts before the yield
        arith.AddIOp(loop.induction_variable, loop.induction_variable)
    assert [type(op) for op in loop.body] == [arith.AddIOp, scf.YieldOp]


def test_for_loop_with_carried_values() -> None:
    index, f32 = ir.IndexType(), ir.F32Type()
    lb, ub, step = (
        arith.ConstantOp(ir.IntegerAttr(v, index)).result for v in (0, 8, 1)
    )
    init = arith.ConstantOp(ir.FloatAttr(0.0, f32)).result
    loop = scf.ForOp(lb, ub, step, init_args=[init])
    assert [r.type for r in loop.results] == [f32]  # derived from init_args
    (acc,) = loop.inner_iter_args
    with ir.InsertionPoint(loop.body):
        scf.YieldOp([arith.AddFOp(acc, acc).result])
    loop.verify()


def test_if_with_and_without_results() -> None:
    i1, i32 = ir.IntegerType(1), ir.IntegerType(32)
    cond = arith.ConstantOp(ir.BoolAttr(True)).result
    plain = scf.IfOp(cond)
    assert isinstance(plain.then_block.terminator, scf.YieldOp)
    assert plain.else_block is None
    else_block = plain.add_else_block()
    assert plain.else_block == else_block and isinstance(
        else_block.terminator, scf.YieldOp
    )
    plain.verify()

    choose = scf.IfOp(cond, result_types=[i32])
    assert choose.else_block is not None and choose.result.type == i32
    for block, value in ((choose.then_block, 1), (choose.else_block, 2)):
        with ir.InsertionPoint(block):
            scf.YieldOp([arith.ConstantOp(ir.IntegerAttr(value, i32)).result])
    choose.verify()
    assert cond.type == i1


def test_while_blocks_match_types() -> None:
    i32 = ir.IntegerType(32)
    init = arith.ConstantOp(ir.IntegerAttr(0, i32)).result
    loop = scf.WhileOp([init], result_types=[i32])
    assert [a.type for a in loop.before_block.arguments] == [i32]
    assert [a.type for a in loop.after_block.arguments] == [i32]
    (x,) = loop.before_block.arguments
    with ir.InsertionPoint(loop.before_block):
        more = arith.CmpIOp(arith.CmpIPredicate.SLT, x, x)
        scf.ConditionOp(more.result, [x])
    with ir.InsertionPoint(loop.after_block):
        scf.YieldOp(loop.after_block.arguments)
    loop.verify()


def test_complete_kernel() -> None:
    f32, index = ir.F32Type(), ir.IndexType()
    buffer_type = ir.MemRefType([16], f32)
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        kernel = func.FuncOp("sum", ir.FunctionType([buffer_type], [f32]))
    with ir.InsertionPoint(kernel.add_entry_block()):
        (buffer,) = kernel.arguments
        lb, ub, step = (
            arith.ConstantOp(ir.IntegerAttr(v, index)).result for v in (0, 16, 1)
        )
        zero = arith.ConstantOp(ir.FloatAttr(0.0, f32)).result
        loop = scf.ForOp(lb, ub, step, init_args=[zero])
        with ir.InsertionPoint(loop.body):
            element = memref.LoadOp(buffer, [loop.induction_variable])
            total = arith.AddFOp(loop.inner_iter_args[0], element.result)
            scf.YieldOp([total.result])
        func.ReturnOp(loop.results)
    module.verify()
    assert (
        "scf.for %arg1 = %c0 to %c16 step %c1 iter_args(%arg2 = %cst) -> (f32)"
        in str(module)
    )


def test_detached_definitions_outlive_their_users() -> None:
    i32 = ir.IntegerType(32)

    def compare() -> arith.CmpIOp:
        c = arith.ConstantOp(ir.IntegerAttr(7, i32))  # detached, only referenced by cmp
        return arith.CmpIOp(arith.CmpIPredicate.EQ, c.result, c.result)

    import gc

    cmp = compare()
    gc.collect()
    definer = cmp.lhs.owner
    assert isinstance(definer, arith.ConstantOp) and definer.value == ir.IntegerAttr(
        7, i32
    )
    module = ir.Module()
    module.body.append(definer)
    module.body.append(cmp)
    module.verify()


def test_wrong_argument_types_raise_type_error() -> None:
    i32 = ir.IntegerType(32)
    with pytest.raises(TypeError):
        func.FuncOp("f", i32)  # type: ignore[arg-type]
    one = arith.ConstantOp(ir.IntegerAttr(1, i32)).result
    with pytest.raises(TypeError):
        arith.CmpIOp("slt", one, one)  # type: ignore[arg-type]  # enums, not strings


def test_constructor_documents_its_parameters() -> None:
    doc = arith.CmpIOp.__init__.__doc__ or ""
    assert "predicate: Attribute ``predicate`` (a member of ``CmpIPredicate``)" in doc
    assert "lhs: Operand ``lhs`` (signless-integer-like)" in doc
    assert "integer comparison operation" in (arith.CmpIOp.__doc__ or "")


def test_variadic_of_variadic_operands() -> None:
    from mlir_python.dialects import llvm

    i32 = ir.IntegerType(32)
    printf_type = llvm.FunctionType(i32, [llvm.PointerType()], variadic=True)
    with ir.InsertionPoint(ir.Module().body):
        llvm.LLVMFuncOp("printf", printf_type)
        text = llvm.string_constant("fmt", "%d")
        one = arith.ConstantOp(ir.IntegerAttr(1, i32)).result
        call = llvm.CallOp(
            [llvm.address_of(text), one],
            var_callee_type=printf_type,
            callee="printf",
            result_type=i32,
        )
    assert call.callee_operands[1] == one and call.op_bundle_operands == []
    assert call.result.type == i32

    fn = func.FuncOp("choose", ir.FunctionType([i32], []))
    entry = fn.add_entry_block()
    default, first, second = (fn.body.append_block() for _ in range(3))
    with ir.InsertionPoint(entry):
        switch = cf.SwitchOp(
            entry.arguments[0],
            default,
            [first, second],
            case_values=ir.DenseIntElementsAttr([1, 2], ir.VectorType([2], i32)),
        )
    for block in (default, first, second):
        with ir.InsertionPoint(block):
            func.ReturnOp()
    fn.verify()
    assert switch.case_operands == [[], []]


def test_switch_case_operands_must_match_destinations() -> None:
    i32 = ir.IntegerType(32)
    fn = func.FuncOp("choose", ir.FunctionType([i32], []))
    entry = fn.add_entry_block()
    default, first = fn.body.append_block(), fn.body.append_block()
    with (
        ir.InsertionPoint(entry),
        pytest.raises(ValueError, match="one group of operands per entry"),
    ):
        cf.SwitchOp(entry.arguments[0], default, [first], case_operands=[[], []])
