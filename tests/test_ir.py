import gc

import pytest

import mlir_python as ir

ADD_MODULE = """
func.func @add(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  return %0 : i32
}
"""


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx:
        yield ctx


def build_add_function() -> ir.Module:
    """Builds ADD_MODULE with the generic operation API."""
    i32 = ir.IntegerType(32)
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        func = ir.Operation.create(
            "func.func",
            attributes={
                "sym_name": ir.StringAttr("add"),
                "function_type": ir.TypeAttr(ir.FunctionType([i32, i32], [i32])),
            },
            regions=1,
        )
    entry = func.regions[0].append_block([i32, i32])
    with ir.InsertionPoint(entry):
        a, b = entry.arguments
        total = ir.Operation.create("arith.addi", operands=[a, b])
        ir.Operation.create("func.return", operands=[total.result])
    return module


def test_parse_and_print_module() -> None:
    module = ir.Module.parse(ADD_MODULE)
    module.verify()
    assert "arith.addi %arg0, %arg1 : i32" in str(module)
    assert ir.Module.parse(str(module)).get_asm() == str(module)


def test_parse_error_is_mlir_error() -> None:
    with pytest.raises(ir.MLIRError, match="failed to parse module") as info:
        ir.Module.parse("func.func @f(")
    assert "error:" in str(info.value)


def test_build_matches_parsed() -> None:
    module = build_add_function()
    module.verify()
    assert str(module) == str(ir.Module.parse(ADD_MODULE))


def test_result_type_inference() -> None:
    module = build_add_function()
    add = module.body.operations[0].regions[0].blocks[0].operations[0]
    assert add.name == "arith.addi"
    assert add.result.type == ir.IntegerType(32)


def test_structure_navigation() -> None:
    module = ir.Module.parse(ADD_MODULE)
    func = module.body.operations[0]
    assert func.name == "func.func" and func.is_registered
    assert func.parent == module
    assert func.block == module.body
    entry = func.regions[0].entry_block
    assert entry is not None
    assert [str(arg.type) for arg in entry.arguments] == ["i32", "i32"]
    add, ret = entry.operations
    assert entry.terminator == ret
    assert list(entry) == [add, ret]
    assert add.operands[0] == entry.arguments[0]
    assert isinstance(add.result, ir.OpResult) and add.result.owner == add
    assert isinstance(entry.arguments[1], ir.BlockArgument)
    assert entry.arguments[1].owner == entry and entry.arguments[1].arg_number == 1
    assert [use.owner for use in add.result.uses] == [ret]
    assert str(add.result) == "%0"
    assert ret.parent == func


def test_attribute_map_includes_inherent_attributes() -> None:
    module = ir.Module.parse(
        "func.func @f(%a: i32) -> i1 {\n"
        "  %0 = arith.cmpi slt, %a, %a {tag} : i32\n"
        "  return %0 : i1\n}"
    )
    cmp = next(iter(module.body.operations[0].regions[0].blocks[0]))
    attrs = cmp.attributes
    assert set(attrs) == {"predicate", "tag"} and len(attrs) == 2
    assert attrs["predicate"] == ir.IntegerAttr(2, ir.IntegerType(64))
    attrs["predicate"] = ir.IntegerAttr(0, ir.IntegerType(64))
    assert "arith.cmpi eq" in str(module)
    del attrs["tag"]
    assert "tag" not in attrs
    with pytest.raises(KeyError):
        del attrs["tag"]
    assert attrs.get("tag") is None


def test_operands_and_uses_can_be_rewired() -> None:
    module = build_add_function()
    entry = module.body.operations[0].regions[0].blocks[0]
    add = entry.operations[0]
    a, b = entry.arguments
    add.operands[1] = a
    assert list(add.operands) == [a, a]
    assert not b.has_uses
    a.replace_all_uses_with(b)
    assert list(add.operands) == [b, b]
    module.verify()


def test_verify_reports_diagnostics() -> None:
    i32 = ir.IntegerType(32)
    op = ir.Operation.create("arith.addi", operands=[], results=[i32])
    with pytest.raises(ir.MLIRError, match="verification failed") as info:
        op.verify()
    assert "operand" in str(info.value)


def test_unregistered_operations_need_opt_in() -> None:
    with pytest.raises(ValueError, match="not registered"):
        ir.Operation.create("foo.bar")
    with pytest.raises(ValueError, match="dialect.op"):
        ir.Operation.create("nodot")
    ctx = ir.Context(allow_unregistered_dialects=True)
    op = ir.Operation.create(
        "foo.bar", results=[ir.IndexType(context=ctx)], context=ctx
    )
    assert not op.is_registered
    assert str(op) == '%0 = "foo.bar"() : () -> index\n'


def test_successors_only_on_terminators() -> None:
    module = ir.Module()
    block = ir.Operation.create("func.func", regions=1).regions[0].append_block()
    with pytest.raises(ValueError, match="terminator"):
        ir.Operation.create(
            "arith.constant",
            successors=[block],
            attributes={"value": ir.IntegerAttr(1)},
        )
    del module


def test_generic_and_debug_printing() -> None:
    module = ir.Module.parse(ADD_MODULE)
    func = module.body.operations[0]
    assert '"func.func"' in func.get_asm(generic=True)
    assert "loc(" in func.get_asm(debug_info=True)
    assert "arith.addi" not in func.get_asm(skip_regions=True)


def test_walk_orders_and_control() -> None:
    module = ir.Module.parse(ADD_MODULE)
    post: list[str] = []
    assert module.walk(lambda op: post.append(op.name)) is ir.WalkResult.ADVANCE
    assert post == ["arith.addi", "func.return", "func.func", "builtin.module"]
    pre: list[str] = []
    module.walk(lambda op: pre.append(op.name), ir.WalkOrder.PRE_ORDER)
    assert pre == ["builtin.module", "func.func", "arith.addi", "func.return"]

    def skip_funcs(op: ir.Operation) -> ir.WalkResult | None:
        pre_skip.append(op.name)
        return ir.WalkResult.SKIP if op.name == "func.func" else None

    pre_skip: list[str] = []
    module.walk(skip_funcs, ir.WalkOrder.PRE_ORDER)
    assert pre_skip == ["builtin.module", "func.func"]

    seen: list[str] = []

    def stop_at_add(op: ir.Operation) -> ir.WalkResult | None:
        seen.append(op.name)
        return ir.WalkResult.INTERRUPT if op.name == "arith.addi" else None

    assert module.walk(stop_at_add) is ir.WalkResult.INTERRUPT
    assert seen == ["arith.addi"]


def test_walk_propagates_exceptions() -> None:
    module = ir.Module.parse(ADD_MODULE)

    def boom(op: ir.Operation) -> None:
        raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        module.walk(boom)


def test_handles_keep_ir_alive() -> None:
    def inner_value() -> ir.OpResult:
        module = ir.Module.parse(ADD_MODULE)
        return module.body.operations[0].regions[0].blocks[0].operations[0].result

    value = inner_value()
    gc.collect()
    assert value.owner.name == "arith.addi"
    assert value.owner.parent is not None
    assert value.owner.parent.parent is not None


def test_inserted_operation_is_owned_by_destination() -> None:
    module = ir.Module()
    op = ir.Operation.create("arith.constant", attributes={"value": ir.IntegerAttr(1)})
    assert op.block is None
    module.body.append(op)
    assert op.block == module.body
    del module
    gc.collect()
    assert op.parent is not None and op.parent.name == "builtin.module"


def test_detach_clone_and_move() -> None:
    module = ir.Module.parse(ADD_MODULE)
    func = module.body.operations[0]
    copy = func.clone()
    assert copy.block is None and copy != func
    module.body.append(copy)
    assert len(module.body) == 2

    detached = copy.detach_from_parent()
    assert detached.block is None and len(module.body) == 1
    ir.InsertionPoint.before(func).insert(detached)
    assert module.body.operations == [detached, func]
    func.move_before(detached)
    assert module.body.operations == [func, detached]
    detached.move_after(func)
    assert module.body.operations == [func, detached]

    other = ir.Module()
    anchor = ir.Operation.create(
        "arith.constant",
        attributes={"value": ir.IntegerAttr(1)},
        ip=ir.InsertionPoint(other.body),
    )
    with pytest.raises(ValueError, match="different tree"):
        detached.move_before(anchor)
    with pytest.raises(ValueError, match="different tree"):
        other.body.append(detached)


def test_cannot_insert_into_itself() -> None:
    module = ir.Module()
    with pytest.raises(ValueError, match="into itself"):
        module.body.append(module)


def test_erase() -> None:
    module = ir.Module.parse(ADD_MODULE)
    entry = module.body.operations[0].regions[0].blocks[0]
    add, _ = entry.operations
    with pytest.raises(ValueError, match="still used"):
        add.erase()
    detached = ir.Operation.create(
        "arith.constant", attributes={"value": ir.IntegerAttr(1)}
    )
    detached.erase()
    with pytest.raises(ValueError, match="erased"):
        _ = detached.name


def test_insertion_points() -> None:
    module = ir.Module.parse(ADD_MODULE)
    entry = module.body.operations[0].regions[0].blocks[0]
    add, ret = entry.operations

    def const(value: int) -> ir.Operation:
        return ir.Operation.create(
            "arith.constant",
            attributes={"value": ir.IntegerAttr(value, ir.IntegerType(32))},
        )

    with ir.InsertionPoint.at_block_begin(entry) as ip:
        current = ir.InsertionPoint.current()
        assert current is not None and current.ref_operation == ip.ref_operation == add
        first = const(1)
    with ir.InsertionPoint.at_block_terminator(entry):
        before_ret = const(2)
    with ir.InsertionPoint.after(add):
        after_add = const(3)
    assert entry.operations == [first, add, after_add, before_ret, ret]
    assert ir.InsertionPoint.current() is None
    assert ir.InsertionPoint(entry).ref_operation == ret  # before the terminator
    assert ir.InsertionPoint.at_block_end(entry).ref_operation is None
    assert ir.InsertionPoint.before(ret).ref_operation == ret


def test_blocks_and_regions() -> None:
    func = ir.Operation.create("func.func", regions=1)
    region = func.regions[0]
    assert len(region) == 0 and region.entry_block is None
    block = region.append_block([ir.IndexType()])
    arg = block.add_argument(ir.F32Type())
    assert [a.arg_number for a in block.arguments] == [0, 1]
    assert arg.type == ir.F32Type()
    block.erase_argument(0)
    assert [a.type for a in block.arguments] == [ir.F32Type()]
    assert region.parent_op == func and block.region == region
    assert block.parent_op == func
    assert region.blocks == [block]


def test_value_type_can_be_changed() -> None:
    module = ir.Module.parse(ADD_MODULE)
    arg = module.body.operations[0].regions[0].blocks[0].arguments[0]
    arg.type = ir.IntegerType(64)
    with pytest.raises(ir.MLIRError):
        module.verify()


def test_module_is_an_operation_and_parse_operation() -> None:
    op = ir.Operation.parse("func.func private @decl(i32)")
    assert op.name == "func.func" and op.block is None
    module = ir.Module()
    assert isinstance(module, ir.Operation) and module.name == "builtin.module"
    module.body.append(op)
    assert "func.func private @decl(i32)" in str(module)
    nested = ir.Module.parse("module { module {} }").body.operations[0]
    assert isinstance(nested, ir.Module)
    with pytest.raises(ValueError, match="exactly one"):
        ir.Operation.parse("func.func private @a()\nfunc.func private @b()")
