"""Dialects defined at runtime (IRDL) through mlir_python.irdl."""

import pytest

import mlir_python as ir
from mlir_python import codegen, irdl
from mlir_python.dialects import arith, func

demo = irdl.Dialect("demo")

Mutex = demo.type("mutex")
Box = demo.type("box", element=irdl.Any())


@demo.operation
class Guard(irdl.Op):
    """Locks ``mutex``; ``value`` is readable while it is held."""

    mutex = irdl.Operand(irdl.BaseOf(Mutex))
    value = irdl.Result(irdl.Any())


@demo.operation
class Pack(irdl.Op):
    """Packs any number of values, with an optional tag, into a box."""

    items = irdl.VariadicOperand(irdl.BaseOf(ir.IntegerType))
    tag = irdl.OptionalOperand()
    box = irdl.Result(irdl.Parametric(Box, irdl.Any()))
    label = irdl.Attr(irdl.BaseOf(ir.StringAttr))


@demo.operation(name="double")
class Twice(irdl.Op):
    """Doubles an i32; lowered to arith.addi below."""

    input = irdl.Operand(irdl.Is(lambda: ir.IntegerType(32)))
    output = irdl.Result(irdl.Is(lambda: ir.IntegerType(32)))


@demo.operation
class Scope(irdl.Op):
    body = irdl.Region()


@pytest.fixture(autouse=True)
def context():
    with ir.Context() as ctx, ir.Location.unknown():
        yield ctx


def test_definition_is_irdl() -> None:
    text = demo.definition()
    assert "irdl.dialect @demo" in text
    assert "irdl.operation @guard" in text
    assert 'irdl.base "!builtin.integer"' in text
    assert "irdl.parametric @demo::@box" in text
    assert Twice.NAME == "demo.double"
    assert Pack.NAME == "demo.pack"


def test_operations_are_typed_views() -> None:
    i32 = ir.IntegerType(32)
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        fn = func.FuncOp("f", ir.FunctionType([Mutex(), i32, i32], []))
    with ir.InsertionPoint(fn.add_entry_block()):
        mutex, a, b = fn.arguments
        guard = Guard.create(mutex, value=i32)
        pack = Pack.create([a, b], None, box=Box(i32), label=ir.StringAttr("pair"))
        tagged = Pack.create([a], guard.value, box=Box(i32), label=ir.StringAttr("t"))
        func.ReturnOp()
    module.verify()
    assert guard.mutex == mutex
    assert guard.value.type == i32
    assert pack.items == [a, b]
    assert pack.tag is None
    assert tagged.items == [a]
    assert tagged.tag == guard.value
    assert str(pack.label) == '"pair"'
    assert Box.matches(pack.box.type) and not Mutex.matches(pack.box.type)
    assert [g.operation for g in Guard.all(module)] == [guard.operation]
    assert len(Pack.all(module)) == 2
    assert Guard.matches(guard.operation) and not Pack.matches(guard.operation)


def test_constraints_are_verified() -> None:
    i32 = ir.IntegerType(32)
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        fn = func.FuncOp("f", ir.FunctionType([i32], []))
    with ir.InsertionPoint(fn.add_entry_block()):
        (x,) = fn.arguments
        Guard.create(x, value=i32)  # x is not a !demo.mutex
        func.ReturnOp()
    with pytest.raises(ir.MLIRError, match="demo.guard"):
        module.verify()


def test_create_checks_its_arguments() -> None:
    with pytest.raises(TypeError, match="takes 1 operands"):
        Guard.create()
    with pytest.raises(TypeError, match="operand 'items' is variadic"):
        Pack.create(None, None)  # type: ignore[arg-type]
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        fn = func.FuncOp("f", ir.FunctionType([Mutex()], []))
    with ir.InsertionPoint(fn.add_entry_block()):
        (mutex,) = fn.arguments
        with pytest.raises(TypeError, match="needs the type of 'value'"):
            Guard.create(mutex)
        with pytest.raises(TypeError, match="unknown fields: colour"):
            Guard.create(mutex, value=Mutex(), colour=Mutex())


def test_lowering_a_runtime_op_to_standard_ops() -> None:
    i32 = ir.IntegerType(32)
    module = ir.Module()
    with ir.InsertionPoint(module.body):
        fn = func.FuncOp("quadruple", ir.FunctionType([i32], [i32]))
    with ir.InsertionPoint(fn.add_entry_block()):
        (x,) = fn.arguments
        once = Twice.create(x)  # the result type comes from Is(i32)
        twice = Twice.create(once.output)
        func.ReturnOp([twice.output])
    for op in Twice.all(module):
        with ir.InsertionPoint.before(op.operation):
            doubled = arith.AddIOp(op.input, op.input).result
        op.replace_with([doubled])
    assert "demo.double" not in str(module)
    compiled = codegen.compile(module)
    quadruple = next(o for o in module.body.operations if isinstance(o, func.FuncOp))
    assert compiled.function(quadruple)(5) == 20


def test_regions_and_types() -> None:
    scope = Scope.create()
    assert len(scope.body.blocks) == 0
    assert str(Box(ir.IntegerType(8))) == "!demo.box<i8>"
    with pytest.raises(TypeError, match="takes 1 parameters"):
        Box()


def test_declarations_close_once_used() -> None:
    late = irdl.Dialect("late")

    @late.operation
    class First(irdl.Op):
        pass

    late.load()
    with pytest.raises(RuntimeError, match="declare 'second' before"):

        @late.operation
        class Second(irdl.Op):
            pass

        del Second
    assert First.NAME == "late.first"
