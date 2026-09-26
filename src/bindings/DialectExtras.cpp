// Hand-written additions to the generated dialect classes.
//
// Generated constructors create operations exactly as described by ODS. The
// hooks here add what MLIR's C++ builders also set up, so a new operation is
// immediately ready to fill in (e.g. an `scf.for` body block with its
// induction variable), and the helpers name the blocks users fill in.
#include "DialectSupport.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>

namespace mlir_python {

namespace {

template <typename Fn>
void addMethod(nb::handle cls, const char *name, Fn &&fn, const char *doc) {
  nb::setattr(cls, name,
              nb::cpp_function(std::forward<Fn>(fn), nb::is_method(),
                               nb::scope(cls), nb::name(name), doc));
}

template <typename Fn>
void addProperty(nb::handle cls, const char *name, Fn &&fn, const char *doc) {
  // Stub generation reads the getter's docstring.
  nb::object getter = nb::cpp_function(std::forward<Fn>(fn), nb::is_method(),
                                       nb::scope(cls), nb::name(name), doc);
  nb::setattr(cls, name,
              nb::module_::import_("builtins")
                  .attr("property")(getter, nb::none(), nb::none(), doc));
}

//===--------------------------------------------------------------------===//
// LLVM dialect types
//===--------------------------------------------------------------------===//

struct PyLLVMPointerType : PyType {};
struct PyLLVMArrayType : PyType {};
struct PyLLVMStructType : PyType {};
struct PyLLVMFunctionType : PyType {};
struct PyLLVMVoidType : PyType {};

/// LLVM types can only be created once the dialect is loaded (MLIR aborts
/// otherwise), so every constructor calls this first.
const ContextHandle &withLLVMDialect(const ContextHandle &context) {
  context->context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  return context;
}

template <typename Py>
Py typeHandle(const ContextHandle &context, mlir::Type type) {
  Py result;
  result.context = context;
  result.type = type;
  return result;
}

/// Calls `getChecked`, turning verifier diagnostics into an MLIRError.
template <typename T, typename... Args>
T checkedType(const ContextHandle &context, const char *what, Args &&...args) {
  DiagnosticCapture capture(&context->context);
  T type = T::getChecked(mlir::detail::getDefaultDiagnosticEmitFn(&context->context),
                         std::forward<Args>(args)...);
  if (!type)
    capture.raise(std::string("invalid ") + what);
  return type;
}

std::vector<mlir::Type> nativeTypes(const ContextHandle &context,
                                    const std::vector<PyType> &types) {
  std::vector<mlir::Type> result;
  for (const PyType &type : types) {
    requireSameContext(context, type.context, "type");
    result.push_back(type.type);
  }
  return result;
}

std::vector<TypeObject> pythonTypes(const ContextHandle &context,
                                    llvm::ArrayRef<mlir::Type> types) {
  std::vector<TypeObject> result;
  for (mlir::Type type : types)
    result.push_back(wrapType(context, type));
  return result;
}

void bindLLVMTypes(nb::module_ llvmModule) {
  using namespace nb::literals;
  registerTypeClass<PyLLVMPointerType, mlir::LLVM::LLVMPointerType>();
  registerTypeClass<PyLLVMArrayType, mlir::LLVM::LLVMArrayType>();
  registerTypeClass<PyLLVMStructType, mlir::LLVM::LLVMStructType>();
  registerTypeClass<PyLLVMFunctionType, mlir::LLVM::LLVMFunctionType>();
  registerTypeClass<PyLLVMVoidType, mlir::LLVM::LLVMVoidType>();

  nb::class_<PyLLVMPointerType, PyType>(
      llvmModule, "PointerType",
      "An opaque pointer, ``!llvm.ptr`` (or ``!llvm.ptr<N>`` in address\n"
      "space N). Pointers carry no pointee type; loads, stores, and\n"
      "address computations name the type they access.")
      .def(nb::new_([](unsigned addressSpace, std::optional<PyContext> context) {
             ContextHandle state = withLLVMDialect(resolveContext(context));
             return typeHandle<PyLLVMPointerType>(
                 state, mlir::LLVM::LLVMPointerType::get(&state->context, addressSpace));
           }),
           "address_space"_a = 0, nb::kw_only(), "context"_a = nb::none(),
           "Create a pointer type in ``address_space`` (0 is the default).")
      .def_prop_ro(
          "address_space",
          [](const PyLLVMPointerType &self) {
            return llvm::cast<mlir::LLVM::LLVMPointerType>(self.type).getAddressSpace();
          },
          "The address space.");

  nb::class_<PyLLVMArrayType, PyType>(
      llvmModule, "ArrayType",
      "A fixed-size LLVM array, ``!llvm.array<N x T>``, e.g. the type of a\n"
      "string constant.")
      .def(nb::new_([](const PyType &element, uint64_t size) {
             withLLVMDialect(element.context);
             return typeHandle<PyLLVMArrayType>(
                 element.context,
                 checkedType<mlir::LLVM::LLVMArrayType>(element.context, "LLVM array type",
                                                        element.type, size));
           }),
           "element_type"_a, "size"_a, "Create ``!llvm.array<size x element_type>``.")
      .def_prop_ro(
          "element_type",
          [](const PyLLVMArrayType &self) {
            return wrapType(self.context,
                            llvm::cast<mlir::LLVM::LLVMArrayType>(self.type).getElementType());
          },
          "The element type.")
      .def_prop_ro(
          "size",
          [](const PyLLVMArrayType &self) {
            return llvm::cast<mlir::LLVM::LLVMArrayType>(self.type).getNumElements();
          },
          "The number of elements.");

  nb::class_<PyLLVMStructType, PyType>(
      llvmModule, "StructType",
      "An LLVM structure, ``!llvm.struct<(T1, T2, ...)>``.")
      .def(nb::new_([](const std::vector<PyType> &elements, bool packed,
                       std::optional<PyContext> context) {
             ContextHandle state = context ? context->state
                                   : !elements.empty() ? elements.front().context
                                                       : resolveContext(std::nullopt);
             withLLVMDialect(state);
             DiagnosticCapture capture(&state->context);
             auto type = mlir::LLVM::LLVMStructType::getLiteralChecked(
                 mlir::detail::getDefaultDiagnosticEmitFn(&state->context),
                 &state->context, nativeTypes(state, elements), packed);
             if (!type)
               capture.raise("invalid LLVM struct type");
             return typeHandle<PyLLVMStructType>(state, type);
           }),
           "elements"_a, "packed"_a = false, nb::kw_only(), "context"_a = nb::none(),
           "Create a literal struct of ``elements``; ``packed`` removes padding.")
      .def_prop_ro(
          "elements",
          [](const PyLLVMStructType &self) {
            return pythonTypes(self.context,
                               llvm::cast<mlir::LLVM::LLVMStructType>(self.type).getBody());
          },
          "The element types.")
      .def_prop_ro(
          "packed",
          [](const PyLLVMStructType &self) {
            return llvm::cast<mlir::LLVM::LLVMStructType>(self.type).isPacked();
          },
          "Whether the struct has no padding.");

  nb::class_<PyLLVMFunctionType, PyType>(
      llvmModule, "FunctionType",
      "An LLVM function signature, ``!llvm.func<R (A, B, ...)>``, as taken\n"
      "by ``LLVMFuncOp``. Unlike the builtin ``FunctionType`` it has one\n"
      "result (``VoidType`` for none) and may be variadic, like ``printf``.")
      .def(nb::new_([](const PyType &result, const std::vector<PyType> &inputs,
                       bool variadic) {
             const ContextHandle &state = withLLVMDialect(result.context);
             return typeHandle<PyLLVMFunctionType>(
                 state, checkedType<mlir::LLVM::LLVMFunctionType>(
                            state, "LLVM function type", result.type,
                            nativeTypes(state, inputs), variadic));
           }),
           "result"_a, "inputs"_a, nb::kw_only(), "variadic"_a = false,
           "Create ``!llvm.func<result (inputs...)>``.\n\n"
           "Args:\n"
           "    result: The result type; ``VoidType()`` for no result.\n"
           "    inputs: The parameter types.\n"
           "    variadic: Whether extra arguments may follow (``...``).")
      .def_prop_ro(
          "result",
          [](const PyLLVMFunctionType &self) {
            return wrapType(
                self.context,
                llvm::cast<mlir::LLVM::LLVMFunctionType>(self.type).getReturnType());
          },
          "The result type.")
      .def_prop_ro(
          "inputs",
          [](const PyLLVMFunctionType &self) {
            return pythonTypes(self.context,
                               llvm::cast<mlir::LLVM::LLVMFunctionType>(self.type).getParams());
          },
          "The parameter types.")
      .def_prop_ro(
          "variadic",
          [](const PyLLVMFunctionType &self) {
            return llvm::cast<mlir::LLVM::LLVMFunctionType>(self.type).isVarArg();
          },
          "Whether the function takes extra arguments.");

  nb::class_<PyLLVMVoidType, PyType>(
      llvmModule, "VoidType",
      "``!llvm.void``: the result type of an LLVM function returning nothing.")
      .def(nb::new_([](std::optional<PyContext> context) {
             ContextHandle state = withLLVMDialect(resolveContext(context));
             return typeHandle<PyLLVMVoidType>(
                 state, mlir::LLVM::LLVMVoidType::get(&state->context));
           }),
           nb::kw_only(), "context"_a = nb::none(), "Create ``!llvm.void``.");
}

mlir::Block *appendBlock(mlir::Region &region, mlir::TypeRange types,
                         mlir::Location location) {
  auto *block = new mlir::Block();
  block->addArguments(
      types, llvm::SmallVector<mlir::Location>(types.size(), location));
  region.push_back(block);
  return block;
}

PyBlock blockHandle(const PyOperation &self, mlir::Block *block) {
  return PyBlock{self.tree, block};
}

//===--------------------------------------------------------------------===//
// Post-create hooks
//===--------------------------------------------------------------------===//

/// Body block with the induction variable and one argument per init value;
/// without init values, also the implicit `scf.yield`.
void completeFor(mlir::Operation *op) {
  auto loop = llvm::cast<mlir::scf::ForOp>(op);
  if (!loop.getRegion().empty())
    return;
  llvm::SmallVector<mlir::Type> types{loop.getLowerBound().getType()};
  llvm::append_range(types, loop.getInitArgs().getTypes());
  appendBlock(loop.getRegion(), types, op->getLoc());
  if (loop.getInitArgs().empty()) {
    mlir::OpBuilder builder(op->getContext());
    mlir::scf::ForOp::ensureTerminator(loop.getRegion(), builder, op->getLoc());
  }
}

/// Then block (and else block when the op has results, which requires one);
/// without results, the implicit `scf.yield`.
void completeIf(mlir::Operation *op) {
  auto ifOp = llvm::cast<mlir::scf::IfOp>(op);
  mlir::OpBuilder builder(op->getContext());
  if (ifOp.getThenRegion().empty()) {
    appendBlock(ifOp.getThenRegion(), {}, op->getLoc());
    if (op->getNumResults() == 0)
      mlir::scf::IfOp::ensureTerminator(ifOp.getThenRegion(), builder,
                                        op->getLoc());
  }
  if (op->getNumResults() > 0 && ifOp.getElseRegion().empty())
    appendBlock(ifOp.getElseRegion(), {}, op->getLoc());
}

/// "before" block taking the init values and "after" block taking the
/// result values.
void completeWhile(mlir::Operation *op) {
  auto loop = llvm::cast<mlir::scf::WhileOp>(op);
  if (loop.getBefore().empty())
    appendBlock(loop.getBefore(), loop.getInits().getTypes(), op->getLoc());
  if (loop.getAfter().empty())
    appendBlock(loop.getAfter(), op->getResultTypes(), op->getLoc());
}

} // namespace

void bindDialectExtras(nb::module_ &m) {
  bindLLVMTypes(nb::borrow<nb::module_>(m.attr("llvm")));
  registerPostCreateHook(mlir::TypeID::get<mlir::scf::ForOp>(), &completeFor);
  registerPostCreateHook(mlir::TypeID::get<mlir::scf::IfOp>(), &completeIf);
  registerPostCreateHook(mlir::TypeID::get<mlir::scf::WhileOp>(),
                         &completeWhile);

  nb::handle funcOp = m.attr("func").attr("FuncOp");
  addMethod(
      funcOp, "add_entry_block",
      [](const PyOperation &self) {
        auto fn = llvm::cast<mlir::func::FuncOp>(checkedOp(self));
        if (!fn.getBody().empty())
          throw std::invalid_argument("function already has a body");
        return blockHandle(self, fn.addEntryBlock());
      },
      "Give this declaration a body: an entry block whose arguments match\n"
      "``function_type.inputs``. Returns the block.\n\n"
      "Raises:\n    ValueError: If the function already has a body.");
  addProperty(
      funcOp, "arguments",
      [](const PyOperation &self) {
        auto fn = llvm::cast<mlir::func::FuncOp>(checkedOp(self));
        if (fn.getBody().empty())
          throw std::invalid_argument(
              "function is a declaration; call add_entry_block() first");
        std::vector<PyBlockArgument> args;
        for (mlir::BlockArgument arg : fn.getArguments())
          args.push_back(PyBlockArgument{{self.tree, arg}});
        return args;
      },
      "The entry block's arguments.\n\nRaises:\n"
      "    ValueError: If the function has no body.");

  nb::handle forOp = m.attr("scf").attr("ForOp");
  addProperty(
      forOp, "body",
      [](const PyOperation &self) {
        return blockHandle(
            self, llvm::cast<mlir::scf::ForOp>(checkedOp(self)).getBody());
      },
      "The loop body. Without init values it already ends in ``scf.yield``,\n"
      "and ``InsertionPoint(body)`` inserts before it.");
  addProperty(
      forOp, "induction_variable",
      [](const PyOperation &self) {
        auto loop = llvm::cast<mlir::scf::ForOp>(checkedOp(self));
        return PyBlockArgument{{self.tree, loop.getInductionVar()}};
      },
      "The loop counter, the body's first argument.");
  addProperty(
      forOp, "inner_iter_args",
      [](const PyOperation &self) {
        auto loop = llvm::cast<mlir::scf::ForOp>(checkedOp(self));
        std::vector<PyBlockArgument> args;
        for (mlir::BlockArgument arg : loop.getRegionIterArgs())
          args.push_back(PyBlockArgument{{self.tree, arg}});
        return args;
      },
      "Body arguments carrying the loop-carried values; yield their next\n"
      "values with ``scf.YieldOp``.");

  nb::handle ifOp = m.attr("scf").attr("IfOp");
  addProperty(
      ifOp, "then_block",
      [](const PyOperation &self) {
        return blockHandle(
            self, llvm::cast<mlir::scf::IfOp>(checkedOp(self)).thenBlock());
      },
      "The block run when the condition holds.");
  addProperty(
      ifOp, "else_block",
      [](const PyOperation &self) -> std::optional<PyBlock> {
        mlir::Region &region =
            llvm::cast<mlir::scf::IfOp>(checkedOp(self)).getElseRegion();
        if (region.empty())
          return std::nullopt;
        return blockHandle(self, &region.front());
      },
      "The block run otherwise, or ``None`` (see ``add_else_block``).");
  addMethod(
      ifOp, "add_else_block",
      [](const PyOperation &self) {
        mlir::Operation *op = checkedOp(self);
        mlir::Region &region = llvm::cast<mlir::scf::IfOp>(op).getElseRegion();
        if (!region.empty())
          throw std::invalid_argument("the if already has an else block");
        mlir::Block *block = appendBlock(region, {}, op->getLoc());
        if (op->getNumResults() == 0) {
          mlir::OpBuilder builder(op->getContext());
          mlir::scf::IfOp::ensureTerminator(region, builder, op->getLoc());
        }
        return blockHandle(self, block);
      },
      "Add the else block (ending in ``scf.yield`` when the if has no\n"
      "results) and return it. An if with results already has one.\n\n"
      "Raises:\n    ValueError: If the else block exists.");

  nb::handle whileOp = m.attr("scf").attr("WhileOp");
  addProperty(
      whileOp, "before_block",
      [](const PyOperation &self) {
        return blockHandle(
            self, llvm::cast<mlir::scf::WhileOp>(checkedOp(self)).getBeforeBody());
      },
      "Computes the condition; ends in ``scf.ConditionOp``.");
  addProperty(
      whileOp, "after_block",
      [](const PyOperation &self) {
        return blockHandle(
            self, llvm::cast<mlir::scf::WhileOp>(checkedOp(self)).getAfterBody());
      },
      "The loop body; ends in ``scf.YieldOp``.");
}

} // namespace mlir_python
