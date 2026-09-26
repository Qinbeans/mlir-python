// Bindings for the IR structure: values, operations, blocks, regions,
// insertion points, and modules.
#include "Core.h"

#include <map>

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Parser/Parser.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace mlir_python {

using namespace nb::literals;

namespace {

struct PyOpOperand {
  PyOpOperand(OpTreeHandle tree, mlir::OpOperand *operand)
      : tree(std::move(tree)), operand(operand), version(versionOf(this->tree)) {}
  OpTreeHandle tree;
  mlir::OpOperand *operand;
  uint64_t version;
};

mlir::OpOperand *get(const PyOpOperand &handle) {
  handle.tree->checkCurrent(handle.version, nullptr);
  return handle.operand;
}

/// Live, list-like view of an operation's operands.
struct PyOperandList {
  PyOperation owner;
};

/// Live, dict-like view of an operation's attributes.
struct PyAttributeMap {
  PyOperation owner;
};

enum class WalkOrder { PreOrder, PostOrder };
enum class WalkResult { Advance, Interrupt, Skip };

using OperationOrBlock = nb::typed<nb::object, std::variant<PyOperation, PyBlock>>;
using WalkCallback =
    nb::typed<nb::callable, std::optional<WalkResult>(PyOperation)>;

//===--------------------------------------------------------------------===//
// Validity and ownership helpers
//===--------------------------------------------------------------------===//

const ContextHandle &contextOf(const OpTreeHandle &tree) {
  return tree->context();
}

OperationObject wrapOp(const OpTreeHandle &tree, mlir::Operation *op) {
  return wrapOperation(tree, op);
}

OptionalOperationObject wrapOptionalOp(const OpTreeHandle &tree,
                                       mlir::Operation *op) {
  if (!op)
    return nb::none();
  return wrapOp(tree, op);
}

llvm::DenseMap<mlir::TypeID, OperationFactory> &operationClasses() {
  static llvm::DenseMap<mlir::TypeID, OperationFactory> classes;
  return classes;
}

std::optional<PyBlock> wrapOptionalBlock(const OpTreeHandle &tree,
                                         mlir::Block *block) {
  if (!block)
    return std::nullopt;
  return PyBlock{tree, block};
}

/// Whether a value defined in `op`'s subtree is used outside it.
bool hasUsesOutside(mlir::Operation *op) {
  bool found = false;
  op->walk([&](mlir::Operation *nested) {
    for (mlir::OpResult result : nested->getResults())
      for (mlir::OpOperand &use : result.getUses())
        if (!op->isAncestor(use.getOwner()))
          found = true;
    return found ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  });
  return found;
}

/// Returns the tree that owns detached operation `op`.
OpTreeHandle rootTreeOf(const ContextHandle &context, mlir::Operation *op) {
  auto it = context->trees.find(op);
  if (it != context->trees.end())
    if (OpTreeHandle tree = it->second.lock())
      return tree;
  // A detached operation we have never seen can only come from the caller's
  // own handle, which must then own it.
  throw std::logic_error("detached operation has no owning tree");
}

/// Moves a detached operation into `block` before `before` (or at the end).
void insertDetached(const PyOperation &op, const OpTreeHandle &destination,
                    mlir::Block *block, mlir::Operation *before) {
  mlir::Operation *native = get(op);
  requireSameContext(contextOf(destination), contextOf(op.tree), "operation");
  if (native->getBlock())
    throw std::invalid_argument(
        "operation is already in a block; call detach_from_parent() first or "
        "use move_before()/move_after()");
  if (mlir::Operation *parent = block->getParentOp())
    if (native == parent || native->isProperAncestor(parent))
      throw std::invalid_argument("cannot insert an operation into itself");
  OpTreeHandle tree = rootTreeOf(contextOf(destination), native);
  OpTreeHandle owner = destination->owner(destination);
  if (owner == tree)
    throw std::invalid_argument("cannot insert an operation into itself");
  tree->attachTo(owner);
  if (before)
    block->getOperations().insert(mlir::Block::iterator(before), native);
  else
    block->push_back(native);
  trackCrossTreeUses(contextOf(destination), native);
}

/// Inserts `op` at `point`, detached or not, keeping ownership consistent.
void insertAt(const PyOperation &op, const PyInsertionPoint &point) {
  checkCurrent(point);
  mlir::Operation *native = get(op);
  if (point.before && point.before->getBlock() != point.block)
    throw std::invalid_argument("insertion point is no longer valid");
  if (!native->getBlock()) {
    insertDetached(op, point.tree, point.block, point.before);
    return;
  }
  // Attached operations may move only within the same tree, so that every
  // existing handle keeps the right owner alive.
  if (op.tree->owner(op.tree) != point.tree->owner(point.tree))
    throw std::invalid_argument(
        "operation belongs to a different tree; call detach_from_parent() "
        "first");
  if (mlir::Operation *parent = point.block->getParentOp())
    if (native == parent || native->isProperAncestor(parent))
      throw std::invalid_argument("cannot move an operation into itself");
  if (point.before)
    native->moveBefore(point.before);
  else
    native->moveBefore(point.block, point.block->end());
}

/// The operation's tree for inserting at `point`, or a new tree if detached.
PyOperation place(const ContextHandle &context, mlir::Operation *op,
                  const std::optional<PyInsertionPoint> &point) {
  if (point) {
    try {
      checkCurrent(*point);
    } catch (...) {
      op->destroy();
      throw;
    }
  }
  if (!point)
    return PyOperation{OpTree::adopt(context, op), op};
  if (point->before && point->before->getBlock() != point->block) {
    op->destroy();
    throw std::invalid_argument("insertion point is no longer valid");
  }
  if (point->before)
    point->block->getOperations().insert(mlir::Block::iterator(point->before),
                                         op);
  else
    point->block->push_back(op);
  return PyOperation{point->tree->owner(point->tree), op};
}

//===--------------------------------------------------------------------===//
// Operation creation
//===--------------------------------------------------------------------===//

mlir::OperationName lookupOperationName(const ContextHandle &context,
                                        const std::string &name) {
  mlir::MLIRContext *native = &context->context;
  auto dot = name.find('.');
  if (dot == std::string::npos || dot == 0 || dot + 1 == name.size())
    throw std::invalid_argument("operation name '" + name +
                                "' must look like 'dialect.op'");
  bool dialectLoaded = native->getOrLoadDialect(name.substr(0, dot)) != nullptr;
  mlir::OperationName opName(name, native);
  if (!opName.isRegistered() && !native->allowsUnregisteredDialects())
    throw std::invalid_argument(
        "operation '" + name + "' is not registered" +
        (dialectLoaded ? "" : " (its dialect is not available)") +
        "; construct the Context with allow_unregistered_dialects=True to "
        "create it anyway");
  return opName;
}

void inferResults(mlir::OperationState &state,
                      const mlir::RegisteredOperationName &info) {
  auto *inference = info.getInterface<mlir::InferTypeOpInterface>();
  mlir::MLIRContext *context = state.getContext();
  mlir::DictionaryAttr attributes = state.attributes.getDictionary(context);

  // Properties must be materialized from the attributes so the inference
  // hook sees inherent attributes such as predicates.
  std::unique_ptr<uint64_t[]> storage;
  mlir::OpaqueProperties properties(nullptr);
  int size = info.getOpPropertyByteSize();
  if (size > 0) {
    storage.reset(new uint64_t[(size + 7) / 8]);
    properties = mlir::OpaqueProperties(storage.get());
    info.initOpProperties(properties, mlir::OpaqueProperties(nullptr));
  }
  auto destroy = llvm::make_scope_exit([&] {
    if (size > 0)
      info.destroyOpProperties(properties);
  });

  DiagnosticCapture capture(context);
  if (size > 0 && !attributes.empty()) {
    auto emitError = [&]() {
      return mlir::emitError(state.location)
             << "invalid attributes for " << state.name << ": ";
    };
    if (mlir::failed(info.setOpPropertiesFromAttribute(state.name, properties,
                                                       attributes, emitError)))
      capture.raise("could not infer result types of '" +
                    state.name.getStringRef().str() + "'");
  }
  if (mlir::failed(inference->inferReturnTypes(
          context, state.location, state.operands, attributes, properties,
          state.regions, state.types)))
    capture.raise("could not infer result types of '" +
                  state.name.getStringRef().str() +
                  "'; pass results= explicitly");
}

OperationObject createGenericOperation(
    const std::string &name, const std::optional<std::vector<PyType>> &results,
    const std::vector<PyValue> &operands,
    const std::optional<std::map<std::string, PyAttribute>> &attributes,
    const std::vector<PyBlock> &successors, unsigned regions,
    const std::optional<PyLocation> &location,
    const std::optional<PyInsertionPoint> &ip,
    const std::optional<PyContext> &context) {
  std::vector<ContextHandle> hints;
  if (!operands.empty())
    hints.push_back(operands.front().tree->context());
  if (results && !results->empty())
    hints.push_back(results->front().context);
  ContextHandle state = context ? context->state
                                : resolveOperationContext(location, ip, hints);
  mlir::OperationName opName = lookupOperationName(state, name);

  mlir::OperationState opState(resolveLocation(state, location), opName);
  for (const PyValue &operand : operands) {
    requireSameContext(state, operand.tree->context(), "operand");
    opState.operands.push_back(get(operand));
  }
  if (attributes)
    for (const auto &[key, attr] : *attributes) {
      requireSameContext(state, attr.context, "attribute");
      opState.addAttribute(key, attr.attribute);
    }
  if (!successors.empty()) {
    if (opName.isRegistered() && !opName.hasTrait<mlir::OpTrait::IsTerminator>())
      throw std::invalid_argument("only terminators can have successors; '" +
                                  name + "' is not a terminator");
    for (const PyBlock &successor : successors) {
      requireSameContext(state, successor.tree->context(), "successor");
      opState.successors.push_back(get(successor));
    }
  }
  for (unsigned i = 0; i < regions; ++i)
    opState.addRegion();

  bool infer = false;
  if (results) {
    for (const PyType &type : *results) {
      requireSameContext(state, type.context, "result type");
      opState.types.push_back(type.type);
    }
  } else {
    std::optional<mlir::RegisteredOperationName> info = opName.getRegisteredInfo();
    infer = info && info->hasInterface<mlir::InferTypeOpInterface>();
  }
  PyOperation op = createOperation(state, opState, infer, ip);
  return wrapOp(op.tree, op.op);
}

//===--------------------------------------------------------------------===//
// Printing
//===--------------------------------------------------------------------===//

mlir::OpPrintingFlags printingFlags(bool generic, bool debugInfo,
                                    std::optional<int64_t> largeElementsLimit,
                                    bool useLocalScope, bool assumeVerified,
                                    bool skipRegions) {
  mlir::OpPrintingFlags flags;
  flags.printGenericOpForm(generic);
  flags.enableDebugInfo(debugInfo);
  if (largeElementsLimit)
    flags.elideLargeElementsAttrs(*largeElementsLimit);
  flags.useLocalScope(useLocalScope);
  flags.assumeVerified(assumeVerified);
  flags.skipRegions(skipRegions);
  return flags;
}

std::string printOperation(mlir::Operation *op,
                           const mlir::OpPrintingFlags &flags) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  op->print(stream, flags);
  return result;
}

std::string printValue(mlir::Value value) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  value.printAsOperand(stream, mlir::OpPrintingFlags().useLocalScope());
  return result;
}

//===--------------------------------------------------------------------===//
// Lists
//===--------------------------------------------------------------------===//

std::vector<ValueObject> wrapValues(const OpTreeHandle &tree,
                                    mlir::ValueRange values) {
  std::vector<ValueObject> result;
  for (mlir::Value value : values)
    result.push_back(wrapValue(tree, value));
  return result;
}

std::vector<OperationObject> operationsOf(const OpTreeHandle &tree,
                                          mlir::Block *block) {
  std::vector<OperationObject> result;
  for (mlir::Operation &op : *block)
    result.push_back(wrapOp(tree, &op));
  return result;
}

std::vector<PyBlock> blocksOf(const OpTreeHandle &tree, mlir::Region *region) {
  std::vector<PyBlock> result;
  for (mlir::Block &block : *region)
    result.push_back(PyBlock{tree, &block});
  return result;
}

size_t normalizeIndex(int64_t index, size_t size) {
  int64_t signedSize = static_cast<int64_t>(size);
  if (index < 0)
    index += signedSize;
  if (index < 0 || index >= signedSize)
    throw nb::index_error("index out of range");
  return static_cast<size_t>(index);
}

std::vector<mlir::Location>
blockArgLocations(const ContextHandle &context, size_t count,
                  const std::optional<std::vector<PyLocation>> &locations) {
  if (!locations)
    return std::vector<mlir::Location>(count,
                                       resolveLocation(context, std::nullopt));
  if (locations->size() != count)
    throw std::invalid_argument(
        "arg_locations must have one entry per argument type");
  std::vector<mlir::Location> result;
  for (const PyLocation &location : *locations) {
    requireSameContext(context, location.context, "location");
    result.push_back(location.location);
  }
  return result;
}

/// Parses `source` into a detached module.
PyModule adoptModule(const ContextHandle &context,
                     mlir::OwningOpRef<mlir::ModuleOp> module,
                     const DiagnosticCapture &capture, const std::string &what) {
  if (!module)
    capture.raise("failed to parse " + what);
  mlir::Operation *op = module.release().getOperation();
  PyModule result;
  static_cast<PyOperation &>(result) = PyOperation{OpTree::adopt(context, op), op};
  return result;
}

constexpr const char *kAsmDoc =
    "Args:\n"
    "    generic: Print the generic form ``\"dialect.op\"(...)``.\n"
    "    debug_info: Include source locations.\n"
    "    large_elements_limit: Elide elements attributes with more than\n"
    "        this many elements.\n"
    "    use_local_scope: Number SSA values as if this were the top level.\n"
    "    assume_verified: Skip the verifier the printer otherwise runs\n"
    "        (and falls back to the generic form on failure).\n"
    "    skip_regions: Omit region bodies.";

} // namespace

void registerOperationClass(mlir::TypeID op, OperationFactory factory) {
  operationClasses()[op] = factory;
}

namespace {
llvm::DenseMap<mlir::TypeID, PostCreateHook> &postCreateHooks() {
  static llvm::DenseMap<mlir::TypeID, PostCreateHook> hooks;
  return hooks;
}
} // namespace

void registerPostCreateHook(mlir::TypeID op, PostCreateHook hook) {
  postCreateHooks()[op] = hook;
}

void runPostCreateHook(mlir::Operation *op) {
  if (std::optional<mlir::RegisteredOperationName> info = op->getRegisteredInfo()) {
    auto it = postCreateHooks().find(info->getTypeID());
    if (it != postCreateHooks().end())
      it->second(op);
  }
}

OperationObject wrapOperation(const OpTreeHandle &tree, mlir::Operation *op) {
  PyOperation handle{tree, op};
  if (std::optional<mlir::RegisteredOperationName> info =
          op->getRegisteredInfo()) {
    auto it = operationClasses().find(info->getTypeID());
    if (it != operationClasses().end())
      return it->second(handle);
  }
  return nb::cast(std::move(handle));
}

ContextHandle resolveOperationContext(
    const std::optional<PyLocation> &location,
    const std::optional<PyInsertionPoint> &ip,
    const std::vector<ContextHandle> &hints) {
  if (location)
    return location->context;
  if (ip)
    return ip->tree->context();
  for (const ContextHandle &hint : hints)
    if (hint)
      return hint;
  return resolveContext(std::nullopt);
}

PyOperation createOperation(const ContextHandle &context,
                            mlir::OperationState &state,
                            bool inferResultTypes,
                            const std::optional<PyInsertionPoint> &ip) {
  std::optional<PyInsertionPoint> point = resolveInsertionPoint(ip);
  if (point)
    requireSameContext(context, point->tree->context(), "insertion point");
  if (inferResultTypes) {
    std::optional<mlir::RegisteredOperationName> info =
        state.name.getRegisteredInfo();
    if (!info || !info->hasInterface<mlir::InferTypeOpInterface>())
      throw std::invalid_argument("'" + state.name.getStringRef().str() +
                                  "' cannot infer its result types");
    inferResults(state, *info);
  }
  PyOperation op = place(context, mlir::Operation::create(state), point);
  trackCrossTreeUses(context, op.op);
  return op;
}

ValueObject wrapValue(const OpTreeHandle &tree, mlir::Value value) {
  if (llvm::isa<mlir::OpResult>(value))
    return nb::cast(PyOpResult{{tree, value}});
  return nb::cast(PyBlockArgument{{tree, value}});
}

void bindIR(nb::module_ &m) {
  nb::enum_<WalkOrder>(m, "WalkOrder", "Traversal order for ``Operation.walk``.")
      .value("PRE_ORDER", WalkOrder::PreOrder,
             "Visit an operation before the operations nested in it.")
      .value("POST_ORDER", WalkOrder::PostOrder,
             "Visit an operation after the operations nested in it. Safe for\n"
             "erasing the visited operation.");

  nb::enum_<WalkResult>(m, "WalkResult",
                        "What a ``walk`` callback wants to happen next.")
      .value("ADVANCE", WalkResult::Advance, "Keep walking (same as ``None``).")
      .value("INTERRUPT", WalkResult::Interrupt, "Stop the walk.")
      .value("SKIP", WalkResult::Skip,
             "Do not visit this operation's nested operations (pre-order).");

  // Declared first so every signature below can name them.
  auto value = nb::class_<PyValue>(
      m, "Value",
      "An SSA value: an operation result (``OpResult``) or a block argument\n"
      "(``BlockArgument``).");
  auto opResult = nb::class_<PyOpResult, PyValue>(
      m, "OpResult", "A value produced by an operation.");
  auto blockArgument = nb::class_<PyBlockArgument, PyValue>(
      m, "BlockArgument", "A value passed into a block.");
  auto opOperand = nb::class_<PyOpOperand>(
      m, "OpOperand", "One use of a value: an operand slot of an operation.");
  auto operation = nb::class_<PyOperation>(
      m, "Operation",
      "A generic MLIR operation.\n\n"
      "An operation created without an insertion point is detached and owned\n"
      "by Python; inserting it into a block transfers ownership to that\n"
      "block's IR. Handles into IR keep the whole tree alive.\n\n"
      "``erase()`` destroys an operation immediately; other handles to it or\n"
      "to IR nested in it must not be used afterwards.");
  auto operandList = nb::class_<PyOperandList>(
      m, "OperandList",
      "Live view of an operation's operands. Supports ``len``, indexing,\n"
      "assignment, and iteration.");
  auto attributeMap = nb::class_<PyAttributeMap>(
      m, "AttributeMap",
      "Live view of an operation's attributes, including inherent ones\n"
      "stored as properties. Supports ``len``, ``in``, indexing, assignment,\n"
      "deletion, and iteration over names.");
  auto block = nb::class_<PyBlock>(
      m, "Block",
      "A list of operations with typed arguments, inside a region.");
  auto region = nb::class_<PyRegion>(
      m, "Region", "A list of blocks attached to an operation.");
  auto insertionPoint = nb::class_<PyInsertionPoint>(
      m, "InsertionPoint",
      "A position in a block where new operations are inserted.\n\n"
      "Enter it with ``with`` to make ``Operation.create`` insert there by\n"
      "default.");
  auto module = nb::class_<PyModule, PyOperation>(
      m, "Module",
      "A ``builtin.module``: the usual top-level container of IR.\n\n"
      "A module is an ``Operation``; navigating to a ``builtin.module``\n"
      "always produces a ``Module``.");
  registerOperationClass<PyModule, mlir::ModuleOp>();
  module.attr("OPERATION_NAME") = "builtin.module";

  //===------------------------------------------------------------------===//
  // Values
  //===------------------------------------------------------------------===//

  value
      .def_prop_rw(
          "type",
          [](const PyValue &self) {
            return wrapType(contextOf(self.tree), get(self).getType());
          },
          [](PyValue &self, const PyType &type) {
            requireSameContext(contextOf(self.tree), type.context, "type");
            get(self).setType(type.type);
          },
          "The value's type. Assigning changes it in place without checks;\n"
          "verify afterwards.")
      .def_prop_ro(
          "context",
          [](const PyValue &self) { return PyContext{contextOf(self.tree)}; },
          "The context the value belongs to.")
      .def_prop_ro(
          "location",
          [](const PyValue &self) {
            return PyLocation{contextOf(self.tree), get(self).getLoc()};
          },
          "Where the value is defined.")
      .def_prop_ro(
          "defining_op",
          [](const PyValue &self) {
            return wrapOptionalOp(self.tree, get(self).getDefiningOp());
          },
          "The operation producing this value, or ``None`` for a block\n"
          "argument.")
      .def_prop_ro(
          "owner",
          [](const PyValue &self) -> OperationOrBlock {
            if (auto result = llvm::dyn_cast<mlir::OpResult>(get(self)))
              return nb::cast(wrapOp(self.tree, result.getOwner()));
            return nb::cast(PyBlock{
                self.tree, llvm::cast<mlir::BlockArgument>(get(self)).getOwner()});
          },
          "The defining operation or, for a block argument, its block.")
      .def_prop_ro(
          "uses",
          [](const PyValue &self) {
            std::vector<PyOpOperand> uses;
            for (mlir::OpOperand &use : get(self).getUses())
              uses.push_back(PyOpOperand{self.tree, &use});
            return uses;
          },
          "Every operand slot that currently reads this value.")
      .def_prop_ro(
          "has_uses",
          [](const PyValue &self) { return !get(self).use_empty(); },
          "Whether anything reads this value.")
      .def(
          "replace_all_uses_with",
          [](PyValue &self, const PyValue &replacement) {
            requireSameContext(contextOf(self.tree),
                               contextOf(replacement.tree), "replacement");
            get(self).replaceAllUsesWith(get(replacement));
            llvm::SmallPtrSet<mlir::Operation *, 8> users;
            for (mlir::OpOperand &use : get(replacement).getUses())
              users.insert(use.getOwner());
            for (mlir::Operation *user : users)
              trackCrossTreeUses(contextOf(self.tree), user);
          },
          "replacement"_a,
          "Make every user of this value read ``replacement`` instead.")
      .def("__eq__", identityEq<PyValue>([](const PyValue &v) -> const void * { return v.value.getAsOpaquePointer(); }),
           "other"_a.none(), nb::sig(kEqSignature))
      .def("__hash__", identityHash<PyValue>([](const PyValue &v) -> const void * { return v.value.getAsOpaquePointer(); }))
      .def("__str__", [](const PyValue &self) { return printValue(get(self)); })
      .def("__repr__", [](nb::handle self) {
        const PyValue &native = nb::cast<const PyValue &>(self);
        return nb::cast<std::string>(self.type().attr("__name__")) + "(" +
               printValue(native.value) + " : " +
               printToString(native.value.getType()) + ")";
      });

  opResult
      .def_prop_ro(
          "owner",
          [](const PyOpResult &self) {
            return wrapOp(self.tree,
                          llvm::cast<mlir::OpResult>(get(self)).getOwner());
          },
          "The operation producing this result.")
      .def_prop_ro(
          "result_number",
          [](const PyOpResult &self) {
            return llvm::cast<mlir::OpResult>(get(self)).getResultNumber();
          },
          "Position among the owner's results.");

  blockArgument
      .def_prop_ro(
          "owner",
          [](const PyBlockArgument &self) {
            return PyBlock{self.tree,
                           llvm::cast<mlir::BlockArgument>(get(self)).getOwner()};
          },
          "The block this argument belongs to.")
      .def_prop_ro(
          "arg_number",
          [](const PyBlockArgument &self) {
            return llvm::cast<mlir::BlockArgument>(get(self)).getArgNumber();
          },
          "Position among the block's arguments.");

  opOperand
      .def_prop_ro(
          "owner",
          [](const PyOpOperand &self) {
            return wrapOp(self.tree, get(self)->getOwner());
          },
          "The operation this operand belongs to.")
      .def_prop_ro(
          "operand_number",
          [](const PyOpOperand &self) {
            return get(self)->getOperandNumber();
          },
          "Position among the owner's operands.")
      .def_prop_ro(
          "value",
          [](const PyOpOperand &self) {
            return wrapValue(self.tree, get(self)->get());
          },
          "The value currently used.");

  //===------------------------------------------------------------------===//
  // Operation
  //===------------------------------------------------------------------===//

  operation
      .def_static(
          "create", &createGenericOperation, "name"_a, nb::kw_only(),
          "results"_a = nb::none(), "operands"_a = std::vector<PyValue>{},
          "attributes"_a = nb::none(), "successors"_a = std::vector<PyBlock>{},
          "regions"_a = 0, "location"_a = nb::none(), "ip"_a = nb::none(),
          "context"_a = nb::none(),
          "Create an operation by name.\n\n"
          "Args:\n"
          "    name: Full name, e.g. ``\"arith.addi\"``. Its dialect is loaded\n"
          "        on demand.\n"
          "    results: Result types. When ``None``, they are inferred for\n"
          "        operations that support it, and empty otherwise.\n"
          "    operands: Operand values.\n"
          "    attributes: Attributes by name, inherent or discardable.\n"
          "    successors: Successor blocks (terminators only).\n"
          "    regions: Number of empty regions to create.\n"
          "    location: Defaults to the current ``Location``, else unknown.\n"
          "    ip: Where to insert; defaults to the current\n"
          "        ``InsertionPoint``. Without one the result is detached.\n"
          "    context: Needed only if nothing else determines the context.\n\n"
          "Raises:\n"
          "    ValueError: If the operation is unknown and the context does\n"
          "        not allow unregistered dialects.\n"
          "    MLIRError: If result types cannot be inferred.")
      .def_static(
          "parse",
          [](const std::string &source, std::optional<PyContext> context) {
            ContextHandle state = resolveContext(context);
            DiagnosticCapture capture(&state->context);
            mlir::ParserConfig config(&state->context);
            mlir::Block block;
            if (mlir::failed(mlir::parseSourceString(source, &block, config)))
              capture.raise("failed to parse operation");
            if (block.getOperations().size() != 1)
              throw std::invalid_argument(
                  "expected exactly one top-level operation, got " +
                  std::to_string(block.getOperations().size()));
            mlir::Operation *op = &block.front();
            op->remove();
            return wrapOp(OpTree::adopt(state, op), op);
          },
          "source"_a, nb::kw_only(), "context"_a = nb::none(),
          "Parse a single detached operation from its textual form.\n\n"
          "Raises:\n"
          "    MLIRError: If ``source`` does not parse.\n"
          "    ValueError: If it holds more than one top-level operation.")
      .def_prop_ro(
          "name",
          [](const PyOperation &self) {
            return get(self)->getName().getStringRef().str();
          },
          "Full operation name, e.g. ``\"func.func\"``.")
      .def_prop_ro(
          "is_registered",
          [](const PyOperation &self) { return get(self)->isRegistered(); },
          "Whether the operation's definition is known to this build.")
      .def_prop_ro(
          "context",
          [](const PyOperation &self) { return PyContext{contextOf(self.tree)}; },
          "The context the operation belongs to.")
      .def_prop_rw(
          "location",
          [](const PyOperation &self) {
            return PyLocation{contextOf(self.tree), get(self)->getLoc()};
          },
          [](PyOperation &self, const PyLocation &location) {
            requireSameContext(contextOf(self.tree), location.context,
                               "location");
            get(self)->setLoc(location.location);
          },
          "Source location.")
      .def_prop_ro(
          "attributes",
          [](const PyOperation &self) {
            get(self);
            return PyAttributeMap{self};
          },
          "Attributes by name (live view).")
      .def_prop_ro(
          "operands",
          [](const PyOperation &self) {
            get(self);
            return PyOperandList{self};
          },
          "Operand values (live view).")
      .def_prop_ro(
          "results",
          [](const PyOperation &self) {
            std::vector<PyOpResult> results;
            for (mlir::OpResult result : get(self)->getResults())
              results.push_back(PyOpResult{{self.tree, result}});
            return results;
          },
          "Result values.")
      .def_prop_ro(
          "result",
          [](const PyOperation &self) {
            mlir::Operation *op = get(self);
            if (op->getNumResults() != 1)
              throw std::invalid_argument(
                  "'" + op->getName().getStringRef().str() + "' has " +
                  std::to_string(op->getNumResults()) +
                  " results; use .results");
            return PyOpResult{{self.tree, op->getResult(0)}};
          },
          "The only result.\n\nRaises:\n"
          "    ValueError: Unless there is exactly one result.")
      .def_prop_ro(
          "regions",
          [](const PyOperation &self) {
            std::vector<PyRegion> regions;
            for (mlir::Region &region : get(self)->getRegions())
              regions.push_back(PyRegion{self.tree, &region});
            return regions;
          },
          "Attached regions.")
      .def_prop_ro(
          "successors",
          [](const PyOperation &self) {
            std::vector<PyBlock> successors;
            for (mlir::Block *successor : get(self)->getSuccessors())
              successors.push_back(PyBlock{self.tree, successor});
            return successors;
          },
          "Successor blocks of a terminator.")
      .def_prop_ro(
          "parent",
          [](const PyOperation &self) {
            return wrapOptionalOp(self.tree, get(self)->getParentOp());
          },
          "The operation whose region contains this one, if any.")
      .def_prop_ro(
          "block",
          [](const PyOperation &self) {
            return wrapOptionalBlock(self.tree, get(self)->getBlock());
          },
          "The block containing this operation, or ``None`` if detached.")
      .def(
          "verify",
          [](const PyOperation &self) {
            mlir::Operation *op = get(self);
            DiagnosticCapture capture(op->getContext());
            if (mlir::failed(mlir::verify(op)))
              capture.raise("verification failed");
          },
          "Check the operation and everything nested in it.\n\n"
          "Raises:\n"
          "    MLIRError: With the verifier's diagnostics.")
      .def(
          "get_asm",
          [](const PyOperation &self, bool generic, bool debugInfo,
             std::optional<int64_t> largeElementsLimit, bool useLocalScope,
             bool assumeVerified, bool skipRegions) {
            return printOperation(get(self),
                                  printingFlags(generic, debugInfo,
                                                largeElementsLimit, useLocalScope,
                                                assumeVerified, skipRegions));
          },
          nb::kw_only(), "generic"_a = false, "debug_info"_a = false,
          "large_elements_limit"_a = nb::none(), "use_local_scope"_a = false,
          "assume_verified"_a = false, "skip_regions"_a = false,
          (std::string("Print the operation to a string.\n\n") + kAsmDoc).c_str())
      .def(
          "clone",
          [](const PyOperation &self) {
            mlir::Operation *copy = get(self)->clone();
            OpTreeHandle tree = OpTree::adopt(contextOf(self.tree), copy);
            trackCrossTreeUses(contextOf(self.tree), copy);
            return wrapOp(tree, copy);
          },
          "A detached deep copy of this operation.")
      .def(
          "detach_from_parent",
          [](const PyOperation &self) {
            mlir::Operation *op = get(self);
            if (!op->getBlock())
              return wrapOp(self.tree, op);
            op->remove();
            OpTreeHandle tree = OpTree::adopt(contextOf(self.tree), op);
            trackCrossTreeUses(contextOf(self.tree), op);
            return wrapOp(tree, op);
          },
          "Remove the operation from its block and return a handle that owns\n"
          "it. Keep using the returned handle: older handles do not keep the\n"
          "detached operation alive.")
      .def(
          "move_before",
          [](const PyOperation &self, const PyOperation &other) {
            mlir::Operation *anchor = get(other);
            if (!anchor->getBlock())
              throw std::invalid_argument("target operation is detached");
            insertAt(self, PyInsertionPoint{other.tree, anchor->getBlock(), anchor});
          },
          "other"_a, "Move this operation immediately before ``other``.")
      .def(
          "move_after",
          [](const PyOperation &self, const PyOperation &other) {
            mlir::Operation *anchor = get(other);
            if (!anchor->getBlock())
              throw std::invalid_argument("target operation is detached");
            insertAt(self, PyInsertionPoint{other.tree, anchor->getBlock(),
                                            anchor->getNextNode()});
          },
          "other"_a, "Move this operation immediately after ``other``.")
      .def(
          "erase",
          [](PyOperation &self) {
            mlir::Operation *op = get(self);
            if (hasUsesOutside(op))
              throw std::invalid_argument(
                  "cannot erase an operation whose results (or nested values) "
                  "are still used");
            if (!op->getBlock()) {
              auto it = contextOf(self.tree)->trees.find(op);
              if (it != contextOf(self.tree)->trees.end())
                if (OpTreeHandle tree = it->second.lock())
                  tree->forget();
            }
            op->erase();
            self.op = nullptr;
          },
          "Destroy the operation and everything nested in it.\n\n"
          "Raises:\n"
          "    ValueError: If a result (or a value nested inside) is still\n"
          "        used elsewhere.")
      .def(
          "walk",
          [](const PyOperation &self, const WalkCallback &callback,
             WalkOrder order) {
            // Exceptions must not unwind through MLIR, which is built
            // without exception support: stop the walk and rethrow after.
            std::exception_ptr error;
            auto visit = [&](mlir::Operation *op) -> mlir::WalkResult {
              try {
                nb::object result = callback(wrapOp(self.tree, op));
                if (result.is_none())
                  return mlir::WalkResult::advance();
                switch (nb::cast<WalkResult>(result)) {
                case WalkResult::Advance:
                  return mlir::WalkResult::advance();
                case WalkResult::Interrupt:
                  return mlir::WalkResult::interrupt();
                case WalkResult::Skip:
                  return mlir::WalkResult::skip();
                }
              } catch (...) {
                error = std::current_exception();
              }
              return mlir::WalkResult::interrupt();
            };
            mlir::WalkResult result =
                order == WalkOrder::PreOrder
                    ? get(self)->walk<mlir::WalkOrder::PreOrder>(visit)
                    : get(self)->walk<mlir::WalkOrder::PostOrder>(visit);
            if (error)
              std::rethrow_exception(error);
            return result.wasInterrupted() ? WalkResult::Interrupt
                                           : WalkResult::Advance;
          },
          "callback"_a, "order"_a = WalkOrder::PostOrder,
          "Call ``callback`` on this operation and every nested one.\n\n"
          "Args:\n"
          "    callback: Returns ``None`` (or ``WalkResult.ADVANCE``) to\n"
          "        continue, ``INTERRUPT`` to stop, or ``SKIP`` (pre-order\n"
          "        only) to skip nested operations. Exceptions propagate.\n"
          "    order: Post-order by default.\n\n"
          "Returns:\n"
          "    ``WalkResult.INTERRUPT`` if the walk stopped early, else\n"
          "    ``WalkResult.ADVANCE``.")
      .def("__eq__", identityEq<PyOperation>([](const PyOperation &v) -> const void * { return v.op; }),
           "other"_a.none(), nb::sig(kEqSignature))
      .def("__hash__", identityHash<PyOperation>([](const PyOperation &v) -> const void * { return v.op; }))
      .def("__str__",
           [](const PyOperation &self) {
             return printOperation(get(self), mlir::OpPrintingFlags());
           })
      .def("__repr__", [](const PyOperation &self) {
        return printOperation(get(self), mlir::OpPrintingFlags());
      });

  operandList
      .def("__len__",
           [](const PyOperandList &self) {
             return get(self.owner)->getNumOperands();
           })
      .def(
          "__getitem__",
          [](const PyOperandList &self, int64_t index) {
            mlir::Operation *op = get(self.owner);
            return wrapValue(self.owner.tree,
                             op->getOperand(normalizeIndex(index, op->getNumOperands())));
          },
          "index"_a)
      .def(
          "__setitem__",
          [](PyOperandList &self, int64_t index, const PyValue &value) {
            mlir::Operation *op = get(self.owner);
            requireSameContext(contextOf(self.owner.tree), contextOf(value.tree),
                               "operand");
            op->setOperand(normalizeIndex(index, op->getNumOperands()), get(value));
            trackCrossTreeUses(contextOf(self.owner.tree), op);
          },
          "index"_a, "value"_a)
      .def("__iter__",
           [](const PyOperandList &self) -> nb::typed<nb::iterator, PyValue> {
             return nb::iter(nb::cast(
                 wrapValues(self.owner.tree, get(self.owner)->getOperands())));
           });

  attributeMap
      .def("__len__",
           [](const PyAttributeMap &self) {
             return get(self.owner)->getAttrs().size();
           })
      .def(
          "__contains__",
          [](const PyAttributeMap &self, const std::string &name) {
            return static_cast<bool>(get(self.owner)->getAttr(name));
          },
          "name"_a)
      .def(
          "__getitem__",
          [](const PyAttributeMap &self, const std::string &name) {
            mlir::Attribute attr = get(self.owner)->getAttr(name);
            if (!attr)
              throw nb::key_error(name.c_str());
            return wrapAttribute(contextOf(self.owner.tree), attr);
          },
          "name"_a)
      .def(
          "__setitem__",
          [](PyAttributeMap &self, const std::string &name,
             const PyAttribute &attr) {
            requireSameContext(contextOf(self.owner.tree), attr.context,
                               "attribute");
            get(self.owner)->setAttr(name, attr.attribute);
          },
          "name"_a, "attr"_a)
      .def(
          "__delitem__",
          [](PyAttributeMap &self, const std::string &name) {
            if (!get(self.owner)->removeAttr(name))
              throw nb::key_error(name.c_str());
          },
          "name"_a)
      .def(
          "get",
          [](const PyAttributeMap &self, const std::string &name) {
            return wrapOptionalAttribute(contextOf(self.owner.tree),
                                         get(self.owner)->getAttr(name));
          },
          "name"_a, "The attribute called ``name``, or ``None``.")
      .def("__iter__",
           [](const PyAttributeMap &self) -> nb::typed<nb::iterator, std::string> {
             std::vector<std::string> names;
             for (mlir::NamedAttribute attr : get(self.owner)->getAttrs())
               names.push_back(attr.getName().str());
             return nb::iter(nb::cast(names));
           })
      .def(
          "keys",
          [](const PyAttributeMap &self) {
            std::vector<std::string> names;
            for (mlir::NamedAttribute attr : get(self.owner)->getAttrs())
              names.push_back(attr.getName().str());
            return names;
          },
          "Attribute names, sorted.")
      .def(
          "items",
          [](const PyAttributeMap &self) {
            std::vector<std::pair<std::string, AttributeObject>> items;
            for (mlir::NamedAttribute attr : get(self.owner)->getAttrs())
              items.emplace_back(attr.getName().str(),
                                 wrapAttribute(contextOf(self.owner.tree),
                                               attr.getValue()));
            return items;
          },
          "``(name, attribute)`` pairs, sorted by name.");

  //===------------------------------------------------------------------===//
  // Block and Region
  //===------------------------------------------------------------------===//

  block
      .def_prop_ro(
          "arguments",
          [](const PyBlock &self) {
            std::vector<PyBlockArgument> args;
            for (mlir::BlockArgument arg : get(self)->getArguments())
              args.push_back(PyBlockArgument{{self.tree, arg}});
            return args;
          },
          "The block's arguments.")
      .def(
          "add_argument",
          [](PyBlock &self, const PyType &type,
             std::optional<PyLocation> location) {
            const ContextHandle &context = contextOf(self.tree);
            requireSameContext(context, type.context, "type");
            mlir::BlockArgument arg = get(self)->addArgument(
                type.type, resolveLocation(context, location));
            return PyBlockArgument{{self.tree, arg}};
          },
          "type"_a, "location"_a = nb::none(),
          "Append an argument of ``type`` and return it.")
      .def(
          "erase_argument",
          [](PyBlock &self, int64_t index) {
            size_t position = normalizeIndex(index, get(self)->getNumArguments());
            if (!get(self)->getArgument(position).use_empty())
              throw std::invalid_argument("argument still has uses");
            get(self)->eraseArgument(position);
          },
          "index"_a, "Remove an unused argument.")
      .def_prop_ro(
          "operations",
          [](const PyBlock &self) { return operationsOf(self.tree, get(self)); },
          "The operations, in order.")
      .def("__len__",
           [](const PyBlock &self) { return get(self)->getOperations().size(); })
      .def("__iter__",
           [](const PyBlock &self) -> nb::typed<nb::iterator, PyOperation> {
             return nb::iter(nb::cast(operationsOf(self.tree, get(self))));
           })
      .def_prop_ro(
          "terminator",
          [](const PyBlock &self) -> OptionalOperationObject {
            if (get(self)->empty() ||
                !get(self)->back().mightHaveTrait<mlir::OpTrait::IsTerminator>())
              return nb::none();
            return wrapOp(self.tree, &get(self)->back());
          },
          "The last operation if it is (or may be) a terminator.")
      .def_prop_ro(
          "parent_op",
          [](const PyBlock &self) {
            return wrapOptionalOp(self.tree, get(self)->getParentOp());
          },
          "The operation owning this block's region.")
      .def_prop_ro(
          "region",
          [](const PyBlock &self) -> std::optional<PyRegion> {
            if (mlir::Region *region = get(self)->getParent())
              return PyRegion{self.tree, region};
            return std::nullopt;
          },
          "The region containing this block.")
      .def(
          "append",
          [](PyBlock &self, const PyOperation &op) {
            insertAt(op, PyInsertionPoint{self.tree, get(self), nullptr});
          },
          "op"_a,
          "Move ``op`` to the end of this block. A detached operation becomes\n"
          "owned by this block's IR.")
      .def("__eq__", identityEq<PyBlock>([](const PyBlock &v) -> const void * { return v.block; }),
           "other"_a.none(), nb::sig(kEqSignature))
      .def("__hash__", identityHash<PyBlock>([](const PyBlock &v) -> const void * { return v.block; }))
      .def("__str__", [](const PyBlock &self) {
        std::string result;
        llvm::raw_string_ostream stream(result);
        get(self)->print(stream);
        return result;
      });

  region
      .def_prop_ro(
          "blocks",
          [](const PyRegion &self) { return blocksOf(self.tree, get(self)); },
          "The blocks, in order. The first is the entry block.")
      .def("__len__",
           [](const PyRegion &self) { return get(self)->getBlocks().size(); })
      .def("__iter__",
           [](const PyRegion &self) -> nb::typed<nb::iterator, PyBlock> {
             return nb::iter(nb::cast(blocksOf(self.tree, get(self))));
           })
      .def_prop_ro(
          "entry_block",
          [](const PyRegion &self) -> std::optional<PyBlock> {
            if (get(self)->empty())
              return std::nullopt;
            return PyBlock{self.tree, &get(self)->front()};
          },
          "The first block, or ``None`` if the region is empty.")
      .def(
          "append_block",
          [](PyRegion &self, const std::vector<PyType> &argTypes,
             std::optional<std::vector<PyLocation>> argLocations) {
            const ContextHandle &context = contextOf(self.tree);
            std::vector<mlir::Type> types;
            for (const PyType &type : argTypes) {
              requireSameContext(context, type.context, "argument type");
              types.push_back(type.type);
            }
            auto *newBlock = new mlir::Block();
            newBlock->addArguments(
                types, blockArgLocations(context, types.size(), argLocations));
            get(self)->push_back(newBlock);
            return PyBlock{self.tree, newBlock};
          },
          "arg_types"_a = std::vector<PyType>{}, "arg_locations"_a = nb::none(),
          "Append a new block with the given argument types and return it.\n\n"
          "Args:\n"
          "    arg_types: Argument types.\n"
          "    arg_locations: One location per argument; the current\n"
          "        location by default.")
      .def_prop_ro(
          "parent_op",
          [](const PyRegion &self) {
            return wrapOp(self.tree, get(self)->getParentOp());
          },
          "The operation owning this region.")
      .def("__eq__", identityEq<PyRegion>([](const PyRegion &v) -> const void * { return v.region; }),
           "other"_a.none(), nb::sig(kEqSignature))
      .def("__hash__", identityHash<PyRegion>([](const PyRegion &v) -> const void * { return v.region; }));

  //===------------------------------------------------------------------===//
  // InsertionPoint
  //===------------------------------------------------------------------===//

  insertionPoint
      .def(nb::new_([](const PyBlock &block) {
             mlir::Operation *terminator = nullptr;
             if (!get(block)->empty() &&
                 get(block)->back().hasTrait<mlir::OpTrait::IsTerminator>())
               terminator = &get(block)->back();
             return PyInsertionPoint{block.tree, get(block), terminator};
           }),
           "block"_a,
           "Insert at the end of ``block``, but before its terminator if it\n"
           "already has one (operations after a terminator are invalid).")
      .def_static(
          "at_block_end",
          [](const PyBlock &block) {
            return PyInsertionPoint{block.tree, get(block), nullptr};
          },
          "block"_a,
          "Insert after everything in ``block``, including a terminator.")
      .def_static(
          "at_block_begin",
          [](const PyBlock &block) {
            return PyInsertionPoint{
                block.tree, get(block),
                get(block)->empty() ? nullptr : &get(block)->front()};
          },
          "block"_a, "Insert at the start of ``block``.")
      .def_static(
          "at_block_terminator",
          [](const PyBlock &block) {
            if (get(block)->empty() ||
                !get(block)->back().mightHaveTrait<mlir::OpTrait::IsTerminator>())
              throw std::invalid_argument("block has no terminator");
            return PyInsertionPoint{block.tree, get(block), &get(block)->back()};
          },
          "block"_a,
          "Insert just before ``block``'s terminator.\n\n"
          "Raises:\n    ValueError: If the block has no terminator.")
      .def_static(
          "before",
          [](const PyOperation &op) {
            mlir::Operation *native = get(op);
            if (!native->getBlock())
              throw std::invalid_argument("operation is detached");
            return PyInsertionPoint{op.tree, native->getBlock(), native};
          },
          "op"_a, "Insert just before ``op``.")
      .def_static(
          "after",
          [](const PyOperation &op) {
            mlir::Operation *native = get(op);
            if (!native->getBlock())
              throw std::invalid_argument("operation is detached");
            return PyInsertionPoint{op.tree, native->getBlock(),
                                    native->getNextNode()};
          },
          "op"_a, "Insert just after ``op``.")
      .def_static(
          "current",
          []() { return resolveInsertionPoint(std::nullopt); },
          "The innermost ``with InsertionPoint`` block's point, if any.")
      .def_prop_ro(
          "block",
          [](const PyInsertionPoint &self) {
            checkCurrent(self);
            return PyBlock{self.tree, self.block};
          },
          "The block operations are inserted into.")
      .def_prop_ro(
          "ref_operation",
          [](const PyInsertionPoint &self) {
            checkCurrent(self);
            return wrapOptionalOp(self.tree, self.before);
          },
          "The operation new ones go before, or ``None`` for the block end.")
      .def(
          "insert",
          [](const PyInsertionPoint &self, const PyOperation &op) {
            insertAt(op, self);
          },
          "op"_a,
          "Move ``op`` to this point. A detached operation becomes owned by\n"
          "this block's IR.")
      .def(
          "__enter__",
          [](nb::handle self) {
            pushInsertionPoint(nb::cast<PyInsertionPoint &>(self));
            return nb::borrow(self);
          },
          nb::sig("def __enter__(self) -> typing.Self"),
          "Make this the default insertion point inside a ``with`` block.")
      .def(
          "__exit__",
          [](PyInsertionPoint &self, nb::handle, nb::handle, nb::handle) {
            popInsertionPoint(self);
          },
          "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(),
          nb::sig("def __exit__(self, exc_type: type[BaseException] | None, "
                  "exc_value: BaseException | None, traceback: object, "
                  "/) -> None"));

  //===------------------------------------------------------------------===//
  // Module
  //===------------------------------------------------------------------===//

  module
      .def(nb::new_([](std::optional<PyLocation> location,
                       std::optional<PyContext> context) {
             ContextHandle state =
                 location ? location->context : resolveContext(context);
             mlir::Operation *op =
                 mlir::ModuleOp::create(resolveLocation(state, location))
                     .getOperation();
             PyModule result;
             static_cast<PyOperation &>(result) =
                 PyOperation{OpTree::adopt(state, op), op};
             return result;
           }),
           "location"_a = nb::none(), nb::kw_only(), "context"_a = nb::none(),
           "Create an empty module.")
      .def_static(
          "parse",
          [](const std::string &source, std::optional<PyContext> context) {
            ContextHandle state = resolveContext(context);
            DiagnosticCapture capture(&state->context);
            return adoptModule(
                state,
                mlir::parseSourceString<mlir::ModuleOp>(source, &state->context),
                capture, "module");
          },
          "source"_a, nb::kw_only(), "context"_a = nb::none(),
          "Parse a module from text. Top-level operations that are not a\n"
          "module are wrapped in one.\n\n"
          "Raises:\n    MLIRError: With the parser's diagnostics.")
      .def_static(
          "parse_file",
          [](const std::string &path, std::optional<PyContext> context) {
            ContextHandle state = resolveContext(context);
            DiagnosticCapture capture(&state->context);
            mlir::ParserConfig config(&state->context);
            return adoptModule(
                state, mlir::parseSourceFile<mlir::ModuleOp>(path, config),
                capture, "'" + path + "'");
          },
          "path"_a, nb::kw_only(), "context"_a = nb::none(),
          "Parse a module from a ``.mlir`` file (text or bytecode).\n\n"
          "Raises:\n    MLIRError: If the file cannot be read or parsed.")
      .def_prop_ro(
          "body",
          [](const PyModule &self) {
            return PyBlock{self.tree,
                           llvm::cast<mlir::ModuleOp>(get(self)).getBody()};
          },
          "The module's single block.");
}

} // namespace mlir_python
