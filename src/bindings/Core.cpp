#include "Core.h"

#include <mlir/IR/BuiltinAttributes.h>

namespace mlir_python {

ContextState::ContextState(const mlir::DialectRegistry &registry)
    : context(registry) {}

//===----------------------------------------------------------------------===//
// OpTree
//===----------------------------------------------------------------------===//

OpTreeHandle OpTree::adopt(const ContextHandle &context, mlir::Operation *op) {
  std::weak_ptr<OpTree> &slot = context->trees[op];
  if (OpTreeHandle existing = slot.lock()) {
    // The operation was inserted somewhere and has been detached again.
    existing->parent_.reset();
    existing->root_ = op;
    return existing;
  }
  auto tree = std::make_shared<OpTree>(context, op);
  slot = tree;
  return tree;
}

OpTree::~OpTree() {
  if (!root_)
    return;
  auto it = context_->trees.find(root_);
  if (it != context_->trees.end() && it->second.expired())
    context_->trees.erase(it);
  if (!parent_) {
    dropExternalUses();
    root_->destroy();
  }
  // `dependencies_` is released after the root is gone, so definitions it
  // keeps alive outlive this tree's uses of them.
}

void OpTree::dropExternalUses() {
  auto isOutside = [&](mlir::Operation *user) {
    return !root_->isAncestor(user);
  };
  root_->walk([&](mlir::Operation *op) {
    for (mlir::OpResult result : op->getResults())
      for (mlir::OpOperand &use : llvm::make_early_inc_range(result.getUses()))
        if (isOutside(use.getOwner()))
          use.drop();
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region) {
        for (mlir::BlockArgument arg : block.getArguments())
          for (mlir::OpOperand &use : llvm::make_early_inc_range(arg.getUses()))
            if (isOutside(use.getOwner()))
              use.drop();
        for (mlir::BlockOperand &use : llvm::make_early_inc_range(block.getUses()))
          if (isOutside(use.getOwner()))
            use.drop();
      }
  });
}

OpTreeHandle OpTree::owner(const OpTreeHandle &self) const {
  OpTreeHandle current = self;
  while (current->parent_)
    current = current->parent_;
  return current;
}

void OpTree::invalidate(const OpTreeHandle &tree, mlir::Operation *survivor) {
  OpTreeHandle owner = tree->owner(tree);
  std::vector<OpTree *> pending{owner.get()};
  while (!pending.empty()) {
    OpTree *current = pending.back();
    pending.pop_back();
    ++current->version_;
    current->survivors_.clear();
    llvm::erase_if(current->attached_,
                   [](const std::weak_ptr<OpTree> &t) { return t.expired(); });
    for (const std::weak_ptr<OpTree> &attached : current->attached_)
      if (OpTreeHandle child = attached.lock())
        pending.push_back(child.get());
  }
  for (mlir::Operation *op = survivor; op; op = op->getParentOp())
    owner->survivors_.insert(op);
}

void OpTree::checkCurrent(uint64_t version, mlir::Operation *op) const {
  if (version == version_)
    return;
  if (op)
    for (const OpTree *tree = this; tree; tree = tree->parent_.get())
      if (tree->survivors_.contains(op))
        return;
  throw std::invalid_argument(
      "this handle is stale: a pass rewrote the IR it points into; navigate "
      "again from the operation the pass ran on");
}

void OpTree::attachTo(OpTreeHandle owner) {
  owner->attached_.push_back(weak_from_this());
  // The owner now holds this tree's IR, so it inherits its dependencies; links
  // that now point back into the owner itself would be cycles, so drop them.
  for (OpTreeHandle &dependency : dependencies_)
    owner->dependencies_.push_back(std::move(dependency));
  dependencies_.clear();
  parent_ = std::move(owner);
  OpTreeHandle ownerHandle = parent_->owner(parent_);
  llvm::erase_if(ownerHandle->dependencies_, [&](const OpTreeHandle &tree) {
    return tree->owner(tree) == ownerHandle;
  });
}

void OpTree::addDependency(const OpTreeHandle &user,
                           const OpTreeHandle &definer) {
  OpTreeHandle userOwner = user->owner(user);
  OpTreeHandle definerOwner = definer->owner(definer);
  if (userOwner == definerOwner ||
      llvm::is_contained(userOwner->dependencies_, definerOwner))
    return;
  userOwner->dependencies_.push_back(std::move(definerOwner));
}

OpTreeHandle ownerTreeOf(const ContextHandle &context, mlir::Operation *op) {
  mlir::Operation *top = op;
  while (mlir::Operation *parent = top->getParentOp())
    top = parent;
  auto it = context->trees.find(top);
  if (it == context->trees.end())
    return nullptr;
  OpTreeHandle tree = it->second.lock();
  return tree ? tree->owner(tree) : nullptr;
}

void trackCrossTreeUses(const ContextHandle &context, mlir::Operation *op) {
  OpTreeHandle tree = ownerTreeOf(context, op);
  if (!tree)
    return;
  auto definerOf = [](mlir::Value value) -> mlir::Operation * {
    if (mlir::Operation *definer = value.getDefiningOp())
      return definer;
    return llvm::cast<mlir::BlockArgument>(value).getOwner()->getParentOp();
  };
  auto link = [&](mlir::Operation *user, mlir::Operation *definer) {
    if (!user || !definer)
      return;
    OpTreeHandle userTree = ownerTreeOf(context, user);
    OpTreeHandle definerTree = ownerTreeOf(context, definer);
    if (userTree && definerTree && userTree != definerTree)
      OpTree::addDependency(userTree, definerTree);
  };
  op->walk([&](mlir::Operation *nested) {
    // Values this subtree reads from elsewhere.
    for (mlir::Value operand : nested->getOperands())
      if (operand)
        link(nested, definerOf(operand));
    for (mlir::Block *successor : nested->getSuccessors())
      link(nested, successor->getParentOp());
    // Values this subtree defines that are read elsewhere.
    for (mlir::OpResult result : nested->getResults())
      for (mlir::OpOperand &use : result.getUses())
        link(use.getOwner(), nested);
  });
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

namespace {

const char *severityName(mlir::DiagnosticSeverity severity) {
  switch (severity) {
  case mlir::DiagnosticSeverity::Note:
    return "note";
  case mlir::DiagnosticSeverity::Warning:
    return "warning";
  case mlir::DiagnosticSeverity::Error:
    return "error";
  case mlir::DiagnosticSeverity::Remark:
    return "remark";
  }
  return "diagnostic";
}

std::string formatDiagnostic(const mlir::Diagnostic &diagnostic) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << diagnostic.getLocation() << ": "
         << severityName(diagnostic.getSeverity()) << ": " << diagnostic;
  for (const mlir::Diagnostic &note : diagnostic.getNotes())
    stream << "\n  " << note.getLocation() << ": note: " << note;
  return result;
}

} // namespace

DiagnosticCapture::DiagnosticCapture(mlir::MLIRContext *context)
    : handler_(context, [this](mlir::Diagnostic &diagnostic) {
        messages_.push_back(formatDiagnostic(diagnostic));
        return mlir::success();
      }) {}

void DiagnosticCapture::raise(const std::string &summary) const {
  std::string message = summary;
  for (const std::string &diagnostic : messages_)
    message += "\n" + diagnostic;
  throw MLIRError(message);
}

//===----------------------------------------------------------------------===//
// Implicit defaults
//===----------------------------------------------------------------------===//

namespace {

thread_local std::vector<ContextHandle> contextStack;
thread_local std::vector<PyLocation> locationStack;
thread_local std::vector<PyInsertionPoint> insertionPointStack;

template <typename T, typename Same>
void popChecked(std::vector<T> &stack, const Same &isSame, const char *what) {
  if (stack.empty() || !isSame(stack.back()))
    throw std::runtime_error(std::string(what) +
                             " exited out of order with its `with` block");
  stack.pop_back();
}

} // namespace

ContextHandle resolveContext(const std::optional<PyContext> &explicitContext) {
  if (explicitContext)
    return explicitContext->state;
  if (!contextStack.empty())
    return contextStack.back();
  if (!locationStack.empty())
    return locationStack.back().context;
  if (!insertionPointStack.empty())
    return insertionPointStack.back().tree->context();
  throw std::invalid_argument(
      "no MLIR context: pass `context=` or enter one with `with Context():`");
}

mlir::Location
resolveLocation(const ContextHandle &context,
                const std::optional<PyLocation> &explicitLocation) {
  if (explicitLocation) {
    requireSameContext(context, explicitLocation->context, "location");
    return explicitLocation->location;
  }
  if (!locationStack.empty() && locationStack.back().context == context)
    return locationStack.back().location;
  return mlir::UnknownLoc::get(&context->context);
}

std::optional<PyInsertionPoint>
resolveInsertionPoint(const std::optional<PyInsertionPoint> &explicitPoint) {
  if (explicitPoint)
    return explicitPoint;
  if (!insertionPointStack.empty())
    return insertionPointStack.back();
  return std::nullopt;
}

void pushContext(const PyContext &context) {
  contextStack.push_back(context.state);
}

void popContext(const PyContext &context) {
  popChecked(
      contextStack,
      [&](const ContextHandle &top) { return top == context.state; },
      "Context");
}

void pushLocation(const PyLocation &location) {
  locationStack.push_back(location);
}

void popLocation(const PyLocation &location) {
  popChecked(
      locationStack,
      [&](const PyLocation &top) {
        return top.context == location.context &&
               top.location == location.location;
      },
      "Location");
}

void pushInsertionPoint(const PyInsertionPoint &point) {
  insertionPointStack.push_back(point);
}

void popInsertionPoint(const PyInsertionPoint &point) {
  popChecked(
      insertionPointStack,
      [&](const PyInsertionPoint &top) {
        return top.block == point.block && top.before == point.before;
      },
      "InsertionPoint");
}

void requireSameContext(const ContextHandle &expected,
                        const ContextHandle &actual, const char *what) {
  if (expected != actual)
    throw std::invalid_argument(std::string(what) +
                                " belongs to a different Context");
}

} // namespace mlir_python
