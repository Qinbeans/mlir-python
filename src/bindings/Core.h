// Shared handle types for the mlir_python bindings.
//
// Ownership model
// ---------------
// * Every native object retains the `ContextState` that owns its
//   `mlir::MLIRContext`, so a context outlives everything derived from it.
// * IR objects (operations, blocks, regions, values) retain an `OpTree`. An
//   `OpTree` owns one detached top-level operation and destroys it when the last
//   Python handle into that IR goes away. When a detached operation is inserted
//   into another tree, its `OpTree` stops owning it and instead retains the
//   destination tree, so existing handles keep the new owner alive.
// * When IR in one tree reads values defined in another (e.g. a detached op
//   using a detached constant), the reading tree retains the defining tree, so
//   definitions outlive their uses. See `trackCrossTreeUses`.
// * Passes rewrite IR behind Python's back. Every IR handle records its tree's
//   version when created; running a pass bumps the version of every tree
//   holding the IR, so older handles raise instead of touching IR a pass may
//   have freed. The operation a pass ran on, and its ancestors, stay valid.
// * Explicitly erasing an operation invalidates handles into it. This mirrors
//   `Operation::erase` and is documented on the Python side.
#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "Registration.h"

namespace mlir_python {

namespace nb = nanobind;

class OpTree;
using OpTreeHandle = std::shared_ptr<OpTree>;

/// State shared by every object created in one Python `Context`.
struct ContextState {
  explicit ContextState(const mlir::DialectRegistry &registry);

  mlir::MLIRContext context;
  /// Live trees keyed by the operation they were created for.
  llvm::DenseMap<mlir::Operation *, std::weak_ptr<OpTree>> trees;
};
using ContextHandle = std::shared_ptr<ContextState>;

/// Keeps a detached operation (and everything nested in it) alive.
class OpTree : public std::enable_shared_from_this<OpTree> {
public:
  /// Takes ownership of a detached operation, reusing an existing tree object
  /// for it when one is still alive.
  static OpTreeHandle adopt(const ContextHandle &context, mlir::Operation *op);

  OpTree(ContextHandle context, mlir::Operation *root)
      : context_(std::move(context)), root_(root) {}
  OpTree(const OpTree &) = delete;
  OpTree &operator=(const OpTree &) = delete;
  ~OpTree();

  [[nodiscard]] const ContextHandle &context() const { return context_; }
  [[nodiscard]] mlir::Operation *root() const { return root_; }

  /// The tree currently responsible for destroying this tree's operation.
  [[nodiscard]] OpTreeHandle owner(const OpTreeHandle &self) const;

  /// Records that the root has been inserted into the tree owned by `owner`.
  void attachTo(OpTreeHandle owner);
  /// Makes the tree owning `user` keep the tree owning `definer` alive.
  static void addDependency(const OpTreeHandle &user,
                            const OpTreeHandle &definer);
  /// Records that the root has been destroyed by other means.
  void forget() { root_ = nullptr; }

  /// Bumped whenever a pass may have rewritten this tree's IR.
  [[nodiscard]] uint64_t version() const { return version_; }

  /// Marks handles into the IR held by `tree`'s owner stale, except handles
  /// to `survivor` and its ancestors (the operation a pass ran on).
  static void invalidate(const OpTreeHandle &tree, mlir::Operation *survivor);

  /// Throws if a handle created at `version` may refer to rewritten IR.
  /// `op` is the handle's operation, if it is one (survivors stay valid).
  void checkCurrent(uint64_t version, mlir::Operation *op) const;

private:
  /// Drops uses of values (and blocks) defined in the root from IR outside
  /// it, so destroying the root cannot abort on remaining uses.
  void dropExternalUses();

  ContextHandle context_;
  mlir::Operation *root_;
  OpTreeHandle parent_;
  /// Owner trees whose values this tree's IR reads.
  std::vector<OpTreeHandle> dependencies_;
  uint64_t version_ = 0;
  /// Trees whose roots were inserted into this tree's IR.
  std::vector<std::weak_ptr<OpTree>> attached_;
  /// Operations whose handles survive the latest invalidation.
  llvm::SmallPtrSet<mlir::Operation *, 4> survivors_;
};

/// The tree that owns `op`'s top-level operation, or null if none does.
OpTreeHandle ownerTreeOf(const ContextHandle &context, mlir::Operation *op);

/// Records keep-alive links for every use crossing between `op`'s subtree and
/// IR owned by other trees, in both directions. Call after anything that can
/// create such uses: creating, rewiring, detaching, or moving operations.
void trackCrossTreeUses(const ContextHandle &context, mlir::Operation *op);

/// Python `Context`.
struct PyContext {
  ContextHandle state;
  [[nodiscard]] mlir::MLIRContext *get() const { return &state->context; }
};

struct PyLocation {
  ContextHandle context;
  mlir::Location location;
};

struct PyType {
  ContextHandle context;
  mlir::Type type;
};

struct PyAttribute {
  ContextHandle context;
  mlir::Attribute attribute;
};

// One C++ struct per Python class so nanobind can model the hierarchy and
// signatures (including generated dialect code) can name precise classes.
struct PyIntegerType : PyType {};
struct PyIndexType : PyType {};
struct PyFloatType : PyType {};
struct PyShapedType : PyType {};
struct PyNoneType : PyType {};
struct PyComplexType : PyType {};
struct PyFunctionType : PyType {};
struct PyTupleType : PyType {};
struct PyOpaqueType : PyType {};
struct PyRankedTensorType : PyShapedType {};
struct PyUnrankedTensorType : PyShapedType {};
struct PyVectorType : PyShapedType {};
struct PyMemRefType : PyShapedType {};
struct PyUnrankedMemRefType : PyShapedType {};

struct PyIntegerAttr : PyAttribute {};
struct PyBoolAttr : PyIntegerAttr {};
struct PyFloatAttr : PyAttribute {};
struct PyStringAttr : PyAttribute {};
struct PyUnitAttr : PyAttribute {};
struct PyTypeAttr : PyAttribute {};
struct PyArrayAttr : PyAttribute {};
struct PyDictAttr : PyAttribute {};
struct PySymbolRefAttr : PyAttribute {};
struct PyFlatSymbolRefAttr : PySymbolRefAttr {};
struct PyDenseElementsAttr : PyAttribute {};
struct PyDenseIntElementsAttr : PyDenseElementsAttr {};
struct PyDenseFPElementsAttr : PyDenseElementsAttr {};
struct PyDenseArrayAttr : PyAttribute {};
struct PyStridedLayoutAttr : PyAttribute {};

// IR handles. Each records its tree's version when created; access the
// pointer through `get(handle)`, which checks that it is still current.

inline uint64_t versionOf(const OpTreeHandle &tree) {
  return tree ? tree->version() : 0;
}

struct PyOperation {
  PyOperation() = default;
  PyOperation(OpTreeHandle tree, mlir::Operation *op)
      : tree(std::move(tree)), op(op), version(versionOf(this->tree)) {}
  OpTreeHandle tree;
  mlir::Operation *op = nullptr;
  uint64_t version = 0;
};

struct PyBlock {
  PyBlock() = default;
  PyBlock(OpTreeHandle tree, mlir::Block *block)
      : tree(std::move(tree)), block(block), version(versionOf(this->tree)) {}
  OpTreeHandle tree;
  mlir::Block *block = nullptr;
  uint64_t version = 0;
};

struct PyRegion {
  PyRegion() = default;
  PyRegion(OpTreeHandle tree, mlir::Region *region)
      : tree(std::move(tree)), region(region), version(versionOf(this->tree)) {}
  OpTreeHandle tree;
  mlir::Region *region = nullptr;
  uint64_t version = 0;
};

struct PyValue {
  PyValue() = default;
  PyValue(OpTreeHandle tree, mlir::Value value)
      : tree(std::move(tree)), value(value), version(versionOf(this->tree)) {}
  OpTreeHandle tree;
  mlir::Value value;
  uint64_t version = 0;
};
struct PyOpResult : PyValue {};
struct PyBlockArgument : PyValue {};

/// A position inside a block: before `before`, or at the end when null.
struct PyInsertionPoint {
  PyInsertionPoint() = default;
  PyInsertionPoint(OpTreeHandle tree, mlir::Block *block, mlir::Operation *before)
      : tree(std::move(tree)), block(block), before(before),
        version(versionOf(this->tree)) {}
  OpTreeHandle tree;
  mlir::Block *block = nullptr;
  mlir::Operation *before = nullptr;
  uint64_t version = 0;
};

inline mlir::Operation *get(const PyOperation &handle) {
  if (!handle.op)
    throw std::invalid_argument("operation has been erased");
  handle.tree->checkCurrent(handle.version, handle.op);
  return handle.op;
}
inline mlir::Block *get(const PyBlock &handle) {
  handle.tree->checkCurrent(handle.version, nullptr);
  return handle.block;
}
inline mlir::Region *get(const PyRegion &handle) {
  handle.tree->checkCurrent(handle.version, nullptr);
  return handle.region;
}
inline mlir::Value get(const PyValue &handle) {
  handle.tree->checkCurrent(handle.version, nullptr);
  return handle.value;
}
inline void checkCurrent(const PyInsertionPoint &point) {
  point.tree->checkCurrent(point.version, nullptr);
}

/// `builtin.module`; also the base of every generated operation class.
struct PyModule : PyOperation {};

/// Raised as `mlir_python.MLIRError`.
class MLIRError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// Collects diagnostics emitted while alive instead of printing them.
class DiagnosticCapture {
public:
  explicit DiagnosticCapture(mlir::MLIRContext *context);
  [[nodiscard]] bool empty() const { return messages_.empty(); }
  /// Throws an `MLIRError` whose message is `summary` plus any diagnostics.
  [[noreturn]] void raise(const std::string &summary) const;

private:
  std::vector<std::string> messages_;
  mlir::ScopedDiagnosticHandler handler_;
};

// Implicit defaults managed by `with` blocks (see Core.cpp).
ContextHandle resolveContext(const std::optional<PyContext> &explicitContext);
mlir::Location resolveLocation(const ContextHandle &context,
                               const std::optional<PyLocation> &explicitLocation);
std::optional<PyInsertionPoint>
resolveInsertionPoint(const std::optional<PyInsertionPoint> &explicitPoint);
void pushContext(const PyContext &context);
void popContext(const PyContext &context);
void pushLocation(const PyLocation &location);
void popLocation(const PyLocation &location);
void pushInsertionPoint(const PyInsertionPoint &point);
void popInsertionPoint(const PyInsertionPoint &point);


/// Python objects whose stub type is the named class (or `X | None`).
using TypeObject = nb::typed<nb::object, PyType>;
using AttributeObject = nb::typed<nb::object, PyAttribute>;
using ValueObject = nb::typed<nb::object, PyValue>;
using OptionalTypeObject = nb::typed<nb::object, std::optional<PyType>>;
using OptionalAttributeObject =
    nb::typed<nb::object, std::optional<PyAttribute>>;

/// Makes `wrapType` produce a specific Python class for a dialect type (see
/// DialectExtras.cpp); builtin types are dispatched in Types.cpp.
using TypeFactory = nb::object (*)(const ContextHandle &context, mlir::Type type);
void registerTypeClass(mlir::TypeID type, TypeFactory factory);
template <typename Py, typename T> void registerTypeClass() {
  registerTypeClass(mlir::TypeID::get<T>(),
                    [](const ContextHandle &context, mlir::Type type) -> nb::object {
                      Py result;
                      result.context = context;
                      result.type = type;
                      return nb::cast(std::move(result));
                    });
}

/// Wraps native values in their most specific Python class.
TypeObject wrapType(const ContextHandle &context, mlir::Type type);
AttributeObject wrapAttribute(const ContextHandle &context,
                              mlir::Attribute attr);
ValueObject wrapValue(const OpTreeHandle &tree, mlir::Value value);
OptionalTypeObject wrapOptionalType(const ContextHandle &context,
                                    mlir::Type type);
OptionalAttributeObject wrapOptionalAttribute(const ContextHandle &context,
                                              mlir::Attribute attr);

/// Python objects whose stub type is `Operation` (or `Operation | None`);
/// at runtime they are the most specific registered operation class.
using OperationObject = nb::typed<nb::object, PyOperation>;
using OptionalOperationObject =
    nb::typed<nb::object, std::optional<PyOperation>>;

/// Converts a generic handle into an instance of a specific operation class.
using OperationFactory = nb::object (*)(const PyOperation &);

/// Makes `wrapOperation` produce the Python class `Py` for C++ op `Op`.
void registerOperationClass(mlir::TypeID op, OperationFactory factory);
template <typename Py, typename Op> void registerOperationClass() {
  registerOperationClass(
      mlir::TypeID::get<Op>(), [](const PyOperation &op) -> nb::object {
        Py result;
        static_cast<PyOperation &>(result) = op;
        return nb::cast(std::move(result));
      });
}

/// Completes an operation after a generated constructor creates it, e.g. by
/// adding the blocks its regions structurally require (see DialectExtras.cpp).
using PostCreateHook = void (*)(mlir::Operation *op);
void registerPostCreateHook(mlir::TypeID op, PostCreateHook hook);
void runPostCreateHook(mlir::Operation *op);

/// Wraps `op` in its most specific registered Python class.
OperationObject wrapOperation(const OpTreeHandle &tree, mlir::Operation *op);

/// Picks the context for a new operation: the location's, the insertion
/// point's, the first non-null hint (e.g. an operand's), else the current one.
ContextHandle resolveOperationContext(
    const std::optional<PyLocation> &location,
    const std::optional<PyInsertionPoint> &ip,
    const std::vector<ContextHandle> &hints);

/// Creates the operation described by `state`, inferring result types first
/// when `inferResultTypes` is set, and inserts it at `ip` (or the current
/// insertion point). Without an insertion point the result is detached.
PyOperation createOperation(const ContextHandle &context,
                            mlir::OperationState &state,
                            bool inferResultTypes,
                            const std::optional<PyInsertionPoint> &ip);

/// Throws unless both objects were created in the same context.
void requireSameContext(const ContextHandle &expected,
                        const ContextHandle &actual, const char *what);

/// Prints any printable MLIR entity to a string.
template <typename T> std::string printToString(const T &value) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  value.print(stream);
  return result;
}

/// `__eq__(other: object)` and `__hash__` from an identity key, so comparing
/// with unrelated objects returns False instead of raising TypeError.
inline constexpr const char *kEqSignature =
    "def __eq__(self, other: object, /) -> bool";

template <typename Py, typename KeyFn> auto identityEq(KeyFn key) {
  return [key](const Py &self, nb::handle other) {
    return nb::isinstance<Py>(other) &&
           key(self) == key(nb::cast<const Py &>(other));
  };
}

template <typename Py, typename KeyFn> auto identityHash(KeyFn key) {
  return [key](const Py &self) { return std::hash<const void *>()(key(self)); };
}

void bindCore(nb::module_ &m);
void bindTypes(nb::module_ &m);
void bindAttributes(nb::module_ &m);
void bindIR(nb::module_ &m);
void bindPasses(nb::module_ &m);
void bindCodegen(nb::module_ &m);
/// Binds every generated dialect module (see CMakeLists.txt).
void bindDialects(nb::module_ &m);
/// Adds hand-written helpers to generated dialect classes.
void bindDialectExtras(nb::module_ &m);

} // namespace mlir_python
