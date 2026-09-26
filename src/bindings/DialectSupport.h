// Runtime support for the generated dialect bindings (tools/gen_dialects.py).
//
// Generated constructors describe an operation with `GeneratedOpState` and
// generated accessors read it back through the helpers below, so the
// generated code stays a flat, readable list of named parameters.
#pragma once

#include "Core.h"

#include <string>
#include <vector>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace mlir_python {

// Defined in Attributes.cpp.
nb::int_ fromAPInt(const llvm::APInt &value, bool isSigned);
double toDouble(llvm::APFloat value);

/// Creates `parent.<name>` and registers it in `sys.modules` so that
/// `from mlir_python._mlir_python.<name> import X` works.
inline nb::module_ defineDialectModule(nb::module_ &parent, const char *name,
                                       const char *doc) {
  nb::module_ module = parent.def_submodule(name, doc);
  nb::module_::import_("sys").attr("modules")[module.attr("__name__")] = module;
  return module;
}

/// Accumulates the pieces of one operation for a generated constructor.
class GeneratedOpState {
public:
  GeneratedOpState(const char *name, const std::optional<PyLocation> &location,
                   const std::optional<PyInsertionPoint> &ip,
                   const std::vector<ContextHandle> &hints)
      : context_(resolveOperationContext(location, ip, hints)),
        builder_(&context_->context),
        state_(resolveLocation(context_, location),
               loadOperationName(context_, name)),
        ip_(ip) {}

  [[nodiscard]] mlir::Builder &builder() { return builder_; }
  [[nodiscard]] const ContextHandle &context() const { return context_; }

  void addOperand(const PyValue &value) {
    requireSameContext(context_, value.tree->context(), "operand");
    state_.operands.push_back(get(value));
    operandSegments_.push_back(1);
  }
  void addOptionalOperand(const std::optional<PyValue> &value) {
    if (value) {
      requireSameContext(context_, value->tree->context(), "operand");
      state_.operands.push_back(get(*value));
    }
    operandSegments_.push_back(value ? 1 : 0);
  }
  void addVariadicOperands(const std::vector<PyValue> &values) {
    for (const PyValue &value : values) {
      requireSameContext(context_, value.tree->context(), "operand");
      state_.operands.push_back(get(value));
    }
    operandSegments_.push_back(static_cast<int32_t>(values.size()));
  }

  /// A variadic-of-variadic operand: several groups of values, flattened, with
  /// each group's size recorded in the attribute `sizesAttribute`.
  void addOperandGroups(const std::vector<std::vector<PyValue>> &groups,
                        const char *sizesAttribute) {
    std::vector<int32_t> sizes;
    int32_t total = 0;
    for (const std::vector<PyValue> &group : groups) {
      for (const PyValue &value : group) {
        requireSameContext(context_, value.tree->context(), "operand");
        state_.operands.push_back(get(value));
      }
      sizes.push_back(static_cast<int32_t>(group.size()));
      total += static_cast<int32_t>(group.size());
    }
    operandSegments_.push_back(total);
    state_.addAttribute(sizesAttribute, builder_.getDenseI32ArrayAttr(sizes));
  }

  void addResultType(const PyType &type) {
    requireSameContext(context_, type.context, "result type");
    state_.types.push_back(type.type);
    resultSegments_.push_back(1);
  }
  void addOptionalResultType(const std::optional<PyType> &type) {
    if (type) {
      requireSameContext(context_, type->context, "result type");
      state_.types.push_back(type->type);
    }
    resultSegments_.push_back(type ? 1 : 0);
  }
  void addVariadicResultTypes(const std::vector<PyType> &types) {
    for (const PyType &type : types) {
      requireSameContext(context_, type.context, "result type");
      state_.types.push_back(type.type);
    }
    resultSegments_.push_back(static_cast<int32_t>(types.size()));
  }

  /// Sets an inherent attribute; a null attribute means "leave unset".
  void setAttribute(const char *name, mlir::Attribute value) {
    if (value)
      state_.addAttribute(name, value);
  }
  /// Sets a caller-provided attribute after checking its context.
  void setAttribute(const char *name, const PyAttribute &value) {
    requireSameContext(context_, value.context, name);
    state_.addAttribute(name, value.attribute);
  }

  void addRegions(unsigned count) {
    for (unsigned i = 0; i < count; ++i)
      state_.addRegion();
  }
  void addSuccessor(const PyBlock &block) {
    requireSameContext(context_, block.tree->context(), "successor");
    state_.successors.push_back(get(block));
  }
  void addSuccessors(const std::vector<PyBlock> &blocks) {
    for (const PyBlock &block : blocks)
      addSuccessor(block);
  }

  /// Creates the operation as `Py`, the generated class for it.
  template <typename Py>
  Py create(bool inferResultTypes, bool operandSegments, bool resultSegments) {
    if (operandSegments)
      state_.addAttribute("operandSegmentSizes",
                          builder_.getDenseI32ArrayAttr(operandSegments_));
    if (resultSegments)
      state_.addAttribute("resultSegmentSizes",
                          builder_.getDenseI32ArrayAttr(resultSegments_));
    Py result;
    static_cast<PyOperation &>(result) =
        createOperation(context_, state_, inferResultTypes, ip_);
    runPostCreateHook(result.op);
    return result;
  }

private:
  static mlir::OperationName loadOperationName(const ContextHandle &context,
                                               const char *name) {
    std::string full = name;
    context->context.getOrLoadDialect(full.substr(0, full.find('.')));
    return mlir::OperationName(full, &context->context);
  }

  ContextHandle context_;
  mlir::Builder builder_;
  mlir::OperationState state_;
  std::optional<PyInsertionPoint> ip_;
  std::vector<int32_t> operandSegments_;
  std::vector<int32_t> resultSegments_;
};

//===----------------------------------------------------------------------===//
// Accessors
//===----------------------------------------------------------------------===//

inline mlir::Operation *checkedOp(const PyOperation &self) { return get(self); }

inline ValueObject singleValue(const PyOperation &self,
                               mlir::Operation::operand_range values) {
  return wrapValue(self.tree, values.front());
}
inline nb::typed<nb::object, std::optional<PyValue>>
optionalValue(const PyOperation &self, mlir::Operation::operand_range values) {
  if (values.empty())
    return nb::none();
  return wrapValue(self.tree, values.front());
}
inline std::vector<ValueObject> valueList(const PyOperation &self,
                                          mlir::Operation::operand_range values) {
  std::vector<ValueObject> result;
  for (mlir::Value value : values)
    result.push_back(wrapValue(self.tree, value));
  return result;
}

/// Splits a variadic-of-variadic operand into its groups, whose sizes are in
/// the attribute `sizesAttribute`.
inline std::vector<std::vector<ValueObject>>
valueGroups(const PyOperation &self, mlir::Operation::operand_range values,
            const char *sizesAttribute) {
  std::vector<std::vector<ValueObject>> groups;
  auto sizes = llvm::dyn_cast_or_null<mlir::DenseI32ArrayAttr>(
      checkedOp(self)->getAttr(sizesAttribute));
  if (!sizes)
    return groups;
  auto it = values.begin();
  for (int32_t size : sizes.asArrayRef()) {
    std::vector<ValueObject> group;
    for (int32_t i = 0; i < size && it != values.end(); ++i, ++it)
      group.push_back(wrapValue(self.tree, *it));
    groups.push_back(std::move(group));
  }
  return groups;
}

inline PyOpResult singleResult(const PyOperation &self,
                               mlir::Operation::result_range results) {
  return PyOpResult{{self.tree, results.front()}};
}
inline std::optional<PyOpResult>
optionalResult(const PyOperation &self, mlir::Operation::result_range results) {
  if (results.empty())
    return std::nullopt;
  return PyOpResult{{self.tree, results.front()}};
}
inline std::vector<PyOpResult> resultList(const PyOperation &self,
                                          mlir::Operation::result_range results) {
  std::vector<PyOpResult> list;
  for (mlir::OpResult result : results)
    list.push_back(PyOpResult{{self.tree, result}});
  return list;
}

inline PyRegion regionAt(const PyOperation &self, unsigned index) {
  return PyRegion{self.tree, &checkedOp(self)->getRegion(index)};
}
inline std::vector<PyRegion> regionsFrom(const PyOperation &self,
                                         unsigned first) {
  std::vector<PyRegion> list;
  mlir::Operation *op = checkedOp(self);
  for (unsigned i = first; i < op->getNumRegions(); ++i)
    list.push_back(PyRegion{self.tree, &op->getRegion(i)});
  return list;
}

inline PyBlock successorAt(const PyOperation &self, unsigned index) {
  return PyBlock{self.tree, checkedOp(self)->getSuccessor(index)};
}
inline std::vector<PyBlock> successorsFrom(const PyOperation &self,
                                           unsigned first) {
  std::vector<PyBlock> list;
  mlir::Operation *op = checkedOp(self);
  for (unsigned i = first; i < op->getNumSuccessors(); ++i)
    list.push_back(PyBlock{self.tree, op->getSuccessor(i)});
  return list;
}

/// The inherent attribute `name`, or null when absent.
inline mlir::Attribute inherentAttr(const PyOperation &self, const char *name) {
  return checkedOp(self)->getAttr(name);
}

/// Replaces (or with a null value, removes) inherent attribute `name`.
inline void setInherentAttr(const PyOperation &self, const char *name,
                            mlir::Attribute value) {
  mlir::Operation *op = checkedOp(self);
  if (value)
    op->setAttr(name, value);
  else
    op->removeAttr(name);
}

//===----------------------------------------------------------------------===//
// Conversions used in generated parameter and accessor code
//===----------------------------------------------------------------------===//

template <typename Py>
Py makeTypeHandle(const ContextHandle &context, mlir::Type type) {
  Py result;
  result.context = context;
  result.type = type;
  return result;
}

template <typename Py>
Py makeAttributeHandle(const ContextHandle &context, mlir::Attribute attr) {
  Py result;
  result.context = context;
  result.attribute = attr;
  return result;
}

/// Checks that `type` is a `T`, naming the parameter in the error.
template <typename T>
T expectType(const ContextHandle &context, const PyType &type,
             const char *parameter, const char *expected) {
  requireSameContext(context, type.context, parameter);
  auto result = llvm::dyn_cast<T>(type.type);
  if (!result)
    throw nb::type_error((std::string(parameter) + " must be " + expected +
                          ", got " + printToString(type.type))
                             .c_str());
  return result;
}

/// Checks that `attr` is a `T`, naming the parameter in the error.
template <typename T>
T expectAttribute(const ContextHandle &context, const PyAttribute &attr,
                  const char *parameter, const char *expected) {
  requireSameContext(context, attr.context, parameter);
  auto result = llvm::dyn_cast<T>(attr.attribute);
  if (!result)
    throw nb::type_error((std::string(parameter) + " must be " + expected +
                          ", got " + printToString(attr.attribute))
                             .c_str());
  return result;
}

/// Converts a Python float to the semantics of `type` (a float type).
inline llvm::APFloat toAPFloat(double value, mlir::Type type) {
  llvm::APFloat result(value);
  bool losesInfo = false;
  result.convert(llvm::cast<mlir::FloatType>(type).getFloatSemantics(),
                 llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return result;
}

template <typename T> std::vector<T> toVector(llvm::ArrayRef<T> values) {
  return std::vector<T>(values.begin(), values.end());
}

} // namespace mlir_python
