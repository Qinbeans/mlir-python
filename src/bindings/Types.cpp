// Bindings for mlir::Type and the builtin type hierarchy.
#include "Core.h"

#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace mlir_python {

using namespace nb::literals;

namespace {

template <typename T> struct PySpecificFloatType : PyFloatType {};

// (C++ type, Python name, textual form) for every builtin float type.
#define MLIR_PYTHON_FLOAT_TYPES(X)                                             \
  X(BFloat16Type, "BF16Type", "bf16")                                          \
  X(Float16Type, "F16Type", "f16")                                             \
  X(FloatTF32Type, "TF32Type", "tf32")                                         \
  X(Float32Type, "F32Type", "f32")                                             \
  X(Float64Type, "F64Type", "f64")                                             \
  X(Float80Type, "F80Type", "f80")                                             \
  X(Float128Type, "F128Type", "f128")                                          \
  X(Float8E5M2Type, "Float8E5M2Type", "f8E5M2")                                \
  X(Float8E4M3Type, "Float8E4M3Type", "f8E4M3")                                \
  X(Float8E4M3FNType, "Float8E4M3FNType", "f8E4M3FN")                          \
  X(Float8E5M2FNUZType, "Float8E5M2FNUZType", "f8E5M2FNUZ")                    \
  X(Float8E4M3FNUZType, "Float8E4M3FNUZType", "f8E4M3FNUZ")                    \
  X(Float8E4M3B11FNUZType, "Float8E4M3B11FNUZType", "f8E4M3B11FNUZ")          \
  X(Float8E3M4Type, "Float8E3M4Type", "f8E3M4")                                \
  X(Float4E2M1FNType, "Float4E2M1FNType", "f4E2M1FN")                          \
  X(Float6E2M3FNType, "Float6E2M3FNType", "f6E2M3FN")                          \
  X(Float6E3M2FNType, "Float6E3M2FNType", "f6E3M2FN")                          \
  X(Float8E8M0FNUType, "Float8E8M0FNUType", "f8E8M0FNU")

template <typename Py, typename T>
Py make(const ContextHandle &context, T type) {
  Py result;
  result.context = context;
  result.type = type;
  return result;
}

/// Calls `getChecked` and turns verifier diagnostics into an MLIRError.
template <typename T, typename... Args>
T getChecked(const ContextHandle &context, const char *what, Args &&...args) {
  DiagnosticCapture capture(&context->context);
  T type = T::getChecked(
      mlir::detail::getDefaultDiagnosticEmitFn(&context->context),
      std::forward<Args>(args)...);
  if (!type)
    capture.raise(std::string("invalid ") + what);
  return type;
}

mlir::Type unwrap(const ContextHandle &context, const PyType &type,
                  const char *what = "type") {
  requireSameContext(context, type.context, what);
  return type.type;
}

std::vector<mlir::Type> unwrapAll(const ContextHandle &context,
                                  const std::vector<PyType> &types) {
  std::vector<mlir::Type> result;
  result.reserve(types.size());
  for (const PyType &type : types)
    result.push_back(unwrap(context, type));
  return result;
}

std::vector<TypeObject> wrapAll(const ContextHandle &context,
                                mlir::TypeRange types) {
  std::vector<TypeObject> result;
  for (mlir::Type type : types)
    result.push_back(wrapType(context, type));
  return result;
}

mlir::Attribute unwrapOptional(const ContextHandle &context,
                               const std::optional<PyAttribute> &attr,
                               const char *what) {
  if (!attr)
    return {};
  requireSameContext(context, attr->context, what);
  return attr->attribute;
}

std::vector<int64_t> toNativeShape(const std::vector<std::optional<int64_t>> &shape) {
  std::vector<int64_t> result;
  result.reserve(shape.size());
  for (const std::optional<int64_t> &dim : shape) {
    if (dim && *dim < 0)
      throw std::invalid_argument(
          "dimension sizes must be non-negative; use None for dynamic");
    result.push_back(dim ? *dim : mlir::ShapedType::kDynamic);
  }
  return result;
}

std::vector<std::optional<int64_t>> toPythonShape(llvm::ArrayRef<int64_t> shape) {
  std::vector<std::optional<int64_t>> result;
  for (int64_t dim : shape)
    result.push_back(mlir::ShapedType::isDynamic(dim)
                         ? std::nullopt
                         : std::optional<int64_t>(dim));
  return result;
}

mlir::ShapedType shaped(const PyShapedType &self) {
  return llvm::cast<mlir::ShapedType>(self.type);
}

mlir::ShapedType ranked(const PyShapedType &self) {
  auto type = shaped(self);
  if (!type.hasRank())
    throw std::invalid_argument("unranked type has no shape");
  return type;
}

std::string reprWithClass(nb::handle self, const std::string &body) {
  return nb::cast<std::string>(self.type().attr("__name__")) + "(" + body + ")";
}

constexpr const char *kContextDoc =
    "context: Context for the type; defaults to the current context.";

} // namespace

namespace {
llvm::DenseMap<mlir::TypeID, TypeFactory> &typeClasses() {
  static llvm::DenseMap<mlir::TypeID, TypeFactory> classes;
  return classes;
}
} // namespace

void registerTypeClass(mlir::TypeID type, TypeFactory factory) {
  typeClasses()[type] = factory;
}

TypeObject wrapType(const ContextHandle &context, mlir::Type type) {
  if (!type)
    throw std::invalid_argument("null type");
  auto registered = typeClasses().find(type.getTypeID());
  if (registered != typeClasses().end())
    return registered->second(context, type);
#define WRAP_FLOAT(CppType, PyName, Text)                                      \
  if (llvm::isa<mlir::CppType>(type))                                          \
    return nb::cast(make<PySpecificFloatType<mlir::CppType>>(context, type));
  MLIR_PYTHON_FLOAT_TYPES(WRAP_FLOAT)
#undef WRAP_FLOAT
  if (llvm::isa<mlir::IntegerType>(type))
    return nb::cast(make<PyIntegerType>(context, type));
  if (llvm::isa<mlir::IndexType>(type))
    return nb::cast(make<PyIndexType>(context, type));
  if (llvm::isa<mlir::NoneType>(type))
    return nb::cast(make<PyNoneType>(context, type));
  if (llvm::isa<mlir::FloatType>(type))
    return nb::cast(make<PyFloatType>(context, type));
  if (llvm::isa<mlir::ComplexType>(type))
    return nb::cast(make<PyComplexType>(context, type));
  if (llvm::isa<mlir::FunctionType>(type))
    return nb::cast(make<PyFunctionType>(context, type));
  if (llvm::isa<mlir::TupleType>(type))
    return nb::cast(make<PyTupleType>(context, type));
  if (llvm::isa<mlir::OpaqueType>(type))
    return nb::cast(make<PyOpaqueType>(context, type));
  if (llvm::isa<mlir::RankedTensorType>(type))
    return nb::cast(make<PyRankedTensorType>(context, type));
  if (llvm::isa<mlir::UnrankedTensorType>(type))
    return nb::cast(make<PyUnrankedTensorType>(context, type));
  if (llvm::isa<mlir::VectorType>(type))
    return nb::cast(make<PyVectorType>(context, type));
  if (llvm::isa<mlir::MemRefType>(type))
    return nb::cast(make<PyMemRefType>(context, type));
  if (llvm::isa<mlir::UnrankedMemRefType>(type))
    return nb::cast(make<PyUnrankedMemRefType>(context, type));
  if (llvm::isa<mlir::ShapedType>(type))
    return nb::cast(make<PyShapedType>(context, type));
  return nb::cast(make<PyType>(context, type));
}

OptionalTypeObject wrapOptionalType(const ContextHandle &context,
                                    mlir::Type type) {
  return type ? nb::object(wrapType(context, type)) : nb::none();
}

void bindTypes(nb::module_ &m) {
  nb::class_<PyType>(
      m, "Type",
      "Base class of every MLIR type.\n\n"
      "Types are immutable and uniqued in their context: two types are equal\n"
      "exactly when they print the same. Objects returned by the API are\n"
      "always the most specific subclass (e.g. ``IntegerType``), so narrow\n"
      "with ``isinstance``.")
      .def_static(
          "parse",
          [](const std::string &source, std::optional<PyContext> context) {
            ContextHandle state = resolveContext(context);
            DiagnosticCapture capture(&state->context);
            mlir::Type type = mlir::parseType(source, &state->context);
            if (!type)
              capture.raise("failed to parse type '" + source + "'");
            return wrapType(state, type);
          },
          "source"_a, nb::kw_only(), "context"_a = nb::none(),
          "Parse a type from its textual form, e.g. ``\"tensor<?xf32>\"``.\n\n"
          "Raises:\n"
          "    MLIRError: If ``source`` is not a valid type.")
      .def_prop_ro(
          "context", [](const PyType &self) { return PyContext{self.context}; },
          "The context this type belongs to.")
      .def("__eq__", identityEq<PyType>([](const PyType &v) -> const void * { return v.type.getAsOpaquePointer(); }),
           "other"_a.none(), nb::sig(kEqSignature))
      .def("__hash__", identityHash<PyType>([](const PyType &v) -> const void * { return v.type.getAsOpaquePointer(); }))
      .def("__str__", [](const PyType &self) { return printToString(self.type); })
      .def("__repr__", [](nb::handle self) {
        return reprWithClass(self, printToString(nb::cast<PyType &>(self).type));
      });

  nb::enum_<mlir::IntegerType::SignednessSemantics>(
      m, "Signedness", "How an ``IntegerType`` interprets its bits.")
      .value("SIGNLESS", mlir::IntegerType::Signless,
             "No signedness; operations decide (``i32``). The common case.")
      .value("SIGNED", mlir::IntegerType::Signed, "Signed (``si32``).")
      .value("UNSIGNED", mlir::IntegerType::Unsigned, "Unsigned (``ui32``).");

  nb::class_<PyIntegerType, PyType>(m, "IntegerType",
                                    "A fixed-width integer type such as "
                                    "``i32``, ``si8``, or ``ui64``.")
      .def(nb::new_([](unsigned width,
                       mlir::IntegerType::SignednessSemantics signedness,
                       std::optional<PyContext> context) {
             ContextHandle state = resolveContext(context);
             return make<PyIntegerType>(
                 state, getChecked<mlir::IntegerType>(
                            state, "integer type", &state->context, width,
                            signedness));
           }),
           "width"_a, nb::kw_only(),
           "signedness"_a = mlir::IntegerType::Signless,
           "context"_a = nb::none(),
           (std::string("Create an integer type of ``width`` bits.\n\nArgs:\n"
                        "    width: Bit width, 1 to 16777215.\n"
                        "    signedness: Signless by default.\n    ") +
            kContextDoc)
               .c_str())
      .def_prop_ro(
          "width",
          [](const PyIntegerType &self) {
            return llvm::cast<mlir::IntegerType>(self.type).getWidth();
          },
          "Bit width.")
      .def_prop_ro(
          "signedness",
          [](const PyIntegerType &self) {
            return llvm::cast<mlir::IntegerType>(self.type).getSignedness();
          },
          "Signedness semantics.")
      .def_prop_ro(
          "is_signless",
          [](const PyIntegerType &self) {
            return llvm::cast<mlir::IntegerType>(self.type).isSignless();
          },
          "Whether the type is signless (``iN``).")
      .def_prop_ro(
          "is_signed",
          [](const PyIntegerType &self) {
            return llvm::cast<mlir::IntegerType>(self.type).isSigned();
          },
          "Whether the type is signed (``siN``).")
      .def_prop_ro(
          "is_unsigned",
          [](const PyIntegerType &self) {
            return llvm::cast<mlir::IntegerType>(self.type).isUnsigned();
          },
          "Whether the type is unsigned (``uiN``).");

  nb::class_<PyIndexType, PyType>(
      m, "IndexType",
      "The target-width integer type used for sizes and indices (``index``).")
      .def(nb::new_([](std::optional<PyContext> context) {
             ContextHandle state = resolveContext(context);
             return make<PyIndexType>(state,
                                      mlir::IndexType::get(&state->context));
           }),
           nb::kw_only(), "context"_a = nb::none(), "Create ``index``.");

  nb::class_<PyNoneType, PyType>(m, "NoneType",
                                 "The unit type ``none``.")
      .def(nb::new_([](std::optional<PyContext> context) {
             ContextHandle state = resolveContext(context);
             return make<PyNoneType>(state,
                                     mlir::NoneType::get(&state->context));
           }),
           nb::kw_only(), "context"_a = nb::none(), "Create ``none``.");

  nb::class_<PyFloatType, PyType>(
      m, "FloatType",
      "Base class of the floating-point types. Construct a specific\n"
      "subclass such as ``F32Type``.")
      .def_prop_ro(
          "width",
          [](const PyFloatType &self) {
            return llvm::cast<mlir::FloatType>(self.type).getWidth();
          },
          "Bit width.")
      .def_prop_ro(
          "mantissa_width",
          [](const PyFloatType &self) {
            return llvm::cast<mlir::FloatType>(self.type).getFPMantissaWidth();
          },
          "Width of the significand, including the implicit bit.");

#define BIND_FLOAT(CppType, PyName, Text)                                      \
  nb::class_<PySpecificFloatType<mlir::CppType>, PyFloatType>(                 \
      m, PyName, "The ``" Text "`` floating-point type.")                      \
      .def(nb::new_([](std::optional<PyContext> context) {                     \
             ContextHandle state = resolveContext(context);                    \
             return make<PySpecificFloatType<mlir::CppType>>(                  \
                 state, mlir::CppType::get(&state->context));                  \
           }),                                                                 \
           nb::kw_only(), "context"_a = nb::none(), "Create ``" Text "``.");
  MLIR_PYTHON_FLOAT_TYPES(BIND_FLOAT)
#undef BIND_FLOAT

  nb::class_<PyComplexType, PyType>(
      m, "ComplexType", "A complex number type, e.g. ``complex<f32>``.")
      .def(nb::new_([](const PyType &elementType) {
             return make<PyComplexType>(
                 elementType.context,
                 getChecked<mlir::ComplexType>(elementType.context,
                                               "complex type",
                                               elementType.type));
           }),
           "element_type"_a,
           "Create ``complex<element_type>``. The element type must be an\n"
           "integer or floating-point type.")
      .def_prop_ro(
          "element_type",
          [](const PyComplexType &self) {
            return wrapType(
                self.context,
                llvm::cast<mlir::ComplexType>(self.type).getElementType());
          },
          "Type of the real and imaginary parts.");

  nb::class_<PyFunctionType, PyType>(
      m, "FunctionType", "A function signature, e.g. ``(i32, f32) -> i1``.")
      .def(nb::new_([](const std::vector<PyType> &inputs,
                       const std::vector<PyType> &results,
                       std::optional<PyContext> context) {
             ContextHandle state =
                 context ? context->state
                 : !inputs.empty()  ? inputs.front().context
                 : !results.empty() ? results.front().context
                                    : resolveContext(std::nullopt);
             return make<PyFunctionType>(
                 state, mlir::FunctionType::get(&state->context,
                                                unwrapAll(state, inputs),
                                                unwrapAll(state, results)));
           }),
           "inputs"_a, "results"_a, nb::kw_only(), "context"_a = nb::none(),
           "Create ``(inputs) -> results``.")
      .def_prop_ro(
          "inputs",
          [](const PyFunctionType &self) {
            return wrapAll(self.context,
                           llvm::cast<mlir::FunctionType>(self.type).getInputs());
          }, "Argument types.")
      .def_prop_ro(
          "results",
          [](const PyFunctionType &self) {
            return wrapAll(
                self.context,
                llvm::cast<mlir::FunctionType>(self.type).getResults());
          }, "Result types.");

  nb::class_<PyTupleType, PyType>(m, "TupleType",
                                  "A fixed-size tuple of types, e.g. "
                                  "``tuple<i32, f32>``.")
      .def(nb::new_([](const std::vector<PyType> &types,
                       std::optional<PyContext> context) {
             ContextHandle state = context ? context->state
                                   : !types.empty() ? types.front().context
                                                    : resolveContext(std::nullopt);
             return make<PyTupleType>(
                 state, mlir::TupleType::get(&state->context,
                                             unwrapAll(state, types)));
           }),
           "types"_a, nb::kw_only(), "context"_a = nb::none(),
           "Create ``tuple<types...>``.")
      .def_prop_ro(
          "types",
          [](const PyTupleType &self) {
            return wrapAll(self.context,
                           llvm::cast<mlir::TupleType>(self.type).getTypes());
          }, "Element types.")
      .def("__len__", [](const PyTupleType &self) {
        return llvm::cast<mlir::TupleType>(self.type).size();
      });

  nb::class_<PyOpaqueType, PyType>(
      m, "OpaqueType",
      "A type from an unregistered dialect, kept as uninterpreted text\n"
      "(``!dialect.data``).")
      .def(nb::new_([](const std::string &dialect, const std::string &data,
                       std::optional<PyContext> context) {
             ContextHandle state = resolveContext(context);
             auto name = mlir::StringAttr::get(&state->context, dialect);
             return make<PyOpaqueType>(
                 state, getChecked<mlir::OpaqueType>(state, "opaque type",
                                                     name, data));
           }),
           "dialect"_a, "data"_a, nb::kw_only(), "context"_a = nb::none(),
           "Create ``!dialect.data``, e.g. ``OpaqueType(\"foo\", \"bar<1>\")``\n"
           "is ``!foo.bar<1>``.")
      .def_prop_ro(
          "dialect",
          [](const PyOpaqueType &self) {
            return llvm::cast<mlir::OpaqueType>(self.type)
                .getDialectNamespace()
                .str();
          },
          "The dialect namespace.")
      .def_prop_ro(
          "data",
          [](const PyOpaqueType &self) {
            return llvm::cast<mlir::OpaqueType>(self.type).getTypeData().str();
          },
          "The uninterpreted type body.");

  nb::class_<PyShapedType, PyType>(
      m, "ShapedType",
      "Base class of tensor, vector, and memref types.\n\n"
      "Shapes are lists of ``int | None``, where ``None`` is a dynamic\n"
      "dimension (``?`` in the textual form).")
      .def_prop_ro(
          "element_type",
          [](const PyShapedType &self) {
            return wrapType(self.context, shaped(self).getElementType());
          }, "The element type.")
      .def_prop_ro(
          "has_rank",
          [](const PyShapedType &self) { return shaped(self).hasRank(); },
          "Whether the number of dimensions is known.")
      .def_prop_ro(
          "rank", [](const PyShapedType &self) { return ranked(self).getRank(); },
          "Number of dimensions.\n\nRaises:\n    ValueError: If unranked.")
      .def_prop_ro(
          "shape",
          [](const PyShapedType &self) {
            return toPythonShape(ranked(self).getShape());
          },
          "Dimension sizes, ``None`` for dynamic ones.\n\n"
          "Raises:\n    ValueError: If unranked.")
      .def_prop_ro(
          "has_static_shape",
          [](const PyShapedType &self) {
            return shaped(self).hasStaticShape();
          },
          "Whether the type is ranked and every dimension is static.")
      .def_prop_ro(
          "num_elements",
          [](const PyShapedType &self) -> std::optional<int64_t> {
            auto type = shaped(self);
            if (!type.hasStaticShape())
              return std::nullopt;
            return type.getNumElements();
          },
          "Total element count, or ``None`` unless the shape is static.");

  nb::class_<PyRankedTensorType, PyShapedType>(
      m, "RankedTensorType", "A tensor of known rank, e.g. ``tensor<4x?xf32>``.")
      .def(nb::new_([](const std::vector<std::optional<int64_t>> &shape,
                       const PyType &elementType,
                       std::optional<PyAttribute> encoding) {
             const ContextHandle &state = elementType.context;
             return make<PyRankedTensorType>(
                 state, getChecked<mlir::RankedTensorType>(
                            state, "tensor type", toNativeShape(shape),
                            elementType.type,
                            unwrapOptional(state, encoding, "encoding")));
           }),
           "shape"_a, "element_type"_a, "encoding"_a = nb::none(),
           "Create ``tensor<shape x element_type>``.\n\n"
           "Args:\n"
           "    shape: Dimension sizes; ``None`` marks a dynamic dimension.\n"
           "    element_type: The element type.\n"
           "    encoding: Optional encoding attribute (e.g. a sparse layout).")
      .def_prop_ro(
          "encoding",
          [](const PyRankedTensorType &self) {
            return wrapOptionalAttribute(
                self.context,
                llvm::cast<mlir::RankedTensorType>(self.type).getEncoding());
          },
          "The encoding attribute, if any.");

  nb::class_<PyUnrankedTensorType, PyShapedType>(
      m, "UnrankedTensorType", "A tensor of unknown rank, e.g. ``tensor<*xf32>``.")
      .def(nb::new_([](const PyType &elementType) {
             return make<PyUnrankedTensorType>(
                 elementType.context,
                 getChecked<mlir::UnrankedTensorType>(
                     elementType.context, "tensor type", elementType.type));
           }),
           "element_type"_a, "Create ``tensor<*xelement_type>``.");

  nb::class_<PyVectorType, PyShapedType>(
      m, "VectorType",
      "A fixed-shape SIMD vector, e.g. ``vector<4xf32>`` or the scalable\n"
      "``vector<[4]xf32>``.")
      .def(nb::new_([](const std::vector<int64_t> &shape,
                       const PyType &elementType,
                       std::optional<std::vector<bool>> scalable) {
             const ContextHandle &state = elementType.context;
             std::vector<bool> scalableDims =
                 scalable ? *scalable : std::vector<bool>(shape.size(), false);
             if (scalableDims.size() != shape.size())
               throw std::invalid_argument(
                   "scalable must have one entry per dimension");
             llvm::SmallVector<bool> native(scalableDims.begin(),
                                            scalableDims.end());
             return make<PyVectorType>(
                 state, getChecked<mlir::VectorType>(
                            state, "vector type", shape, elementType.type,
                            llvm::ArrayRef<bool>(native)));
           }),
           "shape"_a, "element_type"_a, nb::kw_only(),
           "scalable"_a = nb::none(),
           "Create ``vector<shape x element_type>``.\n\n"
           "Args:\n"
           "    shape: Static, positive dimension sizes.\n"
           "    element_type: An integer, index, or float type.\n"
           "    scalable: Per-dimension scalability flags (all ``False`` by\n"
           "        default).")
      .def_prop_ro(
          "scalable_dims",
          [](const PyVectorType &self) {
            auto dims = llvm::cast<mlir::VectorType>(self.type).getScalableDims();
            return std::vector<bool>(dims.begin(), dims.end());
          },
          "Per-dimension scalability flags.")
      .def_prop_ro(
          "is_scalable",
          [](const PyVectorType &self) {
            return llvm::cast<mlir::VectorType>(self.type).isScalable();
          },
          "Whether any dimension is scalable.");

  nb::class_<PyMemRefType, PyShapedType>(
      m, "MemRefType",
      "A ranked reference to a region of memory, e.g. ``memref<4x?xf32>``.")
      .def(nb::new_([](const std::vector<std::optional<int64_t>> &shape,
                       const PyType &elementType,
                       std::optional<PyAttribute> layout,
                       std::optional<PyAttribute> memorySpace) {
             const ContextHandle &state = elementType.context;
             mlir::MemRefLayoutAttrInterface nativeLayout;
             if (mlir::Attribute attr =
                     unwrapOptional(state, layout, "layout")) {
               nativeLayout = llvm::dyn_cast<mlir::MemRefLayoutAttrInterface>(attr);
               if (!nativeLayout)
                 throw std::invalid_argument(
                     "layout must be a memref layout attribute");
             }
             return make<PyMemRefType>(
                 state, getChecked<mlir::MemRefType>(
                            state, "memref type", toNativeShape(shape),
                            elementType.type, nativeLayout,
                            unwrapOptional(state, memorySpace, "memory space")));
           }),
           "shape"_a, "element_type"_a, nb::kw_only(), "layout"_a = nb::none(),
           "memory_space"_a = nb::none(),
           "Create ``memref<shape x element_type, layout, memory_space>``.\n\n"
           "Args:\n"
           "    shape: Dimension sizes; ``None`` marks a dynamic dimension.\n"
           "    element_type: The element type.\n"
           "    layout: A layout attribute such as an affine map or strided\n"
           "        layout; identity when omitted.\n"
           "    memory_space: Optional memory-space attribute.")
      .def_prop_ro(
          "layout",
          [](const PyMemRefType &self) {
            return wrapAttribute(self.context,
                                 llvm::cast<mlir::MemRefType>(self.type).getLayout());
          }, "The layout attribute.")
      .def_prop_ro(
          "memory_space",
          [](const PyMemRefType &self) {
            return wrapOptionalAttribute(
                self.context,
                llvm::cast<mlir::MemRefType>(self.type).getMemorySpace());
          },
          "The memory space, or ``None`` for the default one.");

  nb::class_<PyUnrankedMemRefType, PyShapedType>(
      m, "UnrankedMemRefType",
      "A memref of unknown rank, e.g. ``memref<*xf32>``.")
      .def(nb::new_([](const PyType &elementType,
                       std::optional<PyAttribute> memorySpace) {
             const ContextHandle &state = elementType.context;
             return make<PyUnrankedMemRefType>(
                 state, getChecked<mlir::UnrankedMemRefType>(
                            state, "memref type", elementType.type,
                            unwrapOptional(state, memorySpace, "memory space")));
           }),
           "element_type"_a, nb::kw_only(), "memory_space"_a = nb::none(),
           "Create ``memref<*xelement_type>``.")
      .def_prop_ro(
          "memory_space",
          [](const PyUnrankedMemRefType &self) {
            return wrapOptionalAttribute(
                self.context,
                llvm::cast<mlir::UnrankedMemRefType>(self.type).getMemorySpace());
          },
          "The memory space, or ``None`` for the default one.");
}

} // namespace mlir_python
