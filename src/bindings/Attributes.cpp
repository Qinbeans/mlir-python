// Bindings for mlir::Attribute and the builtin attribute hierarchy.
#include "Core.h"
#include "DialectSupport.h"

#include <map>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallString.h>
#include <mlir/AsmParser/AsmParser.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace mlir_python {

using namespace nb::literals;

namespace {

template <typename T> struct PySpecificDenseArrayAttr : PyDenseArrayAttr {};

// (C++ attribute, C++ element, Python name, Python element type, text).
#define MLIR_PYTHON_DENSE_ARRAYS(X)                                            \
  X(DenseBoolArrayAttr, bool, "DenseBoolArrayAttr", "bool", "i1")              \
  X(DenseI8ArrayAttr, int8_t, "DenseI8ArrayAttr", "int", "i8")                 \
  X(DenseI16ArrayAttr, int16_t, "DenseI16ArrayAttr", "int", "i16")             \
  X(DenseI32ArrayAttr, int32_t, "DenseI32ArrayAttr", "int", "i32")             \
  X(DenseI64ArrayAttr, int64_t, "DenseI64ArrayAttr", "int", "i64")             \
  X(DenseF32ArrayAttr, float, "DenseF32ArrayAttr", "float", "f32")             \
  X(DenseF64ArrayAttr, double, "DenseF64ArrayAttr", "float", "f64")

template <typename Py>
Py make(const ContextHandle &context, mlir::Attribute attr) {
  Py result;
  result.context = context;
  result.attribute = attr;
  return result;
}

template <typename T> T as(const PyAttribute &self) {
  return llvm::cast<T>(self.attribute);
}

//===--------------------------------------------------------------------===//
// Python int <-> APInt
//===--------------------------------------------------------------------===//

/// Converts a Python int to an APInt of `width` bits, rejecting values that
/// do not fit. Signless integers accept both signed and unsigned ranges.
llvm::APInt toAPInt(const nb::int_ &value, unsigned width,
                    mlir::IntegerType::SignednessSemantics signedness) {
  std::string text = nb::cast<std::string>(nb::str(value));
  bool negative = !text.empty() && text[0] == '-';
  unsigned bits =
      std::max(llvm::APInt::getBitsNeeded(text, 10), width) + 1;
  llvm::APInt wide(bits, text, 10);
  bool fits = false;
  switch (signedness) {
  case mlir::IntegerType::Signed:
    fits = wide.getSignificantBits() <= width;
    break;
  case mlir::IntegerType::Unsigned:
    fits = !negative && wide.getActiveBits() <= width;
    break;
  case mlir::IntegerType::Signless:
    fits = negative ? wide.getSignificantBits() <= width
                    : wide.getActiveBits() <= width;
    break;
  }
  if (!fits)
    throw std::overflow_error(text + " does not fit in " +
                              std::to_string(width) + " bits");
  return wide.trunc(width);
}

} // namespace

nb::int_ fromAPInt(const llvm::APInt &value, bool isSigned) {
  llvm::SmallString<32> text;
  value.toString(text, 10, isSigned);
  return nb::int_(nb::str(text.data(), text.size()));
}

namespace {

/// Width and signedness of an integer or index type.
std::pair<unsigned, mlir::IntegerType::SignednessSemantics>
integerTraits(mlir::Type type) {
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
    return {integer.getWidth(), integer.getSignedness()};
  if (llvm::isa<mlir::IndexType>(type))
    return {mlir::IndexType::kInternalStorageBitWidth,
            mlir::IntegerType::Signless};
  throw std::invalid_argument("expected an integer or index type, got " +
                              printToString(type));
}

} // namespace

/// Converts any float semantics to double, rounding if needed.
double toDouble(llvm::APFloat value) {
  bool losesInfo = false;
  value.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &losesInfo);
  return value.convertToDouble();
}

namespace {

bool isSignedForPython(mlir::Type type) {
  return integerTraits(type).second != mlir::IntegerType::Unsigned;
}

mlir::IntegerAttr makeIntegerAttr(mlir::Type type, const nb::int_ &value) {
  auto [width, signedness] = integerTraits(type);
  return mlir::IntegerAttr::get(type, toAPInt(value, width, signedness));
}

mlir::FloatAttr makeFloatAttr(mlir::Type type, double value) {
  if (!llvm::isa<mlir::FloatType>(type))
    throw std::invalid_argument("expected a floating-point type, got " +
                                printToString(type));
  return mlir::FloatAttr::get(type, value);
}

//===--------------------------------------------------------------------===//
// Helpers
//===--------------------------------------------------------------------===//

mlir::Attribute unwrap(const ContextHandle &context, const PyAttribute &attr) {
  requireSameContext(context, attr.context, "attribute");
  return attr.attribute;
}

ContextHandle contextOf(const std::optional<PyContext> &context,
                        const std::vector<PyAttribute> &elements) {
  if (context)
    return context->state;
  if (!elements.empty())
    return elements.front().context;
  return resolveContext(std::nullopt);
}

size_t normalizeIndex(int64_t index, size_t size) {
  int64_t signedSize = static_cast<int64_t>(size);
  if (index < 0)
    index += signedSize;
  if (index < 0 || index >= signedSize)
    throw nb::index_error("index out of range");
  return static_cast<size_t>(index);
}

/// The shaped type a dense elements attribute can be built for.
mlir::ShapedType denseShapedType(const PyType &type) {
  auto shaped = llvm::dyn_cast<mlir::ShapedType>(type.type);
  if (!shaped || !llvm::isa<mlir::RankedTensorType, mlir::VectorType>(shaped))
    throw std::invalid_argument(
        "dense elements need a ranked tensor or vector type");
  if (!shaped.hasStaticShape())
    throw std::invalid_argument("dense elements need a static shape");
  return shaped;
}

void checkElementCount(mlir::ShapedType type, size_t count) {
  if (count != 1 && static_cast<int64_t>(count) != type.getNumElements())
    throw std::invalid_argument(
        "expected " + std::to_string(type.getNumElements()) +
        " elements (or 1 to splat), got " + std::to_string(count));
}

std::vector<AttributeObject> wrapAll(const ContextHandle &context,
                                     llvm::ArrayRef<mlir::Attribute> attrs) {
  std::vector<AttributeObject> result;
  result.reserve(attrs.size());
  for (mlir::Attribute attr : attrs)
    result.push_back(wrapAttribute(context, attr));
  return result;
}

std::string reprWithClass(nb::handle self, const std::string &body) {
  return nb::cast<std::string>(self.type().attr("__name__")) + "(" + body + ")";
}

using AttributeIterator = nb::typed<nb::iterator, PyAttribute>;
using StringIterator = nb::typed<nb::iterator, std::string>;

} // namespace

AttributeObject wrapAttribute(const ContextHandle &context,
                              mlir::Attribute attr) {
  if (!attr)
    throw std::invalid_argument("null attribute");
  if (llvm::isa<mlir::BoolAttr>(attr))
    return nb::cast(make<PyBoolAttr>(context, attr));
  if (llvm::isa<mlir::IntegerAttr>(attr))
    return nb::cast(make<PyIntegerAttr>(context, attr));
  if (llvm::isa<mlir::FloatAttr>(attr))
    return nb::cast(make<PyFloatAttr>(context, attr));
  if (llvm::isa<mlir::StringAttr>(attr))
    return nb::cast(make<PyStringAttr>(context, attr));
  if (llvm::isa<mlir::UnitAttr>(attr))
    return nb::cast(make<PyUnitAttr>(context, attr));
  if (llvm::isa<mlir::TypeAttr>(attr))
    return nb::cast(make<PyTypeAttr>(context, attr));
  if (llvm::isa<mlir::ArrayAttr>(attr))
    return nb::cast(make<PyArrayAttr>(context, attr));
  if (llvm::isa<mlir::DictionaryAttr>(attr))
    return nb::cast(make<PyDictAttr>(context, attr));
  if (llvm::isa<mlir::FlatSymbolRefAttr>(attr))
    return nb::cast(make<PyFlatSymbolRefAttr>(context, attr));
  if (llvm::isa<mlir::SymbolRefAttr>(attr))
    return nb::cast(make<PySymbolRefAttr>(context, attr));
  if (llvm::isa<mlir::DenseIntElementsAttr>(attr))
    return nb::cast(make<PyDenseIntElementsAttr>(context, attr));
  if (llvm::isa<mlir::DenseFPElementsAttr>(attr))
    return nb::cast(make<PyDenseFPElementsAttr>(context, attr));
  if (llvm::isa<mlir::DenseElementsAttr>(attr))
    return nb::cast(make<PyDenseElementsAttr>(context, attr));
#define WRAP_ARRAY(CppAttr, Elt, PyName, PyElt, Text)                          \
  if (llvm::isa<mlir::CppAttr>(attr))                                          \
    return nb::cast(make<PySpecificDenseArrayAttr<mlir::CppAttr>>(context, attr));
  MLIR_PYTHON_DENSE_ARRAYS(WRAP_ARRAY)
#undef WRAP_ARRAY
  if (llvm::isa<mlir::StridedLayoutAttr>(attr))
    return nb::cast(make<PyStridedLayoutAttr>(context, attr));
  return nb::cast(make<PyAttribute>(context, attr));
}

OptionalAttributeObject wrapOptionalAttribute(const ContextHandle &context,
                                              mlir::Attribute attr) {
  return attr ? nb::object(wrapAttribute(context, attr)) : nb::none();
}

void bindAttributes(nb::module_ &m) {
  nb::class_<PyAttribute>(
      m, "Attribute",
      "Base class of every MLIR attribute: compile-time constant data such\n"
      "as numbers, strings, types, and arrays.\n\n"
      "Attributes are immutable and uniqued in their context. Objects\n"
      "returned by the API are always the most specific subclass.")
      .def_static(
          "parse",
          [](const std::string &source, std::optional<PyContext> context) {
            ContextHandle state = resolveContext(context);
            DiagnosticCapture capture(&state->context);
            mlir::Attribute attr = mlir::parseAttribute(source, &state->context);
            if (!attr)
              capture.raise("failed to parse attribute '" + source + "'");
            return wrapAttribute(state, attr);
          },
          "source"_a, nb::kw_only(), "context"_a = nb::none(),
          "Parse an attribute from its textual form, e.g. ``\"42 : i32\"``.\n\n"
          "Raises:\n"
          "    MLIRError: If ``source`` is not a valid attribute.")
      .def_prop_ro(
          "context",
          [](const PyAttribute &self) { return PyContext{self.context}; },
          "The context this attribute belongs to.")
      .def_prop_ro(
          "type",
          [](const PyAttribute &self) -> OptionalTypeObject {
            if (auto typed = llvm::dyn_cast<mlir::TypedAttr>(self.attribute))
              return wrapType(self.context, typed.getType());
            return nb::none();
          },
          "The attribute's type if it has one (e.g. ``i32`` for ``42 : i32``).")
      .def("__eq__", identityEq<PyAttribute>([](const PyAttribute &v) -> const void * { return v.attribute.getAsOpaquePointer(); }),
           "other"_a.none(), nb::sig(kEqSignature))
      .def("__hash__", identityHash<PyAttribute>([](const PyAttribute &v) -> const void * { return v.attribute.getAsOpaquePointer(); }))
      .def("__str__",
           [](const PyAttribute &self) { return printToString(self.attribute); })
      .def("__repr__", [](nb::handle self) {
        return reprWithClass(self,
                             printToString(nb::cast<PyAttribute &>(self).attribute));
      });

  nb::class_<PyIntegerAttr, PyAttribute>(
      m, "IntegerAttr", "An integer constant with an integer or index type.")
      .def(nb::new_([](const nb::int_ &value,
                       const std::optional<std::variant<PyIntegerType, PyIndexType>>
                           &type,
                       std::optional<PyContext> context) {
             std::optional<PyType> base;
             if (type)
               base = std::visit([](const auto &t) { return PyType(t); }, *type);
             ContextHandle state = base ? base->context : resolveContext(context);
             mlir::Type nativeType =
                 base ? base->type : mlir::IntegerType::get(&state->context, 64);
             return make<PyIntegerAttr>(state, makeIntegerAttr(nativeType, value));
           }),
           "value"_a, "type"_a = nb::none(), nb::kw_only(),
           "context"_a = nb::none(),
           "Create ``value : type``.\n\n"
           "Args:\n"
           "    value: Any Python int that fits in ``type``.\n"
           "    type: An ``IntegerType`` or ``IndexType``; ``i64`` if omitted.\n"
           "    context: Used only when ``type`` is omitted.\n\n"
           "Raises:\n"
           "    OverflowError: If ``value`` does not fit.")
      .def_prop_ro(
          "value",
          [](const PyIntegerAttr &self) {
            auto attr = as<mlir::IntegerAttr>(self);
            return fromAPInt(attr.getValue(), isSignedForPython(attr.getType()));
          },
          "The value. Signless integers are read as signed.");

  nb::class_<PyBoolAttr, PyIntegerAttr>(
      m, "BoolAttr", "A boolean constant (``true`` / ``false``), typed ``i1``.")
      .def(nb::new_([](bool value, std::optional<PyContext> context) {
             ContextHandle state = resolveContext(context);
             return make<PyBoolAttr>(state,
                                     mlir::BoolAttr::get(&state->context, value));
           }),
           "value"_a, nb::kw_only(), "context"_a = nb::none(),
           "Create ``true`` or ``false``.")
      .def_prop_ro(
          "value",
          [](const PyBoolAttr &self) { return as<mlir::BoolAttr>(self).getValue(); },
          "The value.")
      .def("__bool__", [](const PyBoolAttr &self) {
        return as<mlir::BoolAttr>(self).getValue();
      });

  nb::class_<PyFloatAttr, PyAttribute>(
      m, "FloatAttr", "A floating-point constant, e.g. ``1.5 : f32``.")
      .def(nb::new_([](double value, std::optional<PyFloatType> type,
                       std::optional<PyContext> context) {
             ContextHandle state = type ? type->context : resolveContext(context);
             mlir::Type nativeType =
                 type ? type->type : mlir::Float64Type::get(&state->context);
             return make<PyFloatAttr>(state, makeFloatAttr(nativeType, value));
           }),
           "value"_a, "type"_a = nb::none(), nb::kw_only(),
           "context"_a = nb::none(),
           "Create ``value : type``, rounding ``value`` to ``type``.\n\n"
           "Args:\n"
           "    value: The value.\n"
           "    type: A ``FloatType``; ``f64`` if omitted.\n"
           "    context: Used only when ``type`` is omitted.")
      .def_prop_ro(
          "value",
          [](const PyFloatAttr &self) {
            return as<mlir::FloatAttr>(self).getValueAsDouble();
          },
          "The value, converted to a Python float.");

  nb::class_<PyStringAttr, PyAttribute>(m, "StringAttr",
                                        "A string constant, e.g. ``\"hello\"``.")
      .def(nb::new_([](const std::string &value,
                       std::optional<PyContext> context) {
             ContextHandle state = resolveContext(context);
             return make<PyStringAttr>(
                 state, mlir::StringAttr::get(&state->context, value));
           }),
           "value"_a, nb::kw_only(), "context"_a = nb::none(),
           "Create a string attribute.")
      .def_prop_ro(
          "value",
          [](const PyStringAttr &self) {
            return as<mlir::StringAttr>(self).getValue().str();
          },
          "The string.");

  nb::class_<PyUnitAttr, PyAttribute>(
      m, "UnitAttr",
      "A marker attribute with no value; its presence is the information.")
      .def(nb::new_([](std::optional<PyContext> context) {
             ContextHandle state = resolveContext(context);
             return make<PyUnitAttr>(state, mlir::UnitAttr::get(&state->context));
           }),
           nb::kw_only(), "context"_a = nb::none(), "Create ``unit``.");

  nb::class_<PyTypeAttr, PyAttribute>(m, "TypeAttr",
                                      "An attribute holding a type.")
      .def(nb::new_([](const PyType &type) {
             return make<PyTypeAttr>(type.context, mlir::TypeAttr::get(type.type));
           }),
           "value"_a, "Wrap ``value`` as an attribute.")
      .def_prop_ro(
          "value",
          [](const PyTypeAttr &self) {
            return wrapType(self.context, as<mlir::TypeAttr>(self).getValue());
          },
          "The wrapped type.");

  nb::class_<PyArrayAttr, PyAttribute>(
      m, "ArrayAttr",
      "An ordered list of attributes, e.g. ``[1 : i32, \"x\"]``. Supports\n"
      "``len``, indexing, and iteration.")
      .def(nb::new_([](const std::vector<PyAttribute> &elements,
                       std::optional<PyContext> context) {
             ContextHandle state = contextOf(context, elements);
             std::vector<mlir::Attribute> native;
             for (const PyAttribute &element : elements)
               native.push_back(unwrap(state, element));
             return make<PyArrayAttr>(
                 state, mlir::ArrayAttr::get(&state->context, native));
           }),
           "elements"_a, nb::kw_only(), "context"_a = nb::none(),
           "Create an array. ``context`` is needed only when ``elements`` is\n"
           "empty and no context is active.")
      .def("__len__",
           [](const PyArrayAttr &self) { return as<mlir::ArrayAttr>(self).size(); })
      .def(
          "__getitem__",
          [](const PyArrayAttr &self, int64_t index) {
            auto attr = as<mlir::ArrayAttr>(self);
            return wrapAttribute(self.context,
                                 attr[normalizeIndex(index, attr.size())]);
          },
          "index"_a)
      .def("__iter__",
           [](const PyArrayAttr &self) -> AttributeIterator {
             return nb::iter(nb::cast(
                 wrapAll(self.context, as<mlir::ArrayAttr>(self).getValue())));
           })
      .def_prop_ro(
          "elements",
          [](const PyArrayAttr &self) {
            return wrapAll(self.context, as<mlir::ArrayAttr>(self).getValue());
          },
          "The elements as a list.");

  nb::class_<PyDictAttr, PyAttribute>(
      m, "DictAttr",
      "A string-keyed dictionary of attributes, sorted by key. Supports\n"
      "``len``, ``in``, indexing by key, and iteration over keys.")
      .def(nb::new_([](const std::map<std::string, PyAttribute> &elements,
                       std::optional<PyContext> context) {
             ContextHandle state =
                 context              ? context->state
                 : !elements.empty() ? elements.begin()->second.context
                                     : resolveContext(std::nullopt);
             std::vector<mlir::NamedAttribute> native;
             for (const auto &[name, attr] : elements)
               native.emplace_back(mlir::StringAttr::get(&state->context, name),
                                   unwrap(state, attr));
             return make<PyDictAttr>(
                 state, mlir::DictionaryAttr::get(&state->context, native));
           }),
           "elements"_a, nb::kw_only(), "context"_a = nb::none(),
           "Create a dictionary. ``context`` is needed only when ``elements``\n"
           "is empty and no context is active.")
      .def("__len__", [](const PyDictAttr &self) {
        return as<mlir::DictionaryAttr>(self).size();
      })
      .def(
          "__contains__",
          [](const PyDictAttr &self, const std::string &key) {
            return static_cast<bool>(as<mlir::DictionaryAttr>(self).get(key));
          },
          "key"_a)
      .def(
          "__getitem__",
          [](const PyDictAttr &self, const std::string &key) {
            mlir::Attribute attr = as<mlir::DictionaryAttr>(self).get(key);
            if (!attr)
              throw nb::key_error(key.c_str());
            return wrapAttribute(self.context, attr);
          },
          "key"_a)
      .def(
          "get",
          [](const PyDictAttr &self, const std::string &key) {
            return wrapOptionalAttribute(self.context,
                                         as<mlir::DictionaryAttr>(self).get(key));
          },
          "key"_a, "The value for ``key``, or ``None``.")
      .def("__iter__",
           [](const PyDictAttr &self) -> StringIterator {
             std::vector<std::string> keys;
             for (mlir::NamedAttribute entry : as<mlir::DictionaryAttr>(self))
               keys.push_back(entry.getName().str());
             return nb::iter(nb::cast(keys));
           })
      .def(
          "keys",
          [](const PyDictAttr &self) {
            std::vector<std::string> keys;
            for (mlir::NamedAttribute entry : as<mlir::DictionaryAttr>(self))
              keys.push_back(entry.getName().str());
            return keys;
          },
          "The keys, sorted.")
      .def(
          "items",
          [](const PyDictAttr &self) {
            std::vector<std::pair<std::string, AttributeObject>> items;
            for (mlir::NamedAttribute entry : as<mlir::DictionaryAttr>(self))
              items.emplace_back(entry.getName().str(),
                                 wrapAttribute(self.context, entry.getValue()));
            return items;
          },
          "``(key, value)`` pairs, sorted by key.");

  nb::class_<PySymbolRefAttr, PyAttribute>(
      m, "SymbolRefAttr",
      "A reference to a symbol, possibly nested: ``@root::@a::@b``.")
      .def(nb::new_([](const std::string &root,
                       const std::vector<std::string> &nested,
                       std::optional<PyContext> context) {
             ContextHandle state = resolveContext(context);
             std::vector<mlir::FlatSymbolRefAttr> refs;
             for (const std::string &name : nested)
               refs.push_back(mlir::FlatSymbolRefAttr::get(&state->context, name));
             return make<PySymbolRefAttr>(
                 state, mlir::SymbolRefAttr::get(&state->context, root, refs));
           }),
           "root"_a, "nested"_a = std::vector<std::string>{}, nb::kw_only(),
           "context"_a = nb::none(),
           "Create ``@root::@nested[0]::...``.")
      .def_prop_ro(
          "root",
          [](const PySymbolRefAttr &self) {
            return as<mlir::SymbolRefAttr>(self).getRootReference().str();
          },
          "The outermost symbol name.")
      .def_prop_ro(
          "leaf",
          [](const PySymbolRefAttr &self) {
            return as<mlir::SymbolRefAttr>(self).getLeafReference().str();
          },
          "The innermost symbol name.")
      .def_prop_ro(
          "nested",
          [](const PySymbolRefAttr &self) {
            std::vector<std::string> names;
            for (mlir::FlatSymbolRefAttr ref :
                 as<mlir::SymbolRefAttr>(self).getNestedReferences())
              names.push_back(ref.getValue().str());
            return names;
          },
          "Names after the root, outermost first.");

  nb::class_<PyFlatSymbolRefAttr, PySymbolRefAttr>(
      m, "FlatSymbolRefAttr", "A reference to a single symbol: ``@name``.")
      .def(nb::new_([](const std::string &name, std::optional<PyContext> context) {
             ContextHandle state = resolveContext(context);
             return make<PyFlatSymbolRefAttr>(
                 state, mlir::FlatSymbolRefAttr::get(&state->context, name));
           }),
           "name"_a, nb::kw_only(), "context"_a = nb::none(), "Create ``@name``.")
      .def_prop_ro(
          "value",
          [](const PyFlatSymbolRefAttr &self) {
            return as<mlir::FlatSymbolRefAttr>(self).getValue().str();
          },
          "The symbol name.");

  nb::class_<PyDenseElementsAttr, PyAttribute>(
      m, "DenseElementsAttr",
      "A constant tensor or vector with every element stored, e.g.\n"
      "``dense<[1, 2]> : tensor<2xi32>``. Use ``DenseIntElementsAttr`` or\n"
      "``DenseFPElementsAttr`` to build one from Python numbers.")
      .def(nb::new_([](const std::vector<PyAttribute> &elements,
                       const PyShapedType &type) {
             mlir::ShapedType shaped = denseShapedType(type);
             checkElementCount(shaped, elements.size());
             std::vector<mlir::Attribute> native;
             for (const PyAttribute &element : elements) {
               mlir::Attribute attr = unwrap(type.context, element);
               auto typed = llvm::dyn_cast<mlir::TypedAttr>(attr);
               if (!llvm::isa<mlir::IntegerAttr, mlir::FloatAttr>(attr) ||
                   typed.getType() != shaped.getElementType())
                 throw std::invalid_argument(
                     "elements must be IntegerAttr or FloatAttr of type " +
                     printToString(shaped.getElementType()));
               native.push_back(attr);
             }
             return make<PyDenseElementsAttr>(
                 type.context, mlir::DenseElementsAttr::get(shaped, native));
           }),
           "elements"_a, "type"_a,
           "Create from element attributes in row-major order.\n\n"
           "Args:\n"
           "    elements: One attribute per element, or a single one to\n"
           "        splat. Each must be typed with the element type.\n"
           "    type: A statically shaped ``RankedTensorType`` or\n"
           "        ``VectorType``.")
      .def_prop_ro(
          "is_splat",
          [](const PyDenseElementsAttr &self) {
            return as<mlir::DenseElementsAttr>(self).isSplat();
          },
          "Whether every element has the same value.")
      .def("__len__",
           [](const PyDenseElementsAttr &self) {
             return as<mlir::DenseElementsAttr>(self).getNumElements();
           })
      .def_prop_ro(
          "elements",
          [](const PyDenseElementsAttr &self) {
            std::vector<AttributeObject> result;
            for (mlir::Attribute attr :
                 as<mlir::DenseElementsAttr>(self).getValues<mlir::Attribute>())
              result.push_back(wrapAttribute(self.context, attr));
            return result;
          },
          "Every element as an attribute, row-major.");

  nb::class_<PyDenseIntElementsAttr, PyDenseElementsAttr>(
      m, "DenseIntElementsAttr",
      "Dense elements whose element type is an integer or index type.")
      .def(nb::new_([](const std::vector<nb::int_> &values,
                       const PyShapedType &type) {
             mlir::ShapedType shaped = denseShapedType(type);
             checkElementCount(shaped, values.size());
             auto [width, signedness] = integerTraits(shaped.getElementType());
             std::vector<llvm::APInt> native;
             for (const nb::int_ &value : values)
               native.push_back(toAPInt(value, width, signedness));
             return make<PyDenseIntElementsAttr>(
                 type.context, mlir::DenseElementsAttr::get(shaped, native));
           }),
           "values"_a, "type"_a,
           "Create from Python ints in row-major order.\n\n"
           "Args:\n"
           "    values: One int per element, or a single one to splat.\n"
           "    type: A statically shaped tensor or vector of integers.\n\n"
           "Raises:\n"
           "    OverflowError: If a value does not fit the element type.")
      .def_prop_ro(
          "values",
          [](const PyDenseIntElementsAttr &self) {
            auto attr = as<mlir::DenseIntElementsAttr>(self);
            bool isSigned = isSignedForPython(attr.getElementType());
            std::vector<nb::int_> result;
            for (const llvm::APInt &value : attr)
              result.push_back(fromAPInt(value, isSigned));
            return result;
          },
          "Every element as a Python int, row-major.");

  nb::class_<PyDenseFPElementsAttr, PyDenseElementsAttr>(
      m, "DenseFPElementsAttr",
      "Dense elements whose element type is a floating-point type.")
      .def(nb::new_([](const std::vector<double> &values,
                       const PyShapedType &type) {
             mlir::ShapedType shaped = denseShapedType(type);
             checkElementCount(shaped, values.size());
             std::vector<mlir::Attribute> native;
             for (double value : values)
               native.push_back(makeFloatAttr(shaped.getElementType(), value));
             return make<PyDenseFPElementsAttr>(
                 type.context, mlir::DenseElementsAttr::get(shaped, native));
           }),
           "values"_a, "type"_a,
           "Create from Python floats in row-major order.\n\n"
           "Args:\n"
           "    values: One float per element, or a single one to splat.\n"
           "    type: A statically shaped tensor or vector of floats.")
      .def_prop_ro(
          "values",
          [](const PyDenseFPElementsAttr &self) {
            std::vector<double> result;
            for (const llvm::APFloat &value : as<mlir::DenseFPElementsAttr>(self))
              result.push_back(toDouble(value));
            return result;
          },
          "Every element as a Python float, row-major.");

  nb::class_<PyDenseArrayAttr, PyAttribute>(
      m, "DenseArrayAttr",
      "Base class of the ``array<T: ...>`` attributes: compact 1-D arrays of\n"
      "a primitive type, often used for operation properties.")
      .def("__len__", [](const PyDenseArrayAttr &self) {
        return as<mlir::DenseArrayAttr>(self).size();
      });

#define BIND_ARRAY(CppAttr, Elt, PyName, PyElt, Text)                          \
  nb::class_<PySpecificDenseArrayAttr<mlir::CppAttr>, PyDenseArrayAttr>(       \
      m, PyName, "An ``array<" Text ": ...>`` attribute.")                     \
      .def(nb::new_([](const std::vector<Elt> &values,                         \
                       std::optional<PyContext> context) {                     \
             ContextHandle state = resolveContext(context);                    \
             llvm::SmallVector<Elt> native(values.begin(), values.end());      \
             return make<PySpecificDenseArrayAttr<mlir::CppAttr>>(             \
                 state, mlir::CppAttr::get(&state->context, native));          \
           }),                                                                 \
           "values"_a, nb::kw_only(), "context"_a = nb::none(),                \
           "Create from a sequence of " PyElt " values.")                      \
      .def_prop_ro(                                                            \
          "values",                                                            \
          [](const PySpecificDenseArrayAttr<mlir::CppAttr> &self) {            \
            llvm::ArrayRef<Elt> values = as<mlir::CppAttr>(self).asArrayRef(); \
            return std::vector<Elt>(values.begin(), values.end());             \
          },                                                                   \
          "The values.")                                                       \
      .def(                                                                    \
          "__getitem__",                                                       \
          [](const PySpecificDenseArrayAttr<mlir::CppAttr> &self,              \
             int64_t index) {                                                  \
            llvm::ArrayRef<Elt> values = as<mlir::CppAttr>(self).asArrayRef(); \
            return values[normalizeIndex(index, values.size())];               \
          },                                                                   \
          "index"_a);
  MLIR_PYTHON_DENSE_ARRAYS(BIND_ARRAY)
#undef BIND_ARRAY

  nb::class_<PyStridedLayoutAttr, PyAttribute>(
      m, "StridedLayoutAttr",
      "A memref layout given by an offset and per-dimension strides, e.g.\n"
      "``strided<[?, 1], offset: 4>``. ``None`` marks a dynamic value.")
      .def(nb::new_([](std::optional<int64_t> offset,
                       const std::vector<std::optional<int64_t>> &strides,
                       std::optional<PyContext> context) {
             ContextHandle state = resolveContext(context);
             auto toNative = [](std::optional<int64_t> value) {
               return value ? *value : mlir::ShapedType::kDynamic;
             };
             std::vector<int64_t> native;
             for (const std::optional<int64_t> &stride : strides)
               native.push_back(toNative(stride));
             DiagnosticCapture capture(&state->context);
             auto attr = mlir::StridedLayoutAttr::getChecked(
                 mlir::detail::getDefaultDiagnosticEmitFn(&state->context),
                 &state->context, toNative(offset), llvm::ArrayRef<int64_t>(native));
             if (!attr)
               capture.raise("invalid strided layout");
             return make<PyStridedLayoutAttr>(state, attr);
           }),
           "offset"_a, "strides"_a, nb::kw_only(), "context"_a = nb::none(),
           "Create ``strided<strides, offset: offset>``.")
      .def_prop_ro(
          "offset",
          [](const PyStridedLayoutAttr &self) -> std::optional<int64_t> {
            int64_t offset = as<mlir::StridedLayoutAttr>(self).getOffset();
            if (mlir::ShapedType::isDynamic(offset))
              return std::nullopt;
            return offset;
          },
          "The offset, or ``None`` if dynamic.")
      .def_prop_ro(
          "strides",
          [](const PyStridedLayoutAttr &self) {
            std::vector<std::optional<int64_t>> result;
            for (int64_t stride : as<mlir::StridedLayoutAttr>(self).getStrides())
              result.push_back(mlir::ShapedType::isDynamic(stride)
                                   ? std::nullopt
                                   : std::optional<int64_t>(stride));
            return result;
          },
          "Strides, ``None`` for dynamic ones.");
}

} // namespace mlir_python
