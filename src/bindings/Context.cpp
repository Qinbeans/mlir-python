// Bindings for Context, Location, and MLIRError.
#include "Core.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace mlir_python {

using namespace nb::literals;

namespace {

PyContext createContext(bool allowUnregisteredDialects, bool multithreading,
                        bool loadAllDialects) {
  mlir::DialectRegistry registry;
  registerLinkedDialects(registry);
  auto state = std::make_shared<ContextState>(registry);
  state->context.allowUnregisteredDialects(allowUnregisteredDialects);
  state->context.disableMultithreading(!multithreading);
  if (loadAllDialects)
    state->context.loadAllAvailableDialects();
  return PyContext{std::move(state)};
}

PyLocation makeLocation(const ContextHandle &context, mlir::Location location) {
  return PyLocation{context, location};
}

std::optional<PyContext> currentContext() {
  try {
    return PyContext{resolveContext(std::nullopt)};
  } catch (const std::invalid_argument &) {
    return std::nullopt;
  }
}

} // namespace

void bindCore(nb::module_ &m) {
  nb::exception<MLIRError>(m, "MLIRError", PyExc_Exception)
      .attr("__doc__") =
      "Raised when MLIR reports an error (parsing, verification, a failing "
      "pass).\n\nThe message lists every diagnostic MLIR emitted, one per "
      "line, as ``location: severity: message``.";

  nb::class_<PyContext>(
      m, "Context",
      "Owns an MLIR context: the dialects it has loaded and the uniqued\n"
      "types, attributes, and locations created in it.\n\n"
      "Everything derived from a context keeps it alive. A context can be\n"
      "entered with ``with`` to make it the default for calls that take an\n"
      "optional ``context=`` argument::\n\n"
      "    with Context():\n"
      "        i32 = IntegerType(32)")
      .def(nb::new_(&createContext), nb::kw_only(),
           "allow_unregistered_dialects"_a = false, "multithreading"_a = true,
           "load_all_dialects"_a = false,
           "Create a context.\n\n"
           "Args:\n"
           "    allow_unregistered_dialects: Accept operations, types, and\n"
           "        attributes from dialects this build does not know.\n"
           "    multithreading: Let MLIR use a thread pool (e.g. for passes).\n"
           "    load_all_dialects: Load every available dialect up front\n"
           "        instead of on first use.")
      .def_prop_rw(
          "allow_unregistered_dialects",
          [](const PyContext &self) {
            return self.get()->allowsUnregisteredDialects();
          },
          [](PyContext &self, bool value) {
            self.get()->allowUnregisteredDialects(value);
          },
          "Whether unknown dialects are accepted.")
      .def_prop_rw(
          "multithreading",
          [](const PyContext &self) {
            return self.get()->isMultithreadingEnabled();
          },
          [](PyContext &self, bool value) {
            self.get()->disableMultithreading(!value);
          },
          "Whether MLIR may use its thread pool.")
      .def_prop_ro(
          "loaded_dialects",
          [](const PyContext &self) {
            std::vector<std::string> names;
            for (mlir::Dialect *dialect : self.get()->getLoadedDialects())
              names.emplace_back(dialect->getNamespace());
            return names;
          },
          "Namespaces of the dialects currently loaded, sorted.")
      .def_prop_ro(
          "available_dialects",
          [](const PyContext &self) {
            std::vector<std::string> names;
            for (llvm::StringRef name :
                 self.get()->getDialectRegistry().getDialectNames())
              names.emplace_back(name);
            return names;
          },
          "Namespaces of every dialect that can be loaded, sorted.")
      .def(
          "load_dialect",
          [](PyContext &self, const std::string &name) {
            if (!self.get()->getOrLoadDialect(name))
              throw std::invalid_argument("unknown dialect '" + name + "'");
          },
          "name"_a,
          "Load the dialect with namespace ``name`` (e.g. ``\"arith\"``).\n\n"
          "Raises:\n"
          "    ValueError: If no such dialect is available.")
      .def(
          "load_all_available_dialects",
          [](PyContext &self) { self.get()->loadAllAvailableDialects(); },
          "Load every dialect in ``available_dialects``.")
      .def(
          "is_registered_operation",
          [](PyContext &self, const std::string &name) {
            auto dot = name.find('.');
            if (dot != std::string::npos)
              self.get()->getOrLoadDialect(name.substr(0, dot));
            return mlir::RegisteredOperationName::lookup(name, self.get())
                .has_value();
          },
          "name"_a,
          "Whether ``name`` (e.g. ``\"arith.addi\"``) is a known operation.\n"
          "Loads the operation's dialect if needed.")
      .def_static("current", &currentContext,
                  "The context of the innermost ``with Context()``,\n"
                  "``with Location``, or ``with InsertionPoint`` block, if any.")
      .def(
          "__enter__",
          [](nb::handle self) {
            pushContext(nb::cast<PyContext &>(self));
            return nb::borrow(self);
          },
          nb::sig("def __enter__(self) -> typing.Self"),
          "Make this the default context inside a ``with`` block.")
      .def(
          "__exit__",
          [](PyContext &self, nb::handle, nb::handle, nb::handle) {
            popContext(self);
          },
          "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(),
          nb::sig("def __exit__(self, exc_type: type[BaseException] | None, "
                  "exc_value: BaseException | None, traceback: object, "
                  "/) -> None"))
      .def("__eq__", identityEq<PyContext>([](const PyContext &v) -> const void * { return v.state.get(); }),
           "other"_a.none(), nb::sig(kEqSignature))
      .def("__hash__", identityHash<PyContext>([](const PyContext &v) -> const void * { return v.state.get(); }))
      .def("__repr__", [](const PyContext &self) {
        return "Context(loaded_dialects=" +
               std::to_string(self.get()->getLoadedDialects().size()) + ")";
      });

  nb::class_<PyLocation>(
      m, "Location",
      "A source location attached to operations and diagnostics.\n\n"
      "Enter a location with ``with`` to make it the default for operations\n"
      "created inside the block.")
      .def_static(
          "unknown",
          [](std::optional<PyContext> context) {
            ContextHandle state = resolveContext(context);
            return makeLocation(state, mlir::UnknownLoc::get(&state->context));
          },
          nb::kw_only(), "context"_a = nb::none(),
          "A location carrying no information (``loc(unknown)``).")
      .def_static(
          "file",
          [](const std::string &filename, unsigned line, unsigned column,
             std::optional<PyContext> context) {
            ContextHandle state = resolveContext(context);
            return makeLocation(state,
                                mlir::FileLineColLoc::get(&state->context,
                                                          filename, line,
                                                          column));
          },
          "filename"_a, "line"_a, "column"_a, nb::kw_only(),
          "context"_a = nb::none(),
          "A ``filename:line:column`` location.")
      .def_static(
          "name",
          [](const std::string &name, std::optional<PyLocation> child,
             std::optional<PyContext> context) {
            ContextHandle state =
                child ? child->context : resolveContext(context);
            auto nameAttr = mlir::StringAttr::get(&state->context, name);
            mlir::Location location =
                child ? mlir::NameLoc::get(nameAttr, child->location)
                      : mlir::NameLoc::get(nameAttr);
            return makeLocation(state, location);
          },
          "name"_a, "child"_a = nb::none(), nb::kw_only(),
          "context"_a = nb::none(),
          "A named location, optionally wrapping a more precise ``child``.")
      .def_static(
          "callsite",
          [](const PyLocation &callee, const PyLocation &caller) {
            requireSameContext(callee.context, caller.context, "caller");
            return makeLocation(
                callee.context,
                mlir::CallSiteLoc::get(callee.location, caller.location));
          },
          "callee"_a, "caller"_a,
          "The location of ``callee`` as called from ``caller``.")
      .def_static(
          "fused",
          [](const std::vector<PyLocation> &locations,
             std::optional<PyAttribute> metadata,
             std::optional<PyContext> context) {
            ContextHandle state =
                !locations.empty() ? locations.front().context
                                   : resolveContext(context);
            std::vector<mlir::Location> native;
            for (const PyLocation &location : locations) {
              requireSameContext(state, location.context, "location");
              native.push_back(location.location);
            }
            mlir::Attribute nativeMetadata;
            if (metadata) {
              requireSameContext(state, metadata->context, "metadata");
              nativeMetadata = metadata->attribute;
            }
            return makeLocation(state,
                                mlir::FusedLoc::get(&state->context, native,
                                                    nativeMetadata));
          },
          "locations"_a, "metadata"_a = nb::none(), nb::kw_only(),
          "context"_a = nb::none(),
          "Several locations merged into one, with optional metadata.")
      .def_static(
          "current",
          []() -> std::optional<PyLocation> {
            std::optional<PyContext> context = currentContext();
            if (!context)
              return std::nullopt;
            return makeLocation(context->state,
                                resolveLocation(context->state, std::nullopt));
          },
          "The default location for new operations: the innermost\n"
          "``with Location`` block, else ``unknown`` in the current context,\n"
          "else ``None``.")
      .def_prop_ro(
          "context",
          [](const PyLocation &self) { return PyContext{self.context}; },
          "The context this location belongs to.")
      .def(
          "__enter__",
          [](nb::handle self) {
            pushLocation(nb::cast<PyLocation &>(self));
            return nb::borrow(self);
          },
          nb::sig("def __enter__(self) -> typing.Self"),
          "Make this the default location inside a ``with`` block.")
      .def(
          "__exit__",
          [](PyLocation &self, nb::handle, nb::handle, nb::handle) {
            popLocation(self);
          },
          "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(),
          nb::sig("def __exit__(self, exc_type: type[BaseException] | None, "
                  "exc_value: BaseException | None, traceback: object, "
                  "/) -> None"))
      .def("__eq__", identityEq<PyLocation>([](const PyLocation &v) -> const void * { return v.location.getAsOpaquePointer(); }),
           "other"_a.none(), nb::sig(kEqSignature))
      .def("__hash__", identityHash<PyLocation>([](const PyLocation &v) -> const void * { return v.location.getAsOpaquePointer(); }))
      .def("__str__",
           [](const PyLocation &self) {
             return printToString(self.location);
           })
      .def("__repr__", [](const PyLocation &self) {
        return "Location(" + printToString(self.location) + ")";
      });
}

} // namespace mlir_python
