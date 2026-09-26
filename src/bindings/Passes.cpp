// Bindings for mlir::PassManager.
#include "Core.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <nanobind/stl/string.h>

namespace mlir_python {

using namespace nb::literals;

namespace {

struct PyPassManager {
  ContextHandle context;
  std::unique_ptr<mlir::PassManager> passManager;
};

std::string pipelineText(const mlir::OpPassManager &pm) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  pm.printAsTextualPipeline(stream);
  return result;
}

} // namespace

void bindPasses(nb::module_ &m) {
  nb::class_<PyPassManager>(
      m, "ParsedPassPipeline",
      "A pass pipeline parsed from MLIR's textual syntax, ready to run.\n\n"
      "Low level: build pipelines with ``mlir_python.PassManager`` and the\n"
      "typed passes in ``mlir_python.passes``, which produce this object.")
      .def_static(
          "parse",
          [](const std::string &pipeline, std::optional<PyContext> context) {
            ContextHandle state = resolveContext(context);
            std::string errors;
            llvm::raw_string_ostream errorStream(errors);
            mlir::FailureOr<mlir::OpPassManager> parsed =
                mlir::parsePassPipeline(pipeline, errorStream);
            if (mlir::failed(parsed))
              throw std::invalid_argument("invalid pass pipeline: " + errors);
            auto pm = std::make_unique<mlir::PassManager>(
                &state->context, parsed->getOpAnchorName());
            static_cast<mlir::OpPassManager &>(*pm) = std::move(*parsed);
            return PyPassManager{state, std::move(pm)};
          },
          "pipeline"_a, nb::kw_only(), "context"_a = nb::none(),
          "Parse an anchored pipeline such as\n"
          "``\"builtin.module(canonicalize, cse)\"``.\n\n"
          "Raises:\n    ValueError: If the pipeline does not parse.")
      .def(
          "enable_verifier",
          [](PyPassManager &self, bool enabled) {
            self.passManager->enableVerifier(enabled);
          },
          "enabled"_a = true,
          "Verify the IR after each pass (on by default).")
      .def(
          "run",
          [](PyPassManager &self, const PyOperation &op) {
            mlir::Operation *native = get(op);
            requireSameContext(self.context, op.tree->context(), "operation");
            DiagnosticCapture capture(&self.context->context);
            mlir::LogicalResult result = mlir::failure();
            {
              nb::gil_scoped_release release;
              result = self.passManager->run(native);
            }
            // Passes may have replaced or erased anything nested in `native`.
            OpTree::invalidate(op.tree, native);
            if (mlir::failed(result))
              capture.raise("pass pipeline failed");
          },
          "op"_a,
          "Run the pipeline on ``op`` in place.\n\n"
          "Raises:\n"
          "    MLIRError: If a pass fails or the result does not verify.")
      .def_prop_ro(
          "anchor",
          [](const PyPassManager &self) {
            return self.passManager->getOpAnchorName().str();
          },
          "Name of the operations this manager runs on.")
      .def_prop_ro(
          "context",
          [](const PyPassManager &self) { return PyContext{self.context}; },
          "The context passes run in.")
      .def("__str__",
           [](const PyPassManager &self) {
             return pipelineText(*self.passManager);
           })
      .def("__repr__", [](const PyPassManager &self) {
        return "ParsedPassPipeline(\"" + pipelineText(*self.passManager) + "\")";
      });
}

} // namespace mlir_python
