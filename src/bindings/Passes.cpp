// Bindings for mlir::PassManager.
#include "Core.h"
#include "ProjectPasses.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <nanobind/stl/string.h>

#include <mutex>
#include <unordered_map>

namespace mlir_python {

using namespace nb::literals;

namespace {

struct PyPassManager {
  ContextHandle context;
  std::unique_ptr<mlir::PassManager> passManager;
};

/// A pipeline running now: the Python tree of the operation it runs on (so
/// Python passes can wrap the operations they get), and the first exception a
/// Python pass raised.
struct Running {
  OpTreeHandle tree;
  std::exception_ptr error;
};
std::mutex runningMutex;
std::unordered_map<mlir::Operation *, Running> running;

/// The callables of registered Python passes (see register_python_pass).
std::vector<std::shared_ptr<PyObject *>> pythonPassFunctions;

/// The running pipeline that `op` is part of.
Running *runningFor(mlir::Operation *op) {
  std::lock_guard<std::mutex> lock(runningMutex);
  for (mlir::Operation *at = op; at; at = at->getParentOp())
    if (auto found = running.find(at); found != running.end())
      return &found->second;
  return nullptr;
}

std::string pipelineText(const mlir::OpPassManager &pm) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  pm.printAsTextualPipeline(stream);
  return result;
}

} // namespace

void bindPasses(nb::module_ &m) {
  m.def(
      "register_python_pass",
      [](const std::string &argument, const std::string &description,
         nb::callable run) {
        // Registered passes live until the process exits, after Python has
        // shut down; the callable is released at interpreter exit (see
        // release_python_passes), after which the pass fails if run.
        auto function = std::make_shared<PyObject *>(run.release().ptr());
        pythonPassFunctions.push_back(function);
        auto callback = std::make_shared<PassCallback>(
            [function](mlir::Operation *op) -> bool {
              nb::gil_scoped_acquire gil;
              Running *pipeline = runningFor(op);
              if (!pipeline || !*function)
                return false; // only pipelines run from Python have a tree
              try {
                nb::handle callable(*function);
                callable(wrapOperation(pipeline->tree, op));
                return true;
              } catch (...) {
                std::lock_guard<std::mutex> lock(runningMutex);
                if (!pipeline->error)
                  pipeline->error = std::current_exception();
                return false;
              }
            });
        registerCallbackPass(argument, description, std::move(callback));
      },
      "argument"_a, "description"_a, "run"_a,
      "Low level: register a pass named ``argument`` that calls ``run`` with\n"
      "each operation it runs on (see ``passes.PythonPass``).");
  m.def(
      "release_python_passes",
      []() {
        for (auto &function : pythonPassFunctions) {
          Py_XDECREF(*function);
          *function = nullptr;
        }
        pythonPassFunctions.clear();
      },
      "Low level: release the callables of registered Python passes (run\n"
      "at interpreter exit).");
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
              std::lock_guard<std::mutex> lock(runningMutex);
              if (!running.emplace(native, Running{op.tree, nullptr}).second)
                throw std::invalid_argument(
                    "a pipeline is already running on this operation");
            }
            {
              nb::gil_scoped_release release;
              result = self.passManager->run(native);
            }
            std::exception_ptr error;
            {
              std::lock_guard<std::mutex> lock(runningMutex);
              error = running[native].error;
              running.erase(native);
            }
            // Passes may have replaced or erased anything nested in `native`.
            OpTree::invalidate(op.tree, native);
            if (error)
              std::rethrow_exception(error);
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
