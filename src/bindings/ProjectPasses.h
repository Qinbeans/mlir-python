// Passes defined by mlir-python, shared with the pass bindings.
#pragma once

#include <functional>
#include <memory>
#include <string>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir_python {

/// Runs a pass on one operation; returns false to fail the pass.
using PassCallback = std::function<bool(mlir::Operation *)>;

/// Registers a pass named `argument` whose instances run `callback` (used for
/// passes written in Python). Registering an argument twice is an error.
void registerCallbackPass(const std::string &argument,
                          const std::string &description,
                          std::shared_ptr<PassCallback> callback);

} // namespace mlir_python
