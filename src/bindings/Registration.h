// Which dialects, extensions, and passes are linked in. Shared by the
// extension module and the build-time introspection tool.
#pragma once

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace mlir_python {

/// Registers the dialects and dialect extensions linked into the module.
void registerLinkedDialects(mlir::DialectRegistry &registry);
/// Registers the passes linked into the module with MLIR's global registry.
void registerLinkedPasses();
/// Registers the passes mlir-python defines (ProjectPasses.td); called by
/// registerLinkedPasses.
void registerProjectPasses();

} // namespace mlir_python
