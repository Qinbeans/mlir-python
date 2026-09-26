// Entry point of the `mlir_python._mlir_python` extension module.
#include "Core.h"

NB_MODULE(_mlir_python, m) {
  m.doc() = "Typed bindings over the MLIR C++ API.";

  mlir_python::registerLinkedPasses();

  // Base classes are bound before the classes that derive from them.
  mlir_python::bindCore(m);
  mlir_python::bindTypes(m);
  mlir_python::bindAttributes(m);
  mlir_python::bindIR(m);
  mlir_python::bindPasses(m);
  mlir_python::bindCodegen(m);
  mlir_python::bindDialects(m);
  mlir_python::bindDialectExtras(m);
}
