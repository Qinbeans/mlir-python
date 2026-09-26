// Build-time tool: prints facts about every registered operation that
// TableGen records do not state directly, as JSON.
//
// Usage: mlir-python-introspect <output.json> [pass-records.json...]
//
// Output: {"operations": {name: facts}, "passes": [registered arguments]},
// where the candidate passes come from `llvm-tblgen --dump-json` records.
//
// ODS adds some interfaces implicitly (for example InferTypeOpInterface when
// every result type is derivable), so the generator asks the compiled dialects
// instead of re-deriving ODS's rules.
#include "../src/bindings/Registration.h"

#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Pass/PassRegistry.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "usage: " << argv[0]
                 << " <output.json> [pass-records.json...]\n";
    return 2;
  }
  mlir::DialectRegistry registry;
  mlir_python::registerLinkedDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::json::Object ops;
  for (mlir::RegisteredOperationName name : context.getRegisteredOperations()) {
    ops[name.getStringRef()] = llvm::json::Object{
        {"dialect", name.getDialectNamespace()},
        {"infers_result_types",
         name.hasInterface<mlir::InferTypeOpInterface>()},
        {"is_terminator", name.hasTrait<mlir::OpTrait::IsTerminator>()},
        {"attr_sized_operand_segments",
         name.hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()},
        {"attr_sized_result_segments",
         name.hasTrait<mlir::OpTrait::AttrSizedResultSegments>()},
    };
  }
  mlir_python::registerLinkedPasses();
  llvm::json::Array passes;
  for (int i = 2; i < argc; ++i) {
    auto buffer = llvm::MemoryBuffer::getFile(argv[i]);
    if (!buffer) {
      llvm::errs() << argv[i] << ": " << buffer.getError().message() << "\n";
      return 1;
    }
    llvm::Expected<llvm::json::Value> records =
        llvm::json::parse((*buffer)->getBuffer());
    if (!records) {
      llvm::errs() << argv[i] << ": " << llvm::toString(records.takeError())
                   << "\n";
      return 1;
    }
    const llvm::json::Object *root = records->getAsObject();
    const llvm::json::Object *instances =
        root ? root->getObject("!instanceof") : nullptr;
    const llvm::json::Array *candidates =
        instances ? instances->getArray("PassBase") : nullptr;
    if (!candidates)
      continue;
    for (const llvm::json::Value &name : *candidates) {
      const llvm::json::Object *record = root->getObject(*name.getAsString());
      std::optional<llvm::StringRef> argument =
          record ? record->getString("argument") : std::nullopt;
      if (argument && mlir::PassInfo::lookup(*argument))
        passes.push_back(argument->str()); // json::Value borrows StringRefs
    }
  }

  std::error_code error;
  llvm::raw_fd_ostream output(argv[1], error);
  if (error) {
    llvm::errs() << argv[1] << ": " << error.message() << "\n";
    return 1;
  }
  output << llvm::json::Value(llvm::json::Object{
                {"operations", std::move(ops)}, {"passes", std::move(passes)}})
         << "\n";
  return 0;
}
