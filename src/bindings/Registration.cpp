// The single place that decides which dialects, extensions, and passes are
// linked into the extension module. Keep in sync with target_link_libraries.
#include "Registration.h"

#include <mlir/IR/DialectRegistry.h>

#include <mlir/Conversion/ArithToEmitC/ArithToEmitC.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h>
#include <mlir/Conversion/FuncToEmitC/FuncToEmitC.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/SCFToEmitC/SCFToEmitC.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h>
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Async/Passes.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Pipelines/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h>
#include <mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/EmitC/IR/EmitC.h>
#include <mlir/Dialect/EmitC/Transforms/Passes.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/IRDL/IR/IRDL.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h>
#include <mlir/Dialect/MemRef/Transforms/BufferViewFlowOpInterfaceImpl.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Vector/Transforms/Passes.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Transforms/Passes.h>

namespace mlir_python {

void registerLinkedDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::cf::ControlFlowDialect, mlir::emitc::EmitCDialect,
                  mlir::func::FuncDialect, mlir::irdl::IRDLDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  mlir::ub::UBDialect, mlir::vector::VectorDialect>();
  mlir::func::registerInlinerExtension(registry);

  // Bufferization (tensor to memref) and buffer deallocation: how each
  // dialect's operations read, write, allocate, and forward buffers.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::memref::registerBufferViewFlowOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);

  // Lowering to the LLVM dialect (used by the `convert-to-llvm` pass).
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  // Loaded as a dependency of tensor and bufferization; convert-to-llvm needs
  // every loaded dialect's lowering.
  mlir::registerConvertComplexToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);

  // Lowering to the EmitC dialect (used by the `convert-to-emitc` pass), from
  // which mlir_python.codegen.to_cpp translates to C and C++.
  mlir::registerConvertArithToEmitCInterface(registry);
  mlir::registerConvertFuncToEmitCInterface(registry);
  mlir::registerConvertMemRefToEmitCInterface(registry);
  mlir::registerConvertSCFToEmitCInterface(registry);

  // Translation from the LLVM dialect to LLVM IR.
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
}

void registerLinkedPasses() {
  mlir::registerTransformsPasses();
  mlir::registerAsyncPasses();
  mlir::registerConvertAsyncToLLVMPass();
  mlir::bufferization::registerBufferizationPasses();
  mlir::bufferization::registerBufferizationPipelines();
  mlir::memref::registerExpandReallocPass();
  mlir::vector::registerVectorPasses();
  mlir::emitc::registerEmitCPasses();
  mlir::registerConvertBufferizationToMemRefPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerConvertMathToLLVMPass();
  mlir::registerConvertToLLVMPass();
  mlir::registerConvertVectorToLLVMPass();
  mlir::registerConvertVectorToSCF();
  mlir::registerConvertToEmitC();
  mlir::registerConvertArithToEmitC();
  mlir::registerConvertFuncToEmitC();
  mlir::registerConvertMathToEmitC();
  mlir::registerConvertMemRefToEmitC();
  mlir::registerSCFToEmitC();
  mlir::registerLiftControlFlowToSCFPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerUBToLLVMConversionPass();
  registerProjectPasses();
}

} // namespace mlir_python
