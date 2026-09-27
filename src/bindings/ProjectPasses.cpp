// Passes defined by mlir-python itself; their options and documentation are
// declared in ProjectPasses.td, which generates the Python classes.
#include "ProjectPasses.h"
#include "Registration.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/EmitC/IR/EmitC.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir_python {
namespace {

struct UpliftWhileToFor
    : public mlir::PassWrapper<UpliftWhileToFor, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UpliftWhileToFor)

  llvm::StringRef getArgument() const final { return "scf-uplift-while-to-for"; }
  llvm::StringRef getDescription() const final {
    return "Turn counted scf.while loops into scf.for";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    mlir::scf::populateUpliftWhileToForPatterns(patterns);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

/// Math operations with a C99 <math.h> counterpart: the double-precision name
/// (C adds an `f` suffix for float; C++ calls the std:: overload). Operations
/// whose name is marked generic are type-generic macros in C.
struct LibmFunction {
  llvm::StringLiteral operation;
  llvm::StringLiteral function;
  bool generic = false;
};
constexpr LibmFunction kLibmFunctions[] = {
    {"math.absf", "fabs"},        {"math.acos", "acos"},
    {"math.acosh", "acosh"},      {"math.asin", "asin"},
    {"math.asinh", "asinh"},      {"math.atan", "atan"},
    {"math.atan2", "atan2"},      {"math.atanh", "atanh"},
    {"math.cbrt", "cbrt"},        {"math.ceil", "ceil"},
    {"math.copysign", "copysign"}, {"math.cos", "cos"},
    {"math.cosh", "cosh"},        {"math.erf", "erf"},
    {"math.erfc", "erfc"},        {"math.exp", "exp"},
    {"math.exp2", "exp2"},        {"math.expm1", "expm1"},
    {"math.floor", "floor"},      {"math.fma", "fma"},
    {"math.isfinite", "isfinite", true}, {"math.isinf", "isinf", true},
    {"math.isnan", "isnan", true}, {"math.isnormal", "isnormal", true},
    {"math.log", "log"},          {"math.log10", "log10"},
    {"math.log1p", "log1p"},      {"math.log2", "log2"},
    {"math.powf", "pow"},         {"math.round", "round"},
    // Rounds half to even under the default rounding mode.
    {"math.roundeven", "nearbyint"}, {"math.sin", "sin"},
    {"math.sinh", "sinh"},        {"math.sqrt", "sqrt"},
    {"math.tan", "tan"},          {"math.tanh", "tanh"},
    {"math.trunc", "trunc"},
};

struct ConvertMathToEmitCLibm
    : public mlir::PassWrapper<ConvertMathToEmitCLibm,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMathToEmitCLibm)

  ConvertMathToEmitCLibm() = default;
  ConvertMathToEmitCLibm(const ConvertMathToEmitCLibm &other)
      : PassWrapper(other) {}

  llvm::StringRef getArgument() const final {
    return "convert-math-to-emitc-libm";
  }
  llvm::StringRef getDescription() const final {
    return "Turn math operations into calls to the C or C++ math library";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::emitc::EmitCDialect>();
  }

  Option<bool> lowerToCpp{
      *this, "lower-to-cpp",
      llvm::cl::desc(
          "Call the C++ library (std::sqrt in <cmath>) instead of C's"),
      llvm::cl::init(false)};

  static bool isScalarFloat(mlir::Type type) {
    return mlir::isa<mlir::Float32Type, mlir::Float64Type>(type);
  }

  /// The call replacing `op`, or null when it has no library counterpart.
  mlir::Value call(mlir::OpBuilder &builder, mlir::Operation *op,
                   const LibmFunction &function) const {
    if (op->getNumResults() != 1 || op->getNumOperands() == 0 ||
        !llvm::all_of(op->getOperandTypes(), isScalarFloat))
      return {};
    mlir::Type operand = op->getOperand(0).getType();
    if (!llvm::all_of(op->getOperandTypes(),
                      [&](mlir::Type type) { return type == operand; }))
      return {};
    std::string callee;
    if (lowerToCpp)
      callee = ("std::" + function.function).str();
    else if (operand.isF32() && !function.generic)
      callee = (function.function + "f").str();
    else
      callee = function.function.str();
    return mlir::emitc::CallOpaqueOp::create(builder, op->getLoc(),
                                             op->getResultTypes(), callee,
                                             op->getOperands())
        .getResult(0);
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    llvm::StringMap<const LibmFunction *> functions;
    for (const LibmFunction &function : kLibmFunctions)
      functions[function.operation] = &function;
    const LibmFunction sqrt{"math.sqrt", "sqrt"};

    llvm::SmallVector<mlir::Operation *> candidates;
    module.walk([&](mlir::Operation *op) {
      llvm::StringRef name = op->getName().getStringRef();
      if (name == "math.rsqrt" || functions.count(name))
        candidates.push_back(op);
    });
    bool called = false;
    mlir::OpBuilder builder(&getContext());
    for (mlir::Operation *op : candidates) {
      builder.setInsertionPoint(op);
      mlir::Value result;
      if (op->getName().getStringRef() == "math.rsqrt") {
        mlir::Value root = call(builder, op, sqrt);
        if (root) {
          mlir::Type type = root.getType();
          mlir::Value one = mlir::arith::ConstantOp::create(
              builder, op->getLoc(), builder.getFloatAttr(type, 1.0));
          result = mlir::arith::DivFOp::create(builder, op->getLoc(), one, root);
        }
      } else {
        result = call(builder, op, *functions[op->getName().getStringRef()]);
      }
      if (!result)
        continue;
      op->getResult(0).replaceAllUsesWith(result);
      op->erase();
      called = true;
    }
    if (!called)
      return;
    llvm::StringRef header = lowerToCpp ? "cmath" : "math.h";
    for (auto include : module.getOps<mlir::emitc::IncludeOp>())
      if (include.getInclude() == header)
        return;
    builder.setInsertionPointToStart(module.getBody());
    mlir::emitc::IncludeOp::create(builder, module.getLoc(),
                                   builder.getStringAttr(header),
                                   builder.getUnitAttr());
  }
};

/// A pass whose work is done by a callback (a pass written in Python).
struct CallbackPass
    : public mlir::PassWrapper<CallbackPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CallbackPass)

  CallbackPass(std::string argument, std::string description,
               std::shared_ptr<PassCallback> callback)
      : argument(std::move(argument)), description(std::move(description)),
        callback(std::move(callback)) {}

  llvm::StringRef getArgument() const final { return argument; }
  llvm::StringRef getDescription() const final { return description; }
  llvm::StringRef getName() const final { return argument; }

  void runOnOperation() override {
    if (!(*callback)(getOperation()))
      signalPassFailure();
  }

  std::string argument;
  std::string description;
  std::shared_ptr<PassCallback> callback;
};

} // namespace

void registerProjectPasses() {
  mlir::PassRegistration<UpliftWhileToFor>();
  mlir::PassRegistration<ConvertMathToEmitCLibm>();
}

void registerCallbackPass(const std::string &argument,
                          const std::string &description,
                          std::shared_ptr<PassCallback> callback) {
  mlir::registerPass([=]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<CallbackPass>(argument, description, callback);
  });
}

} // namespace mlir_python
