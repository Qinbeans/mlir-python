// Passes defined by mlir-python itself. Their records feed tools/gen_passes.py
// like MLIR's own pass definitions; the implementations live in
// ProjectPasses.cpp, registered under the same arguments.

include "mlir/Pass/PassBase.td"

def UpliftWhileToFor : Pass<"scf-uplift-while-to-for"> {
  let summary = "Turn counted scf.while loops into scf.for";
  let description = [{
    Rewrites an `scf.while` loop whose condition compares an induction
    variable against a loop-invariant bound (`slt`/`sgt`) and whose body adds
    a loop-invariant step into an `scf.for` loop, the form that later
    parallelization and vectorization expect. Other loops are left as they
    are. Wraps MLIR's `populateUpliftWhileToForPatterns`.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect"];
}

def ConvertMathToEmitCLibm : Pass<"convert-math-to-emitc-libm", "::mlir::ModuleOp"> {
  let summary = "Turn math operations into calls to the C or C++ math library";
  let description = [{
    Rewrites each scalar `f32`/`f64` math operation with a `<math.h>`
    counterpart (`math.sqrt` to `sqrtf`/`sqrt`, `math.isnan` to `isnan`, ...)
    as an `emitc.call_opaque`, or with `lower-to-cpp`, a call to the
    `<cmath>` overload (`std::sqrt`). `math.rsqrt` becomes a division by the
    square root. The module includes the header when any call is made.
    Covers every math operation `convert-math-to-emitc` does and the rest
    of the C99 library.
  }];
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::emitc::EmitCDialect"];
  let options = [
    Option<"lowerToCpp", "lower-to-cpp", "bool", /*default=*/"false",
           "Call the C++ library (std::sqrt in <cmath>) instead of C's">,
  ];
}
