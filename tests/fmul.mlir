// RUN: %MLIRCheckLLVM

module{
  llvm.func @main() -> i64 {
    %0 = llvm.mlir.constant(0:i64) : i64
    %20 = llvm.mlir.constant(1.250000e+00 : f64) : f64
    %186 = llvm.sitofp %0 : i64 to f64
    %187 = llvm.fmul %20, %186  : f64
    %188 = llvm.fptosi %187 : f64 to i64
    llvm.return %0 : i64
  }
}
