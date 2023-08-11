// RUN: %FileCheckExecReturnStatus %s
// CHECK: 20

module{
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(42.0 : f32) : f32
    %1 = llvm.fadd %0, %0 : f32 // 84
    %2 = llvm.intr.fmuladd(%1, %1, %1) : (f32, f32, f32) -> f32 // 84 * 84 + 84 = 7140
    %3 = llvm.fptosi %2 : f32 to i32
    %4 = llvm.mlir.constant(45 : i32) : i32
    %5 = llvm.mul %3, %4 : i32 // 7140 * 45 = 321300
    %6 = llvm.mlir.constant(256 : i32) : i32
    %7 = llvm.urem %5, %6 : i32 // 321300 mod 256 = 20
    llvm.return %7 : i32 // -> the return status will be 20
  }
}
