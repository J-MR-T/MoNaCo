// RUN: %MLIRCheckLLVM

module{
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(16 : i32) : i32
    %256 = llvm.mlir.constant(256 : i32) : i32
    %false = llvm.mlir.constant(false) : i1
    %mask = llvm.mlir.constant(4294967295 : i32) : i32
    %3 = llvm.mlir.constant(true) : i1
    %4 = llvm.sext %3 : i1 to i32 // ffffffff
    %5 = llvm.sext %3 : i1 to i16 // ffff
    %zext5 = llvm.zext %5 : i16 to i32 // 0000ffff
    %6 = llvm.shl %zext5, %2 : i32 // ffff0000
    %7 = llvm.add %6, %zext5 : i32 // should be the same as %4
    %8 = llvm.and %7, %mask : i32
    %9 = llvm.xor %4, %mask : i32
    %10 = llvm.or %8, %9 : i32
    %11 = llvm.urem %10, %256 : i32
    llvm.return %11 : i32
  }
}
