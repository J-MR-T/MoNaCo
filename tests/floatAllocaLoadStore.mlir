// RUN: %MLIRCheckLLVM
module {
  llvm.mlir.global private unnamed_addr constant @".str.2"("objective value            : %0.0f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func @main() -> i32{
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    llvm.store %2, %1 {alignment = 8 : i64} : f64, !llvm.ptr

    %4 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> f64

    %3 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    llvm.store %3, %1 {alignment = 8 : i64} : f64, !llvm.ptr

    %5 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> f64

    %257 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %6 = llvm.call @printf(%257, %4) : (!llvm.ptr, f64) -> i32
    %7 = llvm.call @printf(%257, %5) : (!llvm.ptr, f64) -> i32

    %8 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %8 : i32
  }
}
