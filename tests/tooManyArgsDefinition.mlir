// RUN: %MLIRCheckLLVM
module{
  llvm.mlir.global private unnamed_addr constant @".str"("Wuhu %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func @myfunc(%0 : !llvm.ptr, %1 : i32, %2 : i32, %3 : i32, %4 : i32, %5 : i32, %6 : i32, %7 : i32, %8 : i32, %9 : i32, %10 : i32, %11 : i32, %12 : i32, %13 : i32, %14 : i32, %15 : i32) -> i32{
    %16 = llvm.call @printf(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)
    %17 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %17 : i32
  }

  llvm.func @main() -> i32 {
    %0  = llvm.mlir.addressof @".str" : !llvm.ptr
    %1  = llvm.mlir.constant(1 :  i32) : i32
    %2  = llvm.mlir.constant(2 :  i32) : i32
    %3  = llvm.mlir.constant(3 :  i32) : i32
    %4  = llvm.mlir.constant(4 :  i32) : i32
    %5  = llvm.mlir.constant(5 :  i32) : i32
    %6  = llvm.mlir.constant(6 :  i32) : i32
    %7  = llvm.mlir.constant(7 :  i32) : i32
    %8  = llvm.mlir.constant(8 :  i32) : i32
    %9  = llvm.mlir.constant(9 :  i32) : i32
    %10 = llvm.mlir.constant(10 : i32) : i32
    %11 = llvm.mlir.constant(11 : i32) : i32
    %12 = llvm.mlir.constant(12 : i32) : i32
    %13 = llvm.mlir.constant(13 : i32) : i32
    %14 = llvm.mlir.constant(14 : i32) : i32
    %15 = llvm.mlir.constant(15 : i32) : i32
    %17 = llvm.call @myfunc(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> (i32)

    llvm.return %17 : i32
  }
}
