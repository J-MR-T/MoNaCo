// RUN: %MLIRCheckLLVM
module{
  llvm.mlir.global private unnamed_addr constant @intStr("%d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    %top = llvm.mlir.constant(400000 : i32) : i32
    %addr = llvm.mlir.addressof @intStr : !llvm.ptr
    llvm.br ^bb1(%0 , %1 : i32, i32)
  ^bb1(%2 : i32, %3 : i32):
    %4 = llvm.add %2, %3 : i32
    %printf1 = llvm.call @printf(%addr, %4) : (!llvm.ptr, i32) -> i32
    %i1 = llvm.icmp "slt" %4, %top : i32
    llvm.cond_br %i1, ^bb2(%4 : i32), ^bb3(%2 : i32)
  ^bb2(%5 : i32):
    %printf2 = llvm.call @printf(%addr, %5) : (!llvm.ptr, i32) -> i32
    llvm.br ^bb1(%3 , %5 : i32, i32)
  ^bb3(%9 : i32):
    %printf3 = llvm.call @printf(%addr, %9) : (!llvm.ptr, i32) -> i32
    llvm.return %0 : i32
  }
}
