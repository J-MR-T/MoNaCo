// RUN: %monaco %s --jit main

module{
  llvm.func @main() -> i32{
    // initialization
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.alloca %1 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %1 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %0, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %1, %4 {alignment = 4 : i64} : i32, !llvm.ptr

    // memcpy
    %len = llvm.mlir.constant(4 : i32) : i32
    "llvm.intr.memcpy" (%4, %3, %len) {isVolatile = 0 : i1} : (!llvm.ptr, !llvm.ptr, i32) -> () // args seem to be dst, src, len, isVolatile
    %5 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32 // this should be 0
    // try memset as well

    // set it to 0x01010101
    %trunced = llvm.trunc %1 : i32 to i8
    "llvm.intr.memset"(%3, %trunced, %len) {isVolatile = 0 : i1} : (!llvm.ptr, i8, i32) -> ()
    // sub 0x01010101
    %6 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %7 = llvm.mlir.constant(16843009 : i32) : i32
    %8 = llvm.sub %6, %7 : i32
    // then or with the previous result, should be 0
    %9 = llvm.or %8, %5 : i32

    llvm.return %9 : i32
  }
}
