// RUN: %monaco %s --jit main

module{
  llvm.mlir.global internal unnamed_addr @opt(false) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : i1
  llvm.func @main() -> i32 {
      // get addrof global, do cond br based on it
      %0 = llvm.mlir.addressof @opt : !llvm.ptr
      %1 = llvm.load %0 {alignment = 8 : i64} : !llvm.ptr -> i1
      llvm.cond_br %1, ^bb1, ^bb2
    ^bb1:
      %2 = llvm.mlir.constant(1 : i32) : i32
      llvm.return %2 : i32
    ^bb2:
      %3 = llvm.mlir.constant(0 : i32) : i32
      llvm.return %3 : i32
  }
}
