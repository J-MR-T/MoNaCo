// RUN: %monaco %s --jit main

module{
  llvm.func @main() -> i32 {
    // test if llvm.select works
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %false = llvm.mlir.constant(0 : i1) : i1
    %true = llvm.mlir.constant(1 : i1) : i1
    %3 = llvm.select %false, %1, %2 : i1, i32 // should be 2
    %check3 = llvm.sub %3, %2 : i32

    %4 = llvm.select %false, %2, %1 : i1, i32 // should be 1
    %check4 = llvm.sub %4, %1 : i32

    %5 = llvm.select %true, %1, %2 : i1, i32 // should be 1
    %check5 = llvm.sub %5, %1 : i32 // TODO doesnt work

    %6 = llvm.select %true, %2, %1 : i1, i32 // should be 2
    %check6 = llvm.sub %6, %2 : i32

    // multiply each check value with a prime, then add, so we know the result is unique, and nothing has cancelled out
    %p1 = llvm.mlir.constant(3 : i32) : i32
    %p2 = llvm.mlir.constant(5 : i32) : i32
    %p3 = llvm.mlir.constant(7 : i32) : i32
    %p4 = llvm.mlir.constant(11 : i32) : i32
    %checkp3 = llvm.mul %check3, %p1 : i32
    %checkp4 = llvm.mul %check4, %p2 : i32
    %checkp5 = llvm.mul %check5, %p3 : i32
    %checkp6 = llvm.mul %check6, %p4 : i32
    %sum = llvm.add %checkp3, %checkp4 : i32
    %sum2 = llvm.add %sum, %checkp5 : i32
    %sum3 = llvm.add %sum2, %checkp6 : i32
    llvm.return %sum3 : i32
  }
}
