// RUN: %monaco %s --jit main | FileCheck --match-full-lines %s

module{
  // external function to take addr of
  llvm.func @putchar(i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind"]}

  // function before call to take addr of
  llvm.func @myputchar(%0 : i32) -> i32{
    %1 = llvm.call @putchar(%0) : (i32) -> i32
    llvm.return %1 : i32
  }
  llvm.func @main(){
    // indirect calls with all 3 cases, output 'a', 'b', 'c', then direct call to '\n'
    // CHECK: abc
    %a = llvm.mlir.constant (97 : i32) : i32
    %b = llvm.mlir.constant (98 : i32) : i32
    %c = llvm.mlir.constant (99 : i32) : i32
    %n = llvm.mlir.constant (10 : i32) : i32

    %0 = llvm.mlir.addressof @putchar : !llvm.ptr
    %1 = llvm.call %0(%a) : !llvm.ptr, (i32) -> i32

    %2 = llvm.mlir.addressof @myputchar : !llvm.ptr
    %3 = llvm.call %2(%b) : !llvm.ptr, (i32) -> i32

    %4 = llvm.mlir.addressof @myputchar2 : !llvm.ptr
    %5 = llvm.call %4(%c) : !llvm.ptr, (i32) -> i32

    %6 = llvm.call @putchar(%n) : (i32) -> i32
    llvm.return
  }
  // function after call to take addr of
  llvm.func @myputchar2(%0 : i32) -> i32{
    %1 = llvm.call @putchar(%0) : (i32) -> i32
    llvm.return %1 : i32
  }
}
