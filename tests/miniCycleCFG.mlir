// RUN: %monaco -fno-codegen-dce -pasm %s | FileCheck %s

// correct RPO is bb0, bb1, bb2, and it shouldn't die doing it :)
module {
  func.func private @main() -> i64 {
    // CHECK: 0x0
	%0 = arith.constant 0 : i1
    // CHECK: 0x1
    // CHECK: 0x2
    cf.br ^bb1
  ^bb2:  // pred: ^bb1
	%2 = arith.constant 2 : i64
    return %2 : i64
  ^bb1:
	%1 = arith.constant 1 : i64
    cf.cond_br %0, ^bb1, ^bb2
  }
}

