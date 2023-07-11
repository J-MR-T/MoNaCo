// commented out for the moment, while I'm checking what traversal is right
// RUN: %monaco -fno-codegen-dce -pasm %s | FileCheck %s

// Correct RPO is: entry/bb0, bb1, bb3, bb4, bb2. Unambiguous
module {
  func.func private @fn(%10: i64, %11: i64) -> i64 {
    %0 = arith.cmpi slt, %10, %11 : i64
    cf.br ^bb1
  ^bb1:
    // CHECK: BB1:
    // CHECK-NEXT: 0x1
	%1 = arith.constant 1 : i64
    cf.cond_br %0, ^bb2, ^bb3
// this has to come last, because of the CFG
  ^bb2:  // pred: ^bb1
    %2 = arith.constant 2 : i64
    return %2 : i64
  ^bb3:  // pred: ^bb1
    // CHECK: BB2:
    // CHECK-NEXT: 0x3
    %3 = arith.constant 3 : i64
    cf.cond_br %0, ^bb2, ^bb4
  ^bb4:
    // CHECK: BB3:
    // CHECK-NEXT: 0x4
    %4 = arith.constant 4 : i64
	cf.br ^bb2

  // this is where bb2 is
    // CHECK: BB4:
    // CHECK-NEXT: 0x2
  }
}

