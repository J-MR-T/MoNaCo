// RUN: %FileCheckAsm %s

// Correct RPO is: entry/bb0, bb1, bb3, bb4, bb2. Unambiguous
module {
  func.func private @main() -> i64 {
    // CHECK: 0x0
	%0 = arith.constant 0 : i1
    cf.br ^bb1
  ^bb1:
    // CHECK: 0x1
	%1 = arith.constant 1 : i64
    cf.cond_br %0, ^bb2, ^bb3
// this has to come last, because of the CFG
  ^bb2:  // pred: ^bb1
    %2 = arith.constant 2 : i64
    return %2 : i64
  ^bb3:  // pred: ^bb1
    // CHECK: 0x3
    %3 = arith.constant 3 : i64
    cf.cond_br %0, ^bb2, ^bb4
  ^bb4:
    // CHECK: 0x4
    %4 = arith.constant 4 : i64
	cf.br ^bb2

  // this is where bb2 is
    // CHECK: 0x2
  }
}

