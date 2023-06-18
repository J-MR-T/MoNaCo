// RUN: %FileCheckAsm %s

// CHECK-NOT: cmp [[REG1:[a-z]+]], [[REG1]]
module {
  func.func private @main(%0: i64   ) -> i1 {
	%1 = arith.constant 3 : i64
    %2 = "arith.cmpi"(%0, %1) <{predicate = 5 : i64}> : (i64, i64) -> i1
    cf.cond_br %2, ^bb1, ^bb2
    ^bb1:
      return %2 : i1
    ^bb2:
      return %2 : i1
  }
}
