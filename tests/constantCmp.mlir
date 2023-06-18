// RUN: %monaco -p isel %s | FileCheck %s

module {
  func.func private @main() -> i1 {
	%0 = arith.constant 2 : i64
	%1 = arith.constant 3 : i64
    %2 = "arith.cmpi"(%0, %1) <{predicate = 5 : i64}> : (i64, i64) -> i1 // 5 measn >= , i.e. 2 >= 3 results in false
    cf.cond_br %2, ^bb1, ^bb2
    // CHECK: JMP{{.*}}bb2
    ^bb1:
      return %2 : i1
    ^bb2:
      return %2 : i1
  }
}
