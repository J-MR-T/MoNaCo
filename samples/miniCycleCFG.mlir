module {
  func.func private @main() -> i64 {
	%0 = arith.constant 0 : i1
    cf.br ^bb1
  ^bb1:
    cf.cond_br %0, ^bb1, ^bb2
  ^bb2:  // pred: ^bb1
    %c48_i64 = arith.constant 48 : i64
    return %c48_i64 : i64
  }
}

