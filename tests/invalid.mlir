// RUN: not %monaco

module {
  func.func private @main(%0: i64   ) -> i1 {
      cf.cond_br %2, ^bb0, ^bb1
    ^bb1:
      return %2 : i1
  }
}
