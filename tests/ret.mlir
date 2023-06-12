// RUN: %monaco %s

module {
  func.func private @giveI8() -> i8 {
    %0 = arith.constant 9 : i8
    return %0 : i8
  }
  func.func private @patternMatchingTest() -> () {
    %0 = call @giveI8() : () -> i8
    return
  }
}
