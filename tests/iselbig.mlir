module {
  func.func private @giveI8() -> i8
  func.func private @giveI16() -> i16
  func.func private @giveI32() -> i32
  func.func private @giveI64() -> i64
  func.func private @patternMatchingTest() -> () {
    %0 = call @giveI8() : () -> i8
    %1 = call @giveI8() : () -> i8
    %2 = call @giveI16() : () -> i16
    %3 = call @giveI16() : () -> i16
    %4 = call @giveI32() : () -> i32
    %5 = call @giveI32() : () -> i32
    %6 = call @giveI64() : () -> i64
    %7 = call @giveI64() : () -> i64
    %8 = "arith.addi"(%0, %1) : (i8, i8) -> i8
    %9 = "arith.addi"(%2, %3) : (i16, i16) -> i16
    %10 = "arith.addi"(%4, %5) : (i32, i32) -> i32
    %11 = "arith.addi"(%6, %7) : (i64, i64) -> i64
    %12 = "arith.constant"() <{value = 8 : i8}> : () -> i8
    %13 = "arith.constant"() <{value = 16 : i16}> : () -> i16
    %14 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 64 : i64}> : () -> i64
    %16 = "arith.addi"(%12, %0) : (i8, i8) -> i8
    %17 = "arith.addi"(%13, %2) : (i16, i16) -> i16
    %18 = "arith.addi"(%14, %4) : (i32, i32) -> i32
    %19 = "arith.addi"(%15, %6) : (i64, i64) -> i64
    %20 = "arith.addi"(%19, %19) : (i64, i64) -> i64
    %21 = "arith.addi"(%0, %12) : (i8, i8) -> i8
    %22 = "arith.addi"(%2, %13) : (i16, i16) -> i16
    %23 = "arith.addi"(%4, %14) : (i32, i32) -> i32
    %24 = "arith.addi"(%6, %15) : (i64, i64) -> i64
    %25 = "arith.cmpi"(%19, %15) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %26 = "arith.extui"(%25) : (i1) -> i64
    %27 = "arith.constant"() <{value = 7 : i64}> : () -> i64
    %28 = "arith.andi"(%15, %27) : (i64, i64) -> i64
    %29 = "arith.shli"(%26, %28) : (i64, i64) -> i64
    "cf.cond_br"(%25)[^bb2, ^bb1] {operand_segment_sizes = array<i32: 1, 0, 0>} : (i1) -> ()
  ^bb1:  // 2 preds: ^bb0, ^bb1
    %30 = "arith.cmpi"(%26, %29) <{predicate = 6 : i64}> : (i64, i64) -> i1
    "cf.cond_br"(%30)[^bb2, ^bb1] {operand_segment_sizes = array<i32: 1, 0, 0>} : (i1) -> ()
  ^bb2:
    %31 = "arith.cmpi"(%26, %29) <{predicate = 0 : i64}> : (i64, i64) -> i1
    "cf.cond_br"(%31)[^bb1, ^bb2] {operand_segment_sizes = array<i32: 1, 0, 0>} : (i1) -> ()
  }
}
