// RUN: %FileCheckExecReturnStatus %s

// CHECK: 119

module {
  func.func private @main() -> i64 {
    %c2000_i64 = arith.constant 2000 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0_i64_0 = arith.constant 0 : i64
    %c0_i64_1 = arith.constant 0 : i64
    %c0_i64_2 = arith.constant 0 : i64
    %c0_i64_3 = arith.constant 0 : i64
    %c0_i64_4 = arith.constant 0 : i64
    cf.br ^bb1(%c2000_i64, %c0_i64_4, %c1_i64, %c0_i64_3 : i64, i64, i64, i64)
  ^bb1(%0: i64, %1: i64, %2: i64, %3: i64):  // 2 preds: ^bb0, ^bb5
    %c2_i64 = arith.constant 2 : i64
    %4 = arith.muli %c2_i64, %0 : i64
    %5 = arith.cmpi slt, %1, %4 : i64
    %6 = arith.extui %5 : i1 to i64
    %c0_i64_5 = arith.constant 0 : i64
    %7 = arith.cmpi ne, %6, %c0_i64_5 : i64
    cf.cond_br %7, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %c1_i64_6 = arith.constant 1 : i64
    %8 = arith.addi %0, %c1_i64_6 : i64
    %c0_i64_7 = arith.constant 0 : i64
    cf.br ^bb3(%0, %c0_i64_7, %2, %c0_i64_7, %8, %3, %1 : i64, i64, i64, i64, i64, i64, i64)
  ^bb3(%9: i64, %10: i64, %11: i64, %12: i64, %13: i64, %14: i64, %15: i64):  // 2 preds: ^bb2, ^bb4
    %16 = arith.cmpi slt, %10, %9 : i64
    %17 = arith.extui %16 : i1 to i64
    %c0_i64_8 = arith.constant 0 : i64
    %18 = arith.cmpi ne, %17, %c0_i64_8 : i64
    cf.cond_br %18, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %c10_i64 = arith.constant 10 : i64
    %19 = arith.muli %11, %c10_i64 : i64
    %20 = arith.addi %12, %19 : i64
    %21 = arith.remsi %20, %13 : i64
    %c1_i64_9 = arith.constant 1 : i64
    %22 = arith.addi %10, %c1_i64_9 : i64
    %23 = arith.divsi %20, %13 : i64
    %c1_i64_10 = arith.constant 1 : i64
    %24 = arith.subi %13, %c1_i64_10 : i64
    %25 = arith.addi %14, %21 : i64
    cf.br ^bb3(%9, %22, %21, %23, %24, %25, %15 : i64, i64, i64, i64, i64, i64, i64)
  ^bb5:  // pred: ^bb3
    %c1_i64_11 = arith.constant 1 : i64
    %26 = arith.addi %15, %c1_i64_11 : i64
    cf.br ^bb1(%9, %26, %11, %14 : i64, i64, i64, i64)
  ^bb6:  // pred: ^bb1
    return %3 : i64
  }
}

