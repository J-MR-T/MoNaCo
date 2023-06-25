// RUN: %FileCheckAsm %s

module{
  func.func private @two(%10 : i64, %11 : i64) -> (i1) {
    %0 = arith.cmpi slt, %10, %11 : i64
    %7 = arith.cmpi ugt, %10, %11 : i64
    %8 = arith.subi %0, %7 : i1
    // CHECK: set
    // CHECK-NEXT: mov byte ptr [[[MEM1:rbp\-0x[0-9a-f]+]]],
    // CHECK: set
    // CHECK-NEXT: mov byte ptr [[[MEM2:rbp\-0x[0-9a-f]+]]],
    // CHECK: sub
    // CHECK-NEXT: mov byte ptr [[[MEM3:rbp\-0x[0-9a-f]+]]],

    // because of block reordering, we need to DAG match these next instructions, because the two sets can occur in any order

    // CHECK-DAG: mov [[REG4:[^,]+]], byte ptr [[[SLOT1:rbp\-0x[0-9a-f]+]]]
    // CHECK-DAG: cmov[[PRED:[a-z]+]] {{.*}}[[[MEM3]]]
    // CHECK-DAG: mov byte ptr [[[SLOT1]]], [[REG4]]
    // CHECK-DAG: j[[PRED]]{{.*}}BB[[BB1:[0-9]]]

    // CHECK-DAG: mov [[REG3:[^,]+]], byte ptr [[[MEM1]]]
    // CHECK-DAG: mov byte ptr [[[SLOT2:rbp\-0x[0-9a-f]+]]], [[REG3]]

    cf.cond_br %0, ^bb1(%8 : i1), ^bb2(%0 : i1)
  ^bb1(%1: i1):
    // CHECK: BB[[BB1]]:
    // CHECK-NEXT: mov al, byte ptr [[[SLOT1]]]
    // CHECK: ret
    return %1 : i1
  ^bb2(%2: i1):
    %3 = arith.constant 1 : i1
    %4 = arith.addi %2, %3 : i1
    cf.cond_br %4, ^bb1(%4 : i1), ^bb2(%4 : i1)
  }
}
