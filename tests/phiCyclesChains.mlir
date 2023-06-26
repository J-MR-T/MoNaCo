// RUN: %FileCheckAsm %s

module {
  func.func private @main() -> () {
    %1 = arith.constant 1 : i64
    // CHECK: mov [[REG1:[^,]+]], 0x1
    // CHECK-NEXT: mov qword ptr [[[MEM1:rbp\-0x[0-9a-f]+]]], [[REG1]]
    // CHECK-NEXT: mov [[REG2:[^,]+]], 0x2
    // CHECK-NEXT: mov qword ptr [[[MEM2:rbp\-0x[0-9a-f]+]]], [[REG2]]
    // CHECK-NEXT: mov [[REG3:[^,]+]], 0x3
    // CHECK-NEXT: mov qword ptr [[[MEM3:rbp\-0x[0-9a-f]+]]], [[REG3]]
    // CHECK-NEXT: mov [[REG4:[^,]+]], 0x4
    // CHECK-NEXT: mov qword ptr [[[MEM4:rbp\-0x[0-9a-f]+]]], [[REG4]]
    // CHECK-NEXT: mov [[REG5:[^,]+]], 0x5
    // CHECK-NEXT: mov qword ptr [[[MEM5:rbp\-0x[0-9a-f]+]]], [[REG5]]
    %2 = arith.constant 2 : i64
    %3 = arith.constant 3 : i64
    %4 = arith.constant 4 : i64
    %5 = arith.constant 5 : i64

    // CHECK: mov {{.*}}, qword ptr [[[MEM1]]]
    // CHECK-NEXT: mov qword ptr [[[SLOT1:rbp\-0x[0-9a-f]+]]]
    // CHECK: mov {{.*}}, qword ptr [[[MEM2]]]
    // CHECK-NEXT: mov qword ptr [[[SLOT2:rbp\-0x[0-9a-f]+]]]
    // CHECK: mov {{.*}}, qword ptr [[[MEM3]]]
    // CHECK-NEXT: mov qword ptr [[[SLOT3:rbp\-0x[0-9a-f]+]]]
    // CHECK: mov {{.*}}, qword ptr [[[MEM4]]]
    // CHECK-NEXT: mov qword ptr [[[SLOT4:rbp\-0x[0-9a-f]+]]]
    // CHECK: mov {{.*}}, qword ptr [[[MEM5]]]
    // CHECK-NEXT: mov qword ptr [[[SLOT5:rbp\-0x[0-9a-f]+]]]
    cf.br ^bb1(%1, %2, %3, %4, %5 : i64, i64, i64, i64, i64)
    ^bb1(%11 : i64, %12 : i64, %13 : i64, %14 : i64, %15 : i64):
      // phi chain/cycle graph: https://dreampuf.github.io/GraphvizOnline/#digraph%20G%20%7B%0A%20%20%20%20subgraph%7B%0A%20%20%20%20%20%20%20%20rank%20%3D%20same%3B%0A%20%20%20%20%20%20%20%201%20-%3E%205%3B%0A%20%20%20%20%20%20%20%205%20-%3E%202%3B%0A%20%20%20%20%20%20%20%202%20-%3E%201%3B%0A%20%20%20%20%7D%0A%20%20%20%203%20-%3E%201%3B%0A%20%20%20%204%20-%3E%202%3B%0A%20%20%20%20%0A%20%20%20%201%20%5Blabel%20%3D%20%22%26phi%3B%E2%82%81%22%5D%3B%0A%20%20%20%202%20%5Blabel%20%3D%20%22%26phi%3B%E2%82%82%22%5D%3B%0A%20%20%20%203%20%5Blabel%20%3D%20%22%26phi%3B%E2%82%83%22%5D%3B%0A%20%20%20%204%20%5Blabel%20%3D%20%22%26phi%3B%E2%82%84%22%5D%3B%0A%20%20%20%205%20%5Blabel%20%3D%20%22%26phi%3B%E2%82%85%22%5D%3B%0A%7D
      // -> correct ordering: 3 <- 1, 4 <- 1 in any order, then the loop, which we'll assume starts at 1, making this implementation independent would be a bit much
      // CHECK: BB
      // CHECK-DAG: mov [[REG1:[^,]+]], qword ptr [[[SLOT1]]]
      // CHECK-DAG: mov qword ptr [[[SLOT3]]], [[REG1]]
      // CHECK-DAG: mov [[REG2:[^,]+]], qword ptr [[[SLOT2]]]
      // CHECK-DAG: mov qword ptr [[[SLOT4]]], [[REG2]]

      // first check that one is copied
      // CHECK: mov [[REG_BREAK:[^,]+]], qword ptr [[[SLOT1]]]
      // CHECK-NEXT: [[[SLOT5]]]
      // CHECK-NEXT: [[[SLOT1]]]
      // CHECK-NEXT: [[[SLOT2]]]
      // CHECK-NEXT: [[[SLOT5]]]
      // CHECK-NEXT: qword ptr [[[SLOT2]]], [[REG_BREAK]]
      cf.br ^bb1(%15, %11, %11, %12, %12 : i64, i64, i64, i64, i64)
  }
}
