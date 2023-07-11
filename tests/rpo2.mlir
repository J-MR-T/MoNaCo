// COM: only one of these continued run lines will ever work, and at least one should work. So run all 4, the sums of their exit codes should be 3, because one will work
// RUN: sum=0;                                                                                                  \
// RUN: %monaco -fno-codegen-dce -pasm %s | FileCheck --check-prefix CHECK-1 %s; sum=$(expr $(echo $?) + $sum); \
// RUN: %monaco -fno-codegen-dce -pasm %s | FileCheck --check-prefix CHECK-2 %s; sum=$(expr $(echo $?) + $sum); \
// RUN: %monaco -fno-codegen-dce -pasm %s | FileCheck --check-prefix CHECK-3 %s; sum=$(expr $(echo $?) + $sum); \
// RUN: %monaco -fno-codegen-dce -pasm %s | FileCheck --check-prefix CHECK-4 %s; sum=$(expr $(echo $?) + $sum); \
// RUN: [ $sum -eq 3 ]

// COM: Dot: https://dreampuf.github.io/GraphvizOnline/#digraph%20G%20%7B%0A%0A%20%20bb0%20-%3E%20bb1%3B%0A%20%20bb0%20-%3E%20bb2%3B%0A%20%20bb1%20-%3E%20bb2%3B%0A%20%20bb2%20-%3E%20bb3%20-%3E%20bb4%20-%3E%20bb2%3B%0A%20%20bb2%20-%3E%20bb5%3B%0A%20%20bb1%20-%3E%20bb6%3B%0A%0A%20%20bb0%20%5Bshape%3DMdiamond%5D%3B%0A%20%20bb5%2C%20bb6%20%5Bshape%3DMsquare%5D%3B%0A%7D

module{
  // 4 possible RPOs:
  // - 0,1,6,2,3,4,5
  // - 0,1,6,2,5,3,4
  // - 0,1,2,3,4,5,6
  // - 0,1,2,5,3,4,6
  func.func private @two(%10: i64, %11: i64) -> () {
    // CHECK-1: mov{{.*0x0$}}
    // CHECK-1: mov{{.*0x1$}}
    // CHECK-1: mov{{.*0x6$}}
    // CHECK-1: mov{{.*0x2$}}
    // CHECK-1: mov{{.*0x3$}}
    // CHECK-1: mov{{.*0x4$}}
    // CHECK-1: mov{{.*0x5$}}

    // CHECK-2: mov{{.*0x0$}}
    // CHECK-2: mov{{.*0x1$}}
    // CHECK-2: mov{{.*0x6$}}
    // CHECK-2: mov{{.*0x2$}}
    // CHECK-2: mov{{.*0x5$}}
    // CHECK-2: mov{{.*0x3$}}
    // CHECK-2: mov{{.*0x4$}}

    // CHECK-3: mov{{.*0x0$}}
    // CHECK-3: mov{{.*0x1$}}
    // CHECK-3: mov{{.*0x2$}}
    // CHECK-3: mov{{.*0x3$}}
    // CHECK-3: mov{{.*0x4$}}
    // CHECK-3: mov{{.*0x5$}}
    // CHECK-3: mov{{.*0x6$}}

    // CHECK-4: mov{{.*0x0$}}
    // CHECK-4: mov{{.*0x1$}}
    // CHECK-4: mov{{.*0x2$}}
    // CHECK-4: mov{{.*0x5$}}
    // CHECK-4: mov{{.*0x3$}}
    // CHECK-4: mov{{.*0x4$}}
    // CHECK-4: mov{{.*0x6$}}

    %0 = arith.cmpi slt, %10, %11 : i64
    %zero = arith.constant 0 : i64
    cf.cond_br %0, ^bb2, ^bb1
  ^bb1:
    %1 = arith.constant 1 : i64
    cf.cond_br %0, ^bb2, ^bb6
  ^bb3:
    %3 = arith.constant 3 : i64
    cf.br ^bb4
  ^bb2:
    %2 = arith.constant 2 : i64
    cf.cond_br %0, ^bb5, ^bb3
  ^bb4:
    %4 = arith.constant 4 : i64
    cf.br ^bb2
  ^bb5:
    %5 = arith.constant 5 : i64
    return
  ^bb6:
    %6 = arith.constant 6 : i64
    return
  }
}
