// COM: only one of these continued run lines will ever work, and at least one should work. So run all 4, the sums of their exit codes should be 3, because one will work
// RUN: sum=0;                                                                                                  \
// RUN: %monaco -fno-codegen-dce -pasm %s | FileCheck --check-prefix CHECK-1 %s; sum=$(expr $(echo $?) + $sum); \
// RUN: %monaco -fno-codegen-dce -pasm %s | FileCheck --check-prefix CHECK-2 %s; sum=$(expr $(echo $?) + $sum); \
// RUN: %monaco -fno-codegen-dce -pasm %s | FileCheck --check-prefix CHECK-3 %s; sum=$(expr $(echo $?) + $sum); \
// RUN: [ $sum -eq 2 ]

// COM: Dot: https://dreampuf.github.io/GraphvizOnline/#digraph%20G%20%7B%0A%0A%20%20bb0%20-%3E%20bb1%20-%3E%20bb1%3B%0A%20%20bb0%20-%3E%20bb4%3B%0A%20%20bb1%20-%3E%20bb2%20-%3E%20bb3%20-%3E%20bb1%3B%0A%20%20bb2%20-%3E%20bb2%3B%0A%20%20bb4%20-%3E%20bb5%20-%3E%20bb6%20-%3E%20bb7%3B%0A%20%20bb5%20-%3E%20bb3%3B%0A%20%20bb4%20-%3E%20bb6%3B%0A%20%20%0A%20%20%0A%20%20bb1%2C%20bb2%2C%20bb3%20%5Bstyle%3Dfilled%2C%20fillcolor%20%3D%20yellow%5D%3B%0A%20%20%0A%0A%20%20bb0%20%5Bshape%3DMdiamond%5D%3B%0A%20%20bb7%20%5Bshape%3DMsquare%5D%3B%0A%7D

module{
  // 3 possible RPOs:
  // - 0,4,5,3,1,2,6,7
  // - 0,4,5,6,7,3,1,2
  // - 0,4,5,6,7,1,2,3
  func.func private @two(%10: i64, %11: i64) -> () {
    // CHECK-1: mov{{.*0x0$}}
    // CHECK-1: mov{{.*0x4$}}
    // CHECK-1: mov{{.*0x5$}}
    // CHECK-1: mov{{.*0x3$}}
    // CHECK-1: mov{{.*0x1$}}
    // CHECK-1: mov{{.*0x2$}}
    // CHECK-1: mov{{.*0x6$}}
    // CHECK-1: mov{{.*0x7$}}

    // CHECK-2: mov{{.*0x0$}}
    // CHECK-2: mov{{.*0x4$}}
    // CHECK-2: mov{{.*0x5$}}
    // CHECK-2: mov{{.*0x6$}}
    // CHECK-2: mov{{.*0x7$}}
    // CHECK-2: mov{{.*0x3$}}
    // CHECK-2: mov{{.*0x1$}}
    // CHECK-2: mov{{.*0x2$}}

    // CHECK-3: mov{{.*0x0$}}
    // CHECK-3: mov{{.*0x4$}}
    // CHECK-3: mov{{.*0x5$}}
    // CHECK-3: mov{{.*0x6$}}
    // CHECK-3: mov{{.*0x7$}}
    // CHECK-3: mov{{.*0x1$}}
    // CHECK-3: mov{{.*0x2$}}
    // CHECK-3: mov{{.*0x3$}}

    %0 = arith.cmpi slt, %10, %11 : i64
    %zero = arith.constant 0 : i64
    cf.cond_br %0, ^bb4, ^bb1
  ^bb1:
    %1 = arith.constant 1 : i64
    cf.cond_br %0, ^bb1, ^bb2
  ^bb3:
    %3 = arith.constant 3 : i64
    cf.br ^bb1
  ^bb2:
    %2 = arith.constant 2 : i64
    cf.cond_br %0, ^bb2, ^bb3
  ^bb4:
    %4 = arith.constant 4 : i64
    cf.cond_br %0, ^bb6, ^bb5
  ^bb5:
    %5 = arith.constant 5 : i64
    cf.cond_br %0, ^bb3, ^bb6
  ^bb6:
    %6 = arith.constant 6 : i64
    cf.br ^bb7
  ^bb7:
    %7 = arith.constant 7 : i64
    return
  }
}
