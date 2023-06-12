// RUN: %monaco -p isel %s | FileCheck --check-prefix ISEL %s
// RUN: %FileCheckAsm -vv --check-prefix ASM %s

module {
  // ISEL: giveI8
  // ISEL-SAME: amd64.gpr8
  func.func private @giveI8() -> i8 {
    %0 = arith.constant 9 : i8
    return %0 : i8
  }
  // ISEL: giveI16
  // ISEL-SAME: amd64.gpr16
  func.func private @giveI16() -> i16 {
    %0 = arith.constant 17 : i16
    return %0 : i16
  }
  // ISEL: giveI32
  // ISEL-SAME: amd64.gpr32
  func.func private @giveI32() -> i32 {
    %0 = arith.constant 33 : i32
    return %0 : i32
  }
  // ISEL: giveI64
  // ISEL-SAME: amd64.gpr64
  func.func private @giveI64() -> i64 {
    %0 = arith.constant 65 : i64
    return %0 : i64
  }
  // ASM-COUNT-4: ret
  // TODO test more stuff on the individual instructions
  func.func private @patternMatchingTest() -> () {
    %0 = call @giveI8() : () -> i8
    // ASM: mov byte ptr [[[MEM1:rbp\-0x[0-9a-f]+]]], al
    %1 = call @giveI8() : () -> i8
    // ASM-NEXT: call
    // ASM-NEXT: mov byte ptr [[[MEM2:rbp\-0x[0-9a-f]+]]], al
    %2 = call @giveI16() : () -> i16
    %3 = call @giveI16() : () -> i16
    %4 = call @giveI32() : () -> i32
    %5 = call @giveI32() : () -> i32
    %6 = call @giveI64() : () -> i64
    %7 = call @giveI64() : () -> i64
    // ASM: mov [[REG1:[^,]+]], byte ptr [[[MEM1]]]
    // ASM-NEXT: mov [[REG2:[^,]+]], byte ptr [[[MEM2]]]
    %8 = "arith.addi"(%0, %1) : (i8, i8) -> i8
    // ASM-NEXT: add [[REG1]], [[REG2]]
    %9 = "arith.addi"(%2, %3) : (i16, i16) -> i16
    %10 = "arith.addi"(%4, %5) : (i32, i32) -> i32
    %11 = "arith.addi"(%6, %7) : (i64, i64) -> i64
    %12 = "arith.constant"() <{value = 8 : i8}> : () -> i8
    %13 = "arith.constant"() <{value = 16 : i16}> : () -> i16
    %14 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    // ISEL: [[VAL1:%[0-9]+]] = {{.*}}MOV64ri{{.*}}instructionInfo{{.*}}64]}>
    // ASM: mov{{.*}}, 0x40
    // ASM-NEXT: mov qword ptr [[[MEM1:rbp\-0x[0-9a-f]+]]]
    %15 = "arith.constant"() <{value = 64 : i64}> : () -> i64
    %16 = "arith.addi"(%12, %0) : (i8, i8) -> i8
    %17 = "arith.addi"(%13, %2) : (i16, i16) -> i16
    %18 = "arith.addi"(%14, %4) : (i32, i32) -> i32
    // ISEL: [[VAL2:%[0-9]+]] = {{.*}}ADD64{{.*}}{{[[VAL1]]|64}}{{.*}}
    // ASM: add r
    // ASM-NEXT: mov qword ptr [[[MEM2:rbp\-0x[0-9a-f]+]]]
    %19 = "arith.addi"(%15, %6) : (i64, i64) -> i64
    // ASM: mov [[REG1:r[0-9a-z]+]], qword ptr [[[MEM2]]]
    // ASM-NEXT: mov [[REG2:r[0-9a-z]+]], qword ptr [[[MEM2]]]
    // ASM-NEXT: add [[REG1]], [[REG2]]
    %20 = "arith.addi"(%19, %19) : (i64, i64) -> i64
    %21 = "arith.addi"(%0, %12) : (i8, i8) -> i8
    %22 = "arith.addi"(%2, %13) : (i16, i16) -> i16
    %23 = "arith.addi"(%4, %14) : (i32, i32) -> i32
    %24 = "arith.addi"(%6, %15) : (i64, i64) -> i64
    // ASM: mov [[REG1:r[0-9a-z]+]], qword ptr [[[MEM2]]]
    // ASM-NEXT: mov [[REG2:r[0-9a-z]+]], qword ptr [[[MEM1]]]
    // ISEL: CMP64rr{{.*}}[[VAL2]], [[VAL1]]
    // TODO this is where it fails, because there's a cmp rax, rax for some reason
    // COM: ASM-NEXT: cmp [[REG1]], [[REG2]]
    %25 = "arith.cmpi"(%19, %15) <{predicate = 5 : i64}> : (i64, i64) -> i1
    // ISEL-NEXT: SETGE{{.*}}gpr8
    // ISEL-NEXT: [[VAL3:%[0-9]+]] = {{.*}}MOVZXr64r8
    %26 = "arith.extui"(%25) : (i1) -> i64
    %27 = "arith.constant"() <{value = 7 : i64}> : () -> i64
    %28 = "arith.andi"(%19, %27) : (i64, i64) -> i64
    // ISEL: SHL{{.*}}[[VAL3]]
    %29 = "arith.shli"(%26, %28) : (i64, i64) -> i64
    // ISEL: CMP64rr{{.*}}[[VAL2]], [[VAL1]]
    "cf.cond_br"(%25)[^bb2, ^bb1] {operand_segment_sizes = array<i32: 1, 0, 0>} : (i1) -> ()
    // ISEL-NEXT: JGE
  ^bb1:  // 2 preds: ^bb0, ^bb1
    %30 = "arith.cmpi"(%26, %29) <{predicate = 6 : i64}> : (i64, i64) -> i1
    "cf.cond_br"(%30)[^bb2, ^bb1] {operand_segment_sizes = array<i32: 1, 0, 0>} : (i1) -> ()
  ^bb2:
    %31 = "arith.cmpi"(%26, %29) <{predicate = 0 : i64}> : (i64, i64) -> i1
    "cf.cond_br"(%31)[^bb1, ^bb2] {operand_segment_sizes = array<i32: 1, 0, 0>} : (i1) -> ()
  }
}
