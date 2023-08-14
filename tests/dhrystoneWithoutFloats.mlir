// RUN: echo 999 | ASAN_OPTIONS=detect_leaks=0 %monaco %s --jit main | FileCheck %s
// 999 runs

// CHECK: Final values of the variables used in the benchmark:
// CHECK-EMPTY:
// CHECK-NEXT: [[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: [[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: [[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: [[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: [[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: 1009{{$}}
// CHECK-NEXT: Number_Of_Runs + 10
// CHECK: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}
// CHECK-NEXT: :{{ *}}[[CHECKVAL:   .*]]{{$}}
// CHECK-NEXT: [[CHECKVAL]]{{$}}

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.mlir.global external local_unnamed_addr @Reg(1 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global external local_unnamed_addr @Float_Rating(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global external local_unnamed_addr @Next_Ptr_Glob() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.null : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global external local_unnamed_addr @Ptr_Glob() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.ptr {
    %0 = llvm.mlir.null : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global private unnamed_addr constant @".str"("DHRYSTONE PROGRAM, SOME STRING\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.1"("DHRYSTONE PROGRAM, 1'ST STRING\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external @Arr_2_Glob() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<50 x array<50 x i32>> {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.undef : !llvm.array<50 x i32>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<50 x i32> 
    %3 = llvm.insertvalue %0, %2[1] : !llvm.array<50 x i32> 
    %4 = llvm.insertvalue %0, %3[2] : !llvm.array<50 x i32> 
    %5 = llvm.insertvalue %0, %4[3] : !llvm.array<50 x i32> 
    %6 = llvm.insertvalue %0, %5[4] : !llvm.array<50 x i32> 
    %7 = llvm.insertvalue %0, %6[5] : !llvm.array<50 x i32> 
    %8 = llvm.insertvalue %0, %7[6] : !llvm.array<50 x i32> 
    %9 = llvm.insertvalue %0, %8[7] : !llvm.array<50 x i32> 
    %10 = llvm.insertvalue %0, %9[8] : !llvm.array<50 x i32> 
    %11 = llvm.insertvalue %0, %10[9] : !llvm.array<50 x i32> 
    %12 = llvm.insertvalue %0, %11[10] : !llvm.array<50 x i32> 
    %13 = llvm.insertvalue %0, %12[11] : !llvm.array<50 x i32> 
    %14 = llvm.insertvalue %0, %13[12] : !llvm.array<50 x i32> 
    %15 = llvm.insertvalue %0, %14[13] : !llvm.array<50 x i32> 
    %16 = llvm.insertvalue %0, %15[14] : !llvm.array<50 x i32> 
    %17 = llvm.insertvalue %0, %16[15] : !llvm.array<50 x i32> 
    %18 = llvm.insertvalue %0, %17[16] : !llvm.array<50 x i32> 
    %19 = llvm.insertvalue %0, %18[17] : !llvm.array<50 x i32> 
    %20 = llvm.insertvalue %0, %19[18] : !llvm.array<50 x i32> 
    %21 = llvm.insertvalue %0, %20[19] : !llvm.array<50 x i32> 
    %22 = llvm.insertvalue %0, %21[20] : !llvm.array<50 x i32> 
    %23 = llvm.insertvalue %0, %22[21] : !llvm.array<50 x i32> 
    %24 = llvm.insertvalue %0, %23[22] : !llvm.array<50 x i32> 
    %25 = llvm.insertvalue %0, %24[23] : !llvm.array<50 x i32> 
    %26 = llvm.insertvalue %0, %25[24] : !llvm.array<50 x i32> 
    %27 = llvm.insertvalue %0, %26[25] : !llvm.array<50 x i32> 
    %28 = llvm.insertvalue %0, %27[26] : !llvm.array<50 x i32> 
    %29 = llvm.insertvalue %0, %28[27] : !llvm.array<50 x i32> 
    %30 = llvm.insertvalue %0, %29[28] : !llvm.array<50 x i32> 
    %31 = llvm.insertvalue %0, %30[29] : !llvm.array<50 x i32> 
    %32 = llvm.insertvalue %0, %31[30] : !llvm.array<50 x i32> 
    %33 = llvm.insertvalue %0, %32[31] : !llvm.array<50 x i32> 
    %34 = llvm.insertvalue %0, %33[32] : !llvm.array<50 x i32> 
    %35 = llvm.insertvalue %0, %34[33] : !llvm.array<50 x i32> 
    %36 = llvm.insertvalue %0, %35[34] : !llvm.array<50 x i32> 
    %37 = llvm.insertvalue %0, %36[35] : !llvm.array<50 x i32> 
    %38 = llvm.insertvalue %0, %37[36] : !llvm.array<50 x i32> 
    %39 = llvm.insertvalue %0, %38[37] : !llvm.array<50 x i32> 
    %40 = llvm.insertvalue %0, %39[38] : !llvm.array<50 x i32> 
    %41 = llvm.insertvalue %0, %40[39] : !llvm.array<50 x i32> 
    %42 = llvm.insertvalue %0, %41[40] : !llvm.array<50 x i32> 
    %43 = llvm.insertvalue %0, %42[41] : !llvm.array<50 x i32> 
    %44 = llvm.insertvalue %0, %43[42] : !llvm.array<50 x i32> 
    %45 = llvm.insertvalue %0, %44[43] : !llvm.array<50 x i32> 
    %46 = llvm.insertvalue %0, %45[44] : !llvm.array<50 x i32> 
    %47 = llvm.insertvalue %0, %46[45] : !llvm.array<50 x i32> 
    %48 = llvm.insertvalue %0, %47[46] : !llvm.array<50 x i32> 
    %49 = llvm.insertvalue %0, %48[47] : !llvm.array<50 x i32> 
    %50 = llvm.insertvalue %0, %49[48] : !llvm.array<50 x i32> 
    %51 = llvm.insertvalue %0, %50[49] : !llvm.array<50 x i32> 
    %52 = llvm.mlir.undef : !llvm.array<50 x array<50 x i32>>
    %53 = llvm.insertvalue %51, %52[0] : !llvm.array<50 x array<50 x i32>> 
    %54 = llvm.insertvalue %51, %53[1] : !llvm.array<50 x array<50 x i32>> 
    %55 = llvm.insertvalue %51, %54[2] : !llvm.array<50 x array<50 x i32>> 
    %56 = llvm.insertvalue %51, %55[3] : !llvm.array<50 x array<50 x i32>> 
    %57 = llvm.insertvalue %51, %56[4] : !llvm.array<50 x array<50 x i32>> 
    %58 = llvm.insertvalue %51, %57[5] : !llvm.array<50 x array<50 x i32>> 
    %59 = llvm.insertvalue %51, %58[6] : !llvm.array<50 x array<50 x i32>> 
    %60 = llvm.insertvalue %51, %59[7] : !llvm.array<50 x array<50 x i32>> 
    %61 = llvm.insertvalue %51, %60[8] : !llvm.array<50 x array<50 x i32>> 
    %62 = llvm.insertvalue %51, %61[9] : !llvm.array<50 x array<50 x i32>> 
    %63 = llvm.insertvalue %51, %62[10] : !llvm.array<50 x array<50 x i32>> 
    %64 = llvm.insertvalue %51, %63[11] : !llvm.array<50 x array<50 x i32>> 
    %65 = llvm.insertvalue %51, %64[12] : !llvm.array<50 x array<50 x i32>> 
    %66 = llvm.insertvalue %51, %65[13] : !llvm.array<50 x array<50 x i32>> 
    %67 = llvm.insertvalue %51, %66[14] : !llvm.array<50 x array<50 x i32>> 
    %68 = llvm.insertvalue %51, %67[15] : !llvm.array<50 x array<50 x i32>> 
    %69 = llvm.insertvalue %51, %68[16] : !llvm.array<50 x array<50 x i32>> 
    %70 = llvm.insertvalue %51, %69[17] : !llvm.array<50 x array<50 x i32>> 
    %71 = llvm.insertvalue %51, %70[18] : !llvm.array<50 x array<50 x i32>> 
    %72 = llvm.insertvalue %51, %71[19] : !llvm.array<50 x array<50 x i32>> 
    %73 = llvm.insertvalue %51, %72[20] : !llvm.array<50 x array<50 x i32>> 
    %74 = llvm.insertvalue %51, %73[21] : !llvm.array<50 x array<50 x i32>> 
    %75 = llvm.insertvalue %51, %74[22] : !llvm.array<50 x array<50 x i32>> 
    %76 = llvm.insertvalue %51, %75[23] : !llvm.array<50 x array<50 x i32>> 
    %77 = llvm.insertvalue %51, %76[24] : !llvm.array<50 x array<50 x i32>> 
    %78 = llvm.insertvalue %51, %77[25] : !llvm.array<50 x array<50 x i32>> 
    %79 = llvm.insertvalue %51, %78[26] : !llvm.array<50 x array<50 x i32>> 
    %80 = llvm.insertvalue %51, %79[27] : !llvm.array<50 x array<50 x i32>> 
    %81 = llvm.insertvalue %51, %80[28] : !llvm.array<50 x array<50 x i32>> 
    %82 = llvm.insertvalue %51, %81[29] : !llvm.array<50 x array<50 x i32>> 
    %83 = llvm.insertvalue %51, %82[30] : !llvm.array<50 x array<50 x i32>> 
    %84 = llvm.insertvalue %51, %83[31] : !llvm.array<50 x array<50 x i32>> 
    %85 = llvm.insertvalue %51, %84[32] : !llvm.array<50 x array<50 x i32>> 
    %86 = llvm.insertvalue %51, %85[33] : !llvm.array<50 x array<50 x i32>> 
    %87 = llvm.insertvalue %51, %86[34] : !llvm.array<50 x array<50 x i32>> 
    %88 = llvm.insertvalue %51, %87[35] : !llvm.array<50 x array<50 x i32>> 
    %89 = llvm.insertvalue %51, %88[36] : !llvm.array<50 x array<50 x i32>> 
    %90 = llvm.insertvalue %51, %89[37] : !llvm.array<50 x array<50 x i32>> 
    %91 = llvm.insertvalue %51, %90[38] : !llvm.array<50 x array<50 x i32>> 
    %92 = llvm.insertvalue %51, %91[39] : !llvm.array<50 x array<50 x i32>> 
    %93 = llvm.insertvalue %51, %92[40] : !llvm.array<50 x array<50 x i32>> 
    %94 = llvm.insertvalue %51, %93[41] : !llvm.array<50 x array<50 x i32>> 
    %95 = llvm.insertvalue %51, %94[42] : !llvm.array<50 x array<50 x i32>> 
    %96 = llvm.insertvalue %51, %95[43] : !llvm.array<50 x array<50 x i32>> 
    %97 = llvm.insertvalue %51, %96[44] : !llvm.array<50 x array<50 x i32>> 
    %98 = llvm.insertvalue %51, %97[45] : !llvm.array<50 x array<50 x i32>> 
    %99 = llvm.insertvalue %51, %98[46] : !llvm.array<50 x array<50 x i32>> 
    %100 = llvm.insertvalue %51, %99[47] : !llvm.array<50 x array<50 x i32>> 
    %101 = llvm.insertvalue %51, %100[48] : !llvm.array<50 x array<50 x i32>> 
    %102 = llvm.insertvalue %51, %101[49] : !llvm.array<50 x array<50 x i32>> 
    llvm.return %102 : !llvm.array<50 x array<50 x i32>>
  }
  llvm.mlir.global private unnamed_addr constant @".str.6"("Ratings using 'float' datatype (%d bytes)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.7"("Ratings using 'double' datatype (%d bytes)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.8"("HZ = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.9"("Please give the number of runs through the benchmark: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.10"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.11"("Execution starts, %d runs through Dhrystone\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @Begin_Time(0 : i64) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : i64
  llvm.mlir.global private unnamed_addr constant @".str.12"("DHRYSTONE PROGRAM, 2'ND STRING\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @Bool_Glob(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global external @Arr_1_Glob() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<50 x i32> {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.undef : !llvm.array<50 x i32>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<50 x i32> 
    %3 = llvm.insertvalue %0, %2[1] : !llvm.array<50 x i32> 
    %4 = llvm.insertvalue %0, %3[2] : !llvm.array<50 x i32> 
    %5 = llvm.insertvalue %0, %4[3] : !llvm.array<50 x i32> 
    %6 = llvm.insertvalue %0, %5[4] : !llvm.array<50 x i32> 
    %7 = llvm.insertvalue %0, %6[5] : !llvm.array<50 x i32> 
    %8 = llvm.insertvalue %0, %7[6] : !llvm.array<50 x i32> 
    %9 = llvm.insertvalue %0, %8[7] : !llvm.array<50 x i32> 
    %10 = llvm.insertvalue %0, %9[8] : !llvm.array<50 x i32> 
    %11 = llvm.insertvalue %0, %10[9] : !llvm.array<50 x i32> 
    %12 = llvm.insertvalue %0, %11[10] : !llvm.array<50 x i32> 
    %13 = llvm.insertvalue %0, %12[11] : !llvm.array<50 x i32> 
    %14 = llvm.insertvalue %0, %13[12] : !llvm.array<50 x i32> 
    %15 = llvm.insertvalue %0, %14[13] : !llvm.array<50 x i32> 
    %16 = llvm.insertvalue %0, %15[14] : !llvm.array<50 x i32> 
    %17 = llvm.insertvalue %0, %16[15] : !llvm.array<50 x i32> 
    %18 = llvm.insertvalue %0, %17[16] : !llvm.array<50 x i32> 
    %19 = llvm.insertvalue %0, %18[17] : !llvm.array<50 x i32> 
    %20 = llvm.insertvalue %0, %19[18] : !llvm.array<50 x i32> 
    %21 = llvm.insertvalue %0, %20[19] : !llvm.array<50 x i32> 
    %22 = llvm.insertvalue %0, %21[20] : !llvm.array<50 x i32> 
    %23 = llvm.insertvalue %0, %22[21] : !llvm.array<50 x i32> 
    %24 = llvm.insertvalue %0, %23[22] : !llvm.array<50 x i32> 
    %25 = llvm.insertvalue %0, %24[23] : !llvm.array<50 x i32> 
    %26 = llvm.insertvalue %0, %25[24] : !llvm.array<50 x i32> 
    %27 = llvm.insertvalue %0, %26[25] : !llvm.array<50 x i32> 
    %28 = llvm.insertvalue %0, %27[26] : !llvm.array<50 x i32> 
    %29 = llvm.insertvalue %0, %28[27] : !llvm.array<50 x i32> 
    %30 = llvm.insertvalue %0, %29[28] : !llvm.array<50 x i32> 
    %31 = llvm.insertvalue %0, %30[29] : !llvm.array<50 x i32> 
    %32 = llvm.insertvalue %0, %31[30] : !llvm.array<50 x i32> 
    %33 = llvm.insertvalue %0, %32[31] : !llvm.array<50 x i32> 
    %34 = llvm.insertvalue %0, %33[32] : !llvm.array<50 x i32> 
    %35 = llvm.insertvalue %0, %34[33] : !llvm.array<50 x i32> 
    %36 = llvm.insertvalue %0, %35[34] : !llvm.array<50 x i32> 
    %37 = llvm.insertvalue %0, %36[35] : !llvm.array<50 x i32> 
    %38 = llvm.insertvalue %0, %37[36] : !llvm.array<50 x i32> 
    %39 = llvm.insertvalue %0, %38[37] : !llvm.array<50 x i32> 
    %40 = llvm.insertvalue %0, %39[38] : !llvm.array<50 x i32> 
    %41 = llvm.insertvalue %0, %40[39] : !llvm.array<50 x i32> 
    %42 = llvm.insertvalue %0, %41[40] : !llvm.array<50 x i32> 
    %43 = llvm.insertvalue %0, %42[41] : !llvm.array<50 x i32> 
    %44 = llvm.insertvalue %0, %43[42] : !llvm.array<50 x i32> 
    %45 = llvm.insertvalue %0, %44[43] : !llvm.array<50 x i32> 
    %46 = llvm.insertvalue %0, %45[44] : !llvm.array<50 x i32> 
    %47 = llvm.insertvalue %0, %46[45] : !llvm.array<50 x i32> 
    %48 = llvm.insertvalue %0, %47[46] : !llvm.array<50 x i32> 
    %49 = llvm.insertvalue %0, %48[47] : !llvm.array<50 x i32> 
    %50 = llvm.insertvalue %0, %49[48] : !llvm.array<50 x i32> 
    %51 = llvm.insertvalue %0, %50[49] : !llvm.array<50 x i32> 
    llvm.return %51 : !llvm.array<50 x i32>
  }
  llvm.mlir.global external local_unnamed_addr @Ch_2_Glob(0 : i8) {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
  llvm.mlir.global private unnamed_addr constant @".str.13"("DHRYSTONE PROGRAM, 3'RD STRING\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @End_Time(0 : i64) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : i64
  llvm.mlir.global private unnamed_addr constant @".str.16"("Int_Glob:            %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.17"("        should be:   %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.18"("Bool_Glob:           %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.19"("Ch_1_Glob:           %c\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.20"("        should be:   %c\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.21"("Ch_2_Glob:           %c\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.22"("Arr_1_Glob[8]:       %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.23"("Arr_2_Glob[8][7]:    %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.26"("  Discr:             %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.27"("  Enum_Comp:         %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.28"("  Int_Comp:          %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.29"("  Str_Comp:          %s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.32"("Int_1_Loc:           %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.33"("Int_2_Loc:           %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.34"("Int_3_Loc:           %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.35"("Enum_Loc:            %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.36"("Str_1_Loc:           %s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.38"("Str_2_Loc:           %s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @User_Time(0 : i64) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : i64
  llvm.mlir.global private unnamed_addr constant @".str.42"("Microseconds for one run through Dhrystone: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str("Dhrystone Benchmark, Version 2.1 (Language: C)\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.44("Program compiled without 'register' attribute\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.45("Execution ends\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.46("Final values of the variables used in the benchmark:\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.47("        should be:   Number_Of_Runs + 10\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.48("Ptr_Glob->\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.50("Next_Ptr_Glob->\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.51("        should be:   DHRYSTONE PROGRAM, SOME STRING\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.52("        should be:   DHRYSTONE PROGRAM, 1'ST STRING\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.53("        should be:   DHRYSTONE PROGRAM, 2'ND STRING\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.54("Removed\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.55("Measured time too small to obtain meaningful results\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.56("Please increase number of runs\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.57("Program compiled with 'register' attribute\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @Int_Glob(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global external local_unnamed_addr @Ch_1_Glob(0 : i8) {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
  llvm.metadata @__llvm_global_metadata {
    llvm.tbaa_root @tbaa_root_3 {id = "Simple C/C++ TBAA"}
    llvm.tbaa_tag @tbaa_tag_0 {access_type = @tbaa_type_desc_1, base_type = @tbaa_type_desc_1, offset = 0 : i64}
    llvm.tbaa_type_desc @tbaa_type_desc_1 {id = "any pointer", members = {<@tbaa_type_desc_2, 0>}}
    llvm.tbaa_type_desc @tbaa_type_desc_2 {id = "omnipotent char", members = {<@tbaa_root_3, 0>}}
    llvm.tbaa_tag @tbaa_tag_4 {access_type = @tbaa_type_desc_1, base_type = @tbaa_type_desc_5, offset = 0 : i64}
    llvm.tbaa_type_desc @tbaa_type_desc_5 {id = "record", members = {<@tbaa_type_desc_1, 0>, <@tbaa_type_desc_2, 8>, <@tbaa_type_desc_2, 12>}}
    llvm.tbaa_tag @tbaa_tag_6 {access_type = @tbaa_type_desc_2, base_type = @tbaa_type_desc_5, offset = 8 : i64}
    llvm.tbaa_tag @tbaa_tag_7 {access_type = @tbaa_type_desc_2, base_type = @tbaa_type_desc_2, offset = 0 : i64}
    llvm.tbaa_tag @tbaa_tag_8 {access_type = @tbaa_type_desc_9, base_type = @tbaa_type_desc_9, offset = 0 : i64}
    llvm.tbaa_type_desc @tbaa_type_desc_9 {id = "int", members = {<@tbaa_type_desc_2, 0>}}
    llvm.tbaa_tag @tbaa_tag_10 {access_type = @tbaa_type_desc_11, base_type = @tbaa_type_desc_11, offset = 0 : i64}
    llvm.tbaa_type_desc @tbaa_type_desc_11 {id = "long", members = {<@tbaa_type_desc_2, 0>}}
  }
  llvm.func @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> i32 attributes {passthrough = ["nounwind", "sspstrong", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.mlir.constant(31 : i64) : i64
    %3 = llvm.mlir.constant(56 : i64) : i64
    %4 = llvm.mlir.null : !llvm.ptr
    %5 = llvm.mlir.addressof @Next_Ptr_Glob : !llvm.ptr
    %6 = llvm.mlir.addressof @Ptr_Glob : !llvm.ptr
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(2 : i32) : i32
    %10 = llvm.mlir.constant(40 : i32) : i32
    %11 = llvm.mlir.constant("DHRYSTONE PROGRAM, SOME STRING\00") : !llvm.array<31 x i8>
    %12 = llvm.mlir.addressof @".str" : !llvm.ptr
    %13 = llvm.mlir.constant(false) : i1
    %14 = llvm.mlir.constant("DHRYSTONE PROGRAM, 1'ST STRING\00") : !llvm.array<31 x i8>
    %15 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %16 = llvm.mlir.constant(10 : i32) : i32
    %17 = llvm.mlir.constant(7 : i64) : i64
    %18 = llvm.mlir.constant(8 : i64) : i64
    %121 = llvm.mlir.addressof @Arr_2_Glob : !llvm.ptr
    %122 = llvm.getelementptr inbounds %121[%7, %18, %17] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<50 x array<50 x i32>>
    %123 = llvm.mlir.constant("Dhrystone Benchmark, Version 2.1 (Language: C)\00") : !llvm.array<47 x i8>
    %124 = llvm.mlir.addressof @str : !llvm.ptr
    %125 = llvm.mlir.addressof @Reg : !llvm.ptr
    %126 = llvm.mlir.constant("Program compiled without 'register' attribute\00") : !llvm.array<46 x i8>
    %127 = llvm.mlir.addressof @str.44 : !llvm.ptr
    %128 = llvm.mlir.constant("Program compiled with 'register' attribute\00") : !llvm.array<43 x i8>
    %129 = llvm.mlir.addressof @str.57 : !llvm.ptr
    %130 = llvm.mlir.addressof @Float_Rating : !llvm.ptr
    %131 = llvm.mlir.constant("Ratings using 'float' datatype (%d bytes)\0A\00") : !llvm.array<43 x i8>
    %132 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %133 = llvm.mlir.constant("Ratings using 'double' datatype (%d bytes)\0A\00") : !llvm.array<44 x i8>
    %134 = llvm.mlir.addressof @".str.7" : !llvm.ptr
    %135 = llvm.mlir.constant("HZ = %d\0A\00") : !llvm.array<9 x i8>
    %136 = llvm.mlir.addressof @".str.8" : !llvm.ptr
    %137 = llvm.mlir.constant(100 : i32) : i32
    %138 = llvm.mlir.constant(1 : i64) : i64
    %139 = llvm.mlir.constant("Please give the number of runs through the benchmark: \00") : !llvm.array<55 x i8>
    %140 = llvm.mlir.addressof @".str.9" : !llvm.ptr
    %141 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %142 = llvm.mlir.addressof @".str.10" : !llvm.ptr
    %143 = llvm.mlir.constant("Execution starts, %d runs through Dhrystone\0A\00") : !llvm.array<45 x i8>
    %144 = llvm.mlir.addressof @".str.11" : !llvm.ptr
    %145 = llvm.mlir.addressof @time : !llvm.ptr
    %146 = llvm.mlir.addressof @Begin_Time : !llvm.ptr
    %147 = llvm.mlir.undef : i32
    %148 = llvm.mlir.constant(65 : i8) : i8
    %149 = llvm.mlir.constant(0 : i8) : i8
    %150 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %151 = llvm.mlir.addressof @Bool_Glob : !llvm.ptr
    %152 = llvm.mlir.constant(66 : i8) : i8
    %153 = llvm.mlir.addressof @Ch_2_Glob : !llvm.ptr
    %154 = llvm.mlir.constant("DHRYSTONE PROGRAM, 2'ND STRING\00") : !llvm.array<31 x i8>
    %155 = llvm.mlir.addressof @".str.12" : !llvm.ptr
    %156 = llvm.mlir.constant(7 : i32) : i32
    %157 = llvm.mlir.constant(3 : i32) : i32
    %158 = llvm.mlir.addressof @Arr_1_Glob : !llvm.ptr
    %159 = llvm.mlir.constant(5 : i32) : i32
    %160 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %161 = llvm.mlir.constant(6 : i32) : i32
    %162 = llvm.mlir.addressof @Func_1 : !llvm.ptr
    %163 = llvm.mlir.constant(67 : i8) : i8
    %164 = llvm.mlir.constant("DHRYSTONE PROGRAM, 3'RD STRING\00") : !llvm.array<31 x i8>
    %165 = llvm.mlir.addressof @".str.13" : !llvm.ptr
    %166 = llvm.mlir.constant(1 : i8) : i8
    %167 = llvm.mlir.constant(9 : i32) : i32
    %168 = llvm.mlir.addressof @End_Time : !llvm.ptr
    %169 = llvm.mlir.constant("Execution ends\00") : !llvm.array<15 x i8>
    %170 = llvm.mlir.addressof @str.45 : !llvm.ptr
    %171 = llvm.mlir.constant("Final values of the variables used in the benchmark:\00") : !llvm.array<53 x i8>
    %172 = llvm.mlir.addressof @str.46 : !llvm.ptr
    %173 = llvm.mlir.constant("Int_Glob:            %d\0A\00") : !llvm.array<25 x i8>
    %174 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %175 = llvm.mlir.constant("        should be:   %d\0A\00") : !llvm.array<25 x i8>
    %176 = llvm.mlir.addressof @".str.17" : !llvm.ptr
    %177 = llvm.mlir.constant("Bool_Glob:           %d\0A\00") : !llvm.array<25 x i8>
    %178 = llvm.mlir.addressof @".str.18" : !llvm.ptr
    %179 = llvm.mlir.constant("Ch_1_Glob:           %c\0A\00") : !llvm.array<25 x i8>
    %180 = llvm.mlir.addressof @".str.19" : !llvm.ptr
    %181 = llvm.mlir.constant("        should be:   %c\0A\00") : !llvm.array<25 x i8>
    %182 = llvm.mlir.addressof @".str.20" : !llvm.ptr
    %183 = llvm.mlir.constant(65 : i32) : i32
    %184 = llvm.mlir.constant("Ch_2_Glob:           %c\0A\00") : !llvm.array<25 x i8>
    %185 = llvm.mlir.addressof @".str.21" : !llvm.ptr
    %186 = llvm.mlir.constant(66 : i32) : i32
    %187 = llvm.getelementptr inbounds %158[%7, %18] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<50 x i32>
    %188 = llvm.mlir.constant("Arr_1_Glob[8]:       %d\0A\00") : !llvm.array<25 x i8>
    %189 = llvm.mlir.addressof @".str.22" : !llvm.ptr
    %190 = llvm.mlir.constant("Arr_2_Glob[8][7]:    %d\0A\00") : !llvm.array<25 x i8>
    %191 = llvm.mlir.addressof @".str.23" : !llvm.ptr
    %192 = llvm.mlir.constant("        should be:   Number_Of_Runs + 10\00") : !llvm.array<41 x i8>
    %193 = llvm.mlir.addressof @str.47 : !llvm.ptr
    %194 = llvm.mlir.constant("Ptr_Glob->\00") : !llvm.array<11 x i8>
    %195 = llvm.mlir.addressof @str.48 : !llvm.ptr
    %196 = llvm.mlir.constant("  Discr:             %d\0A\00") : !llvm.array<25 x i8>
    %197 = llvm.mlir.addressof @".str.26" : !llvm.ptr
    %198 = llvm.mlir.constant("  Enum_Comp:         %d\0A\00") : !llvm.array<25 x i8>
    %199 = llvm.mlir.addressof @".str.27" : !llvm.ptr
    %200 = llvm.mlir.constant("  Int_Comp:          %d\0A\00") : !llvm.array<25 x i8>
    %201 = llvm.mlir.addressof @".str.28" : !llvm.ptr
    %202 = llvm.mlir.constant(17 : i32) : i32
    %203 = llvm.mlir.constant("  Str_Comp:          %s\0A\00") : !llvm.array<25 x i8>
    %204 = llvm.mlir.addressof @".str.29" : !llvm.ptr
    %205 = llvm.mlir.constant("        should be:   DHRYSTONE PROGRAM, SOME STRING\00") : !llvm.array<52 x i8>
    %206 = llvm.mlir.addressof @str.51 : !llvm.ptr
    %207 = llvm.mlir.constant("Next_Ptr_Glob->\00") : !llvm.array<16 x i8>
    %208 = llvm.mlir.addressof @str.50 : !llvm.ptr
    %209 = llvm.mlir.constant(18 : i32) : i32
    %210 = llvm.mlir.constant("Int_1_Loc:           %d\0A\00") : !llvm.array<25 x i8>
    %211 = llvm.mlir.addressof @".str.32" : !llvm.ptr
    %212 = llvm.mlir.constant("Int_2_Loc:           %d\0A\00") : !llvm.array<25 x i8>
    %213 = llvm.mlir.addressof @".str.33" : !llvm.ptr
    %214 = llvm.mlir.constant(13 : i32) : i32
    %215 = llvm.mlir.constant("Int_3_Loc:           %d\0A\00") : !llvm.array<25 x i8>
    %216 = llvm.mlir.addressof @".str.34" : !llvm.ptr
    %217 = llvm.mlir.constant("Enum_Loc:            %d\0A\00") : !llvm.array<25 x i8>
    %218 = llvm.mlir.addressof @".str.35" : !llvm.ptr
    %219 = llvm.mlir.constant("Str_1_Loc:           %s\0A\00") : !llvm.array<25 x i8>
    %220 = llvm.mlir.addressof @".str.36" : !llvm.ptr
    %221 = llvm.mlir.constant("        should be:   DHRYSTONE PROGRAM, 1'ST STRING\00") : !llvm.array<52 x i8>
    %222 = llvm.mlir.addressof @str.52 : !llvm.ptr
    %223 = llvm.mlir.constant("Str_2_Loc:           %s\0A\00") : !llvm.array<25 x i8>
    %224 = llvm.mlir.addressof @".str.38" : !llvm.ptr
    %225 = llvm.mlir.constant("        should be:   DHRYSTONE PROGRAM, 2'ND STRING\00") : !llvm.array<52 x i8>
    %226 = llvm.mlir.addressof @str.53 : !llvm.ptr
    %227 = llvm.mlir.addressof @User_Time : !llvm.ptr
    %228 = llvm.mlir.constant(2 : i64) : i64
    %229 = llvm.mlir.constant("Microseconds for one run through Dhrystone: \00") : !llvm.array<45 x i8>
    %230 = llvm.mlir.addressof @".str.42" : !llvm.ptr
    %231 = llvm.mlir.constant("Removed\00") : !llvm.array<8 x i8>
    %232 = llvm.mlir.addressof @str.54 : !llvm.ptr
    %233 = llvm.mlir.constant("Measured time too small to obtain meaningful results\00") : !llvm.array<53 x i8>
    %234 = llvm.mlir.addressof @str.55 : !llvm.ptr
    %235 = llvm.mlir.constant("Please increase number of runs\00") : !llvm.array<31 x i8>
    %236 = llvm.mlir.addressof @str.56 : !llvm.ptr
    %237 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %238 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %239 = llvm.alloca %0 x !llvm.array<31 x i8> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %240 = llvm.alloca %0 x !llvm.array<31 x i8> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %241 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %242 = llvm.call @malloc(%3) : (i64) -> !llvm.ptr
    llvm.store %242, %5 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr, !llvm.ptr
    %243 = llvm.call @malloc(%3) : (i64) -> !llvm.ptr
    llvm.store %243, %6 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr, !llvm.ptr
    llvm.store %242, %243 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr, !llvm.ptr
    %244 = llvm.getelementptr inbounds %243[%7, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %8, %244 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_6]} : i32, !llvm.ptr
    %245 = llvm.getelementptr inbounds %243[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %9, %245 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i32, !llvm.ptr
    %246 = llvm.getelementptr inbounds %243[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %10, %246 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i32, !llvm.ptr
    %247 = llvm.getelementptr inbounds %243[%7, 2, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    "llvm.intr.memcpy"(%247, %12, %2, %13) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
    "llvm.intr.memcpy"(%239, %15, %2, %13) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
    llvm.store %16, %122 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    %248 = llvm.call @putchar(%16) : (i32) -> i32
    %249 = llvm.call @puts(%124) : (!llvm.ptr) -> i32
    %250 = llvm.load %125 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %251 = llvm.icmp "eq" %250, %8 : i32
    %252 = llvm.select %251, %127, %129 : i1, !llvm.ptr
    %253 = llvm.call @puts(%252) : (!llvm.ptr) -> i32
    %254 = llvm.load %130 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %255 = llvm.icmp "eq" %254, %8 : i32
    llvm.cond_br %255, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %256 = llvm.call @printf(%132, %1) : (!llvm.ptr, i64) -> i32
    llvm.br ^bb3
  ^bb2:  // pred: ^bb0
    %257 = llvm.call @printf(%134, %18) : (!llvm.ptr, i64) -> i32
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %258 = llvm.call @printf(%136, %137) : (!llvm.ptr, i32) -> i32
    %259 = llvm.call @putchar(%16) : (i32) -> i32
    %260 = llvm.icmp "slt" %arg0, %9 : i32
    llvm.cond_br %260, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %261 = llvm.call @printf(%140) : (!llvm.ptr) -> i32
    %262 = llvm.call @__isoc99_scanf(%142, %241) : (!llvm.ptr, !llvm.ptr) -> i32
    %263 = llvm.load %241 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %264 = llvm.call @putchar(%16) : (i32) -> i32
    llvm.br ^bb6(%263 : i32)
  ^bb5:  // pred: ^bb3
    %265 = llvm.getelementptr inbounds %arg1[%138] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %266 = llvm.load %265 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %267 = llvm.call @strtol(%266, %4, %16) : (!llvm.ptr, !llvm.ptr, i32) -> i64
    %268 = llvm.trunc %267 : i64 to i32
    llvm.br ^bb6(%268 : i32)
  ^bb6(%269: i32):  // 2 preds: ^bb4, ^bb5
    %270 = llvm.call @printf(%144, %269) : (!llvm.ptr, i32) -> i32
    %271 = llvm.call %145(%4) : !llvm.ptr, (!llvm.ptr) -> i64
    llvm.store %271, %146 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_10]} : i64, !llvm.ptr
    %272 = llvm.icmp "slt" %269, %0 : i32
    llvm.cond_br %272, ^bb18(%147, %147 : i32, i32), ^bb7(%0 : i32)
  ^bb7(%273: i32):  // 2 preds: ^bb6, ^bb16
    llvm.store %148, %150 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i8, !llvm.ptr
    llvm.store %0, %151 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.store %152, %153 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i8, !llvm.ptr
    "llvm.intr.memcpy"(%240, %155, %2, %13) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
    llvm.store %0, %238 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i32, !llvm.ptr
    %274 = llvm.call @Func_2(%239, %240) : (!llvm.ptr, !llvm.ptr) -> i32
    %275 = llvm.icmp "eq" %274, %8 : i32
    %276 = llvm.zext %275 : i1 to i32
    llvm.store %276, %151 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.store %156, %237 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.call @Proc_7(%9, %157, %237) : (i32, i32, !llvm.ptr) -> ()
    %277 = llvm.load %237 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    llvm.call @Proc_8(%158, %121, %157, %277) : (!llvm.ptr, !llvm.ptr, i32, i32) -> ()
    %278 = llvm.load %6 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %279 = llvm.load %278 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%279, %278, %3, %13) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %280 = llvm.getelementptr inbounds %278[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %159, %280 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i32, !llvm.ptr
    %281 = llvm.getelementptr inbounds %279[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %159, %281 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i32, !llvm.ptr
    %282 = llvm.load %278 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %282, %279 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr, !llvm.ptr
    %283 = llvm.load %6 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %284 = llvm.icmp "eq" %283, %4 : !llvm.ptr
    llvm.cond_br %284, ^bb9(%4 : !llvm.ptr), ^bb8
  ^bb8:  // pred: ^bb7
    %285 = llvm.load %283 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %285, %279 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr, !llvm.ptr
    %286 = llvm.load %6 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb9(%286 : !llvm.ptr)
  ^bb9(%287: !llvm.ptr):  // 2 preds: ^bb7, ^bb8
    %288 = llvm.load %160 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %289 = llvm.getelementptr inbounds %287[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.call @Proc_7(%16, %288, %289) : (i32, i32, !llvm.ptr) -> ()
    %290 = llvm.getelementptr inbounds %279[%7, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %291 = llvm.load %290 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_6]} : !llvm.ptr -> i32
    %292 = llvm.icmp "eq" %291, %8 : i32
    llvm.cond_br %292, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %293 = llvm.getelementptr inbounds %279[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %294 = llvm.getelementptr inbounds %278[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %161, %281 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i32, !llvm.ptr
    %295 = llvm.load %294 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i32
    llvm.call @Proc_6(%295, %293) : (i32, !llvm.ptr) -> ()
    %296 = llvm.load %6 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %297 = llvm.load %296 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %297, %279 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr, !llvm.ptr
    %298 = llvm.load %281 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i32
    llvm.call @Proc_7(%298, %16, %281) : (i32, i32, !llvm.ptr) -> ()
    llvm.br ^bb12
  ^bb11:  // pred: ^bb9
    %299 = llvm.load %278 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%278, %299, %3, %13) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
    llvm.br ^bb12
  ^bb12:  // 2 preds: ^bb10, ^bb11
    %300 = llvm.load %153 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i8
    %301 = llvm.icmp "slt" %300, %148 : i8
    llvm.cond_br %301, ^bb16(%157 : i32), ^bb13(%148, %157 : i8, i32)
  ^bb13(%302: i8, %303: i32):  // 2 preds: ^bb12, ^bb15
    %304 = llvm.load %238 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i32
    %305 = llvm.call %162(%302, %163) : !llvm.ptr, (i8, i8) -> i32
    %306 = llvm.icmp "eq" %304, %305 : i32
    llvm.cond_br %306, ^bb14, ^bb15(%303 : i32)
  ^bb14:  // pred: ^bb13
    llvm.call @Proc_6(%8, %238) : (i32, !llvm.ptr) -> ()
    "llvm.intr.memcpy"(%240, %165, %2, %13) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
    llvm.store %273, %160 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.br ^bb15(%273 : i32)
  ^bb15(%307: i32):  // 2 preds: ^bb13, ^bb14
    %308 = llvm.add %302, %166  : i8
    %309 = llvm.load %153 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i8
    %310 = llvm.icmp "sgt" %308, %309 : i8
    llvm.cond_br %310, ^bb16(%307 : i32), ^bb13(%308, %307 : i8, i32)
  ^bb16(%311: i32):  // 2 preds: ^bb12, ^bb15
    %312 = llvm.add %273, %0  : i32
    %313 = llvm.icmp "eq" %273, %269 : i32
    llvm.cond_br %313, ^bb17, ^bb7(%312 : i32)
  ^bb17:  // pred: ^bb16
    %314 = llvm.load %160 {alignment = 4 : i64} : !llvm.ptr -> i32
    %315 = llvm.load %150 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i8
    %316 = llvm.load %237 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %317 = llvm.mul %311, %157  : i32
    %318 = llvm.sdiv %317, %316  : i32
    %319 = llvm.sub %317, %316  : i32
    %320 = llvm.mul %319, %156  : i32
    %321 = llvm.sub %320, %318  : i32
    %322 = llvm.icmp "eq" %315, %148 : i8
    %323 = llvm.add %318, %167  : i32
    %324 = llvm.sub %323, %314  : i32
    %325 = llvm.select %322, %324, %318 : i1, i32
    llvm.br ^bb18(%325, %321 : i32, i32)
  ^bb18(%326: i32, %327: i32):  // 2 preds: ^bb6, ^bb17
    %328 = llvm.call %145(%4) : !llvm.ptr, (!llvm.ptr) -> i64
    llvm.store %328, %168 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_10]} : i64, !llvm.ptr
    %329 = llvm.call @puts(%170) : (!llvm.ptr) -> i32
    %330 = llvm.call @putchar(%16) : (i32) -> i32
    %331 = llvm.call @puts(%172) : (!llvm.ptr) -> i32
    %332 = llvm.call @putchar(%16) : (i32) -> i32
    %333 = llvm.load %160 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %334 = llvm.call @printf(%174, %333) : (!llvm.ptr, i32) -> i32
    %335 = llvm.call @printf(%176, %159) : (!llvm.ptr, i32) -> i32
    %336 = llvm.load %151 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %337 = llvm.call @printf(%178, %336) : (!llvm.ptr, i32) -> i32
    %338 = llvm.call @printf(%176, %0) : (!llvm.ptr, i32) -> i32
    %339 = llvm.load %150 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i8
    %340 = llvm.sext %339 : i8 to i32
    %341 = llvm.call @printf(%180, %340) : (!llvm.ptr, i32) -> i32
    %342 = llvm.call @printf(%182, %183) : (!llvm.ptr, i32) -> i32
    %343 = llvm.load %153 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i8
    %344 = llvm.sext %343 : i8 to i32
    %345 = llvm.call @printf(%185, %344) : (!llvm.ptr, i32) -> i32
    %346 = llvm.call @printf(%182, %186) : (!llvm.ptr, i32) -> i32
    %347 = llvm.load %187 {alignment = 16 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %348 = llvm.call @printf(%189, %347) : (!llvm.ptr, i32) -> i32
    %349 = llvm.call @printf(%176, %156) : (!llvm.ptr, i32) -> i32
    %350 = llvm.load %122 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %351 = llvm.call @printf(%191, %350) : (!llvm.ptr, i32) -> i32
    %352 = llvm.call @puts(%193) : (!llvm.ptr) -> i32
    %353 = llvm.call @puts(%195) : (!llvm.ptr) -> i32
    %354 = llvm.load %6 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %355 = llvm.getelementptr inbounds %354[%7, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %356 = llvm.load %355 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_6]} : !llvm.ptr -> i32
    %357 = llvm.call @printf(%197, %356) : (!llvm.ptr, i32) -> i32
    %358 = llvm.call @printf(%176, %8) : (!llvm.ptr, i32) -> i32
    %359 = llvm.load %6 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %360 = llvm.getelementptr inbounds %359[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %361 = llvm.load %360 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i32
    %362 = llvm.call @printf(%199, %361) : (!llvm.ptr, i32) -> i32
    %363 = llvm.call @printf(%176, %9) : (!llvm.ptr, i32) -> i32
    %364 = llvm.load %6 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %365 = llvm.getelementptr inbounds %364[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %366 = llvm.load %365 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i32
    %367 = llvm.call @printf(%201, %366) : (!llvm.ptr, i32) -> i32
    %368 = llvm.call @printf(%176, %202) : (!llvm.ptr, i32) -> i32
    %369 = llvm.load %6 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %370 = llvm.getelementptr inbounds %369[%7, 2, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %371 = llvm.call @printf(%204, %370) : (!llvm.ptr, !llvm.ptr) -> i32
    %372 = llvm.call @puts(%206) : (!llvm.ptr) -> i32
    %373 = llvm.call @puts(%208) : (!llvm.ptr) -> i32
    %374 = llvm.load %5 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %375 = llvm.getelementptr inbounds %374[%7, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %376 = llvm.load %375 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_6]} : !llvm.ptr -> i32
    %377 = llvm.call @printf(%197, %376) : (!llvm.ptr, i32) -> i32
    %378 = llvm.call @printf(%176, %8) : (!llvm.ptr, i32) -> i32
    %379 = llvm.load %5 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %380 = llvm.getelementptr inbounds %379[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %381 = llvm.load %380 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i32
    %382 = llvm.call @printf(%199, %381) : (!llvm.ptr, i32) -> i32
    %383 = llvm.call @printf(%176, %0) : (!llvm.ptr, i32) -> i32
    %384 = llvm.load %5 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %385 = llvm.getelementptr inbounds %384[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %386 = llvm.load %385 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i32
    %387 = llvm.call @printf(%201, %386) : (!llvm.ptr, i32) -> i32
    %388 = llvm.call @printf(%176, %209) : (!llvm.ptr, i32) -> i32
    %389 = llvm.load %5 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %390 = llvm.getelementptr inbounds %389[%7, 2, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %391 = llvm.call @printf(%204, %390) : (!llvm.ptr, !llvm.ptr) -> i32
    %392 = llvm.call @puts(%206) : (!llvm.ptr) -> i32
    %393 = llvm.call @printf(%211, %326) : (!llvm.ptr, i32) -> i32
    %394 = llvm.call @printf(%176, %159) : (!llvm.ptr, i32) -> i32
    %395 = llvm.call @printf(%213, %327) : (!llvm.ptr, i32) -> i32
    %396 = llvm.call @printf(%176, %214) : (!llvm.ptr, i32) -> i32
    %397 = llvm.load %237 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %398 = llvm.call @printf(%216, %397) : (!llvm.ptr, i32) -> i32
    %399 = llvm.call @printf(%176, %156) : (!llvm.ptr, i32) -> i32
    %400 = llvm.load %238 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i32
    %401 = llvm.call @printf(%218, %400) : (!llvm.ptr, i32) -> i32
    %402 = llvm.call @printf(%176, %0) : (!llvm.ptr, i32) -> i32
    %403 = llvm.call @printf(%220, %239) : (!llvm.ptr, !llvm.ptr) -> i32
    %404 = llvm.call @puts(%222) : (!llvm.ptr) -> i32
    %405 = llvm.call @printf(%224, %240) : (!llvm.ptr, !llvm.ptr) -> i32
    %406 = llvm.call @puts(%226) : (!llvm.ptr) -> i32
    %407 = llvm.call @putchar(%16) : (i32) -> i32
    %408 = llvm.load %168 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_10]} : !llvm.ptr -> i64
    %409 = llvm.load %146 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_10]} : !llvm.ptr -> i64
    %410 = llvm.sub %408, %409  : i64
    llvm.store %410, %227 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_10]} : i64, !llvm.ptr
    %411 = llvm.icmp "slt" %410, %228 : i64
    llvm.cond_br %411, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %412 = llvm.call @puts(%234) : (!llvm.ptr) -> i32
    %413 = llvm.call @puts(%236) : (!llvm.ptr) -> i32
    %414 = llvm.call @putchar(%16) : (i32) -> i32
    llvm.br ^bb21
  ^bb20:  // pred: ^bb18
    %415 = llvm.call @printf(%230) : (!llvm.ptr) -> i32
    %416 = llvm.call @puts(%232) : (!llvm.ptr) -> i32
    llvm.br ^bb21
  ^bb21:  // 2 preds: ^bb19, ^bb20
    llvm.return %8 : i32
  }
  llvm.func @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["allockind", "9"], ["allocsize", "4294967295"], ["alloc-family", "malloc"], ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func @putchar(i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind"]}
  llvm.func @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind"]}
  llvm.func @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func @__isoc99_scanf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func @strtol(!llvm.ptr {llvm.noundef, llvm.readonly}, !llvm.ptr {llvm.nocapture, llvm.noundef}, i32 {llvm.noundef}) -> i64 attributes {passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func @time(...) -> i64 attributes {passthrough = [["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func @Proc_5() attributes {memory = #llvm.memory_effects<other = write, argMem = write, inaccessibleMem = write>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(65 : i8) : i8
    %1 = llvm.mlir.constant(0 : i8) : i8
    %2 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.addressof @Bool_Glob : !llvm.ptr
    llvm.store %0, %2 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i8, !llvm.ptr
    llvm.store %3, %4 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func @Proc_4() attributes {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %2 = llvm.mlir.constant(65 : i8) : i8
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.addressof @Bool_Glob : !llvm.ptr
    %5 = llvm.mlir.constant(66 : i8) : i8
    %6 = llvm.mlir.addressof @Ch_2_Glob : !llvm.ptr
    %7 = llvm.load %1 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i8
    %8 = llvm.icmp "eq" %7, %2 : i8
    %9 = llvm.zext %8 : i1 to i32
    %10 = llvm.load %4 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %11 = llvm.or %10, %9  : i32
    llvm.store %11, %4 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.store %5, %6 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i8, !llvm.ptr
    llvm.return
  }
  llvm.func @Proc_1(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {passthrough = ["nounwind", "sspstrong", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.null : !llvm.ptr
    %1 = llvm.mlir.addressof @Ptr_Glob : !llvm.ptr
    %2 = llvm.mlir.constant(56 : i64) : i64
    %3 = llvm.mlir.constant(false) : i1
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.mlir.constant(2 : i32) : i32
    %6 = llvm.mlir.constant(0 : i32) : i32
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(5 : i32) : i32
    %9 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %10 = llvm.mlir.constant(10 : i32) : i32
    %11 = llvm.mlir.constant(6 : i32) : i32
    %12 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    %13 = llvm.load %1 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%12, %13, %2, %3) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
    %14 = llvm.getelementptr inbounds %arg0[%4, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %8, %14 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i32, !llvm.ptr
    %15 = llvm.getelementptr inbounds %12[%4, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %8, %15 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i32, !llvm.ptr
    %16 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %16, %12 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr, !llvm.ptr
    %17 = llvm.load %1 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.icmp "eq" %17, %0 : !llvm.ptr
    llvm.cond_br %18, ^bb2(%0 : !llvm.ptr), ^bb1
  ^bb1:  // pred: ^bb0
    %19 = llvm.load %17 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %19, %12 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr, !llvm.ptr
    %20 = llvm.load %1 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb2(%20 : !llvm.ptr)
  ^bb2(%21: !llvm.ptr):  // 2 preds: ^bb0, ^bb1
    %22 = llvm.load %9 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %23 = llvm.getelementptr inbounds %21[%4, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.call @Proc_7(%10, %22, %23) : (i32, i32, !llvm.ptr) -> ()
    %24 = llvm.getelementptr inbounds %12[%4, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_6]} : !llvm.ptr -> i32
    %26 = llvm.icmp "eq" %25, %6 : i32
    llvm.cond_br %26, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %27 = llvm.getelementptr inbounds %12[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %28 = llvm.getelementptr inbounds %arg0[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %11, %15 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i32, !llvm.ptr
    %29 = llvm.load %28 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i32
    llvm.call @Proc_6(%29, %27) : (i32, !llvm.ptr) -> ()
    %30 = llvm.load %1 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %31 = llvm.load %30 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %31, %12 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr, !llvm.ptr
    %32 = llvm.load %15 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i32
    llvm.call @Proc_7(%32, %10, %15) : (i32, i32, !llvm.ptr) -> ()
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    %33 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%arg0, %33, %2, %3) : (!llvm.ptr, !llvm.ptr, i64, i1) -> ()
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    llvm.return
  }
  llvm.func @Proc_2(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %2 = llvm.mlir.constant(65 : i8) : i8
    %3 = llvm.mlir.constant(9 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %6 = llvm.load %1 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i8
    %7 = llvm.icmp "eq" %6, %2 : i8
    llvm.cond_br %7, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %8 = llvm.load %arg0 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %9 = llvm.add %8, %3  : i32
    %10 = llvm.load %5 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %11 = llvm.sub %9, %10  : i32
    llvm.store %11, %arg0 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return
  }
  llvm.func @Proc_3(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {passthrough = ["nounwind", "sspstrong", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.null : !llvm.ptr
    %1 = llvm.mlir.addressof @Ptr_Glob : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.mlir.constant(2 : i32) : i32
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(10 : i32) : i32
    %8 = llvm.load %1 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    %9 = llvm.icmp "eq" %8, %0 : !llvm.ptr
    llvm.cond_br %9, ^bb2(%0 : !llvm.ptr), ^bb1
  ^bb1:  // pred: ^bb0
    %10 = llvm.load %8 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %10, %arg0 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr, !llvm.ptr
    %11 = llvm.load %1 {alignment = 8 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_0]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb2(%11 : !llvm.ptr)
  ^bb2(%12: !llvm.ptr):  // 2 preds: ^bb0, ^bb1
    %13 = llvm.load %3 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %14 = llvm.getelementptr inbounds %12[%4, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.call @Proc_7(%7, %13, %14) : (i32, i32, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @Proc_6(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.mlir.constant(3 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %5 = llvm.mlir.constant(100 : i32) : i32
    %6 = llvm.icmp "eq" %arg0, %0 : i32
    %7 = llvm.select %6, %0, %1 : i1, i32
    llvm.store %7, %arg1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.switch %arg0 : i32, ^bb5 [
      0: ^bb4(%arg0 : i32),
      1: ^bb1,
      2: ^bb2,
      4: ^bb3
    ]
  ^bb1:  // pred: ^bb0
    %8 = llvm.load %4 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %9 = llvm.icmp "sgt" %8, %5 : i32
    %10 = llvm.select %9, %3, %1 : i1, i32
    llvm.br ^bb4(%10 : i32)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb4(%2 : i32)
  ^bb3:  // pred: ^bb0
    llvm.br ^bb4(%0 : i32)
  ^bb4(%11: i32):  // 4 preds: ^bb0, ^bb1, ^bb2, ^bb3
    llvm.store %11, %arg1 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return
  }
  llvm.func @Func_3(%arg0: i32 {llvm.noundef}) -> i32 attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.icmp "eq" %arg0, %0 : i32
    %2 = llvm.zext %1 : i1 to i32
    llvm.return %2 : i32
  }
  llvm.func @Proc_7(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.add %arg0, %0  : i32
    %2 = llvm.add %1, %arg1  : i32
    llvm.store %2, %arg2 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func @Proc_8(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: i32 {llvm.noundef}) attributes {passthrough = ["nofree", "norecurse", "nosync", "nounwind", "sspstrong", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(5 : i32) : i32
    %1 = llvm.mlir.constant(6 : i32) : i32
    %2 = llvm.mlir.constant(35 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.mlir.constant(4 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(25 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %9 = llvm.add %arg2, %0  : i32
    %10 = llvm.sext %9 : i32 to i64
    %11 = llvm.getelementptr inbounds %arg0[%10] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %arg3, %11 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    %12 = llvm.add %arg2, %1  : i32
    %13 = llvm.sext %12 : i32 to i64
    %14 = llvm.getelementptr inbounds %arg0[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %arg3, %14 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    %15 = llvm.add %arg2, %2  : i32
    %16 = llvm.sext %15 : i32 to i64
    %17 = llvm.getelementptr inbounds %arg0[%16] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %9, %17 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    %18 = llvm.getelementptr inbounds %arg1[%10, %10] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<50 x i32>
    llvm.store %9, %18 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    %19 = llvm.add %10, %3  : i64
    %20 = llvm.getelementptr inbounds %arg1[%10, %19] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<50 x i32>
    llvm.store %9, %20 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    %21 = llvm.add %arg2, %4  : i32
    %22 = llvm.sext %21 : i32 to i64
    %23 = llvm.getelementptr inbounds %arg1[%10, %22] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<50 x i32>
    %24 = llvm.load %23 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %25 = llvm.add %24, %5  : i32
    llvm.store %25, %23 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    %26 = llvm.load %11 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : !llvm.ptr -> i32
    %27 = llvm.add %arg2, %6  : i32
    %28 = llvm.sext %27 : i32 to i64
    %29 = llvm.getelementptr inbounds %arg1[%28, %10] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<50 x i32>
    llvm.store %26, %29 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.store %0, %8 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func @Func_1(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> i32 attributes {memory = #llvm.memory_effects<other = write, argMem = write, inaccessibleMem = write>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(255 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i8) : i8
    %3 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.xor %arg1, %arg0  : i32
    %6 = llvm.and %5, %0  : i32
    %7 = llvm.icmp "eq" %6, %1 : i32
    llvm.cond_br %7, ^bb1, ^bb2(%1 : i32)
  ^bb1:  // pred: ^bb0
    %8 = llvm.trunc %arg0 : i32 to i8
    llvm.store %8, %3 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i8, !llvm.ptr
    llvm.br ^bb2(%4 : i32)
  ^bb2(%9: i32):  // 2 preds: ^bb0, ^bb1
    llvm.return %9 : i32
  }
  llvm.func @Func_2(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> i32 attributes {passthrough = ["nofree", "nounwind", "sspstrong", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(2 : i64) : i64
    %1 = llvm.mlir.constant(3 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(10 : i32) : i32
    %4 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(0 : i8) : i8
    %7 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %8 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %9 = llvm.load %8 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i8
    %10 = llvm.getelementptr inbounds %arg1[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %11 = llvm.load %10 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : !llvm.ptr -> i8
    %12 = llvm.icmp "eq" %11, %9 : i8
    llvm.cond_br %12, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    llvm.store %9, %7 {alignment = 1 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_7]} : i8, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb2
  ^bb3:  // pred: ^bb0
    %13 = llvm.call @strcmp(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> i32
    %14 = llvm.icmp "sgt" %13, %2 : i32
    llvm.cond_br %14, ^bb4, ^bb5(%2 : i32)
  ^bb4:  // pred: ^bb3
    llvm.store %3, %4 {alignment = 4 : i64, tbaa = [@__llvm_global_metadata::@tbaa_tag_8]} : i32, !llvm.ptr
    llvm.br ^bb5(%5 : i32)
  ^bb5(%15: i32):  // 2 preds: ^bb3, ^bb4
    llvm.return %15 : i32
  }
  llvm.func @strcmp(!llvm.ptr {llvm.nocapture, llvm.noundef}, !llvm.ptr {llvm.nocapture, llvm.noundef}) -> i32 attributes {memory = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

