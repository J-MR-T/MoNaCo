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

#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "long", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc2, access_type = #tbaa_type_desc2, offset = 0>
#tbaa_tag3 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "record", members = {<#tbaa_type_desc1, 0>, <#tbaa_type_desc, 8>, <#tbaa_type_desc, 12>}>
#tbaa_tag4 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag5 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc, offset = 8>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
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
  llvm.mlir.global external @Arr_2_Glob(dense<0> : tensor<50x50xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<50 x array<50 x i32>>
  llvm.mlir.global private unnamed_addr constant @".str.6"("Ratings using 'float' datatype (%d bytes)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.7"("Ratings using 'double' datatype (%d bytes)\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.8"("HZ = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.9"("Please give the number of runs through the benchmark: \00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.10"("%d\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.11"("Execution starts, %d runs through Dhrystone\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @Begin_Time(0 : i64) {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : i64
  llvm.mlir.global private unnamed_addr constant @".str.12"("DHRYSTONE PROGRAM, 2'ND STRING\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @Bool_Glob(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global external @Arr_1_Glob(dense<0> : tensor<50xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<50 x i32>
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
  llvm.mlir.global private unnamed_addr constant @".str.43"("Removed. Begin_Time: %ld ; End_Time: %ld ; Delta: %ld\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
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
  llvm.mlir.global private unnamed_addr constant @str.54("Measured time too small to obtain meaningful results\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.55("Please increase number of runs\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @str.56("Program compiled with 'register' attribute\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global external local_unnamed_addr @Int_Glob(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global external local_unnamed_addr @Ch_1_Glob(0 : i8) {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
  llvm.func local_unnamed_addr @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> i32 attributes {passthrough = ["nounwind", "sspstrong", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
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
    %19 = llvm.mlir.constant(dense<0> : tensor<50xi32>) : !llvm.array<50 x i32>
    %20 = llvm.mlir.constant(dense<0> : tensor<50x50xi32>) : !llvm.array<50 x array<50 x i32>>
    %21 = llvm.mlir.addressof @Arr_2_Glob : !llvm.ptr
    %22 = llvm.getelementptr inbounds %21[%7, %18, %17] : (!llvm.ptr, i64, i64, i64) -> !llvm.ptr, !llvm.array<50 x array<50 x i32>>
    %23 = llvm.mlir.constant("Dhrystone Benchmark, Version 2.1 (Language: C)\00") : !llvm.array<47 x i8>
    %24 = llvm.mlir.addressof @str : !llvm.ptr
    %25 = llvm.mlir.addressof @Reg : !llvm.ptr
    %26 = llvm.mlir.constant("Program compiled without 'register' attribute\00") : !llvm.array<46 x i8>
    %27 = llvm.mlir.addressof @str.44 : !llvm.ptr
    %28 = llvm.mlir.constant("Program compiled with 'register' attribute\00") : !llvm.array<43 x i8>
    %29 = llvm.mlir.addressof @str.56 : !llvm.ptr
    %30 = llvm.mlir.addressof @Float_Rating : !llvm.ptr
    %31 = llvm.mlir.constant("Ratings using 'float' datatype (%d bytes)\0A\00") : !llvm.array<43 x i8>
    %32 = llvm.mlir.addressof @".str.6" : !llvm.ptr
    %33 = llvm.mlir.constant("Ratings using 'double' datatype (%d bytes)\0A\00") : !llvm.array<44 x i8>
    %34 = llvm.mlir.addressof @".str.7" : !llvm.ptr
    %35 = llvm.mlir.constant("HZ = %d\0A\00") : !llvm.array<9 x i8>
    %36 = llvm.mlir.addressof @".str.8" : !llvm.ptr
    %37 = llvm.mlir.constant(100 : i32) : i32
    %38 = llvm.mlir.constant(1 : i64) : i64
    %39 = llvm.mlir.constant("Please give the number of runs through the benchmark: \00") : !llvm.array<55 x i8>
    %40 = llvm.mlir.addressof @".str.9" : !llvm.ptr
    %41 = llvm.mlir.constant("%d\00") : !llvm.array<3 x i8>
    %42 = llvm.mlir.addressof @".str.10" : !llvm.ptr
    %43 = llvm.mlir.constant("Execution starts, %d runs through Dhrystone\0A\00") : !llvm.array<45 x i8>
    %44 = llvm.mlir.addressof @".str.11" : !llvm.ptr
    %45 = llvm.mlir.addressof @time : !llvm.ptr
    %46 = llvm.mlir.addressof @Begin_Time : !llvm.ptr
    %47 = llvm.mlir.undef : i32
    %48 = llvm.mlir.constant(65 : i8) : i8
    %49 = llvm.mlir.constant(0 : i8) : i8
    %50 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %51 = llvm.mlir.addressof @Bool_Glob : !llvm.ptr
    %52 = llvm.mlir.constant(66 : i8) : i8
    %53 = llvm.mlir.addressof @Ch_2_Glob : !llvm.ptr
    %54 = llvm.mlir.constant("DHRYSTONE PROGRAM, 2'ND STRING\00") : !llvm.array<31 x i8>
    %55 = llvm.mlir.addressof @".str.12" : !llvm.ptr
    %56 = llvm.mlir.constant(7 : i32) : i32
    %57 = llvm.mlir.constant(3 : i32) : i32
    %58 = llvm.mlir.addressof @Arr_1_Glob : !llvm.ptr
    %59 = llvm.mlir.constant(5 : i32) : i32
    %60 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %61 = llvm.mlir.constant(6 : i32) : i32
    %62 = llvm.mlir.addressof @Func_1 : !llvm.ptr
    %63 = llvm.mlir.constant(67 : i8) : i8
    %64 = llvm.mlir.constant("DHRYSTONE PROGRAM, 3'RD STRING\00") : !llvm.array<31 x i8>
    %65 = llvm.mlir.addressof @".str.13" : !llvm.ptr
    %66 = llvm.mlir.constant(1 : i8) : i8
    %67 = llvm.mlir.constant(9 : i32) : i32
    %68 = llvm.mlir.addressof @End_Time : !llvm.ptr
    %69 = llvm.mlir.constant("Execution ends\00") : !llvm.array<15 x i8>
    %70 = llvm.mlir.addressof @str.45 : !llvm.ptr
    %71 = llvm.mlir.constant("Final values of the variables used in the benchmark:\00") : !llvm.array<53 x i8>
    %72 = llvm.mlir.addressof @str.46 : !llvm.ptr
    %73 = llvm.mlir.constant("Int_Glob:            %d\0A\00") : !llvm.array<25 x i8>
    %74 = llvm.mlir.addressof @".str.16" : !llvm.ptr
    %75 = llvm.mlir.constant("        should be:   %d\0A\00") : !llvm.array<25 x i8>
    %76 = llvm.mlir.addressof @".str.17" : !llvm.ptr
    %77 = llvm.mlir.constant("Bool_Glob:           %d\0A\00") : !llvm.array<25 x i8>
    %78 = llvm.mlir.addressof @".str.18" : !llvm.ptr
    %79 = llvm.mlir.constant("Ch_1_Glob:           %c\0A\00") : !llvm.array<25 x i8>
    %80 = llvm.mlir.addressof @".str.19" : !llvm.ptr
    %81 = llvm.mlir.constant("        should be:   %c\0A\00") : !llvm.array<25 x i8>
    %82 = llvm.mlir.addressof @".str.20" : !llvm.ptr
    %83 = llvm.mlir.constant(65 : i32) : i32
    %84 = llvm.mlir.constant("Ch_2_Glob:           %c\0A\00") : !llvm.array<25 x i8>
    %85 = llvm.mlir.addressof @".str.21" : !llvm.ptr
    %86 = llvm.mlir.constant(66 : i32) : i32
    %87 = llvm.getelementptr inbounds %58[%7, %18] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<50 x i32>
    %88 = llvm.mlir.constant("Arr_1_Glob[8]:       %d\0A\00") : !llvm.array<25 x i8>
    %89 = llvm.mlir.addressof @".str.22" : !llvm.ptr
    %90 = llvm.mlir.constant("Arr_2_Glob[8][7]:    %d\0A\00") : !llvm.array<25 x i8>
    %91 = llvm.mlir.addressof @".str.23" : !llvm.ptr
    %92 = llvm.mlir.constant("        should be:   Number_Of_Runs + 10\00") : !llvm.array<41 x i8>
    %93 = llvm.mlir.addressof @str.47 : !llvm.ptr
    %94 = llvm.mlir.constant("Ptr_Glob->\00") : !llvm.array<11 x i8>
    %95 = llvm.mlir.addressof @str.48 : !llvm.ptr
    %96 = llvm.mlir.constant("  Discr:             %d\0A\00") : !llvm.array<25 x i8>
    %97 = llvm.mlir.addressof @".str.26" : !llvm.ptr
    %98 = llvm.mlir.constant("  Enum_Comp:         %d\0A\00") : !llvm.array<25 x i8>
    %99 = llvm.mlir.addressof @".str.27" : !llvm.ptr
    %100 = llvm.mlir.constant("  Int_Comp:          %d\0A\00") : !llvm.array<25 x i8>
    %101 = llvm.mlir.addressof @".str.28" : !llvm.ptr
    %102 = llvm.mlir.constant(17 : i32) : i32
    %103 = llvm.mlir.constant("  Str_Comp:          %s\0A\00") : !llvm.array<25 x i8>
    %104 = llvm.mlir.addressof @".str.29" : !llvm.ptr
    %105 = llvm.mlir.constant("        should be:   DHRYSTONE PROGRAM, SOME STRING\00") : !llvm.array<52 x i8>
    %106 = llvm.mlir.addressof @str.51 : !llvm.ptr
    %107 = llvm.mlir.constant("Next_Ptr_Glob->\00") : !llvm.array<16 x i8>
    %108 = llvm.mlir.addressof @str.50 : !llvm.ptr
    %109 = llvm.mlir.constant(18 : i32) : i32
    %110 = llvm.mlir.constant("Int_1_Loc:           %d\0A\00") : !llvm.array<25 x i8>
    %111 = llvm.mlir.addressof @".str.32" : !llvm.ptr
    %112 = llvm.mlir.constant("Int_2_Loc:           %d\0A\00") : !llvm.array<25 x i8>
    %113 = llvm.mlir.addressof @".str.33" : !llvm.ptr
    %114 = llvm.mlir.constant(13 : i32) : i32
    %115 = llvm.mlir.constant("Int_3_Loc:           %d\0A\00") : !llvm.array<25 x i8>
    %116 = llvm.mlir.addressof @".str.34" : !llvm.ptr
    %117 = llvm.mlir.constant("Enum_Loc:            %d\0A\00") : !llvm.array<25 x i8>
    %118 = llvm.mlir.addressof @".str.35" : !llvm.ptr
    %119 = llvm.mlir.constant("Str_1_Loc:           %s\0A\00") : !llvm.array<25 x i8>
    %120 = llvm.mlir.addressof @".str.36" : !llvm.ptr
    %121 = llvm.mlir.constant("        should be:   DHRYSTONE PROGRAM, 1'ST STRING\00") : !llvm.array<52 x i8>
    %122 = llvm.mlir.addressof @str.52 : !llvm.ptr
    %123 = llvm.mlir.constant("Str_2_Loc:           %s\0A\00") : !llvm.array<25 x i8>
    %124 = llvm.mlir.addressof @".str.38" : !llvm.ptr
    %125 = llvm.mlir.constant("        should be:   DHRYSTONE PROGRAM, 2'ND STRING\00") : !llvm.array<52 x i8>
    %126 = llvm.mlir.addressof @str.53 : !llvm.ptr
    %127 = llvm.mlir.addressof @User_Time : !llvm.ptr
    %128 = llvm.mlir.constant(2 : i64) : i64
    %129 = llvm.mlir.constant("Microseconds for one run through Dhrystone: \00") : !llvm.array<45 x i8>
    %130 = llvm.mlir.addressof @".str.42" : !llvm.ptr
    %131 = llvm.mlir.constant("Removed. Begin_Time: %ld ; End_Time: %ld ; Delta: %ld\0A\00") : !llvm.array<55 x i8>
    %132 = llvm.mlir.addressof @".str.43" : !llvm.ptr
    %133 = llvm.mlir.constant("Measured time too small to obtain meaningful results\00") : !llvm.array<53 x i8>
    %134 = llvm.mlir.addressof @str.54 : !llvm.ptr
    %135 = llvm.mlir.constant("Please increase number of runs\00") : !llvm.array<31 x i8>
    %136 = llvm.mlir.addressof @str.55 : !llvm.ptr
    %137 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %138 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %139 = llvm.alloca %0 x !llvm.array<31 x i8> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %140 = llvm.alloca %0 x !llvm.array<31 x i8> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %141 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %142 = llvm.call @malloc(%3) : (i64) -> !llvm.ptr
    llvm.store %142, %5 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr, !llvm.ptr
    %143 = llvm.call @malloc(%3) : (i64) -> !llvm.ptr
    llvm.store %143, %6 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr, !llvm.ptr
    llvm.store %142, %143 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr, !llvm.ptr
    %144 = llvm.getelementptr inbounds %143[%7, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %8, %144 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : i32, !llvm.ptr
    %145 = llvm.getelementptr inbounds %143[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %9, %145 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %146 = llvm.getelementptr inbounds %143[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %10, %146 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %147 = llvm.getelementptr inbounds %143[%7, 2, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    "llvm.intr.memcpy"(%147, %12, %2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "llvm.intr.memcpy"(%139, %15, %2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.store %16, %22 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    %148 = llvm.call @putchar(%16) : (i32) -> i32
    %149 = llvm.call @puts(%24) : (!llvm.ptr) -> i32
    %150 = llvm.load %25 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %151 = llvm.icmp "eq" %150, %8 : i32
    %152 = llvm.select %151, %27, %29 : i1, !llvm.ptr
    %153 = llvm.call @puts(%152) : (!llvm.ptr) -> i32
    %154 = llvm.load %30 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %155 = llvm.icmp "eq" %154, %8 : i32
    llvm.cond_br %155, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %156 = llvm.call @printf(%32, %1) : (!llvm.ptr, i64) -> i32
    llvm.br ^bb3
  ^bb2:  // pred: ^bb0
    %157 = llvm.call @printf(%34, %18) : (!llvm.ptr, i64) -> i32
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %158 = llvm.call @printf(%36, %37) : (!llvm.ptr, i32) -> i32
    %159 = llvm.call @putchar(%16) : (i32) -> i32
    %160 = llvm.icmp "slt" %arg0, %9 : i32
    llvm.cond_br %160, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %161 = llvm.call @printf(%40) : (!llvm.ptr) -> i32
    %162 = llvm.call @__isoc99_scanf(%42, %141) : (!llvm.ptr, !llvm.ptr) -> i32
    %163 = llvm.load %141 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %164 = llvm.call @putchar(%16) : (i32) -> i32
    llvm.br ^bb6(%163 : i32)
  ^bb5:  // pred: ^bb3
    %165 = llvm.getelementptr inbounds %arg1[%38] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
    %166 = llvm.load %165 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %167 = llvm.call @strtol(%166, %4, %16) : (!llvm.ptr, !llvm.ptr, i32) -> i64
    %168 = llvm.trunc %167 : i64 to i32
    llvm.br ^bb6(%168 : i32)
  ^bb6(%169: i32):  // 2 preds: ^bb4, ^bb5
    %170 = llvm.call @printf(%44, %169) : (!llvm.ptr, i32) -> i32
    %171 = llvm.call %45(%4) : !llvm.ptr, (!llvm.ptr) -> i64
    llvm.store %171, %46 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i64, !llvm.ptr
    %172 = llvm.icmp "slt" %169, %0 : i32
    llvm.cond_br %172, ^bb18(%47, %47 : i32, i32), ^bb7(%0 : i32)
  ^bb7(%173: i32):  // 2 preds: ^bb6, ^bb16
    llvm.store %48, %50 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i8, !llvm.ptr
    llvm.store %0, %51 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.store %52, %53 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i8, !llvm.ptr
    "llvm.intr.memcpy"(%140, %55, %2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.store %0, %138 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %174 = llvm.call @Func_2(%139, %140) : (!llvm.ptr, !llvm.ptr) -> i32
    %175 = llvm.icmp "eq" %174, %8 : i32
    %176 = llvm.zext %175 : i1 to i32
    llvm.store %176, %51 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.store %56, %137 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.call @Proc_7(%9, %57, %137) : (i32, i32, !llvm.ptr) -> ()
    %177 = llvm.load %137 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    llvm.call @Proc_8(%58, %21, %57, %177) : (!llvm.ptr, !llvm.ptr, i32, i32) -> ()
    %178 = llvm.load %6 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %179 = llvm.load %178 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%179, %178, %3) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %180 = llvm.getelementptr inbounds %178[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %59, %180 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %181 = llvm.getelementptr inbounds %179[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %59, %181 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %182 = llvm.load %178 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %182, %179 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr, !llvm.ptr
    %183 = llvm.load %6 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %184 = llvm.icmp "eq" %183, %4 : !llvm.ptr
    llvm.cond_br %184, ^bb9(%4 : !llvm.ptr), ^bb8
  ^bb8:  // pred: ^bb7
    %185 = llvm.load %183 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %185, %179 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr, !llvm.ptr
    %186 = llvm.load %6 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb9(%186 : !llvm.ptr)
  ^bb9(%187: !llvm.ptr):  // 2 preds: ^bb7, ^bb8
    %188 = llvm.load %60 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %189 = llvm.getelementptr inbounds %187[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.call @Proc_7(%16, %188, %189) : (i32, i32, !llvm.ptr) -> ()
    %190 = llvm.getelementptr inbounds %179[%7, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %191 = llvm.load %190 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> i32
    %192 = llvm.icmp "eq" %191, %8 : i32
    llvm.cond_br %192, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %193 = llvm.getelementptr inbounds %179[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %194 = llvm.getelementptr inbounds %178[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %61, %181 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %195 = llvm.load %194 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    llvm.call @Proc_6(%195, %193) : (i32, !llvm.ptr) -> ()
    %196 = llvm.load %6 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %197 = llvm.load %196 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %197, %179 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr, !llvm.ptr
    %198 = llvm.load %181 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    llvm.call @Proc_7(%198, %16, %181) : (i32, i32, !llvm.ptr) -> ()
    llvm.br ^bb12
  ^bb11:  // pred: ^bb9
    %199 = llvm.load %178 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%178, %199, %3) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb12
  ^bb12:  // 2 preds: ^bb10, ^bb11
    %200 = llvm.load %53 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %201 = llvm.icmp "slt" %200, %48 : i8
    llvm.cond_br %201, ^bb16(%57 : i32), ^bb13(%48, %57 : i8, i32)
  ^bb13(%202: i8, %203: i32):  // 2 preds: ^bb12, ^bb15
    %204 = llvm.load %138 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %205 = llvm.call %62(%202, %63) : !llvm.ptr, (i8, i8) -> i32
    %206 = llvm.icmp "eq" %204, %205 : i32
    llvm.cond_br %206, ^bb14, ^bb15(%203 : i32)
  ^bb14:  // pred: ^bb13
    llvm.call @Proc_6(%8, %138) : (i32, !llvm.ptr) -> ()
    "llvm.intr.memcpy"(%140, %65, %2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.store %173, %60 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.br ^bb15(%173 : i32)
  ^bb15(%207: i32):  // 2 preds: ^bb13, ^bb14
    %208 = llvm.add %202, %66  : i8
    %209 = llvm.load %53 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %210 = llvm.icmp "sgt" %208, %209 : i8
    llvm.cond_br %210, ^bb16(%207 : i32), ^bb13(%208, %207 : i8, i32)
  ^bb16(%211: i32):  // 2 preds: ^bb12, ^bb15
    %212 = llvm.add %173, %0  : i32
    %213 = llvm.icmp "eq" %173, %169 : i32
    llvm.cond_br %213, ^bb17, ^bb7(%212 : i32)
  ^bb17:  // pred: ^bb16
    %214 = llvm.load %60 {alignment = 4 : i64} : !llvm.ptr -> i32
    %215 = llvm.load %50 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %216 = llvm.load %137 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %217 = llvm.mul %211, %57  : i32
    %218 = llvm.sdiv %217, %216  : i32
    %219 = llvm.sub %217, %216  : i32
    %220 = llvm.mul %219, %56  : i32
    %221 = llvm.sub %220, %218  : i32
    %222 = llvm.icmp "eq" %215, %48 : i8
    %223 = llvm.add %218, %67  : i32
    %224 = llvm.sub %223, %214  : i32
    %225 = llvm.select %222, %224, %218 : i1, i32
    llvm.br ^bb18(%225, %221 : i32, i32)
  ^bb18(%226: i32, %227: i32):  // 2 preds: ^bb6, ^bb17
    %228 = llvm.call %45(%4) : !llvm.ptr, (!llvm.ptr) -> i64
    llvm.store %228, %68 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i64, !llvm.ptr
    %229 = llvm.call @puts(%70) : (!llvm.ptr) -> i32
    %230 = llvm.call @putchar(%16) : (i32) -> i32
    %231 = llvm.call @puts(%72) : (!llvm.ptr) -> i32
    %232 = llvm.call @putchar(%16) : (i32) -> i32
    %233 = llvm.load %60 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %234 = llvm.call @printf(%74, %233) : (!llvm.ptr, i32) -> i32
    %235 = llvm.call @printf(%76, %59) : (!llvm.ptr, i32) -> i32
    %236 = llvm.load %51 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %237 = llvm.call @printf(%78, %236) : (!llvm.ptr, i32) -> i32
    %238 = llvm.call @printf(%76, %0) : (!llvm.ptr, i32) -> i32
    %239 = llvm.load %50 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %240 = llvm.sext %239 : i8 to i32
    %241 = llvm.call @printf(%80, %240) : (!llvm.ptr, i32) -> i32
    %242 = llvm.call @printf(%82, %83) : (!llvm.ptr, i32) -> i32
    %243 = llvm.load %53 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %244 = llvm.sext %243 : i8 to i32
    %245 = llvm.call @printf(%85, %244) : (!llvm.ptr, i32) -> i32
    %246 = llvm.call @printf(%82, %86) : (!llvm.ptr, i32) -> i32
    %247 = llvm.load %87 {alignment = 16 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %248 = llvm.call @printf(%89, %247) : (!llvm.ptr, i32) -> i32
    %249 = llvm.call @printf(%76, %56) : (!llvm.ptr, i32) -> i32
    %250 = llvm.load %22 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %251 = llvm.call @printf(%91, %250) : (!llvm.ptr, i32) -> i32
    %252 = llvm.call @puts(%93) : (!llvm.ptr) -> i32
    %253 = llvm.call @puts(%95) : (!llvm.ptr) -> i32
    %254 = llvm.load %6 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %255 = llvm.getelementptr inbounds %254[%7, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %256 = llvm.load %255 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> i32
    %257 = llvm.call @printf(%97, %256) : (!llvm.ptr, i32) -> i32
    %258 = llvm.call @printf(%76, %8) : (!llvm.ptr, i32) -> i32
    %259 = llvm.load %6 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %260 = llvm.getelementptr inbounds %259[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %261 = llvm.load %260 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %262 = llvm.call @printf(%99, %261) : (!llvm.ptr, i32) -> i32
    %263 = llvm.call @printf(%76, %9) : (!llvm.ptr, i32) -> i32
    %264 = llvm.load %6 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %265 = llvm.getelementptr inbounds %264[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %266 = llvm.load %265 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %267 = llvm.call @printf(%101, %266) : (!llvm.ptr, i32) -> i32
    %268 = llvm.call @printf(%76, %102) : (!llvm.ptr, i32) -> i32
    %269 = llvm.load %6 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %270 = llvm.getelementptr inbounds %269[%7, 2, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %271 = llvm.call @printf(%104, %270) : (!llvm.ptr, !llvm.ptr) -> i32
    %272 = llvm.call @puts(%106) : (!llvm.ptr) -> i32
    %273 = llvm.call @puts(%108) : (!llvm.ptr) -> i32
    %274 = llvm.load %5 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %275 = llvm.getelementptr inbounds %274[%7, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %276 = llvm.load %275 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> i32
    %277 = llvm.call @printf(%97, %276) : (!llvm.ptr, i32) -> i32
    %278 = llvm.call @printf(%76, %8) : (!llvm.ptr, i32) -> i32
    %279 = llvm.load %5 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %280 = llvm.getelementptr inbounds %279[%7, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %281 = llvm.load %280 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %282 = llvm.call @printf(%99, %281) : (!llvm.ptr, i32) -> i32
    %283 = llvm.call @printf(%76, %0) : (!llvm.ptr, i32) -> i32
    %284 = llvm.load %5 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %285 = llvm.getelementptr inbounds %284[%7, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %286 = llvm.load %285 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %287 = llvm.call @printf(%101, %286) : (!llvm.ptr, i32) -> i32
    %288 = llvm.call @printf(%76, %109) : (!llvm.ptr, i32) -> i32
    %289 = llvm.load %5 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %290 = llvm.getelementptr inbounds %289[%7, 2, 0, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %291 = llvm.call @printf(%104, %290) : (!llvm.ptr, !llvm.ptr) -> i32
    %292 = llvm.call @puts(%106) : (!llvm.ptr) -> i32
    %293 = llvm.call @printf(%111, %226) : (!llvm.ptr, i32) -> i32
    %294 = llvm.call @printf(%76, %59) : (!llvm.ptr, i32) -> i32
    %295 = llvm.call @printf(%113, %227) : (!llvm.ptr, i32) -> i32
    %296 = llvm.call @printf(%76, %114) : (!llvm.ptr, i32) -> i32
    %297 = llvm.load %137 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %298 = llvm.call @printf(%116, %297) : (!llvm.ptr, i32) -> i32
    %299 = llvm.call @printf(%76, %56) : (!llvm.ptr, i32) -> i32
    %300 = llvm.load %138 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %301 = llvm.call @printf(%118, %300) : (!llvm.ptr, i32) -> i32
    %302 = llvm.call @printf(%76, %0) : (!llvm.ptr, i32) -> i32
    %303 = llvm.call @printf(%120, %139) : (!llvm.ptr, !llvm.ptr) -> i32
    %304 = llvm.call @puts(%122) : (!llvm.ptr) -> i32
    %305 = llvm.call @printf(%124, %140) : (!llvm.ptr, !llvm.ptr) -> i32
    %306 = llvm.call @puts(%126) : (!llvm.ptr) -> i32
    %307 = llvm.call @putchar(%16) : (i32) -> i32
    %308 = llvm.load %68 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i64
    %309 = llvm.load %46 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i64
    %310 = llvm.sub %308, %309  : i64
    llvm.store %310, %127 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i64, !llvm.ptr
    %311 = llvm.icmp "slt" %310, %128 : i64
    llvm.cond_br %311, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %312 = llvm.call @puts(%134) : (!llvm.ptr) -> i32
    %313 = llvm.call @puts(%136) : (!llvm.ptr) -> i32
    %314 = llvm.call @putchar(%16) : (i32) -> i32
    llvm.br ^bb21
  ^bb20:  // pred: ^bb18
    %315 = llvm.call @printf(%130) : (!llvm.ptr) -> i32
    %316 = llvm.load %46 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i64
    %317 = llvm.load %68 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i64
    %318 = llvm.sub %317, %316  : i64
    %319 = llvm.call @printf(%132, %316, %317, %318) : (!llvm.ptr, i64, i64, i64) -> i32
    llvm.br ^bb21
  ^bb21:  // 2 preds: ^bb19, ^bb20
    llvm.return %8 : i32
  }
  llvm.func local_unnamed_addr @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["allockind", "9"], ["allocsize", "4294967295"], ["alloc-family", "malloc"], ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @putchar(i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind"]}
  llvm.func local_unnamed_addr @puts(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind"]}
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @__isoc99_scanf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {passthrough = ["nofree", "nounwind", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @strtol(!llvm.ptr {llvm.noundef, llvm.readonly}, !llvm.ptr {llvm.nocapture, llvm.noundef}, i32 {llvm.noundef}) -> i64 attributes {passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @time(...) -> i64 attributes {passthrough = [["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
  llvm.func local_unnamed_addr @Proc_5() attributes {memory = #llvm.memory_effects<other = write, argMem = write, inaccessibleMem = write>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(65 : i8) : i8
    %1 = llvm.mlir.constant(0 : i8) : i8
    %2 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.addressof @Bool_Glob : !llvm.ptr
    llvm.store %0, %2 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i8, !llvm.ptr
    llvm.store %3, %4 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @Proc_4() attributes {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %2 = llvm.mlir.constant(65 : i8) : i8
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.addressof @Bool_Glob : !llvm.ptr
    %5 = llvm.mlir.constant(66 : i8) : i8
    %6 = llvm.mlir.addressof @Ch_2_Glob : !llvm.ptr
    %7 = llvm.load %1 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %8 = llvm.icmp "eq" %7, %2 : i8
    %9 = llvm.zext %8 : i1 to i32
    %10 = llvm.load %4 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %11 = llvm.or %10, %9  : i32
    llvm.store %11, %4 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.store %5, %6 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i8, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @Proc_1(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {passthrough = ["nounwind", "sspstrong", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
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
    %12 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    %13 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%12, %13, %2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %14 = llvm.getelementptr inbounds %arg0[%4, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %8, %14 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %15 = llvm.getelementptr inbounds %12[%4, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %8, %15 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %16 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %16, %12 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr, !llvm.ptr
    %17 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %18 = llvm.icmp "eq" %17, %0 : !llvm.ptr
    llvm.cond_br %18, ^bb2(%0 : !llvm.ptr), ^bb1
  ^bb1:  // pred: ^bb0
    %19 = llvm.load %17 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %19, %12 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr, !llvm.ptr
    %20 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb2(%20 : !llvm.ptr)
  ^bb2(%21: !llvm.ptr):  // 2 preds: ^bb0, ^bb1
    %22 = llvm.load %9 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %23 = llvm.getelementptr inbounds %21[%4, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.call @Proc_7(%10, %22, %23) : (i32, i32, !llvm.ptr) -> ()
    %24 = llvm.getelementptr inbounds %12[%4, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %25 = llvm.load %24 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> i32
    %26 = llvm.icmp "eq" %25, %6 : i32
    llvm.cond_br %26, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %27 = llvm.getelementptr inbounds %12[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    %28 = llvm.getelementptr inbounds %arg0[%4, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.store %11, %15 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %29 = llvm.load %28 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    llvm.call @Proc_6(%29, %27) : (i32, !llvm.ptr) -> ()
    %30 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %31 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %31, %12 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr, !llvm.ptr
    %32 = llvm.load %15 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    llvm.call @Proc_7(%32, %10, %15) : (i32, i32, !llvm.ptr) -> ()
    llvm.br ^bb5
  ^bb4:  // pred: ^bb2
    %33 = llvm.load %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    "llvm.intr.memcpy"(%arg0, %33, %2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    llvm.return
  }
  llvm.func local_unnamed_addr @Proc_2(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}) attributes {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %2 = llvm.mlir.constant(65 : i8) : i8
    %3 = llvm.mlir.constant(9 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %6 = llvm.load %1 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %7 = llvm.icmp "eq" %6, %2 : i8
    llvm.cond_br %7, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %8 = llvm.load %arg0 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %9 = llvm.add %8, %3  : i32
    %10 = llvm.load %5 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %11 = llvm.sub %9, %10  : i32
    llvm.store %11, %arg0 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.return
  }
  llvm.func local_unnamed_addr @Proc_3(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {passthrough = ["nounwind", "sspstrong", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.null : !llvm.ptr
    %1 = llvm.mlir.addressof @Ptr_Glob : !llvm.ptr
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.mlir.constant(2 : i32) : i32
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(10 : i32) : i32
    %8 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %9 = llvm.icmp "eq" %8, %0 : !llvm.ptr
    llvm.cond_br %9, ^bb2(%0 : !llvm.ptr), ^bb1
  ^bb1:  // pred: ^bb0
    %10 = llvm.load %8 {alignment = 8 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> !llvm.ptr
    llvm.store %10, %arg0 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr, !llvm.ptr
    %11 = llvm.load %1 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    llvm.br ^bb2(%11 : !llvm.ptr)
  ^bb2(%12: !llvm.ptr):  // 2 preds: ^bb0, ^bb1
    %13 = llvm.load %3 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %14 = llvm.getelementptr inbounds %12[%4, 2, 0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.record", (ptr, i32, struct<"union.anon", (struct<"struct.anon", (i32, i32, array<31 x i8>)>)>)>
    llvm.call @Proc_7(%7, %13, %14) : (i32, i32, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @Proc_6(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
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
    %8 = llvm.load %4 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %9 = llvm.icmp "sgt" %8, %5 : i32
    %10 = llvm.select %9, %3, %1 : i1, i32
    llvm.br ^bb4(%10 : i32)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb4(%2 : i32)
  ^bb3:  // pred: ^bb0
    llvm.br ^bb4(%0 : i32)
  ^bb4(%11: i32):  // 4 preds: ^bb0, ^bb1, ^bb2, ^bb3
    llvm.store %11, %arg1 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return
  }
  llvm.func local_unnamed_addr @Func_3(%arg0: i32 {llvm.noundef}) -> i32 attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.icmp "eq" %arg0, %0 : i32
    %2 = llvm.zext %1 : i1 to i32
    llvm.return %2 : i32
  }
  llvm.func local_unnamed_addr @Proc_7(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.add %arg0, %0  : i32
    %2 = llvm.add %1, %arg1  : i32
    llvm.store %2, %arg2 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @Proc_8(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: i32 {llvm.noundef}) attributes {passthrough = ["nofree", "norecurse", "nosync", "nounwind", "sspstrong", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
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
    llvm.store %arg3, %11 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    %12 = llvm.add %arg2, %1  : i32
    %13 = llvm.sext %12 : i32 to i64
    %14 = llvm.getelementptr inbounds %arg0[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %arg3, %14 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    %15 = llvm.add %arg2, %2  : i32
    %16 = llvm.sext %15 : i32 to i64
    %17 = llvm.getelementptr inbounds %arg0[%16] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %9, %17 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    %18 = llvm.getelementptr inbounds %arg1[%10, %10] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<50 x i32>
    llvm.store %9, %18 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    %19 = llvm.add %10, %3  : i64
    %20 = llvm.getelementptr inbounds %arg1[%10, %19] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<50 x i32>
    llvm.store %9, %20 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    %21 = llvm.add %arg2, %4  : i32
    %22 = llvm.sext %21 : i32 to i64
    %23 = llvm.getelementptr inbounds %arg1[%10, %22] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<50 x i32>
    %24 = llvm.load %23 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %25 = llvm.add %24, %5  : i32
    llvm.store %25, %23 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    %26 = llvm.load %11 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : !llvm.ptr -> i32
    %27 = llvm.add %arg2, %6  : i32
    %28 = llvm.sext %27 : i32 to i64
    %29 = llvm.getelementptr inbounds %arg1[%28, %10] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<50 x i32>
    llvm.store %26, %29 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.store %0, %8 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @Func_1(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> i32 attributes {memory = #llvm.memory_effects<other = write, argMem = write, inaccessibleMem = write>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "sspstrong", "willreturn", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
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
    llvm.store %8, %3 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i8, !llvm.ptr
    llvm.br ^bb2(%4 : i32)
  ^bb2(%9: i32):  // 2 preds: ^bb0, ^bb1
    llvm.return %9 : i32
  }
  llvm.func local_unnamed_addr @Func_2(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> i32 attributes {passthrough = ["nofree", "nounwind", "sspstrong", ["uwtable", "2"], ["frame-pointer", "none"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(2 : i64) : i64
    %1 = llvm.mlir.constant(3 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(10 : i32) : i32
    %4 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(0 : i8) : i8
    %7 = llvm.mlir.addressof @Ch_1_Glob : !llvm.ptr
    %8 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %9 = llvm.load %8 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %10 = llvm.getelementptr inbounds %arg1[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %11 = llvm.load %10 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i8
    %12 = llvm.icmp "eq" %11, %9 : i8
    llvm.cond_br %12, ^bb1, ^bb3
  ^bb1:  // pred: ^bb0
    llvm.store %9, %7 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i8, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb2
  ^bb3:  // pred: ^bb0
    %13 = llvm.call @strcmp(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> i32
    %14 = llvm.icmp "sgt" %13, %2 : i32
    llvm.cond_br %14, ^bb4, ^bb5(%2 : i32)
  ^bb4:  // pred: ^bb3
    llvm.store %3, %4 {alignment = 4 : i64, tbaa = [#tbaa_tag2]} : i32, !llvm.ptr
    llvm.br ^bb5(%5 : i32)
  ^bb5(%15: i32):  // 2 preds: ^bb3, ^bb4
    llvm.return %15 : i32
  }
  llvm.func local_unnamed_addr @strcmp(!llvm.ptr {llvm.nocapture, llvm.noundef}, !llvm.ptr {llvm.nocapture, llvm.noundef}) -> i32 attributes {memory = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["frame-pointer", "none"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

