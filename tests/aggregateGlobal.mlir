// RUN: %MLIRCheckLLVM
// original C code:
//#include<stdio.h>
//
//struct A{
//    int a;
//    int* b;
//    int c;
//};
//
//struct A as[4] = {0};
//
//int bs[4] = {0};
//
//int main(int argc, char *argv[]) {
//    // print all values, make sure they're zero
//    for(int i = 0; i < 4; i++){
//        printf("as[%d].a = %d\n", i, as[i].a);
//        printf("as[%d].b = %p\n", i, (void*)as[i].b);
//        printf("as[%d].c = %d\n", i, as[i].c);
//        printf("bs[%d] = %d\n", i, bs[i]);
//    }
//
//}
#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  llvm.mlir.global external @as() {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<4 x struct<"struct.A", (i32, ptr, i32)>> {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.null : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.struct<"struct.A", (i32, ptr, i32)>
    %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<"struct.A", (i32, ptr, i32)> 
    %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<"struct.A", (i32, ptr, i32)> 
    %5 = llvm.insertvalue %0, %4[2] : !llvm.struct<"struct.A", (i32, ptr, i32)> 
    %6 = llvm.mlir.undef : !llvm.array<4 x struct<"struct.A", (i32, ptr, i32)>>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.array<4 x struct<"struct.A", (i32, ptr, i32)>> 
    %8 = llvm.insertvalue %5, %7[1] : !llvm.array<4 x struct<"struct.A", (i32, ptr, i32)>> 
    %9 = llvm.insertvalue %5, %8[2] : !llvm.array<4 x struct<"struct.A", (i32, ptr, i32)>> 
    %10 = llvm.insertvalue %5, %9[3] : !llvm.array<4 x struct<"struct.A", (i32, ptr, i32)>> 
    llvm.return %10 : !llvm.array<4 x struct<"struct.A", (i32, ptr, i32)>>
  }
  llvm.mlir.global external @bs(dense<0> : tensor<4xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<4 x i32>
  llvm.mlir.global private unnamed_addr constant @".str"("as[%d].a = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.1"("as[%d].b = %p\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.2"("as[%d].c = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @".str.3"("bs[%d] = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(4 : i32) : i32
    %3 = llvm.mlir.null : !llvm.ptr
    %13 = llvm.mlir.addressof @as : !llvm.ptr
    %14 = llvm.mlir.constant(0 : i64) : i64
    %15 = llvm.mlir.constant("as[%d].a = %d\0A\00") : !llvm.array<15 x i8>
    %16 = llvm.mlir.addressof @".str" : !llvm.ptr
    %17 = llvm.mlir.constant("as[%d].b = %p\0A\00") : !llvm.array<15 x i8>
    %18 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %19 = llvm.mlir.constant(2 : i32) : i32
    %20 = llvm.mlir.constant("as[%d].c = %d\0A\00") : !llvm.array<15 x i8>
    %21 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %22 = llvm.mlir.constant(dense<0> : tensor<4xi32>) : !llvm.array<4 x i32>
    %23 = llvm.mlir.addressof @bs : !llvm.ptr
    %24 = llvm.mlir.constant("bs[%d] = %d\0A\00") : !llvm.array<13 x i8>
    %25 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %26 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %27 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %28 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %29 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %26 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg0, %27 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %28 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %1, %29 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %30 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> i32
    %31 = llvm.icmp "slt" %30, %2 : i32
    llvm.cond_br %31, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %32 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> i32
    %33 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> i32
    %34 = llvm.sext %33 : i32 to i64
    %35 = llvm.getelementptr inbounds %13[%14, %34] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x struct<"struct.A", (i32, ptr, i32)>>
    %36 = llvm.getelementptr inbounds %35[%1, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.A", (i32, ptr, i32)>
    %37 = llvm.load %36 {alignment = 8 : i64} : !llvm.ptr -> i32
    %38 = llvm.call @printf(%16, %32, %37) : (!llvm.ptr, i32, i32) -> i32
    %39 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> i32
    %40 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> i32
    %41 = llvm.sext %40 : i32 to i64
    %42 = llvm.getelementptr inbounds %13[%14, %41] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x struct<"struct.A", (i32, ptr, i32)>>
    %43 = llvm.getelementptr inbounds %42[%1, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.A", (i32, ptr, i32)>
    %44 = llvm.load %43 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %45 = llvm.call @printf(%18, %39, %44) : (!llvm.ptr, i32, !llvm.ptr) -> i32
    %46 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> i32
    %47 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> i32
    %48 = llvm.sext %47 : i32 to i64
    %49 = llvm.getelementptr inbounds %13[%14, %48] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x struct<"struct.A", (i32, ptr, i32)>>
    %50 = llvm.getelementptr inbounds %49[%1, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.A", (i32, ptr, i32)>
    %51 = llvm.load %50 {alignment = 8 : i64} : !llvm.ptr -> i32
    %52 = llvm.call @printf(%21, %46, %51) : (!llvm.ptr, i32, i32) -> i32
    %53 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> i32
    %54 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> i32
    %55 = llvm.sext %54 : i32 to i64
    %56 = llvm.getelementptr inbounds %23[%14, %55] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i32>
    %57 = llvm.load %56 {alignment = 4 : i64} : !llvm.ptr -> i32
    %58 = llvm.call @printf(%25, %53, %57) : (!llvm.ptr, i32, i32) -> i32
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %59 = llvm.load %29 {alignment = 4 : i64} : !llvm.ptr -> i32
    %60 = llvm.add %59, %0  : i32
    llvm.store %60, %29 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1 {loop_annotation = #loop_annotation}
  ^bb4:  // pred: ^bb1
    %61 = llvm.load %26 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %61 : i32
  }
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}
