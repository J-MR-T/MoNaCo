// RUN: %monaco %s --jit main | FileCheck --match-full-lines %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.mlir.global private unnamed_addr constant @".str"("i = %d, j = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.func @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = ["noinline", "nounwind", "optnone", "sspstrong", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(5 : i32) : i32
    %3 = llvm.mlir.constant(6 : i32) : i32
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(7 : i32) : i32
    %6 = llvm.mlir.constant("i = %d, j = %d\0A\00") : !llvm.array<16 x i8>
    %7 = llvm.mlir.addressof @".str" : !llvm.ptr
    %8 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %10 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %8 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg0, %9 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %10 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %2, %11 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %3, %12 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %12, %13 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %14 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %15 = llvm.getelementptr inbounds %14[%4] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %5, %15 {alignment = 4 : i64} : i32, !llvm.ptr
    %16 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    %17 = llvm.load %12 {alignment = 4 : i64} : !llvm.ptr -> i32
    // CHECK: i = 7, j = 6
    %18 = llvm.call @printf(%7, %16, %17) : (!llvm.ptr, i32, i32) -> i32
    llvm.return %1 : i32
  }
  llvm.func @printf(!llvm.ptr {llvm.noundef}, ...) -> i32 attributes {passthrough = [["frame-pointer", "all"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]}
}

