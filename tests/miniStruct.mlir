// RUN: %FileCheckExecOutput --allow-empty %s

// CHECK-NOT: {{.+}}

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = ["noinline", "nounwind", "optnone", "sspstrong", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i64) : i64
    %3 = llvm.mlir.constant(10 : i32) : i32
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(20 : i64) : i64
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.constant(30 : i32) : i32
    %8 = llvm.mlir.constant(40 : i32) : i32
    %9 = llvm.mlir.constant(50 : i32) : i32
    %10 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %13 = llvm.alloca %0 x !llvm.array<2 x struct<"struct.a", (i32, i64, struct<"struct.anon", (i32, i32)>)>> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    llvm.store %1, %10 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg0, %11 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %12 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    %14 = llvm.getelementptr inbounds %13[%2, %2] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x struct<"struct.a", (i32, i64, struct<"struct.anon", (i32, i32)>)>>
    %15 = llvm.getelementptr inbounds %14[%1, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.a", (i32, i64, struct<"struct.anon", (i32, i32)>)>
    llvm.store %3, %15 {alignment = 16 : i64} : i32, !llvm.ptr
    %16 = llvm.getelementptr inbounds %13[%2, %4] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x struct<"struct.a", (i32, i64, struct<"struct.anon", (i32, i32)>)>>
    %17 = llvm.getelementptr inbounds %16[%1, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.a", (i32, i64, struct<"struct.anon", (i32, i32)>)>
    llvm.store %5, %17 {alignment = 8 : i64} : i64, !llvm.ptr
    %18 = llvm.getelementptr inbounds %13[%2, %2] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x struct<"struct.a", (i32, i64, struct<"struct.anon", (i32, i32)>)>>
    %19 = llvm.getelementptr inbounds %18[%1, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.a", (i32, i64, struct<"struct.anon", (i32, i32)>)>
    %20 = llvm.getelementptr inbounds %19[%1, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.anon", (i32, i32)>
    llvm.store %7, %20 {alignment = 16 : i64} : i32, !llvm.ptr
    %21 = llvm.getelementptr inbounds %13[%2, %4] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x struct<"struct.a", (i32, i64, struct<"struct.anon", (i32, i32)>)>>
    %22 = llvm.getelementptr inbounds %21[%1, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.a", (i32, i64, struct<"struct.anon", (i32, i32)>)>
    %23 = llvm.getelementptr inbounds %22[%1, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.anon", (i32, i32)>
    llvm.store %8, %23 {alignment = 4 : i64} : i32, !llvm.ptr
    %24 = llvm.load %11 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %1 : i32
  }
}

