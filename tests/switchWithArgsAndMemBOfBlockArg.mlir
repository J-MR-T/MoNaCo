// RUN: %monaco %s --jit main

module{
  llvm.mlir.global external local_unnamed_addr @Int_Glob(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
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
    %8 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
    %9 = llvm.icmp "sgt" %8, %5 : i32
    %10 = llvm.select %9, %3, %1 : i1, i32
    llvm.br ^bb4(%10 : i32)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb4(%2 : i32)
  ^bb3:  // pred: ^bb0
    llvm.br ^bb4(%0 : i32)
  ^bb4(%11: i32):  // 4 preds: ^bb0, ^bb1, ^bb2, ^bb3
    llvm.store %11, %arg1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return
  }

  llvm.func @main() -> i32{
    %0 = llvm.mlir.constant(0 : i32) : i32
    %42 = llvm.mlir.constant(42 : i32) : i32
    %1 = llvm.mlir.addressof @Int_Glob : !llvm.ptr
    llvm.store %42, %1 {alignment = 4 : i64} : i32, !llvm.ptr // first write 42 there
    llvm.call @Proc_6(%0, %1) : (i32, !llvm.ptr) -> ()
    %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32 // then check that the call wrote 0 there
    llvm.return %2 : i32
  }
}
