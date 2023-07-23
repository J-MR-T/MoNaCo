// RUN: %FileCheckAsm %s

module{
  llvm.func @test1() {
    llvm.return
  }
  // CHECK: ret

  llvm.func @main(){
     llvm.call @test1() : () -> ()
     llvm.return
  }
  // call with negative rip relative addr, so has to start with quite a lot of f's, we just check 1
  // CHECK: call 0xf
  // CHECK: ret

  func.func @test2(){
    return
  }
  // CHECK: ret

  func.func @foo(){
    call @test2() : () -> ()
    return
  }
  // CHECK: call 0xf
  // CHECK: ret

}
