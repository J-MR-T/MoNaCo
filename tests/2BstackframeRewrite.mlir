// RUN: %FileCheckAsm %s

// the function tests out if the encoder/regallocer can handle patching in a 2 Byte stack frame allocation
// CHECK-NOT: non-decodable
module {
  func.func private @giveI64() -> i64 {
    %0 = arith.constant 65 : i64
    return %0 : i64
  }
  func.func private @moreThan1BstackframeSize() -> () {
    %7 = call @giveI64() : () -> i64
    %15 = "arith.constant"() <{value = 64 : i64}> : () -> i64
    %25 = "arith.cmpi"(%7, %15) <{predicate = 5 : i64}> : (i64, i64) -> i1
    %26 = "arith.extui"(%25) : (i1) -> i64
    %27 = "arith.extui"(%25) : (i1) -> i64
    %28 = "arith.extui"(%25) : (i1) -> i64
    %29 = "arith.extui"(%25) : (i1) -> i64
    %30 = "arith.extui"(%25) : (i1) -> i64
    %31 = "arith.extui"(%25) : (i1) -> i64
    %32 = "arith.extui"(%25) : (i1) -> i64
    %33 = "arith.extui"(%25) : (i1) -> i64
    %34 = "arith.extui"(%25) : (i1) -> i64
    %35 = "arith.extui"(%25) : (i1) -> i64
    %36 = "arith.extui"(%25) : (i1) -> i64
    %37 = "arith.extui"(%25) : (i1) -> i64
    %38 = "arith.extui"(%25) : (i1) -> i64
    %39 = "arith.extui"(%25) : (i1) -> i64
    %40 = "arith.extui"(%25) : (i1) -> i64
    %41 = "arith.extui"(%25) : (i1) -> i64
    %42 = "arith.extui"(%25) : (i1) -> i64
    %43 = "arith.extui"(%25) : (i1) -> i64
    %44 = "arith.extui"(%25) : (i1) -> i64
    %45 = "arith.extui"(%25) : (i1) -> i64
    %46 = "arith.extui"(%25) : (i1) -> i64
    %47 = "arith.extui"(%25) : (i1) -> i64
    %48 = "arith.extui"(%25) : (i1) -> i64
    %49 = "arith.extui"(%25) : (i1) -> i64
    %50 = "arith.extui"(%25) : (i1) -> i64
    %51 = "arith.extui"(%25) : (i1) -> i64
    %52 = "arith.extui"(%25) : (i1) -> i64
    %53 = "arith.extui"(%25) : (i1) -> i64
    %54 = "arith.extui"(%25) : (i1) -> i64
    %55 = "arith.extui"(%25) : (i1) -> i64
    %56 = "arith.extui"(%25) : (i1) -> i64
    %57 = "arith.extui"(%25) : (i1) -> i64
    %58 = "arith.extui"(%25) : (i1) -> i64
    %59 = "arith.extui"(%25) : (i1) -> i64
    %60 = "arith.extui"(%25) : (i1) -> i64
    %61 = "arith.extui"(%25) : (i1) -> i64
    %62 = "arith.extui"(%25) : (i1) -> i64
    %63 = "arith.extui"(%25) : (i1) -> i64
    %64 = "arith.extui"(%25) : (i1) -> i64
    %65 = "arith.extui"(%25) : (i1) -> i64
    %66 = "arith.extui"(%25) : (i1) -> i64
    %67 = "arith.extui"(%25) : (i1) -> i64
    %68 = "arith.extui"(%25) : (i1) -> i64
    %69 = "arith.extui"(%25) : (i1) -> i64
    %70 = "arith.extui"(%25) : (i1) -> i64
    %71 = "arith.extui"(%25) : (i1) -> i64
    %72 = "arith.extui"(%25) : (i1) -> i64
    %73 = "arith.extui"(%25) : (i1) -> i64
    %74 = "arith.extui"(%25) : (i1) -> i64
    %75 = "arith.extui"(%25) : (i1) -> i64
    %76 = "arith.extui"(%25) : (i1) -> i64
    %77 = "arith.extui"(%25) : (i1) -> i64
    %78 = "arith.extui"(%25) : (i1) -> i64
    %79 = "arith.extui"(%25) : (i1) -> i64
    %80 = "arith.extui"(%25) : (i1) -> i64
    %81 = "arith.extui"(%25) : (i1) -> i64
    %82 = "arith.extui"(%25) : (i1) -> i64
    %83 = "arith.extui"(%25) : (i1) -> i64
    %84 = "arith.extui"(%25) : (i1) -> i64
    %85 = "arith.extui"(%25) : (i1) -> i64
    %86 = "arith.extui"(%25) : (i1) -> i64
    return
  }
}
