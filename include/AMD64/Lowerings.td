#ifndef AMD64_LOWERINGS
#define AMD64_LOWERINGS

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "AMD64/AMD64Ops.td"

// TODO assertion failure in mlir-tblgen (https://github.com/llvm/llvm-project/issues/62674)
//def MovExamplePat : Pat<(Arith_ConstantOp I64Attr:$attr), (MOV64ri $attr)>;

// TODO remove this, this is just an example, and is not even correct in that, because ADD8ri is not finished yet, and $attr is just wrong
//def AddExamplePat : Pat<(Arith_AddIOp $input, (Arith_ConstantOp $attr)), (ADD8ri $input, $input, $attr)>;

//def SubExamplePat : Pat<(Arith_SubIOp $op1, $op2), (TestOp64 $op1, $op2)>;

//def ExamplePat2 : Pat<(Arith_SubIOp $op1, (Arith_ConstantOp $op2)), (TestOp64 $op1, $op1)>;

#endif
