#ifndef AMD64_LOWERINGS
#define AMD64_LOWERINGS

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "AMD64/AMD64Ops.td"

// TODO remove this, this is just an example, and is not even correct in that, because ADD8ri is not finished yet, and $attr is just wrong
//def AddExamplePat : Pat<(Arith_AddIOp $input, (Arith_ConstantOp $attr)), (ADD8ri $input, $input, $attr)>;

#endif
