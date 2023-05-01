#ifndef AMD64_OPS_H
#define AMD64_OPS_H

// TODO depending on if this is changed, remove this include here
#include "fadec-enc.h"

#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/Dialect.h"
// predefined interfaces
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
// my own interfaces TODO i don't think this is right
#include "AMD64/InstructionOpInterface.h.inc"

#include "mlir/IR/Builders.h"

#include "mlir/IR/BuiltinTypes.h"
//#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
//#include "mlir/IR/BuiltinTypeInterfaces.h"

#include "AMD64/AMD64Types.h"

// TODO is this right at this point? or does it need to come before/after AMD64Ops.h.inc?
// pull all enum type definitions in
#include "AMD64/AMD64OpsEnums.h.inc"

#define GET_OP_CLASSES
#include "AMD64/AMD64Ops.h.inc"

#endif

