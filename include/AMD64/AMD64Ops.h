#ifndef AMD64_OPS_H
#define AMD64_OPS_H

#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "mlir/IR/Builders.h"

#include "mlir/IR/BuiltinTypes.h"
//#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
//#include "mlir/IR/BuiltinTypeInterfaces.h"

#include "AMD64/AMD64Types.h"

#define GET_OP_CLASSES
#include "AMD64/AMD64Ops.h.inc"


#endif

