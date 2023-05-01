#include "AMD64/AMD64Ops.h"
#include "AMD64/AMD64Dialect.h"
#include "mlir/IR/OpImplementation.h"

#include <cstdint>

#define GET_OP_CLASSES
#include "AMD64/AMD64Ops.cpp.inc"

#include "AMD64/AMD64OpsEnums.cpp.inc"

#include "AMD64/InstructionOpInterface.cpp.inc"
