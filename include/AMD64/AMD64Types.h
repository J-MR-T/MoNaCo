#ifndef AMD64_TYPES_H
#define AMD64_TYPES_H

#include "mlir/IR/BuiltinTypes.h"

#include "fadec-enc.h"

using FeMnem = uint64_t;

#define GET_TYPEDEF_CLASSES
#include "AMD64/AMD64OpsTypes.h.inc"

#endif // AMD64_TYPES_H

