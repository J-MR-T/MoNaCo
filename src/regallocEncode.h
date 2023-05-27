#pragma once

#include <mlir/IR/BuiltinOps.h>
#include "AMD64/AMD64Types.h"

bool debugEncodeOp(mlir::ModuleOp mod, amd64::InstructionOpInterface op);
bool regallocEncode(std::vector<uint8_t>& buf, mlir::ModuleOp mod, bool debug = false);
bool regallocEncodeRepeated(std::vector<uint8_t>& buf, mlir::ModuleOp mod, bool debug = false);
