#pragma once

#include <mlir/IR/BuiltinOps.h>
#include "AMD64/AMD64Types.h"

bool debugEncodeOp(mlir::ModuleOp mod, amd64::InstructionOpInterface op);
/// on error: returns nullptr
/// otherwise: returns the address at which to start executing, if JIT is enabled, otherwise returns buf
uint8_t* regallocEncode(uint8_t* buf, uint8_t* bufEnd, mlir::ModuleOp mod, bool dumpAsm = false, bool jit = false, llvm::StringRef startSymbolIfJIT = llvm::StringRef());
bool regallocEncodeRepeated(uint8_t* buf, uint8_t* bufEnd, mlir::ModuleOp mod);
