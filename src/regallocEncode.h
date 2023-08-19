#pragma once

#include <mlir/IR/BuiltinOps.h>
#include "AMD64/AMD64Types.h"

bool debugEncodeOp(mlir::ModuleOp mod, amd64::InstructionOpInterface op);
/// on error: returns nullptr
/// otherwise: returns the address at which to start executing, if JIT is enabled, otherwise returns buf
MCDescriptor regallocEncode(uint8_t* buf, uint8_t* bufEnd, mlir::ModuleOp mod, amd64::GlobalsInfo&& globals, bool dumpAsm = false, bool jit = false, llvm::StringRef startSymbolIfJIT = llvm::StringRef());
