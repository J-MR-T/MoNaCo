#pragma once

#include <cstdint>
#include "AMD64/AMD64Types.h"
#include <llvm/ADT/SmallVector.h>

namespace mlir{
struct Operation;
struct TypeConverter;
struct RewritePatternSet;
}

bool isel(mlir::Operation* regionOp, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns);
bool maximalIsel(mlir::Operation* regionOp, amd64::GlobalsInfo& globals);
