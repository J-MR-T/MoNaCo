#pragma once
#include <mlir/IR/Operation.h>

/// TODO
/// takes an operation and does isel on its regions
void prototypeIsel(mlir::Operation* regionOp);
