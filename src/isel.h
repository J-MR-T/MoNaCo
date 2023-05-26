#pragma once
#include <mlir/IR/Operation.h>

/// TODO
/// takes an operation and does isel on its regions
/// returns whether doing isel failed
bool prototypeIsel(mlir::Operation* regionOp);
