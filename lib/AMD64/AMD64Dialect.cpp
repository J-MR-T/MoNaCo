
#include "AMD64/AMD64Dialect.h"
#include "AMD64/AMD64Ops.h"

using namespace mlir;
using namespace amd64;

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "AMD64/AMD64OpsTypes.cpp.inc"

#include "AMD64/AMD64OpsDialect.cpp.inc"

void AMD64Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AMD64/AMD64Ops.cpp.inc"
      >();

  // TODO this might have to be AMD64OpsTypes instead of AMD64Ops
  addAttributes<
#define GET_ATTRDEF_LIST
#include "AMD64/AMD64Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "AMD64/AMD64OpsTypes.cpp.inc"
      >();
  // fallback, in case this thing above doesn't work: add types manually, like this:
  //addTypes<mlir::b::PointerType>();
}
