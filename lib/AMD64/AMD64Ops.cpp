#include "AMD64/AMD64Ops.h"
#include "AMD64/AMD64Dialect.h"

#include "mlir/IR/OpImplementation.h"

#include <concepts>
#include <cstdint>
#include <err.h>

#include <mlir/Support/LogicalResult.h>
using mlir::success;
using mlir::failure;
using mlir::failed;

#include "mlir/IR/OperationSupport.h"

// either include ODSSupport.h (which doesn't have all of the required convertToAttribute/convertFromAttribute methods for properties)
//#include "mlir/IR/ODSSupport.h"

// or define them yourself:
template<std::signed_integral T>
mlir::LogicalResult convertFromAttribute(T &storage,
                                         ::mlir::Attribute attr,
                                         ::mlir::InFlightDiagnostic *diag) {
  auto valueAttr = dyn_cast<mlir::IntegerAttr>(attr);
  if (!valueAttr) {
    if (diag)
      *diag << "expected IntegerAttr for key `value`";
    return failure();
  }
  storage = valueAttr.getValue().getSExtValue();
  return success();
}
template<std::unsigned_integral T>
mlir::LogicalResult convertFromAttribute(T &storage,
                                         ::mlir::Attribute attr,
                                         ::mlir::InFlightDiagnostic *diag) {
  auto valueAttr = dyn_cast<mlir::IntegerAttr>(attr);
  if (!valueAttr) {
    if (diag)
      *diag << "expected IntegerAttr for key `value`";
    return failure();
  }
  storage = valueAttr.getValue().getZExtValue();
  return success();
}
template<std::integral T>
mlir::Attribute convertToAttribute(mlir::MLIRContext* ctx, T storage) {
  return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, sizeof(T)*8), storage);
}

template<typename T>
mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader&, T&) {
    errx(EXIT_FAILURE, "Conversion to/from bytecode not implemented");
    return mlir::failure();
}

template<typename T>
void writeToMlirBytecode(::mlir::DialectBytecodeWriter&, T&) {
    errx(EXIT_FAILURE, "Conversion to/from bytecode not implemented");
}

template<typename T>
mlir::Attribute convertToAttribute(mlir::MLIRContext* ctx, T){
    errx(EXIT_FAILURE, "Conversion to/from attribute not implemented");
    return mlir::UnitAttr::get(ctx);
}

template<typename T>
mlir::LogicalResult convertFromAttribute(T&, ::mlir::Attribute, ::mlir::InFlightDiagnostic*){
    errx(EXIT_FAILURE, "Conversion to/from attribute not implemented");
    return mlir::failure();
}

// someone didn't qualify their types in mlir-tblgen between 583d492c6 (where this wasn't necessary) and a403d75be7 (where it is)
using namespace llvm;
using namespace mlir;
//template<typename T>
//using ArrayRef = llvm::ArrayRef<T>;

#define GET_OP_CLASSES
#include "AMD64/AMD64Ops.cpp.inc"

#include "AMD64/AMD64OpsEnums.cpp.inc"

#include "AMD64/AMD64OpInterfaces.cpp.inc"
#include "AMD64/AMD64TypeInterfaces.cpp.inc"
