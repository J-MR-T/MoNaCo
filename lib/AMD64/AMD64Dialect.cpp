
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

namespace amd64{
struct MemoryOperandAttr : public mlir::Attribute::AttrBase<MemoryOperandAttr, mlir::Attribute, mlir::AttributeStorage> { // TODO needs an actual storage class, with displacement and scale ints
    // inherit necessary constructors
    using Base::Base;

    /// vaguely inspired by the VerCapExtAttr from SPIRVAttributes.h/cpp, which fulfills a similar purpose (structured info about an op)

    /// Create an instance from concrete values, these get converted to attributes which are passed to the second get() function, which passes them to the Base
    static MemoryOperandAttr get(int32_t displacement, uint8_t scale, mlir::MLIRContext* ctx){
        mlir::Builder builder(ctx);
        auto displacementAttr = builder.getI32IntegerAttr(displacement);
        auto scaleAttr = builder.getI8IntegerAttr(scale);
        return get(displacementAttr, scaleAttr, ctx);
    }

    static MemoryOperandAttr get(mlir::IntegerAttr displacement, mlir::IntegerAttr scale, mlir::MLIRContext* ctx){
        return Base::get(ctx, displacement, scale);
    }

    /// Getters for the values
    int32_t getDisplacement(){
        return getImpl()->displacement.getInt();
    }

    uint8_t getScale(){
        return getImpl()->scale.getInt();
    }

};
} // end namespace amd64

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
