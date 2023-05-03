#ifndef AMD64_TYPES_H
#define AMD64_TYPES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "fadec-enc.h"

#include "mlir/IR/OpDefinition.h"

using FeMnem = uint64_t;

// this is heavily based on mlir/test/lib/Dialect/Test/TestOps.td/cpp/h

// TODO make this into an mlir attribute so it can be included on every instruction Op
/// to support saving register info on multi result instructions (up to 2 results for now)
struct ResultRegisters{
	/// first reg is saved in lower 16 bits, second in upper 16 bits
	uint32_t content;

    ResultRegisters(): content(0){}

    ResultRegisters(FeReg reg1, FeReg reg2): content(transform(reg1) | (transform(reg2) << 16)){
        assert(static_cast<uint16_t>(reg1) == reg1 && static_cast<uint16_t>(reg2) == reg2 && "registers contain too much information to be saved in this class");
    }

	inline FeReg getReg1() const {
		return transform(content & 0xFFFFu);
	}

	inline FeReg getReg2() const {
		return transform((content >> 16) & 0xFFFFu);
	}

    inline void setReg1(FeReg reg){
        content = (content & 0xFFFF0000u) | transform(reg);
    }

    inline void setReg2(FeReg reg){
        content = (content & 0x0000FFFFu) | (transform(reg) << 16);
    }

    inline void setRegs(FeReg reg1, FeReg reg2){
        content = (transform(reg1) | (transform(reg2) << 16));
    }

    // these 3 are to use this as an mlir property:
    inline mlir::Attribute asAttribute(mlir::MLIRContext* ctx) const{
        mlir::Builder builder(ctx);
        return builder.getI32IntegerAttr(content);
    }
    inline static mlir::LogicalResult setFromAttr(ResultRegisters& prop, mlir::Attribute attr, mlir::InFlightDiagnostic* diag){
        mlir::IntegerAttr intAttr = attr.dyn_cast<mlir::IntegerAttr>();
        if(!intAttr){
            if(diag)
                *diag << "expected integer attribute for ResultRegisters";
            return mlir::failure();
        }
        prop.content = intAttr.getInt();
        return mlir::success();
    }
    inline llvm::hash_code hash() const {
        return llvm::hash_value(content);
    }

private:
    /// if the internal representation of FeReg changes, this needs to be updated
	inline static uint16_t transform(FeReg reg){
		return reg;
	}

	inline static FeReg transform(uint16_t reg){
		return static_cast<FeReg>(reg);
	}
};

// === traits ===

// the way to define these seems so hacky...
namespace mlir::OpTrait{

/// lots of x86 instructions have the first operand as the destination -> this trait signals that
template<unsigned N>
class Operand1IsDestN{
public:
    template <typename ConcreteType>
    class Impl:public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
    };
};

/// TODO maybe this is entirely stupid
template<unsigned N, FeReg reg>
class OperandNIsConstrainedToReg{
public:
    template <typename ConcreteType>
    class Impl:public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
    };
};

} // namespace mlir::OpTrait

#define GET_TYPEDEF_CLASSES
#include "AMD64/AMD64OpsTypes.h.inc"

#endif // AMD64_TYPES_H

