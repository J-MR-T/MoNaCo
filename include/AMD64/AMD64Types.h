#ifndef AMD64_TYPES_H
#define AMD64_TYPES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "fadec-enc.h"

#include "mlir/IR/OpDefinition.h"

using FeMnem = uint64_t;

// this is heavily based on mlir/test/lib/Dialect/Test/TestOps.td/cpp/h

/// to support saving register info on multi result instructions (up to 2 results for now)
struct ResultRegisters{
	uint16_t reg1;
	uint16_t reg2;

    ResultRegisters(): reg1(0), reg2(0){}

    ResultRegisters(FeReg reg1, FeReg reg2): reg1(transform(reg1)), reg2(transform(reg2)){
        assert(static_cast<uint16_t>(reg1) == reg1 && static_cast<uint16_t>(reg2) == reg2 && "registers contain too much information to be saved in this class");
    }

	/// if this class is implicitly cast to a FeReg, return the first register, as this is the most common case
	inline operator FeReg() const{
		return getReg1();
	}

	inline FeReg getReg1() const{
		return transform(reg1);
	}

	inline FeReg getReg2() const{
		return transform(reg2);
	}

	inline void setReg1(FeReg reg){
		reg1 = transform(reg);
	}

	inline void setReg2(FeReg reg){
		reg2 = transform(reg);
	}

    // these 3 are to use this as an mlir property:
    inline mlir::Attribute asAttribute(mlir::MLIRContext* ctx) const{
        mlir::Builder builder(ctx);
        return builder.getI32IntegerAttr(reg1 | (reg2 << 16));
    }
    inline static mlir::LogicalResult setFromAttr(ResultRegisters& prop, mlir::Attribute attr, mlir::InFlightDiagnostic* diag){
        mlir::IntegerAttr intAttr = attr.dyn_cast<mlir::IntegerAttr>();
        if(!intAttr){
            if(diag)
                *diag << "expected integer attribute for ResultRegisters";
            return mlir::failure();
        }
        prop.reg1 = intAttr.getInt() & 0xFFFF;
        prop.reg2 = intAttr.getInt() & 0xFFFF0000;
        return mlir::success();
    }
    inline llvm::hash_code hash() const {
        return llvm::hash_value({reg1, reg2});
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

/// overarching property class for all instructions
struct InstructionInfo{
	ResultRegisters regs;

	// these 3 are to use this as an mlir property:
	inline mlir::Attribute asAttribute(mlir::MLIRContext* ctx) const{
		mlir::Builder builder(ctx);
		return regs.asAttribute(ctx);
	}
	inline static mlir::LogicalResult setFromAttr(InstructionInfo& prop, mlir::Attribute attr, mlir::InFlightDiagnostic* diag){
		return ResultRegisters::setFromAttr(prop.regs, attr, diag);
	}
	inline llvm::hash_code hash() const {
		return regs.hash();
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
/// TODO just do static method on op, == 0 to test if constraints exist
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

