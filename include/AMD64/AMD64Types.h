#ifndef AMD64_TYPES_H
#define AMD64_TYPES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "fadec-enc.h"

#include "mlir/IR/OpDefinition.h"

using FeMnem = uint64_t;

namespace amd64{

// TODO i need two seperate things here, right? operand register constraints and result register constraints
// TODO what is the best way to have multiple of these, without taking up tons of space?
// TODO put a static op iface method to get some kind of collection of these two
struct OperandRegisterConstraint{
    uint8_t whichOperand;
    FeReg reg;
};

struct ResultRegisterConstraint{
    uint8_t whichResult;
    FeReg reg;
};

// TODO put the rest of the stuff into the namespace as well
}

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

    inline uint32_t combined() const{
        return reg1 | (reg2 << 16);
    }

    // these 3 are to use this as an mlir property:
    inline mlir::Attribute asAttribute(mlir::MLIRContext* ctx) const{
        mlir::Builder builder(ctx);
        return builder.getI32IntegerAttr(combined());
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
	// TODO maybe optimize this later, but for now this is fine
	int64_t imm; // only has a useful value if the instruction has the HasImm trait

	// these 3 are to use this as an mlir property:
	inline mlir::Attribute asAttribute(mlir::MLIRContext* ctx) const{
        mlir::Builder builder(ctx);

        return builder.getI64ArrayAttr({regs.combined(), imm}); // this is somewhat inefficient, but there is no better way to do this as far as I knwo, and it shouldn't happend during normal compilation
	}
	inline static mlir::LogicalResult setFromAttr(InstructionInfo& prop, mlir::Attribute attr, mlir::InFlightDiagnostic* diag){
        auto arrayAttr = attr.cast<mlir::ArrayAttr>();
        if(!arrayAttr){
            if(diag)
                *diag << "expected array attribute for InstructionInfo";
            return mlir::failure();
        }
        auto intAttrs = arrayAttr.getValue();

        auto immAttr = intAttrs[1].dyn_cast<mlir::IntegerAttr>();
        if(!immAttr){
            if(diag)
                *diag << "expected integer attribute for InstructionInfo immediate";
            return mlir::failure();
        }
        prop.imm = immAttr.getInt();

		return ResultRegisters::setFromAttr(prop.regs, *intAttrs.begin(), diag);
	}
	inline llvm::hash_code hash() const {
		return llvm::hash_combine(regs.hash(), imm);
	}
};

// === traits ===

// the way to define these seems so hacky...
namespace mlir::OpTrait{

/// TODO does this need to be parametrized? Is 1 = 1 enough?
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

template<typename ConcreteType>
class HasImm : public TraitBase<ConcreteType, HasImm> {
};

} // namespace mlir::OpTrait

namespace mlir::TypeTrait{
} // namespace mlir::TypeTrait


// my own interfaces 
#include "AMD64/AMD64OpInterfaces.h.inc"
#include "AMD64/AMD64TypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "AMD64/AMD64OpsTypes.h.inc"

#endif // AMD64_TYPES_H

