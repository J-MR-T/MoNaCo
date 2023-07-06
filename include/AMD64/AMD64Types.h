#ifndef AMD64_TYPES_H
#define AMD64_TYPES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "fadec-enc.h"

#include "mlir/IR/OpDefinition.h"
#include <llvm/ADT/StringMap.h>

using FeMnem = uint64_t;

namespace amd64{

namespace conditional{
enum predicate{
    Z = 0, // E
    NZ,    // NE
    L,
    GE,
    LE,
    G,
    C,     // B
    NC,    // AE
    BE,
    A,
    NONE = 0xFF
};

inline predicate invert(predicate pred){
    return static_cast<predicate>(pred ^ 0x1);
}
}

// TODO template this and only have one of the two

// chmpxchg16b is the worst for this
struct OperandRegisterConstraint{
#define NO_CONSTRAINT -1
    int8_t which = NO_CONSTRAINT;
    FeReg reg;

    /// TODO I hope this doesn't make it less efficient, because its not POD anymore
    bool constrainsReg() const{
        return which != NO_CONSTRAINT;
    }
};

struct ResultRegisterConstraint{
    int8_t which = NO_CONSTRAINT;
    FeReg reg;

    bool constrainsReg() const{
        return which != NO_CONSTRAINT;
    }
};

// TODO think if 'which' can be eliminated somehow, this is quite ugly
// a light wrapper around a pair
template<typename T>
struct Constraints{
    T first, second;

    constexpr Constraints(T first, T second): first(first), second(second){}
    constexpr Constraints(std::pair<T, T> pair): first(pair.first), second(pair.second){}

    operator std::pair<T, T>(){
        return std::make_pair(first, second);
    }

    // TODO also a bit ugly
    // TODO I think this is also not right, because the operand index i, with which it is called, doesn't always equate to the ith *register* operand (memory op in the middle/...)
    T operator [](int i){
        if(i == 0)
            return first.which == 0 ? first : T();
        else if(i == 1)
            return first.which == 1 ? first : (second.which == 1 ? second : T());
        else
            llvm_unreachable("invalid index");
    }
};
using OperandRegisterConstraints = Constraints<OperandRegisterConstraint>;
using ResultRegisterConstraints  = Constraints<ResultRegisterConstraint>;

// TODO maybe actually don't put any of this into the namespace?
struct GlobalSymbolInfo{
    llvm::SmallVector<uint8_t, 8> bytes;
    unsigned alignment;
    // TODO linkage/visibility

    intptr_t addrInDataSection;
};

using GlobalsInfo = llvm::StringMap<GlobalSymbolInfo>;

// TODO put the rest of the stuff into the namespace as well
}

// this is heavily based on mlir/test/lib/Dialect/Test/TestOps.td/cpp/h

/// to support saving register info on multi result instructions (up to 2 results for now)
struct ResultRegisters{
    // TODO these take up 32 bits each, even tho the information fits within 16 bits by simply shifting, and into 8 bits from an information theory perspective. But this makes accessing the registers much easier, so optimize this later.
    FeReg reg1;
    FeReg reg2;

    ResultRegisters(): reg1((FeReg)FE_NOREG), reg2((FeReg)FE_NOREG){}

    ResultRegisters(FeReg reg1, FeReg reg2): reg1(reg1), reg2(reg2){
        // assert this, because the combine methods would otherwise fail
        assert(static_cast<uint16_t>(reg1) == reg1 && static_cast<uint16_t>(reg2) == reg2 && "registers contain too much information to be saved in this class");
    }

	/// if this class is implicitly cast to a FeReg, return the first register, as this is the most common case
    inline bool reg1Empty() const{
        return reg1 == FE_NOREG;
    }

    inline bool reg2Empty() const{
        return reg2 == FE_NOREG;
    }

    inline uint32_t combined() const{
        return reg1 | (reg2 << 16);
    }

    inline void setFromCombined(uint32_t combined){
        reg1 = static_cast<FeReg>(combined & 0x0000FFFF);
        reg2 = static_cast<FeReg>(combined & 0xFFFF0000);
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
        prop.setFromCombined(intAttr.getInt());
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
	int64_t imm; // only has a useful value if the instruction has the Special<HasImm> trait

    /// returns if true if not all of the registers were constrained, i.e. if there is still something to do on this instruction for the register allocator
    inline bool setRegsFromConstraints(const amd64::ResultRegisterConstraints& constraints){
        auto allConstrained = true;
        for(auto constraint : {constraints.first, constraints.second}){
            allConstrained &= constraint.constrainsReg();

            if(constraint.constrainsReg()){
                if(constraint.which == 0){
                    assert((regs.reg1Empty() || regs.reg1 == constraint.reg) && "register constraint mismatch");

                    regs.reg1 = constraint.reg;
                }else if(constraint.which == 1){
                    assert((regs.reg2Empty() || regs.reg2 == constraint.reg) && "register constraint mismatch");

                    regs.reg2 = constraint.reg;
                }else{
                    assert(false && "invalid result number");
                }
            }
        }

        return !allConstrained;
    }

	// these 3 are to use this as an mlir property:
	inline mlir::Attribute asAttribute(mlir::MLIRContext* ctx) const{
        mlir::Builder builder(ctx);

        return builder.getI64ArrayAttr({regs.combined(), imm}); // this is somewhat inefficient, but there is no better way to do this as far as I know, and it shouldn't happend during normal compilation
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
namespace amd64{
enum class Special{
    IDIV,         // sign-extending of the upper part
    DIV,          // zero-extending of the upper part
    HasImm
};
}

// the way to define these seems so hacky...
namespace mlir::OpTrait{

// TODO does this need to be parametrized? Is 1 = 1 enough?
/// lots of x86 instructions have the first operand as the destination -> this trait signals that
template<unsigned N>
class Operand0IsDestN{
public:
    template<typename ConcreteType>
    class Impl:public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
    };
};

template<amd64::Special specialKind>
class SpecialCase{
public:
    template<typename ConcreteType>
    class Impl:public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
    };
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

