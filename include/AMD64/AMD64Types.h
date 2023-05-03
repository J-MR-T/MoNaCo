#ifndef AMD64_TYPES_H
#define AMD64_TYPES_H

#include "mlir/IR/BuiltinTypes.h"

#include "fadec-enc.h"

#include "mlir/IR/OpDefinition.h"

using FeMnem = uint64_t;

// TODO make this into an mlir attribute so it can be included on every instruction Op
/// to support saving register info on multi result instructions (up to 2 results for now)
struct ResultRegisters{
	/// first reg is saved in lower 16 bits, second in upper 16 bits
	uint32_t regs;

    ResultRegisters(FeReg reg1, FeReg reg2): regs(transform(reg1) | (transform(reg2) << 16)){
        assert(static_cast<uint16_t>(reg1) == reg1 && static_cast<uint16_t>(reg2) == reg2 && "registers contain too much information to be saved in this class");
    }

	inline FeReg getReg1() const {
		return transform(regs & 0xFFFFu);
	}

	inline FeReg getReg2() const {
		return transform((regs >> 16) & 0xFFFFu);
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

