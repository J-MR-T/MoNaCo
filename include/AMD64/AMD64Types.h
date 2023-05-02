#ifndef AMD64_TYPES_H
#define AMD64_TYPES_H

#include "mlir/IR/BuiltinTypes.h"

#include "fadec-enc.h"

using FeMnem = uint64_t;

// TODO maybe this is premature optimization..., but I think it certainly makes the code more readable
struct RegConstraint{
    // lowest 8 bits represent the first constraint, next 8 bits represent the second constraint, etc.
    uint32_t constraints;

#define NO_CONSTRAINT static_cast<FeReg>(0)
    RegConstraint(FeReg constraint1 = NO_CONSTRAINT, FeReg constraint2 = NO_CONSTRAINT, FeReg constraint3 = NO_CONSTRAINT, FeReg constraint4 = NO_CONSTRAINT) : 
        constraints(transform(constraint1) | transform(constraint2) << 8 | transform(constraint3) << 16 | transform(constraint4) << 24) {}
#undef NO_CONSTRAINT

    template<uint8_t N>
    requires (N < 4)
    bool constrainsOpN(FeReg reg){
        // TODO is this right?
		constexpr auto shiftLeftBy = N*8;
        return (constraints & (0xFFFFFFFF << shiftLeftBy)) == static_cast<uint32_t>(transform(reg) << shiftLeftBy);
    }

private:
    uint8_t transform(FeReg r){
        return r >> 8;
    }

};

#define GET_TYPEDEF_CLASSES
#include "AMD64/AMD64OpsTypes.h.inc"

#endif // AMD64_TYPES_H

