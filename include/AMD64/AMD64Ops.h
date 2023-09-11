#ifndef AMD64_OPS_H
#define AMD64_OPS_H

// TODO depending on if this is changed, remove this include here
#include "fadec-enc.h"

// note: this fixed "OpAsmOpInterface not found" errors... (because OpImplementation.h defines AsmParser, the implementation of which includes the OpAsmOpInterface)
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/IR/Dialect.h"
// predefined interfaces
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"


#include "mlir/IR/Builders.h"

#include "mlir/IR/BuiltinTypes.h"
//#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
//#include "mlir/IR/BuiltinTypeInterfaces.h"

#include "AMD64/AMD64Types.h"

namespace amd64{

// TODO decide: should this always return the register constraint, or should we leave it up to the regallocator to set those constraints as the actual registers later on?
// TODO also: Is there some way to set return register constraints on ops on creation? That might save a bit of time. Although do that at the end, no premature optimization :).
// TODO needs an overload for block args
// TODO put in a different file

/// Get the register of an OpResult of any instruction.
/// The result number is used to figure out which register the result belongs to.
inline FeReg& registerOf(mlir::OpResult result){
    auto instr = mlir::cast<amd64::InstructionOpInterface>(result.getDefiningOp());

    auto& regs = instr.instructionInfo().regs;
    if(result.getResultNumber() == 0){
        return regs.reg1;
    }else{
        assert(result.getResultNumber() == 1 && "Operands seems to have more than 2 results");
        return regs.reg2;
    }
}

/// Get the register of an Operation of an instruction with exactly one result.
inline FeReg& registerOf(amd64::InstructionOpInterface instr){
    auto& regs = instr.instructionInfo().regs;
    assert(instr->getNumResults() == 1 && "Operands seems to have more than 1 result");
    return regs.reg1;
}

/// Get the register of an Operation of an instruction with exactly one result.
inline FeReg& registerOf(mlir::Operation* op){
    auto instr = mlir::dyn_cast<amd64::InstructionOpInterface>(op);
    assert(instr && "expected an instruction op");
    return registerOf(instr);
}

/// Get the register of a block argument
inline FeReg& registerOf(mlir::BlockArgument& arg, mlir::DenseMap<mlir::BlockArgument, FeReg>& blockArgToReg){
    return blockArgToReg[arg];
}

inline FeReg& registerOf(mlir::Value value, mlir::DenseMap<mlir::BlockArgument, FeReg>* blockArgToReg = nullptr){
    if (auto result = value.dyn_cast<mlir::OpResult>()) {
        return registerOf(result);
    } else if(auto blockArg = value.dyn_cast<mlir::BlockArgument>()){
        assert(blockArgToReg && "blockArgToReg must be provided if value is a block argument");
        return registerOf(blockArg, *blockArgToReg);
    } else { // TODO I think this is impossible, right?
        return registerOf(value.getDefiningOp());
    }
}

inline bool isFPReg(const FeReg& reg){
    // TODO this is neither nice nor API stable, but it works
    return (reg - FE_XMM0) >= 0;
}

inline bool isGPReg(const FeReg& reg){
    // TODO this is neither nice nor API stable, but it works
    return reg >= FE_AX && reg <= FE_BH;
}

} // end namespace amd64

// my own interfaces are included in AMD64Types.h

// TODO is this right at this point? or does it need to come before/after AMD64Ops.h.inc?
// should be right here, so that Ops.h can use the types
// pull all enum type definitions in
#include "AMD64/AMD64OpsEnums.h.inc"

#define GET_OP_CLASSES
#include "AMD64/AMD64Ops.h.inc"

namespace amd64{

// aliases
using JE  = JZ;
using JNE = JNZ;
using JB  = JC;
using JAE = JNC;

using SETE8r  = SETZ8r;
using SETNE8r = SETNZ8r;
using SETB8r  = SETC8r;
using SETAE8r = SETNC8r;

} // end namespace amd64

#endif
