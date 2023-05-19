#ifndef AMD64_OPS_H
#define AMD64_OPS_H

// TODO depending on if this is changed, remove this include here
#include "fadec-enc.h"

#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/Dialect.h"
// predefined interfaces
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"

// note: this fixed "OpAsmOpInterface not found" errors... (because OpImplementation.h defines AsmParser, the implementation of which includes the OpAsmOpInterface)
#include "mlir/IR/OpImplementation.h"

#include "mlir/IR/Builders.h"

#include "mlir/IR/BuiltinTypes.h"
//#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
//#include "mlir/IR/BuiltinTypeInterfaces.h"

#include "AMD64/AMD64Types.h"

namespace amd64{
    /// Get the register of an OpResult of any instruction.
    /// The result number is used to figure out which register the result belongs to.
    inline FeReg registerOf(mlir::OpResult result){
        assert(mlir::isa<amd64::InstructionOpInterface>(result.getDefiningOp()) && "expected an instruction op");

        auto& regs = mlir::dyn_cast<amd64::InstructionOpInterface>(result.getDefiningOp()).instructionInfo().regs;
        if(result.getResultNumber() == 0){
            return regs.getReg1();
        }else{
            assert(result.getResultNumber() == 1 && "Operands seems to have more than 2 results");
            return regs.getReg2();
        }
    }
    /// Get the register of an Operation of an instruction with exactly one result.
    inline FeReg registerOf(mlir::Operation* op){
        assert(mlir::isa<amd64::InstructionOpInterface>(op) && "expected an instruction op");

        auto& regs = mlir::dyn_cast<amd64::InstructionOpInterface>(op).instructionInfo().regs;
        assert(op->getNumResults() == 1 && "Operands seems to have more than 1 result");
        return regs.getReg1();
    }
    // TODO handle block args at some point
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

