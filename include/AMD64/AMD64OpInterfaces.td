#ifndef AMD64_OP_INTERFACES
#define AMD64_OP_INTERFACES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"


def InstructionOpInterface : OpInterface<"InstructionOpInterface">{
  let description = [{
    This is an interface for all AMD64 instructions. Their opcode (as a fadec mnemonic), instruction info, and register constraints can be queried using this interface.
  }];

  let methods = [
    InterfaceMethod<"Return a mutable reference to the InstructionInfo property containing non-static op information, i.e. information that can be different for different instances of the same op.", "InstructionInfo&", "instructionInfo", /*args=*/ (ins ), /*methodBody=*/ [{
      return $_op.getProperties().instructionInfoImpl;
    }]>,
    StaticInterfaceMethod<"Return the fadec mnemonic for this instruction op, revealing the opcode of the instruction. See the fadec encoder library's documentation for detail.", "FeMnem",                            "getFeMnemonic">,
    StaticInterfaceMethod<"Return all register constraints that this instructions' operands have to obey. Note that the Operand0IsDestN trait has priority over any operand register constraint for operand 0, should one exist, it is simply ignored if Operand0IsDestN is present.", "amd64::OperandRegisterConstraints", "getOperandRegisterConstraints">,
    StaticInterfaceMethod<"Return all register constraints that this instructions' return values have to obey. These constraints interact with Operand0IsDestN as one would expect, they fixe operand 0 to have the same register as the specified destination register constraint.", "amd64::ResultRegisterConstraints",  "getResultRegisterConstraints">,
  ];

  let cppNamespace = "::amd64";
}

def EncodeOpInterface : OpInterface<"EncodeOpInterface">{
  let description = [{
    This is an interface for AMD64 Memory Ops. They can specify how they are encoded using this interface.
  }];

  let methods = [
    InterfaceMethod<"Uses the fadec encoder library to return the encoding of the current op (can be an operand to another instruction). How operands and instructions are represented is documented in the fadec encoder API documentation.", "FeOp", "encode", (ins "mlir::DenseMap<mlir::BlockArgument, FeReg>*":$blockArgToReg)>, // this whole passing blockArgToReg around is such terrible design, but I don't knwo how to do it in a better way, until we can save stuff directly on block arguments
    InterfaceMethod<"Try to get the MLIR value for the base value of an SIBD-memory operand. Returns std::nullopt if this is not such an operand or has no base.", "std::optional<mlir::Value>", "getBaseGeneric">,
    InterfaceMethod<"Try to get the MLIR value for the index value of an SIBD-memory operand. Returns std::nullopt if this is not such an operand or has no index.", "std::optional<mlir::Value>", "getIndexGeneric">,
  ];

  let cppNamespace = "::amd64";
}

def PredicateOpInterface : OpInterface<"PredicateOpInterface">{
  let description = [{
    This is an interface for all AMD64 Ops which have prediactes. They can be queried using this interface.
  }];

  let methods = [
    InterfaceMethod<"Return the predicate as a custom `::amd64::conditional::predicate` enum type", "::amd64::conditional::predicate",  "getPredicate">,
  ];

  let cppNamespace = "::amd64";
}

// can maybe simplify this, by also extending MLIRs built-in branch op interface
// no, I believe that would actually make it more complicated. This is easier, because it always has exactly 2 successors.
def ConditionalJumpOpInterface : OpInterface<"ConditionalJumpOpInterface", [InstructionOpInterface, PredicateOpInterface]>{
  let description = [{
    This is an interface for AMD64 Jump Ops. They can specify how they are inverted using this interface, and their destinations and destination arguments can be queried.
  }];

  let methods = [
    StaticInterfaceMethod<"Returns the `FeMnem` mnemonic describing the inverted jump, i.e. the Jump that would execute iff this does not execute", "FeMnem",                           "getInvertedMnem">, // technically mnemonic inversion could be done with just xoring with 1 (the lowest bit is the condition code), but that's not very API-stable...
    InterfaceMethod<"Return the block argument operand range for the destination block targeted by this jump if the condition is true",       "::mlir::Operation::operand_range", "getTrueDestOperands">,
    InterfaceMethod<"Return the block argument operand range for the destination block targeted by this jump if the condition is false",       "::mlir::Operation::operand_range", "getFalseDestOperands">,
    InterfaceMethod<"Return the mutable block argument operand range for the destination block targeted by this jump if the condition is true",       "::mlir::MutableOperandRange",      "getTrueDestOperandsMutable">,
    InterfaceMethod<"Return the mutable block argument operand range for the destination block targeted by this jump if the condition is false",       "::mlir::MutableOperandRange",      "getFalseDestOperandsMutable">,
    InterfaceMethod<"Return the destination block targeted by this jump if the condition is true",       "::mlir::Block*",                   "getTrueDest">,
    InterfaceMethod<"Return the destination block targeted by this jump if the condition is false",       "::mlir::Block*",                   "getFalseDest">,
  ];

  let cppNamespace = "::amd64";
}

#endif
