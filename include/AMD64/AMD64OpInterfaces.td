#ifndef AMD64_OP_INTERFACES
#define AMD64_OP_INTERFACES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"


// TODO straighten this out when its done
def InstructionOpInterface : OpInterface<"InstructionOpInterface">{
  let description = [{
    This is an interface for all AMD64 instructions. TODO
  }];

  let methods = [
    InterfaceMethod<"TODO", "InstructionInfo&", "instructionInfo", /*args=*/ (ins ), /*methodBody=*/ [{
      return $_op.getProperties().instructionInfoImpl;
    }]>,
    StaticInterfaceMethod<"TODO", "FeMnem", "getFeMnemonic">,
    StaticInterfaceMethod<"TODO", "amd64::OperandRegisterConstraints", "getOperandRegisterConstraints">, // TODO document that Operand0IsDestN has priority over operand register constraints, i.e. if that is set, then any operand register constraint for operand 0 is ignored
    StaticInterfaceMethod<"TODO", "amd64::ResultRegisterConstraints",  "getResultRegisterConstraints">,
  ];

  let cppNamespace = "::amd64";
}

// TODO straighten this out when its done
def EncodeOpInterface : OpInterface<"EncodeOpInterface">{
  let description = [{
    This is an interface for AMD64 Memory Ops. They can specify how they are encoded using this interface.
  }];

  let methods = [
    InterfaceMethod<"TODO", "FeOp", "encode">
  ];

  let cppNamespace = "::amd64";
}

def ConditionalJumpInterface : OpInterface<"ConditionalJumpInterface">{
  let description = [{
    This is an interface for AMD64 Jump Ops. They can specify how they are inverted using this interface, and their destinations and destination arguments can be queried.
  }];

  let methods = [
    InterfaceMethod<"TODO", "::amd64::ConditionalJumpInterface", "invert",                       (ins "::mlir::OpBuilder&":$builder)>,
    InterfaceMethod<"TODO", "::mlir::Operation::operand_range",  "getTrueDestOperands">,
    InterfaceMethod<"TODO", "::mlir::Operation::operand_range",  "getFalseDestOperands">,
    InterfaceMethod<"TODO", "::mlir::MutableOperandRange",       "getTrueDestOperandsMutable">,
    InterfaceMethod<"TODO", "::mlir::MutableOperandRange",       "getFalseDestOperandsMutable">,
    InterfaceMethod<"TODO", "::mlir::Block*",                    "getTrueDest">,
    InterfaceMethod<"TODO", "::mlir::Block*",                    "getFalseDest">,
  ];

  let cppNamespace = "::amd64";
}

#endif
