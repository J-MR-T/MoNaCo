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
    // TODO for now 
    //StaticInterfaceMethod<"TODO", "std::pair<amd64::OperandRegisterConstraint", "getOperandRegisterConstraints">,
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

#endif
