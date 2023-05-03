#ifndef AMD64_INSTRUCTION_OP_INTERFACE
#define AMD64_INSTRUCTION_OP_INTERFACE

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"


// TODO straighten this out when its done
def InstructionOpInterface : OpInterface<"InstructionOpInterface">{
  let description = [{
    This is an interface for all AMD64 instructions. TODO
  }];

  let methods = [
    // TODO registers as property maybe as a function ResultRegisters getRegs() { return getProperties().regs; } ? should be possible for this to be found on every Op, i think?
    InterfaceMethod<"TODO", "ResultRegisters&", "getRegs">,
    StaticInterfaceMethod<"TODO", "FeMnem", "getFeMnemonic">,
  ];

  let cppNamespace = "::amd64";
}

#endif
