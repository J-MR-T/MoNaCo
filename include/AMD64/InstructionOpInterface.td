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
    StaticInterfaceMethod<"TODO", "uint64_t", "getFeMnemonic">,
  ];

  let cppNamespace = "::amd64";
}

#endif
