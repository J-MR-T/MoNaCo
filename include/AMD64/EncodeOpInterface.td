#ifndef AMD64_ENCODE_OP_INTERFACE
#define AMD64_ENCODE_OP_INTERFACE

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"


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
