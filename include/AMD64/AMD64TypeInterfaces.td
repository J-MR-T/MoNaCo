#ifndef AMD64_TYPE_INTERFACES
#define AMD64_TYPE_INTERFACES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"


def RegisterTypeInterface : TypeInterface<"RegisterTypeInterface">{
  let description = [{
    TODO
  }];

  let methods = [
    StaticInterfaceMethod<"TODO", "unsigned", "getBitwidth">,
  ];

  let cppNamespace = "::amd64";
}

def GPRegisterTypeInterface : TypeInterface<"GPRegisterTypeInterface", [RegisterTypeInterface]>{
  let description = [{
    TODO
  }];

  let cppNamespace = "::amd64";
}

def FPRegisterTypeInterface : TypeInterface<"FPRegisterTypeInterface", [RegisterTypeInterface]>{
  let description = [{
    TODO
  }];

  let cppNamespace = "::amd64";
}

#endif
