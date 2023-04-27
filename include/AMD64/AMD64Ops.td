#ifndef AMD64OPS
#define AMD64OPS

include "AMD64/AMD64Dialect.td"

include "mlir/IR/AttrTypeBase.td"

include "mlir/IR/OpBase.td"

include "mlir/IR/BuiltinTypes.td"
// don't include this: "mlir/IR/BuiltinOps.td". Breaks everything for some reason
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinDialect.td"

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// === types ===

// base type
class AMD64_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AMD64_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif
