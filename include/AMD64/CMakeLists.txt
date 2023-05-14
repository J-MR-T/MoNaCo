add_mlir_dialect(AMD64Ops amd64)
add_mlir_doc(AMD64Dialect AMD64Dialect AMD64/ -gen-dialect-doc)
add_mlir_doc(AMD64Ops AMD64Ops AMD64/ -gen-op-doc)

# one would expect 'add_mlir_interface' to add any type of interface, but it covers op interfaces ...
add_mlir_interface(AMD64OpInterfaces)

# ... so do it manually for the type interfaces
set(LLVM_TARGET_DEFINITIONS AMD64TypeInterfaces.td)
mlir_tablegen(AMD64TypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(AMD64TypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRAMD64TypeInterfacesIncGen)


# enums, attrs
set(LLVM_TARGET_DEFINITIONS AMD64Ops.td)
mlir_tablegen(AMD64OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(AMD64OpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(AMD64OpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(AMD64OpsAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRAMD64OpsEnumsIncGen)

# Lowerings
set(LLVM_TARGET_DEFINITIONS Lowerings.td)
mlir_tablegen(Lowerings.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRAMD64LoweringsIncGen)
