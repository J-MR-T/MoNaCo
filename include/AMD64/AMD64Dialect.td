#ifndef AMD64_DIALECT
#define AMD64_DIALECT

// important: This file is not used directly, only by being included in the AMD64Ops file!
// that explains why "AMD64OpsDialect.[h/cpp].inc" needs to be used elsewhere, because this file content only *really* exists in BOps.

// include stuff for tablegen
include "mlir/IR/OpBase.td"

def AMD64_Dialect : Dialect {
  let name = "amd64";

  let summary = "An Instruction Selection dialect for the x86-64/AMD64 architecture";

  let description = [{
    TODO
  }];

  let cppNamespace = "::amd64";

  let useDefaultTypePrinterParser = 1;

  // TODO performance test this later
  let usePropertiesForAttributes = 1;

  // TODO get this to work
  //let isExtensible = 1;
}

class AMD64_Op<string mnemonic, list<Trait> traits = []> :
        Op<AMD64_Dialect, mnemonic, traits>;

#endif

