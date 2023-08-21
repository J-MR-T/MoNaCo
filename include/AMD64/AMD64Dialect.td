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
  // the problem is this issue: https://github.com/llvm/llvm-project/issues/63072
  // its now solved, I haven't upgraded to the newest MLIR yet, but experimentally tried commit https://github.com/llvm/llvm-project/commit/87a14216cd4058154fe59a4031e6fb962ebf0462, and it works.
  //let isExtensible = 1;
}

class AMD64_Op<string mnemonic, list<Trait> traits = []> :
        Op<AMD64_Dialect, mnemonic, traits>;

#endif

