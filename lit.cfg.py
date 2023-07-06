import lit.formats
import os

# TODO find an option to automatically set -j1 per default for lit

config.name = "MoNaCo: MLIR to Native Compiler"
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mlir', '.c']

config.excludes = ['samples']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, 'tests')

config.substitutions.append(("%monaco", os.path.join(config.test_source_root, "monaco")))
config.substitutions.append(("%FileCheckAsm", r"%monaco -p asm %s | FileCheck"))
# doesn't work anymore with the mmapping stuff, but we don't need it, just use the jit execution engine
# config.substitutions.append(("%FileCheckExecReturnStatus", r"""outfileName=$(basename -z %basename_t .mlir); %monaco %s $outfileName && objcopy --input-target=binary --output-target=elf64-x86-64 --rename-section '.data=.text' $outfileName %t_2 && gcc -c -x c <(echo "int main(int argc, char** argv){ return _binary_"$outfileName"_start(argc, argv); } ") -o %t_3 && gcc %t_2 %t_3 -o %t_4 && %t_4; echo $? | FileCheck"""))
config.substitutions.append(("%FileCheckExecReturnStatus", r"""%monaco --jit "main" %s; echo $? | FileCheck"""))
config.substitutions.append(("%FileCheckExecOutput", r"""%monaco --jit "main" %s | FileCheck"""))
config.substitutions.append(("%RunC", r"""clang -S -emit-llvm %s -o - 2>/dev/null | mlir-translate --import-llvm 2>/dev/null | %monaco /dev/stdin --jit "argv0" """))

config.recursiveExpansionLimit = 3

