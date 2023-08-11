import lit.formats
import os

config.name = "MoNaCo: MLIR to Native Compiler"
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mlir', '.c']

config.excludes = ['samples']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, 'tests')

monacoExe = os.getenv("MONACO_EXE")
if monacoExe is None:
    monacoExe = os.path.join(config.test_source_root, "monaco")

monacoExe = os.path.abspath(monacoExe)

config.substitutions.append(("%monaco", monacoExe))
config.substitutions.append(("%FileCheckAsm", r"%monaco -p asm %s | FileCheck"))
# doesn't work anymore with the mmapping stuff, but we don't need it, just use the jit execution engine
# config.substitutions.append(("%FileCheckExecReturnStatus", r"""outfileName=$(basename -z %basename_t .mlir); %monaco %s $outfileName && objcopy --input-target=binary --output-target=elf64-x86-64 --rename-section '.data=.text' $outfileName %t_2 && gcc -c -x c <(echo "int main(int argc, char** argv){ return _binary_"$outfileName"_start(argc, argv); } ") -o %t_3 && gcc %t_2 %t_3 -o %t_4 && %t_4; echo $? | FileCheck"""))
config.substitutions.append(("%FileCheckExecReturnStatus", r"""%monaco --jit "main" %s; echo $? | FileCheck"""))
# would be nice to be able to do this on all optimization levels
config.substitutions.append(("%RunC", r"""clang -S -emit-llvm %s -o - 2>/dev/null | mlir-translate --import-llvm 2>/dev/null | %monaco /dev/stdin --jit """))

config.recursiveExpansionLimit = 3

