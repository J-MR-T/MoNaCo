import lit.formats
import os

# TODO find an option to automatically set -j1 per default for lit

config.name = "MoNaCo: MLIR to Native Compiler"
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mlir']

config.excludes = ['samples']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, 'tests')

config.substitutions.append(("%monaco", os.path.join(config.test_source_root, "monaco")))
config.substitutions.append(("%FileCheckAsm", r"%monaco -p asm %s | FileCheck"))

config.recursiveExpansionLimit = 3

