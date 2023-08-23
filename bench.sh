monaco=${1:-moreBuilds/build_rel/bin/monaco}
flags=${2:--fcodegen-dce -fomit-one-use-value-spills -b --iterations=1}

function run (){
    # echo what will be run, then run it. Will be called with run command
    echo Running \""$@"\" >&2
    "$@"
}

function monaco (){
    run "$monaco" $flags "$@"
}

ct=measurements/comptime

monaco    --jit main samples/dhrystone/O0Patched.mlir > $ct/dhrystoneO0Rel.txt
monaco    --jit main samples/dhrystone/O2Patched.mlir > $ct/dhrystoneO2Rel.txt
monaco    --jit main samples/mcf/O0Patched.mlir       > $ct/mcfO0Rel.txt
monaco    --jit main samples/mcf/O2Patched.mlir       > $ct/mcfO2Rel.txt
monaco -F --jit main samples/dhrystone/O0Patched.mlir > $ct/dhrystoneO0LLVM.txt
monaco -F --jit main samples/dhrystone/O2Patched.mlir > $ct/dhrystoneO2LLVM.txt
monaco -F --jit main samples/mcf/O0Patched.mlir       > $ct/mcfO0LLVM.txt
monaco -F --jit main samples/mcf/O2Patched.mlir       > $ct/mcfO2LLVM.txt
