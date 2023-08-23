monaco=${1:-moreBuilds/build_rel/bin/monaco}
flags=${2:--fcodegen-dce -fomit-one-use-value-spills -bcomptime --iterations=1000}
dhrystoneRuns=${3:-1000000000}

function run (){
    # echo what will be run, then run it. Will be called with run command
    date >&2
    echo Running \""$@"\" >&2
    "$@"
    date >&2
}

function monaco (){
    run "$monaco" $flags "$@"
}

ct=measurements/comptime
mkdir -p $ct

monaco    --jit main samples/dhrystone/O0Patched.mlir > $ct/dhrystoneO0Rel.txt
monaco    --jit main samples/dhrystone/O2Patched.mlir > $ct/dhrystoneO2Rel.txt
monaco    --jit main samples/mcf/O0Patched.mlir       > $ct/mcfO0Rel.txt
monaco    --jit main samples/mcf/O2Patched.mlir       > $ct/mcfO2Rel.txt
monaco -F --jit main samples/dhrystone/O0Patched.mlir > $ct/dhrystoneO0LLVM.txt
monaco -F --jit main samples/dhrystone/O2Patched.mlir > $ct/dhrystoneO2LLVM.txt
monaco -F --jit main samples/mcf/O0Patched.mlir       > $ct/mcfO0LLVM.txt
monaco -F --jit main samples/mcf/O2Patched.mlir       > $ct/mcfO2LLVM.txt

flags=${2:--fcodegen-dce -fomit-one-use-value-spills -bruntime --iterations=10}

rt=measurements/runtime
mkdir -p $rt

monaco    --jit main samples/dhrystone/O0Patched.mlir                                              > $rt/dhrystoneO0Rel.txt < <(echo $dhrystoneRuns)
monaco    --jit main samples/dhrystone/O2Patched.mlir                                              > $rt/dhrystoneO2Rel.txt < <(echo $dhrystoneRuns)
monaco    --jit 'main samples/mcf/run_base_test_mytest-m64.0000/inp.in' samples/mcf/O0Patched.mlir > $rt/mcfO0Rel.txt
monaco    --jit 'main samples/mcf/run_base_test_mytest-m64.0000/inp.in' samples/mcf/O2Patched.mlir > $rt/mcfO2Rel.txt
monaco -F --jit main samples/dhrystone/O0Patched.mlir                                              > $rt/dhrystoneO0LLVM.txt < <(echo $dhrystoneRuns)
monaco -F --jit main samples/dhrystone/O2Patched.mlir                                              > $rt/dhrystoneO2LLVM.txt < <(echo $dhrystoneRuns)
monaco -F --jit 'main samples/mcf/run_base_test_mytest-m64.0000/inp.in' samples/mcf/O0Patched.mlir > $rt/mcfO0LLVM.txt
monaco -F --jit 'main samples/mcf/run_base_test_mytest-m64.0000/inp.in' samples/mcf/O2Patched.mlir > $rt/mcfO2LLVM.txt
