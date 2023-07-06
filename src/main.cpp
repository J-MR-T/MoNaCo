#include <chrono>
#include <llvm/Support/CommandLine.h>
#include <string>

#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Analysis/Liveness.h>
#include <regex>

#include "AMD64/AMD64Dialect.h"
#include "AMD64/AMD64Ops.h"
#include "isel.h"
#include "fallback.h"
#include "regallocEncode.h"

int main(int argc, char *argv[]) {
#define MEASURE_TIME_START(point) auto point ## _start = std::chrono::high_resolution_clock::now()

#define MEASURE_TIME_END(point) auto point ## _end = std::chrono::high_resolution_clock::now()

#define MEASURED_TIME_AS_SECONDS(point, iterations) std::chrono::duration_cast<std::chrono::duration<double>>(point ## _end - point ## _start).count()/(static_cast<double>(iterations))


    ArgParse::parse(argc, argv);

    auto& args = ArgParse::args;

    if(args.help()){
        ArgParse::printHelp(argv[0]);
        return EXIT_SUCCESS;
    }

    enum {PRINT_NONE = 0x0, PRINT_INPUT = 0x1, PRINT_ISEL = 0x2, PRINT_ASM = 0x4};
    int printOpts = PRINT_NONE;
    if(args.fallback() && args.noFallback()){
        errx(EXIT_FAILURE, "Cannot specify both --fallback and --no-fallback");
    }else if((*args.input).empty()){
        errx(EXIT_FAILURE, "Input file cannot be empty");
    }else if(args.output() && (*args.output).empty()){
        errx(EXIT_FAILURE, "Output file cannot be empty");
    }else if(args.iterations() && (*args.iterations).empty()){
        errx(EXIT_FAILURE, "Iterations cannot be empty");
    }else if(args.jit() && args.benchmark()){
        errx(EXIT_FAILURE, "Benchmarking the JIT execution time is not yet supported");
    }else if(args.jit() && *args.jit == ""){
        errx(EXIT_FAILURE, "Cannot JIT without arguments");
    }

    if(args.print()){
        auto print_sv = *args.print;
        if(print_sv.contains("input")) printOpts |= PRINT_INPUT;
        if(print_sv.contains("isel"))  printOpts |= PRINT_ISEL;
        if(print_sv.contains("asm"))   printOpts |= PRINT_ASM;

        if(printOpts == PRINT_NONE)
            errx(EXIT_FAILURE, "Invalid argument for --print: %s", std::string{print_sv}.c_str());
    }
    
    if(args.debug()){
        llvm::DebugFlag = true;
        printOpts |= PRINT_INPUT | PRINT_ISEL | PRINT_ASM;
    }

    mlir::MLIRContext ctx;
    ctx.loadAllAvailableDialects();
    ctx.loadDialect<amd64::AMD64Dialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::cf::ControlFlowDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();
    ctx.loadDialect<mlir::LLVM::LLVMDialect>();
    // TODO not sure about this
    ctx.allowUnregisteredDialects();

    auto inputFile = ArgParse::args.input() ? *ArgParse::args.input : "-";

    auto owningModRef = readMLIRMod(inputFile, ctx);

    if(printOpts & PRINT_INPUT){
        llvm::outs() << termcolor::make(termcolor::red, "Input:\n");
        owningModRef->print(llvm::outs());
    }

    auto maybeWriteToFile = [](auto* buf, size_t size){
        if(args.output()){
            std::error_code ec;
            llvm::raw_fd_ostream out(*args.output, ec);
            if(ec){
                err(EXIT_FAILURE, "Could not open output file %s", std::string{*args.output}.c_str());
            }
            out.write(reinterpret_cast<const char*>(buf), size);
        }
    };

    int prot = PROT_READ|PROT_WRITE;
    if(args.jit())
        prot |= PROT_EXEC;


    if(args.benchmark()){
        // TODO what happens if this throws an exception? Is that fine?
        unsigned iterations = 1;
        if(args.iterations()){
            int iterationsI = 0;
            auto iters_sv = *args.iterations;
            auto [ptr, ec] = std::from_chars(iters_sv.data(), iters_sv.data() + iters_sv.size(), iterationsI);
            if(ec != std::errc() || ptr != iters_sv.data() + iters_sv.size()){
                errx(EXIT_FAILURE, "Could not parse iterations argument");
            }else if(iterationsI <= 0){
                errx(EXIT_FAILURE, "Iterations must be a positive integer");
            }
            iterations = static_cast<unsigned>(iterationsI);
        }

        std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modClones(2*iterations);
        for(unsigned i = 0; i < modClones.size(); i++){
            modClones[i] = mlir::OwningOpRef<mlir::ModuleOp>(owningModRef->clone());
        }

        if(args.fallback()){
            llvm::TargetOptions opt;
            opt.EnableFastISel = true;

            // TODO:
            // const char* myopt = "-pass-remarks-missed=sdagisel";

            MEASURE_TIME_START(totalLLVM);
            for(auto i = 0u; i < iterations; i++){
                auto obj = llvm::SmallVector<char, 0>();
                fallbackToLLVMCompilation(*modClones[i], obj, opt);
            }

            MEASURE_TIME_END(totalLLVM);

            llvm::outs() << "LLVM Fallback compilation took " << MEASURED_TIME_AS_SECONDS(totalLLVM, iterations) << " seconds on average over " << iterations << " iterations\n";
        }else{
            // allocate 2 GiB (small code model)
            auto [start, end] = mmapSpace(2ll*1024ll*1024ll*1024ll, prot);

            if(!start || !end)
                err(EXIT_FAILURE, "mmap");

            MEASURE_TIME_START(totalMLIR);

            for(unsigned i = 0; i < iterations; i++){
                // first pass: ISel
                // TODO does this slow it down?
                amd64::GlobalsInfo globals;
                maximalIsel(*modClones[i], globals);

                // second pass: RegAlloc + encoding
                // - will need a third pass in between to do liveness analysis later
                regallocEncodeRepeated(start, end, *modClones[i], std::move(globals));
            }

            MEASURE_TIME_END(totalMLIR);

            MEASURE_TIME_START(iselMLIR);

            // TODO does this slow it down?
            std::vector<amd64::GlobalsInfo> globalsClones(iterations);
            for(unsigned i = iterations; i < 2*iterations; i++){
                maximalIsel(*modClones[i], globalsClones[i-iterations]);
            }
            MEASURE_TIME_END(iselMLIR);

            // TODO this is experimental
            MEASURE_TIME_START(liveness);
            for(unsigned i = iterations; i < 2*iterations; i++){
                mlir::Liveness liveness(*modClones[i]);
            }
            MEASURE_TIME_END(liveness);

            MEASURE_TIME_START(regallocMLIR);
            for(unsigned i = iterations; i < 2*iterations; i++){
                regallocEncodeRepeated(start, end, *modClones[i], std::move(globalsClones[i-iterations]));
            }
            MEASURE_TIME_END(regallocMLIR);

            llvm::outs() << "ISel + RegAlloc + encoding took " << MEASURED_TIME_AS_SECONDS(totalMLIR, iterations) << " seconds on average over "     << iterations                                         << " iterations\n";
            llvm::outs() << "ISel repeated "                   << iterations                                      << " times without RegAlloc took " << MEASURED_TIME_AS_SECONDS(iselMLIR,     iterations) << " seconds on average\n";
            llvm::outs() << "RegAlloc repeated "               << iterations                                      << " times without ISel took "     << MEASURED_TIME_AS_SECONDS(regallocMLIR, iterations) << " seconds on average\n";
            llvm::outs() << "Combining these two times gives " << MEASURED_TIME_AS_SECONDS(iselMLIR, iterations) + MEASURED_TIME_AS_SECONDS(regallocMLIR, iterations) << " seconds on average, be aware that the last three measurements do not represent realistic use-case of these functions!\n";
            llvm::outs() << "Experimental: Liveness analysis took " << MEASURED_TIME_AS_SECONDS(liveness, iterations) << " seconds on average over " << iterations << " iterations\n";
        }
    }else if(args.fallback()){
        auto obj = llvm::SmallVector<char, 0>();
        if(fallbackToLLVMCompilation(*owningModRef, obj))
            return EXIT_FAILURE;

        maybeWriteToFile(obj.data(), obj.size());
    }else{ // "normal" case
        // allocate 2 GiB (small code model)
        auto [start, end] = mmapSpace(2ll*1024ll*1024ll*1024ll, prot);

        if(!start || !end)
            err(EXIT_FAILURE, "mmap");

        // first pass: ISel
        amd64::GlobalsInfo globals;
        auto iselFailed = maximalIsel(*owningModRef, globals);

        if(printOpts & PRINT_ISEL){
            llvm::outs() << termcolor::make(termcolor::red, "After ISel:\n");
            owningModRef->print(llvm::outs());
        }

        if(iselFailed)
            errx(EXIT_FAILURE, "ISel failed");

        // second pass: RegAlloc + encoding
        // - will need a third pass in between to do liveness analysis later
        auto* execStart = regallocEncode(start, end, *owningModRef, std::move(globals), printOpts & PRINT_ASM, args.jit(), "main");

        if(args.jit()){
            if(execStart == nullptr || execStart == NULL)
                errx(EXIT_FAILURE, "Could not find main function");

            // TODO this is totally ugly, but thats what Cpp gets for not including a proper string split function
            // split jit argv with spaces
            const auto jitArgvStr = std::string{*args.jit};
            std::regex regexz("[ ]+");
            std::vector<std::string> split(std::sregex_token_iterator(jitArgvStr.begin(), jitArgvStr.end(), regexz, -1), std::sregex_token_iterator());
            // TODO vector
            const char** jitArgv = new const char*[split.size() + 1];
            for(unsigned i = 0; i < split.size(); i++){
                jitArgv[i] = split[i].c_str();
            }
            jitArgv[split.size()] = nullptr;

            // TODO technically have to be non-const
            using main_t = int(*)(int, const char**);
            auto main = reinterpret_cast<main_t>(execStart);

            // TODO the miniStruct test program currently overwrites the stack slot of rbp, resulting in 0xa being popped to rbp instead of the original value, that's why it segfaults after returning
            auto ret =  main(split.size(), jitArgv);
            delete[] jitArgv;
            return ret;
        }

        maybeWriteToFile(start, end - start);
    }

    return EXIT_SUCCESS;
}
