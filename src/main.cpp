#include <chrono>
#include <string>

#include <llvm/IR/LLVMRemarkStreamer.h>

#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Analysis/Liveness.h>

#include "AMD64/AMD64Dialect.h"
#include "AMD64/AMD64Ops.h"
#include "isel.h"
#include "fallback.h"
#include "regallocEncode.h"

int main(int argc, char *argv[]) {
#define MEASURE_TIME_START(point) auto point ## _start = std::chrono::high_resolution_clock::now()

#define MEASURE_TIME_END(point) auto point ## _end = std::chrono::high_resolution_clock::now()

#define MEASURED_TIME_AS_SECONDS(point, iterations) std::chrono::duration_cast<std::chrono::duration<double>>(point ## _end - point ## _start).count()/(static_cast<double>(iterations))

    // TODO would be nice to be able to disable leak sanitizer before JIT execution, to only detect host leaks. Maybe with suppresions? This doesn't work:
    //setenv("ASAN_OPTIONS", "detect_leaks=0", 1);

    auto& features = ArgParse::features;

    ArgParse::parse(argc, argv);

    auto& args = ArgParse::args;

    if(args.help()){
        ArgParse::printHelp(argv[0]);
        return EXIT_SUCCESS;
    }

    enum {PRINT_NONE = 0x0, PRINT_INPUT = 0x1, PRINT_ISEL = 0x2, PRINT_ASM = 0x4};
    int printOpts = PRINT_NONE;
    // TODO rework once ArgParse::features is done
    if((*args.input).empty()){
        errx(EXIT_FAILURE, "Input file cannot be empty");
    }else if(args.output() && (*args.output).empty()){
        errx(EXIT_FAILURE, "Output file cannot be empty");
    }else if(args.iterations() && (*args.iterations).empty()){
        errx(EXIT_FAILURE, "Iterations cannot be empty");
    }else if(args.jit() && *args.jit == ""){
        errx(EXIT_FAILURE, "Cannot JIT without arguments");
    }

    if(args.forceFallback())
        features["force-fallback"] = true;

    // parse features
    if(args.featuresArg()){
        ArgParse::parseFeatures();
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
    // TODO not sure about these 3
    mlir::registerLLVMDialectTranslation(ctx);
    mlir::registerBuiltinDialectTranslation(ctx);
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

    llvm::TargetOptions fallbackTargetOpts;
    fallbackTargetOpts.EnableFastISel = true;
    llvm::CodeGenOpt::Level fallbackOptLevel = llvm::CodeGenOpt::Level::None;

    llvm::LLVMContext fallbackLLVMCtx;
    IFDEBUG(
    // TODO maybe don't use stderr for this
    // see https://www.llvm.org/docs/Remarks.html, we make use of setupLLVMOptimizationRemarks to count fast isel to selectionDAG fallbacks
    // like -pass-remarks-missed=sdagisel, just the output ios a bit less readable

    if(llvm::setupLLVMOptimizationRemarks(fallbackLLVMCtx, llvm::errs(), 
        "sdagisel",
        "yaml", false))
        errx(EXIT_FAILURE, "Could not setup LLVM optimization remarks to count fallbacks from fast isel to selectionDAG");
    )

    auto wrapFl = [](auto&& f) {
        llvm::write_double(llvm::outs(), std::move(f), llvm::FloatStyle::Fixed, 8);
        return "";
    };

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

        bool jit = args.jit();
        if(features["force-fallback"]){
            std::vector<llvm::LLVMContext*> fallbackLLVMCtxs;
            fallbackLLVMCtxs.resize(iterations);
            for(auto i = 0u; i < iterations; i++){
                // don't know how to clone a context, but atm this doesn't have any options or anything except the optimization remarks, and those are only used in debug mode, so just create a new one
                fallbackLLVMCtxs[i] = new llvm::LLVMContext();
            }

            MEASURE_TIME_START(totalLLVM);
            for(auto i = 0u; i < iterations; i++){
                auto obj = llvm::SmallVector<char, 0>();
                fallbackToLLVMCompilation(*modClones[i], *fallbackLLVMCtxs[i], &obj, jit, /* execute */ false, fallbackTargetOpts, fallbackOptLevel);
            }

            MEASURE_TIME_END(totalLLVM);

            for(auto i = 0u; i < iterations; i++){
                delete fallbackLLVMCtxs[i];
            }

            llvm::outs() << "LLVM Fallback compilation took " << wrapFl(MEASURED_TIME_AS_SECONDS(totalLLVM, iterations)) << " seconds on average over " << iterations << " iterations\n";
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
                regallocEncode(start, end, *modClones[i], std::move(globals), false, jit, "main");
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
                regallocEncode(start, end, *modClones[i], std::move(globalsClones[i-iterations]), false, jit, "main");
            }
            MEASURE_TIME_END(regallocMLIR);

            llvm::outs() << "ISel + RegAlloc + encoding took " << wrapFl(MEASURED_TIME_AS_SECONDS(totalMLIR, iterations)) << " seconds on average over "     << iterations                                         << " iterations\n";
            llvm::outs() << "ISel repeated "                   << iterations                                      << " times without RegAlloc took " << wrapFl(MEASURED_TIME_AS_SECONDS(iselMLIR,     iterations)) << " seconds on average\n";
            llvm::outs() << "RegAlloc repeated "               << iterations                                      << " times without ISel took "     << wrapFl(MEASURED_TIME_AS_SECONDS(regallocMLIR, iterations)) << " seconds on average\n";
            llvm::outs() << "Combining these two times gives " << wrapFl(MEASURED_TIME_AS_SECONDS(iselMLIR, iterations) + MEASURED_TIME_AS_SECONDS(regallocMLIR, iterations)) << " seconds on average, be aware that the last measurements combined do not represent realistic use-case of these functions!\n";
            llvm::outs() << "Experimental: Liveness analysis took " << wrapFl(MEASURED_TIME_AS_SECONDS(liveness, iterations)) << " seconds on average over " << iterations << " iterations\n";
        }
    }else if(features["force-fallback"]){
        auto obj = llvm::SmallVector<char, 0>();
        // TODO add option for fallback optimization level
        if(fallbackToLLVMCompilation(*owningModRef, fallbackLLVMCtx, &obj, args.jit(), /* execute */ true, fallbackTargetOpts, fallbackOptLevel))
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
        // TODO maybe use jit argv[0] instead of main at some point

        if(args.jit()){
            if(execStart == nullptr || execStart == NULL)
                errx(EXIT_FAILURE, "Could not find main function");

            auto main = reinterpret_cast<main_t>(execStart);

            auto [jitArgc, jitArgv] = ArgParse::parseJITArgv();

            if(printOpts != PRINT_NONE)
                llvm::outs() << termcolor::make(termcolor::red, "JIT execution output:\n");

            fflush(stdout);

            auto ret =  main(jitArgc, jitArgv);
            return ret;
        }

        maybeWriteToFile(start, end - start);
    }

    return EXIT_SUCCESS;
}
