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
    // steady_clock == MONOTONIC, best for performance measurements
#define MEASURE_TIME_START(point) auto point ## _start = std::chrono::steady_clock::now()

#define MEASURE_TIME_END(point) auto point ## _end = std::chrono::steady_clock::now()

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
    enum {BENCH_NONE = 0x0, BENCH_COMPTIME = 0x1, BENCH_RUNTIME = 0x2};
    int benchOpts = BENCH_NONE;
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
        auto printArgValues = args.print.values();
        if(llvm::find(printArgValues, "input") != printArgValues.end()) printOpts |= PRINT_INPUT;
        if(llvm::find(printArgValues, "isel")  != printArgValues.end()) printOpts |= PRINT_ISEL;
        if(llvm::find(printArgValues, "asm")   != printArgValues.end()) printOpts |= PRINT_ASM;

        if(printOpts == PRINT_NONE)
            errx(EXIT_FAILURE, "Invalid argument for --print, valid arguments are: input, isel, asm");
    }

    if(args.benchmark()){
        auto benchArgValues = args.benchmark.values();
        if(llvm::find(benchArgValues, "comptime") != benchArgValues.end()) benchOpts |= BENCH_COMPTIME;
        if(llvm::find(benchArgValues, "runtime")  != benchArgValues.end()) benchOpts |= BENCH_RUNTIME;
        if(llvm::find(benchArgValues, "all")      != benchArgValues.end()) benchOpts |= BENCH_COMPTIME | BENCH_RUNTIME;
        if(benchOpts == BENCH_NONE)
            errx(EXIT_FAILURE, "Invalid argument for --benchmark, valid arguments are: 'comptime', 'runtime', 'all'");

        if(benchOpts & BENCH_RUNTIME && !args.jit())
            errx(EXIT_FAILURE, "Cannot benchmark run-time without JIT");
    }

    if(args.debug()){
        llvm::errs() << "Debugging enabled\n";
#ifdef NDEBUG
        llvm::errs() << "This is not recommended in a release build\n";
#endif
    }

    if(args.debugMore()){
        llvm::errs() << "More debugging enabled\n";
#ifdef NDEBUG
        llvm::errs() << "This is not recommended in a release build\n";
#endif

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

    if(benchOpts & BENCH_COMPTIME){
        std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modClones(2*iterations);
        for(unsigned i = 0; i < modClones.size(); i++){
            modClones[i] = mlir::OwningOpRef<mlir::ModuleOp>(owningModRef->clone());
        }

        llvm::outs() << "Measurement description;Measurement (usually time in seconds);Iterations\n";

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
                // also get the function address, again to make it fair, as MoNaCo also does this (also might not actually do anything otherwise)
                fallbackToLLVMCompilation(*modClones[i], *fallbackLLVMCtxs[i], &obj, jit, fallbackTargetOpts, fallbackOptLevel).second->getFunctionAddress("main");
            }

            MEASURE_TIME_END(totalLLVM);

            for(auto i = 0u; i < iterations; i++){
                delete fallbackLLVMCtxs[i];
            }

            llvm::outs() << "LLVM compilation;" << wrapFl(MEASURED_TIME_AS_SECONDS(totalLLVM, iterations)) << ";" << iterations << "\n";
        }else{
            // allocate 2 GiB (small code model)
            auto [start, end] = mmapSpace(2ll*1024ll*1024ll*1024ll, PROT_READ|PROT_WRITE /* execute will be added later, for security*/);

            MEASURE_TIME_START(totalMLIR);

            if(!start || !end)
                err(EXIT_FAILURE, "mmap");

            for(unsigned i = 0; i < iterations; i++){
                // first pass: ISel
                amd64::GlobalsInfo globals;
                maximalIsel(*modClones[i], globals);

                // second pass: RegAlloc + encoding
                // - will need a third pass in between to do liveness analysis later
                auto codeInfo = regallocEncode(start, end, *modClones[i], std::move(globals), false, jit, "main");
            }

            MEASURE_TIME_END(totalMLIR);

            std::vector<amd64::GlobalsInfo> globalsClones(iterations);

            MEASURE_TIME_START(iselMLIR);
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

            auto combined         = MEASURED_TIME_AS_SECONDS(totalMLIR,    iterations);
            auto iselIsolated     = MEASURED_TIME_AS_SECONDS(iselMLIR,     iterations);
            auto regallocIsolated = MEASURED_TIME_AS_SECONDS(regallocMLIR, iterations);
            auto sumIsolated      = iselIsolated + regallocIsolated;
            auto deviation        = std::abs(combined/sumIsolated - 1);

            auto livenessIsolated = MEASURED_TIME_AS_SECONDS(liveness, iterations);

            llvm::outs()                             <<
                "ISel + RegAlloc + Encoding in one;" << wrapFl(combined)         << ";" << iterations << "\n"  <<
                "ISel isolated;"                     << wrapFl(iselIsolated)     << ";" << iterations << "\n"  <<
                "RegAlloc + Encoding isolated;"      << wrapFl(regallocIsolated) << ";" << iterations << "\n"  <<
                "Sum of all isolated;"               << wrapFl(sumIsolated)      << ";" << iterations << "\n"  <<
                "Deviation isolated vs combined;"    << wrapFl(deviation)        << ";" << iterations << "\n"  <<
                "Separate Liveness:"                 << wrapFl(livenessIsolated) << ";" << iterations << "\n";
        }
    }

    if(benchOpts & BENCH_RUNTIME){
        if(features["force-fallback"]){
            auto obj = llvm::SmallVector<char, 0>();
            // TODO add option for fallback optimization level
            auto [ret, engineUP] = fallbackToLLVMCompilation(*owningModRef, fallbackLLVMCtx, &obj, args.jit(), /* execute */ fallbackTargetOpts, fallbackOptLevel);
            if(ret != 0)
                errx(EXIT_FAILURE, "LLVM compilation failed");

            auto main = reinterpret_cast<main_t>(engineUP->getFunctionAddress("main"));
            auto [jitArgc, jitArgv] = ArgParse::parseJITArgv();
            MEASURE_TIME_START(totalLLVM);
            for(unsigned i = 0; i < iterations; i++){
                main(jitArgc, jitArgv);
            }
            MEASURE_TIME_END(totalLLVM);

            llvm::outs() << "LLVM execution;" << wrapFl(MEASURED_TIME_AS_SECONDS(totalLLVM, iterations)) << ";" << iterations << "\n";
        }else{
            // allocate 2 GiB (small code model)
            auto [start, end] = mmapSpace(2ll*1024ll*1024ll*1024ll, PROT_READ|PROT_WRITE /* execute will be added later, for security*/);

            if(!start || !end)
                err(EXIT_FAILURE, "mmap");

            amd64::GlobalsInfo globals;
            auto iselFailed = maximalIsel(*owningModRef, globals);
            if(iselFailed)
                errx(EXIT_FAILURE, "ISel failed");

            auto codeInfo = regallocEncode(start, end, *owningModRef, std::move(globals), false, args.jit(), "main");

            auto execStart = (main_t)codeInfo.startSymbolAddr;
            if(!execStart)
                errx(EXIT_FAILURE, "Could not find main function");

            mprotect(codeInfo.textSectionStart, codeInfo.bufEnd - codeInfo.textSectionStart, PROT_READ|PROT_EXEC);

            auto [jitArgc, jitArgv] = ArgParse::parseJITArgv();

            MEASURE_TIME_START(totalMoNaCo);
            for(unsigned i = 0; i < iterations; i++){
                execStart(jitArgc, jitArgv);
            }
            MEASURE_TIME_END(totalMoNaCo);

            llvm::outs() << "MoNaCo execution;" << wrapFl(MEASURED_TIME_AS_SECONDS(totalMoNaCo, iterations)) << ";" << iterations << "\n";
        }
    }

    if(benchOpts != BENCH_NONE)
        // stop after benchmarking
        return EXIT_SUCCESS;

    if(features["force-fallback"]){
        auto obj = llvm::SmallVector<char, 0>();
        // TODO add option for fallback optimization level
        auto [ret, engineUP] = fallbackToLLVMCompilation(*owningModRef, fallbackLLVMCtx, &obj, args.jit(), /* execute */ fallbackTargetOpts, fallbackOptLevel);
        if(ret != 0)
            errx(EXIT_FAILURE, "LLVM compilation failed");

        if(args.jit()){
            auto main = reinterpret_cast<main_t>(engineUP->getFunctionAddress("main"));
            auto [jitArgc, jitArgv] = ArgParse::parseJITArgv();
            return main(jitArgc, jitArgv);
        }

        maybeWriteToFile(obj.data(), obj.size());
    }else{ // "normal" case
        // allocate 2 GiB (small code model)
        auto [start, end] = mmapSpace(2ll*1024ll*1024ll*1024ll, PROT_READ|PROT_WRITE /* execute will be added later, for security*/);

        if(!start || !end)
            err(EXIT_FAILURE, "mmap");

        // first pass: ISel
        amd64::GlobalsInfo globals;
        // TODO fall back on error
        auto iselFailed = maximalIsel(*owningModRef, globals);

        if(printOpts & PRINT_ISEL){
            llvm::outs() << termcolor::make(termcolor::red, "After ISel:\n");
            owningModRef->print(llvm::outs());
        }

        if(iselFailed)
            errx(EXIT_FAILURE, "ISel failed");

        // second pass: RegAlloc + encoding
        // - will need a third pass in between to do liveness analysis later
        // TODO fall back on error
        MCDescriptor codeInfo = regallocEncode(start, end, *owningModRef, std::move(globals), printOpts & PRINT_ASM, args.jit(), "main");
        // TODO maybe use jit argv[0] instead of main at some point

		// TODO omg the new code for dhyrstone is for some reason modifying itself...
		// to find out more, make r-x using mprotect, then it should segfault where the erroneous write to code is

        if(args.jit()){
            auto execStart = codeInfo.startSymbolAddr;
            if(execStart == nullptr || execStart == NULL)
                errx(EXIT_FAILURE, "Could not find main function");

            auto main = reinterpret_cast<main_t>(execStart);

            auto [jitArgc, jitArgv] = ArgParse::parseJITArgv();

            if(printOpts != PRINT_NONE)
                llvm::outs() << termcolor::make(termcolor::red, "JIT execution output:\n");

			DEBUGLOG("main is at: " << (void*)main);

            fflush(stdout);

            // make the code non-writable, but executable
            // regallocEncode asserts that the text section starts on a page boundary
            mprotect(codeInfo.textSectionStart, codeInfo.bufEnd - codeInfo.textSectionStart, PROT_READ|PROT_EXEC);

            auto ret =  main(jitArgc, jitArgv);
            return ret;
        }

        maybeWriteToFile(start, end - start);
    }

    return EXIT_SUCCESS;
}
