#pragma once

#include <llvm/IR/Module.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

// to translate from MLIR Dialect soup to LLVM dialect
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

// to translate from LLVM Dialect to LLVM IR
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

// to translate from LLVM IR to machine code
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/MC/MCContext.h>

#include <llvm/ExecutionEngine/ExecutionEngine.h>

#include "util.h"

// TODO to truly make it useful as a library, make the rewrite pattern set a parameter

/// returns it failed
inline bool lowerToLLVMDialect(mlir::ModuleOp mod) noexcept{
    IFDEBUG(llvm::setCurrentDebugType("dialect-conversion")); // like debug-only=dialect-conversion

    mlir::MLIRContext& ctx = *mod.getContext();
    mlir::LLVMConversionTarget target(ctx);
    // or: mlir::ConversionTarget ... plus target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    // needed for other conversions for the pre-existing dialect, as well as for the b::PointerType
    mlir::LLVMTypeConverter typeConverter(&ctx);
    assert(typeConverter.useOpaquePointers() && "opaque pointers are required for the lowering to llvm");

    mlir::RewritePatternSet patterns(&ctx);
    //mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

    return mlir::failed(mlir::applyFullConversion(mod, target, std::move(patterns)));
}

/// returns whether it failed
inline bool llvmCompileMod(llvm::Module& mod, llvm::SmallVector<char, 0>& outputVec, llvm::TargetOptions TargetOpt, llvm::CodeGenOpt::Level OptLevel){
    // adapted from https://www.llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl08.html
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    mod.setTargetTriple(targetTriple);

    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);

    // Print an error and exit if we couldn't find the requested target.
    // This generally occurs if we've forgotten to initialise the
    // TargetRegistry or we have a bogus target triple.
    if (!target) {
        llvm::errs() << error;
        return -1;
    }

    auto CPU         = "generic";
    auto features    = "";

    // apparently, PIC is not the default
    auto RM = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);

    // For some reason, the targetMachine needs to be deleted manually, so encapsulate it in a unique_ptr
    auto targetMachineUP = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(targetTriple, CPU, features, TargetOpt, RM));

    auto DL = targetMachineUP->createDataLayout();
    mod.setDataLayout(DL);

    targetMachineUP->setOptLevel(OptLevel);

    std::error_code ec;

    if(ec){
        llvm::errs() << "Could not open file: " << ec.message() << "\n";
        return -1;
    }

    // old pass manager for back-end
    llvm::legacy::PassManager pass;

    llvm::MCContext mcctx(llvm::Triple(targetTriple), nullptr, nullptr, nullptr);
    auto* mcctxPtr = &mcctx;

    auto out = llvm::raw_svector_ostream(outputVec);

    if(targetMachineUP->addPassesToEmitMC(pass, mcctxPtr, out)){
        llvm::errs() << "TargetMachine can't emit MC of this type" << "\n";
        return -1;
    }

    pass.run(mod);

    return 0;
}

inline std::unique_ptr<llvm::ExecutionEngine> llvmJITCompileMod(std::unique_ptr<llvm::Module> llvmModUP, llvm::TargetOptions TargetOpt, llvm::CodeGenOpt::Level OptLevel){
    // don't let the linker throw away the file
    LLVMLinkInMCJIT();

    llvm::EngineBuilder builder(std::move(llvmModUP));
    builder.setEngineKind(llvm::EngineKind::JIT);
    std::string error;
    builder.setErrorStr(&error);
    builder.setOptLevel(OptLevel);
    builder.setTargetOptions(TargetOpt);
    builder.setRelocationModel(llvm::Reloc::PIC_);
    builder.setCodeModel(llvm::CodeModel::Small);

    // TODO probably need to deallocate this in some way or another
    auto engine = std::unique_ptr<llvm::ExecutionEngine>(builder.create());
    //DEBUGLOG("lazy compilation: " << engine->isCompilingLazily());
    if (!engine)
        err(1, "could not create engine: %s", error.c_str());

    return engine;
}

/// returns 0 on success, -1 on failure, whatever the JIT compiled program returns if jit is passed
inline std::pair<int, std::unique_ptr<llvm::ExecutionEngine>> fallbackToLLVMCompilation(mlir::ModuleOp mlirMod, llvm::LLVMContext& llvmCtx, llvm::SmallVector<char, 0>* obj, bool jit, llvm::TargetOptions TargetOpt = {}, llvm::CodeGenOpt::Level OptLevel = llvm::CodeGenOpt::None){
    if(!obj && !jit){
        llvm::errs() << "No output specified\n";
        return {-1, nullptr};
    }

    // mlir mod -> llvm dialect mod
    if(lowerToLLVMDialect(mlirMod)){
        llvm::errs() << "Could not lower to LLVM dialect\n";
        return {-1, nullptr};
    }

    auto* mlirCtx = mlirMod.getContext();
    mlir::registerBuiltinDialectTranslation(*mlirCtx);
    mlir::registerLLVMDialectTranslation(*mlirCtx);

    // llvm dialect mod -> llvm mod
    auto llvmModUP = mlir::translateModuleToLLVMIR(mlirMod, llvmCtx);

    if(!llvmModUP){
        llvm::errs() << "Could not translate MLIR module to LLVM IR\n";
    }

    llvm::InitializeAllTargetInfos();
    //llvm::InitializeAllTargets();
    //llvm::InitializeAllTargetMCs();
    //llvm::InitializeAllAsmParsers();
    //llvm::InitializeAllAsmPrinters();
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetDisassembler();
    
    if(jit){
        auto engineUP = llvmJITCompileMod(std::move(llvmModUP), TargetOpt, OptLevel);
        if(!engineUP){
            llvm::errs() << "Could not JIT compile LLVM module\n";
            return {-1, nullptr};
        }

        return {0, std::move(engineUP)};
    }else if(llvmCompileMod(*llvmModUP, *obj, TargetOpt, OptLevel) != 0){
        llvm::errs() << "Could not compile LLVM module\n";
        return {-1, nullptr};
    }
    return {0, nullptr};
}

