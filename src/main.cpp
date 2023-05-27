#include <chrono>
#include <string>

#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "AMD64/AMD64Dialect.h"
#include "AMD64/AMD64Ops.h"
#include "isel.h"
#include "fallback.h"
#include "regallocEncode.h"

// TODO delete all of this later
void testOpCreation(mlir::ModuleOp mod){
    mlir::MLIRContext* ctx = mod.getContext();

    llvm::errs() << termcolor::red << "=== Encoding tests ===\n" << termcolor::reset ;

    auto gpr8 = amd64::gpr8Type::get(ctx);
    assert(gpr8.isa<amd64::RegisterTypeInterface>() && gpr8.dyn_cast<amd64::RegisterTypeInterface>().getBitwidth() == 8 && "gpr8 is not a register type");

    auto builder = mlir::OpBuilder(ctx);
    auto loc = builder.getUnknownLoc();
    builder.setInsertionPointToStart(mod.getBody());

    auto imm8_1 = builder.create<amd64::MOV8ri>(loc);
    imm8_1.instructionInfo().imm = 1;
    imm8_1.instructionInfo().regs.reg1 = FE_CX;
    auto imm8_2 = builder.create<amd64::MOV8ri>(loc);
    imm8_2.instructionInfo().imm = 2;
    imm8_2.instructionInfo().regs.reg1 = FE_R8;

    auto add8rr = builder.create<amd64::ADD8rr>(loc, imm8_1, imm8_2);
    add8rr.instructionInfo().regs.reg1 = FE_CX;

    mlir::Operation* generic = add8rr;

    auto opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    FeMnem YAAAAAY = opInterface.getFeMnemonic();

    assert(YAAAAAY == FE_ADD8rr);

    auto mul8r = builder.create<amd64::MUL8r>(loc, imm8_1, imm8_2);
    generic = mul8r;
    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    assert(mul8r.hasTrait<mlir::OpTrait::Operand0IsDestN<0>::Impl>());

    auto [resC1, resC2] = mul8r.getResultRegisterConstraints();
    assert(resC1.which == 0 && resC1.reg == FE_AX && resC2.which == 1 && resC2.reg == FE_AH);

    auto mul16r = builder.create<amd64::MUL16r>(loc, imm8_1, imm8_2);
    generic = mul16r;
    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    assert(mul16r.hasTrait<mlir::OpTrait::Operand0IsDestN<0>::Impl>());

    auto [resC3, resC4] = mul16r.getResultRegisterConstraints();
    assert(resC3.which == 0 && resC3.reg == FE_AX && resC4.which == 1 && resC4.reg == FE_DX);

    auto regsTest = builder.create<amd64::CMP8rr>(loc, imm8_1, imm8_2);


    regsTest.instructionInfo().regs = {FE_AX, FE_DX};
    assert(regsTest.instructionInfo().regs.reg1 == FE_AX && regsTest.instructionInfo().regs.reg2 == FE_DX);

    generic = regsTest;
    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    assert(regsTest.instructionInfo().regs.reg1 == FE_AX && regsTest.instructionInfo().regs.reg2 == FE_DX);

    // immediate stuff
    auto immTest = builder.create<amd64::MOV8ri>(loc, 42);
    immTest.instructionInfo().regs = {FE_AX, FE_DX};

    // encoding test for simple things
    debugEncodeOp(mod, immTest);
    debugEncodeOp(mod, add8rr);

    // memory operand Op: interface encode to let the memory op define how it is encoded using FE_MEM
    auto memSIBD = builder.create<amd64::MemSIBD>(loc, /* base */ add8rr, /* index */ imm8_2);
    memSIBD.getProperties().scale = 2;
    memSIBD.getProperties().displacement = 10;
    assert(memSIBD.getProperties().scale == 2);
    assert(memSIBD.getProperties().displacement == 10);

    auto memSIBD2 = builder.create<amd64::MemSIBD>(loc, /* base */ add8rr, /* scale*/ 4, /* index */ imm8_2, /* displacement */ 20); // basically 'byte ptr [rcx + 4*r8 + 20]'
    assert(memSIBD2.getProperties().scale == 4);
    assert(memSIBD2.getProperties().displacement == 20);

    auto sub8mi = builder.create<amd64::SUB8mi>(loc, memSIBD2);
    sub8mi.instructionInfo().regs.reg1 = FE_BX;
    sub8mi.instructionInfo().imm = 42;

    debugEncodeOp(mod, sub8mi);

    auto jmpTestFn = builder.create<mlir::func::FuncOp>(loc, "jmpTest", mlir::FunctionType::get(ctx, {}, {}));;
    auto entryBB = jmpTestFn.addEntryBlock();

    auto call = builder.create<amd64::CALL>(loc, gpr8, "jmpTest", mlir::ValueRange{});
    debugEncodeOp(mod, call);


    builder.setInsertionPointToStart(entryBB);
    auto targetBlock1 = jmpTestFn.addBlock();
    auto targetBlock2 = jmpTestFn.addBlock();
    auto imm64 = builder.create<amd64::MOV64ri>(loc, 42);
    builder.create<amd64::ADD64rr>(loc, imm64, imm64);
    builder.create<amd64::JMP>(loc, targetBlock1);
    builder.setInsertionPointToStart(targetBlock1);
    builder.create<amd64::ADD64rr>(loc, imm64, imm64);

    llvm::errs() << termcolor::red << "=== Jump inversion test ===\n" << termcolor::reset ;
    auto jnz = builder.create<amd64::JNZ>(loc, mlir::ValueRange{}, mlir::ValueRange{}, targetBlock1, targetBlock2);
    jmpTestFn.dump();

    auto jz = jnz.invert(builder);
    jnz->replaceAllUsesWith(jz);
    jnz->erase();

    jmpTestFn.dump();
}

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

    llvm::DebugFlag = args.debug();

    mlir::MLIRContext ctx;
    ctx.loadAllAvailableDialects();
    ctx.loadDialect<amd64::AMD64Dialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::cf::ControlFlowDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();
    ctx.loadDialect<mlir::LLVM::LLVMDialect>();

    auto inputFile = ArgParse::args.input() ? *ArgParse::args.input : "-";

    auto owningModRef = readMLIRMod(inputFile, ctx);

    if(args.benchmark()){
        // TODO what happens if this throws an exception? Is that fine?
        unsigned iterations = std::stoi(std::string{args.iterations() ? *args.iterations : "1"});

        std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modClones(iterations);
        for(unsigned i = 0; i < iterations; i++){
            modClones[i] = mlir::OwningOpRef<mlir::ModuleOp>(owningModRef->clone());
        }

        if(args.isel()){
            MEASURE_TIME_START(totalMLIR);

            std::vector<uint8_t> encoded;
            for(unsigned i = 0; i < iterations; i++){
                // first pass: ISel
                prototypeIsel(*modClones[i]);

                // second pass: RegAlloc + encoding
                // - will need a third pass in between to do liveness analysis later
                regallocEncodeRepeated(encoded, *modClones[i]);
            }

            MEASURE_TIME_END(totalMLIR);

            llvm::outs() << "ISel + dummy RegAlloc + encoding took " << MEASURED_TIME_AS_SECONDS(totalMLIR, iterations) << " seconds on average over " << iterations << " iterations\n";

            // TODO the encoding certainly won't work with block args currently
        }else if(args.fallback()){
            MEASURE_TIME_START(totalLLVM);

            for(auto i = 0u; i < iterations; i++){
                auto obj = llvm::SmallVector<char, 0>();
                fallbackToLLVMCompilation(*modClones[i], obj);
            }

            MEASURE_TIME_END(totalLLVM);

            llvm::outs() << "LLVM Fallback compilation took " << MEASURED_TIME_AS_SECONDS(totalLLVM, iterations) << " seconds on average over " << iterations << " iterations\n";
        }
    }else if(args.isel()){
        // first pass: ISel
        prototypeIsel(*owningModRef);

        if(args.debug()){
            llvm::outs() << "After ISel:\n";
            owningModRef->dump();
        }

        std::vector<uint8_t> encoded;
        // second pass: RegAlloc + encoding
        // - will need a third pass in between to do liveness analysis later
        regallocEncode(encoded, *owningModRef, args.debug());
    }else if(args.fallback()){
        auto obj = llvm::SmallVector<char, 0>();
        return fallbackToLLVMCompilation(*owningModRef, obj);
    }else if(args.debug()){
        mlir::OpBuilder builder(&ctx);
        auto testMod = mlir::OwningOpRef<mlir::ModuleOp>(builder.create<mlir::ModuleOp>(builder.getUnknownLoc()));
        testOpCreation(*testMod);
    }

    return EXIT_SUCCESS;
}
