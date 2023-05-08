#include <fadec-enc.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "util.h"
#include "AMD64/AMD64Dialect.h"
#include "AMD64/AMD64Ops.h"

// for patternmatching
#include "mlir/IR/PatternMatch.h"
namespace {
#include "AMD64/Lowerings.cpp.inc"
}

// TODO delete all of this later
void testStuff(mlir::ModuleOp mod){
    mlir::MLIRContext* ctx = mod.getContext();

    auto gpr8 = amd64::gpr8Type::get(ctx);
    auto builder = mlir::OpBuilder(ctx);
    auto loc = builder.getUnknownLoc();
    builder.setInsertionPointToStart(mod.getBody());


    auto imm1 = builder.create<amd64::MOV8ri>(loc);
    imm1.getProperties().imm = 1;
    auto imm2 = builder.create<amd64::MOV8ri>(loc);
    imm2.getProperties().imm = 2;

    auto add8rr = builder.create<amd64::ADD8rr>(loc, imm1, imm2);
    auto add8mi = builder.create<amd64::ADD8mi>(loc, imm1, imm2);

    mlir::Operation* generic = add8rr;

    auto opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    FeMnem YAAAAAY = opInterface.getFeMnemonic();

    assert(YAAAAAY == FE_ADD8rr);

    generic = add8mi;
    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    YAAAAAY = opInterface.getFeMnemonic();

    assert(YAAAAAY == FE_ADD8mi);

    auto mul8r = builder.create<amd64::MUL8r>(loc, imm1, imm2);
    generic = mul8r;

    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);

    assert(mul8r.hasTrait<mlir::OpTrait::Operand1IsDestN<1>::Impl>());

    assert((mul8r.hasTrait<mlir::OpTrait::OperandNIsConstrainedToReg<1, FE_AX>::Impl>())); // ah and al, not dx/ax
    // maybe a better way would just be a static method on the op interface, so that we can *get* the constrained register, not just check if it exists

    auto regsTest = builder.create<amd64::CMP8rr>(loc, imm1, imm2);

    regsTest.instructionInfo().regs = {FE_AX, FE_DX};
    assert(regsTest.instructionInfo().regs.getReg1() == FE_AX && regsTest.instructionInfo().regs.getReg2() == FE_DX);

    generic = regsTest;
    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    assert(regsTest.instructionInfo().regs.getReg1() == FE_AX && regsTest.instructionInfo().regs.getReg2() == FE_DX);

    // memory operand Op: interface encode to let the memory op define how it is encoded using FE_MEM
}

int main(int argc, char *argv[]) {
    ArgParse::parse(argc, argv);

    if(ArgParse::args.help()){
        ArgParse::printHelp(argv[0]);
        return EXIT_SUCCESS;
    }

    mlir::MLIRContext ctx;
    ctx.loadAllAvailableDialects();
    ctx.loadDialect<amd64::AMD64Dialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::cf::ControlFlowDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();
    ctx.loadDialect<mlir::LLVM::LLVMDialect>();

    auto inputFile = ArgParse::args.input() ? *ArgParse::args.input : "-";

    auto owningModRef = readMLIRMod(inputFile, ctx);

    testStuff(owningModRef.get());

    return 0;
}
