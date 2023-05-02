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

    auto imm1 = builder.create<amd64::MOV8ri>(loc, 1);
    auto imm2 = builder.create<amd64::MOV8ri>(loc, 2);

    auto add8rr = builder.create<amd64::ADD8rr>(loc, imm1, imm2);
    auto add8mi = builder.create<amd64::ADD8mi>(loc, imm1, imm2);

    mlir::Operation* generic = add8rr;

    auto opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    auto YAAAAAY = opInterface.getFeMnemonic();

    assert(YAAAAAY == FE_ADD8rr);

    generic = add8mi;
    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    YAAAAAY = opInterface.getFeMnemonic();

    assert(YAAAAAY == FE_ADD8mi);

    auto mul8r = builder.create<amd64::MUL8r>(loc, imm1, builder.getIntegerAttr(builder.getIntegerType(32), FeReg::FE_AX));
    generic = mul8r;

    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    if(auto constraints = opInterface.getRegConstraints()){
        //constraints
    }
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
