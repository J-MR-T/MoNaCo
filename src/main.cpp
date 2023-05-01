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

int main(int argc, char *argv[])
{
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

    // TODO delete all of this later

    auto gpr8 = amd64::gpr8Type::get(&ctx);
    auto builder = mlir::OpBuilder(&ctx);
    builder.setInsertionPointToStart(owningModRef->getBody());

    auto imm1 = builder.create<amd64::MOV8ri>(builder.getUnknownLoc(), 1);
    auto imm2 = builder.create<amd64::MOV8ri>(builder.getUnknownLoc(), 2);

    auto add8rr = builder.create<amd64::ADD8rr>(builder.getUnknownLoc(), imm1, imm2);
    auto add8mi = builder.create<amd64::ADD8mi>(builder.getUnknownLoc(), imm1, imm2);

    mlir::Operation* generic = add8rr;

    auto opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    auto YAAAAAY = opInterface.getFeMnemonic();

    return 0;
}
