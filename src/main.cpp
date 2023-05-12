#include <fadec-enc.h>
#include <fadec.h>

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

void prototypeEncode(mlir::Operation* op){
    // there will be many case distinctions here, the alternative would be a general 'encode' interface, where every instruction defines its own encoding.
    // that would be more extensible, but probably even more code

    auto instrOp = mlir::dyn_cast<amd64::InstructionOpInterface>(op);
    assert(instrOp && "Operation does not implement InstructionOpInterface");

    FeMnem mnemonic = instrOp.getFeMnemonic();

    uint8_t buf[16]; // TODO obviously this needs to be variable etc. later, but for now it's enough, one x86 instruction is never more than 15 bytes
    uint8_t* cur = buf;
    
    // TODO needs special handling for jumps of course, as well as for calls

    // actual number of operands should be: op->getNumOperands() + (hasImm?1:0)
    assert(instrOp->getNumOperands() < 4 && "Too many operands for instruction");

    // TODO zeros are fine, right?
    // first operand of the encoded instruction is the result register, if there is one
    std::optional<FeOp> resultReg{};
    FeOp operands[4] = {0};

    if(!instrOp->hasTrait<mlir::OpTrait::ZeroResults>())
        resultReg = instrOp.instructionInfo().regs.getReg1();

    unsigned i = 0;
    // fadec only needs the dest1 == op1 once, so in this case we skip that first register operand
    if(instrOp->hasTrait<mlir::OpTrait::Operand1IsDestN<1>::Impl>() && 
        /* first operand is register operand: */
            instrOp->getNumOperands() > 0 &&
            instrOp->getOperand(0).getType().hasTrait<mlir::TypeTrait::IsRegisterType>()){
        i++;
    }

    llvm::errs() << "Encoding: "; instrOp.dump();

    DEBUGLOG("num operands: " << instrOp->getNumOperands() << " starting encode at " << i);
    for(unsigned feOpIndex = 0; i < instrOp->getNumOperands(); i++, feOpIndex++){
        auto operandValue = instrOp->getOperand(i);
        if(auto encodeInterface = mlir::dyn_cast<amd64::EncodeOpInterface>(operandValue.getDefiningOp())){
            encodeInterface.dump();
            operands[feOpIndex] = encodeInterface.encode();
        }else{
            // TODO maaaaaybe it's possible to do this with a TypeInterface on the gpr types, which have a method to return the register they're in, instead of registerOf, but that is a much simpler solution for now

            // TODO should be able to put this code somewhere else so that everyone can access this
            auto asOpResult = mlir::dyn_cast<mlir::OpResult>(operandValue);

            assert(asOpResult && "Operand is neither a memory op, nor an OpResult"); // TODO oh god what about block args. also: how do i do register allocation and asm emitting at the same time with block args? They don't know where to store their arg to immediately, do they?
            operands[feOpIndex] = amd64::registerOf(asOpResult);
        }
    }

    // immediate operand
    if(instrOp->hasTrait<mlir::OpTrait::HasImm>()){
        operands[i] = instrOp.instructionInfo().imm;
        DEBUGLOG("Added immediate operand");
    }

    // TODO performance test this version, which just passes all operands, against a version which only passes the operands which are actually used, through some hiddeous case distinctions
    // TODO also maybe make the operands smaller once all instructions are defined
    // TODO distinguish between the cases where the resultReg exists and where it doesn't
    // actual:
    int ret;
    if(resultReg)
        ret = fe_enc64(&cur, mnemonic, *resultReg, operands[0], operands[1], operands[2], operands[3]);
    else
        ret = fe_enc64(&cur, mnemonic, operands[0], operands[1], operands[2], operands[3]);
    // test, TODO delete these later:
    //int ret = fe_enc64(&cur, amd64::MOV8ri::getFeMnemonic(), FE_AX, 42);
    //int ret = fe_enc64(&cur, amd64::ADD64rr::getFeMnemonic(), FE_AX, FE_AX, FE_DX);
    //int ret = fe_enc64(&cur, amd64::SUB8mi::getFeMnemonic(), FE_MEM(FE_CX, 4, FE_R8, 20), 5);

    if(ret < 0){
        llvm::errs() << "Test encoding went wrong :(\nOperands:\n";
        for(unsigned i = 0; i < 4; i++){
            llvm::errs() << operands[i] << "\n";
        }
    }

    // decode & print to test if it works
    FdInstr instr; 
    ret = fd_decode(buf, sizeof(buf), 64, 0, &instr);
    if(ret < 0){
        llvm::errs() << "Test encoding resulted in non-decodable instruction :(\n";
    }else{
        char fmtbuf[64];
        fd_format(&instr, fmtbuf, sizeof(fmtbuf));
        llvm::errs() << "Test encoding resulted in: " << fmtbuf << "\n";
    }
}

// TODO delete all of this later
void testStuff(mlir::ModuleOp mod){
    mlir::MLIRContext* ctx = mod.getContext();

    auto gpr8 = amd64::gpr8Type::get(ctx);
    (void) gpr8;

    auto builder = mlir::OpBuilder(ctx);
    auto loc = builder.getUnknownLoc();
    builder.setInsertionPointToStart(mod.getBody());


    auto imm1 = builder.create<amd64::MOV8ri>(loc);
    imm1.instructionInfo().imm = 1;
    imm1.instructionInfo().regs.setReg1(FE_CX);
    auto imm2 = builder.create<amd64::MOV8ri>(loc);
    imm2.instructionInfo().imm = 2;
    imm2.instructionInfo().regs.setReg1(FE_R8);

    auto add8rr = builder.create<amd64::ADD8rr>(loc, imm1, imm2);
    add8rr.instructionInfo().regs.setReg1(FE_CX);

    mlir::Operation* generic = add8rr;

    auto opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    FeMnem YAAAAAY = opInterface.getFeMnemonic();

    assert(YAAAAAY == FE_ADD8rr);

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

    // immediate stuff
    auto immTest = builder.create<amd64::MOV8ri>(loc, 42);
    immTest.instructionInfo().regs = {FE_AX, FE_DX};

    // encoding test for simple things
    prototypeEncode(immTest);
    prototypeEncode(add8rr);

    // memory operand Op: interface encode to let the memory op define how it is encoded using FE_MEM
    auto memSIBD = builder.create<amd64::MemSIBD>(loc, /* base */ add8rr, /* index */ imm2);
    memSIBD.getProperties().scale = 2;
    memSIBD.getProperties().displacement = 10;
    assert(memSIBD.getProperties().scale == 2);
    assert(memSIBD.getProperties().displacement == 10);

    auto memSIBD2 = builder.create<amd64::MemSIBD>(loc, /* base */ add8rr, /* scale*/ 4, /* index */ imm2, /* displacement */ 20); // basically 'byte ptr [rcx + 4*r8 + 20]'
    assert(memSIBD2.getProperties().scale == 4);
    assert(memSIBD2.getProperties().displacement == 20);

    auto sub8mi = builder.create<amd64::SUB8mi>(loc, memSIBD2);
    sub8mi.instructionInfo().regs.setReg1(FE_BX);
    sub8mi.instructionInfo().imm = 42;

    prototypeEncode(sub8mi);

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
