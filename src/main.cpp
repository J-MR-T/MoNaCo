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

#include "isel.h"

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
            instrOp->getOperand(0).getType().isa<amd64::RegisterTypeInterface>()){
        i++;
    }

    llvm::errs() << "Encoding: "; instrOp.dump();

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
    }

    // TODO performance test this version, which just passes all operands, against a version which only passes the operands which are actually used, through some hiddeous case distinctions
    // TODO also maybe make the operands smaller once all instructions are defined, and we know that there are no more than x
    int ret;
    if(resultReg)
        ret = fe_enc64(&cur, mnemonic, *resultReg, operands[0], operands[1], operands[2], operands[3]);
    else
        ret = fe_enc64(&cur, mnemonic, operands[0], operands[1], operands[2], operands[3]);

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
void testOpCreation(mlir::ModuleOp mod){
    mlir::MLIRContext* ctx = mod.getContext();

    auto gpr8 = amd64::gpr8Type::get(ctx);
    assert(gpr8.isa<amd64::RegisterTypeInterface>() && gpr8.dyn_cast<amd64::RegisterTypeInterface>().getBitwidth() == 8 && "gpr8 is not a register type");

    auto builder = mlir::OpBuilder(ctx);
    auto loc = builder.getUnknownLoc();
    builder.setInsertionPointToStart(mod.getBody());

    auto imm8_1 = builder.create<amd64::MOV8ri>(loc);
    imm8_1.instructionInfo().imm = 1;
    imm8_1.instructionInfo().regs.setReg1(FE_CX);
    auto imm8_2 = builder.create<amd64::MOV8ri>(loc);
    imm8_2.instructionInfo().imm = 2;
    imm8_2.instructionInfo().regs.setReg1(FE_R8);

    auto add8rr = builder.create<amd64::ADD8rr>(loc, imm8_1, imm8_2);
    add8rr.instructionInfo().regs.setReg1(FE_CX);

    mlir::Operation* generic = add8rr;

    auto opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    FeMnem YAAAAAY = opInterface.getFeMnemonic();

    assert(YAAAAAY == FE_ADD8rr);

    auto mul8r = builder.create<amd64::MUL8r>(loc, imm8_1, imm8_2);
    generic = mul8r;

    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);

    assert(mul8r.hasTrait<mlir::OpTrait::Operand1IsDestN<1>::Impl>());

    assert((mul8r.hasTrait<mlir::OpTrait::OperandNIsConstrainedToReg<1, FE_AX>::Impl>())); // ah and al, not dx/ax
    // maybe a better way would just be a static method on the op interface, so that we can *get* the constrained register, not just check if it exists

    auto regsTest = builder.create<amd64::CMP8rr>(loc, imm8_1, imm8_2);


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
    auto memSIBD = builder.create<amd64::MemSIBD>(loc, /* base */ add8rr, /* index */ imm8_2);
    memSIBD.getProperties().scale = 2;
    memSIBD.getProperties().displacement = 10;
    assert(memSIBD.getProperties().scale == 2);
    assert(memSIBD.getProperties().displacement == 10);

    auto memSIBD2 = builder.create<amd64::MemSIBD>(loc, /* base */ add8rr, /* scale*/ 4, /* index */ imm8_2, /* displacement */ 20); // basically 'byte ptr [rcx + 4*r8 + 20]'
    assert(memSIBD2.getProperties().scale == 4);
    assert(memSIBD2.getProperties().displacement == 20);

    auto sub8mi = builder.create<amd64::SUB8mi>(loc, memSIBD2);
    sub8mi.instructionInfo().regs.setReg1(FE_BX);
    sub8mi.instructionInfo().imm = 42;

    prototypeEncode(sub8mi);


    auto jmpTestFn = builder.create<mlir::func::FuncOp>(loc, "jmpTest", mlir::FunctionType::get(ctx, {}, {}));;

    builder.setInsertionPointToStart(jmpTestFn.addEntryBlock());
    auto targetBlock = jmpTestFn.addBlock();
    auto imm64 = builder.create<amd64::MOV64ri>(loc, 42);
    builder.create<amd64::ADD64rr>(loc, imm64, imm64);
    builder.create<amd64::JMP>(loc, targetBlock);
    builder.setInsertionPointToStart(targetBlock);
    builder.create<amd64::ADD64rr>(loc, imm64, imm64);
    builder.create<mlir::func::ReturnOp>(loc);

    // TODO maybe premature optimization, just do map[block*] -> address and a vector of unresolved jumps for now. a vector that maps blocks by number to address would be faster, could be accomplished using indices of the functions region's blocks

    // idea: (although probably with a POD struct instead of a tuple)
    llvm::SmallDenseMap<mlir::Block*, 
            std::tuple<uint16_t /* number of jumps to this block that have already been encountered */,
                       uint8_t** /* value of 'cur' obj file pointer at start of block */,
                       llvm::SmallVector<uint8_t**, 2> /* unresolved jumps to this block */
            >, 16>
        blockJmpInfo;
    // every time we enter a new block: check if there is already a map entry for the new block:
    //   if not: we add it (if it has any predecessors), with a number of 0, a pointer to the current obj file pointer, and no unresolved jumps
    //   if yes: we have already found a jump to this block, so set the current obj file pointer to 'here' and resolve all unresolved jumps to this block, and remove them from the vector.
    //      If the number of jumps that have already been encountered is == the number of predecessors, remove the map entry
    //
    // every time we jump to a block: check if there is already a map entry for the target:
    //   if yes: Check if the cur obj file pointer is null.
    //      if yes (that means we have not encountered the block itself yet): 
    //             Add the current obj file pointer to the unresolved jumps vector and increment the number of jumps that have already been encountered.
    //      if no: Use normal jump encoding, get and use the target block pointer (the uint8_t** from the tuple), and increment the number of jumps that have already been encountered.
    //             If the number of jumps that have already been encountered is == the number of predecessors, remove the map entry
    //   if no: add a map entry with a number of 0, a nullptr, and no unresolved jumps.
    //             If the number of jumps that have already been encountered is == the number of predecessors, remove the map entry (i.e. in this case don't add it in the first place)

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

    (void) owningModRef;

    mlir::OpBuilder builder(&ctx);
    auto testMod = mlir::OwningOpRef<mlir::ModuleOp>(builder.create<mlir::ModuleOp>(builder.getUnknownLoc()));
    testOpCreation(*testMod);

    testMod = mlir::OwningOpRef<mlir::ModuleOp>(builder.create<mlir::ModuleOp>(builder.getUnknownLoc()));
    prototypeIsel(*testMod);

    return 0;
}
