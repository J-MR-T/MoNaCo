#include <fadec-enc.h>
#include <fadec.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Pass/Pass.h>

#include "util.h"
#include "AMD64/AMD64Dialect.h"
#include "AMD64/AMD64Ops.h"

// for patternmatching
#include "mlir/IR/PatternMatch.h"
namespace {
#include "AMD64/Lowerings.cpp.inc"

#define DECLARE_RMI(opname, bitwidth)        \
    using INSTrr = opname ## bitwidth ## rr; \
    using INSTri = opname ## bitwidth ## ri; \
    using INSTrm = opname ## bitwidth ## rm; \
    using INSTmr = opname ## bitwidth ## mr; \
    using INSTmi = opname ## bitwidth ## mi;

// TODO maybe it's better to let the lambda return a logical result and forward it to the outside, there might still be a failure in there
#define MATCH_BW_RMI(opname, requestedBitwidth, bwTemplateParamToMatch, typeToMatch, lambda)                             \
    if constexpr (bwTemplateParamToMatch == requestedBitwidth) {                                                         \
        if (typeToMatch.getBitwidth() == requestedBitwidth) {                                                            \
            DECLARE_RMI(opname, requestedBitwidth);                                                                      \
            auto l = lambda;                                                                                             \
            return l.template operator()<requestedBitwidth>();                                                                  \
        }else {                                                                                                          \
            return rewriter.notifyMatchFailure(op /* implicit param */, "expected " #requestedBitwidth " bit " #opname); \
        }                                                                                                                \
    }

#define MATCH_8_16_32_64_RMI(opname, bwTemplateParamToMatch, lambda)                                                                                                 \
    auto typeToMatch= getTypeConverter()->convertType(op.getType()).template dyn_cast<amd64::RegisterTypeInterface>();                                               \
    assert(typeToMatch && "expected register type"); /* TODO for now this is an assert, but it might have to be turned into an actual match failure at some point */ \
    MATCH_BW_RMI(opname,       8,   bwTemplateParamToMatch,  typeToMatch,  lambda)                                                                                   \
    else MATCH_BW_RMI(opname,  16,  bwTemplateParamToMatch,  typeToMatch,  lambda)                                                                                   \
    else MATCH_BW_RMI(opname,  32,  bwTemplateParamToMatch,  typeToMatch,  lambda)                                                                                   \
    else MATCH_BW_RMI(opname,  64,  bwTemplateParamToMatch,  typeToMatch,  lambda)                                                                                   \
    else return rewriter.notifyMatchFailure(op, "Invalid bitwidth, 128 bit not implemented yet, everything else is invalid");

// TODO use something akin to the structure of ConvertOpToLLVMPattern instead of ConversionPattern

// try implementing a few test patterns in C++, because tablegen currently dies when trying to define patterns using operations with properties
template<unsigned bw>
struct AddIPat : public mlir::OpConversionPattern<mlir::arith::AddIOp>{
    using OpConversionPattern<mlir::arith::AddIOp>::OpConversionPattern;

    // TODO for whatever reason, this method doesn't even get *called*, the matching instantly faisl
    mlir::LogicalResult matchAndRewrite(mlir::arith::AddIOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        // this generates 4 patterns, depending on the bitwidth
        MATCH_8_16_32_64_RMI(amd64::ADD, bw, [&]<unsigned actualBitwidth>(){
            // TODO it would be nice to use folds for matching mov's and folding them into the add, but that's not possible right now, so we either have to match it here (see commit 8df6c7d), or ignore it for now
            // TODO an alternative would be to generate custom builders for the RR versions, which check if their argument is a movxxri and then fold it into the RR, resulting in an RI version. That probably wouldn't work because the returned thing would of course expect an RR version, not an RI version
            rewriter.replaceOpWithNewOp<INSTrr>(op, adaptor.getLhs(), adaptor.getRhs());
            return mlir::success();
        })
    }
};

template<unsigned bw>
struct ConstantIntPat : public mlir::OpConversionPattern<mlir::arith::ConstantIntOp>{
    using OpConversionPattern<mlir::arith::ConstantIntOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(mlir::arith::ConstantIntOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        // this generates 4 patterns, depending on the bitwidth
        MATCH_8_16_32_64_RMI(amd64::MOV, bw, [&]<unsigned actualBitwidth>(){
            rewriter.replaceOpWithNewOp<INSTri>(op, op.value());
            return mlir::success();
        })
    }
};


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
void testStuff(mlir::ModuleOp mod){
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


















    // pattern matching tests

    builder.setInsertionPointToEnd(&mod.getRegion().front());
    auto patternMatchingTestFn = builder.create<mlir::func::FuncOp>(loc, "patternMatchingTest", mlir::FunctionType::get(ctx, {}, {}));;
    auto entryBB = patternMatchingTestFn.addEntryBlock();
    builder.setInsertionPointToStart(entryBB);

    llvm::SmallVector<mlir::Value, 32> undeadifyThese;

    auto imm8_3 = builder.create<amd64::MOV8ri>(loc, 8);
    auto imm8_4 = builder.create<amd64::MOV8ri>(loc, 9);
    auto imm16_1 = builder.create<amd64::MOV16ri>(loc, 16);
    auto imm16_2 = builder.create<amd64::MOV16ri>(loc, 17);
    auto imm32_1 = builder.create<amd64::MOV32ri>(loc, 32);
    auto imm32_2 = builder.create<amd64::MOV32ri>(loc, 33);
    auto imm64_1 = builder.create<amd64::MOV64ri>(loc, 64);
    auto imm64_2 = builder.create<amd64::MOV64ri>(loc, 65);

    builder.create<mlir::arith::AddIOp>(loc, imm8_3, imm8_4);
    builder.create<mlir::arith::AddIOp>(loc, imm16_1, imm16_2);
    builder.create<mlir::arith::AddIOp>(loc, imm32_1, imm32_2);
    builder.create<mlir::arith::AddIOp>(loc, imm64_1, imm64_2);

    auto const8 = builder.create<mlir::arith::ConstantIntOp>(loc, 8, builder.getI8Type());
    auto const16 = builder.create<mlir::arith::ConstantIntOp>(loc, 16, builder.getI16Type());
    auto const32 = builder.create<mlir::arith::ConstantIntOp>(loc, 32, builder.getI32Type());
    auto const64 = builder.create<mlir::arith::ConstantIntOp>(loc, 64, builder.getI64Type());

    builder.create<mlir::arith::AddIOp>(loc, const8, imm8_3);
    builder.create<mlir::arith::AddIOp>(loc, const16, imm16_1);
    builder.create<mlir::arith::AddIOp>(loc, const32, imm32_1);
    auto add = builder.create<mlir::arith::AddIOp>(loc, const64, imm64_1);

    builder.create<mlir::arith::AddIOp>(loc, add, add);

    builder.create<mlir::arith::AddIOp>(loc, imm8_3, const8);
    builder.create<mlir::arith::AddIOp>(loc, imm16_1, const16);
    builder.create<mlir::arith::AddIOp>(loc, imm32_1, const32);
    builder.create<mlir::arith::AddIOp>(loc, imm64_1, const64);


    builder.create<mlir::func::ReturnOp>(loc);

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::IntegerType type) -> std::optional<mlir::Type>{
        switch(type.getIntOrFloatBitWidth()) {
            case 8:
                return amd64::gpr8Type::get(type.getContext());
            case 16:
                return amd64::gpr16Type::get(type.getContext());
            case 32:
                return amd64::gpr32Type::get(type.getContext());
            case 64:
                return amd64::gpr64Type::get(type.getContext());

            default:
                assert(false && "unhandled bitwidth");
        }
    });

    // all register types are already legal
    // TODO this is probably not needed in the end, because we shouldn't encounter them at first. But for now with manually created ops it is needed.
    typeConverter.addConversion([](amd64::RegisterTypeInterface type) -> std::optional<mlir::Type>{
        return type;
    });

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<amd64::AMD64Dialect>();
    //target.addLegalDialect<mlir::func::FuncDialect>();
    // func is legal, except for returns and calls, as soon as those have instructions
    //target.addIllegalOp<mlir::func::ReturnOp>();

    mlir::RewritePatternSet patterns(ctx);
    //populateWithGenerated(patterns); // does `patterns.add<SubExamplePat>(ctx);`, ... for all tablegen generated patterns

#define ADD_PATTERN(pattern, bw) patterns.add<pattern<bw>>(typeConverter, ctx);
#define ADD_PATTERN_8_16_32_64(pattern) ADD_PATTERN(pattern, 8) ADD_PATTERN(pattern, 16) ADD_PATTERN(pattern, 32) ADD_PATTERN(pattern, 64)

    ADD_PATTERN_8_16_32_64(AddIPat);
    ADD_PATTERN_8_16_32_64(ConstantIntPat);

#undef ADD_PATTERN
#undef ADD_PATTERN_8_16_32_64

    llvm::DebugFlag = true;
    //llvm::setCurrentDebugType("greedy-rewriter");
    //llvm::setCurrentDebugType("dialect-conversion");

    DEBUGLOG("Before pattern matching:");
    patternMatchingTestFn.dump();
    //auto result = mlir::applyPatternsAndFoldGreedily(patternMatchingTestFn, std::move(patterns)); // TODO OOOOOH. I think this only applise normal rewrite patterns, not conversion patterns...
    auto result = mlir::applyPartialConversion(patternMatchingTestFn, target, std::move(patterns));
    if(result.failed()){
        llvm::errs() << "Pattern matching failed :(\n";
    }else{
        llvm::errs() << "Pattern matching succeeded :)\n";
    }
    DEBUGLOG("After pattern matching:");
    patternMatchingTestFn.dump();

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
    testStuff(*testMod);

    return 0;
}
