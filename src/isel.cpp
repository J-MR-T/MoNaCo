#include <llvm/Support/Debug.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Pass/Pass.h>
#include <mlir/IR/PatternMatch.h>

#include "util.h"
#include "AMD64/AMD64Dialect.h"
#include "AMD64/AMD64Ops.h"

// anonymous namespace to contain patterns
namespace {
#include "AMD64/Lowerings.cpp.inc"

// use this to mark that a specific type of instruction is not available to use in the lambda of a pattern
using NOT_AVAILABLE = int;

/// somewhat generic pattern matching struct
template<
    typename opClassToMatch, unsigned bitwidth,
    auto lambda,
    typename INSTrr = NOT_AVAILABLE, typename INSTri = NOT_AVAILABLE, typename INSTrm = NOT_AVAILABLE, typename INSTmi = NOT_AVAILABLE, typename INSTmr = NOT_AVAILABLE,
    // default template parameters start
    int benefit = 1,
    auto bitwidthMatchLambda = []<unsigned innerBitwidth, typename thisType, typename OpAdaptor>(thisType thiis, opClassToMatch op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter){ 
        auto typeToMatch= thiis->getTypeConverter()->convertType(op.getType()).template dyn_cast<amd64::RegisterTypeInterface>();
        // TODO might have to become actual match failure at some point
        assert(typeToMatch && "expected register type");
        // TODO this assertion currently fails wrongly on a conditional branch
        //assert((op->getNumOperands() == 0 || typeToMatch == thiis->getTypeConverter()->convertType(op->getOperand(0).getType()).template dyn_cast<amd64::RegisterTypeInterface>()) && "this simple bitwidth matcher assumes that the type of the op and the type of the operands are the same");

        if(typeToMatch.getBitwidth() != bitwidth)
            return rewriter.notifyMatchFailure(op, "bitwidth mismatch");

        return mlir::success();
    }
>
struct MatchRMI : public mlir::OpConversionPattern<opClassToMatch>{
    MatchRMI(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context) : mlir::OpConversionPattern<opClassToMatch>(typeConverter, context, benefit){}
    using OpAdaptor = typename mlir::OpConversionPattern<opClassToMatch>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(opClassToMatch op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        auto bitwidthMatchResult = bitwidthMatchLambda.template operator()<bitwidth, decltype(this), OpAdaptor>(this, op, adaptor, rewriter);
        if(bitwidthMatchResult.failed())
            return bitwidthMatchResult;

        return lambda.template operator()<bitwidth, OpAdaptor, INSTrr, INSTri, INSTrm, INSTmi, INSTmr>(op, adaptor, rewriter);
    }
};

#define PATTERN_FOR_BITWIDTH(bitwidth, patternName, opClassToMatch, opPrefixToReplaceWith, lambda, ...)                                                                                                                   \
    using patternName ##  bitwidth = MatchRMI<opClassToMatch,  bitwidth,  lambda,                                                                                                                                         \
        opPrefixToReplaceWith ##  bitwidth ## rr, opPrefixToReplaceWith ##  bitwidth ## ri, opPrefixToReplaceWith ##  bitwidth ## rm, opPrefixToReplaceWith ##  bitwidth ## mi, opPrefixToReplaceWith ##  bitwidth ## mr, \
        ## __VA_ARGS__ >;

#define PATTERN(patternName, opClassToMatch, opPrefixToReplaceWith, lambda, ...)                         \
    PATTERN_FOR_BITWIDTH(8,  patternName, opClassToMatch, opPrefixToReplaceWith, lambda,## __VA_ARGS__ ) \
    PATTERN_FOR_BITWIDTH(16, patternName, opClassToMatch, opPrefixToReplaceWith, lambda,## __VA_ARGS__ ) \
    PATTERN_FOR_BITWIDTH(32, patternName, opClassToMatch, opPrefixToReplaceWith, lambda,## __VA_ARGS__ ) \
    PATTERN_FOR_BITWIDTH(64, patternName, opClassToMatch, opPrefixToReplaceWith, lambda,## __VA_ARGS__ )

// TODO an alternative would be to generate custom builders for the RR versions, which check if their argument is a movxxri and then fold it into the RR, resulting in an RI version. That probably wouldn't work because the returned thing would of course expect an RR version, not an RI version
auto binOpMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(auto op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<INSTrr>(op, adaptor.getLhs(), adaptor.getRhs());
    return mlir::success();
};

// it would be nice to use folds for matching mov's and folding them into the add, but that's not possible right now, so we either have to match it here, or ignore it for now (binOpMatchReplace's approach)
// I finally found out when to use the OpAdaptor and when not to: The OpAdaptor seems to give access to the operands in their already converted form, whereas the op itself still has all operands in their original form.
// In this case we need to access the operand in the original form, to check if it was a constant, we're not interested in what it got converted to
auto binOpAndImmMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(auto op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    auto constantOp = mlir::dyn_cast<mlir::arith::ConstantIntOp>(op.getLhs().getDefiningOp());
    auto other = adaptor.getRhs();

    if(!constantOp){
        constantOp = mlir::dyn_cast<mlir::arith::ConstantIntOp>(op.getRhs().getDefiningOp());
        other = adaptor.getLhs();
    }

    if(!constantOp ||
        // immediate is max 32 bit, otherwise we have to generate a mov for it
        constantOp.value() > std::numeric_limits<int32_t>::max() || constantOp.value() < std::numeric_limits<int32_t>::min()
    ){
        // -> we need to use the RR version, there is no pure immediate operand
        rewriter.replaceOpWithNewOp<INSTrr>(op, adaptor.getLhs(), adaptor.getRhs());
    }else{
        // -> there is a pure immediate operand, which fits into the instruction -> we can use the RI version to save the MOVxxri
        auto newOp = rewriter.replaceOpWithNewOp<INSTri>(op, other);
        newOp.instructionInfo().imm = constantOp.value();
    }

    return mlir::success();
};

// TODO think about whether to define different patterns with this or not etc.
// TODO this binOpAndImmMatchReplace could be split up into multiple patterns, but that might be slower
PATTERN(AddIPat, mlir::arith::AddIOp, amd64::ADD, binOpAndImmMatchReplace, 2);
PATTERN(SubIPat, mlir::arith::SubIOp, amd64::SUB, binOpMatchReplace);
PATTERN(AndIPat, mlir::arith::AndIOp, amd64::AND, binOpMatchReplace);
PATTERN(OrIPat,  mlir::arith::OrIOp,  amd64::OR,  binOpMatchReplace);
PATTERN(XOrIPat, mlir::arith::XOrIOp, amd64::XOR, binOpMatchReplace);

auto cmpIBitwidthMatcher =
    []<unsigned innerBitwidth, typename thisType, typename OpAdaptor>(thisType thiis, mlir::arith::CmpIOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter){
        // cmp always has i1 as a result type, so we need to match the arguments' bitwidths
        auto tc = thiis->getTypeConverter();
        auto lhsNewType = tc->convertType(op.getLhs().getType()).template dyn_cast<amd64::RegisterTypeInterface>();
        auto rhsNewType = tc->convertType(op.getRhs().getType()).template dyn_cast<amd64::RegisterTypeInterface>();

        assert(lhsNewType && rhsNewType && "cmp's operands should be convertible to register types");

        bool success = lhsNewType.getBitwidth() == innerBitwidth && lhsNewType == rhsNewType;
        return mlir::failure(!success);
    };
// TODO specify using benefits, that this has to have lower priority than the version which matches a jump with a cmp as an argument (which doesn't exist yet).
PATTERN(CmpIPat, mlir::arith::CmpIOp, amd64::CMP, binOpMatchReplace, 1, cmpIBitwidthMatcher);

template <bool isRem>
auto matchDivRem = []<unsigned actualBitwidth, typename OpAdaptor,
     typename DIVr, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
     >(auto op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    auto div = rewriter.create<DIVr>(op->getLoc(), adaptor.getLhs(), adaptor.getRhs());
    if constexpr(isRem)
        rewriter.replaceOp(op, div.getRemainder());
    else
        rewriter.replaceOp(op, div.getQuotient());
    return mlir::success();
};

#define MULI_DIVI_PAT(bitwidth)                                                                                          \
    using MulIPat ## bitwidth = MatchRMI<mlir::arith::MulIOp, bitwidth, binOpMatchReplace, amd64::MUL ## bitwidth ## r>; \
    using DivUIPat ## bitwidth = MatchRMI<mlir::arith::DivUIOp, bitwidth,                                                \
        matchDivRem<false>,                                                                                              \
        amd64::DIV ## bitwidth ## r>;                                                                                    \
    using DivSIPat ## bitwidth = MatchRMI<mlir::arith::DivSIOp, bitwidth,                                                \
        matchDivRem<false>,                                                                                              \
        amd64::IDIV ## bitwidth ## r>;                                                                                   \
    using RemUIPat ## bitwidth = MatchRMI<mlir::arith::RemUIOp, bitwidth,                                                \
        matchDivRem<true>,                                                                                               \
        amd64::DIV ## bitwidth ## r>;                                                                                    \
    using RemSIPat ## bitwidth = MatchRMI<mlir::arith::RemSIOp, bitwidth,                                                \
        matchDivRem<true>,                                                                                               \
        amd64::IDIV ## bitwidth ## r>;

MULI_DIVI_PAT(8); MULI_DIVI_PAT(16); MULI_DIVI_PAT(32); MULI_DIVI_PAT(64);

#define SHIFT_PAT(bitwidth)                                                                                                                               \
    using ShlIPat ## bitwidth  = MatchRMI<mlir::arith::ShLIOp,  bitwidth, binOpMatchReplace, amd64::SHL ## bitwidth ## rr, amd64::SHL ## bitwidth ## ri>; \
    using ShrUIPat ## bitwidth = MatchRMI<mlir::arith::ShRUIOp, bitwidth, binOpMatchReplace, amd64::SHR ## bitwidth ## rr, amd64::SHL ## bitwidth ## ri>; \
    using ShrSIPat ## bitwidth = MatchRMI<mlir::arith::ShRSIOp, bitwidth, binOpMatchReplace, amd64::SAR ## bitwidth ## rr, amd64::SHL ## bitwidth ## ri>;

SHIFT_PAT(8); SHIFT_PAT(16); SHIFT_PAT(32); SHIFT_PAT(64);

auto movMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(mlir::arith::ConstantIntOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<INSTri>(op, op.value());
    return mlir::success();
};

PATTERN(ConstantIntPat, mlir::arith::ConstantIntOp, amd64::MOV, movMatchReplace);

// branches

auto branchMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename JMP /* it's weird that leaving out the NOT_AVAILABLE works here, but not in matchDivRem */
     >(mlir::cf::BranchOp br, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<JMP>(br, br.getDestOperands(), br.getDest());
    return mlir::success();
};
using BrPat = MatchRMI<mlir::cf::BranchOp, 64, branchMatchReplace, amd64::JMP, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, []{return mlir::success(); }>;

struct MatchCondBr : public mlir::OpConversionPattern<mlir::cf::CondBranchOp> {
    MatchCondBr(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context) : mlir::OpConversionPattern<mlir::cf::CondBranchOp>(typeConverter, context, 3){}

    mlir::LogicalResult matchAndRewrite(mlir::cf::CondBranchOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        auto cmpi = mlir::dyn_cast<mlir::arith::CmpIOp>(op.getCondition().getDefiningOp());
        if(!cmpi)
            return mlir::failure(); // TODO this will not cover all cases yet. i1 arithmetic is a problem...

        // cmp should already have been replaced by the cmp pattern, so we don't need to do that here

        auto ops1 = adaptor.getTrueDestOperands();
        auto ops2 = adaptor.getFalseDestOperands();

        auto block1 = op.getTrueDest();
        auto block2 = op.getFalseDest();

        if(auto constI1 = mlir::dyn_cast<mlir::arith::ConstantIntOp>(op.getCondition().getDefiningOp())){
            // unconditional branch, if it's a constant condition
            if(constI1.value()){
                rewriter.replaceOpWithNewOp<amd64::JMP>(op, ops1, block1);
            }else{
                rewriter.replaceOpWithNewOp<amd64::JMP>(op, ops2, block2);
            }
        }else{
            // conditional branch
            using mlir::arith::CmpIPredicate;

            switch(cmpi.getPredicate()){
                case CmpIPredicate::eq:  rewriter.replaceOpWithNewOp<amd64::JE>(op,  ops1, ops2, block1, block2); break;
                case CmpIPredicate::ne:  rewriter.replaceOpWithNewOp<amd64::JNE>(op, ops1, ops2, block1, block2); break;
                case CmpIPredicate::slt: rewriter.replaceOpWithNewOp<amd64::JL>(op,  ops1, ops2, block1, block2); break;
                case CmpIPredicate::sle: rewriter.replaceOpWithNewOp<amd64::JLE>(op, ops1, ops2, block1, block2); break;
                case CmpIPredicate::sgt: rewriter.replaceOpWithNewOp<amd64::JG>(op,  ops1, ops2, block1, block2); break;
                case CmpIPredicate::sge: rewriter.replaceOpWithNewOp<amd64::JGE>(op, ops1, ops2, block1, block2); break;
                case CmpIPredicate::ult: rewriter.replaceOpWithNewOp<amd64::JB>(op,  ops1, ops2, block1, block2); break;
                case CmpIPredicate::ule: rewriter.replaceOpWithNewOp<amd64::JBE>(op, ops1, ops2, block1, block2); break;
                case CmpIPredicate::ugt: rewriter.replaceOpWithNewOp<amd64::JA>(op,  ops1, ops2, block1, block2); break;
                case CmpIPredicate::uge: rewriter.replaceOpWithNewOp<amd64::JAE>(op, ops1, ops2, block1, block2); break;
            }
        }
        return mlir::success();
    }
};

} // end anonymous namespace


/// TODO
/// takes an operation and does isel on its regions
void prototypeIsel(mlir::Operation* regionOp){
    // TODO actually use regionOp
    using namespace amd64;
    using namespace mlir::arith;

    auto ctx = regionOp->getContext();
    mlir::OpBuilder builder(ctx);
    auto loc = regionOp->getLoc();

    // TODO pattern matching tests, remove later
    auto owningModRef = mlir::OwningOpRef<mlir::ModuleOp>(builder.create<mlir::ModuleOp>(builder.getUnknownLoc()));
    mlir::ModuleOp mod = owningModRef.get();

    builder.setInsertionPointToEnd(&mod.getRegion().front());
    auto patternMatchingTestFn = builder.create<mlir::func::FuncOp>(loc, "patternMatchingTest", mlir::FunctionType::get(ctx, {}, {}));;
    auto entryBB = patternMatchingTestFn.addEntryBlock();
    builder.setInsertionPointToStart(entryBB);

    auto imm8_3 = builder.create<MOV8ri>(loc, 8);
    auto imm8_4 = builder.create<MOV8ri>(loc, 9);
    auto imm16_1 = builder.create<MOV16ri>(loc, 16);
    auto imm16_2 = builder.create<MOV16ri>(loc, 17);
    auto imm32_1 = builder.create<MOV32ri>(loc, 32);
    auto imm32_2 = builder.create<MOV32ri>(loc, 33);
    auto imm64_1 = builder.create<MOV64ri>(loc, 64);
    auto imm64_2 = builder.create<MOV64ri>(loc, 65);

    builder.create<AddIOp>(loc, imm8_3, imm8_4);
    builder.create<AddIOp>(loc, imm16_1, imm16_2);
    builder.create<AddIOp>(loc, imm32_1, imm32_2);
    builder.create<AddIOp>(loc, imm64_1, imm64_2);

    auto const8 = builder.create<ConstantIntOp>(loc, 8, builder.getI8Type());
    auto const16 = builder.create<ConstantIntOp>(loc, 16, builder.getI16Type());
    auto const32 = builder.create<ConstantIntOp>(loc, 32, builder.getI32Type());
    auto const64 = builder.create<ConstantIntOp>(loc, 64, builder.getI64Type());

    builder.create<AddIOp>(loc, const8, imm8_3);
    builder.create<AddIOp>(loc, const16, imm16_1);
    builder.create<AddIOp>(loc, const32, imm32_1);
    auto add = builder.create<AddIOp>(loc, const64, imm64_1);

    assert(mlir::dyn_cast<ConstantIntOp>(add.getLhs().getDefiningOp()));

    builder.create<AddIOp>(loc, add, add);

    builder.create<AddIOp>(loc, imm8_3,  const8);
    builder.create<AddIOp>(loc, imm16_1, const16);
    builder.create<AddIOp>(loc, imm32_1, const32);
    builder.create<AddIOp>(loc, imm64_1, const64);

    auto targetBlock = patternMatchingTestFn.addBlock();

    auto cmp = builder.create<CmpIOp>(loc, CmpIPredicate::sge, add, const64);
    builder.create<mlir::cf::CondBranchOp>(loc, cmp, targetBlock, entryBB);

    builder.setInsertionPointToStart(targetBlock);


    builder.create<mlir::func::ReturnOp>(loc);

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::IntegerType type) -> std::optional<mlir::Type>{
        switch(type.getIntOrFloatBitWidth()) {
            case 8:  return amd64::gpr8Type ::get(type.getContext());
            case 16: return amd64::gpr16Type::get(type.getContext());
            // cmp is not matched using the result type (always i1), but with the operand type, so this doesn't apply there.
            case 1:  // TODO  Internally, MLIR seems to need to convert i1's though, so we will handle them as i32 for now.
            case 32: return amd64::gpr32Type::get(type.getContext());
            case 64: return amd64::gpr64Type::get(type.getContext());

            default: assert(false && "unhandled bitwidth in typeConverter");
        }
    });

    // all register and memloc types are already legal
    // TODO this is probably not needed in the end, because we shouldn't encounter them at first. But for now with manually created ops it is needed.
    typeConverter.addConversion([](amd64::RegisterTypeInterface type) -> std::optional<mlir::Type>{
        return type;
    });
    typeConverter.addConversion([](amd64::memLocType type) -> std::optional<mlir::Type>{
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

#define ADD_PATTERN(patternName) patterns.add<patternName ## 8, patternName ## 16, patternName ## 32, patternName ## 64>(typeConverter, ctx);
    ADD_PATTERN(ConstantIntPat);
    ADD_PATTERN(AddIPat);
    ADD_PATTERN(SubIPat);
    ADD_PATTERN(MulIPat);
    ADD_PATTERN(CmpIPat);
    ADD_PATTERN(AndIPat);
    ADD_PATTERN(OrIPat);
    ADD_PATTERN(XOrIPat);
    ADD_PATTERN(MulIPat);
    ADD_PATTERN(DivUIPat);
    ADD_PATTERN(DivSIPat);
    ADD_PATTERN(RemSIPat);
    ADD_PATTERN(RemSIPat);
    ADD_PATTERN(ShlIPat);
    ADD_PATTERN(ShrSIPat);
    ADD_PATTERN(ShrUIPat);
    patterns.add<MatchCondBr>(typeConverter, ctx);
#undef ADD_PATTERN
    

    llvm::DebugFlag = true;
    //llvm::setCurrentDebugType("greedy-rewriter");
    //llvm::setCurrentDebugType("dialect-conversion");

    DEBUGLOG("Before pattern matching:");
    patternMatchingTestFn.dump();
    //auto result = mlir::applyPatternsAndFoldGreedily(patternMatchingTestFn, std::move(patterns)); // TODO I think this only applies normal rewrite patterns, not conversion patterns...
    auto result = mlir::applyPartialConversion(patternMatchingTestFn, target, std::move(patterns));
    if(result.failed()){
        llvm::errs() << "Pattern matching failed :(\n";
    }else{
        llvm::errs() << "Pattern matching succeeded :)\n";
    }
    DEBUGLOG("After pattern matching:");
    patternMatchingTestFn.dump();
}
