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
using NOT_AVAILABLE = void;

template <typename opClassToMatch>
auto defaultBitwidthMatchLambda = []<unsigned bitwidth, typename thisType, typename OpAdaptor>(thisType thiis, opClassToMatch op, OpAdaptor, mlir::ConversionPatternRewriter& rewriter){
    mlir::Type opType;
    if constexpr (opClassToMatch::template hasTrait<mlir::OpTrait::OneResult>())
        opType = op.getType();
    else
        opType = op->getResult(0).getType();

    auto typeToMatch= thiis->getTypeConverter()->convertType(opType).template dyn_cast<amd64::RegisterTypeInterface>();
    // TODO might have to become actual match failure at some point, when we have more than just register types
    assert(typeToMatch && "expected register type");
    // TODO this assertion currently fails wrongly on a conditional branch
    //assert((op->getNumOperands() == 0 || typeToMatch == thiis->getTypeConverter()->convertType(op->getOperand(0).getType()).template dyn_cast<amd64::RegisterTypeInterface>()) && "this simple bitwidth matcher assumes that the type of the op and the type of the operands are the same");

    if(typeToMatch.getBitwidth() != bitwidth)
        return rewriter.notifyMatchFailure(op, "bitwidth mismatch");

    return mlir::success();
};
auto ignoreBitwidthMatchLambda = []<unsigned, typename thisType, typename OpAdaptor>(thisType, auto, OpAdaptor, mlir::ConversionPatternRewriter&){
    return mlir::success();
};


/// somewhat generic pattern matching struct
template<
    typename opClassToMatch, unsigned bitwidth,
    auto lambda,
    // default template parameters start
    typename INSTrr = NOT_AVAILABLE, typename INSTri = NOT_AVAILABLE, typename INSTrm = NOT_AVAILABLE, typename INSTmi = NOT_AVAILABLE, typename INSTmr = NOT_AVAILABLE,
    int benefit = 1,
// if i specify the default arg inline, that instantly crashes clangd. But using a separate variable reduces code duplication anyway, so thanks I guess?
    auto bitwidthMatchLambda = defaultBitwidthMatchLambda<opClassToMatch>
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
    auto constantOp = mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(op.getLhs().getDefiningOp());
    auto other = adaptor.getRhs();

    if(!constantOp){
        constantOp = mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(op.getRhs().getDefiningOp());
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
        auto lhsNewType = tc->convertType(op.getLhs().getType()).template dyn_cast<amd64::RegisterTypeInterface>(); // TODO maybe we can use the adaptor and just use .getType on that instead of bothering with the type converter. Not just here, but everywhere
        auto rhsNewType = tc->convertType(op.getRhs().getType()).template dyn_cast<amd64::RegisterTypeInterface>();

        assert(lhsNewType && rhsNewType && "cmp's operands should be convertible to register types");

        bool success = lhsNewType.getBitwidth() == innerBitwidth && lhsNewType == rhsNewType;
        return mlir::failure(!success);
    };
// TODO specify using benefits, that this has to have lower priority than the version which matches a jump with a cmp as an argument (which doesn't exist yet).
auto cmpIMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename CMPrr, typename CMPri, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
     >(mlir::arith::CmpIOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {

    auto loc = op->getLoc();
    rewriter.create<CMPrr>(loc, adaptor.getLhs(), adaptor.getRhs());

    if(!op->use_empty()){
        using mlir::arith::CmpIPredicate;
        switch(op.getPredicate()){
            case CmpIPredicate::eq:  rewriter.replaceOpWithNewOp<amd64::SETE8r>(op);  break;
            case CmpIPredicate::ne:  rewriter.replaceOpWithNewOp<amd64::SETNE8r>(op); break;
            case CmpIPredicate::slt: rewriter.replaceOpWithNewOp<amd64::SETL8r>(op);  break;
            case CmpIPredicate::sle: rewriter.replaceOpWithNewOp<amd64::SETLE8r>(op); break;
            case CmpIPredicate::sgt: rewriter.replaceOpWithNewOp<amd64::SETG8r>(op);  break;
            case CmpIPredicate::sge: rewriter.replaceOpWithNewOp<amd64::SETGE8r>(op); break;
            case CmpIPredicate::ult: rewriter.replaceOpWithNewOp<amd64::SETB8r>(op);  break;
            case CmpIPredicate::ule: rewriter.replaceOpWithNewOp<amd64::SETBE8r>(op); break;
            case CmpIPredicate::ugt: rewriter.replaceOpWithNewOp<amd64::SETA8r>(op);  break;
            case CmpIPredicate::uge: rewriter.replaceOpWithNewOp<amd64::SETAE8r>(op); break;
        }
    }else{
        rewriter.eraseOp(op); // we need to replace the root op, the CMP doesn't have a result, so replace it with nothing
    }
    return mlir::success();
};
PATTERN(CmpIPat, mlir::arith::CmpIOp, amd64::CMP, cmpIMatchReplace, 1, cmpIBitwidthMatcher);

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

// sign/zero extensions

/// ZExt from i1 pattern
/// TODO maybe do this pattern for sign extend? But sign extending an i1 seems pretty useless
struct ExtUII1Pat : mlir::OpConversionPattern<mlir::arith::ExtUIOp> {
    ExtUII1Pat(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context) : mlir::OpConversionPattern<mlir::arith::ExtUIOp>(typeConverter, context, 1){}

    mlir::LogicalResult matchAndRewrite(mlir::arith::ExtUIOp zextOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        // we're only matching i1s here
        if(!zextOp.getIn().getType().isInteger(1))
            return rewriter.notifyMatchFailure(zextOp, "this pattern only extends i1s");

        // to be precise, we're only matching cmps for the moment, although this might change later
        auto cmpi = mlir::dyn_cast<mlir::arith::CmpIOp>(zextOp.getIn().getDefiningOp());
        if(!cmpi)
            return rewriter.notifyMatchFailure(zextOp, "only cmps are supported for i1 extension for now");

        assert(zextOp.getOut().getType().isIntOrFloat() && "extui with non-int result type");

        // TODO this doesn't make any sense without a CMOVcc/SETcc for/after the cmp
        switch(zextOp.getOut().getType().getIntOrFloatBitWidth()){
            case 8:  rewriter.replaceOp(zextOp, adaptor.getIn() /* the cmpi should be replaced by a SETcc in another pattern, as this sets an 8 bit register anyway, we don't need to do anything here */); break;
            case 16: rewriter.replaceOpWithNewOp<amd64::MOVZXr16r8>(zextOp, adaptor.getIn()); break;
            case 32: rewriter.replaceOpWithNewOp<amd64::MOVZXr32r8>(zextOp, adaptor.getIn()); break;
            case 64: rewriter.replaceOpWithNewOp<amd64::MOVZXr64r8>(zextOp, adaptor.getIn()); break;

            default:
                return rewriter.notifyMatchFailure(zextOp, "unsupported bitwidth for i1 extension");
        }
        return mlir::success();
    }
};


/// the inBitwidth matches the bitwidth of the input operand to the extui, which needs to be different per pattern, because the corresponding instruction differs.
template<unsigned inBitwidth>
/// the outBitwidth matches the bitwidth of the result of the extui, which also affects which instruction is used.
auto truncExtUiSiBitwidthMatcher = []<unsigned outBitwidth, typename thisType, typename OpAdaptor>(thisType thiis, auto op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter){
    // out bitwidth
    auto failure = defaultBitwidthMatchLambda<decltype(op)>.template operator()<outBitwidth, thisType, OpAdaptor>(thiis, op, adaptor, rewriter);
    if(failure.failed())
        return failure;

    // in bitwidth
    return defaultBitwidthMatchLambda<decltype(op)>.template operator()<inBitwidth, thisType, OpAdaptor>(thiis, op, adaptor, rewriter);
};

auto truncExtUiSiMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename MOVZX, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
>(auto szextOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<MOVZX>(szextOp, adaptor.getIn());
    return mlir::success();
};

/// only for 16-64 bits outBitwidth, for 8 we have a special pattern. There are more exceptions: Because not all versions of MOVZX exist, MOVZXr8r8 wouldn't make sense (also invalid in MLIR), MOVZXr64r32 is just a MOV, etc.
#define EXT_UI_SI_PAT(outBitwidth, inBitwidth) \
    using ExtUIPat ## outBitwidth ## _ ## inBitwidth = MatchRMI<mlir::arith::ExtUIOp, outBitwidth, truncExtUiSiMatchReplace, amd64::MOVZX ## r ## outBitwidth ## r ## inBitwidth, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<inBitwidth>>; \
    using ExtSIPat ## outBitwidth ## _ ## inBitwidth = MatchRMI<mlir::arith::ExtSIOp, outBitwidth, truncExtUiSiMatchReplace, amd64::MOVSX ## r ## outBitwidth ## r ## inBitwidth, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<inBitwidth>>;

// generalizable cases:
EXT_UI_SI_PAT(16, 8);
EXT_UI_SI_PAT(32, 8); EXT_UI_SI_PAT(32, 16);
EXT_UI_SI_PAT(64, 8); EXT_UI_SI_PAT(64, 16);

// cases that are still valid in mlir, but not covered here:
// - 32 -> 64 (just a MOV)
// - any weird integer types, but we ignore those anyway
using ExtUIPat64_32 = MatchRMI<mlir::arith::ExtUIOp, 64, truncExtUiSiMatchReplace, amd64::MOV32rr, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<32>>;
// for sign extend, the pattern above would work, but for simplicity, just do it manually here:
using ExtSIPat64_32 = MatchRMI<mlir::arith::ExtSIOp, 64, truncExtUiSiMatchReplace, amd64::MOVSXr64r32, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<32>>;

// trunc
#define TRUNC_PAT(outBitwidth, inBitwidth) \
    using TruncPat ## outBitwidth ## _ ## inBitwidth = MatchRMI<mlir::arith::TruncIOp, outBitwidth, truncExtUiSiMatchReplace, amd64::MOV ## outBitwidth ## rr, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<inBitwidth>>;
TRUNC_PAT(8, 16); TRUNC_PAT(8, 32); TRUNC_PAT(8, 64);
TRUNC_PAT(16, 32); TRUNC_PAT(16, 64);
TRUNC_PAT(32, 64);

// branches
auto branchMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename JMP, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
     >(mlir::cf::BranchOp br, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<JMP>(br, adaptor.getDestOperands(), br.getDest());
    return mlir::success();
};
using BrPat = MatchRMI<mlir::cf::BranchOp, 64, branchMatchReplace, amd64::JMP, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, ignoreBitwidthMatchLambda>;

struct CondBrPat : public mlir::OpConversionPattern<mlir::cf::CondBranchOp> {
    CondBrPat(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context) : mlir::OpConversionPattern<mlir::cf::CondBranchOp>(typeConverter, context, 3){}

    mlir::LogicalResult matchAndRewrite(mlir::cf::CondBranchOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        auto ops1 = adaptor.getTrueDestOperands();
        auto ops2 = adaptor.getFalseDestOperands();

        auto block1 = op.getTrueDest();
        auto block2 = op.getFalseDest();

        if(auto constI1 = mlir::dyn_cast<mlir::arith::ConstantIntOp>(op.getCondition().getDefiningOp())){
            // unconditional branch, if it's a constant condition
            if(constI1.value())
                rewriter.replaceOpWithNewOp<amd64::JMP>(op, ops1, block1);
            else
                rewriter.replaceOpWithNewOp<amd64::JMP>(op, ops2, block2);
            return mlir::success();
        }

        auto cmpi = mlir::dyn_cast<mlir::arith::CmpIOp>(op.getCondition().getDefiningOp());
        if(!cmpi){
            return mlir::failure(); // TODO this will not cover all cases yet. handle i1 arithmetic here
        }

        // cmp should already have been replaced by the cmp pattern, so we don't need to do that here
        // but the cmp can be arbitrarily far away, so we need to possibly reinsert it here.

        if(cmpi->getNextNode() != op){
            // TODO IR/Builders.cpp suggests that this gets inserted at the right point, but check again
            // clone the cmpi, it will then have no uses and get pattern matched with the normal cmp pattern as we want. The SET from that pattern shouldn't be generated, because the cmpi is dead.
            rewriter.clone(*cmpi);
        }

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

        return mlir::success();
    }
};

auto callMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename CALL, typename MOVrr, typename gprType, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
     >(mlir::func::CallOp callOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    if(callOp.getNumResults() > 1)
        return rewriter.notifyMatchFailure(callOp, "multiple return values not supported");

    // TODO needs type conversion for operands. Wait, maybe the adaptor takes care of this
    rewriter.replaceOpWithNewOp<CALL>(callOp, gprType::get(callOp->getContext()), callOp.getCalleeAttr(), adaptor.getOperands());
    return mlir::success();
};

// calls
#define CALL_PAT(bitwidth) \
    using CallPat ## bitwidth = MatchRMI<mlir::func::CallOp, bitwidth, callMatchReplace, amd64::CALL, amd64::MOV ## bitwidth ## rr, amd64::gpr ## bitwidth ## Type>

CALL_PAT(8); CALL_PAT(16); CALL_PAT(32); CALL_PAT(64);

// returns
auto returnMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename RET, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
     >(mlir::func::ReturnOp returnOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    mlir::Value retOperand;
    if(returnOp.getNumOperands() > 1)
        return rewriter.notifyMatchFailure(returnOp, "multiple return values not supported");
    else if(returnOp.getNumOperands() == 0)
        retOperand = rewriter.create<amd64::MOV64ri>(returnOp.getLoc(), 0); // TODO this is not right yet, need to know the return type for the right bitwidth
    else
        retOperand = adaptor.getOperands().front();

    rewriter.replaceOpWithNewOp<RET>(returnOp, retOperand);
    return mlir::success();
};

using ReturnPat = MatchRMI<mlir::func::ReturnOp, 64, returnMatchReplace, amd64::RET, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, ignoreBitwidthMatchLambda>;

// see https://mlir.llvm.org/docs/DialectConversion/#type-converter and https://mlir.llvm.org/docs/DialectConversion/#region-signature-conversion
// -> this is necessary to convert the types of the region ops
/// similar to `RegionOpConversion` from OpenMPToLLVM.cpp
struct FuncPat : public mlir::OpConversionPattern<mlir::func::FuncOp>{
    FuncPat(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context) : mlir::OpConversionPattern<mlir::func::FuncOp>(typeConverter, context, 1){}

    mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp func, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        // TODO the whole signature conversion doesn't work, instead we use the typeconverter to convert the function type, which isn't pretty, but it works
        //mlir::TypeConverter::SignatureConversion signatureConversion(func.getNumArguments());
        //signatureConversion.getConvertedTypes()
        //
        //if(failed(getTypeConverter()->convertSignatureArgs(func.getFunctionType(), signatureConversion)))
        //    return mlir::failure();

        auto newFunc = rewriter.create<mlir::func::FuncOp>(
            func.getLoc(), func.getName(),
            mlir::dyn_cast<mlir::FunctionType>(getTypeConverter()->convertType(adaptor.getFunctionType())) /* this is also probably wrong, type converter still says its illegal afterwards */,
            adaptor.getSymVisibilityAttr(),
            adaptor.getArgAttrsAttr(),
            adaptor.getResAttrsAttr());
        rewriter.inlineRegionBefore(func.getRegion(), newFunc.getRegion(), newFunc.getRegion().end());
        if (failed(rewriter.convertRegionTypes(&newFunc.getRegion(), *getTypeConverter())))
            return mlir::failure();

        rewriter.replaceOp(func, newFunc->getResults());
        return mlir::success();
    }
};

// TODO remove this, this is a skeleton to show that the bare 'ConversionPattern' class also works
struct TestConvPatternWOOp : public mlir::ConversionPattern{
    TestConvPatternWOOp(mlir::TypeConverter& tc, mlir::MLIRContext* context) : mlir::ConversionPattern(tc, "test.conv.pat", 1, context){}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        (void)op; (void)operands; (void)rewriter;
        return mlir::failure();
    }
};

} // end anonymous namespace


// TODO make it possible to
// - select from different preexisting pattern sets based on optimization level (if there is a performance difference between the sets)
//      (maybe 'populate with -Ox patterns' methods?)
// - register custom type conversions
// - register custom patterns to the pattern set

/// takes an operation and does isel on its regions
bool prototypeIsel(mlir::Operation* regionOp){
    using namespace amd64;
    using namespace mlir::arith;

    auto ctx = regionOp->getContext();

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::IntegerType type) -> std::optional<mlir::Type>{
        switch(type.getIntOrFloatBitWidth()) {
            // cmp is not matched using the result type (always i1), but with the operand type, so this doesn't apply there.
            case 1:  // TODO  Internally, MLIR seems to need to convert i1's though, so we will handle them as i8 for now, also because SETcc returns i8.
            case 8:  return amd64::gpr8Type ::get(type.getContext());
            case 16: return amd64::gpr16Type::get(type.getContext());
            case 32: return amd64::gpr32Type::get(type.getContext());
            case 64: return amd64::gpr64Type::get(type.getContext());

            default: assert(false && "unhandled bitwidth in typeConverter");
        }
    });

    // all register and memloc types are already legal
    // TODO this is probably not needed in the end, because we shouldn't encounter them in the first place. But for now with manually created ops it is needed.
    typeConverter.addConversion([](amd64::RegisterTypeInterface type) -> std::optional<mlir::Type>{
        return type;
    });
    typeConverter.addConversion([](amd64::memLocType type) -> std::optional<mlir::Type>{
        return type;
    });
    // TODO this might be entirely stupid
    typeConverter.addConversion([&typeConverter](mlir::FunctionType functionType) -> std::optional<mlir::Type>{
        llvm::SmallVector<mlir::Type, 6> inputs;
        llvm::SmallVector<mlir::Type, 1> results;
        if(mlir::failed(typeConverter.convertTypes(functionType.getInputs(), inputs))
                || mlir::failed(typeConverter.convertTypes(functionType.getResults(), results)))
            return {};

        return mlir::FunctionType::get(functionType.getContext(), 
            inputs, results
        );
    });

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<amd64::AMD64Dialect>();

    // from https://discourse.llvm.org/t/lowering-and-type-conversion-of-block-arguments/63570
    // funcs are only legal once their args have been converted
    // TODO if this costs a lot of performance, try defining an amd64::FuncOp and make all of func illegal
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) { return typeConverter.isLegal(op.getFunctionType()); });
    // this also needs a pattern to rewrite func ops, see `FuncPat`

    //target.addLegalDialect<mlir::func::FuncDialect>();
    // func is legal, except for returns and calls, as soon as those have instructions
    //target.addIllegalOp<mlir::func::ReturnOp>();

    mlir::RewritePatternSet patterns(ctx);
    //populateWithGenerated(patterns); // does `patterns.add<SubExamplePat>(ctx);`, ... for all tablegen generated patterns

    // TODO probably put this pattern adding near the actual patterns, so its harder to forget, and there are multiple pattern sets to choose from, depending on the optimization level
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
    patterns.add<ExtUII1Pat>(typeConverter, ctx);
    patterns.add<ExtUIPat16_8, ExtUIPat32_8, ExtUIPat64_8, ExtUIPat32_16, ExtUIPat64_16, ExtUIPat64_32>(typeConverter, ctx);
    patterns.add<ExtSIPat16_8, ExtSIPat32_8, ExtSIPat64_8, ExtSIPat32_16, ExtSIPat64_16, ExtSIPat64_32>(typeConverter, ctx);
    patterns.add<TruncPat8_16, TruncPat8_32, TruncPat8_64, TruncPat16_32, TruncPat16_64, TruncPat32_64>(typeConverter, ctx);

    patterns.add<BrPat>(typeConverter, ctx);
    patterns.add<CondBrPat>(typeConverter, ctx);

    ADD_PATTERN(CallPat);
    patterns.add<ReturnPat>(typeConverter, ctx);

    patterns.add<FuncPat>(typeConverter, ctx);

    patterns.add<TestConvPatternWOOp>(typeConverter, ctx);
#undef ADD_PATTERN
    
    //llvm::setCurrentDebugType("greedy-rewriter");
    //llvm::setCurrentDebugType("dialect-conversion");

    //auto result = mlir::applyPatternsAndFoldGreedily(patternMatchingTestFn, std::move(patterns)); // TODO I think this only applies normal rewrite patterns, not conversion patterns...
    return mlir::failed(mlir::applyPartialConversion(regionOp, target, std::move(patterns)));
}
