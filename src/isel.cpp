#include <span>
#include <thread>

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/PassManager.h>

#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Pass/Pass.h>
#include <mlir/IR/PatternMatch.h>

#include "util.h"
#include "isel.h"
#include "AMD64/AMD64Dialect.h"
#include "AMD64/AMD64Ops.h"

using GlobalsInfo = amd64::GlobalsInfo;
using GlobalSymbolInfo = amd64::GlobalSymbolInfo;

// anonymous namespace to contain patterns
namespace {
#include "AMD64/Lowerings.cpp.inc"

using namespace mlir;

// use this to mark that a specific type of instruction is not available to use in the lambda of a pattern
using NOT_AVAILABLE = void;

template <typename OpTy, bool matchZeroResult = false>
auto defaultBitwidthMatchLambda = []<unsigned bitwidth, typename thisType, typename OpAdaptor>(thisType thiis, OpTy op, OpAdaptor, mlir::ConversionPatternRewriter& rewriter){
    auto matchZeroResultReturn = [](){
        if constexpr(matchZeroResult)
            return mlir::success();
        else
            return mlir::failure();
        
    };

    if constexpr (OpTy::template hasTrait<mlir::OpTrait::ZeroResults>()){
        return matchZeroResultReturn();
    }
    if constexpr(OpTy::template hasTrait<mlir::OpTrait::VariadicResults>()){
        if(op.getNumResults() == 0)
            return matchZeroResultReturn();
    }

    mlir::Type opType;
    if constexpr (OpTy::template hasTrait<mlir::OpTrait::OneResult>())
        opType = op.getType();
    else
        opType = op->getResult(0).getType();

    auto type = thiis->getTypeConverter()->convertType(opType);
    // TODO might be slow
    if(!type)
        return rewriter.notifyMatchFailure(op, "type conversion failed");

    auto typeToMatch= type.template dyn_cast<amd64::RegisterTypeInterface>();
    //assert(typeToMatch && "expected register type");
    if(!typeToMatch)
        return rewriter.notifyMatchFailure(op, "expected register type");
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
    typename OpTy, unsigned bitwidth,
    auto lambda,
    // default template parameters start
    typename INSTrr = NOT_AVAILABLE, typename INSTri = NOT_AVAILABLE, typename INSTrm = NOT_AVAILABLE, typename INSTmi = NOT_AVAILABLE, typename INSTmr = NOT_AVAILABLE,
    int benefit = 1,
// if i specify the default arg inline, that instantly crashes clangd. But using a separate variable reduces code duplication anyway, so thanks I guess?
    auto bitwidthMatchLambda = defaultBitwidthMatchLambda<OpTy>
>
struct MatchRMI : public mlir::OpConversionPattern<OpTy>{
    MatchRMI(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<OpTy>(tc, ctx, benefit){}
    using OpAdaptor = typename mlir::OpConversionPattern<OpTy>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        auto bitwidthMatchResult = bitwidthMatchLambda.template operator()<bitwidth, decltype(this), OpAdaptor>(this, op, adaptor, rewriter);
        if(bitwidthMatchResult.failed())
            return bitwidthMatchResult;

        return lambda.template operator()<bitwidth, OpAdaptor, INSTrr, INSTri, INSTrm, INSTmi, INSTmr>(op, adaptor, rewriter);
    }
};

#define PATTERN_FOR_BITWIDTH(bitwidth, patternName, OpTy, opPrefixToReplaceWith, lambda, ...)                                                                                                                   \
    using patternName ##  bitwidth = MatchRMI<OpTy,  bitwidth,  lambda,                                                                                                                                         \
        opPrefixToReplaceWith ##  bitwidth ## rr, opPrefixToReplaceWith ##  bitwidth ## ri, opPrefixToReplaceWith ##  bitwidth ## rm, opPrefixToReplaceWith ##  bitwidth ## mi, opPrefixToReplaceWith ##  bitwidth ## mr, \
        ## __VA_ARGS__ >;

#define PATTERN(patternName, opTy, opPrefixToReplaceWith, lambda, ...)                         \
    PATTERN_FOR_BITWIDTH(8,  patternName, opTy, opPrefixToReplaceWith, lambda,## __VA_ARGS__ ) \
    PATTERN_FOR_BITWIDTH(16, patternName, opTy, opPrefixToReplaceWith, lambda,## __VA_ARGS__ ) \
    PATTERN_FOR_BITWIDTH(32, patternName, opTy, opPrefixToReplaceWith, lambda,## __VA_ARGS__ ) \
    PATTERN_FOR_BITWIDTH(64, patternName, opTy, opPrefixToReplaceWith, lambda,## __VA_ARGS__ )

// TODO technically I could replace the OpAdaptor template arg everywhere with a simple `auto adaptor` parameter
// TODO even better: use a typename T for the op, and then use T::Adaptor for the adaptor

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
    []<unsigned innerBitwidth, typename thisType, typename OpAdaptor>(thisType thiis, auto op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter){
        // cmp always has i1 as a result type, so we need to match the arguments' bitwidths
        auto tc = thiis->getTypeConverter();
        auto lhsNewType = tc->convertType(op.getLhs().getType()).template dyn_cast<amd64::RegisterTypeInterface>(); // TODO maybe we can use the adaptor and just use .getType on that instead of bothering with the type converter. Not just here, but everywhere
        auto rhsNewType = tc->convertType(op.getRhs().getType()).template dyn_cast<amd64::RegisterTypeInterface>();

        assert(lhsNewType && rhsNewType && "cmp's operands should be convertible to register types");

        bool success = lhsNewType.getBitwidth() == innerBitwidth && lhsNewType == rhsNewType;
        return mlir::failure(!success);
    };
// TODO specify using benefits, that this has to have lower priority than the version which matches a jump with a cmp as an argument (which doesn't exist yet).
template<typename CmpPredicate>
auto cmpIMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename CMPrr, typename CMPri, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
     >(auto op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {

    auto loc = op->getLoc();
    rewriter.create<CMPrr>(loc, adaptor.getLhs(), adaptor.getRhs());

    if(!op->use_empty()){
        switch(op.getPredicate()){
            case CmpPredicate::eq:  rewriter.replaceOpWithNewOp<amd64::SETE8r>(op);  break;
            case CmpPredicate::ne:  rewriter.replaceOpWithNewOp<amd64::SETNE8r>(op); break;
            case CmpPredicate::slt: rewriter.replaceOpWithNewOp<amd64::SETL8r>(op);  break;
            case CmpPredicate::sle: rewriter.replaceOpWithNewOp<amd64::SETLE8r>(op); break;
            case CmpPredicate::sgt: rewriter.replaceOpWithNewOp<amd64::SETG8r>(op);  break;
            case CmpPredicate::sge: rewriter.replaceOpWithNewOp<amd64::SETGE8r>(op); break;
            case CmpPredicate::ult: rewriter.replaceOpWithNewOp<amd64::SETB8r>(op);  break;
            case CmpPredicate::ule: rewriter.replaceOpWithNewOp<amd64::SETBE8r>(op); break;
            case CmpPredicate::ugt: rewriter.replaceOpWithNewOp<amd64::SETA8r>(op);  break;
            case CmpPredicate::uge: rewriter.replaceOpWithNewOp<amd64::SETAE8r>(op); break;
        }
    }else{
        rewriter.eraseOp(op); // we need to replace the root op, the CMP doesn't have a result, so replace it with nothing
    }
    return mlir::success();
};
PATTERN(CmpIPat, mlir::arith::CmpIOp, amd64::CMP, cmpIMatchReplace<mlir::arith::CmpIPredicate>, 1, cmpIBitwidthMatcher);

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
struct ExtUII1Pat : mlir::OpConversionPattern<mlir::arith::ExtUIOp> {
    ExtUII1Pat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::arith::ExtUIOp>(tc, ctx, 2){}

    mlir::LogicalResult matchAndRewrite(mlir::arith::ExtUIOp zextOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        // we're only matching i1s here
        if(!zextOp.getIn().getType().isInteger(1))
            return rewriter.notifyMatchFailure(zextOp, "this pattern only extends i1s");

        // to be precise, we're only matching cmps for the moment, although this might change later
        if(!zextOp.getIn().isa<mlir::OpResult>())
            return rewriter.notifyMatchFailure(zextOp, "i1 zext pattern only matches cmps, this seems to be a block arg");

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

// TODO sign extension for i1


/// the inBitwidth matches the bitwidth of the input operand to the extui, which needs to be different per pattern, because the corresponding instruction differs.
template<unsigned inBitwidth, auto getIn, auto getOut>
/// the outBitwidth matches the bitwidth of the result of the extui, which also affects which instruction is used.
auto truncExtUiSiBitwidthMatcher = []<unsigned outBitwidth, typename thisType, typename OpAdaptor>(thisType thiis, auto op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter){
    // TODO check that this doesn't accidentally match gpr8 which have an i1 as source, that would be wrong, at least for sign extensions

    // out bitwidth
    auto failure = defaultBitwidthMatchLambda<decltype(op)>.template operator()<outBitwidth, thisType, OpAdaptor>(thiis, op, adaptor, rewriter);
    if(failure.failed())
        return rewriter.notifyMatchFailure(op, "out bitwidth mismatch");

    // in bitwidth
    mlir::Type opType = getIn(adaptor).getType();
    auto typeToMatch= thiis->getTypeConverter()->convertType(opType).template dyn_cast<amd64::RegisterTypeInterface>();
    assert(typeToMatch && "expected register type");

    if(typeToMatch.getBitwidth() != inBitwidth)
        return rewriter.notifyMatchFailure(op, "in bitwidth mismatch");

    return mlir::success();
};

template<unsigned inBitwidth, auto getIn, auto getOut>
auto truncExtUiSiMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename MOVZX, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
>(auto szextOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    // we need to take care to truncate an i1 to 0/1, before we 'actually' use it, i.e. do real computations with it on the other side of the MOVZX
    // i1's are represented as 8 bits currently, so we only need to check this in patterns which extend 8 bit
    if constexpr (inBitwidth == 8){
        if(getIn(szextOp).getType().isInteger(1)){
            // assert the 8 bits for the i1
            assert(mlir::dyn_cast<amd64::RegisterTypeInterface>(getIn(adaptor).getType()).getBitwidth() == 8);

            // and it with 1, to truncate it
            auto AND = rewriter.create<amd64::AND8ri>(szextOp.getLoc(), getIn(adaptor));
            AND.instructionInfo().imm = 0x1;
            rewriter.replaceOpWithNewOp<MOVZX>(szextOp, AND);
            return mlir::success();
        }
    }
        

    rewriter.replaceOpWithNewOp<MOVZX>(szextOp, getIn(adaptor));
    return mlir::success();
};

auto arithGetIn = [](auto adaptorOrOp){ return adaptorOrOp.getIn(); };
auto arithGetOut = [](auto adaptorOrOp){ return adaptorOrOp.getOut(); };
/// only for 16-64 bits outBitwidth, for 8 we have a special pattern. There are more exceptions: Because not all versions of MOVZX exist, MOVZXr8r8 wouldn't make sense (also invalid in MLIR), MOVZXr64r32 is just a MOV, etc.
#define EXT_UI_SI_PAT(outBitwidth, inBitwidth) \
    using ExtUIPat ## outBitwidth ## _ ## inBitwidth = MatchRMI<mlir::arith::ExtUIOp, outBitwidth, truncExtUiSiMatchReplace<inBitwidth, arithGetIn, arithGetOut>, amd64::MOVZX ## r ## outBitwidth ## r ## inBitwidth, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<inBitwidth, arithGetIn, arithGetOut>>; \
    using ExtSIPat ## outBitwidth ## _ ## inBitwidth = MatchRMI<mlir::arith::ExtSIOp, outBitwidth, truncExtUiSiMatchReplace<inBitwidth, arithGetIn, arithGetOut>, amd64::MOVSX ## r ## outBitwidth ## r ## inBitwidth, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<inBitwidth, arithGetIn, arithGetOut>>;

// generalizable cases:
EXT_UI_SI_PAT(16, 8);
EXT_UI_SI_PAT(32, 8); EXT_UI_SI_PAT(32, 16);
EXT_UI_SI_PAT(64, 8); EXT_UI_SI_PAT(64, 16);

#undef EXT_UI_SI_PAT

// cases that are still valid in mlir, but not covered here:
// - 32 -> 64 (just a MOV)
// - any weird integer types, but we ignore those anyway
using ExtUIPat64_32 = MatchRMI<mlir::arith::ExtUIOp, 64, truncExtUiSiMatchReplace<32, arithGetIn, arithGetOut>, amd64::MOV32rr, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<32, arithGetIn, arithGetOut>>;
// for sign extend, the pattern above would work, but for simplicity, just do it manually here:
using ExtSIPat64_32 = MatchRMI<mlir::arith::ExtSIOp, 64, truncExtUiSiMatchReplace<32, arithGetIn, arithGetOut>, amd64::MOVSXr64r32, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<32, arithGetIn, arithGetOut>>;

// trunc
#define TRUNC_PAT(outBitwidth, inBitwidth) \
    using TruncPat ## outBitwidth ## _ ## inBitwidth = MatchRMI<mlir::arith::TruncIOp, outBitwidth, truncExtUiSiMatchReplace<inBitwidth, arithGetIn, arithGetOut>, amd64::MOV ## outBitwidth ## rr, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<inBitwidth, arithGetIn, arithGetOut>>;
TRUNC_PAT(8, 16); TRUNC_PAT(8, 32); TRUNC_PAT(8, 64);
TRUNC_PAT(16, 32); TRUNC_PAT(16, 64);
TRUNC_PAT(32, 64);

#undef TRUNC_PAT

// branches
auto branchMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename JMP, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
     >(auto br, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<JMP>(br, adaptor.getDestOperands(), br.getDest());
    return mlir::success();
};
using BrPat = MatchRMI<mlir::cf::BranchOp, 64, branchMatchReplace, amd64::JMP, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, ignoreBitwidthMatchLambda>;

template<typename IntegerCmpOp, typename thisType, typename OpAdaptor>
auto condBrMatchReplace = [](thisType thiis, auto /* some kind of cond branch, either cf or llvm */ op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
        auto ops1 = adaptor.getTrueDestOperands();
        auto ops2 = adaptor.getFalseDestOperands();

        auto* block1 = op.getTrueDest();
        auto* block2 = op.getFalseDest();

        assert(thiis->getTypeConverter()->convertType(mlir::IntegerType::get(thiis->getContext(), 1)) == amd64::gpr8Type::get(thiis->getContext()));

        auto generalI1Case = [&](){
            // and the condition i1 with 1, then do JNZ
            auto andri = rewriter.create<amd64::AND8ri>(op.getLoc(), adaptor.getCondition());
            andri.instructionInfo().imm = 1;

            rewriter.replaceOpWithNewOp<amd64::JNZ>(op, ops1, ops2, block1, block2);
            return mlir::success();
        };

        // if its a block argument, emit a CMP and a conditional JMP
        if(adaptor.getCondition().template isa<mlir::BlockArgument>()) [[unlikely]]{
            // need to do this in case of a block argument at the start, because otherwise calling getDefiningOp() will fail. This is also called if no other case matches
            return generalI1Case();
        } else if(auto constI1AsMov8 = mlir::dyn_cast<amd64::MOV8ri>(adaptor.getCondition().getDefiningOp())) [[unlikely]]{
            // constant conditions, that can either occur naturally or through folding, will result in i1 constants that get matched by the constant int pattern, and thus converted to MOV8ri (because i1 is modeled as 8 bit)

            // do an unconditional JMP, if it's a constant condition
            if(constI1AsMov8.instructionInfo().imm)
                rewriter.replaceOpWithNewOp<amd64::JMP>(op, ops1, block1);
            else
                rewriter.replaceOpWithNewOp<amd64::JMP>(op, ops2, block2);

            // TODO how do I remove this MOV8ri, if it's only used here? erasing it results in an error about failing to legalize the erased op
            return mlir::success();
        } else if(auto setccPredicate = mlir::dyn_cast<amd64::PredicateInterface>(adaptor.getCondition().getDefiningOp())){
            // conditional branch

            auto cmpi = mlir::dyn_cast<IntegerCmpOp>(op.getCondition().getDefiningOp());
            auto CMP = setccPredicate->getPrevNode();
            assert(cmpi && CMP && "Conditional branch with SETcc, but without cmpi and CMP");

            // we're using the SETcc here, because the cmpi might have been folded, so we need to get to the original CMP, which is the CMP before the SETcc, then the SETcc has the right predicate

            // cmp should already have been replaced by the cmp pattern, so we don't need to do that here
            // but the cmp can be arbitrarily far away, so we need to reinsert it here, except if it's immediately before our current op (cond.br), and it's the replacement for the SETcc that we're using
            if(!(cmpi->getNextNode() == op && setccPredicate->getNextNode() == cmpi)){
                // TODO IR/Builders.cpp suggests that this gets inserted at the right point, but check again
                // clone the cmpi, it will then have no uses and get pattern matched with the normal cmp pattern as we want. The SET from that pattern shouldn't be generated, because the cmpi is dead.
                rewriter.clone(*CMP);
            }


            using namespace amd64::conditional;

            switch(setccPredicate.getPredicate()){
                case Z:  rewriter.replaceOpWithNewOp<amd64::JE>(op,  ops1, ops2, block1, block2); break;
                case NZ: rewriter.replaceOpWithNewOp<amd64::JNE>(op, ops1, ops2, block1, block2); break;
                case L:  rewriter.replaceOpWithNewOp<amd64::JL>(op,  ops1, ops2, block1, block2); break;
                case LE: rewriter.replaceOpWithNewOp<amd64::JLE>(op, ops1, ops2, block1, block2); break;
                case G:  rewriter.replaceOpWithNewOp<amd64::JG>(op,  ops1, ops2, block1, block2); break;
                case GE: rewriter.replaceOpWithNewOp<amd64::JGE>(op, ops1, ops2, block1, block2); break;
                case C:  rewriter.replaceOpWithNewOp<amd64::JB>(op,  ops1, ops2, block1, block2); break;
                case BE: rewriter.replaceOpWithNewOp<amd64::JBE>(op, ops1, ops2, block1, block2); break;
                case A:  rewriter.replaceOpWithNewOp<amd64::JA>(op,  ops1, ops2, block1, block2); break;
                case NC: rewriter.replaceOpWithNewOp<amd64::JAE>(op, ops1, ops2, block1, block2); break;
                default: llvm_unreachable("unknown predicate");
            }

            //if(setccPredicate->use_empty())
                //rewriter.eraseOp(setccPredicate);

            return mlir::success();
        }else{
            // general i1 arithmetic
            return generalI1Case();
        }
};

struct CondBrPat : public mlir::OpConversionPattern<mlir::cf::CondBranchOp> {
    CondBrPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::cf::CondBranchOp>(tc, ctx, 3){}

    mlir::LogicalResult matchAndRewrite(mlir::cf::CondBranchOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        return condBrMatchReplace<mlir::arith::CmpIOp,decltype(this), OpAdaptor>(this, op, adaptor, rewriter);
    }
};

auto callMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename CALL, typename MOVrr, typename gprType, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
     >(auto callOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    if(callOp.getNumResults() > 1)
        return rewriter.notifyMatchFailure(callOp, "multiple return values not supported");

    // direct call and indirect call are just handled the same as in the llvm dialect, the first op can be a ptr. So we don't do anything here, and check if the attribute exists in encoding, if it does, move it to AX and start one after the normal operand with moving to the operand registers
    if(callOp.getNumResults() == 0)
        rewriter.replaceOpWithNewOp<CALL>(callOp, TypeRange(), callOp.getCalleeAttr(), adaptor.getOperands());
    else
        rewriter.replaceOpWithNewOp<CALL>(callOp, gprType::get(callOp->getContext()), callOp.getCalleeAttr(), adaptor.getOperands());

    return mlir::success();
};

#define CALL_PAT(bitwidth) \
    using CallPat ## bitwidth = MatchRMI<mlir::func::CallOp, bitwidth, callMatchReplace, amd64::CALL, amd64::MOV ## bitwidth ## rr, amd64::gpr ## bitwidth ## Type, NOT_AVAILABLE, NOT_AVAILABLE, 1, defaultBitwidthMatchLambda<mlir::func::CallOp, true>>

CALL_PAT(8); CALL_PAT(16); CALL_PAT(32); CALL_PAT(64);

#undef CALL_PAT

// TODO maybe AND i1's before returning them
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
// TODO i found this other way with `populateAnyFunctionOpInterfaceTypeConversionPattern`, performance test the two against one another
//struct FuncPat : public mlir::OpConversionPattern<mlir::func::FuncOp>{
//    FuncPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::func::FuncOp>(tc, ctx, 1){}
//
//    mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp func, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
//        // TODO the whole signature conversion doesn't work, instead we use the typeconverter to convert the function type, which isn't pretty, but it works
//        //mlir::TypeConverter::SignatureConversion signatureConversion(func.getNumArguments());
//        //signatureConversion.getConvertedTypes()
//        //
//        //if(failed(getTypeConverter()->convertSignatureArgs(func.getFunctionType(), signatureConversion)))
//        //    return mlir::failure();
//
//        auto newFunc = rewriter.create<mlir::func::FuncOp>(
//            func.getLoc(), func.getName(),
//            mlir::dyn_cast<mlir::FunctionType>(getTypeConverter()->convertType(adaptor.getFunctionType())) [> this is also probably wrong, type converter still says its illegal afterwards <],
//            adaptor.getSymVisibilityAttr(),
//            adaptor.getArgAttrsAttr(),
//            adaptor.getResAttrsAttr());
//        rewriter.inlineRegionBefore(func.getRegion(), newFunc.getRegion(), newFunc.getRegion().end());
//        if (failed(rewriter.convertRegionTypes(&newFunc.getRegion(), *getTypeConverter())))
//            return rewriter.notifyMatchFailure(func, "failed to convert region types");
//
//        //convertFuncOpTypes(func, *getTypeConverter(), rewriter);
//
//        rewriter.replaceOp(func, newFunc->getResults());
//        return mlir::success();
//    }
//};

// TODO remove this, this is a skeleton to show that the bare 'ConversionPattern' class also works
struct TestConvPatternWOOp : public mlir::ConversionPattern{
    TestConvPatternWOOp(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::ConversionPattern(tc, "test.conv.pat", 1, ctx){}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
        (void)op; (void)operands; (void)rewriter;
        return mlir::failure();
    }
};

// llvm test patterns
struct LLVMGEPPattern : public mlir::OpConversionPattern<LLVM::GEPOp>{
    LLVMGEPPattern(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<LLVM::GEPOp>(tc, ctx, 1){}

    mlir::LogicalResult matchAndRewrite(LLVM::GEPOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        auto  dl = mlir::DataLayout::closest(op);
        // TODO this is wrong, we don't want to add the size of the whole type

        auto getBytesOffsetAndType = [&](auto type, int64_t elemNum) -> std::pair<int64_t, mlir::Type>  {
            using namespace mlir::LLVM;

            if(LLVMStructType structType = type.template dyn_cast<LLVMStructType>()){
                assert(elemNum >= 0 && "struct element number cannot be negative");
                auto elemNumUnsigned = static_cast<uint64_t>(elemNum);

                auto structParts = structType.getBody();

                // modified version of LLVMStructType::getTypeSizeInBits
                unsigned structSizeUpToNow = 0;
                unsigned structAlignment = 1;
                for (auto [i, element] : llvm::enumerate(structParts)){
                    // stop if we've reached the element we want to address
                    if(i == elemNumUnsigned)
                        return {structSizeUpToNow, element}; // TODO hope that there's no dangling reference to element here,  but i think its a reference, and the structParts is a reference itself, so it should be fine

                    unsigned elementAlignment = structType.isPacked() ? 1 : dl.getTypeABIAlignment(element);
                    // Add padding to the struct size to align it to the abi alignment of the
                    // element type before then adding the size of the element
                    structSizeUpToNow = llvm::alignTo(structSizeUpToNow, elementAlignment);
                    structSizeUpToNow += dl.getTypeSize(element);

                    // The alignment requirement of a struct is equal to the strictest alignment
                    // requirement of its elements.
                    structAlignment = std::max(elementAlignment, structAlignment);
                }
                llvm_unreachable("element index out of bounds");
            }else if(LLVMArrayType arrayType = type.template dyn_cast<LLVMArrayType>()){
                return {dl.getTypeSize(arrayType.getElementType()) * elemNum, arrayType.getElementType()};
            }else if(LLVMPointerType ptrType = type.template dyn_cast<LLVMPointerType>()){
                // in this case, we just always use the source element type, because there should only ever be one index in this case
                assert(op.getIndices().size() == 1 && "only one index is supported for pointer element types");
                return {dl.getTypeSize(op.getSourceElementType()) * elemNum, mlir::Type()};
            }else if(IntegerType intType = type.template dyn_cast<IntegerType>()){
                // in this case, we just always use the source element type, because there should only ever be one index in this case
                assert(op.getIndices().size() == 1 && "only one index is supported for int element types");
                return {dl.getTypeSize(intType) * elemNum, mlir::Type()};
            }else{
                op.dump();
                type.dump();
                llvm_unreachable("unhandled type");
            }
        };

        // there is no adaptor.getIndices(), the adaptor only gives access to the dynamic indices. so we iterate over all of the indices, and if we find a dynamic one, use the rewriter to remap it
        auto indices = op.getIndices();
        // TODO check for allocas to optimize if possible
        auto currentIndexComputationValue = adaptor.getBase();

        // TODO the other case is some weird vector thing, i'd rather have it fail for now, if that is encountered
        assert(op.getElemType().has_value());
        mlir::Type currentType = *op.getElemType();

        for(auto indexPtr_u : indices){
            assert(getTypeConverter()->convertType(currentIndexComputationValue.getType()) == amd64::gpr64Type::get(getContext()) && "only 64 bit pointers are supported");

            if(mlir::Value val = indexPtr_u.dyn_cast<mlir::Value>()){
                // no dynamic struct indices please
                assert(!mlir::isa<LLVM::LLVMStructType>(currentType) && "dynamic struct indices are not allowed, this should be fixed in the verification of GEP in the llvm dialect!");

                auto scaled = rewriter.create<amd64::IMUL64rri>(op.getLoc(), rewriter.getRemappedValue(val));

                // we perform the computation analogously, but just for ptr/array types, so use 1 as the index
                std::tie(scaled.instructionInfo().imm, currentType) = getBytesOffsetAndType(currentType, 1);
                currentIndexComputationValue = rewriter.create<amd64::ADD64rr>(op.getLoc(), currentIndexComputationValue, scaled);
            }else{
                // has to be integer attr otherwise
                auto indexInt = indexPtr_u.get<mlir::IntegerAttr>().getValue().getSExtValue();
                if(indexInt == 0)
                    continue;

                auto addri = rewriter.create<amd64::ADD64ri>(op.getLoc(), currentIndexComputationValue);
                std::tie(addri.instructionInfo().imm, currentType) = getBytesOffsetAndType(currentType, indexInt);
                currentIndexComputationValue = addri;
            }
        }

        rewriter.replaceOp(op, currentIndexComputationValue);
        return mlir::success();
    }
};

auto llvmLoadMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(LLVM::LoadOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    // TODO this is an ugly hack, because this op gets unrealized conversion casts as args (ptr in this case), because the ptr type gets converted to an i64, instead of a memloc, so the alloca returning a memloc doesn't work
    auto ptr = adaptor.getAddr();
    if(ptr.template isa<mlir::OpResult>() && mlir::isa<amd64::LEA64rm>(ptr.getDefiningOp()) && 
        mlir::isa<mlir::OpResult>(ptr.getDefiningOp()->getOperand(0))) /* && */ if(auto allocaOp = mlir::dyn_cast<amd64::AllocaOp>(ptr.getDefiningOp()->getOperand(0).getDefiningOp())){
            rewriter.replaceOpWithNewOp<INSTrm>(op, allocaOp);
            return mlir::success();
        }

    auto mem = rewriter.create<amd64::MemB>(op.getLoc(), ptr);
    rewriter.replaceOpWithNewOp<INSTrm>(op, mem);
    return mlir::success();
};

PATTERN(LLVMLoadPat, LLVM::LoadOp, amd64::MOV, llvmLoadMatchReplace);

auto llvmStoreMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(LLVM::StoreOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    // TODO this is an ugly hack, because this op gets unrealized conversion casts as args (ptr in this case), because the ptr type gets converted to an i64, instead of a memloc, so the alloca returning a memloc doesn't work
    auto ptr = adaptor.getAddr();
    auto val = adaptor.getValue();
    if(ptr.template isa<mlir::OpResult>() && mlir::isa<amd64::LEA64rm>(ptr.getDefiningOp()) && 
        mlir::isa<mlir::OpResult>(ptr.getDefiningOp()->getOperand(0))) /* && */ if(auto allocaOp = mlir::dyn_cast<amd64::AllocaOp>(ptr.getDefiningOp()->getOperand(0).getDefiningOp())){
        rewriter.replaceOpWithNewOp<INSTmr>(op, allocaOp, val);
        return mlir::success();
    }

    auto mem = rewriter.create<amd64::MemB>(op.getLoc(), ptr);
    rewriter.replaceOpWithNewOp<INSTmr>(op, mem, val);
    return mlir::success();
};

PATTERN(LLVMStorePat, LLVM::StoreOp, amd64::MOV, llvmStoreMatchReplace, 1, []<unsigned bitwidth, typename thisType, typename OpAdaptor>(thisType thiis, LLVM::StoreOp op, OpAdaptor, mlir::ConversionPatternRewriter& rewriter){ 
    if(mlir::dyn_cast<amd64::RegisterTypeInterface>(thiis->getTypeConverter()->convertType(op.getValue().getType())).getBitwidth() == bitwidth)
        return mlir::success();
    return rewriter.notifyMatchFailure(op, "bitwidth mismatch");
});

struct LLVMAllocaPat : public mlir::OpConversionPattern<mlir::LLVM::AllocaOp>{
    LLVMAllocaPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<LLVM::AllocaOp>(tc, ctx, 1){}

    mlir::LogicalResult matchAndRewrite(LLVM::AllocaOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        // TODO maybe this can be improved when considering that the alloca is only ever used as a ptr by GEP, load, store, and ptrtoint. In this case the lea is technically only needed for ptrtoint
        auto numElemsVal = op.getArraySize();
        auto numElems = mlir::cast<LLVM::ConstantOp>(numElemsVal.getDefiningOp()).getValue().cast<mlir::IntegerAttr>().getValue().getSExtValue();

        // TODO AllocaOp::print does this a bit differently -> use that?
        auto  dl = mlir::DataLayout::closest(op);
        assert(op.getElemType().has_value());
        auto elemSize = dl.getTypeSize(*op.getElemType());

        auto alloca = rewriter.create<amd64::AllocaOp>(op.getLoc(), elemSize*numElems);
        // TODO technically this is garbage, because we already know the exact number, but we need to distinguish between memloc and i64 types, normal ptr are i64, to enable arithmetic, but allocas have to give a memloc. I need to get it done, and this should work
        rewriter.replaceOpWithNewOp<amd64::LEA64rm>(op, alloca);

        return mlir::success();
    }
};

struct IntrinsicsPattern : public mlir::OpConversionPattern<LLVM::CallIntrinsicOp>{
    // TODO

};

auto llvmMovMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(LLVM::ConstantOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<INSTri>(op, adaptor.getValue().template cast<mlir::IntegerAttr>().getValue().getSExtValue());
    return mlir::success();
};
PATTERN(LLVMConstantIntPat, LLVM::ConstantOp, amd64::MOV, llvmMovMatchReplace, 2);

// TODO this is obviously not finished
struct LLVMConstantStringPat : public mlir::OpConversionPattern<LLVM::ConstantOp>{
    LLVMConstantStringPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<LLVM::ConstantOp>(tc, ctx, 1){}

    mlir::LogicalResult matchAndRewrite(LLVM::ConstantOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        if(op.use_empty()){
            rewriter.eraseOp(op);
            return mlir::success();
        }

        return mlir::failure();
    }
};

struct LLVMAddrofPat : public mlir::OpConversionPattern<LLVM::AddressOfOp>{
    LLVMAddrofPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<LLVM::AddressOfOp>(tc, ctx, 1){}

    mlir::LogicalResult matchAndRewrite(LLVM::AddressOfOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        auto global = mlir::SymbolTable::lookupNearestSymbolFrom(op, adaptor.getGlobalNameAttr());
        if(auto globalOp = mlir::dyn_cast<LLVM::GlobalOp>(global)){
            // we don't necessarily know the offset of the global yet, might need to be resolved later
            // TODO maybe pass globals and check if it's in there already to avoid making the extra op. Might also be slower, not sure
            rewriter.replaceOpWithNewOp<amd64::AddrOfGlobal>(op, adaptor.getGlobalNameAttr());
            return mlir::success();
        }else if(auto func = mlir::dyn_cast<LLVM::LLVMFuncOp>(global)){
            rewriter.replaceOpWithNewOp<amd64::MOV64ri>(op, (intptr_t) checked_dlsym(func.getName()));
            return mlir::success();
        }
        llvm_unreachable("TODO what?");
    }
};

auto llvmReturnMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(LLVM::ReturnOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    // TODO arg is optional, handle it not existing
    rewriter.replaceOpWithNewOp<INSTrr>(op, adaptor.getArg());
    return mlir::success();
};

// probably wrong
using LLVMReturnPat = MatchRMI<LLVM::ReturnOp, 64, llvmReturnMatchReplace, amd64::RET, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, ignoreBitwidthMatchLambda>;

struct LLVMFuncPat : public mlir::OpConversionPattern<LLVM::LLVMFuncOp>{
    LLVMFuncPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<LLVM::LLVMFuncOp>(tc, ctx, 1){}

    mlir::LogicalResult matchAndRewrite(LLVM::LLVMFuncOp func, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        auto llvmFnType = func.getFunctionType();

        llvm::SmallVector<mlir::Type> convertedArgTypes;
        auto res = getTypeConverter()->convertTypes(llvmFnType.getParams(), convertedArgTypes);
        if(mlir::failed(res))
            return rewriter.notifyMatchFailure(func, "failed to convert arg types");

        mlir::Type convertedReturnType{};
        if(auto retType = llvmFnType.getReturnType(); retType != LLVM::LLVMVoidType::get(func.getContext())){
            convertedReturnType = getTypeConverter()->convertType(retType);
            if(!convertedReturnType)
                return rewriter.notifyMatchFailure(func, "failed to convert return type");
        }

        auto newFunc = rewriter.create<mlir::func::FuncOp>(
            func.getLoc(), func.getName(),
            rewriter.getFunctionType(convertedArgTypes, convertedReturnType == mlir::Type{} ? TypeRange() : TypeRange(convertedReturnType)),
            rewriter.getStringAttr(/* this is apparently a different kind of visibility: LLVM::stringifyVisibility(adaptor.getVisibility_()) */ "private"),
            adaptor.getArgAttrsAttr(),
            adaptor.getResAttrsAttr());
        rewriter.inlineRegionBefore(func.getRegion(), newFunc.getRegion(), newFunc.getRegion().end());
        if (failed(rewriter.convertRegionTypes(&newFunc.getRegion(), *getTypeConverter())))
            return rewriter.notifyMatchFailure(func, "failed to convert region types");

        rewriter.replaceOp(func, newFunc->getResults());
        return mlir::success();
    }
};

struct LLVMGlobalPat : public mlir::OpConversionPattern<LLVM::GlobalOp>{
    // TODO instead of this, pass the buffer pointer and write there immediately, then make use a GlobalsInfo map that simply maps to that address
    GlobalsInfo& globals;

    LLVMGlobalPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx, GlobalsInfo& globals) : mlir::OpConversionPattern<LLVM::GlobalOp>(tc, ctx, 1), globals(globals){ }

    mlir::LogicalResult matchAndRewrite(LLVM::GlobalOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        // if its a declaration: handle specially, we can already look up the address of the symbol and write it there, or fail immediately
        GlobalSymbolInfo unfinishedGlobal;
        intptr_t& addr = unfinishedGlobal.addrInDataSection = (intptr_t) nullptr;
        auto& bytes = unfinishedGlobal.bytes = {};

        auto symbol = op.getSymName();
        if(op.isDeclaration()){
            // this address is allowed to be 0, checked_dlsym handles an actual error through dlerror()
            DEBUGLOG("external symbol: " << symbol << ", getting address from environment");
            addr = (intptr_t) checked_dlsym(symbol);
        }

        // TODO globals with initalization region
        if(auto initBlock = op.getInitializerBlock()){
            auto fail = [&](){
                return rewriter.notifyMatchFailure(op, "globals with complex initializer blocks are not supported yet");
            };

            if(initBlock->getOperations().size() == 2){
                // TODO dyn cast operand as ptr?
                auto llvmNull = mlir::dyn_cast<LLVM::NullOp>(initBlock->getOperations().front());
                auto ret = mlir::dyn_cast<LLVM::ReturnOp>(initBlock->getOperations().back());
                if(!llvmNull || !ret || ret.getOperands().size() != 1 || ret.getOperand(0) != llvmNull)
                    return fail();

                auto byteSize = sizeof(intptr_t);
                bytes.resize(byteSize);
                auto nullptr_ = nullptr;
                assert(byteSize == sizeof(nullptr_) && "nullptr is not the same size as a pointer");
                assert(memcmp(bytes.data(), &nullptr_, byteSize) == 0 && "nullptr is not all 0s");
                unfinishedGlobal.alignment = byteSize;
                goto finish; // TODO make nicer with a lambda or smth
            }

            return fail();
        }


        // get raw bytes of the value of this global
        // heavily inspired by ModuleTranslation::convertGlobals/LLVM::detail:getLLVMConstant
        // TODO pointer type globals
        if(auto attr = op.getValueOrNull()){
            // TODO maybe try to optimize the ordering?
            if(auto denseElementsAttr = attr.dyn_cast<mlir::DenseElementsAttr>()){
                auto rawData = denseElementsAttr.getRawData();
                bytes.resize(rawData.size());
                memcpy(bytes.data(), rawData.data(), rawData.size());
            }else if(auto strAttr = attr.dyn_cast<mlir::StringAttr>()){
                // TODO is this guaranteed to be null terminated, if it comes from a global?
                bytes.resize(strAttr.getValue().size());
                memcpy(bytes.data(), strAttr.getValue().data(), strAttr.getValue().size());
                assert(unfinishedGlobal.bytes.back() == '\0');
            }else if(auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()){
                auto size = intAttr.getType().getIntOrFloatBitWidth()/8;
                bytes.resize(size);
                memcpyToLittleEndianBuffer(bytes.data(), intAttr.getValue().getSExtValue(), size);
            }else if(auto floatAttr = attr.dyn_cast<mlir::FloatAttr>()){
                op.dump();
                EXIT_TODO;
            }else if(auto funcAttr = attr.dyn_cast<mlir::FlatSymbolRefAttr>()){
                op.dump();
                EXIT_TODO;
            }else{
                op.dump();
                llvm_unreachable("sad");
            }
        }else{
            DEBUGLOG("external symbol: " << symbol << ", getting address from environment");
            addr = (intptr_t) checked_dlsym(symbol);
        }

    finish:
        unfinishedGlobal.alignment = static_cast<unsigned>(adaptor.getAlignment().value_or(mlir::DataLayout::closest(op).getTypeSize(adaptor.getGlobalType())));

        // TODO handle visibility
        globals.insert({symbol, std::move(unfinishedGlobal)});
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

#define CALL_PAT(bitwidth) \
    using LLVMCallPat ## bitwidth = MatchRMI<LLVM::CallOp, bitwidth, callMatchReplace, amd64::CALL, amd64::MOV ## bitwidth ## rr, amd64::gpr ## bitwidth ## Type, NOT_AVAILABLE, NOT_AVAILABLE, 1, defaultBitwidthMatchLambda<mlir::LLVM::CallOp, true>>

CALL_PAT(8); CALL_PAT(16); CALL_PAT(32); CALL_PAT(64);

#undef CALL_PAT

PATTERN(LLVMICmpPat, LLVM::ICmpOp, amd64::CMP, cmpIMatchReplace<LLVM::ICmpPredicate>, 1, cmpIBitwidthMatcher);

// TODO order the LLVM patterns in a more structured way

using LLVMBrPat = MatchRMI<LLVM::BrOp, 64, branchMatchReplace, amd64::JMP, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, ignoreBitwidthMatchLambda>;

struct LLVMCondBrPat : public mlir::OpConversionPattern<LLVM::CondBrOp> {
    LLVMCondBrPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<LLVM::CondBrOp>(tc, ctx, 3){}

    mlir::LogicalResult matchAndRewrite(LLVM::CondBrOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        return condBrMatchReplace<LLVM::ICmpOp, decltype(this), OpAdaptor>(this, op, adaptor, rewriter);
    }
};

struct CaseInfo{
    int64_t comparisonValue;
    mlir::Block* block;
    mlir::OperandRange operands;
};

// TODO maybe do this non-recursively at some point, but that's too annoying for now
template<typename CMPri>
void binarySearchSwitchLowering(mlir::Location loc, mlir::ConversionPatternRewriter& rewriter, mlir::Value adaptedValue, CaseInfo defaultDest, size_t pivotIndex, std::span<CaseInfo> caseInfoSection){
    DEBUGLOG("binarySearchSwitchLowering: pivotIndex = " << pivotIndex << ", caseInfoSection.size() = " << caseInfoSection.size());
    auto* currentBlock = rewriter.getInsertionBlock();

    // TODO recursion end condition 1
    if(caseInfoSection.empty() || pivotIndex >= caseInfoSection.size()){
        // jump to default block on empty case section
        rewriter.replaceAllUsesWith(currentBlock, defaultDest.block);
        rewriter.eraseBlock(currentBlock);
        return;
    }

    auto pivotInfo = caseInfoSection[pivotIndex];

    // if the value is equal, jump to the block
    auto cmp = rewriter.create<CMPri>(loc, adaptedValue);
    cmp.instructionInfo().imm = pivotInfo.comparisonValue;

    auto createBlock = [&](mlir::Block* insertAfter){
        // make a new block, but don't let the rewriter do it, because it would switch to it immediately, we want to do that manually (this is akin to OpBuilder::createBlock)
        auto* newBlock = new mlir::Block();
        // no arguments to this block
        // insert it after the current block
        currentBlock->getParent()->getBlocks().insertAfter(insertAfter->getIterator(), newBlock); // TODO the block ordering of this is no quite right yet
        rewriter.notifyBlockCreated(newBlock);
        return newBlock;
    };

    // if the size is 1, we can simply jump to the block or the default block, recursion end condition 2
    if(caseInfoSection.size() == 1){
        rewriter.create<amd64::JE>(loc, pivotInfo.operands, defaultDest.operands, pivotInfo.block, defaultDest.block);
        return;
    }

    auto* searchLowerOrUpperHalf = createBlock(currentBlock);
    auto* searchLowerHalf = createBlock(searchLowerOrUpperHalf); // TODO probably create these immediately before the recursive call, to optimize the block ordering
    auto* searchUpperHalf = createBlock(searchLowerHalf);

    rewriter.create<amd64::JE>(loc, pivotInfo.operands, mlir::ValueRange(), pivotInfo.block, searchLowerOrUpperHalf);

    // if the value is less than the pivot, search the lower half, otherwise search the upper half
    rewriter.setInsertionPointToEnd(searchLowerOrUpperHalf);
    rewriter.create<amd64::JL>(loc, mlir::ValueRange(), mlir::ValueRange(), searchLowerHalf, searchUpperHalf);

    rewriter.setInsertionPointToEnd(searchLowerHalf);
    binarySearchSwitchLowering<CMPri>(loc, rewriter, adaptedValue, defaultDest, pivotIndex/2, caseInfoSection.first(pivotIndex));

    rewriter.setInsertionPointToEnd(searchUpperHalf);
    // TODO check that this is modmod  done using a bitwise and with 1
    binarySearchSwitchLowering<CMPri>(loc, rewriter, adaptedValue, defaultDest, (pivotIndex % 2 == 1) ? (pivotIndex/2) : (pivotIndex/2 - 1) , caseInfoSection.last(caseInfoSection.size() - pivotIndex - 1 /* leave out the pivot itself */));
}


// TODO extend this to mlir.cf.switch
auto switchMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename CMPrr, typename CMPri, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE, typename = NOT_AVAILABLE
     >(LLVM::SwitchOp op, LLVM::SwitchOp::Adaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {

    auto caseValuesOpt = op.getCaseValues();
    if(!caseValuesOpt.has_value())
        return rewriter.notifyMatchFailure(op, "switches without case values are not supported yet");

    // TODO case operands/block args

    // lower the switch using binary search, so sort the cases

    auto caseValues = mlir::cast<mlir::DenseIntElementsAttr>(*caseValuesOpt);
    // TODO would be nicer to have the right int type here, maybe template arg this?
    llvm::SmallVector<CaseInfo, 8> caseValuesIntSorted;
    caseValuesIntSorted.reserve(caseValues.size());
    for(auto [caseValue, block, operands] : llvm::zip(caseValues, op.getCaseDestinations(), op.getCaseOperands()))
        caseValuesIntSorted.push_back({caseValue.getSExtValue(), block, operands});

    // TODO this might be super slow
    std::sort(caseValuesIntSorted.begin(), caseValuesIntSorted.end(), [](auto a, auto b){return a.comparisonValue < b.comparisonValue;});

    // now do the actual binary search
    auto pivotIndex = caseValuesIntSorted.size()/2;

    auto defaultCase = op.getDefaultDestination();
    assert(defaultCase && "switches without default cases are not allowed (I think)");
    binarySearchSwitchLowering<CMPri>(op->getLoc(), rewriter, adaptor.getValue(), {0, defaultCase, op.getDefaultOperands()}, pivotIndex, caseValuesIntSorted);

    rewriter.eraseOp(op);

    return mlir::success();
};

PATTERN(LLVMSwitchPat, LLVM::SwitchOp, amd64::CMP, switchMatchReplace, 1, /* bitwidth matcher: basically default, but we cant use the result */ []<unsigned bitwidth, typename thisType, typename OpAdaptor>(thisType thiis, LLVM::SwitchOp op, OpAdaptor, mlir::ConversionPatternRewriter& rewriter){
    // TODO this is almost the same as defaultBitwidthMatchLambda, try to merge them
    mlir::Type opType = op.getValue().getType();

    auto type = thiis->getTypeConverter()->convertType(opType);
    // TODO might be slow
    if(!type)
        return rewriter.notifyMatchFailure(op, "type conversion failed");

    auto typeToMatch= type.template dyn_cast<amd64::RegisterTypeInterface>();
    //assert(typeToMatch && "expected register type");
    if(!typeToMatch)
        return rewriter.notifyMatchFailure(op, "expected register type");
    // TODO this assertion currently fails wrongly on a conditional branch
    //assert((op->getNumOperands() == 0 || typeToMatch == thiis->getTypeConverter()->convertType(op->getOperand(0).getType()).template dyn_cast<amd64::RegisterTypeInterface>()) && "this simple bitwidth matcher assumes that the type of the op and the type of the operands are the same");

    if(typeToMatch.getBitwidth() != bitwidth)
        return rewriter.notifyMatchFailure(op, "bitwidth mismatch");

    return mlir::success();
});

// Arithmetic stuff

auto llvmGetIn = [](auto adaptorOrOp){ return adaptorOrOp.getArg(); };
auto llvmGetOut = [](auto adaptorOrOp){ return adaptorOrOp.getRes(); };

// LLVM SExt/ZExt/Trunc patterns, same as MLIR above, read up there on why this is divided into these weird cases
#define SZEXT_PAT(outBitwidth, inBitwidth) \
    using LLVMZExtPat ## outBitwidth ## _ ## inBitwidth = MatchRMI<LLVM::ZExtOp, outBitwidth, truncExtUiSiMatchReplace<inBitwidth, llvmGetIn, llvmGetOut>, amd64::MOVZX ## r ## outBitwidth ## r ## inBitwidth, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<inBitwidth, llvmGetIn, llvmGetOut>>; \
    using LLVMSExtPat ## outBitwidth ## _ ## inBitwidth = MatchRMI<LLVM::SExtOp, outBitwidth, truncExtUiSiMatchReplace<inBitwidth, llvmGetIn, llvmGetOut>, amd64::MOVSX ## r ## outBitwidth ## r ## inBitwidth, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<inBitwidth, llvmGetIn, llvmGetOut>>;

// generalizable cases:
SZEXT_PAT(16, 8);
SZEXT_PAT(32, 8); SZEXT_PAT(32, 16);
SZEXT_PAT(64, 8); SZEXT_PAT(64, 16);

#undef SZEXT_PAT

using LLVMZExtPat64_32 = MatchRMI<LLVM::ZExtOp, 64, truncExtUiSiMatchReplace<32, llvmGetIn, llvmGetOut>, amd64::MOV32rr, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<32, llvmGetIn, llvmGetOut>>;
using LLVMSExtPat64_32 = MatchRMI<LLVM::SExtOp, 64, truncExtUiSiMatchReplace<32, llvmGetIn, llvmGetOut>, amd64::MOVSXr64r32, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<32, llvmGetIn, llvmGetOut>>;
// trunc
#define TRUNC_PAT(outBitwidth, inBitwidth) \
    using LLVMTruncPat ## outBitwidth ## _ ## inBitwidth = MatchRMI<LLVM::TruncOp, outBitwidth, truncExtUiSiMatchReplace<inBitwidth, llvmGetIn, llvmGetOut>, amd64::MOV ## outBitwidth ## rr, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, 1, truncExtUiSiBitwidthMatcher<inBitwidth, llvmGetIn, llvmGetOut>>;
TRUNC_PAT(8, 16); TRUNC_PAT(8, 32); TRUNC_PAT(8, 64);
TRUNC_PAT(16, 32); TRUNC_PAT(16, 64);
TRUNC_PAT(32, 64);

#undef TRUNC_PAT

PATTERN(LLVMAddPat, LLVM::AddOp, amd64::ADD, binOpAndImmMatchReplace, 2);
PATTERN(LLVMSubPat, LLVM::SubOp, amd64::SUB, binOpMatchReplace);
PATTERN(LLVMAndPat, LLVM::AndOp, amd64::AND, binOpMatchReplace);
PATTERN(LLVMOrPat,  LLVM::OrOp,  amd64::OR,  binOpMatchReplace);
PATTERN(LLVMXOrPat, LLVM::XOrOp, amd64::XOR, binOpMatchReplace);

template<typename OpTy, auto matchAndRewriteInject, unsigned benefit = 1>
struct SimplePat : public mlir::OpConversionPattern<OpTy>{
    using OpAdaptor = typename mlir::OpConversionPattern<OpTy>::OpAdaptor;

    SimplePat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<OpTy>(tc, ctx, benefit){}

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        return matchAndRewriteInject(op, adaptor, rewriter);
    }
};

using LLVMNullPat = SimplePat<LLVM::NullOp, [](auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter){
    rewriter.replaceOpWithNewOp<amd64::MOV64ri>(op, (intptr_t) nullptr);
    return mlir::success();
}>;

auto zeroReplace = [](auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter){
    // TODO warn about this
    rewriter.replaceOpWithNewOp<amd64::MOV64ri>(op, 0);
    return mlir::success();
};

using LLVMUndefPat = SimplePat<LLVM::UndefOp, zeroReplace>;
using LLVMPoisonPat = SimplePat<LLVM::PoisonOp, zeroReplace>;

auto ptrIntReplace =  [](auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter){
    // TODO assert pointer/int bitwidths are 64
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return mlir::success();
};
using LLVMIntToPtrPat = SimplePat<LLVM::IntToPtrOp, ptrIntReplace>;
using LLVMPtrToIntPat = SimplePat<LLVM::PtrToIntOp, ptrIntReplace>;
using LLVMEraseMetadataPat = SimplePat<LLVM::MetadataOp, [](auto op, auto, mlir::ConversionPatternRewriter& rewriter){
    rewriter.eraseOp(op);
    return mlir::success();
}>;


} // end anonymous namespace


void populateLLVMToAMD64TypeConversions(mlir::TypeConverter& tc){
    tc.addConversion([](mlir::LLVM::LLVMPointerType type) -> std::optional<mlir::Type>{
        return amd64::gpr64Type::get(type.getContext());
    });
    // TODO
}

void populateDefaultTypesToAMD64TypeConversions(mlir::TypeConverter& tc){
    tc.addConversion([](mlir::IntegerType type) -> std::optional<mlir::Type>{
        switch(type.getIntOrFloatBitWidth()) {
            // cmp is not matched using the result type (always i1), but with the operand type, so this doesn't apply there.
            case 1:  // TODO  Internally, MLIR seems to need to convert i1's though, so we will handle them as i8 for now, also because SETcc returns i8.
            case 8:  return amd64::gpr8Type ::get(type.getContext());
            case 16: return amd64::gpr16Type::get(type.getContext());
            case 32: return amd64::gpr32Type::get(type.getContext());
            case 64: return amd64::gpr64Type::get(type.getContext());

            default: llvm_unreachable("unhandled bitwidth in tc");
        }
    });

    // all register and memloc types are already legal
    // TODO this is probably not needed in the end, because we shouldn't encounter them in the first place. But for now with manually created ops it is needed.
    tc.addConversion([](amd64::RegisterTypeInterface type) -> std::optional<mlir::Type>{
        return type;
    });
    tc.addConversion([](amd64::memLocType type) -> std::optional<mlir::Type>{
        return type;
    });
    // TODO this might be entirely stupid
    tc.addConversion([&tc](mlir::FunctionType functionType) -> std::optional<mlir::Type>{
        llvm::SmallVector<mlir::Type, 6> inputs;
        llvm::SmallVector<mlir::Type, 1> results;
        if(mlir::failed(tc.convertTypes(functionType.getInputs(), inputs))
                || mlir::failed(tc.convertTypes(functionType.getResults(), results)))
            return {};

        return mlir::FunctionType::get(functionType.getContext(), 
            inputs, results
        );
    });
}

void populateArithToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc){
    auto* ctx = patterns.getContext();
#define PATTERN_BITWIDTHS(patternName) patternName ## 8, patternName ## 16, patternName ## 32, patternName ## 64
    // TODO even more fine grained control over which patterns to use
    patterns.add<
        PATTERN_BITWIDTHS(ConstantIntPat),
        PATTERN_BITWIDTHS(AddIPat),
        PATTERN_BITWIDTHS(SubIPat),
        PATTERN_BITWIDTHS(MulIPat),
        PATTERN_BITWIDTHS(CmpIPat),
        PATTERN_BITWIDTHS(AndIPat),
        PATTERN_BITWIDTHS(OrIPat),
        PATTERN_BITWIDTHS(XOrIPat),
        PATTERN_BITWIDTHS(MulIPat),
        PATTERN_BITWIDTHS(DivUIPat),
        PATTERN_BITWIDTHS(DivSIPat),
        PATTERN_BITWIDTHS(RemSIPat),
        PATTERN_BITWIDTHS(RemSIPat),
        PATTERN_BITWIDTHS(ShlIPat),
        PATTERN_BITWIDTHS(ShrSIPat),
        PATTERN_BITWIDTHS(ShrUIPat),
        ExtUII1Pat,
        ExtUIPat16_8, ExtUIPat32_8, ExtUIPat64_8, ExtUIPat32_16, ExtUIPat64_16, ExtUIPat64_32,
        ExtSIPat16_8, ExtSIPat32_8, ExtSIPat64_8, ExtSIPat32_16, ExtSIPat64_16, ExtSIPat64_32,
        TruncPat8_16, TruncPat8_32, TruncPat8_64, TruncPat16_32, TruncPat16_64, TruncPat32_64
    >(tc, ctx);

    //patterns.add<TestConvPatternWOOp>(tc, ctx);
}

void populateCFToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc){
    auto* ctx = patterns.getContext();
    patterns.add<BrPat, CondBrPat, ReturnPat>(tc, ctx);
}

void populateFuncToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc){
    auto* ctx = patterns.getContext();
    patterns.add<PATTERN_BITWIDTHS(CallPat)>(tc, ctx);
    mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, tc);
}

void populateLLVMToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc, GlobalsInfo& globals){
    patterns.add<
        PATTERN_BITWIDTHS(LLVMConstantIntPat),
        PATTERN_BITWIDTHS(LLVMAddPat),
        PATTERN_BITWIDTHS(LLVMSubPat),
        PATTERN_BITWIDTHS(LLVMAndPat),
        PATTERN_BITWIDTHS(LLVMOrPat),
        PATTERN_BITWIDTHS(LLVMXOrPat),
        LLVMNullPat, LLVMUndefPat, LLVMPoisonPat, LLVMEraseMetadataPat,
        LLVMGEPPattern, LLVMAllocaPat, PATTERN_BITWIDTHS(LLVMLoadPat), PATTERN_BITWIDTHS(LLVMStorePat), LLVMPtrToIntPat, LLVMIntToPtrPat,
        LLVMReturnPat, LLVMFuncPat,
        LLVMConstantStringPat, LLVMAddrofPat,
        PATTERN_BITWIDTHS(LLVMCallPat),
        LLVMBrPat, LLVMCondBrPat, PATTERN_BITWIDTHS(LLVMSwitchPat),
        PATTERN_BITWIDTHS(LLVMICmpPat),
        LLVMZExtPat16_8,  LLVMZExtPat32_8,  LLVMZExtPat64_8,  LLVMZExtPat32_16,  LLVMZExtPat64_16,  LLVMZExtPat64_32,
        LLVMSExtPat16_8,  LLVMSExtPat32_8,  LLVMSExtPat64_8,  LLVMSExtPat32_16,  LLVMSExtPat64_16,  LLVMSExtPat64_32,
        LLVMTruncPat8_16, LLVMTruncPat8_32, LLVMTruncPat8_64, LLVMTruncPat16_32, LLVMTruncPat16_64, LLVMTruncPat32_64
    >(tc, patterns.getContext());
    patterns.add<LLVMGlobalPat>(tc, patterns.getContext(), globals);
}


void populateAnyKnownAMD64TypeConversionsConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc, GlobalsInfo& globals){
    //populateWithGenerated(patterns);
    populateDefaultTypesToAMD64TypeConversions(tc);
    populateArithToAMD64ConversionPatterns(patterns, tc);
    populateFuncToAMD64ConversionPatterns(patterns, tc);
    populateCFToAMD64ConversionPatterns(patterns, tc);
    populateLLVMToAMD64TypeConversions(tc);
    populateLLVMToAMD64ConversionPatterns(patterns, tc, globals);
}
#undef PATTERN_BITWIDTHS

/// takes an operation and does isel on its regions using all available type conversions and conversion patterns
/// use the populate* functions, a custom rewrite pattern set and type converter, and the `isel` function to customize the pattern set and isel to your liking
[[nodiscard("You should check whether ISel succeeded!")]] bool maximalIsel(mlir::Operation* regionOp, GlobalsInfo& globals){
    mlir::RewritePatternSet patterns(regionOp->getContext());
    mlir::TypeConverter typeConverter;
    populateAnyKnownAMD64TypeConversionsConversionPatterns(patterns, typeConverter, globals);
    return isel(regionOp, typeConverter, patterns);
}

/// takes an operation and does isel on its regions
/// if you want to use all known type and pattern conversions, use `maximalIsel`
/// use the populate* functions to customize the pattern set to your liking
[[nodiscard("You should check whether ISel succeeded!")]] bool isel(mlir::Operation* regionOp, mlir::TypeConverter& typeConverter, mlir::RewritePatternSet& patterns){
    using namespace amd64;
    using namespace mlir::arith;

    auto* ctx = regionOp->getContext();

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<amd64::AMD64Dialect>();

    // from https://discourse.llvm.org/t/lowering-and-type-conversion-of-block-arguments/63570
    // funcs are only legal once their args have been converted
    // TODO if this costs a lot of performance, try defining an amd64::FuncOp and make all of func illegal
    // TODO can these be put in any of the populate* functions?
    target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) && typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<mlir::LLVM::LLVMFuncOp>([&](mlir::LLVM::LLVMFuncOp op) {
        return typeConverter.isLegal(op.getFunctionType().getParams()) && typeConverter.isLegal(op.getFunctionType().getReturnType()) && typeConverter.isLegal(&op.getBody());
    });
    
    //llvm::setCurrentDebugType("greedy-rewriter");
    //llvm::setCurrentDebugType("dialect-conversion");

    // TODO try to get greedy driver to work
    //auto result = mlir::applyPatternsAndFoldGreedily(patternMatchingTestFn, std::move(patterns)); // TODO I think this only applies normal rewrite patterns, not conversion patterns...
                                                                                                    // TODO actually no, i think this is a problem with this driver only applying patterns to the top level ops in the region
    // TODO try to do this concurrently, for function regions

    //mlir::applyOpPatternsAndFold({range.begin(), range.end()}, std::move(patterns));
    return mlir::failed(mlir::applyFullConversion(regionOp, target, std::move(patterns)));
}
