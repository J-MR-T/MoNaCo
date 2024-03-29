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

// to let llvm handle translating globals
#include <llvm/IR/IRBuilder.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

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
using NA = void;

// std::derived_from does not work, because interfaces that inherit from each other don't *actually* inherit anything in Cpp, they just have conversion operators
template<typename Derived, typename Base>
concept MLIRInterfaceDerivedFrom = requires(Derived d){
    { d } -> std::convertible_to<Base>;
};

template <MLIRInterfaceDerivedFrom<amd64::RegisterTypeInterface> RegisterTy, bool matchZeroResult = false>
auto defaultBitwidthMatchLambda = []<unsigned bitwidth>(auto thiis, auto op, typename mlir::OpConversionPattern<decltype(op)>::OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter){
    using OpTy = decltype(op);

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
    if(!type)
        return rewriter.notifyMatchFailure(op, "type conversion failed");

    auto typeToMatch= type.template dyn_cast<RegisterTy>();
    if(!typeToMatch)
        return rewriter.notifyMatchFailure(op, "not the correct register type");

    if(typeToMatch.getBitwidth() != bitwidth)
        return rewriter.notifyMatchFailure(op, "bitwidth mismatch");

    return mlir::success();
};

auto intOrFloatBitwidthMatchLambda = defaultBitwidthMatchLambda<amd64::RegisterTypeInterface>;
auto intBitwidthMatchLambda = defaultBitwidthMatchLambda<amd64::GPRegisterTypeInterface>;
auto floatBitwidthMatchLambda = defaultBitwidthMatchLambda<amd64::FPRegisterTypeInterface>;

auto matchAllLambda = []<unsigned>(auto, auto, auto, mlir::ConversionPatternRewriter&){
    return mlir::success();
};

template<typename T, typename OpTy, typename AdaptorTy>
concept MatchReplaceLambda = requires(T lambda, OpTy op, AdaptorTy adaptor, mlir::ConversionPatternRewriter& rewriter){
    { lambda.template operator()<8, NA, NA, NA, NA, NA>(op, adaptor, rewriter) } -> std::convertible_to<mlir::LogicalResult>;
};

/// somewhat generic pattern matching struct
template<
    typename OpTy,
    unsigned bitwidth,
    auto lambda,
    // default template parameters start
    // if i specify the default arg inline, that instantly crashes clangd. But using a separate variable reduces code duplication anyway, so thanks I guess?
    auto bitwidthMatchLambda = intBitwidthMatchLambda,
    int benefit = 1,
    typename INST1 = NA, typename INST2 = NA, typename INST3 = NA, typename INST4 = NA, typename INST5 = NA
>
//requires MatchReplaceLambda<decltype(lambda), OpTy, typename mlir::OpConversionPattern<OpTy>::OpAdaptor>
struct Match : public mlir::OpConversionPattern<OpTy>{
    Match(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<OpTy>(tc, ctx, benefit){}
    using OpAdaptor = typename mlir::OpConversionPattern<OpTy>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        auto bitwidthMatchResult = bitwidthMatchLambda.template operator()<bitwidth>(this, op, adaptor, rewriter);
        if(bitwidthMatchResult.failed())
            return bitwidthMatchResult;

        return lambda.template operator()<bitwidth, INST1, INST2, INST3, INST4, INST5>(op, adaptor, rewriter);
    }
};

#define PATTERN_FOR_BITWIDTH_INT_MRI(bitwidth, patternName, OpTy, opPrefixToReplaceWith, lambda, ...) \
    using patternName ##  bitwidth = Match<OpTy,  bitwidth,  lambda, ## __VA_ARGS__,                  \
        opPrefixToReplaceWith ##  bitwidth ## rr, opPrefixToReplaceWith ##  bitwidth ## ri, opPrefixToReplaceWith ##  bitwidth ## rm, opPrefixToReplaceWith ##  bitwidth ## mi, opPrefixToReplaceWith ##  bitwidth ## mr>;

#define PATTERN_INT_1(patternName, opTy, opPrefixToReplaceWith, lambda, matchLambda, benefit)                \
    PATTERN_FOR_BITWIDTH_INT_MRI(8,  patternName, opTy, opPrefixToReplaceWith, lambda, matchLambda, benefit) \
    PATTERN_FOR_BITWIDTH_INT_MRI(16, patternName, opTy, opPrefixToReplaceWith, lambda, matchLambda, benefit) \
    PATTERN_FOR_BITWIDTH_INT_MRI(32, patternName, opTy, opPrefixToReplaceWith, lambda, matchLambda, benefit) \
    PATTERN_FOR_BITWIDTH_INT_MRI(64, patternName, opTy, opPrefixToReplaceWith, lambda, matchLambda, benefit)

#define PATTERN_INT_2(patternName, opTy, opPrefixToReplaceWith, lambda, matchLambda) \
    PATTERN_INT_1(patternName, opTy, opPrefixToReplaceWith, lambda, matchLambda, 1)

#define PATTERN_INT_3(patternName, opTy, opPrefixToReplaceWith, lambda) \
    PATTERN_INT_2(patternName, opTy, opPrefixToReplaceWith, lambda, intBitwidthMatchLambda)

#define GET_MACRO(_1, _2, _3, _4, _5, _6, name, ...) name
// default args for the macros
#define PATTERN_INT(...) GET_MACRO(__VA_ARGS__, PATTERN_INT_1, PATTERN_INT_2, PATTERN_INT_3)(__VA_ARGS__)

template<typename OpTy, auto matchAndRewriteInject, unsigned benefit = 1>
struct SimplePat : public mlir::OpConversionPattern<OpTy>{
    using OpAdaptor = typename mlir::OpConversionPattern<OpTy>::OpAdaptor;

    SimplePat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<OpTy>(tc, ctx, benefit){}

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        return matchAndRewriteInject(op, adaptor, rewriter);
    }
};

template<typename OpTy>
struct ErasePat : public mlir::OpConversionPattern<OpTy>{
    using OpAdaptor = typename mlir::OpConversionPattern<OpTy>::OpAdaptor;

    ErasePat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<OpTy>(tc, ctx, 1){}

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        rewriter.eraseOp(op);
        return mlir::success();
    }
};


// TODO technically I could replace the OpAdaptor template arg everywhere with a simple `auto adaptor` parameter
// TODO even better: use a typename T for the op, and then use T::Adaptor for the adaptor

// TODO an alternative would be to generate custom builders for the RR versions, which check if their argument is a movxxri and then fold it into the RR, resulting in an RI version. That probably wouldn't work because the returned thing would of course expect an RR version, not an RI version
auto binOpMatchReplace = []<unsigned actualBitwidth,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<INSTrr>(op, adaptor.getLhs(), adaptor.getRhs());
    return mlir::success();
};

// it would be nice to use folds for matching mov's and folding them into the add, but that's not possible right now, so we either have to match it here, or ignore it for now (binOpMatchReplace's approach)
// I finally found out when to use the OpAdaptor and when not to: The OpAdaptor seems to give access to the operands in their already converted form, whereas the op itself still has all operands in their original form.
// In this case we need to access the operand in the original form, to check if it was a constant, we're not interested in what it got converted to
auto binOpAndImmMatchReplace = []<unsigned actualBitwidth,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    // TODO change this so it can also match constants from the llvm dialect
    auto constantOp = mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(op.getLhs().getDefiningOp());
    auto other = adaptor.getRhs();

    if(!constantOp){
        constantOp = mlir::dyn_cast_or_null<mlir::arith::ConstantIntOp>(op.getRhs().getDefiningOp());
        other = adaptor.getLhs();
    }

    if(!constantOp ||
        // immediate is max 32 bit, otherwise we have to generate a mov for it
        fitsInto32BitImm(constantOp.value())
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
PATTERN_INT(AddIPat, arith::AddIOp, amd64::ADD, binOpAndImmMatchReplace, intBitwidthMatchLambda, 2);
PATTERN_INT(SubIPat, arith::SubIOp, amd64::SUB, binOpMatchReplace);
PATTERN_INT(AndIPat, arith::AndIOp, amd64::AND, binOpMatchReplace);
PATTERN_INT(OrIPat,  arith::OrIOp,  amd64::OR,  binOpMatchReplace);
PATTERN_INT(XOrIPat, arith::XOrIOp, amd64::XOR, binOpMatchReplace);

auto cmpIBitwidthMatcher =
    []<unsigned innerBitwidth>(auto thiis, auto op, auto, mlir::ConversionPatternRewriter&){
        // cmp always has i1 as a result type, so we need to match the arguments' bitwidths
        auto tc = thiis->getTypeConverter();
        auto lhsNewType = tc->convertType(op.getLhs().getType()).template dyn_cast<amd64::GPRegisterTypeInterface>(); // TODO maybe we can use the adaptor and just use .getType on that instead of bothering with the type converter. Not just here, but everywhere
        auto rhsNewType = tc->convertType(op.getRhs().getType()).template dyn_cast<amd64::GPRegisterTypeInterface>();

        assert(lhsNewType && rhsNewType && "cmp's operands should be convertible to register types");

        bool success = lhsNewType.getBitwidth() == innerBitwidth && lhsNewType == rhsNewType;
        return mlir::failure(!success);
    };
// TODO specify using benefits, that this has to have lower priority than the version which matches a jump with a cmp as an argument (which doesn't exist yet).
template<typename CmpPredicate>
auto cmpIMatchReplace = []<unsigned actualBitwidth,
     typename CMPrr, typename CMPri, typename, typename, typename
     >(auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {

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
PATTERN_INT(CmpIPat, mlir::arith::CmpIOp, amd64::CMP, cmpIMatchReplace<mlir::arith::CmpIPredicate>, cmpIBitwidthMatcher);

template <bool isRem>
auto matchDivRem = []<unsigned actualBitwidth,
     typename DIVr, typename, typename, typename, typename
     >(auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    auto div = rewriter.create<DIVr>(op->getLoc(), adaptor.getLhs(), adaptor.getRhs());
    if constexpr(isRem)
        rewriter.replaceOp(op, div.getRemainder());
    else
        rewriter.replaceOp(op, div.getQuotient());
    return mlir::success();
};

#define MULI_DIVI_PAT(bitwidth)                                                                                                                 \
    using MulIPat ## bitwidth = Match<mlir::arith::MulIOp, bitwidth, binOpMatchReplace, intBitwidthMatchLambda, 1, amd64::MUL ## bitwidth ## r>; \
    using DivUIPat ## bitwidth = Match<mlir::arith::DivUIOp, bitwidth,                                                                          \
        matchDivRem<false>, intBitwidthMatchLambda, 1,                                                                                          \
        amd64::DIV ## bitwidth ## r>;                                                                                                           \
    using DivSIPat ## bitwidth = Match<mlir::arith::DivSIOp, bitwidth,                                                                          \
        matchDivRem<false>,intBitwidthMatchLambda, 1,                                                                                           \
        amd64::IDIV ## bitwidth ## r>;                                                                                                          \
    using RemUIPat ## bitwidth = Match<mlir::arith::RemUIOp, bitwidth,                                                                          \
        matchDivRem<true>, intBitwidthMatchLambda, 1,                                                                                           \
        amd64::DIV ## bitwidth ## r>;                                                                                                           \
    using RemSIPat ## bitwidth = Match<mlir::arith::RemSIOp, bitwidth,                                                                          \
        matchDivRem<true>, intBitwidthMatchLambda, 1,                                                                                           \
        amd64::IDIV ## bitwidth ## r>;

MULI_DIVI_PAT(8); MULI_DIVI_PAT(16); MULI_DIVI_PAT(32); MULI_DIVI_PAT(64);

#undef MULI_DIVI_PAT

#define SHIFT_PAT(bitwidth)                                                                                                                               \
    using ShlIPat ## bitwidth  = Match<mlir::arith::ShLIOp,  bitwidth, binOpMatchReplace, intBitwidthMatchLambda, 1, amd64::SHL ## bitwidth ## rr, amd64::SHL ## bitwidth ## ri>; \
    using ShrUIPat ## bitwidth = Match<mlir::arith::ShRUIOp, bitwidth, binOpMatchReplace, intBitwidthMatchLambda, 1, amd64::SHR ## bitwidth ## rr, amd64::SHR ## bitwidth ## ri>; \
    using ShrSIPat ## bitwidth = Match<mlir::arith::ShRSIOp, bitwidth, binOpMatchReplace, intBitwidthMatchLambda, 1, amd64::SAR ## bitwidth ## rr, amd64::SAR ## bitwidth ## ri>;

SHIFT_PAT(8); SHIFT_PAT(16); SHIFT_PAT(32); SHIFT_PAT(64);

#undef SHIFT_PAT

auto movMatchReplace = []<unsigned actualBitwidth,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(mlir::arith::ConstantIntOp op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<INSTri>(op, op.value());
    return mlir::success();
};

PATTERN_INT(ConstantIntPat, mlir::arith::ConstantIntOp, amd64::MOV, movMatchReplace);

// sign/zero extensions

/// ZExt from i1 pattern
//struct ExtUII1Pat : mlir::OpConversionPattern<mlir::arith::ExtUIOp> {
//    ExtUII1Pat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<mlir::arith::ExtUIOp>(tc, ctx, 2){}
//
//    mlir::LogicalResult matchAndRewrite(mlir::arith::ExtUIOp zextOp, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
//        // we're only matching i1s here
//        if(!zextOp.getIn().getType().isInteger(1))
//            return rewriter.notifyMatchFailure(zextOp, "this pattern only extends i1s");
//
//        // to be precise, we're only matching cmps for the moment, although this might change later
//        if(!zextOp.getIn().isa<mlir::OpResult>())
//            return rewriter.notifyMatchFailure(zextOp, "i1 zext pattern only matches cmps, this seems to be a block arg");
//
//        auto cmpi = mlir::dyn_cast<mlir::arith::CmpIOp>(zextOp.getIn().getDefiningOp());
//        if(!cmpi)
//            return rewriter.notifyMatchFailure(zextOp, "only cmps are supported for i1 extension for now");
//
//        assert(zextOp.getOut().getType().isIntOrFloat() && "extui with non-int result type");
//
//        // TODO this doesn't make any sense without a CMOVcc/SETcc for/after the cmp
//        switch(zextOp.getOut().getType().getIntOrFloatBitWidth()){
//            case 8:  rewriter.replaceOp(zextOp, adaptor.getIn() [> the cmpi should be replaced by a SETcc in another pattern, as this sets an 8 bit register anyway, we don't need to do anything here <]); break;
//            case 16: rewriter.replaceOpWithNewOp<amd64::MOVZXr16r8>(zextOp, adaptor.getIn()); break;
//            case 32: rewriter.replaceOpWithNewOp<amd64::MOVZXr32r8>(zextOp, adaptor.getIn()); break;
//            case 64: rewriter.replaceOpWithNewOp<amd64::MOVZXr64r8>(zextOp, adaptor.getIn()); break;
//
//            default:
//                return rewriter.notifyMatchFailure(zextOp, "unsupported bitwidth for i1 extension");
//        }
//        return mlir::success();
//    }
//};

// TODO sign extension for i1


/// the inBitwidth matches the bitwidth of the input operand to the extui, which needs to be different per pattern, because the corresponding instruction differs.
template<unsigned inBitwidth, auto getIn, auto getOut>
/// the outBitwidth matches the bitwidth of the result of the extui, which also affects which instruction is used.
/// ---
/// works with both floats and ints
auto truncExtUiSiBitwidthMatcher = []<unsigned outBitwidth>(auto thiis, auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter){
    // TODO shouldn't this use getOut(op) instead of op?
    // out bitwidth
    auto failure = intOrFloatBitwidthMatchLambda.operator()<outBitwidth>(thiis, op, adaptor, rewriter);
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

template<unsigned inBitwidth, auto getIn, auto getOut, amd64::SizeChange::Kind kind>
auto truncExtUiSiMatchReplace = []<unsigned outBitwidth,
     typename MOVSZX, typename SR8ri, typename, typename, typename
>(auto szextOp, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    // we need to take care to truncate an i1 to 0/1, before we 'actually' use it, i.e. do real computations with it on the other side of the MOVZX
    // i1's are represented as 8 bits currently, so we only need to check this in patterns which extend 8 bit
    if constexpr (inBitwidth == 8) /* && */ if(getIn(szextOp).getType().isInteger(1)){

            // assert the 8 bits for the i1
            assert(mlir::dyn_cast<amd64::GPRegisterTypeInterface>(getIn(adaptor).getType()).getBitwidth() == 8);
        if constexpr(kind == amd64::SizeChange::Kind::SExt){
            static_assert(outBitwidth > 8, "This pattern can't sign extend i1 to i8");

            // shift it left, then shift it right, to make a mask
            auto SHL = rewriter.create<amd64::SHL8ri>(szextOp.getLoc(), getIn(adaptor));
            auto SAR = rewriter.create<amd64::SAR8ri>(szextOp.getLoc(), SHL);
            SHL.instructionInfo().imm = SAR.instructionInfo().imm = 0x7;
            // now its sign extended to the 8 bits, then lets let the MOVSZX do its thing
            rewriter.replaceOpWithNewOp<MOVSZX>(szextOp, SAR);
            return mlir::success();
        }else{
            static_assert(kind == amd64::SizeChange::Kind::ZExt);

            // and it with 1
            auto AND = rewriter.create<amd64::AND8ri>(szextOp.getLoc(), getIn(adaptor));
            AND.instructionInfo().imm = 0x1;
            rewriter.replaceOpWithNewOp<MOVSZX>(szextOp, AND);
            return mlir::success();
        }
    }

    if constexpr(std::is_same_v<MOVSZX, amd64::MOV32rr>) // use the mov to 64 version of MOV32rr
        if constexpr(inBitwidth == 32 && outBitwidth == 64)
            rewriter.replaceOpWithNewOp<MOVSZX>(szextOp, amd64::gpr64Type::get(rewriter.getContext()), getIn(adaptor));
        else if constexpr(inBitwidth == 64 && outBitwidth == 32)
            rewriter.replaceOpWithNewOp<MOVSZX>(szextOp, amd64::gpr32Type::get(rewriter.getContext()), getIn(adaptor));
        else 
            static_assert(false, "this pattern can not be used to translate between the same bitwidth");
    else
        rewriter.replaceOpWithNewOp<MOVSZX>(szextOp, getIn(adaptor));
    return mlir::success();
};

auto arithGetIn = [](auto adaptorOrOp){ return adaptorOrOp.getIn(); };
auto arithGetOut = [](auto adaptorOrOp){ return adaptorOrOp.getOut(); };
template<unsigned inBitwidth, amd64::SizeChange::Kind kind>
auto arithTruncExtUiSiMatchReplace = truncExtUiSiMatchReplace<inBitwidth, arithGetIn, arithGetIn, kind>;
template<unsigned inBitwidth>
auto arithTruncExtUiSiBitwidthMatcher = truncExtUiSiBitwidthMatcher<inBitwidth, arithGetIn, arithGetIn>;

/// only for 16-64 bits outBitwidth, for 8 we have a special pattern. There are more exceptions: Because not all versions of MOVZX exist, MOVZXr8r8 wouldn't make sense (also invalid in MLIR), MOVZXr64r32 is just a MOV, etc.
#define EXT_UI_SI_PAT(outBitwidth, inBitwidth) \
    using ExtUIPat ## inBitwidth ## _to_ ## outBitwidth = Match<mlir::arith::ExtUIOp, outBitwidth, arithTruncExtUiSiMatchReplace<inBitwidth, amd64::SizeChange::ZExt>, arithTruncExtUiSiBitwidthMatcher<inBitwidth>, 1, amd64::MOVZX ## r ## outBitwidth ## r ## inBitwidth>; \
    using ExtSIPat ## inBitwidth ## _to_ ## outBitwidth = Match<mlir::arith::ExtSIOp, outBitwidth, arithTruncExtUiSiMatchReplace<inBitwidth, amd64::SizeChange::SExt>, arithTruncExtUiSiBitwidthMatcher<inBitwidth>, 1, amd64::MOVSX ## r ## outBitwidth ## r ## inBitwidth, NA, NA, NA, NA>;

// generalizable cases:
EXT_UI_SI_PAT(16, 8);
EXT_UI_SI_PAT(32, 8); EXT_UI_SI_PAT(32, 16);
EXT_UI_SI_PAT(64, 8); EXT_UI_SI_PAT(64, 16);

#undef EXT_UI_SI_PAT

// cases that are still valid in mlir, but not covered here:
// - 32 -> 64 (just a MOV)
// - any weird integer types, but we ignore those anyway
using ExtUIPat32_to_64 = Match<mlir::arith::ExtUIOp, 64, arithTruncExtUiSiMatchReplace<32, amd64::SizeChange::ZExt>, arithTruncExtUiSiBitwidthMatcher<32>, 1, amd64::MOV32rr>;
// for sign extend, the pattern above would work, but for simplicity, just do it manually here:
using ExtSIPat32_to_64 = Match<mlir::arith::ExtSIOp, 64, arithTruncExtUiSiMatchReplace<32, amd64::SizeChange::SExt>, arithTruncExtUiSiBitwidthMatcher<32>, 1, amd64::MOVSXr64r32>;

// trunc
#define TRUNC_PAT(outBitwidth, inBitwidth) \
    using TruncPat ## inBitwidth ## _to_ ## outBitwidth = Match<mlir::arith::TruncIOp, outBitwidth, arithTruncExtUiSiMatchReplace<inBitwidth, amd64::SizeChange::Trunc>, arithTruncExtUiSiBitwidthMatcher<inBitwidth>, 1, amd64::MOV ## outBitwidth ## rr>;
TRUNC_PAT(8, 16); TRUNC_PAT(8, 32); TRUNC_PAT(8, 64);
TRUNC_PAT(16, 32); TRUNC_PAT(16, 64);
TRUNC_PAT(32, 64);

#undef TRUNC_PAT

// branches
auto branchMatchReplace = []<unsigned actualBitwidth,
     typename JMP, typename, typename, typename, typename
     >(auto br, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<JMP>(br, adaptor.getDestOperands(), br.getDest());
    return mlir::success();
};
using BrPat = Match<mlir::cf::BranchOp, 64, branchMatchReplace, matchAllLambda, 1, amd64::JMP>;

template<typename IntegerCmpOp>
auto condBrMatchReplace = [](auto thiis, auto /* some kind of cond branch, either cf or llvm */ op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
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
    } else if(auto setccPredicate = mlir::dyn_cast<amd64::PredicateOpInterface>(adaptor.getCondition().getDefiningOp())){
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
        return condBrMatchReplace<mlir::arith::CmpIOp>(this, op, adaptor, rewriter);
    }
};

auto callMatchReplace = []<unsigned actualBitwidth,
     typename RegisterTy, typename, typename, typename, typename
     >(auto callOp, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    if(callOp.getNumResults() > 1)
        return rewriter.notifyMatchFailure(callOp, "multiple return values not supported");

    // direct call and indirect call are just handled the same as in the llvm dialect, the first op can be a ptr. So we don't do anything here, and check if the attribute exists in encoding, if it does, move it to AX and start one after the normal operand with moving to the operand registers
    if(callOp.getNumResults() == 0)
        rewriter.replaceOpWithNewOp<amd64::CALL>(callOp, TypeRange(),                        callOp.getCalleeAttr(), /* is guaranteed external */ false, adaptor.getOperands());
    else
        rewriter.replaceOpWithNewOp<amd64::CALL>(callOp, RegisterTy::get(callOp->getContext()), callOp.getCalleeAttr(), /* is guaranteed external */ false, adaptor.getOperands());

    return mlir::success();
};

#define CALL_PAT(bitwidth) \
    using IntCallPat ## bitwidth = Match<mlir::func::CallOp, bitwidth, callMatchReplace, defaultBitwidthMatchLambda<amd64::GPRegisterTypeInterface, true>, 1, amd64::gpr ## bitwidth ## Type>

CALL_PAT(8); CALL_PAT(16); CALL_PAT(32); CALL_PAT(64);
using FloatCallPat32 = Match<mlir::func::CallOp, 32, callMatchReplace, defaultBitwidthMatchLambda<amd64::FPRegisterTypeInterface, true>, 1, amd64::fpr32Type>;
using FloatCallPat64 = Match<mlir::func::CallOp, 64, callMatchReplace, defaultBitwidthMatchLambda<amd64::FPRegisterTypeInterface, true>, 1, amd64::fpr64Type>;

#undef CALL_PAT

// TODO maybe AND i1's before returning them
// returns
auto returnMatchReplace = []<unsigned actualBitwidth,
     typename, typename, typename, typename, typename
     >(auto returnOp, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    mlir::Value retOperand;
    if(returnOp.getNumOperands() > 1)
        return rewriter.notifyMatchFailure(returnOp, "multiple return values not supported");
    else if(returnOp.getNumOperands() == 0)
        // if there is a zero op return, the function does not return anythign ,so we can just mov 0 to rax and return that.
        // TODO consider omitting the return
        retOperand = rewriter.create<amd64::MOV64ri>(returnOp.getLoc(), 0);
    else
        retOperand = adaptor.getOperands().front();

    rewriter.replaceOpWithNewOp<amd64::RET>(returnOp, retOperand);
    return mlir::success();
};

using ReturnPat = Match<mlir::func::ReturnOp, 64, returnMatchReplace, matchAllLambda, 1>;

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
                llvm_unreachable("gep struct element index out of bounds");
            }else if(LLVMArrayType arrayType = type.template dyn_cast<LLVMArrayType>()){
                return {dl.getTypeSize(arrayType.getElementType()) * elemNum, arrayType.getElementType()};
            }else if(LLVMPointerType ptrType = type.template dyn_cast<LLVMPointerType>()){
                // in this case, we just always use the source element type, and then index into that

                // TODO this assertion is wrong, what we would actually want to assert, is that *this* index is the first one
                //assert(op.getIndices().size() <= 2 && "only up to two indices supported for pointer element types");
                return {dl.getTypeSize(op.getSourceElementType()) * elemNum, op.getSourceElementType()};
            }else if(IntegerType intType = type.template dyn_cast<IntegerType>()){
                // TODO this assertion probably wrong for the same reasons as above, just that this should always be the last index.
                //assert(op.getIndices().size() == 1 && "only one index is supported for int element types");
                // TODO rethink this, im not sure this makes sense. It seems to work atm, but just the fact that we're using the type size of the int type here, and the type size of the element source type above doesn't make sense. The two are '1 layer apart', so they shouldn't be used on the same layer, right?
                return {dl.getTypeSize(intType) * elemNum, mlir::Type()};
            }else{
                op.dump();
                type.dump();
                llvm_unreachable("unhandled gep base type");
            }
        };

        // there is no adaptor.getIndices(), the adaptor only gives access to the dynamic indices. so we iterate over all of the indices, and if we find a dynamic one, use the rewriter to remap it
        auto indices = op.getIndices();
        // TODO check for allocas to optimize if possible
        auto currentIndexComputationValue = adaptor.getBase();

        // TODO the other case is some weird vector thing, i'd rather have it fail for now, if that is encountered
        assert(op.getElemType().has_value());

        // we start by indexing into the base type
        mlir::Type currentlyIndexedType = op.getBase().getType();

        for(auto indexPtr_u : indices){
            assert(getTypeConverter()->convertType(currentIndexComputationValue.getType()) == amd64::gpr64Type::get(getContext()) && "only 64 bit pointers are supported");

            if(mlir::Value val = indexPtr_u.dyn_cast<mlir::Value>()){
                // no dynamic struct indices please
                assert(!mlir::isa<LLVM::LLVMStructType>(currentlyIndexedType) && "dynamic struct indices are not allowed, this should be fixed in the verification of GEP in the llvm dialect!");

                //llvm::errs() << "value to be scaled in gep: "; op.dump(); llvm::errs() << "  original value: "; val.dump(); llvm::errs() << "  remapped value:"; rewriter.getRemappedValue(val).dump();
                auto scaled = rewriter.create<amd64::IMUL64rri>(op.getLoc(), rewriter.getRemappedValue(val));

                // we perform the computation analogously, but just for ptr/array types, so use 1 as the index
                std::tie(scaled.instructionInfo().imm, currentlyIndexedType) = getBytesOffsetAndType(currentlyIndexedType, 1);
                currentIndexComputationValue = rewriter.create<amd64::ADD64rr>(op.getLoc(), currentIndexComputationValue, scaled);
            }else{
                // has to be integer attr otherwise
                auto indexInt = indexPtr_u.get<mlir::IntegerAttr>().getValue().getSExtValue();

                int64_t byteOffset;
                std::tie(byteOffset, currentlyIndexedType) = getBytesOffsetAndType(currentlyIndexedType, indexInt);

                // if the offset is zero, we don't have to create an instruction, but we do need to change the indexed type
                if(byteOffset == 0)
                    continue;

                auto addri = rewriter.create<amd64::ADD64ri>(op.getLoc(), currentIndexComputationValue);
                addri.instructionInfo().imm = byteOffset;
                currentIndexComputationValue = addri;
            }
        }

        rewriter.replaceOp(op, currentIndexComputationValue);
        return mlir::success();
    }
};

template<typename INSTrm>
auto llvmLoadMatchReplace = [](LLVM::LoadOp op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    // TODO this is an ugly hack, because this op gets unrealized conversion casts as args (ptr in this case), because the ptr type gets converted to an i64, instead of a memloc, so the alloca returning a memloc doesn't work
    // even if we perform the conversion casts ourselves and insert a 1-2 type conversion from ptr to memloc/i64, this still doesn't work
    auto ptr = adaptor.getAddr();
    if(auto cast = mlir::dyn_cast_if_present<mlir::UnrealizedConversionCastOp>(ptr.getDefiningOp())){
        assert(cast->getNumOperands() == 1 && mlir::isa<amd64::memLocType>(cast->getOperand(0).getType()));
        rewriter.replaceOpWithNewOp<INSTrm>(op, cast->getOperand(0));
        return mlir::success();
    }

    auto mem = rewriter.create<amd64::MemB>(op.getLoc(), ptr);
    rewriter.replaceOpWithNewOp<INSTrm>(op, mem);
    return mlir::success();
};

auto llvmIntLoadMatchReplace = []<unsigned actualBitwidth,
     typename, typename, typename INSTrm, typename, typename
     >(LLVM::LoadOp op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    return llvmLoadMatchReplace<INSTrm>(op, adaptor, rewriter);
};

template<typename INSTrm>
auto llvmFloatLoadMatchReplace = []<unsigned actualBitwidth,
     typename, typename, typename, typename, typename
     >(LLVM::LoadOp op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    return llvmLoadMatchReplace<INSTrm>(op, adaptor, rewriter);
};

PATTERN_INT(LLVMIntLoadPat, LLVM::LoadOp, amd64::MOV, llvmIntLoadMatchReplace);
using LLVMFloatLoadPat32 = Match<LLVM::LoadOp, 32, llvmFloatLoadMatchReplace<amd64::MOVSSrm>, floatBitwidthMatchLambda>;
using LLVMFloatLoadPat64 = Match<LLVM::LoadOp, 64, llvmFloatLoadMatchReplace<amd64::MOVSDrm>, floatBitwidthMatchLambda>;

template<typename INSTmr>
auto llvmStoreMatchReplace = [](LLVM::StoreOp op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    // TODO this is an ugly hack, because this op gets unrealized conversion casts as args (ptr in this case), because the ptr type gets converted to an i64, instead of a memloc, so the alloca returning a memloc doesn't work
    // even if we perform the conversion casts ourselves and insert a 1-2 type conversion from ptr to memloc/i64, this still doesn't work
    auto ptr = adaptor.getAddr();
    auto val = adaptor.getValue();
    if(auto cast = mlir::dyn_cast_if_present<mlir::UnrealizedConversionCastOp>(ptr.getDefiningOp())){
        assert(cast->getNumOperands() == 1 && mlir::isa<amd64::memLocType>(cast->getOperand(0).getType()));
        rewriter.replaceOpWithNewOp<INSTmr>(op, cast->getOperand(0), val);
        return mlir::success();
    }

    auto mem = rewriter.create<amd64::MemB>(op.getLoc(), ptr);
    rewriter.replaceOpWithNewOp<INSTmr>(op, mem, val);
    return mlir::success();
};

auto llvmIntStoreMatchReplace = []<unsigned actualBitwidth,
     typename, typename, typename, typename, typename INSTmr
     >(LLVM::StoreOp op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    return llvmStoreMatchReplace<INSTmr>(op, adaptor, rewriter);
};

template<typename INSTmr>
auto llvmFloatStoreMatchReplace = []<unsigned actualBitwidth,
     typename, typename, typename, typename, typename
     >(LLVM::StoreOp op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    return llvmStoreMatchReplace<INSTmr>(op, adaptor, rewriter);
};

template <typename RegisterTy>
requires MLIRInterfaceDerivedFrom<RegisterTy, amd64::RegisterTypeInterface>
auto llvmStoreBitwidthMatcher = []<unsigned bitwidth>(auto thiis, LLVM::StoreOp op, auto, mlir::ConversionPatternRewriter& rewriter){ 
    if(auto gprType = mlir::dyn_cast<RegisterTy>(thiis->getTypeConverter()->convertType(op.getValue().getType()))){
        if(gprType.getBitwidth() == bitwidth)
            return mlir::success();

        return rewriter.notifyMatchFailure(op, "bitwidth mismatch");
    }

    return rewriter.notifyMatchFailure(op, "expected other register type");
};
PATTERN_INT(LLVMIntStorePat, LLVM::StoreOp, amd64::MOV, llvmIntStoreMatchReplace, llvmStoreBitwidthMatcher<amd64::GPRegisterTypeInterface>);
using LLVMFloatStorePat32 = Match<LLVM::StoreOp, 32, llvmFloatStoreMatchReplace<amd64::MOVSSmr>, llvmStoreBitwidthMatcher<amd64::FPRegisterTypeInterface>>;
using LLVMFloatStorePat64 = Match<LLVM::StoreOp, 64, llvmFloatStoreMatchReplace<amd64::MOVSDmr>, llvmStoreBitwidthMatcher<amd64::FPRegisterTypeInterface>>;

struct LLVMAllocaPat : public mlir::OpConversionPattern<mlir::LLVM::AllocaOp>{
    LLVMAllocaPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<LLVM::AllocaOp>(tc, ctx, 1){}

    mlir::LogicalResult matchAndRewrite(LLVM::AllocaOp op, OpAdaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        // TODO maybe this can be improved when considering that the alloca is only ever used as a ptr by GEP, load, store, and ptrtoint. In this case the lea is technically only needed for ptrtoint
        auto numElemsVal = op.getArraySize();
        auto constantElemNumOp = mlir::dyn_cast<LLVM::ConstantOp>(numElemsVal.getDefiningOp());
        if(!constantElemNumOp)
            return rewriter.notifyMatchFailure(op, "only constant allocas supported for now");
        auto numElems = constantElemNumOp.getValue().cast<mlir::IntegerAttr>().getValue().getSExtValue();

        // TODO AllocaOp::print does this a bit differently -> use that?
        auto  dl = mlir::DataLayout::closest(op);
        assert(op.getElemType().has_value());
        auto elemSize = dl.getTypeSize(*op.getElemType());

        // gets converted to i64 with target materialization from type converter
        rewriter.replaceOpWithNewOp<amd64::AllocaOp>(op, elemSize*numElems);
        return mlir::success();
    }
};

auto llvmMovMatchReplace = []<unsigned actualBitwidth,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(LLVM::ConstantOp op, auto adaptor, mlir::ConversionPatternRewriter& rewriter) {
    auto intAttr = adaptor.getValue().template cast<mlir::IntegerAttr>();
    if(!intAttr)
        return rewriter.notifyMatchFailure(op, "expected integer constant");

    rewriter.replaceOpWithNewOp<INSTri>(op, intAttr.getValue().getSExtValue());
    return mlir::success();
};
PATTERN_INT(LLVMConstantIntPat, LLVM::ConstantOp, amd64::MOV, llvmMovMatchReplace, intBitwidthMatchLambda, 2);

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
        auto globalOrFunc = mlir::SymbolTable::lookupNearestSymbolFrom(op, adaptor.getGlobalNameAttr());
        assert(globalOrFunc && "global from addrof not found");
        if(auto globalOp = mlir::dyn_cast<LLVM::GlobalOp>(globalOrFunc)){
            // we don't necessarily know the offset of the global yet, might need to be resolved later
            // TODO maybe pass globals and check if it's in there already to avoid making the extra op. Might also be slower, not sure
            rewriter.replaceOpWithNewOp<amd64::AddrOfGlobal>(op, adaptor.getGlobalNameAttr());
        } else {
            // calling dlsym here is a mere optimization. It could also be left to AddrOfFunc, but that would require another symbol table look-up, and those are glacially slow already...
            auto funcOp = mlir::cast<mlir::FunctionOpInterface>(globalOrFunc);
            if(!funcOp.isExternal())
                // unknown until later
                rewriter.replaceOpWithNewOp<amd64::AddrOfFunc>(op, adaptor.getGlobalNameAttr());
            else
                rewriter.replaceOpWithNewOp<amd64::MOV64ri>(op, (intptr_t) checked_dlsym(adaptor.getGlobalName())); // TODO only do this if args.jit 
        }
        return mlir::success();
    }
};

using LLVMReturnPat = Match<LLVM::ReturnOp, 64, returnMatchReplace, matchAllLambda, 1, amd64::RET>;

struct LLVMFuncPat : public mlir::OpConversionPattern<LLVM::LLVMFuncOp>{
    LLVMFuncPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<LLVM::LLVMFuncOp>(tc, ctx, 1){}

    mlir::LogicalResult matchAndRewrite(LLVM::LLVMFuncOp func, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        if(func.isVarArg() && !func.isExternal())
            return rewriter.notifyMatchFailure(func, "vararg functions are not supported yet");

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
    GlobalsInfo& globals;

    LLVMGlobalPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx, GlobalsInfo& globals) : mlir::OpConversionPattern<LLVM::GlobalOp>(tc, ctx, 1), globals(globals){ }

    mlir::LogicalResult matchAndRewrite(LLVM::GlobalOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        // if its a declaration: handle specially, we can already look up the address of the symbol and write it there, or fail immediately
        //llvm::errs() << "handling global: "; op.dump();
        GlobalSymbolInfo unfinishedGlobal;
        intptr_t& addr = unfinishedGlobal.addrInDataSection = (intptr_t) nullptr;
        auto& bytes = unfinishedGlobal.bytes = {};

        auto symbol = op.getSymName();
        if(op.isDeclaration()){
            // this address is allowed to be 0, checked_dlsym handles an actual error through dlerror()
            DEBUGLOG("external symbol: " << symbol << ", getting address from environment");
            addr = (intptr_t) checked_dlsym(symbol);
        }

        auto insertGlobalAndEraseOp = [&](){
            globals.insert({symbol, std::move(unfinishedGlobal)});
            rewriter.eraseOp(op);
            return mlir::success();
        };

        auto mlirGetTypeSize = [&](auto type){
            return mlir::DataLayout::closest(op).getTypeSize(adaptor.getGlobalType());
        };

        auto fail = [&](StringRef msg = "globals with complex initializers are not supported yet"){
            return rewriter.notifyMatchFailure(op, msg);
        };

        auto letLLVMDoIt = [&](){
            // do the same thing that LLVM does itself (ModuleTranslation::convertGlobals), this is quite ugly, but can handle the most amount of cases
            // TODO performance test if creating the module/context globally (or with `static` here)once is faster than locally every time
            // TODO because globals (that might need to be translated by llvm) can cross reference each other, we can't create isolated modules only containing one global, we need to create one big llvm module containing all globals, to allow for that. But only do it if it actually happens 
            // TODO calling functions in global initializers would be even more difficult.
            llvm::LLVMContext llvmCtx;

            MLIRContext& mlirCtx = *op.getContext();

            auto miniModule = mlir::OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&mlirCtx)));
            miniModule->getBody()->push_back(op.clone());

            auto llvmModule = translateModuleToLLVMIR(miniModule.get(), llvmCtx);
            if(!llvmModule)
                return fail("failed to translate global to llvm ir");

            assert(!llvmModule->empty() && llvmModule->global_size() == 1 && "expected exactly one global in the module");

            auto llvmGetTypeSize = [&](auto type){
                return llvmModule->getDataLayout().getTypeSizeInBits(type) / 8;
            };

            auto llvmConstant = llvmModule->globals().begin()->getInitializer();
            if(!llvmConstant)
                return fail("failed to get initializer of global");

            unfinishedGlobal.alignment = op.getAlignment().value_or(0);

            assert(llvm::isa<llvm::ConstantData>(llvmConstant));

            assert(mlirGetTypeSize(adaptor.getGlobalType()) == llvmGetTypeSize(llvmConstant->getType()));

            // get bytes from the constant
            if(auto sequential = llvm::dyn_cast<llvm::ConstantDataSequential>(llvmConstant)){
                DEBUGLOG("constant data sequential");
                auto bytes = sequential->getRawDataValues();
                assert(bytes.size() == mlirGetTypeSize(adaptor.getGlobalType()) && "global size mismatch");
                unfinishedGlobal.bytes.resize(bytes.size());

                memcpy(unfinishedGlobal.bytes.data(), bytes.data(), bytes.size());
            }else if(auto zero = llvm::dyn_cast<llvm::ConstantAggregateZero>(llvmConstant)){
                DEBUGLOG("constant aggregate zero");
                unfinishedGlobal.bytes.resize(mlirGetTypeSize(adaptor.getGlobalType()));
                // TODO use `zero` to validate the size


#ifndef NDEBUG
                for(auto byte : unfinishedGlobal.bytes)
                    assert(byte == 0 && "constant aggregate zero is not all 0s");
#endif
            }else{
                return fail("failed to get raw data values of constant, TODO");
            }
            return mlir::success();
        };

        // TODO globals with initalization region
        if(auto initBlock = op.getInitializerBlock()){
            if(initBlock->getOperations().size() == 2){
                // TODO dyn cast operand as ptr?
                // TODO this only remains, because it was there first, technically this case would be covered by the llvm-ir translation. But as that is probably quite slow, and this is a common case, let's leave it for now.
                auto llvmNull = mlir::dyn_cast<LLVM::NullOp>(initBlock->getOperations().front());
                auto ret = mlir::cast<LLVM::ReturnOp>(initBlock->getOperations().back()); // has to be a return, according to doc
                if(!llvmNull || ret.getOperands().size() != 1 || ret.getOperand(0) != llvmNull)
                    return fail();

                auto byteSize = sizeof(intptr_t);
                bytes.resize(byteSize);
                auto nullptr_ = nullptr;
                assert(byteSize == sizeof(nullptr_) && "nullptr is not the same size as a pointer");
                assert(memcmp(bytes.data(), &nullptr_, byteSize) == 0 && "nullptr is not all 0s");
                unfinishedGlobal.alignment = op.getAlignment().value_or(byteSize);
            }else{
                auto logResult =  letLLVMDoIt();
                if(mlir::failed(logResult))
                    return logResult;
            }
            return insertGlobalAndEraseOp();
        }


        // get raw bytes of the value of this global
        // heavily inspired by ModuleTranslation::convertGlobals/LLVM::detail:getLLVMConstant
        // TODO pointer type globals
        if(auto attr = op.getValueOrNull()){
            DEBUGLOG("attr");
            // TODO maybe try to optimize the ordering?
            if(auto denseElementsAttr = attr.dyn_cast<mlir::DenseElementsAttr>()){
                auto maybeRange = denseElementsAttr.tryGetValues<uint8_t>();
                if(static_cast<mlir::LogicalResult>(maybeRange).succeeded()){
                    auto range = maybeRange.value();
                    bytes.resize(range.size());
                    for(auto [i, byte] : llvm::enumerate(range)){
                        bytes[i] = byte;
                    }
                }else{
                    auto logResult = letLLVMDoIt();
                    if(mlir::failed(logResult))
                        return logResult;

                    return insertGlobalAndEraseOp();
                }
                //auto rawData = denseElementsAttr.getRawData();
                //DEBUGLOG("denseElementsAttr, with size " << rawData.size());
                //bytes.resize(rawData.size());
                //memcpy(bytes.data(), rawData.data(), rawData.size());
            }else if(auto strAttr = attr.dyn_cast<mlir::StringAttr>()){
                // TODO is this guaranteed to be null terminated, if it comes from a global?
                bytes.resize(strAttr.getValue().size());
                memcpy(bytes.data(), strAttr.getValue().data(), strAttr.getValue().size());
                assert(unfinishedGlobal.bytes.back() == '\0');
            }else if(auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()){
                assert((intAttr.getType().getIntOrFloatBitWidth() == 1 || intAttr.getType().getIntOrFloatBitWidth() == 8 || intAttr.getType().getIntOrFloatBitWidth() == 16 || intAttr.getType().getIntOrFloatBitWidth() == 32 || intAttr.getType().getIntOrFloatBitWidth() == 64) && "global of invalid bitwidth");

                auto size = intAttr.getType().getIntOrFloatBitWidth()/8;
                auto val = intAttr.getValue().getSExtValue();
                if(size == 0){ // has to have been an i1 -> round up to 1 byte
                    size = 1;
                    val&=1;
                }


                bytes.resize(size);
                memcpyToLittleEndianBuffer(bytes.data(), val, size);
            }else if(auto floatAttr = attr.dyn_cast<mlir::FloatAttr>()){
                assert((floatAttr.getType().getIntOrFloatBitWidth() == 32 || floatAttr.getType().getIntOrFloatBitWidth() == 64) && "global of invalid bitwidth");

                auto size = floatAttr.getType().getIntOrFloatBitWidth()/8;
                bytes.resize(size);
                if(size == 4)
                    memcpyToLittleEndianBuffer(bytes.data(), static_cast<float>(floatAttr.getValueAsDouble()), size);
                else
                    memcpyToLittleEndianBuffer(bytes.data(), floatAttr.getValueAsDouble(), size);
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

        unfinishedGlobal.alignment = static_cast<unsigned>(adaptor.getAlignment().value_or(mlirGetTypeSize(adaptor.getGlobalType())));

        // TODO handle visibility
        return insertGlobalAndEraseOp();
    }
};

#define CALL_PAT(bitwidth) \
    using LLVMIntCallPat ## bitwidth = Match<LLVM::CallOp, bitwidth, callMatchReplace, /* match ints, floats, and zero res */ defaultBitwidthMatchLambda<amd64::GPRegisterTypeInterface, true>, 1, amd64::gpr ## bitwidth ## Type>

CALL_PAT(8); CALL_PAT(16); CALL_PAT(32); CALL_PAT(64);
using LLVMFloatCallPat32 = Match<LLVM::CallOp, 32, callMatchReplace, defaultBitwidthMatchLambda<amd64::FPRegisterTypeInterface, true>, 1, amd64::fpr32Type>;
using LLVMFloatCallPat64 = Match<LLVM::CallOp, 64, callMatchReplace, defaultBitwidthMatchLambda<amd64::FPRegisterTypeInterface, true>, 1, amd64::fpr64Type>;

#undef CALL_PAT

PATTERN_INT(LLVMICmpPat, LLVM::ICmpOp, amd64::CMP, cmpIMatchReplace<LLVM::ICmpPredicate>, cmpIBitwidthMatcher);

// TODO order the LLVM patterns in a more structured way

using LLVMBrPat = Match<LLVM::BrOp, 64, branchMatchReplace, matchAllLambda, 1, amd64::JMP>;

struct LLVMCondBrPat : public mlir::OpConversionPattern<LLVM::CondBrOp> {
    LLVMCondBrPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<LLVM::CondBrOp>(tc, ctx, 3){}

    mlir::LogicalResult matchAndRewrite(LLVM::CondBrOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        return condBrMatchReplace<LLVM::ICmpOp>(this, op, adaptor, rewriter);
    }
};

struct CaseInfo{
    int64_t comparisonValue;
    mlir::Block* block;
    mlir::ValueRange operands;
};

// TODO maybe do this non-recursively at some point, but that's too annoying for now
template<typename CMPri, typename CMPrr, typename MOVri>
void binarySearchSwitchLowering(mlir::Location loc, mlir::ConversionPatternRewriter& rewriter, mlir::Value adaptedValue, CaseInfo defaultDest, std::span<CaseInfo> caseInfoSection){
    DEBUGLOG("binarySearchSwitchLowering: pivotIndex = " << caseInfoSection.size()/2 << ", caseInfoSection.size() = " << caseInfoSection.size());
    auto* currentBlock = rewriter.getInsertionBlock();

    // TODO recursion end condition 1
    if(caseInfoSection.empty()){
        // jump to default block on empty case section
        rewriter.replaceAllUsesWith(currentBlock, defaultDest.block);
        rewriter.eraseBlock(currentBlock);
        return;
    }

    auto pivotIndex = caseInfoSection.size() / 2;
    auto pivotInfo = caseInfoSection[pivotIndex];

    // if the value is equal, jump to the block
    // check whether comparison value can be encoded as a 32 bit immediate, otherwise do a MOVri, then CMPrr
    if(fitsInto32BitImm(pivotInfo.comparisonValue)){
        auto cmp = rewriter.create<CMPri>(loc, adaptedValue);
        cmp.instructionInfo().imm = pivotInfo.comparisonValue;
    }else{
        auto movri = rewriter.create<MOVri>(loc, pivotInfo.comparisonValue);
        rewriter.create<CMPrr>(loc, adaptedValue, movri);
    }
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
    binarySearchSwitchLowering<CMPri, CMPrr, MOVri>(loc, rewriter, adaptedValue, defaultDest, caseInfoSection.first(pivotIndex));

    rewriter.setInsertionPointToEnd(searchUpperHalf);
    // TODO check that this is modmod  done using a bitwise and with 1
    binarySearchSwitchLowering<CMPri, CMPrr, MOVri>(loc, rewriter, adaptedValue, defaultDest,
        caseInfoSection.last(caseInfoSection.size() - pivotIndex - 1 /* leave out the pivot itself */));
}


// TODO extend this to mlir.cf.switch
auto switchMatchReplace = []<unsigned actualBitwidth,
     typename CMPri, typename CMPrr, typename MOVri, typename, typename
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
    auto adaptedCaseOperands = adaptor.getCaseOperands();
    for(auto [caseValue, block, operands] : llvm::zip(caseValues, op.getCaseDestinations(), adaptor.getCaseOperands()))
        caseValuesIntSorted.push_back({caseValue.getSExtValue(), block, operands});

    // TODO this might be super slow
    std::sort(caseValuesIntSorted.begin(), caseValuesIntSorted.end(), [](auto a, auto b){return a.comparisonValue < b.comparisonValue;});

    // now do the actual binary search
    auto defaultCase = op.getDefaultDestination();
    assert(defaultCase && "switches without default cases are not allowed (I think)");
    binarySearchSwitchLowering<CMPri, CMPrr, MOVri>(op->getLoc(), rewriter, adaptor.getValue(), {0, defaultCase, adaptor.getDefaultOperands()}, caseValuesIntSorted);

    rewriter.eraseOp(op);

    return mlir::success();
};

// bitwidth matcher: basically default, but we cant use the result, have to use `getValue` at the start
auto switchBitwidthMatcher = []<unsigned bitwidth>(auto thiis, LLVM::SwitchOp op, auto, mlir::ConversionPatternRewriter& rewriter){
    // TODO this is almost the same as defaultBitwidthMatchLambda, try to merge them
    mlir::Type opType = op.getValue().getType();

    auto type = thiis->getTypeConverter()->convertType(opType);
    // TODO might be slow
    if(!type)
        return rewriter.notifyMatchFailure(op, "type conversion failed");

    auto typeToMatch= type.template dyn_cast<amd64::GPRegisterTypeInterface>();
    //assert(typeToMatch && "expected register type");
    if(!typeToMatch)
        return rewriter.notifyMatchFailure(op, "expected register type");
    // TODO this assertion currently fails wrongly on a conditional branch
    //assert((op->getNumOperands() == 0 || typeToMatch == thiis->getTypeConverter()->convertType(op->getOperand(0).getType()).template dyn_cast<amd64::GPRegisterTypeInterface>()) && "this simple bitwidth matcher assumes that the type of the op and the type of the operands are the same");

    if(typeToMatch.getBitwidth() != bitwidth)
        return rewriter.notifyMatchFailure(op, "bitwidth mismatch");

    return mlir::success();
};

// TODO define for cf.switch
#define SWITCH_PAT(name, OpTy, bitwidth)\
    using name ## bitwidth = Match<OpTy, bitwidth, switchMatchReplace, switchBitwidthMatcher, 1, amd64::CMP ## bitwidth ## ri, amd64::CMP ## bitwidth ## rr, amd64::MOV ## bitwidth ## ri>;

SWITCH_PAT(LLVMSwitchPat, LLVM::SwitchOp, 8);
SWITCH_PAT(LLVMSwitchPat, LLVM::SwitchOp, 16);
SWITCH_PAT(LLVMSwitchPat, LLVM::SwitchOp, 32);
SWITCH_PAT(LLVMSwitchPat, LLVM::SwitchOp, 64);

// Arithmetic stuff

auto llvmGetIn = [](auto adaptorOrOp){ return adaptorOrOp.getArg(); };
auto llvmGetOut = [](auto adaptorOrOp){ return adaptorOrOp.getRes(); };

template<unsigned inBitwidth, amd64::SizeChange::Kind kind>
auto llvmTruncExtUiSiMatchReplace = truncExtUiSiMatchReplace<inBitwidth, llvmGetIn, llvmGetOut, kind>;
template<unsigned inBitwidth>
auto llvmTruncExtUiSiBitwidthMatcher = truncExtUiSiBitwidthMatcher<inBitwidth, llvmGetIn, llvmGetOut>;

// LLVM SExt/ZExt/Trunc patterns, same as MLIR above, read up there on why this is divided into these weird cases
#define SZEXT_PAT(outBitwidth, inBitwidth) \
    using LLVMZExtPat ## inBitwidth ## _to_ ## outBitwidth = Match<LLVM::ZExtOp, outBitwidth, llvmTruncExtUiSiMatchReplace<inBitwidth, amd64::SizeChange::ZExt>, llvmTruncExtUiSiBitwidthMatcher<inBitwidth>, 1, amd64::MOVZX ## r ## outBitwidth ## r ## inBitwidth>; \
    using LLVMSExtPat ## inBitwidth ## _to_ ## outBitwidth = Match<LLVM::SExtOp, outBitwidth, llvmTruncExtUiSiMatchReplace<inBitwidth, amd64::SizeChange::SExt>, llvmTruncExtUiSiBitwidthMatcher<inBitwidth>, 1, amd64::MOVSX ## r ## outBitwidth ## r ## inBitwidth>;

// generalizable cases:
SZEXT_PAT(16, 8);
SZEXT_PAT(32, 8); SZEXT_PAT(32, 16);
SZEXT_PAT(64, 8); SZEXT_PAT(64, 16);

#undef SZEXT_PAT

using LLVMZExtPat32_to_64 = Match<LLVM::ZExtOp, 64, llvmTruncExtUiSiMatchReplace<32, amd64::SizeChange::ZExt>, llvmTruncExtUiSiBitwidthMatcher<32>, 1, amd64::MOV32rr>;
using LLVMSExtPat32_to_64 = Match<LLVM::SExtOp, 64, llvmTruncExtUiSiMatchReplace<32, amd64::SizeChange::SExt>, llvmTruncExtUiSiBitwidthMatcher<32>, 1, amd64::MOVSXr64r32>;
// trunc
#define TRUNC_PAT(outBitwidth, inBitwidth) \
    using LLVMTruncPat ## inBitwidth ## _to_ ## outBitwidth = Match<LLVM::TruncOp, outBitwidth, llvmTruncExtUiSiMatchReplace<inBitwidth, amd64::SizeChange::Trunc>, llvmTruncExtUiSiBitwidthMatcher<inBitwidth>, 1, amd64::MOV ## outBitwidth ## rr>;
TRUNC_PAT(8, 16); TRUNC_PAT(8, 32); TRUNC_PAT(8, 64);
TRUNC_PAT(16, 32); TRUNC_PAT(16, 64);
TRUNC_PAT(32, 64);

#undef TRUNC_PAT

PATTERN_INT(LLVMAddPat, LLVM::AddOp, amd64::ADD, binOpMatchReplace);
PATTERN_INT(LLVMSubPat, LLVM::SubOp, amd64::SUB, binOpMatchReplace);
PATTERN_INT(LLVMAndPat, LLVM::AndOp, amd64::AND, binOpMatchReplace);
PATTERN_INT(LLVMOrPat,  LLVM::OrOp,  amd64::OR,  binOpMatchReplace);
PATTERN_INT(LLVMXOrPat, LLVM::XOrOp, amd64::XOR, binOpMatchReplace);

#define MUL_DIV_PAT(bitwidth)                                                                                                               \
    using LLVMMulPat ## bitwidth = Match<LLVM::MulOp, bitwidth, binOpMatchReplace, intBitwidthMatchLambda, 1, amd64::MUL ## bitwidth ## r>; \
    using LLVMUDivPat ## bitwidth = Match<LLVM::UDivOp, bitwidth,                                                                           \
        matchDivRem<false>, intBitwidthMatchLambda, 1,                                                                                      \
        amd64::DIV ## bitwidth ## r>;                                                                                                       \
    using LLVMSDivPat ## bitwidth = Match<LLVM::SDivOp, bitwidth,                                                                           \
        matchDivRem<false>, intBitwidthMatchLambda, 1,                                                                                      \
        amd64::IDIV ## bitwidth ## r>;                                                                                                      \
    using LLVMURemPat ## bitwidth = Match<LLVM::URemOp, bitwidth,                                                                           \
        matchDivRem<true>, intBitwidthMatchLambda, 1,                                                                                       \
        amd64::DIV ## bitwidth ## r>;                                                                                                       \
    using LLVMSRemPat ## bitwidth = Match<LLVM::SRemOp, bitwidth,                                                                           \
        matchDivRem<true>, intBitwidthMatchLambda, 1,                                                                                       \
        amd64::IDIV ## bitwidth ## r>;

MUL_DIV_PAT(8); MUL_DIV_PAT(16); MUL_DIV_PAT(32); MUL_DIV_PAT(64);

#undef MUL_DIV_PAT

#define SHIFT_PAT(bitwidth)                                                                                                                               \
    using LLVMShlPat ## bitwidth  = Match<LLVM::ShlOp,  bitwidth, binOpMatchReplace, intBitwidthMatchLambda, 1, amd64::SHL ## bitwidth ## rr, amd64::SHL ## bitwidth ## ri>; \
    using LLVMLShrPat ## bitwidth = Match<LLVM::LShrOp, bitwidth, binOpMatchReplace, intBitwidthMatchLambda, 1, amd64::SHR ## bitwidth ## rr, amd64::SHR ## bitwidth ## ri>; \
    using LLVMAShrPat ## bitwidth = Match<LLVM::AShrOp, bitwidth, binOpMatchReplace, intBitwidthMatchLambda, 1, amd64::SAR ## bitwidth ## rr, amd64::SAR ## bitwidth ## ri>;

SHIFT_PAT(8); SHIFT_PAT(16); SHIFT_PAT(32); SHIFT_PAT(64);

#undef SHIFT_PAT

using LLVMNullPat = SimplePat<LLVM::NullOp, [](auto op, auto, mlir::ConversionPatternRewriter& rewriter){
    rewriter.replaceOpWithNewOp<amd64::MOV64ri>(op, (intptr_t) nullptr);
    return mlir::success();
}>;

auto zeroReplace = [](auto op, auto, mlir::ConversionPatternRewriter& rewriter){
    // TODO warn about this
    //      actually maybe not, MLIR uses undefs so often in normal situations...
    rewriter.replaceOpWithNewOp<amd64::MOV64ri>(op, 0);
    return mlir::success();
};

auto zeroReplaceTemplated = []<unsigned actualBitwidth,
     typename, typename MOVri, typename, typename, typename
     >(auto op, auto, mlir::ConversionPatternRewriter& rewriter){
    // TODO warn about this
    //      actually maybe not, MLIR uses undefs so often in normal situations...
    rewriter.replaceOpWithNewOp<MOVri>(op, 0);
    return mlir::success();
};

PATTERN_INT(LLVMUndefPat,  LLVM::UndefOp,  amd64::MOV, zeroReplaceTemplated, intOrFloatBitwidthMatchLambda);
PATTERN_INT(LLVMPoisonPat, LLVM::PoisonOp, amd64::MOV, zeroReplaceTemplated, intOrFloatBitwidthMatchLambda);

auto replaceWithOp0 =  [](auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter){
    // TODO assert pointer/int bitwidths are 64
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return mlir::success();
};

using LLVMFreezePat   = SimplePat<LLVM::FreezeOp,   replaceWithOp0>;
using LLVMIntToPtrPat = SimplePat<LLVM::IntToPtrOp, replaceWithOp0>;
using LLVMPtrToIntPat = SimplePat<LLVM::PtrToIntOp, replaceWithOp0>;

//using LLVMEraseMetadataPat = SimplePat<LLVM::MetadataOp, [](auto op, auto, mlir::ConversionPatternRewriter& rewriter){
//    rewriter.eraseOp(op);
//    return mlir::success();
//}>;

using LLVMUnreachablePat = SimplePat<LLVM::UnreachableOp, [](auto op, auto, mlir::ConversionPatternRewriter& rewriter){
    if(ArgParse::features["unreachable-abort"])
        rewriter.replaceOpWithNewOp<amd64::CALL>(op, TypeRange(), mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "abort"), /* is guaranteed external */ true, ValueRange());
    else
        rewriter.eraseOp(op);
    return mlir::success();
}>;


auto selectMatchReplace = []<unsigned actualBitwidth,
    typename, typename, typename, typename, typename
    >(auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter){
    // to handle the most cases possible, just do branching. For integer selects, CMOVs would theoretically be possible, but hard to get working in SSA, and for FP selects, there are no CMOVs.

    mlir::Block* trueBlock = new mlir::Block();
    rewriter.notifyBlockCreated(trueBlock);
    mlir::Block* falseBlock = new mlir::Block();
    rewriter.notifyBlockCreated(falseBlock);

    mlir::Block* originalBlock = rewriter.getInsertionBlock();
    rewriter.create<LLVM::CondBrOp>(op->getLoc(), op.getCondition(), trueBlock, falseBlock); // TODO translated transitively atm, probably not the fastest, but easier

    mlir::Block* mergeBlock = rewriter.splitBlock(originalBlock, rewriter.getInsertionPoint());
    trueBlock->insertBefore(mergeBlock);
    falseBlock->insertBefore(mergeBlock);

    rewriter.setInsertionPointToEnd(trueBlock);
    rewriter.create<amd64::JMP>(op->getLoc(), mergeBlock, adaptor.getTrueValue());
    rewriter.setInsertionPointToEnd(falseBlock);
    rewriter.create<amd64::JMP>(op->getLoc(), mergeBlock, adaptor.getFalseValue());

    rewriter.setInsertionPointToEnd(mergeBlock); // continue the rewriting there
    rewriter.replaceOp(op, mergeBlock->addArgument(adaptor.getTrueValue().getType(), op->getLoc()));

    return mlir::success();
};
#define SELECT_PAT(bitwidth, lambda) \
    using LLVMSelectPat ## bitwidth = Match<LLVM::SelectOp, bitwidth, lambda>;

SELECT_PAT(8,  selectMatchReplace);
SELECT_PAT(16, selectMatchReplace);
SELECT_PAT(32, selectMatchReplace);
SELECT_PAT(64, selectMatchReplace);

#undef SELECT_PAT

// intrinsics

using LLVMIntrMemCpyPat = SimplePat<LLVM::MemcpyOp, [](auto op, LLVM::MemcpyOp::Adaptor adaptor, mlir::ConversionPatternRewriter& rewriter){
    rewriter.replaceOpWithNewOp<amd64::CALL>(op,
        TypeRange(),
        mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "memcpy"),
        /* is guaranteed external */ true,
        ValueRange({
            adaptor.getDst(),
            adaptor.getSrc(),
            adaptor.getLen()
        })
    );
    return mlir::success();
}, 2>;

using LLVMIntrMemSetPat = SimplePat<LLVM::MemsetOp, [](auto op, auto adaptor, mlir::ConversionPatternRewriter& rewriter){
    // first zero extend the first arg, as it is an 8 bit signless int in LLVM (and interpreted as such by memset), but passed as a 32 bit int
    static_assert(sizeof(int) == 4, "non 32 bit int platforms not supported yet");
    auto zextVal = rewriter.create<amd64::MOVZXr32r8>(op.getLoc(), adaptor.getVal());
    rewriter.replaceOpWithNewOp<amd64::CALL>(op,
        TypeRange(),
        mlir::FlatSymbolRefAttr::get(rewriter.getContext(), "memset"),
        /* is guaranteed external */ true,
        ValueRange({
            adaptor.getDst(),
            zextVal,
            adaptor.getLen()
        })
    );
    return mlir::success();
}, 2>;

template<bool isMin>
auto sminSmaxMatchReplace = []<unsigned,
    typename, typename, typename, typename, typename
    > (auto op, auto, mlir::ConversionPatternRewriter& rewriter){
    LLVM::ICmpPredicate pred;
    if constexpr(isMin)
        pred = LLVM::ICmpPredicate::slt;
    else
        pred = LLVM::ICmpPredicate::sgt;
    auto icmp = rewriter.create<LLVM::ICmpOp>(op.getLoc(), pred, op.getA(), op.getB());
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, icmp, op.getA(), op.getB());
    return mlir::success();
};
PATTERN_INT(LLVMIntrSMinPat, LLVM::SMinOp, amd64::MOV, sminSmaxMatchReplace<true>);
PATTERN_INT(LLVMIntrSMaxPat, LLVM::SMaxOp, amd64::MOV, sminSmaxMatchReplace<false>);

auto absMatchReplace = []<unsigned bitwidth,
    typename XORrr, typename SUBrr, typename SARri, typename, typename
    >(LLVM::AbsOp op, auto adaptor, mlir::ConversionPatternRewriter& rewriter){
    // (see https://stackoverflow.com/a/14194764)
    //auto mask = rewriter.create<SARri>(op.getLoc(), adaptor.getIn());
    //mask.getProperties().instructionInfoImpl.imm = bitwidth - 1;
    //auto xorr = rewriter.create<XORrr>(op.getLoc(), adaptor.getIn(), mask);
    //rewriter.replaceOpWithNewOp<SUBrr>(op, xorr, mask);
    // do it with a select instead
    auto zero = rewriter.create<LLVM::ConstantOp>(op.getLoc(), op.getType(), 0);
    auto isGreaterOrEqualToZero = rewriter.create<LLVM::ICmpOp>(op.getLoc(), LLVM::ICmpPredicate::sge, adaptor.getIn(), zero);
    auto negated = rewriter.create<LLVM::SubOp>(op.getLoc(), zero, op.getIn());
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, isGreaterOrEqualToZero, op.getIn(), negated);
    return mlir::success();
};
#define ABS_PAT(bitwidth)\
    using LLVMIntrAbsPat ## bitwidth = Match<LLVM::AbsOp, bitwidth, absMatchReplace, intBitwidthMatchLambda, 1, amd64::XOR ## bitwidth ## rr, amd64::SUB ## bitwidth ## rr, amd64::SAR ## bitwidth ## ri>;

ABS_PAT(8); ABS_PAT(16); ABS_PAT(32); ABS_PAT(64);


// I guess this is the simplest way of getting rid of these
using LLVMIntrAssumePat        = ErasePat<LLVM::AssumeOp>;
using LLVMIntrLifetimeStartPat = ErasePat<LLVM::LifetimeStartOp>;
using LLVMIntrLifetimeEndPat   = ErasePat<LLVM::LifetimeEndOp>;
using LLVMIntrDbgValuePat      = ErasePat<LLVM::DbgValueOp>;

struct LLVMIntrGenericPat : public mlir::OpConversionPattern<LLVM::CallIntrinsicOp>{
    LLVMIntrGenericPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx) : mlir::OpConversionPattern<LLVM::CallIntrinsicOp>(tc, ctx, 1){}

    mlir::LogicalResult matchAndRewrite(LLVM::CallIntrinsicOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        op.dump();
        // TODO can also ignore some intrinsics, like debug or lifetime related ones
        return rewriter.notifyMatchFailure(op, "unknown intrinsic" + adaptor.getIntrin());
    }

};

// float patterns

// TODO for mcf still necesasry: FMA intr. 2 options: either use FMA extension (but those are vector based instructions), or just split it up
auto fmaMatchReplace = []<unsigned actualBitwidth,
    typename MULS, typename ADDS, typename, typename, typename
    >(LLVM::FMulAddOp op, mlir::OpConversionPattern<LLVM::FMulAddOp>::OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter){
    auto mul = rewriter.create<MULS>(op.getLoc(), adaptor.getA(), adaptor.getB());
    rewriter.replaceOpWithNewOp<ADDS>(op, mul, adaptor.getC());
    return mlir::success();
};

#define FMA_PAT(bitwidth, suff) \
    using LLVMIntrFMulAddPat ## bitwidth = Match<LLVM::FMulAddOp, bitwidth, fmaMatchReplace, floatBitwidthMatchLambda, 2, amd64::MULS ## suff ## rr, amd64::ADDS ## suff ## rr>;

FMA_PAT(32, S);
FMA_PAT(64, D);

// TODO maaaybe separate into 32/64 pat? was necessary for 8/16/32/64, but here its only half...
struct LLVMConstantFloatPat : public mlir::OpConversionPattern<LLVM::ConstantOp> {
    // TODO this is of course a bad idea as a static variable, and cannot possibly work with 32/64 bit float patterns, change this later
    static unsigned floatConstCount;
    GlobalsInfo& globals;

    LLVMConstantFloatPat(mlir::TypeConverter& tc, mlir::MLIRContext* ctx, GlobalsInfo& globals) : mlir::OpConversionPattern<LLVM::ConstantOp>(tc, ctx, 1), globals(globals){}

    // TODO probably rewrite this using integer moves and then moves to float regs, because that doesn't need all these global shenanigans which probably cost a lot of compile time
    mlir::LogicalResult matchAndRewrite(LLVM::ConstantOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        // TODO bitwidth match etc, put this in a proper bitwidth indep structure
        auto floatAttr = adaptor.getValue().template dyn_cast<mlir::FloatAttr>();
        if(!floatAttr)
            return rewriter.notifyMatchFailure(op, "expected float constant");

        double d = floatAttr.getValueAsDouble();
        static_assert(std::endian::native == std::endian::little, "big endian not supported yet");

        if(floatAttr.getType().getIntOrFloatBitWidth() == 32){
            float f = d;

            auto bytesPtr = reinterpret_cast<const char*>(&f);

            auto twine = "f32_" + Twine(floatConstCount++);
            auto str = twine.str();

            globals[str] = GlobalSymbolInfo{.bytes = SmallVector<uint8_t, 8>(bytesPtr, bytesPtr + 4), .alignment = 4};

            rewriter.replaceOpWithNewOp<amd64::MOVSSrm>(op, 
                rewriter.create<amd64::MemB>(op.getLoc(),
                    rewriter.create<amd64::AddrOfGlobal>(op.getLoc(), str)));
        }else{
            auto bytesPtr = reinterpret_cast<const char*>(&d);
            auto twine = "f64_" + Twine(floatConstCount++);
            auto str = twine.str();
            globals[str] = GlobalSymbolInfo{.bytes = SmallVector<uint8_t, 8>(bytesPtr, bytesPtr + 8), .alignment = 8};

            rewriter.replaceOpWithNewOp<amd64::MOVSDrm>(op, 
                rewriter.create<amd64::MemB>(op.getLoc(),
                    rewriter.create<amd64::AddrOfGlobal>(op.getLoc(), str)));
        }

        return mlir::success();
    }
};

// out of line definition for statics is so weird
unsigned LLVMConstantFloatPat::floatConstCount = 0;

#define PATTERN_FLOAT(name, opTy, opPrefixToReplaceWith, lambda, bitwidthMatchLambda, benefit)               \
    using name ## 32  = Match<opTy, 32, lambda, bitwidthMatchLambda, benefit, opPrefixToReplaceWith ## Srr>; \
    using name ## 64  = Match<opTy, 64, lambda, bitwidthMatchLambda, benefit, opPrefixToReplaceWith ## Drr>;

PATTERN_FLOAT(LLVMFAddPat, LLVM::FAddOp, amd64::ADDS, binOpMatchReplace, floatBitwidthMatchLambda, 1)
PATTERN_FLOAT(LLVMFSubPat, LLVM::FSubOp, amd64::SUBS, binOpMatchReplace, floatBitwidthMatchLambda, 1)
PATTERN_FLOAT(LLVMFMulPat, LLVM::FMulOp, amd64::MULS, binOpMatchReplace, floatBitwidthMatchLambda, 1)
PATTERN_FLOAT(LLVMFDivPat, LLVM::FDivOp, amd64::DIVS, binOpMatchReplace, floatBitwidthMatchLambda, 1)

#define FLOAT_CVT_PAT(inBitwidth)                                                                                                                                                                                      \
    using LLVMFPToSIPat ## inBitwidth ## _to_ ## 32 = Match<LLVM::FPToSIOp, 32, llvmTruncExtUiSiMatchReplace<inBitwidth, amd64::SizeChange::None>, llvmTruncExtUiSiBitwidthMatcher<inBitwidth>, 1, amd64::CVT ## SS ## 2 ## SI ## 32 ## rr>; \
    using LLVMFPToSIPat ## inBitwidth ## _to_ ## 64 = Match<LLVM::FPToSIOp, 64, llvmTruncExtUiSiMatchReplace<inBitwidth, amd64::SizeChange::None>, llvmTruncExtUiSiBitwidthMatcher<inBitwidth>, 1, amd64::CVT ## SD ## 2 ## SI ## 64 ## rr>; \
    using LLVMSIToFPPat ## inBitwidth ## _to_ ## 32 = Match<LLVM::SIToFPOp, 32, llvmTruncExtUiSiMatchReplace<inBitwidth, amd64::SizeChange::None>, llvmTruncExtUiSiBitwidthMatcher<inBitwidth>, 1, amd64::CVT ## SI ## 2 ## SS ## 32 ## rr>; \
    using LLVMSIToFPPat ## inBitwidth ## _to_ ## 64 = Match<LLVM::SIToFPOp, 64, llvmTruncExtUiSiMatchReplace<inBitwidth, amd64::SizeChange::None>, llvmTruncExtUiSiBitwidthMatcher<inBitwidth>, 1, amd64::CVT ## SI ## 2 ## SD ## 64 ## rr>;

FLOAT_CVT_PAT(32);
FLOAT_CVT_PAT(64);

#undef FLOAT_CVT_PAT



} // end anonymous namespace

// TODO maybe expose the patterns and let the user decide which ones to use, instead of defining endless populate functions. Type conversions as global lambdas. I'd keep a 'populate all' function though. Biggest problem with this is stupid Cpp with header vs impl files...

void populateLLVMToAMD64TypeConversions(mlir::TypeConverter& tc){
    // TODO this would be a much nicer solution, but for some reason, things that shouldn't accept memlocs (should instead invoke the materialization), do... (call for example)
    //tc.addConversion([](mlir::LLVM::LLVMPointerType type, llvm::SmallVectorImpl<mlir::Type>& possibleTypes) {
    //    possibleTypes.push_back(amd64::gpr64Type::get(type.getContext()));
    //    possibleTypes.push_back(amd64::memLocType::get(type.getContext()));
    //    return mlir::success();
    //});
    tc.addConversion([](mlir::LLVM::LLVMPointerType type) -> std::optional<mlir::Type>{
        return amd64::gpr64Type::get(type.getContext());
    });
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

            default: return {};
        }
    });

    tc.addConversion([](mlir::FloatType type) -> std::optional<mlir::Type>{
        switch(type.getIntOrFloatBitWidth()) {
            // cmp is not matched using the result type (always i1), but with the operand type, so this doesn't apply there.
            case 32:  return amd64::fpr32Type::get(type.getContext());
            case 64:  return amd64::fpr64Type::get(type.getContext());

            default: return {};
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
    // memloc to ptr(i64) conversion
    tc.addTargetMaterialization([](mlir::OpBuilder& builder, amd64::gpr64Type type, mlir::ValueRange inputs, mlir::Location loc) -> std::optional<mlir::Value>{
        if(inputs.size() != 1)
            return nullptr; // unrecoverable materialization, nullopt would mean recoverable
        if(!inputs[0].getType().isa<amd64::memLocType>())
            return {}; // possibly recoverable
        return builder.create<amd64::LEA64rm>(loc, type, inputs);
    });
    // ptr(i64) to memloc conversion
    //tc.addTargetMaterialization([](mlir::OpBuilder& builder, amd64::memLocType type, mlir::ValueRange inputs, mlir::Location loc) -> std::optional<mlir::Value>{
    //    if(inputs.size() != 1 || !inputs[0].getType().isa<amd64::gpr64Type>())
    //        return nullptr; // unrecoverable materialization, nullopt would mean recoverable
    //    if(auto lea = inputs[0].getDefiningOp<amd64::LEA64rm>())
    //        return lea.getMem();
    //
    //    return builder.create<amd64::MemB>(loc, type, inputs);
    //});
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

#define PATTERN_INT_BITWIDTHS(patternName) patternName ## 8, patternName ## 16, patternName ## 32, patternName ## 64
#define FP_PATTERN_BITWIDTHS(pat) pat ## 32, pat ## 64

void populateArithToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc){
    auto* ctx = patterns.getContext();
    // TODO even more fine grained control over which patterns to use
    patterns.add<
        PATTERN_INT_BITWIDTHS(ConstantIntPat),
        PATTERN_INT_BITWIDTHS(AddIPat),
        PATTERN_INT_BITWIDTHS(SubIPat),
        PATTERN_INT_BITWIDTHS(MulIPat),
        PATTERN_INT_BITWIDTHS(CmpIPat),
        PATTERN_INT_BITWIDTHS(AndIPat),
        PATTERN_INT_BITWIDTHS(OrIPat),
        PATTERN_INT_BITWIDTHS(XOrIPat),
        PATTERN_INT_BITWIDTHS(DivUIPat),
        PATTERN_INT_BITWIDTHS(DivSIPat),
        PATTERN_INT_BITWIDTHS(RemSIPat),
        PATTERN_INT_BITWIDTHS(RemSIPat),
        PATTERN_INT_BITWIDTHS(ShlIPat),
        PATTERN_INT_BITWIDTHS(ShrSIPat),
        PATTERN_INT_BITWIDTHS(ShrUIPat),
        ExtUIPat8_to_16, ExtUIPat8_to_32, ExtUIPat8_to_64, ExtUIPat16_to_32, ExtUIPat16_to_64, ExtUIPat32_to_64,
        ExtSIPat8_to_16, ExtSIPat8_to_32, ExtSIPat8_to_64, ExtSIPat16_to_32, ExtSIPat16_to_64, ExtSIPat32_to_64,
        TruncPat16_to_8, TruncPat32_to_8, TruncPat64_to_8, TruncPat32_to_16, TruncPat64_to_16, TruncPat64_to_32
    >(tc, ctx);

    //patterns.add<TestConvPatternWOOp>(tc, ctx);
}

void populateCFToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc){
    auto* ctx = patterns.getContext();
    patterns.add<BrPat, CondBrPat, ReturnPat>(tc, ctx);
}

void populateFuncToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc){
    auto* ctx = patterns.getContext();
    patterns.add<PATTERN_INT_BITWIDTHS(IntCallPat), FP_PATTERN_BITWIDTHS(FloatCallPat)>(tc, ctx);
    mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, tc);
}

void populateLLVMIntArithmeticToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc, GlobalsInfo& globals){
    patterns.add<
        PATTERN_INT_BITWIDTHS(LLVMAddPat),
        PATTERN_INT_BITWIDTHS(LLVMSubPat),
        PATTERN_INT_BITWIDTHS(LLVMAndPat),
        PATTERN_INT_BITWIDTHS(LLVMOrPat),
        PATTERN_INT_BITWIDTHS(LLVMXOrPat),
        PATTERN_INT_BITWIDTHS(LLVMMulPat),
        PATTERN_INT_BITWIDTHS(LLVMUDivPat),
        PATTERN_INT_BITWIDTHS(LLVMSDivPat),
        PATTERN_INT_BITWIDTHS(LLVMURemPat),
        PATTERN_INT_BITWIDTHS(LLVMSRemPat),
        PATTERN_INT_BITWIDTHS(LLVMShlPat),
        PATTERN_INT_BITWIDTHS(LLVMLShrPat),
        PATTERN_INT_BITWIDTHS(LLVMAShrPat)
    >(tc, patterns.getContext());
}

void populateLLVMFloatArithmeticToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc, GlobalsInfo& globals){
    patterns.add<LLVMConstantFloatPat>(tc, patterns.getContext(), globals);
    patterns.add<
        FP_PATTERN_BITWIDTHS(LLVMFAddPat),
        FP_PATTERN_BITWIDTHS(LLVMFSubPat),
        FP_PATTERN_BITWIDTHS(LLVMFMulPat),
        FP_PATTERN_BITWIDTHS(LLVMFDivPat),
        FP_PATTERN_BITWIDTHS(LLVMFloatCallPat),
        LLVMFPToSIPat32_to_32, LLVMFPToSIPat32_to_64, LLVMFPToSIPat64_to_32, LLVMFPToSIPat64_to_64,
        LLVMSIToFPPat32_to_32, LLVMSIToFPPat32_to_64, LLVMSIToFPPat64_to_32, LLVMSIToFPPat64_to_64,
        FP_PATTERN_BITWIDTHS(LLVMFloatLoadPat), FP_PATTERN_BITWIDTHS(LLVMFloatStorePat),
        FP_PATTERN_BITWIDTHS(LLVMIntrFMulAddPat)
        >(tc, patterns.getContext());
#undef FP_PATTERN_BITWIDTHS
}


void populateLLVMMiscToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc, GlobalsInfo& globals){
    patterns.add<
        PATTERN_INT_BITWIDTHS(LLVMConstantIntPat),
        LLVMSelectPat8, LLVMSelectPat16, LLVMSelectPat32, LLVMSelectPat64,
        LLVMNullPat, PATTERN_INT_BITWIDTHS(LLVMUndefPat), PATTERN_INT_BITWIDTHS(LLVMPoisonPat), LLVMFreezePat, LLVMUnreachablePat,
        LLVMGEPPattern, LLVMAllocaPat, PATTERN_INT_BITWIDTHS(LLVMIntLoadPat), PATTERN_INT_BITWIDTHS(LLVMIntStorePat), LLVMPtrToIntPat, LLVMIntToPtrPat,
        LLVMReturnPat, LLVMFuncPat,
        LLVMConstantStringPat, LLVMAddrofPat,
        PATTERN_INT_BITWIDTHS(LLVMIntCallPat),
        LLVMBrPat, LLVMCondBrPat, PATTERN_INT_BITWIDTHS(LLVMSwitchPat),
        PATTERN_INT_BITWIDTHS(LLVMICmpPat),
        LLVMZExtPat8_to_16,  LLVMZExtPat8_to_32,  LLVMZExtPat8_to_64,  LLVMZExtPat16_to_32,  LLVMZExtPat16_to_64,  LLVMZExtPat32_to_64,
        LLVMSExtPat8_to_16,  LLVMSExtPat8_to_32,  LLVMSExtPat8_to_64,  LLVMSExtPat16_to_32,  LLVMSExtPat16_to_64,  LLVMSExtPat32_to_64,
        LLVMTruncPat16_to_8, LLVMTruncPat32_to_8, LLVMTruncPat64_to_8, LLVMTruncPat32_to_16, LLVMTruncPat64_to_16, LLVMTruncPat64_to_32,
        // intrinsics
        LLVMIntrMemCpyPat, LLVMIntrMemSetPat,
        PATTERN_INT_BITWIDTHS(LLVMIntrSMinPat), PATTERN_INT_BITWIDTHS(LLVMIntrSMaxPat), PATTERN_INT_BITWIDTHS(LLVMIntrAbsPat),
        /* these all get replaced with nothing */ LLVMIntrAssumePat, LLVMIntrDbgValuePat, LLVMIntrLifetimeStartPat, LLVMIntrLifetimeEndPat,
        LLVMIntrGenericPat
    >(tc, patterns.getContext());
    patterns.add<LLVMGlobalPat>(tc, patterns.getContext(), globals);
}

void populateAllLLVMToAMD64ConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc, GlobalsInfo& globals){
    populateLLVMMiscToAMD64ConversionPatterns(patterns, tc, globals);
    populateLLVMIntArithmeticToAMD64ConversionPatterns(patterns, tc, globals);
    populateLLVMFloatArithmeticToAMD64ConversionPatterns(patterns, tc, globals);
}


void populateAnyKnownAMD64TypeConversionsConversionPatterns(mlir::RewritePatternSet& patterns, mlir::TypeConverter& tc, GlobalsInfo& globals){
    //populateWithGenerated(patterns);
    populateDefaultTypesToAMD64TypeConversions(tc);
    populateArithToAMD64ConversionPatterns(patterns, tc);
    populateFuncToAMD64ConversionPatterns(patterns, tc);
    populateCFToAMD64ConversionPatterns(patterns, tc);
    populateLLVMToAMD64TypeConversions(tc);
    populateAllLLVMToAMD64ConversionPatterns(patterns, tc, globals);
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

    // TODO this is the current, 'proper' way:
    return mlir::failed(mlir::applyFullConversion(regionOp, target, std::move(patterns)));
}
