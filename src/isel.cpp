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
template<typename opClassToMatch, unsigned bitwidth, typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr, auto lambda, int benefit = 1>
struct MatchRMI : public mlir::OpConversionPattern<opClassToMatch>{
    MatchRMI(mlir::TypeConverter& typeConverter, mlir::MLIRContext* context) : mlir::OpConversionPattern<opClassToMatch>(typeConverter, context, benefit){}
    using OpAdaptor = typename mlir::OpConversionPattern<opClassToMatch>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(opClassToMatch op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override {
        auto typeToMatch= this->template getTypeConverter()->convertType(op.getType()).template dyn_cast<amd64::RegisterTypeInterface>();
        // TODO might have to become actual match failure at some point
        assert(typeToMatch && "expected register type");

        if(typeToMatch.getBitwidth() != bitwidth)
            return rewriter.notifyMatchFailure(op, "bitwidth mismatch");

        return lambda.template operator()<bitwidth, OpAdaptor, INSTrr, INSTri, INSTrm, INSTmi, INSTmr>(op, adaptor, rewriter);
    }
};

#define PATTERN_FOR_BITWIDTH(bitwidth, patternName, opClassToMatch, opPrefixToReplaceWith, lambda, ...)                                                                                                                   \
    using patternName ##  bitwidth = MatchRMI<opClassToMatch,  bitwidth,                                                                                                                                                  \
        opPrefixToReplaceWith ##  bitwidth ## rr, opPrefixToReplaceWith ##  bitwidth ## ri, opPrefixToReplaceWith ##  bitwidth ## rm, opPrefixToReplaceWith ##  bitwidth ## mi, opPrefixToReplaceWith ##  bitwidth ## mr, \
        lambda,## __VA_ARGS__ >;

#define PATTERN(patternName, opClassToMatch, opPrefixToReplaceWith, lambda, ...)              \
    PATTERN_FOR_BITWIDTH(8,  patternName, opClassToMatch, opPrefixToReplaceWith, lambda,## __VA_ARGS__ ) \
    PATTERN_FOR_BITWIDTH(16, patternName, opClassToMatch, opPrefixToReplaceWith, lambda,## __VA_ARGS__ ) \
    PATTERN_FOR_BITWIDTH(32, patternName, opClassToMatch, opPrefixToReplaceWith, lambda,## __VA_ARGS__ ) \
    PATTERN_FOR_BITWIDTH(64, patternName, opClassToMatch, opPrefixToReplaceWith, lambda,## __VA_ARGS__ )

// TODO it would be nice to use folds for matching mov's and folding them into the add, but that's not possible right now, so we either have to match it here (see commit 8df6c7d), or ignore it for now
// TODO an alternative would be to generate custom builders for the RR versions, which check if their argument is a movxxri and then fold it into the RR, resulting in an RI version. That probably wouldn't work because the returned thing would of course expect an RR version, not an RI version
auto binOpMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(auto op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<INSTrr>(op, adaptor.getLhs(), adaptor.getRhs());
    return mlir::success();
};

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
// TODO specify using benefits, that this has to have lower priority than the version which matches a jump with a cmp as an argument (which doesn't exist yet).
PATTERN(CmpIPat, mlir::arith::CmpIOp, amd64::CMP, binOpMatchReplace);
PATTERN(AndIPat, mlir::arith::AndIOp, amd64::AND, binOpMatchReplace);
PATTERN(OrIPat,  mlir::arith::OrIOp,  amd64::OR,  binOpMatchReplace);
PATTERN(XOrIPat, mlir::arith::XOrIOp, amd64::XOR, binOpMatchReplace);

#define MULI_PAT(bitwidth)                                                                       \
    using MulIPat ## bitwidth = MatchRMI<mlir::arith::MulIOp, bitwidth,                          \
        amd64::MUL ## bitwidth ## r, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, NOT_AVAILABLE, \
        binOpMatchReplace>;

MULI_PAT(8); MULI_PAT(16); MULI_PAT(32); MULI_PAT(64);


auto movMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(mlir::arith::ConstantIntOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<INSTri>(op, op.value());
    return mlir::success();
};

PATTERN(ConstantIntPat, mlir::arith::ConstantIntOp, amd64::MOV, movMatchReplace);

} // end anonymous namespace


/// TODO
/// takes an operation and does isel on its regions
void prototypeIsel(mlir::Operation* regionOp){
    // TODO actually use regionOp

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

    assert(mlir::dyn_cast<mlir::arith::ConstantIntOp>(add.getLhs().getDefiningOp()));

    builder.create<mlir::arith::AddIOp>(loc, add, add);

    builder.create<mlir::arith::AddIOp>(loc, imm8_3,  const8);
    builder.create<mlir::arith::AddIOp>(loc, imm16_1, const16);
    builder.create<mlir::arith::AddIOp>(loc, imm32_1, const32);
    builder.create<mlir::arith::AddIOp>(loc, imm64_1, const64);


    builder.create<mlir::func::ReturnOp>(loc);

    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::IntegerType type) -> std::optional<mlir::Type>{
        switch(type.getIntOrFloatBitWidth()) {
            case 8: return  amd64::gpr8Type ::get(type.getContext());
            case 16: return amd64::gpr16Type::get(type.getContext());
            case 32: return amd64::gpr32Type::get(type.getContext());
            case 64: return amd64::gpr64Type::get(type.getContext());

            default: assert(false && "unhandled bitwidth");
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
