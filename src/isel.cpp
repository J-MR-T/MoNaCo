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

/// somewhat generic pattern matching struct
template<typename opClassToMatch, unsigned bitwidth, typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr, auto lambda>
struct MatchRMI : public mlir::OpConversionPattern<opClassToMatch>{
    using mlir::OpConversionPattern<opClassToMatch>::OpConversionPattern;
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

#define PATTERN(patternName, opClassToMatch, opPrefixToReplaceWith, lambda)                                                                                        \
    using patternName ##  8 = MatchRMI<opClassToMatch,  8,                                                                                                         \
        opPrefixToReplaceWith ##  8rr, opPrefixToReplaceWith ##  8ri, opPrefixToReplaceWith ##  8rm, opPrefixToReplaceWith ##  8mi, opPrefixToReplaceWith ##  8mr, \
        lambda>;                                                                                                                                                   \
    using patternName ##  16 = MatchRMI<opClassToMatch,16,                                                                                                         \
        opPrefixToReplaceWith ## 16rr, opPrefixToReplaceWith ## 16ri, opPrefixToReplaceWith ## 16rm, opPrefixToReplaceWith ## 16mi, opPrefixToReplaceWith ## 16mr, \
        lambda>;                                                                                                                                                   \
    using patternName ## 32 = MatchRMI<opClassToMatch,32,                                                                                                          \
        opPrefixToReplaceWith ## 32rr, opPrefixToReplaceWith ## 32ri, opPrefixToReplaceWith ## 32rm, opPrefixToReplaceWith ## 32mi, opPrefixToReplaceWith ## 32mr, \
        lambda>;                                                                                                                                                   \
    using patternName ## 64 = MatchRMI<opClassToMatch,64,                                                                                                          \
        opPrefixToReplaceWith ## 64rr, opPrefixToReplaceWith ## 64ri, opPrefixToReplaceWith ## 64rm, opPrefixToReplaceWith ## 64mi, opPrefixToReplaceWith ## 64mr, \
        lambda>;                                                                                                                                                   \

// TODO it would be nice to use folds for matching mov's and folding them into the add, but that's not possible right now, so we either have to match it here (see commit 8df6c7d), or ignore it for now
// TODO an alternative would be to generate custom builders for the RR versions, which check if their argument is a movxxri and then fold it into the RR, resulting in an RI version. That probably wouldn't work because the returned thing would of course expect an RR version, not an RI version
auto binOpMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(auto op, OpAdaptor adaptor, mlir ::ConversionPatternRewriter &rewriter) {
    rewriter.replaceOpWithNewOp<INSTrr>(op, adaptor.getLhs(), adaptor.getRhs());
    return mlir::success();
};

PATTERN(AddIPat, mlir::arith::AddIOp, amd64::ADD, binOpMatchReplace);
PATTERN(SubIPat, mlir::arith::SubIOp, amd64::SUB, binOpMatchReplace);
// TODO somehow specify, that this has to have lower priority/benefit (why can i not specify a benefit anywhere in the conversion op :() than the version which matches a jump using a cmp (which doesn't exist yet)
PATTERN(CmpIPat, mlir::arith::CmpIOp, amd64::CMP, binOpMatchReplace);

auto movMatchReplace = []<unsigned actualBitwidth, typename OpAdaptor,
     typename INSTrr, typename INSTri, typename INSTrm, typename INSTmi, typename INSTmr
     >(mlir::arith::ConstantIntOp op, OpAdaptor adaptor, mlir ::ConversionPatternRewriter &rewriter) {
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

    builder.create<mlir::arith::AddIOp>(loc, add, add);

    builder.create<mlir::arith::AddIOp>(loc, imm8_3, const8);
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

#define ADD_PATTERN(patternName) patterns.add<patternName ## 8, patternName ## 16, patternName ## 32, patternName ## 64>(typeConverter, ctx);
    ADD_PATTERN(AddIPat);
    ADD_PATTERN(SubIPat);
    ADD_PATTERN(CmpIPat);
    ADD_PATTERN(ConstantIntPat);
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
