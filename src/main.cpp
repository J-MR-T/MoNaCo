#include <chrono>
#include <string>

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
#include "fallback.h"

// TODO put encoding + regalloc in its own file

inline mlir::Operation* getFuncForCall(mlir::ModuleOp mod, auto call){
    // TODO performance test if this cache is worth it (or even works)
    static llvm::DenseMap<mlir::SymbolRefAttr, mlir::Operation*> cache;

    // get function back from call
    mlir::CallInterfaceCallable callee = call.getCallableForCallee();
    auto [it, _] = cache.try_emplace(callee.get<mlir::SymbolRefAttr>(), mlir::SymbolTable::lookupNearestSymbolFrom(mod, callee.get<mlir::SymbolRefAttr>()));
    return it->second;
}

struct Encoder{
    mlir::ModuleOp mod;
    // TODO test mmap against this later
    // actually not, don't measure writing to a file at all, just write it to mem, compare that against llvm
    std::vector<uint8_t> buf;
    uint8_t* cur;

    // Has to use an actual map instead of a vector, because a jump/call doesn't know the index of the target block
    mlir::DenseMap<mlir::Block*, uint8_t*> blocksToBuffer;

    struct UnresolvedBranchInfo{
        uint8_t* whereToEncode;
        mlir::Block* target;
        FeMnem kind; // this always already contains all the info, for example FE_JMPL, so the resolving routine needs to just pass the value of this to the encoder
        // TODO probably assert FE_JMPL in the resolver
    };
    mlir::SmallVector<UnresolvedBranchInfo, 64> unresolvedBranches;

    Encoder(mlir::ModuleOp mod) : mod(mod), buf(), cur() {
        // reserve 1MiB for now
        buf.reserve(1 << 20);
        cur = buf.data();
    }

    /// can only be called with instructions that can actually be encoded
    /// assumes that there is enough space in the buffer, don't use this is you don't know what you're doing
    /// returns if the encoding failed
    bool encodeOp(amd64::InstructionOpInterface instrOp){

        // there will be many case distinctions here, the alternative would be a general 'encode' interface, where every instruction defines its own encoding.
        // that would be more extensible, but probably even more code, and slower
        using namespace mlir::OpTrait;
        using mlir::dyn_cast;
        using Special = amd64::Special; // TODO unnecessary, as soon as this is properly in the namespace

        assert(cur + 15 <= buf.data() + buf.size() && "Buffer is too small to encode instruction");

        if(instrOp->hasTrait<Operand0IsDestN<0>::Impl>()){
            assert(!(opConstr1.which == 0 || opConstr2.which == 0)  && "Operand 0 is constrained to a register, but is also constrained to the destination register");
        }
        auto [resConstr1, _] = instrOp.getResultRegisterConstraints();

        FeMnem mnemonic = instrOp.getFeMnemonic();

        /// encodes the op that was given, in the non-special case
        auto encodeNormally = [this, &mnemonic, &instrOp, &resConstr1](){
            // actual number of operands should be: op->getNumOperands() + (hasImm?1:0)
            assert(instrOp->getNumOperands() < 4 && "Too many operands for instruction");

            // TODO zeros are fine, right?
            // first operand of the encoded instruction is the result register, if there is one
            FeOp operands[4] = {0};
            unsigned mlirOpOperandsCovered = 0, machineOperandsCovered = 0;

            // TODO neither constrained result, nor constrained operand registers should be passed to the encoder
            // wait, sure about constrained operand registers? I'm pretty sure mul doesn't need AX passed, but what about the shifts? I am also pretty sure that they need CL passed explicitly...
            if(!instrOp->hasTrait<ZeroResults>() && !resConstr1.constrainsReg() /* implies that the second one doesn't constrain either */)
                operands[machineOperandsCovered++] = instrOp.instructionInfo().regs.reg1;

            // fadec only needs the dest1 == op1 once, so in this case we skip that first register operand
            if(instrOp->hasTrait<Operand0IsDestN<0>::Impl>() && 
                    /* first operand is register operand: */
                    instrOp->getNumOperands() > 0 &&
                    instrOp->getOperand(0).getType().isa<amd64::RegisterTypeInterface>()
            )
                mlirOpOperandsCovered++;

            for(; mlirOpOperandsCovered < instrOp->getNumOperands(); mlirOpOperandsCovered++, machineOperandsCovered++){
                assert(machineOperandsCovered < 4 && "Something went deeply wrong, there are more than 4 operands");

                auto operandValue = instrOp->getOperand(mlirOpOperandsCovered);
                if(auto encodeInterface = dyn_cast<amd64::EncodeOpInterface>(operandValue.getDefiningOp())){
                    encodeInterface.dump();
                    operands[machineOperandsCovered] = encodeInterface.encode();
                }else{
                    auto asOpResult = dyn_cast<mlir::OpResult>(operandValue);

                    assert(asOpResult && "Operand is neither a memory op, nor an OpResult");
                    // as long as there is no 'op-result' interface, this is probably the only way to do it
                    operands[machineOperandsCovered] = amd64::registerOf(asOpResult);
                }
            }

            // immediate operand
            if(instrOp->hasTrait<SpecialCase<Special::HasImm>::Impl>()){
                assert(machineOperandsCovered < 3 && "Something went deeply wrong, there are more than 4 operands");
                operands[machineOperandsCovered] = instrOp.instructionInfo().imm;
            }

            // TODO performance test this version, which just passes all operands, against a version which only passes the operands which are actually used, through some hiddeous case distinctions
            // TODO also maybe make the operands smaller once all instructions are defined, and we know that there are no more than x
            return fe_enc64(&cur, mnemonic, operands[0], operands[1], operands[2], operands[3]);
        };

        // if we already know it is in the blocksToBuffer map, use that, otherwise, register an unresolved branch, and encode a placeholder
        auto encodeJump = [this](mlir::Block* targetBB, FeMnem mnemonic) -> int {
            if(auto it = blocksToBuffer.find(targetBB); it != blocksToBuffer.end()){
                // TODO no FE_JMPL needed, right? the encoding can be as small as possible. If it is needed, change it everywhere!
                return fe_enc64(&cur, mnemonic, (intptr_t) it->second);
            }else{
                // with FE_JMPL to ensure enough space
                unresolvedBranches.push_back({cur, targetBB, mnemonic | FE_JMPL});
                // placeholder
                return fe_enc64(&cur, mnemonic | FE_JMPL, (intptr_t) cur);
            }
        };
        // if the jump is to the next instruction/block, don't encode it
        auto maybeEncodeJump = [&encodeJump](mlir::Block* targetBB, FeMnem mnemonic, mlir::Block* nextBB) -> int {
            if (targetBB == nextBB) {
                return 0;
            }
            return encodeJump(targetBB, mnemonic);
        };

        bool failed = false;
        // TODO optimize the ordering of this, probably jmp>call>div
        // TODO special cases
        // - jumps
        // - calls
        // - DIV/IDIV because rdx needs to be zeroed/sign extended (2 separate cases)
        // - returns (wait is that actually a special case?)
        
        if(instrOp->hasTrait<SpecialCase<Special::DIV>::Impl>()) [[unlikely]] {
            assert(cur + 30 <= buf.data() + buf.size() && "Buffer is too small to encode div like instruction");

            // in this case we need to simply XOR edx, edx, which also zeroes the upper 32 bits of rdx
            failed |= fe_enc64(&cur, FE_XOR32rr, FE_DX, FE_DX);

            // then encode the div normally
            encodeNormally();
        }else if(instrOp->hasTrait<SpecialCase<Special::IDIV>::Impl>()) [[unlikely]] {
            assert(cur + 30 <= buf.data() + buf.size() && "Buffer is too small to encode div like instruction");

            auto resultType = instrOp->getResult(0).getType().cast<amd64::RegisterTypeInterface>();
            assert(resultType && "Result of div like instruction is not a register type");

            // the CQO family is FE_C_SEPxx, CBW (which we need for 8 bit div) is FE_C_EX16

            // sign extend ax into dx:ax (for different sizes), for 8 bit sign extend al into ax
            switch(resultType.getBitwidth()){
                case 8:  failed |= fe_enc64(&cur, FE_C_EX16);  break;
                case 16: failed |= fe_enc64(&cur, FE_C_SEP16); break;
                case 32: failed |= fe_enc64(&cur, FE_C_SEP32); break;
                case 64: failed |= fe_enc64(&cur, FE_C_SEP64); break;
                default: assert(false && "Result of div like instruction is not a register type"); // TODO generally think about llvm_unreachable vs assert(false)
            }

            failed |= encodeNormally();
        }else if(auto call = mlir::dyn_cast<amd64::CALL>(instrOp.getOperation())) [[unlikely]] {
            // get the entry block of the corresponding function, jump there
            auto maybeFunc = getFuncForCall(mod, call);
            if(!maybeFunc){
                llvm::errs() << "Call to unknown function, relocations not implemented yet\n";
                return failed = true; // readability
            }

            auto func = mlir::dyn_cast<mlir::func::FuncOp>(maybeFunc);
            assert(func);

            if(func.isExternal()){
                llvm::errs() << "Call to external function, relocations not implemented yet\n";
                return failed = true; // readability
            }

            auto entryBB = &func.getBlocks().front();
            assert(entryBB && "Function has no entry block");

            // can't use maybeEncodeJump here, because a call always needs to be encoded
            failed |= encodeJump(entryBB, mnemonic);
        }else if (auto jmp = mlir::dyn_cast<amd64::JMP>(instrOp.getOperation())) [[unlikely]] {
            auto targetBB = jmp.getDest();

            maybeEncodeJump(targetBB, mnemonic, instrOp->getBlock()->getNextNode());
        }else if(auto jcc = mlir::dyn_cast<amd64::ConditionalJumpInterface>(instrOp.getOperation())) [[unlikely]] {
            auto trueBB = jcc.getTrueDest();
            auto falseBB = jcc.getFalseDest();
            assert(trueBB && falseBB && "Conditional jump has no true or no false destination");

            auto nextBB = instrOp->getBlock()->getNextNode();

            // try to generate minimal jumps (it will always be at least one though)
            if(trueBB == nextBB){
                // if we branch to the subsequent block on true, invert the condition, then encode the conditional branch to the false block, don't do anything else

                // technically mnemonic inversion could be done with just xoring with 1 (the lowest bit is the condition code), but that's not very API-stable...
                mlir::OpBuilder builder(instrOp);
                jcc = jcc.invert(builder); // TODO as mentioned in AMD64Ops.td, this could be improved, by not inverting the condition, but simply inverting the mnemonic

                failed |= encodeJump(falseBB, jcc.getFeMnemonic());
            }else{
                // in this case we can let `encodeJump` take care of generating minimal jumps
                failed |= maybeEncodeJump(trueBB,  mnemonic, nextBB);
                failed |= maybeEncodeJump(falseBB, FE_JMP,   nextBB);
            }
        }else{
            failed |= encodeNormally();
        }

        return failed;
    }

    bool debugEncodeOp(amd64::InstructionOpInterface op){
        auto curBefore = cur;
        auto failed = encodeOp(op);
        if(failed)
            llvm::errs() << "Encoding went wrong :(. Still trying to decode:\n";

        // decode & print to test if it works
        FdInstr instr; 
        failed = fd_decode(curBefore, buf.size() - (curBefore - buf.data()), 64, 0, &instr) < 0;
        if(failed){
            llvm::errs() << "Test encoding resulted in non-decodable instruction :(\n";
        }else{
            char fmtbuf[64];
            fd_format(&instr, fmtbuf, sizeof(fmtbuf));
            llvm::errs() << fmtbuf << "\n";
        }
        return failed;
    }

    /// encodes all ops in the functions contained in the module
    template<auto encodeImpl = &Encoder::encodeOp> // TODO look at the -O3 assembly at some point to make absolutely sure this doesn't actually result in an indirect function call. This should be optimized away
    void encodeModRegion(mlir::ModuleOp mod){
        // TODO this is not right, doesn't expand the vector yet
        for(auto func : mod.getOps<mlir::func::FuncOp>()){
            // regions are assumed to be disjoint
            auto& blocklist = func.getBlocks();
            // TODO prologue
            // - stack frame setup
            // - save callee saved registers
            // - (more?)

            // TODO needs to call regalloc of course, but that's not implemented yet
            for(auto& block : blocklist){
                // map block to start of block in buffer
                blocksToBuffer[&block] = cur;

                for(auto& op : block)
                    if(auto instrOp = mlir::dyn_cast<amd64::InstructionOpInterface>(op)) [[likely]] // don't try to encode memory ops etc.
                        (this->*encodeImpl)(instrOp);
            }
            // TODO resolving unresolvedBranches after every function might give better cache locality, and be better if we don't exceed the in-place-allocated limit, consider doing that

            // TODO epilogue
            // - restore callee saved registers
            // - (more?)
        }
        for(auto [whereToEncode, target, kind] : unresolvedBranches){
            auto it = blocksToBuffer.find(target);
            // TODO think about it again, but I'm pretty sure this can never occur, because we already fail at the call site, if a call target is not a block in the module
            assert(it != blocksToBuffer.end() && "Unresolved branch target is not a block in the module");

            uint8_t* whereToJump = it->second;
            fe_enc64(&whereToEncode, kind, (uintptr_t) whereToJump);
        }
    }

    void dumpAfterEncodingDone(){
        // dump the entire buffer
        // decode & print to test if it works
        auto max = cur;

        for(uint8_t* cur = buf.data(); cur < max;){
            FdInstr instr; 
            auto numBytesEncoded = fd_decode(cur, max - cur, 64, 0, &instr);
            if(numBytesEncoded < 0){
                llvm::errs() << "Test encoding resulted in non-decodable instruction :(. Trying to find next decodable instruction...\n";
                cur++;
            }else{
                char fmtbuf[64];
                fd_format(&instr, fmtbuf, sizeof(fmtbuf));
                llvm::errs() << fmtbuf << "\n";
                cur += numBytesEncoded;
            }
        }
    }
};

// TODO delete all of this later
void testOpCreation(mlir::ModuleOp mod){
    Encoder encoder(mod);
    mlir::MLIRContext* ctx = mod.getContext();

    llvm::errs() << termcolor::red << "=== Encoding tests ===\n" << termcolor::reset ;

    auto gpr8 = amd64::gpr8Type::get(ctx);
    assert(gpr8.isa<amd64::RegisterTypeInterface>() && gpr8.dyn_cast<amd64::RegisterTypeInterface>().getBitwidth() == 8 && "gpr8 is not a register type");

    auto builder = mlir::OpBuilder(ctx);
    auto loc = builder.getUnknownLoc();
    builder.setInsertionPointToStart(mod.getBody());

    auto imm8_1 = builder.create<amd64::MOV8ri>(loc);
    imm8_1.instructionInfo().imm = 1;
    imm8_1.instructionInfo().regs.reg1 = FE_CX;
    auto imm8_2 = builder.create<amd64::MOV8ri>(loc);
    imm8_2.instructionInfo().imm = 2;
    imm8_2.instructionInfo().regs.reg1 = FE_R8;

    auto add8rr = builder.create<amd64::ADD8rr>(loc, imm8_1, imm8_2);
    add8rr.instructionInfo().regs.reg1 = FE_CX;

    mlir::Operation* generic = add8rr;

    auto opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    FeMnem YAAAAAY = opInterface.getFeMnemonic();

    assert(YAAAAAY == FE_ADD8rr);

    auto mul8r = builder.create<amd64::MUL8r>(loc, imm8_1, imm8_2);
    generic = mul8r;
    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    assert(mul8r.hasTrait<mlir::OpTrait::Operand0IsDestN<0>::Impl>());

    auto [resC1, resC2] = mul8r.getResultRegisterConstraints();
    assert(resC1.which == 0 && resC1.reg == FE_AX && resC2.which == 1 && resC2.reg == FE_AH);

    auto mul16r = builder.create<amd64::MUL16r>(loc, imm8_1, imm8_2);
    generic = mul16r;
    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    assert(mul16r.hasTrait<mlir::OpTrait::Operand0IsDestN<0>::Impl>());

    auto [resC3, resC4] = mul16r.getResultRegisterConstraints();
    assert(resC3.which == 0 && resC3.reg == FE_AX && resC4.which == 1 && resC4.reg == FE_DX);

    auto regsTest = builder.create<amd64::CMP8rr>(loc, imm8_1, imm8_2);


    regsTest.instructionInfo().regs = {FE_AX, FE_DX};
    assert(regsTest.instructionInfo().regs.reg1 == FE_AX && regsTest.instructionInfo().regs.reg2 == FE_DX);

    generic = regsTest;
    opInterface = mlir::dyn_cast<amd64::InstructionOpInterface>(generic);
    assert(regsTest.instructionInfo().regs.reg1 == FE_AX && regsTest.instructionInfo().regs.reg2 == FE_DX);

    // immediate stuff
    auto immTest = builder.create<amd64::MOV8ri>(loc, 42);
    immTest.instructionInfo().regs = {FE_AX, FE_DX};

    // encoding test for simple things
    encoder.debugEncodeOp(immTest);
    encoder.debugEncodeOp(add8rr);

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
    sub8mi.instructionInfo().regs.reg1 = FE_BX;
    sub8mi.instructionInfo().imm = 42;

    encoder.debugEncodeOp(sub8mi);

    auto jmpTestFn = builder.create<mlir::func::FuncOp>(loc, "jmpTest", mlir::FunctionType::get(ctx, {}, {}));;
    auto entryBB = jmpTestFn.addEntryBlock();

    auto call = builder.create<amd64::CALL>(loc, gpr8, "jmpTest", mlir::ValueRange{});
    encoder.debugEncodeOp(call);


    auto callOp = builder.create<mlir::func::CallOp>(loc, jmpTestFn, mlir::ValueRange{});
    // get function back from call
    assert(getFuncForCall(mod, callOp) == jmpTestFn);

    // call to unknown function, should return a nullptr
    auto callOp2 = builder.create<mlir::func::CallOp>(loc, "aaa", gpr8, mlir::ValueRange{});
    assert(getFuncForCall(mod, callOp2) == nullptr);


    builder.setInsertionPointToStart(entryBB);
    auto targetBlock1 = jmpTestFn.addBlock();
    auto targetBlock2 = jmpTestFn.addBlock();
    auto imm64 = builder.create<amd64::MOV64ri>(loc, 42);
    builder.create<amd64::ADD64rr>(loc, imm64, imm64);
    builder.create<amd64::JMP>(loc, targetBlock1);
    builder.setInsertionPointToStart(targetBlock1);
    builder.create<amd64::ADD64rr>(loc, imm64, imm64);

    llvm::errs() << termcolor::red << "=== Jump inversion test ===\n" << termcolor::reset ;
    auto jnz = builder.create<amd64::JNZ>(loc, mlir::ValueRange{}, mlir::ValueRange{}, targetBlock1, targetBlock2);
    jmpTestFn.dump();

    auto jz = jnz.invert(builder);
    jnz->replaceAllUsesWith(jz);
    jnz->erase();

    jmpTestFn.dump();
}

/// gives every value in this op's region it's constraint, or rbx as first result register, r8 as second
/// does not recurse into nested regions. i.e. call this on function ops, not on the module op
/// doesn't handle block args at all atm
void dummyRegalloc(mlir::Operation* op){
    assert(op->getNumRegions() == 1); // TODO maybe support more later, lets see

    for(auto& block : op->getRegions().front()){
        for (auto& op : block.getOperations()){
            // TODO block args
            // then probably use more overloads of registerOf, to handle this a bit more generically

            auto instrOp = mlir::dyn_cast<amd64::InstructionOpInterface>(op);
            if(!instrOp) [[unlikely]] // memory op or similar
                continue;

            assert(instrOp->getNumResults() <= 2 && "more than 2 results not supported"); // TODO make proper failure

            auto& instructionInfo = instrOp.instructionInfo();

            // result registers
            bool maybeStillEmptyRegs = instructionInfo.setRegsFromConstraints(instrOp.getResultRegisterConstraints());
            if(maybeStillEmptyRegs){
                // TODO this isn't the cleanest solution, probably make some sort of Register wrapper which can be empty
                if(instructionInfo.regs.reg1Empty())
                    instructionInfo.regs.reg1 = FE_BX;

                if(instructionInfo.regs.reg2Empty())
                    instructionInfo.regs.reg2 = FE_R8;
            }

            // operand register constraints
            auto opConstrs = instrOp.getOperandRegisterConstraints();
            for(auto& opConstr : {opConstrs.first, opConstrs.second}){
                if(opConstr.constrainsReg()){
                    // set the result register of the operands defining instruction to the constrained register
                    auto asOpResult = op.getOperand(opConstr.which).dyn_cast<mlir::OpResult>();
                    assert(asOpResult && "operand register constraint on non-result operand");

                    amd64::registerOf(asOpResult) = opConstr.reg;
                }
            }
        }
    }

}

int main(int argc, char *argv[]) {
#define MEASURE_TIME_START(point) auto point ## _start = std::chrono::high_resolution_clock::now()

#define MEASURE_TIME_END(point) auto point ## _end = std::chrono::high_resolution_clock::now()

#define MEASURED_TIME_AS_SECONDS(point, iterations) std::chrono::duration_cast<std::chrono::duration<double>>(point ## _end - point ## _start).count()/(static_cast<double>(iterations))


    ArgParse::parse(argc, argv);

    auto& args = ArgParse::args;

    if(args.help()){
        ArgParse::printHelp(argv[0]);
        return EXIT_SUCCESS;
    }

    llvm::DebugFlag = args.debug();

    mlir::MLIRContext ctx;
    ctx.loadAllAvailableDialects();
    ctx.loadDialect<amd64::AMD64Dialect>();
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::cf::ControlFlowDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();
    ctx.loadDialect<mlir::LLVM::LLVMDialect>();

    auto inputFile = ArgParse::args.input() ? *ArgParse::args.input : "-";

    auto owningModRef = readMLIRMod(inputFile, ctx);

    if(args.benchmark()){
        // TODO what happens if this throws an exception? Is that fine?
        unsigned iterations = std::stoi(std::string{args.iterations() ? *args.iterations : "1"});

        std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modClones(iterations);
        for(unsigned i = 0; i < iterations; i++){
            modClones[i] = mlir::OwningOpRef<mlir::ModuleOp>(owningModRef->clone());
        }

        if(args.isel()){
            MEASURE_TIME_START(totalMLIR);

            for(unsigned i = 0; i < iterations; i++){
                Encoder encoder(*modClones[i]);
                prototypeIsel(*modClones[i]);
                // to test the encoding
                for(auto& funcOpaque : owningModRef->getOps()){
                    auto func = mlir::dyn_cast<mlir::func::FuncOp>(funcOpaque);
                    assert(func);

                    dummyRegalloc(func);
                }
                encoder.encodeModRegion<&Encoder::encodeOp>(*owningModRef);
            }

            MEASURE_TIME_END(totalMLIR);

            llvm::outs() << "ISel + dummy RegAlloc + encoding took " << MEASURED_TIME_AS_SECONDS(totalMLIR, iterations) << " seconds\n";

            // TODO the encoding certainly won't work with block args currently
        }else if(args.fallback()){
            MEASURE_TIME_START(totalLLVM);

            for(auto i = 0u; i < iterations; i++){
                auto obj = llvm::SmallVector<char, 0>();
                fallbackToLLVMCompilation(*modClones[i], obj);
            }

            MEASURE_TIME_END(totalLLVM);

            llvm::outs() << "LLVM Fallback compilation took " << MEASURED_TIME_AS_SECONDS(totalLLVM, iterations) << " seconds\n";
        }
    }else if(args.isel()){
        Encoder encoder(*owningModRef);
        prototypeIsel(*owningModRef);
        // to test the encoding
        for(auto& funcOpaque : owningModRef->getOps()){
            auto func = mlir::dyn_cast<mlir::func::FuncOp>(funcOpaque);
            assert(func);

            dummyRegalloc(func);
        }
        encoder.encodeModRegion<&Encoder::encodeOp>(*owningModRef);

        if(args.debug()){
            llvm::outs() << "After ISel:\n";
            owningModRef->dump();
            llvm::outs() << "Encoded:\n";
            encoder.dumpAfterEncodingDone();
        }
    }else if(args.fallback()){
        auto obj = llvm::SmallVector<char, 0>();
        return fallbackToLLVMCompilation(*owningModRef, obj);
    }else if(args.debug()){
        mlir::OpBuilder builder(&ctx);
        auto testMod = mlir::OwningOpRef<mlir::ModuleOp>(builder.create<mlir::ModuleOp>(builder.getUnknownLoc()));
        testOpCreation(*testMod);
    }

    return EXIT_SUCCESS;
}
