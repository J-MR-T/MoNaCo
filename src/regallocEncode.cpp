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

// TODO very small test indicates, that the cache doesn't work (i.e. performs worse), if there aren't many calls to a function
mlir::Operation* getFuncForCall(mlir::ModuleOp mod, auto call, llvm::DenseMap<mlir::SymbolRefAttr, mlir::Operation*>& cache){
    // get function back from call
    mlir::CallInterfaceCallable callee = call.getCallableForCallee();
    auto [it, _] = cache.try_emplace(callee.get<mlir::SymbolRefAttr>(), mlir::SymbolTable::lookupNearestSymbolFrom(mod, callee.get<mlir::SymbolRefAttr>()));
    return it->second;
}

struct Encoder{
    // TODO test mmap against this later
    // actually not, don't measure writing to a file at all, just write it to mem, compare that against llvm
    std::vector<uint8_t>& buf;
    uint8_t* cur;

    // Has to use an actual map instead of a vector, because a jump/call doesn't know the index of the target block
    mlir::DenseMap<mlir::Block*, uint8_t*> blocksToBuffer;
    llvm::DenseMap<mlir::SymbolRefAttr, mlir::Operation*> symbolrefToFuncCache;

    struct UnresolvedBranchInfo{
        uint8_t* whereToEncode;
        mlir::Block* target;
        FeMnem kind; // this always already contains all the info, for example FE_JMPL, so the resolving routine needs to just pass the value of this to the encoder
        // TODO probably assert FE_JMPL in the resolver
    };
    mlir::SmallVector<UnresolvedBranchInfo, 64> unresolvedBranches;

	Encoder(std::vector<uint8_t>& buf) : buf(buf), cur() {
        // reserve 1MiB for now
        buf.reserve(1 << 20);
        cur = buf.data();
    }

    /// can only be called with instructions that can actually be encoded
    /// assumes that there is enough space in the buffer, don't use this is you don't know what you're doing
    /// returns if the encoding failed
	bool encodeOp(mlir::ModuleOp mod, amd64::InstructionOpInterface instrOp){

        // there will be many case distinctions here, the alternative would be a general 'encode' interface, where every instruction defines its own encoding.
        // that would be more extensible, but probably even more code, and slower
        using namespace mlir::OpTrait;
        using mlir::dyn_cast;
        using Special = amd64::Special; // TODO unnecessary, as soon as this is properly in the namespace

        assert(cur + 15 <= buf.data() + buf.capacity() && "Buffer is too small to encode instruction");

#ifndef NDEBUG
		auto [opConstr1, opConstr2] = instrOp.getOperandRegisterConstraints();
		assert((!instrOp->hasTrait<Operand0IsDestN<0>::Impl>() || !(opConstr1.which == 0 || opConstr2.which == 0))  && "Operand 0 is constrained to a register, but is also constrained to the destination register");
#endif
        auto [resConstr1, _] = instrOp.getResultRegisterConstraints();

        FeMnem mnemonic = instrOp.getFeMnemonic();

		// define some helpers, then employ them in different combinations, depending on the case

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
                // TODO no FE_JMPL needed, right? the encoding can be as small as possible
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
            assert(cur + 30 <= buf.data() + buf.capacity() && "Buffer is too small to encode div like instruction");

            // in this case we need to simply XOR edx, edx, which also zeroes the upper 32 bits of rdx
            failed |= fe_enc64(&cur, FE_XOR32rr, FE_DX, FE_DX);

            // then encode the div normally
            encodeNormally();
        }else if(instrOp->hasTrait<SpecialCase<Special::IDIV>::Impl>()) [[unlikely]] {
            assert(cur + 30 <= buf.data() + buf.capacity() && "Buffer is too small to encode div like instruction");

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
            auto maybeFunc = getFuncForCall(mod, call, symbolrefToFuncCache);
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

	bool debugEncodeOp(mlir::ModuleOp mod, amd64::InstructionOpInterface op){
        auto curBefore = cur;
        auto failed = encodeOp(mod, op);
        if(failed)
            llvm::errs() << "Encoding went wrong :(. Still trying to decode:\n";

        // decode & print to test if it works
        FdInstr instr; 
        failed = fd_decode(curBefore, buf.capacity() - (curBefore - buf.data()), 64, 0, &instr) < 0;
        if(failed){
            llvm::errs() << "Test encoding resulted in non-decodable instruction :(\n";
        }else{
            char fmtbuf[64];
            fd_format(&instr, fmtbuf, sizeof(fmtbuf));
            llvm::errs() << fmtbuf << "\n";
        }
        return failed;
    }

	// TODO probably put this outside the encoder itself, as it also needs to call the register allocation.
    /// encodes all ops in the functions contained in the module
    template<auto encodeImpl = &Encoder::encodeOp> // TODO look at the -O3 assembly at some point to make absolutely sure this doesn't actually result in an indirect function call. This should be optimized away
    bool encodeModRegion(mlir::ModuleOp mod){
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
                        (this->*encodeImpl)(mod, instrOp);
            }
            // TODO resolving unresolvedBranches after every function might give better cache locality, and be better if we don't exceed the in-place-allocated limit, consider doing that

            // TODO epilogue
            // - restore callee saved registers
            // - (more?)
        }
        for(auto [whereToEncode, target, kind] : unresolvedBranches){
			assert(target);
			assert(target->getParent()->getParentOp()->getParentOp() == mod.getOperation() && "Unresolved branch target is not in the module");

            auto it = blocksToBuffer.find(target);
            // TODO think about it again, but I'm pretty sure this can never occur, because we already fail at the call site, if a call target is not a block in the module, but if it can occur normally, make this assertion an actual failure
            assert(it != blocksToBuffer.end() && "Unresolved branch target has not been visited");

            uint8_t* whereToJump = it->second;
            fe_enc64(&whereToEncode, kind, (uintptr_t) whereToJump);
        }

		// TODO return actual failure
		return false;
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

	template<auto encodeImpl = &Encoder::encodeOp>
	static bool encodeOpStateless(mlir::ModuleOp mod, amd64::InstructionOpInterface op){
		std::vector<uint8_t> buf;
		Encoder encoder(buf);
		return (encoder.*encodeImpl)(mod, op);
	}

	void reset(){
		cur = buf.data();
		blocksToBuffer.clear();
		unresolvedBranches.clear();
		symbolrefToFuncCache.clear();
	}
};

bool debugEncodeOp(mlir::ModuleOp mod, amd64::InstructionOpInterface op){
	return Encoder::encodeOpStateless<&Encoder::debugEncodeOp>(mod, op);
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

inline bool regallocEncodeImpl(Encoder& encoder, mlir::ModuleOp mod, bool debug){
	for(auto& funcOpaque : mod.getOps()){
		auto func = mlir::dyn_cast<mlir::func::FuncOp>(funcOpaque);
		assert(func);

		dummyRegalloc(func);
	}
	bool failed;
	if(!debug){
		failed = encoder.encodeModRegion<&Encoder::encodeOp>(mod);
	}else{
		failed = encoder.encodeModRegion<&Encoder::debugEncodeOp>(mod);
		encoder.dumpAfterEncodingDone();
	}
	return failed;
}

// TODO parameters for optimization level (-> which regallocer to use)
bool regallocEncode(std::vector<uint8_t>& buf, mlir::ModuleOp mod, bool debug){
	Encoder encoder(buf);
	return regallocEncodeImpl(encoder, mod, debug);
}

// TODO test if this is actually any faster
bool regallocEncodeRepeated(std::vector<uint8_t>& buf, mlir::ModuleOp mod, bool debug){
	static Encoder encoder(buf);
	bool failed = regallocEncodeImpl(encoder, mod, debug);
	encoder.reset();
	return failed;
}
