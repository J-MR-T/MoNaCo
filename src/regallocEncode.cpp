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
    mlir::DenseMap<mlir::BlockArgument, FeReg>& blockArgToRegs;

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

	Encoder(std::vector<uint8_t>& buf, mlir::DenseMap<mlir::BlockArgument, FeReg>& blockArgToRegs) : buf(buf), cur(), blockArgToRegs(blockArgToRegs){
        // reserve 1MiB for now
        buf.reserve(1 << 20);
        cur = buf.data();
    }

    // placing this here, because it's related to `encodeOp`
private:

    // if we already know the target block is in the blocksToBuffer map, use that, otherwise, register an unresolved branch, and encode a placeholder
    inline auto encodeJump(mlir::Block* targetBB, FeMnem mnemonic) -> int {
        if(auto it = blocksToBuffer.find(targetBB); it != blocksToBuffer.end()){
            // TODO no FE_JMPL needed, right? the encoding can be as small as possible
            return encodeRaw(mnemonic, (intptr_t) it->second);
        }else{
            // with FE_JMPL to ensure enough space
            unresolvedBranches.push_back({cur, targetBB, mnemonic | FE_JMPL});
            // placeholder
            return encodeRaw(mnemonic | FE_JMPL, (intptr_t) cur);
        }
    };
    // if the jump is to the next instruction/block, don't encode it
    inline auto maybeEncodeJump(mlir::Block* targetBB, FeMnem mnemonic, mlir::Block* nextBB) -> int {
        if (targetBB == nextBB)
            return 0;

        return encodeJump(targetBB, mnemonic);
    };

public:
    /// returns whether or not this failed
    template<typename... args_t>
    inline bool encodeRaw(FeMnem mnem, args_t... args){
        // this looks very ugly, but is a result of having to interface with the C API. It's not actually slow, so this is annoying, but not a big problem
        if constexpr(sizeof...(args) == 0)
            return fe_enc64(&cur, mnem);
        else if constexpr(sizeof...(args) == 1)
            return fe_enc64(&cur, mnem, args..., 0, 0, 0);
        else if constexpr(sizeof...(args) == 2)
            return fe_enc64_impl(&cur, mnem, args..., 0, 0);
        else if constexpr(sizeof...(args) == 3)
            return fe_enc64_impl(&cur, mnem, args..., 0);
        else if constexpr(sizeof...(args) == 4)
            return fe_enc64_impl(&cur, mnem, args...);
        else
            static_assert(false, "Too many arguments");
    }

    bool encodeJMP(amd64::JMP jmp){
        auto targetBB = jmp.getDest();

        return maybeEncodeJump(targetBB, jmp.getFeMnemonic(), jmp->getBlock()->getNextNode());
    }

    /// encode a Jcc, encoding the minimal number of jumps
    /// TODO this is a bit hiddeous: have to slightly break abstraction, because this needs the register allocator to do some work
    template<auto destSetupNoCrit, auto destSetupCrit>
    bool encodeJcc(auto& regallocer, amd64::ConditionalJumpInterface jcc){
        auto trueBB = jcc.getTrueDest();
        auto falseBB = jcc.getFalseDest();
        assert(trueBB && falseBB && "Conditional jump has no true or no false destination");

        auto nextBB = jcc->getBlock()->getNextNode();

        // TODO are branches to the entry block valid in MLIR? If so, this needs to be changed, because the entry block has an implicit predecessor
        auto trueIsCritical = trueBB->getSinglePredecessor() == nullptr; // it has at least one predecessor (parent block of jcc), so if it has no single predecessor, it has more than one
        auto falseIsCritical = falseBB->getSinglePredecessor() == nullptr;

        /// TODO also check that this gets compiled into something efficient
        auto setupFalse = [&](){
            if(falseIsCritical)
                (regallocer.*destSetupCrit)(jcc.getFalseDest()->getArguments(), jcc.getFalseDestOperands(), jcc);
            else
                (regallocer.*destSetupNoCrit)(jcc.getFalseDest()->getArguments(), jcc.getFalseDestOperands(), jcc);
        };

        auto setupTrue = [&](){
            if(trueIsCritical)
                (regallocer.*destSetupCrit)(jcc.getTrueDest()->getArguments(), jcc.getTrueDestOperands(), jcc);
            else
                (regallocer.*destSetupNoCrit)(jcc.getTrueDest()->getArguments(), jcc.getTrueDestOperands(), jcc);
        };

        // try to generate minimal jumps (it will always be at least one though)
        bool failed = false;
        if(trueBB == nextBB){
            // if we branch to the subsequent block on true, invert the condition, then encode the conditional branch to the false block, don't do anything else
            setupFalse();
            failed |= encodeJump(falseBB, jcc.invert());
            setupTrue();
        }else{
            // in this case we can let `maybeEncodeJump` take care of generating minimal jumps
            setupTrue();
            failed |= maybeEncodeJump(trueBB,  jcc.getFeMnemonic(), nextBB);
            setupFalse();
            failed |= maybeEncodeJump(falseBB, FE_JMP,   nextBB);
        }
        return failed;
    }

    // TODO look through all the callsites of this, if it still makes sense to use this
    /// can only be called with instructions that can actually be encoded
    /// assumes that there is enough space in the buffer, don't use this is you don't know what you're doing
    /// cannot encode terminators except for return, use encodeTerminatorOp instead
    /// returns if the encoding failed
	bool encodeIntraBlockOp(mlir::ModuleOp mod, amd64::InstructionOpInterface instrOp){

        // there will be many case distinctions here, the alternative would be a general 'encode' interface, where every instruction defines its own encoding.
        // that would be more extensible, but probably even more code, and slower
        using namespace mlir::OpTrait;
        using mlir::dyn_cast;
        using Special = amd64::Special; // TODO unnecessary, as soon as this is properly in the namespace

        assert(cur + 15 <= buf.data() + buf.capacity() && "Buffer is too small to encode instruction");
        assert(!mlir::isa<amd64::JMP>(instrOp.getOperation()) && !mlir::isa<amd64::ConditionalJumpInterface>(instrOp.getOperation()) && "Use encodeJMP or encodeJcc instead");

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
                }else if(auto asOpResult = dyn_cast<mlir::OpResult>(operandValue)){
                    // as long as there is no 'op-result' interfaces in mlir, this is probably the only way to do it
                    operands[machineOperandsCovered] = amd64::registerOf(asOpResult);
                }else{
                    auto blockArg = dyn_cast<mlir::BlockArgument>(operandValue);
                    assert(blockArg && "Operand to encode is neither a memory op, an OpResult, nor a BlockArgument");

                    // this seems overcomplicated, but it makes the most sense to put all of this functionality into the registerOf method, such that ideally no other code has to ever touch the way registers are stored.
                    operands[machineOperandsCovered] = amd64::registerOf(blockArg, blockArgToRegs);
                }
            }

            // immediate operand
            if(instrOp->hasTrait<SpecialCase<Special::HasImm>::Impl>()){
                assert(machineOperandsCovered < 3 && "Something went deeply wrong, there are more than 4 operands");
                operands[machineOperandsCovered] = instrOp.instructionInfo().imm;
            }

            // TODO performance test this version, which just passes all operands, against a version which only passes the operands which are actually used, through some hiddeous case distinctions
            // TODO also maybe make the operands smaller once all instructions are defined, and we know that there are no more than x
            return encodeRaw(mnemonic, operands[0], operands[1], operands[2], operands[3]);
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
            failed |= encodeRaw(FE_XOR32rr, FE_DX, FE_DX);

            // then encode the div normally
            encodeNormally();
        }else if(instrOp->hasTrait<SpecialCase<Special::IDIV>::Impl>()) [[unlikely]] {
            assert(cur + 30 <= buf.data() + buf.capacity() && "Buffer is too small to encode div like instruction");

            auto resultType = instrOp->getResult(0).getType().cast<amd64::RegisterTypeInterface>();
            assert(resultType && "Result of div like instruction is not a register type");

            // the CQO family is FE_C_SEPxx, CBW (which we need for 8 bit div) is FE_C_EX16

            // sign extend ax into dx:ax (for different sizes), for 8 bit sign extend al into ax
            switch(resultType.getBitwidth()){
                case 8:  failed |= encodeRaw(FE_C_EX16);  break;
                case 16: failed |= encodeRaw(FE_C_SEP16); break;
                case 32: failed |= encodeRaw(FE_C_SEP32); break;
                case 64: failed |= encodeRaw(FE_C_SEP64); break;
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
        }else if(auto jcc = mlir::dyn_cast<amd64::ConditionalJumpInterface>(instrOp.getOperation())) [[unlikely]] {
        }else{
            failed |= encodeNormally();
        }

        return failed;
    }

    bool resolveBranches(){
        for(auto [whereToEncode, target, kind] : unresolvedBranches){
            assert(target);
            //assert(target->getParent()->getParentOp()->getParentOp() == mod.getOperation() && "Unresolved branch target is not in the module");

            auto it = blocksToBuffer.find(target);
            // TODO think about it again, but I'm pretty sure this can never occur, because we already fail at the call site, if a call target is not a block in the module, but if it can occur normally, make this assertion an actual failure
            assert(it != blocksToBuffer.end() && "Unresolved branch target has not been visited");

            uint8_t* whereToJump = it->second;
            fe_enc64(&whereToEncode, kind, (uintptr_t) whereToJump);
        }
        return true;
    }

	bool debugEncodeOp(mlir::ModuleOp mod, amd64::InstructionOpInterface op){
        auto curBefore = cur;
        auto failed = encodeIntraBlockOp(mod, op);
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

	template<auto encodeImpl = &Encoder::encodeIntraBlockOp>
	static bool encodeOpStateless(mlir::ModuleOp mod, amd64::InstructionOpInterface op){
		std::vector<uint8_t> buf;
        mlir::DenseMap<mlir::BlockArgument, FeReg> empty;
		Encoder encoder(buf, empty);
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

/// represents a ***permanent*** storage slot/location for a value. i.e. if a value gets moved there, it will always be there, as long as it's live
struct ValueSlot{
    enum Kind{
        Register,
        Stack,
    } kind;

    // this union doesn't really do anything, just makes the code using it a bit nicer
    union{
        /// set only from the FeReg enum
        FeReg reg;
        /// set only via FE_MEM
        FeOp mem; // TODO this is probably actually a bad idea, and I should just have a memory op here. If I decide to encode stuff directly, without the builder, that is
    };
    uint8_t bitwidth;

    static bool isSlotRegister(FeReg reg){
        switch(reg){
            case FE_AX:
            case FE_CX:
            case FE_DX:
            case FE_IP:
            case FE_BP:
            case FE_SP:
                return false;
            default: // TODO technically not all others are, but all that we use are. Maybe add a check for that?
                return true;
        }
    }
};

template<typename Derived>
struct AbstractRegAllocerEncoder{
    /// we store 'where' a value is stored using this map
    /// the register in which the value is found in a particular use of the value, is stored inline with the values definition. For multiple uses, this gets overridden multiple timse, which is fine, because it's all a single pass.
    /// TODO Block args don't have a definition to store the current register inline with, so we will use a registerOf overload and a map for them. Think about whether this is possible to achieve with just a vector as the map. Probably not, because we need to store multiple blocks of block args, so the arg number is not usable as a key.
    // TODO we could try to reduce map usage, and thus hopefully improve performance, by saving the ValueSlot of instruction ops inline, and only using the map for block args. Let's first see if this is even necessary, would probably increase code complexity a lot
    mlir::DenseMap<mlir::Value, ValueSlot> valueToSlot;

    mlir::DenseMap<mlir::BlockArgument, FeReg> blockArgToReg;

    // We're doing single pass register allocation and encoding. But it is not entirely possible to do this, without any coupling between the two components, just using the IR. This is because the instruction-selected IR is still too high-level at some points (conditional branches in particular), and we need to generate a lot of new instructions during regalloc. Some of these can be represented as IR, but for many of them it is easier to simply encode them directly
    // We try to minimize coupling by only having the register allocator know about the encoder, and not also the other way around.
    Encoder encoder;
    
    mlir::ModuleOp mod;

    uint64_t stackSize = 0;
    uint8_t* stackAllocationInstruction = nullptr; // TODO set this

    AbstractRegAllocerEncoder(mlir::ModuleOp mod, std::vector<uint8_t>& buf) : encoder(buf, blockArgToReg), mod(mod) {}

    // TODO also check this for blc, i think i forgot it there

    // repeatedly overwrites the register of the value itself, as it's older values are no longer needed, because they are already encoded, the encoder always uses the current register
    inline void loadValueForUse(mlir::Value val, uint8_t useOperandNumber, amd64::OperandRegisterConstraint constraint){
        static_cast<Derived*>(this)->loadValueForUseImpl(val, useOperandNumber, constraint);
    }

    inline void allocateEncodeValueDef(amd64::InstructionOpInterface def){
        static_cast<Derived*>(this)->allocateEncodeValueDefImpl(def);
    }

    /// returns whether it failed
    bool run(){
        bool failed = false; // TODO
        // TODO this is not right, doesn't expand the vector yet
        for(auto func : mod.getOps<mlir::func::FuncOp>()){
            // TODO uncomment once it exists
            //emitPrologue(func);

            // regions are assumed to be disjoint
            // TODO this can't be in any order, this needs to be RPO
            for(auto& block : func.getBlocks()){
                // map block to start of block in buffer
                encoder.blocksToBuffer[&block] = encoder.cur;

                // iterate over all but the last instruction
                auto endIt = block.end();
                for(auto& op: llvm::make_range(block.begin(), --endIt)){

                    if(auto instr = mlir::dyn_cast<amd64::InstructionOpInterface>(&op)) [[likely]]{
                        DEBUGLOG("Allocating for instruction: " << instr << ":");

                        // two operands max for now
                        // TODO make this nicer
                        assert(instr->getNumOperands() <= 2);
                        for(auto [i, operand] : llvm::enumerate(instr->getOperands())){
                            loadValueForUse(operand, i, instr.getOperandRegisterConstraints()[i]);
                        }

                        // TODO call this as soon as its implemented without a builder
                        allocateEncodeValueDef(instr);
                    }
                }
                handleTerminator(endIt);
            }

            // TODO resolving unresolvedBranches after every function might give better cache locality, and be better if we don't exceed the in-place-allocated limit, consider doing that
        }
        failed|= encoder.resolveBranches();
        return failed;
    }

protected:
    inline FeReg& registerOf(mlir::Value val){
        if(auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)){
            return amd64::registerOf(blockArg, blockArgToReg);
        }else{
            return amd64::registerOf(val);
        }
    }

    /// overload to save a map lookup, if the slot is already known
    inline bool moveFromOperandRegToSlot(mlir::Value val){
        return moveFromOperandRegToSlot(val, valueToSlot[val]);
    }

    // TODO the case distinction depending on the slot kind can be avoided for allocateEncodeValueDef, by using a template and if constexpr, but see if that actually makes the code faster first
    /// move from the register the value is currently in, to the slot, or from the operand register override, if it is set
    inline bool moveFromOperandRegToSlot(mlir::Value val, ValueSlot slot, FeReg operandRegOverride = (FeReg) FE_NOREG){
        /// the register the value is currently in
        FeReg& valReg = registerOf(val);

        if(operandRegOverride != (FeReg) FE_NOREG)
            valReg = operandRegOverride;

        if(slot.kind == ValueSlot::Register){
            FeMnem mnem;
            switch(slot.bitwidth){
                // TODO try to eliminate partial register deps
                case 8:  mnem = FE_MOV8rr;  break;
                case 16: mnem = FE_MOV16rr; break;
                case 32: mnem = FE_MOV32rr; break;
                case 64: mnem = FE_MOV64rr; break;
                default: assert(false && "invalid bitwidth");
            }
            bool failed = encoder.encodeRaw(mnem, slot.reg, valReg);

            // TODO think again about whether this is really necessary
            // we're moving into the register of the arg, so overwrite the register of the value to be in the slot, so it's location is up to date again
            // TODO -> this means we have to have already encoded the instruction using the value from the operand register
            valReg = slot.reg;
            return failed;
        }else{
            assert(slot.kind == ValueSlot::Stack && "slot neither register nor stack");
            FeMnem mnem;
            switch(slot.bitwidth){
                // TODO try to eliminate partial register deps
                case 8:  mnem = FE_MOV8mr;  break;
                case 16: mnem = FE_MOV16mr; break;
                case 32: mnem = FE_MOV32mr; break;
                case 64: mnem = FE_MOV64mr; break;
                default: assert(false && "invalid bitwidth");
            }
            return encoder.encodeRaw(mnem, slot.mem, valReg);
        }
    }

    // TODO the case distinction depending on the slot kind can be avoided for allocateEncodeValueDef, by using a template and if constexpr, but see if that actually makes the code faster first
    inline bool moveFromSlotToOperandReg(mlir::Value val, ValueSlot slot, FeReg reg){
        bool failed;
        if (slot.kind == ValueSlot::Register){
            FeMnem mnem;
            switch(slot.bitwidth){
                // TODO try to eliminate partial register deps
                case 8:  mnem = FE_MOV8rr;  break;
                case 16: mnem = FE_MOV16rr; break;
                case 32: mnem = FE_MOV32rr; break;
                case 64: mnem = FE_MOV64rr; break;
                default: assert(false && "invalid bitwidth");
            }
            failed = encoder.encodeRaw(mnem, reg, slot.reg);
        }else{
            FeMnem mnem;
            switch(slot.bitwidth){
                // TODO try to eliminate partial register deps
                case 8:  mnem = FE_MOV8rm;  break;
                case 16: mnem = FE_MOV16rm; break;
                case 32: mnem = FE_MOV32rm; break;
                case 64: mnem = FE_MOV64rm; break;
                default: assert(false && "invalid bitwidth");
            }
            failed = encoder.encodeRaw(mnem, reg, slot.mem);
        }

        // we're moving into some operand register -> set the val to be in that register
        registerOf(val) = reg;
        // TODO return whether it failed
        return failed;
    }

    inline bool moveFromSlotToOperandReg(mlir::Value val, FeReg reg){
        return moveFromSlotToOperandReg(val, valueToSlot[val], reg);
    }

    /// only use: moving phis
    /// returns whether it failed
    inline bool moveFromSlotToSlot(mlir::Value from, mlir::Value to, FeReg memMemMoveReg){
        // TODO maybe do a case for when the slot is the same

        // simply calling the two movs separately is an option, but in every case but a mem-mem move it will generate one MOVrr too much. That probably doesn't cost much performance though
        // TODO -> improve this later

        auto failed = moveFromSlotToOperandReg(from, memMemMoveReg);
        failed |= moveFromOperandRegToSlot(to, valueToSlot[to], memMemMoveReg);
        return failed;
    }

    /// only use: conditionally moving phis
    // TODO failure etc.
    inline void condMoveFromSlotToSlot(ValueSlot fromSlot, mlir::Value conditionallyMoveTo, FeReg memMemMoveReg, amd64::ConditionalJumpInterface terminator){
        // this is not the most efficient way in terms of result code, but to keep complexity down at the start, this is fine
        // TODO -> improve this later

        // the idea is:
        // 1. unconditionally move the current value conditionallyMoveTo, to a register
        // 2. conditionally move `moveFrom` to that register, using a CMOVxxyyrr or CMOVxxyyrm (xx: bitwidth, yy: condition code)
        // 3. unconditionally move the value back from the register to the slot of conditionallyMoveTo

        // (as stated above, this can be improved, specifically in the register to register, or memory to register case, where we can just use a conditional conditional move directly, without this overriding business)

        // 1.
        moveFromSlotToOperandReg(conditionallyMoveTo, memMemMoveReg);

        // 2.
        // just always use a 64 bit cmov for simplicity, the upper bits don't matter, because the move back to the slot will only use the lower bits
        FeMnem mnem;
        bool isRegister = fromSlot.kind == ValueSlot::Register;
        switch(terminator.getPredicate()){
            case amd64::conditional::Z:  mnem = isRegister ? FE_CMOVZ64rr  : FE_CMOVZ64rm;  break;
            case amd64::conditional::NZ: mnem = isRegister ? FE_CMOVNZ64rr : FE_CMOVNZ64rm; break;
            case amd64::conditional::L:  mnem = isRegister ? FE_CMOVL64rr  : FE_CMOVL64rm;  break;
            case amd64::conditional::GE: mnem = isRegister ? FE_CMOVGE64rr : FE_CMOVGE64rm; break;
            case amd64::conditional::LE: mnem = isRegister ? FE_CMOVLE64rr : FE_CMOVLE64rm; break;
            case amd64::conditional::G:  mnem = isRegister ? FE_CMOVG64rr  : FE_CMOVG64rm;  break;
            case amd64::conditional::C:  mnem = isRegister ? FE_CMOVC64rr  : FE_CMOVC64rm;  break;
            case amd64::conditional::NC: mnem = isRegister ? FE_CMOVNC64rr : FE_CMOVNC64rm; break;
            case amd64::conditional::BE: mnem = isRegister ? FE_CMOVBE64rr : FE_CMOVBE64rm; break;
            case amd64::conditional::A:  mnem = isRegister ? FE_CMOVA64rr  : FE_CMOVA64rm;  break;
            default: assert(false && "invalid condition code");
        }
        encoder.encodeRaw(mnem, memMemMoveReg, isRegister ? fromSlot.reg : fromSlot.mem);

        // 3.
        moveFromOperandRegToSlot(conditionallyMoveTo, valueToSlot[conditionallyMoveTo], memMemMoveReg);
    }

    template<bool isCriticalEdge>
    inline void handleCFGEdgeHelper(mlir::Block::BlockArgListType blockArgs, mlir::Operation::operand_range blockArgOperands, mlir::Operation* terminator){
        // need to break memory memory moves with a temp register
        FeReg memMemMoveReg = FE_AX;

        /// only guaranteed to be valid if isCriticalEdge is true
        auto condJump = mlir::cast<amd64::ConditionalJumpInterface>(terminator);
        if constexpr(isCriticalEdge){
            assert(condJump && "critical edges can only be on conditional jumps");

            assert((condJump.getTrueDest()->getSinglePredecessor() == nullptr || condJump.getFalseDest()->getSinglePredecessor() == nullptr) && "neither of the successors of a supposed critical edge has multiple predecessors");
        }

        auto handleChainElement = [this, memMemMoveReg, condJump](mlir::Value from, mlir::Value to){
            if constexpr(!isCriticalEdge)
                (void) condJump,
                moveFromSlotToSlot(from, to, memMemMoveReg);
            else
                condMoveFromSlotToSlot(valueToSlot[from], to, memMemMoveReg, condJump);
        };

        // TODO do the pointers make sense here?
        llvm::SmallVector<std::pair<mlir::BlockArgument*, mlir::BlockArgument*>, 8> mapArgNumberToArgAndArgItReads(blockArgs.size(), {nullptr, nullptr});
        llvm::SmallVector<int16_t, 8> numReaders(blockArgs.size(), 0); // indexed by the block arg number, i.e. the slots for the dependent args are empty/0 in here

        // find the args which are independent, and those which need a topological sorting
        for(auto [arg, operand] : llvm::zip(blockArgs, blockArgOperands)){
            if(auto operandArg = operand.dyn_cast<mlir::BlockArgument>(); operandArg && operandArg.getParentBlock() == arg.getParentBlock()){
                if(operandArg == arg) { // if its a self-reference/edge, we can simply ignore it, because its SSA: the value only gets defined once, and a ValueSlot only ever holds exactly one value during that value's liveness. In this case this value is this phi, and it is obviously still live, by being used as a block arg, so we can just leave it where it is, don't need to emit any code for it
                    continue;
                } else /* otherwise we've found a dependant phi */ {
                    mapArgNumberToArgAndArgItReads[arg.getArgNumber()] = {&arg, &operandArg};

                    assert(numReaders.size()> operandArg.getArgNumber());
                    numReaders[operandArg.getArgNumber()] += 1;
                }
            }else{
                // no dependencies, just load the values in any order -> just in the order we encounter them now
                handleChainElement(operand, arg);
            }
        }

        // handle in tpoplogical order
        assert(mapArgNumberToArgAndArgItReads.size() < std::numeric_limits<int16_t>::max() && "too many dependent block args"); // TODO maybe make an actual failure?
        for(auto [arg, argThatItReads] : mapArgNumberToArgAndArgItReads){
            // this map contains all phis, so we can just skip the ones which are independent
            if(arg == nullptr)
                continue;

            assert(arg != argThatItReads && "self-reference in phi, should have been handled earlier");

            // this handles the arg, if it has no dependencies anymore. It's a loop, because it might unblock new args, which then need to be handled
            while(numReaders[arg->getArgNumber()] == 0){
                // found the start of a chain

                handleChainElement(*argThatItReads, *arg); // need to move the value being read *from* its slot *to* the slot of the phi

                // to ensure that this arg is never handled again (it might be visited again by the for loop, if it has been handled because a reader has unblocked it), set it's reader number to -1
                numReaders[arg->getArgNumber()] = -1;

                // we've removed a reader from this, might have unblocked it
                numReaders[argThatItReads->getArgNumber()] -= 1;

                // -> try if it's unblocked, by re-setting the arg and argThatItReads
                arg = argThatItReads; // we're looking at the phi we just read now
                argThatItReads = mapArgNumberToArgAndArgItReads[arg->getArgNumber()].second; // get what it reads from the map
            }
        }

        // if there are any entries left which have numReaders != -1, then there is a cycle, and all numReaders != -1 should be exactly == 1
        // break the cycle by copying any of the values in the cycle

        FeReg phiCycleBreakReg = FE_CX;

        // find the first of the cycle
        for(auto [arg, argThatItReads] : mapArgNumberToArgAndArgItReads){
            if(arg == nullptr || numReaders[arg->getArgNumber()] == -1)
                continue;

            assert(numReaders[arg->getArgNumber()] == 1 && "cycle in phi dependencies, but some of the cycle's members have more than one reader, something is wrong here");

            // save temporarily
            moveFromSlotToOperandReg(*arg, phiCycleBreakReg);
            numReaders[arg->getArgNumber()] = 0;

            // we're iterating until what we're reading doesn't have 1 reader -> it has to be the first, so we do the special copy from the cycle break register, after the loop
            while(numReaders[argThatItReads->getArgNumber()] == 1){
                handleChainElement(*argThatItReads, *arg); // need to move the value being read *from* its slot *to* the slot of the phi

                // we've removed a reader from this
                numReaders[argThatItReads->getArgNumber()] = 0;

                arg = argThatItReads; // we're looking at the phi we just read now
                argThatItReads = mapArgNumberToArgAndArgItReads[arg->getArgNumber()].second; // get what it reads from the map
            }

            // need to move from the cycle break to this last arg, to complete the loop
            // and we need to do that conditionally, if this is a critical edge
            if constexpr(!isCriticalEdge){
                moveFromOperandRegToSlot(*arg, valueToSlot[*arg], phiCycleBreakReg); // the problem here is we to move from FE_CX/phiCycleBreakReg, to the slot of the arg, not from the register that the value is currently in, so we have to override it
            }else{
                ValueSlot cycleBreakPsuedoSlot = {.kind = ValueSlot::Register, .reg = phiCycleBreakReg, .bitwidth = valueToSlot[*argThatItReads].bitwidth};

                condMoveFromSlotToSlot(cycleBreakPsuedoSlot, *arg, FE_AX, condJump);
            }

            break;
        }
    }

    inline void handleTerminator(mlir::Block::iterator it){
        // We need to deal with PHIs/block args here.
        // Because we're traversing in RPO, we know that the block args are already allocated.
        // Because this is an unconditional branch, we cannot be on a critical edge, so we can just use normal MOVs to load the values
        // We still have to take care of PHI cycles/chains here, i.e. do a kind of topological sort, which handles a cycle by copying one of the values in the cycle to a temporary register

        if(auto ret = mlir::dyn_cast<amd64::RET>(*it)){
            // TODO call emitEpilogue when its finalized
            loadValueForUse(ret.getOperand(), 0, ret.getOperandRegisterConstraints().first);

            encoder.encodeIntraBlockOp(mod, ret);
        }else if (auto jmp = mlir::dyn_cast<amd64::JMP>(*it)){
            auto blockArgs = jmp.getDest()->getArguments();
            auto blockArgOperands = jmp.getDestOperands();

            handleCFGEdgeHelper<false>(blockArgs, blockArgOperands, jmp.getOperation());
            encoder.encodeJMP(jmp);
        }else if(auto jcc = mlir::dyn_cast<amd64::ConditionalJumpInterface>(*it)){
            // this is a problem, because we only have a single jump here, which gets encoded into 2 (max) later on, but we need to place the MOVs in between those two
            // so we pass the handleCFGEdge helper as an argument (parameterized depending on whether or not this is a critical edge), to be called by the encoder function
            encoder.encodeJcc<&AbstractRegAllocerEncoder<Derived>::handleCFGEdgeHelper<false>, &AbstractRegAllocerEncoder<Derived>::handleCFGEdgeHelper<true>>(*this, jcc);
        }
    }
};

struct StackRegAllocer : public AbstractRegAllocerEncoder<StackRegAllocer>{
public:
    using AbstractRegAllocerEncoder::AbstractRegAllocerEncoder;

    void loadValueForUseImpl(mlir::Value val, uint8_t useOperandNumber, amd64::OperandRegisterConstraint constraint){
        assert(valueToSlot.contains(val) && "value not allocated");

        auto slot = valueToSlot[val];
        assert(slot.kind == ValueSlot::Kind::Stack && "stack register allocator encountered a register slot");

        FeReg whichReg;
        if(constraint.constrainsReg()) [[unlikely]] {
            assert(constraint.which == useOperandNumber && "constraint doesn't match use");
            whichReg = constraint.reg;
        }else{
            // TODO allocating the registers in this fixed way has the advantage of compile speed and simplicity, but it is quite error prone...
            // strategy: the first op can always have AX, because it isn't needed as the constraint for any second op
            //           the second operand can always have CX
            if(useOperandNumber == 0)
                whichReg = FE_AX;
            else if (useOperandNumber == 1)
                whichReg = FE_CX;
            else
                assert(false && "more than 2 operands not supported yet");
        }

        moveFromSlotToOperandReg(val, slot, whichReg);
    }

    void allocateEncodeValueDefImpl(amd64::InstructionOpInterface def){
#ifndef NDEBUG
        for(auto results: def->getResults()){
            assert(!isAllocated(results) && "value already allocated");
        }
#endif

        auto [resConstr1, resConst2] = def.getResultRegisterConstraints();
        auto& instructionInfo = def.instructionInfo();

        // TODO try if an array is more efficient
        llvm::SmallVector<std::tuple<mlir::OpResult, ValueSlot, FeReg>, 2> resultInfoForSpilling(def->getNumResults());

        for(auto [i, result] : llvm::enumerate(def->getResults())){
            std::get<0>(resultInfoForSpilling[i]) = result;

            uint8_t bitwidth = mlir::dyn_cast<amd64::RegisterTypeInterface>(result.getType()).getBitwidth();
            auto memLoc = static_cast<FeOp>(FE_MEM(FE_BP, 0,0, stackSize));
            std::get<1>(resultInfoForSpilling[i]) = valueToSlot[result] = ValueSlot{.kind = ValueSlot::Kind::Stack, .mem = memLoc, .bitwidth = bitwidth};

            // TODO check that this is done as a shift
            stackSize += bitwidth / 8;

            // allocate registers for the result
            if(i == 0){
                // always assign AX to the first result
                std::get<2>(resultInfoForSpilling[i]) = instructionInfo.regs.reg1 = FE_AX;
                assert((!resConstr1.constrainsReg() || resConstr1.which != 0 || resConstr1.reg == FE_AX) && "constraint doesn't match result");
            }else if(i == 1){
                // always assign DX to the second result
                std::get<2>(resultInfoForSpilling[i]) = instructionInfo.regs.reg2 = FE_DX;
                assert((!resConst2.constrainsReg() || resConst2.which != 1 || resConst2.reg == FE_DX) && "constraint doesn't match result");
            }else{
                // TODO this doesn't need to be a proper failure for now, right? Because we only define instrs with at most 2 results
                assert(false && "more than 2 results not supported yet");
            }

            // spill store after the instruction
            // problem: as this is only supposed to happen after the instruction is encoded, and the information we just attached above is needed for encoding, -> we can't do this here, have to do it in a loop after it's encoded
        }

        // encode the instruction, now that it has all the information it needs
        encoder.encodeIntraBlockOp(mod, def);

        def->getNumResults();

        // now do the spilling
        for(auto [result, slot, reg] : resultInfoForSpilling){
            moveFromOperandRegToSlot(result, slot, reg);
        }
    }

private:
    inline bool isAllocated(mlir::Value val){
        return valueToSlot.contains(val);
    }
};

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

// TODO move these next 3 to the AbstractRegAllocerEncoder

// TODO these need a regallocer as an arg
inline void emitPrologue(Encoder& encoder){
    // TODO decide on wether a pure 'encode' wrapper in the Encoder class is unnecessary, or actually makes the code nicer
    // TODO prologue
    // - stack frame setup
    // - save callee saved registers
    // - (more?)
}

inline void emitEpilogue(Encoder& encoder){
    // TODO epilogue
    // - restore callee saved registers
    // - stack frame cleanup
    // - (more?)
}


// TODO parameters for optimization level (-> which regallocer to use)
bool regallocEncode(std::vector<uint8_t>& buf, mlir::ModuleOp mod, bool debug){
	//Encoder encoder(buf, );
    //bool failed = regallocEncodeImpl(encoder, mod);
    //if(debug)
    //    encoder.dumpAfterEncodingDone();
    //return failed;
    StackRegAllocer regallocer(mod, buf);
    bool failed = regallocer.run();
    if(debug)
        regallocer.encoder.dumpAfterEncodingDone();
    return failed;
}

// TODO test if this is actually any faster
bool regallocEncodeRepeated(std::vector<uint8_t>& buf, mlir::ModuleOp mod){
	//static Encoder encoder(buf);
    //bool failed = regallocEncodeImpl(encoder, mod);
    //encoder.reset();
	//return failed;
    return true;
}
