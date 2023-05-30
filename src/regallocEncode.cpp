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

    // placing this here, because it's related to `encodeOp`
private:

    // if we already know the target block is in the blocksToBuffer map, use that, otherwise, register an unresolved branch, and encode a placeholder
    inline auto encodeJump(mlir::Block* targetBB, FeMnem mnemonic) -> int {
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
    inline auto maybeEncodeJump(mlir::Block* targetBB, FeMnem mnemonic, mlir::Block* nextBB) -> int {
        if (targetBB == nextBB) {
            return 0;
        }
        return encodeJump(targetBB, mnemonic);
    };

public:

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
struct AbstractRegAllocer{
    mlir::DenseMap<mlir::Value, ValueSlot> valueToSlot;

    uint64_t stackSize = 0;
    uint8_t* stackAllocationInstruction = nullptr; // TODO set this

    // TODO these probably have to return iterators to the next instruction to look at, so that the caller knows where to continue iterating
    // TODO also check this for blc, i think i forgot it there

    // TODO for now this is done with a builder, to be a bit more safe and forgiving. If it costs significant amounts of time, maybe try to encode it directly.
    // - this will obviously cost speed
    // - the problem is, that if we do it via the IR, we have no way of setting a register on a spill load/store, beacuse it woudln't be an mlir op, just an encoded instruction. One way around this would be to repeatedly overwrite the register of the value itself, as it's older values are no longer needed, because they are already encoded, but this is quite dangerous
    inline void loadValueForUse(mlir::OpBuilder& builder, mlir::Value val, mlir::OpOperand& use, amd64::OperandRegisterConstraint constraint){
        static_cast<Derived*>(this)->loadValueForUseImpl(builder, val, use, constraint);
    }

    // TODO for now this is done with a builder, to be a bit more safe and forgiving. If it costs significant amounts of time, maybe encode it directly.
    inline void allocateValueDef(mlir::OpBuilder& builder, amd64::InstructionOpInterface def){
        static_cast<Derived*>(this)->allocateValueDefImpl(builder, def);
    }

    inline amd64::InstructionOpInterface moveFromOperandRegToSlot(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val, ValueSlot slot){
#ifndef NDEBUG
        auto instrOpIntVal = mlir::dyn_cast<amd64::InstructionOpInterface>(val);
        assert(instrOpIntVal && "value is not an instruction");
        auto& opReg1 = amd64::registerOf(instrOpIntVal);
        assert((opReg1 == FE_NOREG || opReg1 == FE_AX || opReg1 == FE_CX || opReg1 == FE_DX) && "value is not in operand register");
#endif

        amd64::InstructionOpInterface mov;
        if(slot.kind == ValueSlot::Register){
            switch(slot.bitwidth){
                // TODO try to eliminate partial register deps
                case 8:  mov = builder.create<amd64::MOV8rr> (loc, val); break;
                case 16: mov = builder.create<amd64::MOV16rr>(loc, val); break;
                case 32: mov = builder.create<amd64::MOV32rr>(loc, val); break;
                case 64: mov = builder.create<amd64::MOV64rr>(loc, val); break;
                default: assert(false && "invalid bitwidth");            break;
            }

            // we're moving into the register of the arg
            mov.instructionInfo().regs.reg1 = slot.reg;
        }else{
            assert(slot.kind == ValueSlot::Stack && "slot neither register nor stack");
            // TODO also factor this out later
            auto memOp = builder.create<amd64::RawMemoryOp>(loc, slot.mem);
            switch(slot.bitwidth){
                // TODO try to eliminate partial register deps
                case 8:  mov = builder.create<amd64::MOV8mr> (loc, memOp, val); break;
                case 16: mov = builder.create<amd64::MOV16mr>(loc, memOp, val); break;
                case 32: mov = builder.create<amd64::MOV32mr>(loc, memOp, val); break;
                case 64: mov = builder.create<amd64::MOV64mr>(loc, memOp, val); break;
                default: assert(false && "invalid bitwidth");                   break;
            }
        }
    }

    template<ValueSlot::Kind kind>
    inline amd64::InstructionOpInterface moveFromSlotToOperandReg(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val, ValueSlot slot, FeReg reg = (FeReg) FE_NOREG){
        assert(slot.kind == kind && "slot kind doesn't match with the compile-time chosen version of this function");

        amd64::InstructionOpInterface mov;
        if constexpr(kind == ValueSlot::Register){
            switch(slot.bitwidth){
                // TODO try to eliminate partial register deps
                case 8:  mov = builder.create<amd64::MOV8rr> (loc, val); break;
                case 16: mov = builder.create<amd64::MOV16rr>(loc, val); break;
                case 32: mov = builder.create<amd64::MOV32rr>(loc, val); break;
                case 64: mov = builder.create<amd64::MOV64rr>(loc, val); break;
                default: assert(false && "invalid bitwidth");            break;
            }
        }else{
            assert(kind == ValueSlot::Stack && "slot neither register nor stack");
            // TODO also factor this out later
            auto memOp = builder.create<amd64::RawMemoryOp>(loc, slot.mem);
            switch(slot.bitwidth){
                // TODO try to eliminate partial register deps
                case 8:  mov = builder.create<amd64::MOV8rm> (loc, memOp); break;
                case 16: mov = builder.create<amd64::MOV16rm>(loc, memOp); break;
                case 32: mov = builder.create<amd64::MOV32rm>(loc, memOp); break;
                case 64: mov = builder.create<amd64::MOV64rm>(loc, memOp); break;
                default: assert(false && "invalid bitwidth");              break;
            }
        }

        // we're moving into some operand register
        if(reg != (FeReg) FE_NOREG)
            mov.instructionInfo().regs.reg1 = reg;
    }

    inline amd64::InstructionOpInterface moveFromSlotToOperandReg(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value val){
        auto slot = valueToSlot[val];
        return moveFromSlotToOperandReg<slot.kind>(builder, loc, val, slot);
    }

    // TODO call this in the main routine, by doing a for loop which iterates the block until just before the end, then and calls this separately for the last instruction,
    inline void handleTerminator(mlir::OpBuilder& builder, mlir::Block::iterator it){
        // TODO probably templatize (what a word) this, so it can either do conditional moves for critical edges, or normal moves otherwise
        auto moveBlockArgOperandIntoPlace = [this, &builder](mlir::Location loc, mlir::BlockArgument& arg, mlir::Value operand){
            auto slot = valueToSlot[arg];

            // we're moving from an operand register we don't know yet, to the slot
            amd64::InstructionOpInterface phiMov = moveFromOperandRegToSlot(builder, loc, operand, slot);

            auto* insertPointRestore = &*builder.getInsertionPoint();
            // now load the operand from wherever it is to the operand register, so that it can be used in the mov
            builder.setInsertionPoint(phiMov);
            loadValueForUse(builder, operand, phiMov->getOperand(0), {.which = NO_CONSTRAINT, .reg = FE_NOREG});

            builder.setInsertionPoint(insertPointRestore);
        };

        // We need to deal with PHIs/block args here.
        // Because we're traversing in RPO, we know that the block args are already allocated.
        // Because this is an unconditional branch, we cannot be on a critical edge, so we can just use normal MOVs to load the values
        // We still have to take care of PHI cycles/chains here, i.e. do a kind of topological sort, which handles a cycle by copying one of the values in the cycle to a temporary register

        auto handleCFGEdge = [this, &builder, &moveBlockArgOperandIntoPlace](mlir::Location loc, mlir::Block::BlockArgListType blockArgs, mlir::Operation::operand_range blockArgOperands){

            llvm::SmallVector<std::pair<mlir::BlockArgument*, mlir::BlockArgument*>, 8> mapArgNumberToArgAndArgItReads(blockArgs.size(), {nullptr, nullptr});
            llvm::SmallVector<int16_t, 8> numReaders(blockArgs.size(), 0); // indexed by the block arg number, i.e. the slots for the dependent args are empty/0 in here

            // find the args which are independent, and those which need a topological sorting
            for(auto [arg, operand] : llvm::zip(blockArgs, blockArgOperands)){
                if(auto operandArg = operand.dyn_cast<mlir::BlockArgument>(); operandArg && operandArg.getParentBlock() == arg.getParentBlock()){
                    if(operandArg == arg) { // if its a self-reference/edge, we can simply ignore it, because its SSA: the value only gets defined once, and a ValueSlot only ever holds exactly one value during that value's liveness. In this case this value is this phi, and it is obviously still live, by being used as a block arg, so we can just leave it where it is, don't need to emit any code for it
                        continue;
                    } else /* otherwise we've found a dependant phi */ {
                        // TODO wait i hope this isn't a dangling reference. Does this reference the auto local var here, or does it reference the correct actual block arg?
                        mapArgNumberToArgAndArgItReads[arg.getArgNumber()] = {&arg, &operandArg};

                        assert(numReaders.size()> operandArg.getArgNumber());
                        numReaders[operandArg.getArgNumber()] += 1;
                    }
                }else{
                    // no dependencies, just load the values in any order -> just in the order we encounter them now
                    moveBlockArgOperandIntoPlace(loc, arg, operand);
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

                    // do chain element
                    moveBlockArgOperandIntoPlace(loc, arg, argThatItReads);

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
            amd64::InstructionOpInterface cycleBreakSpillLoad = nullptr;

            // find the first of the cycle
            for(auto [arg, argThatItReads] : mapArgNumberToArgAndArgItReads){
                if(arg == nullptr || numReaders[arg->getArgNumber()] == -1)
                    continue;

                assert(numReaders[arg->getArgNumber()] == 1 && "cycle in phi dependencies, but some of the cycle's members have more than one reader, something is wrong here");

                // save temporarily
                cycleBreakSpillLoad = moveFromSlotToOperandReg(builder, loc, arg);
                amd64::registerOf(cycleBreakSpillLoad) = phiCycleBreakReg;

                while(numReaders[arg->getArgNumber()] == 1){
                    // handle chain element
                    moveBlockArgOperandIntoPlace(loc, arg, argThatItReads);

                    // we've removed a reader from this
                    numReaders[argThatItReads->getArgNumber()] -= 1;

                    arg = argThatItReads; // we're looking at the phi we just read now
                    argThatItReads = mapArgNumberToArgAndArgItReads[arg->getArgNumber()].second; // get what it reads from the map
                }

                // need to move from the cycle break to this last arg, to complete the loop
                moveFromOperandRegToSlot(builder, loc, arg, cycleBreakSpillLoad);

                break;
            }

            // check that none of the instructions use the phi cycle break register in between it being set and it being used at the end
#ifndef NDEBUG
            for(auto cur = cycleBreakSpillLoad->getNextNode(); cur; cur = cur->getNextNode()){
                if(auto curInstr = mlir::dyn_cast<amd64::InstructionOpInterface>(cur))
                    assert(amd64::registerOf(curInstr) != phiCycleBreakReg && "cycle break register used in between it being set and it being used");
            }
#endif
        };

        if(auto ret = mlir::dyn_cast<amd64::RET>(*it)){
            // TODO call emitEpilogue when its finalized
            loadValueForUse(builder, ret.getOperand(), ret.getOperand(), ret.getResultRegisterConstraints().first);
        }else if (auto jmp = mlir::dyn_cast<amd64::JMP>(*it)){
            auto blockArgs = jmp.getDest()->getArguments();
            auto blockArgOperands = jmp.getDestOperands();

            handleCFGEdge(jmp.getLoc(), blockArgs, blockArgOperands);
        }else if(auto jcc = mlir::dyn_cast<amd64::ConditionalJumpInterface>(*it)){
            // TODO this is a problem, because we only have a single jump here, which gets encoded into 2 (max) later on, but we need to place the MOVs in between those two
        }
    }
};

struct StackRegAllocer : public AbstractRegAllocer<StackRegAllocer>{
    void loadValueForUseImpl(mlir::OpBuilder& builder, mlir::Value val, mlir::OpOperand& use, amd64::OperandRegisterConstraint constraint){
        assert(valueToSlot.contains(val) && "value not allocated");

        auto slot = valueToSlot[val];
        assert(slot.kind == ValueSlot::Kind::Stack && "stack register allocator encountered a register slot");

        auto user = mlir::dyn_cast<amd64::InstructionOpInterface>(use.getOwner());
        assert(user && "register allocator encountered a non-instruction user");

        amd64::InstructionOpInterface spillLoad = moveFromSlotToOperandReg<ValueSlot::Stack>(builder, user->getLoc(), val, slot);

        // rewrite the use
        use.set(spillLoad->getResult(0));

        if(constraint.constrainsReg()) [[unlikely]] {
            assert(constraint.which == use.getOperandNumber() && "constraint doesn't match use");
            // TODO make sure this doesn't cost performance
            amd64::registerOf(spillLoad) = constraint.reg;
        }else{
            // TODO allocating the registers in this fixed way has the advantage of compile speed, but it is quite error prone...
            // strategy: the first op can always have AX, because it isn't needed as the constraint for any second op
            //           the second operand can always have CX
            if(use.getOperandNumber() == 0)
                amd64::registerOf(spillLoad) = FE_AX;
            else if (use.getOperandNumber() == 1)
                amd64::registerOf(spillLoad) = FE_CX;
            else
                assert(false && "more than 2 operands not supported yet");
        }
    }

    void allocateValueDefImpl(mlir::OpBuilder& builder, amd64::InstructionOpInterface def){
#ifndef NDEBUG
        for(auto results: def->getResults()){
            assert(!isAllocated(results) && "value already allocated");
        }
#endif

        auto [resConstr1, resConst2] = def.getResultRegisterConstraints();
        auto& instructionInfo = def.instructionInfo();

        for(auto [i, result] : llvm::enumerate(def->getResults())){
            uint8_t bitwidth = mlir::dyn_cast<amd64::RegisterTypeInterface>(result.getType()).getBitwidth();
            auto memLoc = static_cast<FeOp>(FE_MEM(FE_BP, 0,0, stackSize));
            valueToSlot[result] = ValueSlot{.kind = ValueSlot::Kind::Stack, .mem = memLoc, .bitwidth = bitwidth};

            // TODO check that this is done as a shift
            stackSize += bitwidth / 8;

            // allocate registers for the result
            if(i == 0){
                // always assign AX to the first result
                instructionInfo.regs.reg1 = FE_AX;
                assert((!resConstr1.constrainsReg() || resConstr1.which != 0 || resConstr1.reg == FE_AX) && "constraint doesn't match result");
            }else if(i == 1){
                // always assign DX to the second result
                instructionInfo.regs.reg2 = FE_DX;
                assert((!resConst2.constrainsReg() || resConst2.which != 1 || resConst2.reg == FE_DX) && "constraint doesn't match result");
            }else{
                // TODO this doesn't need to be a proper failure for now, right? Because we only define instrs with at most 2 results
                assert(false && "more than 2 results not supported yet");
            }

            auto memoryOp = builder.create<amd64::RawMemoryOp>(def->getLoc(), memLoc);

            // spill store after the instruction
            //TODO this isn't the nicest
            switch(bitwidth){
                case 8:  builder.create<amd64::MOV8mr> (def->getLoc(), memoryOp, result); break;
                case 16: builder.create<amd64::MOV16mr>(def->getLoc(), memoryOp, result); break;
                case 32: builder.create<amd64::MOV32mr>(def->getLoc(), memoryOp, result); break;
                case 64: builder.create<amd64::MOV64mr>(def->getLoc(), memoryOp, result); break;
                default: assert(false && "invalid bitwidth");
            }
        }
    }

private:
    bool isAllocated(mlir::Value val){
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


inline bool regallocEncodeImpl(Encoder& encoder, mlir::ModuleOp mod){
    // TODO update this appropriately everywhere
    bool failed = false;

    // TODO get rid of this, once there's an actual regallocer
	for(auto& funcOpaque : mod.getOps()){
		auto func = mlir::dyn_cast<mlir::func::FuncOp>(funcOpaque);
		assert(func);

		dummyRegalloc(func);
	}

    // TODO this is not right, doesn't expand the vector yet
    for(auto func : mod.getOps<mlir::func::FuncOp>()){
        // regions are assumed to be disjoint
        auto& blocklist = func.getBlocks();

        emitPrologue(encoder);

        // TODO needs to call regalloc of course, but that's not implemented yet
        for(auto& block : blocklist){
            // map block to start of block in buffer
            encoder.blocksToBuffer[&block] = encoder.cur;

            for(auto& op : block)
                if(auto instrOp = mlir::dyn_cast<amd64::InstructionOpInterface>(op)) [[likely]] // don't try to encode memory ops etc.
                    encoder.encodeIntraBlockOp(mod, instrOp);
        }
        // TODO resolving unresolvedBranches after every function might give better cache locality, and be better if we don't exceed the in-place-allocated limit, consider doing that
    }

    failed |= encoder.resolveBranches();
    return failed;
}

// TODO parameters for optimization level (-> which regallocer to use)
bool regallocEncode(std::vector<uint8_t>& buf, mlir::ModuleOp mod, bool debug){
	Encoder encoder(buf);
    bool failed = regallocEncodeImpl(encoder, mod);
    if(debug)
        encoder.dumpAfterEncodingDone();
    return failed;
}

// TODO test if this is actually any faster
bool regallocEncodeRepeated(std::vector<uint8_t>& buf, mlir::ModuleOp mod){
	static Encoder encoder(buf);
    bool failed = regallocEncodeImpl(encoder, mod);
    encoder.reset();
	return failed;
}
