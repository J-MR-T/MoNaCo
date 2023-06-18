#include <fadec-enc.h>
#include <fadec.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

// `ForwardIterator` for use with `walk` is defined here ...
#include <mlir/IR/Visitors.h>
// ... but `ReverseIterator`, also for use with `walk`, is defined here:
#include <mlir/IR/Iterators.h>
#include <type_traits>

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
    mlir::DenseMap<mlir::Block*, uint8_t*> blocksToBuffer; // TODO might make sense to make this a reference, and have the regallocer own it, because it needs it too
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

    // TODO these are using absolute jumps, right? Relative makes more sense
    /// if we already know the target block is in the blocksToBuffer map, use that, otherwise, register an unresolved branch, and encode a placeholder
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
    /// if the jump is to the next instruction/block, don't encode it
    inline auto maybeEncodeJump(mlir::Block* targetBB, FeMnem mnemonic, mlir::Block* nextBB) -> int {
        // TODO currently this is left out, because the nextBB is being set linearly at the callsites, but now that we're (correctl) traversing the CFG in RPO, the block ordering in SSA and in machine code are no longer the same

        //if (targetBB == nextBB)
            //return 0;

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
                if(auto blockArg = dyn_cast<mlir::BlockArgument>(operandValue)) [[unlikely]]{
                    // this seems overcomplicated, but it makes the most sense to put all of this functionality into the registerOf method, such that ideally no other code has to ever touch the way registers are stored.
                    operands[machineOperandsCovered] = amd64::registerOf(blockArg, blockArgToRegs);
                }else if(auto asOpResult = dyn_cast<mlir::OpResult>(operandValue)){
                    // as long as there is no 'op-result' interfaces in mlir, this is probably the only way to do it
                    operands[machineOperandsCovered] = amd64::registerOf(asOpResult);
                }else if(auto encodeInterface = dyn_cast<amd64::EncodeOpInterface>(operandValue.getDefiningOp())){
                    operands[machineOperandsCovered] = encodeInterface.encode();
                }else{
                    assert(false && "Operand is neither block argument, nor op result, nor memory op");
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
        // - calls
        // - DIV/IDIV because rdx needs to be zeroed/sign extended (2 separate cases)
        //
        // jumps are handled separately
        
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
        }else{
            failed |= encodeNormally();
        }

        return failed;
    }

    /// returns a pointer to the next instruction to encode.
    [[nodiscard("Saving a pointer to the current instruction needs to be done in the caller, this only returns it.")]] uint8_t* saveCur(){
        return cur;
    }

    /// restores the pointer to the current instruction to the given value.
    void restoreCur(uint8_t* newCur){
        cur = newCur;
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
            llvm::errs() << "encoding resulted in non-decodable instruction :(\n";
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

        llvm::outs() << termcolor::make(termcolor::red, "Decoded assembly:\n");

        for(uint8_t* cur = buf.data(); cur < max;){
            FdInstr instr; 
            auto numBytesEncoded = fd_decode(cur, max - cur, 64, 0, &instr);
            if(numBytesEncoded < 0){
                llvm::errs() << "Encoding resulted in non-decodable instruction :(. Trying to find next decodable instruction...\n";
                cur++;
            }else{
                char fmtbuf[64];
                fd_format(&instr, fmtbuf, sizeof(fmtbuf));
                llvm::outs() <<  fmtbuf << "\n";
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

    /// size of the stack of the current function, measured in bytes from the base pointer. Includes saving of callee-saved registers. This is for easy access to the current top of stack
    uint32_t stackSizeFromBP = 0;
    /// number of bytes used for special purposes, like saving return address and BP
    uint8_t specialStackBytes = 0;
    // -> the sum of these two needs to be 16 byte aligned, that's the criterion we will use to top up the allocation

    /// the SUB64rr rsp, xxx instruction that allocates the stackframe
    uint8_t* stackAllocationInstruction = nullptr;
    /// the ADD64rr rsp, xxx instruction***s*** that deallocate the stackframe
    llvm::SmallVector<uint8_t*, 8> stackDeallocationInstructions;

    /// the number of bytes we need to allocate at the end doesn't include the preallocated stack bytes (specialStackBytes + what is needed for saving callee saved regs), because they are already allocated, so these need to be saved, to be subtracted from the total
    uint8_t preAllocatedStackBytes = 0;

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
        bool failed = false; // TODO |= this in the right places
        // TODO this is not right, doesn't expand the vector yet
        for(auto func : mod.getOps<mlir::func::FuncOp>()){
            if(func.isExternal())
                continue;

            emitPrologue(func);

            // this lambda is only for readability, to separate the traversal from the actual handling of a block
            auto encodeBlock = [&](mlir::Block* block){
                // TODO this is an assertion, for the RPO thingy im trying, but we just stop, for the other algo
                //assert(!encoder.blocksToBuffer.contains(block) && "Already encoded this block");
                if(encoder.blocksToBuffer.contains(block))
                    return;

                encoder.blocksToBuffer[block] = encoder.cur;

                // map block to start of block in buffer

                // iterate over all but the last instruction
                auto endIt = block->end();
                for(auto& op: llvm::make_range(block->begin(), --endIt)){
                    if(auto instr = mlir::dyn_cast<amd64::InstructionOpInterface>(&op)) [[likely]]{
                        // two operands max for now
                        // TODO make this nicer
                        assert(instr->getNumOperands() <= 2);
                        for(auto [i, operand] : llvm::enumerate(instr->getOperands())){
                            loadValueForUse(operand, i, instr.getOperandRegisterConstraints()[i]);
                        }

                        allocateEncodeValueDef(instr);
                    }
                }
                handleTerminator(endIt);
            };

            // do a reverse post order traversal (RPOT) of the CFG, to make sure we encounter all definitions of uses before their uses.
            // except for the entry block, blocks with block args, should have a terminator that jumps to that block visited before the block arg is ever used. So we can allocate a slot at that point
            // a reverse post order traversal, is effectively a topological sorting, by lowest number of *incoming* edges/predecessors.
            // we will make this a DAG, by ignoring back- and self-edges
            // the blocksToBuffer map can already be used as a 'visited' set, because we only add to it, when we will certainly encode the block, so at any point we find a block from it in this traversal, the block is certainly encoded.

            auto* entryBlock = &func.getBlocks().front();
            // TODO this fails, take care of it...
            //assert(entryBlock->hasNoSuccessors() && "Apparently MLIR allows branching to the entry block");  // I'm not sure this would actually break anything, but I am aware of assuming this, so let's assert it
            llvm::SmallVector<mlir::Block*, 4> worklist({entryBlock});

            // TODO this is wrong atm, because if we encounter a loop, we will never enter it
            while(!worklist.empty()){
                auto* currentBlock = worklist.pop_back_val();
                // when we enter this loop with the current block, we *will encode it for certain*
                encodeBlock(currentBlock);

                // TODO reevaluate this code in the loop below, I'm almost sure a DFS pre-order works too, so this is what comes before the continue;
                for(auto* possibleNextBlock : currentBlock->getSuccessors()){
                    if(!encoder.blocksToBuffer.contains(possibleNextBlock))
                        worklist.push_back(possibleNextBlock);
                }

                continue;

                // decide which blocks to encode next
                for(auto* possibleNextBlock : currentBlock->getSuccessors()){
                    // skip self edges *from* the current block, to itself
                    if(currentBlock == possibleNextBlock)
                        continue;

                    // if all predecessors are already encoded , we can encode it
                    bool allPredecessorsEncoded = true;
                    for(auto* pred : possibleNextBlock->getPredecessors())
                    if(pred != possibleNextBlock /* self edges are alright, we can still handle that block, even if we havent seen it*/ && !encoder.blocksToBuffer.contains(pred))
                        allPredecessorsEncoded = false; // TODO could do this with an elaborate &=, see if it makes a performance difference

                    // if we have encoded all predecessors (or the possible next block has only itself as predecessor), we can visit the block now
                    if(allPredecessorsEncoded)
                        worklist.push_back(possibleNextBlock);

                    // if we haven't, then by definition we will see the block again later, when we encode it's last unencoded predecessor, so handle it then
                }
            }

            // TODO resolving unresolvedBranches after every function might give better cache locality, and be better if we don't exceed the in-place-allocated limit, consider doing that

            auto end = encoder.saveCur();

            assert(stackAllocationInstruction);
            // TODO not sure i should assert this
            //assert(!stackDeallocationInstructions.empty()); // at least one return

            assert(specialStackBytes <= preAllocatedStackBytes && "preAllocatedStackBytes should include all bytes, including special ones");
            assert(stackSizeFromBP + specialStackBytes >= preAllocatedStackBytes && "stack + special bytes should be larger than preallocated bytes");


            // fix all stack frame allocations/deallocations with final size of stack frame
            auto totalSize = stackSizeFromBP + specialStackBytes;
            auto paddingForAlignment = 16 - (totalSize % 16);

            // don't need to allocate what's already allocated, but do need to pad
            auto allocationSize = totalSize - preAllocatedStackBytes + paddingForAlignment;

            // immediate encoding is tricky, we can't just encode a bigger immediate into a smaller space, and if we get a bigger space at the start, reencoding with a smaller immediate changes the operand size byte again and now we have too many bytes
            // -> patch in the individual immediate bytes

            auto patch4ByteImm = [](uint8_t* start, uint32_t value){
                // all x86(-64) instructions are little endian, but in accessing allocationSize, we have to take care of endianness

                if constexpr(std::endian::native == std::endian::little){
                    // little endian, so we can just copy the bytes
                    memcpy(start, &value, 4); // TODO can i do this in a more C++-y way?
                }else{
                    static_assert(std::endian::native == std::endian::big && "endianness is neither big nor little, what is it then?");

                    // big endian, so we have to reverse the bytes
                    // TODO this is wrong, seems that std::copy/reverse copy access the memory at +4, which is UB
                    //std::reverse_copy(&value, &value+4, start);
                    static_assert(false && "TODO");
                }
            };
            // allocation
            // `sub rax, 0x01000000` is: 0x 48 81 EC 00 00 00 01
            patch4ByteImm(stackAllocationInstruction+3, allocationSize);

            // deallocation
            for(auto instr : stackDeallocationInstructions){
                encoder.restoreCur(instr);
                // `add rax, 0x01000000` is: 0x 48 81 C4 00 00 00 01
                patch4ByteImm(instr+3, allocationSize);
            }

            encoder.restoreCur(end);

            stackDeallocationInstructions.clear();

#ifndef NDEBUG
            // this is not necessary, but can catch bugs with the assert above, so do it in debug builds
            stackAllocationInstruction = nullptr;
#endif
        }

        failed |= encoder.resolveBranches();

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
    inline bool moveFromOperandRegToSlot(mlir::Value fromVal, ValueSlot toSlot, FeReg operandRegOverride = (FeReg) FE_NOREG){
        /// the register the value is currently in
        FeReg& fromValReg = registerOf(fromVal);

        if(operandRegOverride != (FeReg) FE_NOREG)
            fromValReg = operandRegOverride;

        if(toSlot.kind == ValueSlot::Register){
            FeMnem mnem;
            switch(toSlot.bitwidth){
                // TODO try to eliminate partial register deps
                case 8:  mnem = FE_MOV8rr;  break;
                case 16: mnem = FE_MOV16rr; break;
                case 32: mnem = FE_MOV32rr; break;
                case 64: mnem = FE_MOV64rr; break;
                default: fromVal.dump(); DEBUGLOG("reg: " << toSlot.reg << "\t bw: " << toSlot.bitwidth); assert(false && "invalid bitwidth");
            }
            bool failed = encoder.encodeRaw(mnem, toSlot.reg, fromValReg);

            // TODO think again about whether this is really necessary
            // we're moving into the register of the arg, so overwrite the register of the value to be in the slot, so it's location is up to date again
            // TODO -> this means we have to have already encoded the instruction using the value from the operand register
            fromValReg = toSlot.reg;
            return failed;
        }else{
            assert(toSlot.kind == ValueSlot::Stack && "slot neither register nor stack");
            FeMnem mnem;
            switch(toSlot.bitwidth){
                // TODO try to eliminate partial register deps
                case 8:  mnem = FE_MOV8mr;  break;
                case 16: mnem = FE_MOV16mr; break;
                case 32: mnem = FE_MOV32mr; break;
                case 64: mnem = FE_MOV64mr; break;
                default: assert(false && "invalid bitwidth");
            }
            return encoder.encodeRaw(mnem, toSlot.mem, fromValReg);
        }
    }

    // TODO the case distinction depending on the slot kind can be avoided for allocateEncodeValueDef, by using a template and if constexpr, but see if that actually makes the code faster first
    inline bool moveFromSlotToOperandReg(mlir::Value val, ValueSlot slot, FeReg reg){
        // TODO can probably avoid removing the same value to the same operand register, if it's already there, can probably be checked using registerOf(val) == reg
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
        assert(valueToSlot.contains(val));
        return moveFromSlotToOperandReg(val, valueToSlot[val], reg);
    }

    /// only use: moving phis
    /// returns whether it failed
    inline bool moveFromSlotToSlot(mlir::Value from, mlir::Value to, FeReg memMemMoveReg){
        assert(valueToSlot.contains(to));
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
        // TODO return if something went wrong

        // need to break memory memory moves with a temp register
        FeReg memMemMoveReg = FE_AX;

        /// only guaranteed to be valid if isCriticalEdge is true
        auto condJump = mlir::dyn_cast<amd64::ConditionalJumpInterface>(terminator);
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
                // load the operand into the slot of the arg
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
        auto tryAllocateSlotsForBlockArgsOfSuccessor = [this](mlir::Block::BlockArgListType blockArgs){
            for(auto arg : blockArgs){
                // allocate a slot for the block arg
                // TODO this needs to be functionality of a concrete regallocer implementation, not the abstract one, because it decides where to put block args
                uint8_t bitwidth = arg.getType().cast<amd64::RegisterTypeInterface>().getBitwidth();
                valueToSlot.try_emplace(arg, ValueSlot{.kind = ValueSlot::Stack, .mem = allocateNewStackslot(bitwidth), .bitwidth = bitwidth});
                // TODO a possible optimization would be to use another map for blocks whose blockargs have already been allocated, and skip this loop entirely. Might be worth it for blocks with a lot of args. But might cost performance otherwise
            }
        };

        if(auto ret = mlir::dyn_cast<amd64::RET>(*it)){
            // first load the operand, because it is very probably on the stack, and it will get moved into AX (return reg), which we don't touch in the epilogue
            loadValueForUse(ret.getOperand(), 0, ret.getOperandRegisterConstraints().first);
            emitEpilogue();

            // TODO could also do this directly, see if it makes a performance difference
            encoder.encodeIntraBlockOp(mod, ret);
        }else if (auto jmp = mlir::dyn_cast<amd64::JMP>(*it)){
            auto blockArgs = jmp.getDest()->getArguments();
            tryAllocateSlotsForBlockArgsOfSuccessor(blockArgs);

            auto blockArgOperands = jmp.getDestOperands();

            handleCFGEdgeHelper<false>(blockArgs, blockArgOperands, jmp.getOperation());
            encoder.encodeJMP(jmp);
        }else if(auto jcc = mlir::dyn_cast<amd64::ConditionalJumpInterface>(*it)){
            for(auto dst: {jcc.getTrueDest(), jcc.getFalseDest()}){
                auto blockArgs = dst->getArguments();
                tryAllocateSlotsForBlockArgsOfSuccessor(blockArgs);
            }

            // this is a problem, because we only have a single jump here, which gets encoded into 2 (max) later on, but we need to place the MOVs in between those two
            // so we pass the handleCFGEdge helper as an argument (parameterized depending on whether or not this is a critical edge), to be called by the encoder function
            encoder.encodeJcc<&AbstractRegAllocerEncoder<Derived>::handleCFGEdgeHelper<false>, &AbstractRegAllocerEncoder<Derived>::handleCFGEdgeHelper<true>>(*this, jcc);
        }
    }

    inline void emitPrologue(mlir::func::FuncOp func){
        static_cast<Derived*>(this)->emitPrologueImpl(func);
    }

    inline void emitEpilogue(){
        static_cast<Derived*>(this)->emitEpilogueImpl();
    }

    /// allocates a new stackslot and returns an FeOp that can be used to access it
    inline FeOp allocateNewStackslot(uint8_t bitwidth){ // TODO are there alignment problems here? I think not, but it might be worth aligning every slot to 8 bytes or smth anyway?
        return FE_MEM(FE_BP, 0,0, -(stackSizeFromBP += bitwidth / 8));
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
            FeOp memLoc = allocateNewStackslot(bitwidth);
            std::get<1>(resultInfoForSpilling[i]) = valueToSlot[result] = ValueSlot{.kind = ValueSlot::Kind::Stack, .mem = memLoc, .bitwidth = bitwidth};

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



        // now do the spilling
        for(auto [result, slot, reg] : resultInfoForSpilling){
            moveFromOperandRegToSlot(result, slot, reg);
        }
    }

    void emitPrologueImpl(mlir::func::FuncOp func){
        // - save callee saved registers
        //      - push rbp
        //      - mov rbp, rsp
        //      - push {rbx, r12, r13, r14, r15}
        //      - at this point, the stack is 8 bytes off of being 16 byte aligned, because we've pushed 6*8=3*16 bytes -> the alignment is the same as immediately after the call
        // - stack frame setup
        //      - one sub rsp, xxx to reserve space for spilling etc., this also needs to take care of alignment
        //          - needs to be rewritten at the end, once the final size of the stackframe is known
        // - spill arguments

        encoder.encodeRaw(FE_PUSHr, FE_BP);
        encoder.encodeRaw(FE_MOV64rr, FE_BP, FE_SP);

        for (auto reg : {FE_BX, FE_R12, FE_R13, FE_R14, FE_R15})
            encoder.encodeRaw(FE_PUSHr, reg);

        stackSizeFromBP = 5*8;
        specialStackBytes = 2*8; // return address and saved rbp
        preAllocatedStackBytes = stackSizeFromBP + specialStackBytes;

        // this position needs to be rewritten, with the final stackSizeFromBP at the end
        stackAllocationInstruction = encoder.saveCur();
        encoder.encodeRaw(FE_SUB64ri, FE_SP, 0x01000000); // 0x01000000 isn't the final value, just a placeholder that ensures that the operand size byte is it's maximum possible value -> we have space to encode a big allocation

        // spill args
        static constexpr FeReg argRegs[] = {FE_DI, FE_SI, FE_DX, FE_CX, FE_R8, FE_R9};

        for(auto [i, arg] : llvm::enumerate(func.getArguments())){
            if(i < 6){
                auto argAsRegType = mlir::cast<amd64::RegisterTypeInterface>(arg.getType());
                uint8_t bitwidth = argAsRegType.getBitwidth();
                auto slot = valueToSlot[arg] = ValueSlot{.kind = ValueSlot::Stack, .mem = allocateNewStackslot(bitwidth), .bitwidth = bitwidth};

                moveFromOperandRegToSlot(arg, slot, argRegs[i]);
            }else{
                EXIT_TODO_X("more than 6 args not implemented yet");
            }
        }
    }

    void emitEpilogueImpl(){
        // - stack frame cleanup
        //      - add rsp, stackSize
        //          - needs to be rewritten at the end, once the final size of the stackfrae is known
        // - restore callee saved registers
        //      - pop {r15, r14, r13, r12, rbx}
        //      - mov rsp, rbp
        //      - pop rbp

        // needs to be rewritten later
        stackDeallocationInstructions.push_back(encoder.saveCur());
        encoder.encodeRaw(FE_ADD64ri, FE_SP, 0x01000000); // 0x01000000 isn't the final value, just a placeholder that ensures that the operand size byte is it's maximum possible value -> we have space to encode a big allocation


        for (auto reg : {FE_R15, FE_R14, FE_R13, FE_R12, FE_BX})
            encoder.encodeRaw(FE_POPr, reg);

        encoder.encodeRaw(FE_MOV64rr, FE_SP, FE_BP);
        encoder.encodeRaw(FE_POPr, FE_BP);
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


// TODO parameters for optimization level (-> which regallocer to use)
bool regallocEncode(std::vector<uint8_t>& buf, mlir::ModuleOp mod, bool dumpAsm){
    StackRegAllocer regallocer(mod, buf);
    bool failed = regallocer.run();
    if(dumpAsm)
        regallocer.encoder.dumpAfterEncodingDone();

    // TODO this is a super ugly solution which needs to get fixed later, but since I'm not sure if the vector is even going to stay, i dont want to waste time on a better solution yet
    std::vector copy(buf.begin(), buf.begin() + (regallocer.encoder.cur - regallocer.encoder.buf.data()));
    buf = std::move(copy);
    return failed;
}

// TODO test if this is actually any faster
bool regallocEncodeRepeated(std::vector<uint8_t>& buf, mlir::ModuleOp mod){
    StackRegAllocer regallocer(mod, buf);
    return regallocer.run();
}
