#include <fadec-enc.h>
#include <fadec.h>

#include <llvm/Support/Format.h>
#include <llvm/Support/Alignment.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Analysis/Liveness.h>
#include <dlfcn.h>

#include <type_traits>

#include "isel.h"
#include "util.h"
#include "AMD64/AMD64Dialect.h"
#include "AMD64/AMD64Ops.h"

using GlobalsInfo = amd64::GlobalsInfo;
using Special = amd64::Special;

// TODO reconsider this whole returning failed, doesn't make much sense here, it should never do that.

// TODO very small test indicates, that the cache doesn't work (i.e. performs worse), if there aren't many calls to a function
// TODO a simple string map would probably be better anyways, DenseMaps waste a lot of key space too
mlir::Operation* getFuncForCall(mlir::ModuleOp mod, mlir::CallInterfaceCallable callee, llvm::DenseMap<mlir::SymbolRefAttr, mlir::Operation*>& cache){
    auto [it, _] = cache.try_emplace(callee.get<mlir::SymbolRefAttr>(), mlir::SymbolTable::lookupNearestSymbolFrom(mod, callee.get<mlir::SymbolRefAttr>()));
    return it->second;
}
mlir::Operation* getFuncForCall(mlir::ModuleOp mod, auto callee, llvm::DenseMap<mlir::SymbolRefAttr, mlir::Operation*>& cache){
    // get function back from call
    return getFuncForCall(mod, callee.getCallableForCallee(), cache);
}

// TODO maybe merge this with the AbstractRegAllocerEncoder
struct Encoder{
    uint8_t* cur;
    uint8_t* bufStart;
    uint8_t* bufEnd;
    mlir::DenseMap<mlir::BlockArgument, FeReg>& blockArgToReg; // TODO get rid of this

    // Has to use an actual map instead of a vector, because a jump/call doesn't know the index of the target block
    mlir::DenseMap<mlir::Block*, uint8_t*> blocksToBuffer; // TODO might make sense to make this a reference, and have the regallocer own it, because it needs it too
    // TODO the destructor for this dense map errors for large input files with 'corrupted linked list' when used with -b and 20 iterations, and 'double free or corruption (!prev)' when used without -b (both perf build)
    //      this is most likely due to the vector not being expanded yet

    llvm::DenseMap<mlir::SymbolRefAttr, mlir::Operation*> symbolrefToFuncCache;

    struct UnresolvedBranchInfo{
        uint8_t* whereToEncode;
        mlir::Block* target;
        FeMnem kind; // this always already contains all the info, for example FE_JMPL, so the resolving routine needs to just pass the value of this to the encoder
    };
    mlir::SmallVector<UnresolvedBranchInfo, 64> unresolvedBranches;

    // TODO could also be a template argument, try performance testing that.
    bool jit;

    /// bufEnd works just like an end iterator, it points one *after* the last real byte of the buffer
	Encoder(uint8_t* buf, uint8_t* bufEnd, mlir::DenseMap<mlir::BlockArgument, FeReg>& blockArgToReg, bool jit) : cur(buf), bufStart(buf), bufEnd(bufEnd), blockArgToReg(blockArgToReg), jit(jit){}

    // placing this here, because it's related to `encodeOp`
private:

    // NOTE: Fadec uses relative jumps by default, which is what we want

    /// if we already know the target block is in the blocksToBuffer map, use that, otherwise, register an unresolved branch, and encode a placeholder
    int encodeJump(mlir::Block* targetBB, FeMnem mnemonic) {
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
    int maybeEncodeJump(mlir::Block* targetBB, FeMnem mnemonic, mlir::Block* nextBB) {
        if (targetBB == nextBB)
            return 0;

        return encodeJump(targetBB, mnemonic);
    };

public:
    /// returns whether or not this failed
    /// TODO rework the whole return failed thing in encoding, encoding is technically never suppsoed to fail
    template<typename... args_t>
    bool encodeRaw(FeMnem mnem, args_t... args){
        // TODO performance test if this slows it down
        if (cur >= bufEnd)
            errx(EXIT_FAILURE, "Critical: Out of memory");

        bool failed = false;

        // this looks very ugly, but is a result of having to interface with the C API. It's not actually slow, so this is annoying, but not a big problem
        if constexpr(sizeof...(args) == 0)
            failed = fe_enc64(&cur, mnem);
        else if constexpr(sizeof...(args) == 1)
            failed = fe_enc64(&cur, mnem, args..., 0, 0, 0);
        else if constexpr(sizeof...(args) == 2)
            failed = fe_enc64_impl(&cur, mnem, args..., 0, 0);
        else if constexpr(sizeof...(args) == 3)
            failed = fe_enc64_impl(&cur, mnem, args..., 0);
        else if constexpr(sizeof...(args) == 4)
            failed = fe_enc64_impl(&cur, mnem, args...);
        else
            static_assert(false, "Too many arguments");

        assert(!failed && "encodeRaw failed");
        return failed;
    }

    bool encodeJMP(amd64::JMP jmp, mlir::Block* nextBB = nullptr){
        auto targetBB = jmp.getDest();

        return maybeEncodeJump(targetBB, jmp.getFeMnemonic(), nextBB);
    }

    /// encode a Jcc, encoding the minimal number of jumps
    /// TODO this is a bit hiddeous: have to slightly break abstraction, because this needs the register allocator to do some work
    template<auto destSetupNoCrit, auto destSetupCrit>
    bool encodeJcc(auto& regallocer, amd64::ConditionalJumpInterface jcc, mlir::Block* nextBB = nullptr){
        auto trueBB = jcc.getTrueDest();
        auto falseBB = jcc.getFalseDest();
        assert(trueBB && falseBB && "Conditional jump has no true or no false destination");

        // TODO are branches to the entry block valid in MLIR? If so, this needs to be changed, because the entry block has an implicit predecessor
        auto trueIsCritical = trueBB->getSinglePredecessor() == nullptr; // it has at least one predecessor (parent block of jcc), so if it has no single predecessor, it has more than one
        auto falseIsCritical = falseBB->getSinglePredecessor() == nullptr;

        /// TODO also check that this gets compiled into something efficient
        auto setupFalse = [&](bool crit, amd64::conditional::predicate cmovPredicate = amd64::conditional::NONE){
            if(crit)
                (regallocer.*destSetupCrit)(jcc.getFalseDest()->getArguments(), jcc.getFalseDestOperands(), cmovPredicate);
            else
                (regallocer.*destSetupNoCrit)(jcc.getFalseDest()->getArguments(), jcc.getFalseDestOperands(), cmovPredicate);
        };

        auto setupTrue = [&](bool crit, amd64::conditional::predicate cmovPredicate = amd64::conditional::NONE){
            if(crit)
                (regallocer.*destSetupCrit)(jcc.getTrueDest()->getArguments(), jcc.getTrueDestOperands(), cmovPredicate);
            else
                (regallocer.*destSetupNoCrit)(jcc.getTrueDest()->getArguments(), jcc.getTrueDestOperands(), cmovPredicate);
        };

        // try to generate minimal jumps (it will always be at least one though)
        bool failed = false;
        if(trueBB == nextBB){
            // if we branch to the subsequent block on true, invert the condition, then encode the conditional branch to the false block, don't do anything else
            setupFalse(falseIsCritical, amd64::conditional::invert(jcc.getPredicate()));
            failed |= encodeJump(falseBB, jcc.getInvertedMnem());
            // if we reach here, this branch always gets taken, so we we don't need to encode conditional movs
            setupTrue(false);
        }else{
            // in this case we can let `maybeEncodeJump` take care of generating minimal jumps
            setupTrue(trueIsCritical, jcc.getPredicate());
            failed |= maybeEncodeJump(trueBB,  jcc.getFeMnemonic(), nextBB);
            // if we reach here, this branch always gets taken, so we we don't need to encode conditional movs
            setupFalse(false);
            failed |= maybeEncodeJump(falseBB, FE_JMP, nextBB);
        }
        return failed;
    }


    bool encodeCall(mlir::ModuleOp mod, amd64::CALL call){
        // indirect calls via pointer, that pointer has already been moved to FE_AX
        auto fnNameOpt = call.getCallee();
        if(!fnNameOpt.has_value()){
            encodeRaw(FE_CALLr, FE_AX);
            return false;
        }
        auto fnName = *fnNameOpt;

        // has to be a direct call

        auto handleCallToExternal = [&](){
            if(jit){
                auto name = fnName;

                // resolve symbol
                // TODO try if caching dlsym results improves performance
                intptr_t symbolPtr = (intptr_t) checked_dlsym(name);

                assert(symbolPtr && "Symbol not found");

                // TODO make this a direct call
                //      doesn't seem possible, this is basically guaranteed to be more than 2GB away
                encodeRaw(FE_MOV64ri, FE_AX, symbolPtr);
                return encodeRaw(FE_CALLr, FE_AX);
            }else{
                llvm::errs() << "Call to external function, relocations not implemented yet\n";
                return true;
            }
        };

        // this also includes functions, that don't necessarily have an MLIR declaration, that getFuncForCall could find. Examples include intrinsic translation such as memcpy, memset, or stuff like abort
        if(call.getIsGuaranteedExternal())
            return handleCallToExternal();

        // get the entry block of the corresponding function, jump there
        auto maybeFunc = getFuncForCall(mod, call, symbolrefToFuncCache);
        if(!maybeFunc){
            llvm::errs() << "Call to unknown function, relocations not implemented yet\n";
            return true;
        }

        auto func = mlir::cast<mlir::func::FuncOp>(maybeFunc);

        if(func.isExternal())
            return handleCallToExternal();

        auto entryBB = &func.getBlocks().front();
        assert(entryBB && "Function has no entry block");

        // can't use maybeEncodeJump here, because a call always needs to be encoded
        return encodeJump(entryBB, FE_CALL);
    }

    /// can only be called with instructions that can actually be encoded
    /// assumes that there is enough space in the buffer, don't use this is you don't know what you're doing
    /// cannot encode terminators except for return, use encodeJMP/encodeJcc instead
    /// returns whether the encoding failed
	bool encodeIntraBlockOp(amd64::InstructionOpInterface instrOp){

        // there will be many case distinctions here, the alternative would be a general 'encode' interface, where every instruction defines its own encoding.
        // that would be more extensible, but probably even more code, and slower
        using namespace mlir::OpTrait;
        using mlir::dyn_cast;

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
                // first operand is register operand:
                instrOp->getNumOperands() > 0 &&
                instrOp->getOperand(0).getType().isa<amd64::RegisterTypeInterface>()
            ){
                mlirOpOperandsCovered++;

#ifndef NDEBUG
                // check that the operand register and the result register match
                FeReg regOfOperand;
                regOfOperand = amd64::registerOf(instrOp->getOperand(0), &blockArgToReg);
                // either all operands are the same, or the first operand is the destination register
                // TODO this is just a temporary solution to catch more bugs. In actuality, it's not okay if an op has multiple equal operands and the result register differs, because if we use that value, we will use it from the wrong point
                assert(((instrOp->getOperands().size() >= 2 && llvm::all_equal(instrOp->getOperands())) || regOfOperand == instrOp.instructionInfo().regs.reg1) && "Operand 0 is constrained to the destination register, but operand 0 register and destination register differ");
#endif
            }

            for(; mlirOpOperandsCovered < instrOp->getNumOperands(); mlirOpOperandsCovered++, machineOperandsCovered++){
                assert(machineOperandsCovered < 4 && "Something went deeply wrong, there are more than 4 operands");

                auto operandValue = instrOp->getOperand(mlirOpOperandsCovered);
                // TODO performance test the ordering here, try to get the op result closer to the start
                if(auto blockArg = dyn_cast<mlir::BlockArgument>(operandValue)) [[unlikely]]{
                    // first check if its a block arg, otherwise it has to be an op result
                    // this seems overcomplicated, but it makes the most sense to put all of this functionality into the registerOf method, such that ideally no other code has to ever touch the way registers are stored.
                    operands[machineOperandsCovered] = amd64::registerOf(blockArg, blockArgToReg);
                }else if(auto encodeInterface = dyn_cast<amd64::EncodeOpInterface>(operandValue.getDefiningOp())) [[unlikely]]{
                    // first check encode op interface, because everything that's not a blockarg is an op result
                    operands[machineOperandsCovered] = encodeInterface.encode(&blockArgToReg);
                }else{
                    auto asOpResult = mlir::cast<mlir::OpResult>(operandValue);
                    // as long as there are no 'op-result' interfaces in mlir, this is probably the only way to do it
                    operands[machineOperandsCovered] = amd64::registerOf(asOpResult);
                }
            }

            // immediate operand
            if(instrOp->hasTrait<SpecialCase<Special::HasImm>::Impl>()){
                assert(machineOperandsCovered < 3 && "Something went deeply wrong, there are more than 4 operands");
                operands[machineOperandsCovered] = instrOp.instructionInfo().imm;
            }

            // TODO also maybe make the operands smaller once all instructions are defined, and we know that there are no more than x
            return encodeRaw(mnemonic, operands[0], operands[1], operands[2], operands[3]);
        };

        bool failed = false;
        // special cases
        // - calls: not handled here, see encodeCall
        // - DIV/IDIV because rdx needs to be zeroed/sign extended (2 separate cases)
        // - jumps: not handled here, see encodeJMP/encodeJcc
        assert(mnemonic != FE_CALL && "Calls are handled by encodeCall");
        assert(mnemonic != FE_JMP && "Unconditional jumps are handled by encodeJMP");
        assert(!mlir::isa<amd64::ConditionalJumpInterface>(instrOp.getOperation()) && "Conditional jumps are handled by encodeJcc");
        
        if(instrOp->hasTrait<SpecialCase<Special::DIV>::Impl>()) [[unlikely]] {
            // in this case we need to simply XOR edx, edx, which also zeroes the upper 32 bits of rdx
            failed |= encodeRaw(FE_XOR32rr, FE_DX, FE_DX);
            
            // TODO if the mapping of operand reg to value gets used, need to set the mapping for DX to nullptr here

            // then encode the div normally
            failed |= encodeNormally();
        }else if(instrOp->hasTrait<SpecialCase<Special::IDIV>::Impl>()) [[unlikely]] {
            auto resultType = instrOp->getResult(0).getType().cast<amd64::GPRegisterTypeInterface>();
            assert(resultType && "Result of div like instruction is not a gp register type");

            // the CQO family is FE_C_SEPxx, CBW (which we need for 8 bit div) is FE_C_EX16

            // sign extend ax into dx:ax (for different sizes), for 8 bit sign extend al into ax
            switch(resultType.getBitwidth()){
                case 8:  failed |= encodeRaw(FE_C_EX16);  break;
                case 16: failed |= encodeRaw(FE_C_SEP16); break;
                case 32: failed |= encodeRaw(FE_C_SEP32); break;
                case 64: failed |= encodeRaw(FE_C_SEP64); break;
                default: llvm_unreachable("Result of div like instruction is not a register type");
            }

            failed |= encodeNormally();
        }else {
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
            assert((kind & FE_JMPL) && "wrong encoding for branch target");

            auto it = blocksToBuffer.find(target);
            // TODO think about it again, but I'm pretty sure this can never occur, because we already fail at the call site, if a call target is not a block in the module, but if it can occur normally, make this assertion an actual failure
            assert(it != blocksToBuffer.end() && "Unresolved branch target has not been visited");

            uint8_t* whereToJump = it->second;
            auto ret = fe_enc64(&whereToEncode, kind, (uintptr_t) whereToJump);
            (void) ret;
            assert(ret == 0 && "Failed to resolve branch target");
        }
        return true;
    }

    void dumpAfterEncodingDone(mlir::ModuleOp mod, GlobalsInfo& globals){
        // dump the entire buffer
        // decode & print to test if it works
        auto max = cur;

        llvm::outs() << termcolor::make(termcolor::red, "Decoded assembly:\n");

        // this is a not-very-performant way to get the block boundaries and globals, but it serves its purpose, no need for optimization in dump output
        // block boundaries
        llvm::SmallVector<uint8_t*, 64> blockStartsSorted;
        for(auto [block, buf] : blocksToBuffer){
            blockStartsSorted.push_back(buf);
        }
        std::sort(blockStartsSorted.begin(), blockStartsSorted.end());

        // globals
        llvm::SmallVector<std::pair<llvm::StringRef, amd64::GlobalSymbolInfo>, 64> globalsSorted;
        for(auto sym : globals.keys()){
            globalsSorted.push_back({sym, globals[sym]});
        }

        // also add external functions to the globals
        for(auto func : mod.getOps<mlir::func::FuncOp>()){
            if(func.isExternal())
                globalsSorted.push_back({func.getName(), amd64::GlobalSymbolInfo{
                    .bytes = {},
                    .alignment = 0,
                    .addrInDataSection = (intptr_t)checked_dlsym(func.getName())}});
        }

        std::sort(globalsSorted.begin(), globalsSorted.end(), [](auto a, auto b){
            return a.second.addrInDataSection < b.second.addrInDataSection;
        });

        // .data section
        llvm::outs() << termcolor::make(termcolor::red, ".data") << "\n";

        uint8_t* cur = bufStart;
        for(auto [sym, globalRef] : globalsSorted){
            auto& global = globalRef;
            if(global.alignment != 0)
                cur = (uint8_t*) llvm::alignTo((intptr_t) cur, global.alignment);

            if(global.addrInDataSection != (intptr_t) cur){
                DEBUGLOG("abnormal address: " << (void*) global.addrInDataSection << " vs " << (void*) cur);
                llvm::errs() << termcolor::red << "Warning: Global " << sym << " is not at the expected address in the data section. Could be external\n" << termcolor::reset;
            }
            auto* start = cur;
            auto len = global.bytes.size();

            // hexdump the bytes:
            llvm::outs() << termcolor::make(termcolor::magenta, sym) << ": ";
            for(uint8_t* cur = start; cur < start + len; cur++){
                llvm::outs() << llvm::format_hex_no_prefix(*(cur++), 2);
                if(cur < start + len)
                    llvm::outs() << llvm::format_hex_no_prefix(*cur, 2) << " ";
            }
            // as string:
            llvm::outs() << "(\"";
            for(uint8_t* cur = start; cur < start + len; cur++){
                if(*cur >= ' ' && *cur <= '~'){
                    llvm::outs() << *cur;
                }else if(*cur == '\n'){
                    llvm::outs() << "\\n";
                }else if(*cur == '\t'){
                    llvm::outs() << "\\t";
                }else{
                    llvm::outs() << ".";
                }
            }
            llvm::outs() << "\")";
            if(ArgParse::args.debug()){
                llvm::outs() << "\t\t" << "#byte " << llvm::format_hex((cur - bufStart), 0);
            }
            llvm::outs() << "\n";

            cur += global.bytes.size();
        }

        // .text section
        llvm::outs() << termcolor::make(termcolor::red, ".text") << "\n";

        uint32_t blockNum = 0;
        for(; cur < max;){
            FdInstr instr; 
            auto numBytesEncoded = fd_decode(cur, max - cur, 64, 0, &instr);
            if(numBytesEncoded < 0){
                llvm::errs() << "Encoding resulted in non-decodable instruction :(. Trying to find next decodable instruction...\n";
                cur++;
            }else{
                if(blockNum < blockStartsSorted.size() &&  blockStartsSorted[blockNum] == cur){
                    llvm::outs() << termcolor::red << "BB" << blockNum << ":" << termcolor::reset << "\n";
                    blockNum++;
                }

                static char fmtbuf[64];
                fd_format(&instr, fmtbuf, sizeof(fmtbuf));
                llvm::outs() <<  fmtbuf;
                auto fdType = FD_TYPE(&instr);
                // if its a jump/call, try to print which block it's to
                if(fdType == FDI_JMP   ||
                    fdType == FDI_JC   ||
                    fdType == FDI_JNC  ||
                    fdType == FDI_JZ   ||
                    fdType == FDI_JNZ  ||
                    fdType == FDI_JA   ||
                    fdType == FDI_JBE  ||
                    fdType == FDI_JL   ||
                    fdType == FDI_JGE  ||
                    fdType == FDI_JLE  ||
                    fdType == FDI_JG   ||
                    fdType == FDI_JMPF || fdType == FDI_CALL){

                    auto target = cur + numBytesEncoded + FD_OP_IMM(&instr, 0);
                    auto it = std::lower_bound(blockStartsSorted.begin(), blockStartsSorted.end(), target);
                    if(it != blockStartsSorted.end() && *it == target){
                        llvm::outs() << termcolor::red << " -> BB" << (it - blockStartsSorted.begin()) << termcolor::reset;
                    }

                }else if(fdType == FDI_MOVABS && FD_OP_TYPE(&instr, 1) == FD_OT_IMM){
                    // probably mov loading global -> try to find symbol

                    auto globalAddr = (intptr_t) FD_OP_IMM(&instr, 1);
                    auto it = std::lower_bound(globalsSorted.begin(), globalsSorted.end(), globalAddr, [](auto a, auto b){
                        return a.second.addrInDataSection < b;
                    });
                    if(it != globalsSorted.end() && it->second.addrInDataSection == globalAddr){
                        llvm::outs() << termcolor::red << " -> sym: " << it->first << ", byte: ";

                        if(it->second.addrInDataSection < (intptr_t) bufStart || it->second.addrInDataSection >= (intptr_t) bufEnd)
                            llvm::outs() << "<external>" << termcolor::reset;
                        else
                            llvm::outs() << it->second.addrInDataSection - (intptr_t) bufStart << termcolor::reset;
                    }
                }
                if(ArgParse::args.debug()){
                    llvm::outs() << "\t\t" << "#byte " << llvm::format_hex((cur - bufStart), 0);
                }

                llvm::outs() << "\n";
                cur += numBytesEncoded;
            }
        }
    }

};

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
        FeOp mem;
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

    [[nodiscard]] inline bool isFPReg() const {
        assert(kind == Register && "only registers can be FP");
        return amd64::isFPReg(reg);
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

    std::pair<llvm::StringRef /* start symbol */, uint8_t* /*address*/> startSymbolInfo;

    GlobalsInfo globals;

    mlir::ModuleOp mod;

    // === information for patching in stuff at the end ===

    llvm::SmallVector<std::tuple<uint8_t* /* where */, mlir::Block* /* function entry */>, 8> incompleteFunctionAddressMOVABS;

    /// size of the stack of the current function, measured in bytes from the base pointer. Includes saving of callee-saved registers. This is for easy access to the current top of stack
    uint32_t stackSizeFromBP = 0;
    /// number of bytes used for special purposes, like saving return address and BP
    uint8_t specialStackBytes = 0;
    // -> the sum of these two needs to be 16 byte aligned, that's the criterion we will use to top up the allocation

    /// the SUB64rr rsp, xxx instruction that allocates the stackframe
    uint8_t* stackAllocationInstruction = nullptr;
    /// the ADD64rr rsp, xxx instruction*s* that deallocate the stackframe
    llvm::SmallVector<uint8_t*, 8> stackDeallocationInstructions;

    /// the number of bytes we need to allocate at the end doesn't include the preallocated stack bytes (specialStackBytes + what is needed for saving callee saved regs), because they are already allocated, so these need to be saved, to be subtracted from the total
    uint8_t preAllocatedStackBytes = 0;

    FeReg guaranteedAvailableStorageReg = resetGuaranteedAvailableStorageReg();

    static_assert(FE_BX - 0x103 == 0);
    static_assert(FE_SI == FE_BX + 3) ;
    static_assert(FE_SP > FE_BX && FE_SP < FE_SI && FE_BP > FE_BX && FE_BP < FE_SI);

    FeReg resetGuaranteedAvailableStorageReg(){
        return guaranteedAvailableStorageReg = FE_BX;
    }

    FeReg nextGuaranteedAvailableStorageReg(){
        // TODO highly dependent on enum layout
        if (guaranteedAvailableStorageReg==FE_IP || guaranteedAvailableStorageReg == FE_NOREG){
            return guaranteedAvailableStorageReg = FE_NOREG;
        }else if(guaranteedAvailableStorageReg == FE_BX){
            guaranteedAvailableStorageReg = FE_R12; // first of the other callee saved registers, we could also use the other storage regs, but then we'd have to take care of saving them at calls etc.
            return FE_BX;
        }else{
            auto cur = guaranteedAvailableStorageReg;
            guaranteedAvailableStorageReg = static_cast<FeReg>(static_cast<int>(guaranteedAvailableStorageReg)+1);
            return cur;
        }
    }

    // TODO currently not in use
#if 0
    mlir::Value operandRegToValue[3] = {/* AX */nullptr, /* CX */ nullptr, /* DX */ nullptr};

    static consteval unsigned operandRegValueIndex(FeReg reg){
        // AX, CX, DX are operand registers
        // XMM0 to XMM2 as well
        if (reg == FE_AX)      return 0;
        else if (reg == FE_CX) return 1;
        else if (reg == FE_DX) return 2;
        else static_assert(false, "invalid register");
    }
#endif

    /// bufEnd works just like an end iterator, it points one *after* the last real byte of the buffer
    AbstractRegAllocerEncoder(mlir::ModuleOp mod, uint8_t* buf, uint8_t* bufEnd, GlobalsInfo&& globals, bool jit, llvm::StringRef startSymbolIfJIT) : encoder(buf, bufEnd, blockArgToReg, jit), startSymbolInfo({startSymbolIfJIT, nullptr}), globals(std::move(globals)), mod(mod) {}

    // TODO also check this for blc, i think i forgot it there

    // repeatedly overwrites the register of the value itself, as it's older values are no longer needed, because they are already encoded, the encoder always uses the current register
    void loadValueForUse(mlir::Value val, uint8_t useOperandNumber, amd64::OperandRegisterConstraint constraint){
        static_cast<Derived*>(this)->loadValueForUseImpl(val, useOperandNumber, constraint);
    }

    void allocateEncodeValueDef(amd64::InstructionOpInterface def){
        static_cast<Derived*>(this)->allocateEncodeValueDefImpl(def);
    }

    template<typename T, unsigned N>
    using SmallPtrSetVector = llvm::SetVector<T*, llvm::SmallVector<T*, N>, llvm::SmallPtrSet<T*, 8>>;

    /// returns whether it failed
    bool run(){
        // write globals to sort of .data section
        for(auto sym : globals.keys()){
            DEBUGLOG("Writing global " << sym << " to data section");
            auto& cur = encoder.cur;

            auto it = globals.find(sym);
            assert(it != globals.end() && "global not found in globals map");
            auto& global = it->second;

            // skip external/finished global
            if(global.addrInDataSection != 0){
                DEBUGLOG("Global already written, skipping");
                continue;
            }

            // align
            if(global.alignment != 0){
                cur = (uint8_t*) llvm::alignTo((intptr_t) cur, global.alignment);
                assert((intptr_t) cur % global.alignment == 0 && "global alignment has gone wrong");
            }

            memcpy(cur, global.bytes.data(), global.bytes.size());
            global.addrInDataSection = (intptr_t) cur;
            assert(globals[sym].addrInDataSection == (intptr_t) cur && "global address wasn't written correctly");
            assert(memcmp(cur, global.bytes.data(), global.bytes.size()) == 0 && "global wasn't written correctly");

            cur += global.bytes.size();
        }

        // .text section
        bool failed = false; // TODO |= this in the right places
        for(auto func : mod.getOps<mlir::func::FuncOp>()){
            if(func.isExternal())
                continue;

            // try to save start symbol
            if(encoder.jit && func.getName() == startSymbolInfo.first){
                startSymbolInfo.second = encoder.cur;
                DEBUGLOG("Found start symbol!");
            }

            auto* entryBlock = &func.getBlocks().front();
            assert(entryBlock->hasNoPredecessors() && "MLIR should disallow branching to the entry block -> MLIR bug!"); // this has to be assumed, because the entry block does stack allocation etc., we don't want that to happen multiple times
            // The only things that can target entry blocks are function calls, and they need to include the prologue, so we have to set this here. The rest of the map is filled in the encode routine below
            encoder.blocksToBuffer[entryBlock] = encoder.cur;

            emitPrologue(func);

            // this lambda is only for readability, to separate the traversal from the actual handling of a block
            auto encodeBlock = [&]<bool isEntryBlock = false>(mlir::Block* block, mlir::Block* nextBlock = nullptr){
                // currently, all blocks except the entry block have writeToBlockBufferMap = true
                if constexpr(!isEntryBlock){
                    assert(!encoder.blocksToBuffer.contains(block) && "Already encoded this block");
                    encoder.blocksToBuffer[block] = encoder.cur;
                }

                // map block to start of block in buffer

                // iterate over all but the last instruction
                auto endIt = block->end();
                for(auto& op: llvm::make_range(block->begin(), --endIt)){
                    if constexpr(!isEntryBlock)
                        assert(!mlir::isa<amd64::AllocaOp>(&op) && "found alloca outside entry block");

                    if(auto instr = mlir::dyn_cast<amd64::InstructionOpInterface>(&op)) [[likely]]{
                        // for a call: handle things differently, load all operands directly into the correct registers
                        if(instr.getFeMnemonic() == FE_CALL) [[unlikely]]{
                            auto call = mlir::cast<amd64::CALL>(instr);

                            llvm::SmallVector<mlir::Value, 8> stackArgs;

                            // emit args
                            auto moveOperands = [&]<auto intArgRegs, unsigned numIntRegArgs, auto floatArgRegs, unsigned numFloatRegArgs>(){
                                auto intOperandIndex   = 0u;
                                auto floatOperandIndex = 0u;

                                for(auto operand: instr->getOperands()){
                                    bool opIsInt = operand.getType().isa<amd64::GPRegisterTypeInterface>();
                                    if(opIsInt){
                                        if(intOperandIndex < numIntRegArgs)
                                            moveFromSlotToOperandReg(operand, valueToSlot[operand], intArgRegs[intOperandIndex++]);
                                        else
                                            // stack args are passed in reverse order. So we have to PUSHr/PUSHm the values in reverse order, so we collect them here first, then iterate over them afterwards
                                            stackArgs.push_back(operand);
                                        
                                    }else{
                                        assert(floatOperandIndex < numFloatRegArgs && "remove once float stack args have been tested");

                                        moveFromSlotToOperandReg(operand, valueToSlot[operand], floatArgRegs[floatOperandIndex++]);
                                        // the current approach for ints cant work, because 
                                    }
                                }

                                // if there is an uneven number of stack args, we have to 'push' an additional 8 bytes to keep the stack 16-byte aligned
                                if(stackArgs.size() % 2 == 1)
                                    encoder.encodeRaw(FE_SUB64ri, FE_SP, 8);

                                // now push all the stack args, in reverse order
                                // don't forget to pop them again afterwards
                                for(int i = stackArgs.size() - 1; i >= 0; i--){
                                    auto& arg = stackArgs[i];
                                    auto& slot = valueToSlot[arg];

                                    // using 8 byte PUSHes here should be correct, because the ABI specifies sizes get rounded up to 8 bytes (page 23)
                                    if(slot.kind == ValueSlot::Register)
                                        encoder.encodeRaw(FE_PUSHr, slot.reg);
                                    else
                                        encoder.encodeRaw(FE_PUSHm, slot.mem);
                                }
                            };

                            static constexpr FeReg floatArgRegs[] = {FE_XMM0, FE_XMM1, FE_XMM2, FE_XMM3, FE_XMM4, FE_XMM5, FE_XMM6, FE_XMM7};
                            if(call.getCallee()){
                                // direct call
                                static constexpr FeReg intArgRegs[] = {FE_DI, FE_SI, FE_DX, FE_CX, FE_R8, FE_R9};
                                moveOperands.template operator()<intArgRegs, comptimeArraySize(intArgRegs), floatArgRegs, comptimeArraySize(floatArgRegs)>();
                            }else{
                                // indirect call, first operand is the pointer to what to call, rest is the same
                                static constexpr FeReg intArgRegs[] = {FE_AX, FE_DI, FE_SI, FE_DX, FE_CX, FE_R8, FE_R9};
                                moveOperands.template operator()<intArgRegs, comptimeArraySize(intArgRegs), floatArgRegs, comptimeArraySize(floatArgRegs)>();
                            }

                            // TODO part of this needs to be factored out to a kind of "allocated but dont spill or encode" function. Alternatively template variants (either with boolean template arguments, or as specializations) of the allocateEncodeValueDef function that don't do that.
                            // TODO remove the assert, once the rework of encoding failure is done
                            auto ret = encoder.encodeCall(mod, call);
                            assert(ret == false && "encoding call failed");
                            if(call.getNumResults() > 0){
                                assert(call.getNumResults() == 1 && "multiple results not supported yet");

                                auto result = call.getResult(0);
                                auto resultBw = mlir::cast<amd64::RegisterTypeInterface>(result.getType()).getBitwidth();
                                assert(resultBw < std::numeric_limits<uint8_t>::max() && "bitwidth too high");
                                auto& slot = valueToSlot[result] = ValueSlot{.kind = ValueSlot::Stack, .mem = allocateNewStackslot(resultBw), .bitwidth = static_cast<uint8_t>(resultBw)};
                                bool resultIsInt = result.getType().isa<amd64::GPRegisterTypeInterface>();
                                if(resultIsInt)
                                    moveFromOperandRegToSlot(result, slot, FE_AX);
                                else
                                    moveFromOperandRegToSlot(result, slot, FE_XMM0);
                            }

                            // now 'pop' all the stack args. We don't care about what happened to them, so we can just do a single big ADD64ri
                            auto addsize = stackArgs.size() * 8;
                            // if there is an uneven number of stack args, we have to 'push' an additional 8 bytes to keep the stack 16-byte aligned
                            if(stackArgs.size() % 2 == 1)
                                addsize += 8;

                            if(addsize != 0)
                                encoder.encodeRaw(FE_ADD64ri, FE_SP, addsize);

                            continue;
                        }

                        // ignore the instruction, if it's pure and it's uses are empty
                        // TODO might cost performance, maybe remove for quick allocer
                        // TODO also check that the 'codegen-dce enabled' check doesn't cost performance. But it should really only be a single bool comparison, as the rest of that is constexpr, and that comparison will always be predicted correctly after the first branch miss
                        if(ArgParse::features["codegen-dce"] && mlir::isOpTriviallyDead(&op)) [[unlikely]]
                            continue;

                        // two operands max for now
                        // TODO make this nicer
                        assert(instr->getNumOperands() <= 2);
                        for(auto [i, operand] : llvm::enumerate(instr->getOperands())){
                            // TODO BIIIIG problem: We're setting the register of the value (via registerOf) to be the one it's moved to, which makes sense for the most part, we can overwrite it because it's only needed once, but not if the operation has the same SSA value as input multiple times. The encoder will then get the idea that all operands are in the same register. This is correct for their values, but as that register might get overwritten in the result, we have a huge mess on our hands in that case.
                            // its probably alright, if we just assume that all 'same as first' operands get killed, then use operand registers for that

                            // TODO is this enough?
                            assert((!operand.isa<mlir::OpResult>() || !mlir::isa<amd64::AddrOfGlobal>(operand.getDefiningOp())) && "AddrOfGlobal should have been replaced at the time of use");

                            auto operandRegConstraint = instr.getOperandRegisterConstraints()[i];

                            // for memloc operands, we have to load all register values, the base and the index, into registers
                            if(mlir::isa<mlir::OpResult>(operand)) /* && */ if(auto memOp = mlir::dyn_cast<amd64::EncodeOpInterface>(operand.getDefiningOp())) [[unlikely]]{
                                if(auto base = memOp.getBaseGeneric().value_or(nullptr)){
                                    loadValueForUse(base, 0, operandRegConstraint);
                                    if(auto index = memOp.getIndexGeneric().value_or(nullptr))
                                        loadValueForUse(index, 1, operandRegConstraint);
                                }else if(auto index = memOp.getIndexGeneric().value_or(nullptr)){
                                    loadValueForUse(index, 0, operandRegConstraint);
                                }
                                continue;
                            }

                            loadValueForUse(operand, i, operandRegConstraint);
                        }

                        // TODO this is alright for the moment, as it always allocates AX to the first int result and XMM0 to the first float result, which is what we want for a call. But as soon as there is a more complex register allocator, this need sto change
                        // there is a static assertion elsewhere (because StackRegAllocer is not declared yet) to check this, but the problem is here
                        allocateEncodeValueDef(instr);
                    }else if(auto addrofGlobalOp = mlir::dyn_cast<amd64::AddrOfGlobal>(&op)) [[unlikely]]{
                        // TODO this is not very nice
                        auto it = globals.find(addrofGlobalOp.getName());
                        assert(it != globals.end() && "unknown global");
                        DEBUGLOG("addrofGlobal \"" << addrofGlobalOp.getName() << "\" at " << addrofGlobalOp.getLoc() << " is at " << it->second.addrInDataSection << " in data section");
                        uint8_t* globalAddr = (uint8_t*) it->second.addrInDataSection;
                        assert((intptr_t)globalAddr == it->second.addrInDataSection && "global cast lost information");
                        assert(globalAddr != nullptr && "global not in data section");
                        assert((globalAddr - encoder.cur) == ((intptr_t)globalAddr - (intptr_t)encoder.cur) && "global cast lost information");

                        mlir::OpBuilder builder(&op);

                        // TODO what's the tradeoff here: RIP relative addressing LEA vs. absolute addressing with a 64bit immediate MOV
                        // advantage of 64 bit mov: no raw mem op, and external symbols don't need special handling
                        /*
                            auto rawMemForLea = builder.create<amd64::RawMemoryOp>(addrofGlobalOp.getLoc(), FE_MEM(FE_IP, 0, FE_NOREG, globalAddr-encoder.cur));
                            auto lea = builder.create<amd64::LEA64rm>(addrofGlobalOp.getLoc(), rawMemForLea);
                            addrofGlobalOp.replaceAllUsesWith(lea->getResult(0));

                            allocateEncodeValueDef(lea);
                        */
                        auto mov64ri = builder.create<amd64::MOV64ri>(addrofGlobalOp.getLoc(), (intptr_t)globalAddr);
                        addrofGlobalOp.replaceAllUsesWith(mov64ri->getResult(0));
                        allocateEncodeValueDef(mov64ri);
                    }else if(auto addrofFuncOp = mlir::dyn_cast<amd64::AddrOfFunc>(&op)) [[unlikely]]{
                        // TODO this is not very nice
                        mlir::CallInterfaceCallable callable = addrofFuncOp.getNameAttr();
                        auto maybeFunc = getFuncForCall(mod, callable, encoder.symbolrefToFuncCache);

                        // we will use a mov64ri to move various pointers/addresses into place, but we need to replace the original ops use
                        auto allocateEncodeMOV64ri = [&](intptr_t addr){
                            assert((addr > std::numeric_limits<int32_t>::max() || addr < std::numeric_limits<int32_t>::min()) && "addr cannot be allowed to fit in 32 bit, we want to use the maximum width encoding");
                            // TODO maaaaybe merge with addrofglobal? has similar code
                            mlir::OpBuilder builder(&op);
                            auto mov64ri = builder.create<amd64::MOV64ri>(addrofFuncOp.getLoc(), addr);
                            addrofFuncOp.replaceAllUsesWith(mov64ri->getResult(0));
                            allocateEncodeValueDef(mov64ri);
                        };

                        auto handleExternal = [&](){
                            // TODO at the moment this can't even be called with external things, so maybe remove this in the future
                            llvm_unreachable("not tested with external funcs yet, should not be called");
                            
                            if(encoder.jit){
                                // try to find the function in the environment
                                intptr_t fnPtr = (intptr_t) checked_dlsym(addrofFuncOp.getName());
                                assert(fnPtr && "Symbol not found");

                                allocateEncodeMOV64ri(fnPtr);
                            }else{
                                llvm::errs() << "Call to external function, relocations not implemented yet";
                            }
                        };

                        // function has a declaration
                        if(maybeFunc){
                            auto func = mlir::cast<mlir::func::FuncOp>(maybeFunc);
                            if(func.isExternal()){
                                handleExternal();
                            }else{
                                mlir::Block* entryBlock = &func.getBody().front();
                                assert(entryBlock && "function has no entry block");

                                // if we've already written this, we can just use the address we've written it to
                                if(auto entryBlockAddrIt = encoder.blocksToBuffer.find(entryBlock); entryBlockAddrIt != encoder.blocksToBuffer.end()){
                                    allocateEncodeMOV64ri((intptr_t)entryBlockAddrIt->second);
                                }else{
                                    // if we don't know it yet, generate a dummy instruction to be patched later

                                    // TODO this is only luckily the right buf location, this is *actually* the location of the first load for the operands. In this case there *should be none*, but this is still not clean
                                    auto bufLocation = encoder.saveCur();
                                    allocateEncodeMOV64ri(0x0100000000000000 /* to reserve space for the 64bit pointer immediate */);

                                    incompleteFunctionAddressMOVABS.emplace_back(bufLocation, entryBlock);
                                }
                            }
                        }else{
                            // its probably external
                            handleExternal();
                        }
                    }

                    if constexpr(isEntryBlock){
                        if(auto alloca = mlir::dyn_cast<amd64::AllocaOp>(&op)){
                            auto& prop = alloca.getProperties();
                            prop.rbpOffset = -(stackSizeFromBP += prop.size);
                        }
                    }
                }
                handleTerminator(endIt, nextBlock);
            };

            // do a reverse post order traversal (RPOT) of the CFG, to make sure we encounter all definitions of uses before their uses, and that we see as many predecessors as possible before a block, and loops stay together
            // we will make this a DAG, by ignoring back-(including self-)edges


            // TODO probably yeet this if constexpr at some point, this is just for performance testing. Or maybe make it dependent on the optimization level, if there is a significant difference
            // TODO make a feature arg for it
            constexpr bool useRPO = true;

            // set vector might be quite expensive, but it seems alright
            if constexpr(useRPO){
                // build post order
                SmallPtrSetVector<mlir::Block, 8> worklist;
                worklist.insert(entryBlock);
                SmallPtrSetVector<mlir::Block, 32> postOrder;

                while(!worklist.empty()){
                    auto* currentBlock = worklist.back();

                    bool allSuccessorsInPO = true;
                    for(auto* succ: currentBlock->getSuccessors()){
                        if(!postOrder.contains(succ)){
                            if(!worklist.insert(succ)){ // `insert` returns true if an element was inserted
                                // -> found a back edge if this returns false, back edges are fine to ignore, we can still handle the current block
                            }else{
                                // if we inserted something, handle it immediately
                                allSuccessorsInPO = false;
                                break; // because allSuccessorsInPO is now false, this block won't be removed from the worklist, i.e. we will come back to handle the other successors
                            }
                        }
                    }

                    if(allSuccessorsInPO){
                        postOrder.insert(currentBlock);
                        worklist.pop_back();
                    }
                }

                assert(postOrder.back() == entryBlock && "Entry block is not last in post order traversal");

                // apparently, I could just reuse llvm::ReversePostOrderTraversal here, see ModuleTranslation.cpp:565 getTopologicallySortedBlocks :
                //auto range = llvm::ReversePostOrderTraversal<mlir::Block*>(entryBlock);

                auto range = llvm::reverse(postOrder);
                auto beginPlusOne = range.begin();
                ++beginPlusOne;
                auto oneBeforeEnd = range.end();
                --oneBeforeEnd;

                // range can't be empty, always contains at least the entry block
                assert(!range.empty() && "function RPO without entry block");

                bool atLeast2Blocks = beginPlusOne != range.end();

                // special case for the entry block: don't fill the blocksToBuffer for it, this was done before the prologue was emitted.
                encodeBlock.template operator()<true>(*range.begin(), atLeast2Blocks ? *beginPlusOne : nullptr);

                if(atLeast2Blocks){
                    // iterate over the first n minus one blocks while passing the next one
                    for(auto blockIt = beginPlusOne; blockIt != oneBeforeEnd; ++blockIt){
                        // pass the next block, for encoding the minimum number of jumps
                        encodeBlock(*blockIt, *std::next(blockIt));
                    }

                    // just encode the last block, can't pass the next block
                    encodeBlock(*oneBeforeEnd);
                }
            }else{
                // do a DFS pre-order instead, which has enough guarantees, but is simpler

                llvm::SmallVector<mlir::Block*, 8> worklist{entryBlock};
                while(!worklist.empty()){
                    auto* currentBlock = worklist.pop_back_val();

                    if(encoder.blocksToBuffer.contains(currentBlock)) // skip doubly added blocks
                        continue;

                    encodeBlock(currentBlock);

                    for(auto* succ: currentBlock->getSuccessors()){
                        worklist.push_back(succ);
                    }
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

            auto patch4ByteImm = [&](uint8_t* start, uint32_t val){ memcpyToLittleEndianBuffer(start, val); };
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
            resetGuaranteedAvailableStorageReg();
        }

        // resolve all things that need to know every block
        failed |= encoder.resolveBranches();
        // this also includes the function address moves
        auto cur = encoder.saveCur();
        for(auto [buf, entryBlock] : incompleteFunctionAddressMOVABS){
            encoder.restoreCur(buf);
            intptr_t addr = (intptr_t) encoder.blocksToBuffer[entryBlock];
            assert((addr > std::numeric_limits<int32_t>::max() || addr < std::numeric_limits<int32_t>::min()) && "addr cannot be allowed to fit in 32 bit, we want to use the maximum width encoding");
            // TODO this just uses FE_AX by convention/hardcoded, change it so that this is saved within the incompleteFunctionAddressMOVABS map
            encoder.encodeRaw(FE_MOV64ri,  FE_AX, addr);
        }
        encoder.restoreCur(cur);

        return failed;
    }

protected:
    /// overload to save a map lookup, if the slot is already known
    bool moveFromOperandRegToSlot(mlir::Value val){
        return moveFromOperandRegToSlot(val, valueToSlot[val]);
    }

    // TODO the case distinction depending on the slot kind can be avoided for allocateEncodeValueDef, by using a template and if constexpr, but see if that actually makes the code faster first
    /// move from the register the value is currently in, to the slot, or from the operand register override, if it is set
    bool moveFromOperandRegToSlot(mlir::Value fromVal, const ValueSlot& toSlot, const FeReg operandRegOverride = (FeReg) FE_NOREG){
        /// the register the value is currently in
        FeReg& fromValReg = amd64::registerOf(fromVal, &blockArgToReg);

        if(operandRegOverride != (FeReg) FE_NOREG)
            fromValReg = operandRegOverride;

        bool sourceIsFPReg = amd64::isFPReg(fromValReg);

        if(toSlot.kind == ValueSlot::Register){
            if(fromValReg == toSlot.reg)
                return false;

            bool destIsFPReg = toSlot.isFPReg();

            // TODO for now, we don't allow moves between int and float regs using this method
            assert(iff(destIsFPReg, sourceIsFPReg) && "moves between int and float regs not supported in moveFromOperandRegToSlot");

            FeMnem mnem;

            switch(toSlot.bitwidth + (static_cast<uint32_t>(sourceIsFPReg) << 8)){
                // TODO try to eliminate partial register deps
                case 8:  mnem = FE_MOV8rr;  break;
                case 16: mnem = FE_MOV16rr; break;
                case 32: mnem = FE_MOV32rr; break;
                case 64: mnem = FE_MOV64rr; break;

                case 0x0100 + 32: mnem = FE_SSE_MOVSSrr; break;
                case 0x0100 + 64: mnem = FE_SSE_MOVSDrr; break;

                default: llvm_unreachable("invalid bitwidth");
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
            switch(toSlot.bitwidth + (static_cast<uint32_t>(sourceIsFPReg) << 8)){
                // TODO try to eliminate partial register deps
                case 8:  mnem = FE_MOV8mr;  break;
                case 16: mnem = FE_MOV16mr; break;
                case 32: mnem = FE_MOV32mr; break;
                case 64: mnem = FE_MOV64mr; break;

                case 0x0100 + 32: mnem = FE_SSE_MOVSSmr; break;
                case 0x0100 + 64: mnem = FE_SSE_MOVSDmr; break;

                default: llvm_unreachable("invalid bitwidth");
            }
            return encoder.encodeRaw(mnem, toSlot.mem, fromValReg);
        }
    }

    // TODO the fp reg checks could also slow this down quite a lot
    // TODO the case distinction depending on the slot kind can be avoided for allocateEncodeValueDef, by using a template and if constexpr, but see if that actually makes the code faster first
    bool moveFromSlotToOperandReg(mlir::Value val, ValueSlot slot, FeReg reg){
        // TODO can probably avoid re-moving the same value to the same operand register, but we need more information for this, registerOf(val) == reg is not enough, because the register of a value does not get overwritten, when another value gets moved there. So we need a register -> value map (vector) to do this
        bool failed;
        bool destIsFPReg = amd64::isFPReg(reg);

        if (slot.kind == ValueSlot::Register){
            if(reg == slot.reg)
                goto success;

            bool sourceIsFPReg = slot.isFPReg();

            // TODO for now, we don't allow moves between int and float regs using this method
            assert(iff(destIsFPReg, sourceIsFPReg) && "moves between int and float regs not supported in moveFromSlotToOperandReg");

            FeMnem mnem;
            // TODO idk, is this more or less ugly than an additional if around it?
            switch(slot.bitwidth + (static_cast<uint32_t>(destIsFPReg) << 8)){
                // TODO try to eliminate partial register deps
                case 8:  mnem = FE_MOV8rr;  break;
                case 16: mnem = FE_MOV16rr; break;
                case 32: mnem = FE_MOV32rr; break;
                case 64: mnem = FE_MOV64rr; break;

                case 0x0100 + 32: mnem = FE_SSE_MOVSSrr; break;
                case 0x0100 + 64: mnem = FE_SSE_MOVSDrr; break;

                default: llvm_unreachable("invalid bitwidth");
            }
            failed = encoder.encodeRaw(mnem, reg, slot.reg);
        }else{
            FeMnem mnem;
            switch(slot.bitwidth + (static_cast<uint32_t>(destIsFPReg) << 8)){
                // TODO try to eliminate partial register deps
                case 8:  mnem = FE_MOV8rm;  break;
                case 16: mnem = FE_MOV16rm; break;
                case 32: mnem = FE_MOV32rm; break;
                case 64: mnem = FE_MOV64rm; break;

                case 0x0100 + 32: mnem = FE_SSE_MOVSSrm; break;
                case 0x0100 + 64: mnem = FE_SSE_MOVSDrm; break;

                default: llvm_unreachable("invalid bitwidth");
            }
            failed = encoder.encodeRaw(mnem, reg, slot.mem);
        }

    success:

        // we're moving into some operand register -> set the val to be in that register
        amd64::registerOf(val, &blockArgToReg) = reg;
        // TODO return whether it failed
        return failed;
    }

    bool moveFromSlotToOperandReg(mlir::Value val, const FeReg reg){
        assert(valueToSlot.contains(val));
        return moveFromSlotToOperandReg(val, valueToSlot[val], reg);
    }

    /// only use: moving phis
    /// returns whether it failed
    bool moveFromSlotToSlot(mlir::Value from, mlir::Value to, const FeReg memMemMoveReg){
        assert(!amd64::isFPReg(memMemMoveReg) && "mem-mem moves with fp regs not supported yet");
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
    void condMoveFromSlotToSlot(ValueSlot fromSlot, mlir::Value conditionallyMoveTo, const FeReg memMemMoveReg, amd64::conditional::predicate cmovPredicate){
        assert(!amd64::isFPReg(memMemMoveReg) && "(conditional) mem-mem moves with fp regs not supported yet");
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

        using namespace amd64::conditional;
        switch(cmovPredicate){
            case Z:  mnem = isRegister ? FE_CMOVZ64rr  : FE_CMOVZ64rm;  break;
            case NZ: mnem = isRegister ? FE_CMOVNZ64rr : FE_CMOVNZ64rm; break;
            case L:  mnem = isRegister ? FE_CMOVL64rr  : FE_CMOVL64rm;  break;
            case GE: mnem = isRegister ? FE_CMOVGE64rr : FE_CMOVGE64rm; break;
            case LE: mnem = isRegister ? FE_CMOVLE64rr : FE_CMOVLE64rm; break;
            case G:  mnem = isRegister ? FE_CMOVG64rr  : FE_CMOVG64rm;  break;
            case C:  mnem = isRegister ? FE_CMOVC64rr  : FE_CMOVC64rm;  break;
            case NC: mnem = isRegister ? FE_CMOVNC64rr : FE_CMOVNC64rm; break;
            case BE: mnem = isRegister ? FE_CMOVBE64rr : FE_CMOVBE64rm; break;
            case A:  mnem = isRegister ? FE_CMOVA64rr  : FE_CMOVA64rm;  break;
            default: llvm_unreachable("invalid condition code");
        }
        encoder.encodeRaw(mnem, memMemMoveReg, isRegister ? (FeOp) fromSlot.reg : (FeOp) fromSlot.mem);

        // 3.
        moveFromOperandRegToSlot(conditionallyMoveTo, valueToSlot[conditionallyMoveTo], memMemMoveReg);
    }

    template<bool isCriticalEdge>
    void handleCFGEdgeHelper(mlir::Block::BlockArgListType blockArgs, mlir::Operation::operand_range blockArgOperands, amd64::conditional::predicate cmovPredicateIfCriticalJcc = amd64::conditional::predicate::NONE){
        // TODO return if something went wrong

        // need to break memory memory moves with a temp register
        const FeReg memMemMoveReg = FE_AX;

        if constexpr(isCriticalEdge)
            assert(cmovPredicateIfCriticalJcc != amd64::conditional::predicate::NONE);

        auto handleChainElement = [this, memMemMoveReg, cmovPredicateIfCriticalJcc](mlir::Value from, mlir::Value to){
            if constexpr(!isCriticalEdge)
                (void) cmovPredicateIfCriticalJcc,
                moveFromSlotToSlot(from, to, memMemMoveReg);
            else
                condMoveFromSlotToSlot(valueToSlot[from], to, memMemMoveReg, cmovPredicateIfCriticalJcc);
        };

        // TODO do the pointers make sense here?
        llvm::SmallVector<std::pair<mlir::BlockArgument, mlir::BlockArgument>, 8> mapArgNumberToArgAndArgItReads(blockArgs.size(), {nullptr, nullptr});
        llvm::SmallVector<int16_t, 8> numReaders(blockArgs.size(), 0); // indexed by the block arg number, i.e. the slots for the dependent args are empty/0 in here

        // find the args which are independent, and those which need a topological sorting
        for(auto [arg, operand] : llvm::zip(blockArgs, blockArgOperands)){
            if(auto operandArg = operand.dyn_cast<mlir::BlockArgument>(); operandArg && operandArg.getParentBlock() == arg.getParentBlock()){
                if(operandArg == arg) { // if its a self-reference/edge, we can simply ignore it, because its SSA: the value only gets defined once, and a ValueSlot only ever holds exactly one value during that value's liveness. In this case this value is this phi, and it is obviously still live, by being used as a block arg, so we can just leave it where it is, don't need to emit any code for it
                    continue;
                } else /* otherwise we've found a dependant phi */ {
                    mapArgNumberToArgAndArgItReads[arg.getArgNumber()] = {arg, operandArg};

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
            while(numReaders[arg.getArgNumber()] == 0){
                // found the start of a chain

                handleChainElement(argThatItReads, arg); // need to move the value being read *from* its slot *to* the slot of the phi

                // to ensure that this arg is never handled again (it might be visited again by the for loop, if it has been handled because a reader has unblocked it), set it's reader number to -1
                numReaders[arg.getArgNumber()] = -1;

                // we've removed a reader from this, might have unblocked it
                numReaders[argThatItReads.getArgNumber()] -= 1;

                // -> try if it's unblocked, by re-setting the arg and argThatItReads
                arg = argThatItReads; // we're looking at the phi we just read now
                argThatItReads = mapArgNumberToArgAndArgItReads[arg.getArgNumber()].second; // get what it reads from the map
            }
        }

        // if there are any entries left which have numReaders != -1, then there is a cycle, and all numReaders != -1 should be exactly == 1
        // break the cycle by copying any of the values in the cycle

        FeReg phiCycleBreakReg = FE_CX;

        // find the first of the cycle
        for(auto [arg, argThatItReads] : mapArgNumberToArgAndArgItReads){
            if(arg == nullptr || numReaders[arg.getArgNumber()] == -1)
                continue;

            assert(numReaders[arg.getArgNumber()] == 1 && "cycle in phi dependencies, but some of the cycle's members have more than one reader, something is wrong here");

            // save temporarily
            moveFromSlotToOperandReg(arg, phiCycleBreakReg);
            numReaders[arg.getArgNumber()] = 0;

            // we're iterating until what we're reading doesn't have 1 reader -> it has to be the first, so we do the special copy from the cycle break register, after the loop
            while(numReaders[argThatItReads.getArgNumber()] == 1){
                handleChainElement(argThatItReads, arg); // need to move the value being read *from* its slot *to* the slot of the phi

                // we've removed a reader from this
                numReaders[argThatItReads.getArgNumber()] = 0;

                arg = argThatItReads; // we're looking at the phi we just read now
                argThatItReads = mapArgNumberToArgAndArgItReads[arg.getArgNumber()].second; // get what it reads from the map
            }

            // need to move from the cycle break to this last arg, to complete the loop
            // and we need to do that conditionally, if this is a critical edge
            if constexpr(!isCriticalEdge){
                moveFromOperandRegToSlot(arg, valueToSlot[arg], phiCycleBreakReg); // the problem here is we to move from FE_CX/phiCycleBreakReg, to the slot of the arg, not from the register that the value is currently in, so we have to override it
            }else{
                ValueSlot cycleBreakPsuedoSlot = {.kind = ValueSlot::Register, .reg = phiCycleBreakReg, .bitwidth = valueToSlot[argThatItReads].bitwidth};

                condMoveFromSlotToSlot(cycleBreakPsuedoSlot, arg, FE_AX, cmovPredicateIfCriticalJcc);
            }

            break;
        }
    }

    void handleTerminator(mlir::Block::iterator instrIt, mlir::Block* nextBlock){
        // We need to deal with PHIs/block args here.
        // Because we're traversing in RPO, we know that the block args are already allocated.
        auto tryAllocateSlotsForBlockArgsOfSuccessor = [this, instrIt](mlir::Block::BlockArgListType blockArgs){
            for(auto arg : blockArgs){
                // TODO float block args currently not supported
                // TODO this needs to be functionality of a concrete regallocer implementation, not the abstract one, because it decides where to put block args

                // allocate a slot for the block arg
                // if we have storage registers free in this function, use them. Otherwise, allocate a stack slot

                uint8_t bitwidth = arg.getType().cast<amd64::GPRegisterTypeInterface>().getBitwidth();

                ValueSlot slot;
                if(auto reg = nextGuaranteedAvailableStorageReg(); reg != FE_NOREG){
                    slot = ValueSlot{.kind = ValueSlot::Register, .reg = reg, .bitwidth = bitwidth};
                }else{
                    slot = ValueSlot{.kind = ValueSlot::Stack, .mem = allocateNewStackslot(bitwidth), .bitwidth = bitwidth};
                }

                valueToSlot.try_emplace(arg, slot);
                // TODO a possible optimization would be to use another map for blocks whose blockargs have already been allocated, and skip this loop entirely. Might be worth it for blocks with a lot of args. But might cost performance otherwise
            }
        };

        if(auto ret = mlir::dyn_cast<amd64::RET>(*instrIt)){
            // first load the operand into AX
            // TODO our RET always has exactly one operand, but it might be a good idea to support no operand return in RET, instead of converting a no operand cf.ret to a RET with a dummy operand
            assert(ret->getNumOperands() == 1 && "RET with != 1 operand, someone forgot to change this line!");
            moveFromSlotToOperandReg(ret.getOperand(), FE_AX);
            emitEpilogue();

            // TODO could also do this directly (simply encode RET), see if it makes a performance difference
            encoder.encodeIntraBlockOp(ret);
        }else if (auto jmp = mlir::dyn_cast<amd64::JMP>(*instrIt)){
            auto blockArgs = jmp.getDest()->getArguments();
            tryAllocateSlotsForBlockArgsOfSuccessor(blockArgs);

            auto blockArgOperands = jmp.getDestOperands();

            handleCFGEdgeHelper<false>(blockArgs, blockArgOperands);
            encoder.encodeJMP(jmp, nextBlock);
        }else if(auto jcc = mlir::dyn_cast<amd64::ConditionalJumpInterface>(*instrIt)){
            for(auto dst: {jcc.getTrueDest(), jcc.getFalseDest()}){
                auto blockArgs = dst->getArguments();
                tryAllocateSlotsForBlockArgsOfSuccessor(blockArgs);
            }

            // this is a problem, because we only have a single jump here, which gets encoded into 2 (max) later on, but we need to place the MOVs in between those two
            // so we pass the handleCFGEdge helper as an argument (parameterized depending on whether or not this is a critical edge), to be called by the encoder function
            // TODO next is currently, because the nextBB is being set linearly at the callsites, but the traversal is not linear anymore, so this needs to be passed correctly
            encoder.encodeJcc<&AbstractRegAllocerEncoder<Derived>::handleCFGEdgeHelper<false>, &AbstractRegAllocerEncoder<Derived>::handleCFGEdgeHelper<true>>(*this, jcc, nextBlock);
        }
    }

    void emitPrologue(mlir::func::FuncOp func){
        static_cast<Derived*>(this)->emitPrologueImpl(func);
    }

    void emitEpilogue(){
        static_cast<Derived*>(this)->emitEpilogueImpl();
    }

    /// allocates a new stackslot and returns an FeOp that can be used to access it
    FeOp allocateNewStackslot(uint8_t bitwidth){ // TODO are there alignment problems here? I think not, but it might be worth aligning every slot to 8 bytes or smth anyway?
        return FE_MEM(FE_BP, 0,0, -(stackSizeFromBP += bitwidth / 8));
    }
};

struct StackRegAllocer : public AbstractRegAllocerEncoder<StackRegAllocer>{
public:
    using AbstractRegAllocerEncoder::AbstractRegAllocerEncoder;

    void loadValueForUseImpl(mlir::Value val, uint8_t useOperandNumber, amd64::OperandRegisterConstraint constraint){
#ifndef NDEBUG
        if(!valueToSlot.contains(val)){
            llvm::errs() << "value not allocated: " << val << " (in use number #" << useOperandNumber<< ")\n";
        }
        assert(valueToSlot.contains(val) && "value not allocated");
#endif

        auto slot = valueToSlot[val];
        assert(implies(slot.kind == ValueSlot::Kind::Register, mlir::isa<mlir::BlockArgument>(val)) && "stack register allocator encountered a register slot on a non-block arg");

        FeReg whichReg;
        if(constraint.constrainsReg()) [[unlikely]] {
            assert(constraint.which == useOperandNumber && "constraint doesn't match use");
            whichReg = constraint.reg;
        }else{
            // TODO allocating the registers in this fixed way has the advantage of compile speed and simplicity, but it is quite error prone...
            // strategy: the first op can always have AX, because it isn't needed as the constraint for any second op
            //           the second operand can always have CX

            if(mlir::isa<amd64::GPRegisterTypeInterface>(val.getType())){
                if(useOperandNumber == 0)
                    whichReg = FE_AX;
                else if (useOperandNumber == 1)
                    whichReg = slot.kind == ValueSlot::Register ? slot.reg : FE_CX;
                else if (useOperandNumber == 2)
                    whichReg = FE_DX;
                else
                    llvm_unreachable("more than 3 int operands not supported yet");
            }else{
                assert(mlir::isa<amd64::FPRegisterTypeInterface>(val.getType()) && "unknown register type");
                if(useOperandNumber == 0)
                    whichReg = FE_XMM0;
                else if (useOperandNumber == 1)
                    whichReg = slot.kind == ValueSlot::Register ? slot.reg : FE_XMM1;
                else if (useOperandNumber == 2)
                    whichReg = FE_XMM2;
                else
                    llvm_unreachable("more than 3 float operands not supported yet");
            }
        }

        moveFromSlotToOperandReg(val, slot, whichReg);
    }

    // TODO return failure
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
            auto& [spillResult, spillSlot, spillReg] = resultInfoForSpilling[i];
            spillResult = result;

            auto resultAsRegType = mlir::dyn_cast<amd64::RegisterTypeInterface>(result.getType());
            uint8_t bitwidth = resultAsRegType.getBitwidth();
            FeOp memLoc = allocateNewStackslot(bitwidth);
            spillSlot = valueToSlot[result] = ValueSlot{.kind = ValueSlot::Kind::Stack, .mem = memLoc, .bitwidth = bitwidth};

            // TODO handle float results

            // allocate registers for the result
            if(mlir::isa<amd64::GPRegisterTypeInterface>(resultAsRegType)){
                if(i == 0){
                    // always assign AX to the first result
                    spillReg = instructionInfo.regs.reg1 = FE_AX;
                    assert(implies(resConstr1.constrainsReg(), resConstr1.which != 0 || resConstr1.reg == FE_AX) && "constraint doesn't match result");
                }else if(i == 1){
                    // always assign DX to the second result
                    spillReg = instructionInfo.regs.reg2 = FE_DX;
                    assert(implies(resConst2.constrainsReg(), resConst2.which != 1 || resConst2.reg == FE_DX) && "constraint doesn't match result");
                }else{
                    // TODO this doesn't need to be a proper failure for now, right? Because we only define instrs with at most 2 results
                    llvm_unreachable("more than 2 int results not supported yet");
                }
            }else{
                assert(mlir::isa<amd64::FPRegisterTypeInterface>(resultAsRegType) && "result is neither GP nor FP register type");
                if(i == 0){
                    // always assign XMM0 to the first result
                    spillReg = instructionInfo.regs.reg1 = FE_XMM0;
                    assert(implies(resConstr1.constrainsReg(), resConstr1.which != 0 || resConstr1.reg == FE_XMM0) && "constraint doesn't match result");
                }else if(i == 1){
                    // always assign XMM1 to the second result
                    spillReg = instructionInfo.regs.reg2 = FE_XMM1;
                    assert(implies(resConst2.constrainsReg(), resConst2.which != 1 || resConst2.reg == FE_XMM1) && "constraint doesn't match result");
                }else{
                    // TODO this doesn't need to be a proper failure for now, right? Because we only define instrs with at most 2 results
                    llvm_unreachable("more than 2 float results not supported yet");
                }
            }

            // spill store after the instruction
            // problem: as this is only supposed to happen after the instruction is encoded, and the information we just attached above is needed for encoding, -> we can't do this here, have to do it in a loop after it's encoded
        }

        // encode the instruction, now that it has all the information it needs
        encoder.encodeIntraBlockOp(def);



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
                // TODO float function args not supported yet 

                auto argAsRegType = mlir::cast<amd64::GPRegisterTypeInterface>(arg.getType());
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

        // this instruction needs to be rewritten later
        stackDeallocationInstructions.push_back(encoder.saveCur());
        encoder.encodeRaw(FE_ADD64ri, FE_SP, 0x01000000); // 0x01000000 isn't the final value, just a placeholder that ensures that the operand size byte is it's maximum possible value -> we have space to encode a big allocation


        for (auto reg : {FE_R15, FE_R14, FE_R13, FE_R12, FE_BX})
            encoder.encodeRaw(FE_POPr, reg);

        encoder.encodeRaw(FE_MOV64rr, FE_SP, FE_BP);
        encoder.encodeRaw(FE_POPr, FE_BP);
    }

private:
    bool isAllocated(mlir::Value val){
        return valueToSlot.contains(val);
    }
};

// start of better regalloc deleted in 5672526, if I ever need it again

// TODO parameters for optimization level (-> which regallocer to use)
uint8_t* regallocEncode(uint8_t* buf, uint8_t* bufEnd, mlir::ModuleOp mod, GlobalsInfo&& globals, bool dumpAsm, bool jit, llvm::StringRef startSymbolIfJIT){
    StackRegAllocer regallocer(mod, buf, bufEnd, std::move(globals), jit, startSymbolIfJIT);

    // just to again ensure that this is not changed without caution
    static_assert(std::is_same<decltype(regallocer), StackRegAllocer>::value, "You need to change handling in the abstract reg allocer, like how calls are encoded (implicit reg constraints), and more, before you use a new reg allocer!");
    regallocer.run();
    if(dumpAsm)
        regallocer.encoder.dumpAfterEncodingDone(mod, regallocer.globals);

    return regallocer.startSymbolInfo.second;
}
