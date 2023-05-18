#ifndef AMD64OPS
#define AMD64OPS

include "AMD64/AMD64Dialect.td"

include "mlir/IR/AttrTypeBase.td"

include "mlir/IR/OpBase.td"

include "mlir/IR/BuiltinTypes.td"
// don't include this: "mlir/IR/BuiltinOps.td". Breaks everything for some reason
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinDialect.td"

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// my interfaces
include "AMD64/AMD64OpInterfaces.td"
include "AMD64/AMD64TypeInterfaces.td"

include "mlir/IR/EnumAttr.td"

// TODO add op traits, like commutative and pure, to the ops

// === traits ===

// This is ignored, if operand 1 is not a register
class Operand0IsDestN<int N>
  : ParamNativeOpTrait<"Operand0IsDestN", !cast<string>(N)>;

def HasImm : NativeOpTrait<"HasImm">;

// === types ===

/*  TEST, TODO remove later

def Case1: I64EnumAttrCase<"Case1", 0x55aaff>;

def MyIntEnum : I64EnumAttr<"MyIntEnum", "example", [Case1]>{
  let cppNamespace = "::amd64";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

*/

// base type
class AMD64_Type<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<AMD64_Dialect, name, traits, baseCppClass> {
  let mnemonic = name;
}

def memLoc : AMD64_Type<"memLoc">;

class AMD64_RegisterType<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : AMD64_Type<name, traits # [RegisterTypeInterface], baseCppClass> {
  int bitwidth = ?;
  
  let extraClassDeclaration = "inline static unsigned getBitwidth() { return " # bitwidth # "; }";
}

// to get the register of a value of AMD64_GPRegisterType, use `registerOf()`.
class AMD64_GPRegisterType<string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : AMD64_RegisterType<name, traits, baseCppClass> {
}

// GP types

foreach bitwidth = [8, 16, 32, 64] in {
  def gpr#bitwidth : AMD64_GPRegisterType<"gpr"#bitwidth>;
}

// TODO this doesn't seem optimal
def anygprType: TypeConstraint<Or<[gpr8.predicate, gpr16.predicate, gpr32.predicate, gpr64.predicate]>>;

// === memory ops ===
class MemOp<string mnemonic, list<Trait> traits = []>: AMD64_Op<mnemonic, traits # [EncodeOpInterface]>{
  code encoding = ?;
  let extraClassDeclaration = [{
    inline FeOp encode() {
      }] # encoding # [{
    }
  }];
}

// TODO for local variables/memref.alloca/llvm.alloca: define a memop-like alloca operation, that gives back a memloc and also implements the encoding interface, it can then be used just as a normal variable
//      not sure about that anymore

// trying to reduce code duplication with this somewhat generic mem op, which will be instantiated with different bits.
// sadly looks quite hiddeous, but that's better than 5 different memory ops with basically the same code with a few things stripped out
class MemSIBD_subset<string mnemonic, list<Trait> traits = []>: MemOp<mnemonic, traits>{
  bit hasBase = ?;
  bit hasScaleIndex = ?;
  bit hasDisplacement = ?;

  assert !or(hasBase, hasDisplacement), "a memop without a base and without a displacement does not make sense";
  
  let arguments =
    !con(
      // base
      !if(hasBase, (ins anygprType:$base), (ins)),
      // scale + index
      !if(hasScaleIndex, (ins Property<"uint8_t">:$scale, anygprType:$index), (ins)),
      // displacement
      !if(hasDisplacement, (ins Property<"int64_t">:$displacement), (ins))
    );
  let results = (outs memLoc:$memloc);

  // TODO I'm pretty sure the encoding for just displacement won't make much sense, test that later
  let encoding = [{
    return FE_MEM( }]
      # !if(hasBase, "registerOf(mlir::dyn_cast<mlir::OpResult>(getBase())),", "0,")
      # !if(hasScaleIndex, [{
      getProperties().scale,
      registerOf(mlir::dyn_cast<mlir::OpResult>(getIndex())),
      }], "0, 0, ")
      # !if(hasDisplacement, "getProperties().displacement", "0")
      # [{
    );
  }];

  // custom builder to initialize the properties, this is currently the only way to do it
  let builders = [
    OpBuilder<
      // arguments
      !con(
        // base
        !if(hasBase,         (ins "::mlir::Value":$base),                       (ins)),
        // scale + index
        !if(hasScaleIndex,   (ins "uint8_t":$scaleArg, "::mlir::Value":$index), (ins)),
        // displacement
        !if(hasDisplacement, (ins "int64_t":$displacementArg),                  (ins))
      ), [{
      auto& prop = $_state.getOrAddProperties<MemSIBD::Properties>();
      (void) prop;
      }]
      # !if(hasScaleIndex,   "prop.scale = scaleArg;",               "")
      # !if(hasDisplacement, "prop.displacement = displacementArg;", "")
      # [{
      build($_builder, $_state
      }]
      # !if(hasBase,       ", base",  "")
      # !if(hasScaleIndex, ", index", "")
      # [{
      );
    }]>
  ];
}

// memory op with base, scale, index, and displacement, most general case
let hasBase = 1, hasScaleIndex = 1, hasDisplacement = 1 in
def MemSIBD : MemSIBD_subset<"MemSIBD">;

// memory op with just base, scale, and index
let hasBase = 1, hasScaleIndex = 1, hasDisplacement = 0 in
def MemSIB : MemSIBD_subset<"MemSIB">;

// memory op with just base and displacement
let hasBase = 1, hasScaleIndex = 0, hasDisplacement = 1 in
def MemBD : MemSIBD_subset<"MemBD">;

// memory op with just scale, index, and displacement (useful for example for LEAs)
let hasBase = 0, hasScaleIndex = 1, hasDisplacement = 1 in
def MemSID : MemSIBD_subset<"MemSID">;

// memory op with just base
let hasBase = 1, hasScaleIndex = 0, hasDisplacement = 0 in
def MemB : MemSIBD_subset<"MemB">;

// memory op with just displacement
let hasBase = 0, hasScaleIndex = 0, hasDisplacement = 1 in
def MemD : MemSIBD_subset<"MemD">;

// === properties ===

// this is heavily based on mlir/test/lib/Dialect/Test/TestOps.td|cpp|h
// one big instruction info property, so that the tablegen doesn't have to change, when new info is added
def InstructionInfoProp : Property<"InstructionInfo"> {
  let convertToAttribute = "$_storage.asAttribute($_ctxt)";
  let convertFromAttribute = "return InstructionInfo::setFromAttr($_storage, $_attr, $_diag);";
  let hashProperty = "$_storage.hash();";
}

// === ops (= instructions) ===

// use the InstructionOpInterface to inject generic functionality into all instructions
class Instruction<string mnemonic, list<Trait> traits = []> :
  AMD64_Op<mnemonic, traits # [InstructionOpInterface]> {

  string operandConstraint1 = "{.whichOperand = -1, .hasReg = false}";
  string operandConstraint2 = "{.whichOperand = -1, .hasReg = false}";

  string resultConstraint1 = "{.whichResult = -1, .hasReg = false}";
  string resultConstraint2 = "{.whichResult = -1, .hasReg = false}";

  // basically equivalent to arguments, but arguments are conditionally furthered based on traits, and more info, so these here are the 'raw' operands.
  dag operands = ?;

  // this var is currently unused, but are not hurtful either
  // if the instruction has an immediate, it has the HasImm trait
  defvar hasImm = !not(!empty(!filter(trait, traits, !eq(trait, HasImm))));

  list<code> extraExtraClassDeclarations = [];
  let extraClassDeclaration = "static constexpr FeMnem getFeMnemonic() { return FE_" # mnemonic # "; }" /* declaration, because getFeMnemonic is inline */ #
    "static constexpr amd64::OperandRegisterConstraints getOperandRegisterConstraints () { return {" # operandConstraint1 # "," # operandConstraint2 # "}; } " #
    "static constexpr amd64::ResultRegisterConstraints getResultRegisterConstraints () {   return {" # resultConstraint1  # "," # resultConstraint2  # "}; } " #
    "InstructionInfo& instructionInfo();" #
    !interleave(extraExtraClassDeclarations, "\n");

  list<code> extraExtraClassDefinitions = [];
  let extraClassDefinition = "InstructionInfo& $cppClass::instructionInfo() { return this->getProperties().instructionInfoImpl; }" # !interleave(extraExtraClassDefinitions, "\n");

  // always append the instruction info property. Immediates are stored in the instruction info
  let arguments = !con(operands, (ins InstructionInfoProp:$instructionInfoImpl));
}

// Test Op without properties, for DRR
def TestOp64 : AMD64_Op<"test64", []>{
  let arguments = (ins gpr64:$src1, gpr64:$src2);
  let results = (outs gpr64:$dst);
}

// immediate instruction multiclass
multiclass Instr_i<string mnemonic, list<Trait> traits = []> {
  // TODO probably needs custom builder or declaration to initialize the immediate I don't think this is possible generically, as we would have to repeat all possible arguments. Let's hope there is in-tree support for this soon
  def "i" : Instruction<mnemonic # "i", traits # [HasImm]>;
}

// produces "rr", "rm", "mr", "mi", "ri" variants, depending on the bits set here
multiclass Instr_rmi<string mnemonic, AMD64_GPRegisterType result, AMD64_GPRegisterType op1, AMD64_GPRegisterType op2, list<Trait> traits = [], bit produceMem = 1, bit produceImm = 1, bit produceRR = 1> {
  defvar rr_Traits = traits;
  defvar ri_Traits = !listremove(traits, [Commutative]);
  defvar rm_mr_mi_Traits = !listremove(traits, [Pure, Commutative]);
  // don't give back memLocs as a result, only leads to quadratic chaining of encodings etc. If necessary, rewrite instructions so that they point to the same memory operand
  if produceMem then
    let results = (outs) in {
      let operands = (ins memLoc:$src1, op2:$src2) in
      def "mr" : Instruction<mnemonic # "mr", rm_mr_mi_Traits>;

      if produceImm then
        let operands = (ins memLoc:$src1) in
        defm "m" : Instr_i<mnemonic # "m", rm_mr_mi_Traits>;
    }

  let results = (outs result: $dst) in {
    if produceRR then
      let operands = (ins op1:$src1, op2:$src2) in 
      def "rr" : Instruction<mnemonic # "rr", rr_Traits>;

    if produceMem then
      let operands = (ins op1:$src1, memLoc:$src2) in
      def "rm" : Instruction<mnemonic # "rm", rm_mr_mi_Traits>;

    if produceImm then
      let operands = (ins op1:$src1) in
      defm "r" : Instr_i<mnemonic # "r", ri_Traits>;
  }
}

multiclass Instr8_16_32_64_rmi<string mnemonic, list<Trait> traits = [], bit produceMem = 1, bit produceImm = 1, bit produceRR = 1> {

  foreach suffix = ["8", "16", "32", "64"] in {
    defvar gprType = !cast<AMD64_GPRegisterType>("gpr"#suffix);
    defm suffix : Instr_rmi<mnemonic#suffix, gprType, gprType, gprType, traits, produceMem, produceImm, produceRR>;
  }

  /* this foreach is the same as:
    defm "8" : Instr_rmi<mnemonic# "8",  gpr8,  gpr8,  gpr8,  traits>;
    defm "16": Instr_rmi<mnemonic# "16", gpr16, gpr16, gpr16, traits>;
    defm "32": Instr_rmi<mnemonic# "32", gpr32, gpr32, gpr32, traits>;
    defm "64": Instr_rmi<mnemonic# "64", gpr64, gpr64, gpr64, traits>;
  */
}

/* ways to define an instr:

// manually:
 
def ADD8rr : Instruction<"ADD8rr"> {
  let operands = (ins gpr8:$src1, gpr8:$src2);
  let results = (outs gpr8:$dst);
}

// for a specified bitwidth:

defm ADD8 : Instr_rmi<"ADD8", gpr8, gpr8, gpr8>;

// for all bitwidths:

defm ADD : Instr8_16_32_64_rmi<"ADD">;

*/

// ops defined through multiclasses
defm ADD : Instr8_16_32_64_rmi<"ADD", [Operand0IsDestN<0>]>;
defm SUB : Instr8_16_32_64_rmi<"SUB", [Operand0IsDestN<0>]>;
defm CMP : Instr8_16_32_64_rmi<"CMP">;
// MOVxxri/MOVxxmi are generated below, as they are special for their custom builders
defm MOV : Instr8_16_32_64_rmi<"MOV", [Operand0IsDestN<0>], /* produceMem */ 1, /* produceImm */ 0>;

defm AND : Instr8_16_32_64_rmi<"AND", [Operand0IsDestN<0>]>;
defm OR : Instr8_16_32_64_rmi<"OR", [Operand0IsDestN<0>]>;
defm XOR : Instr8_16_32_64_rmi<"XOR", [Operand0IsDestN<0>]>;

// miscellaneous 'special' instructions dependant bitwidth
foreach bitwidth = ["8", "16", "32", "64"] in {
  defvar gprType = !cast<AMD64_GPRegisterType>("gpr"#bitwidth);


  // zero result instructions
  let results = (outs) in {
    // for 8 bit it's ah:al, not dx:ax (i love you x86)
    let resultConstraint1  = "{.whichResult = 0, .reg = FE_AX}", resultConstraint2  = !if(!eq(bitwidth, "8"), "{.whichResult = 1, .reg = FE_AH}", "{.whichResult = 1, .reg = FE_DX}") in 
    let operands = (ins gprType:$src1, memLoc:$src2) in
    def MUL # bitwidth # "m": Instruction<"MUL" # bitwidth # "m", [Operand0IsDestN<0>]>;
  }

  // one register result instructions
  let results = (outs gprType:$dst) in {
    // for 8 bit it's ah:al, not dx:ax (i love you x86)
    let resultConstraint1  = "{.whichResult = 0, .reg = FE_AX}", resultConstraint2  = !if(!eq(bitwidth, "8"), "{.whichResult = 1, .reg = FE_AH}", "{.whichResult = 1, .reg = FE_DX}") in 
    let operands = (ins gprType:$src1, gprType:$src2) in 
    def MUL # bitwidth # "r" : Instruction<"MUL" # bitwidth # "r", [Operand0IsDestN<0>]>;

    // only MOVxxri/MOVxxmi variants, the others are generated normally above

    let operands = (ins) in // TODO wait: should these also have an operand? It kind of doesn't make sense, but the other ri's have to have one. Maybe use a dummy one?
    // currently native properties don't get a generated builder to set their value upon op building, so we do that ourselves. Sadly this is very hard to do for any immediate op (because we need to replicate all args), so for now it only works on MOVs
    let builders = [
      OpBuilder<(ins "int" # bitwidth # "_t":$immArg), [{
        $_state.getOrAddProperties<InstructionInfo>().imm = immArg;
        build($_builder, $_state);
      }]>
    ] in
    defm MOV # bitwidth # "r" : Instr_i<"MOV" # bitwidth # "r", [Pure]>;

    let operands = (ins memLoc) in
    let builders = [
      OpBuilder<(ins "::mlir::Value":$memLocArg, "int" # bitwidth # "_t":$immArg), [{
        $_state.getOrAddProperties<InstructionInfo>().imm = immArg;
        build($_builder, $_state, memLocArg);
      }]>
    ] in
    defm MOV # bitwidth # "m" : Instr_i<"MOV" # bitwidth # "m">;

    // shifts
    // TODO theoretically there are mi/mr variants, but define those once we need them
    let operandConstraint1 = "{.whichOperand = 1, .reg = FE_CX}" in
    let operands = (ins gprType:$src1, gprType:$src2) in {
      def SHL # bitwidth # "rr" : Instruction<"SHL" # bitwidth # "rr", [Operand0IsDestN<0>]>;
      def SHR # bitwidth # "rr" : Instruction<"SHR" # bitwidth # "rr", [Operand0IsDestN<0>]>;
      def SAR # bitwidth # "rr" : Instruction<"SAR" # bitwidth # "rr", [Operand0IsDestN<0>]>;
    }

    let operands = (ins gprType:$src1) in {
      defm SHL # bitwidth # "r" : Instr_i<"SHL" # bitwidth # "r", [Operand0IsDestN<0>]>;
      defm SHR # bitwidth # "r" : Instr_i<"SHR" # bitwidth # "r", [Operand0IsDestN<0>]>;
      defm SAR # bitwidth # "r" : Instr_i<"SAR" # bitwidth # "r", [Operand0IsDestN<0>]>;
    }

    // movsx/movzx
    if !not(!eq(bitwidth, "8")) then {
      foreach innerBitwidth = ["8", "16", "32"] in {
        defvar innerGprType = !cast<AMD64_GPRegisterType>("gpr"#innerBitwidth);

        // movsx
        let operands = (ins gprType:$src1, innerGprType:$src2) in
        def MOVSX # "r" # bitwidth # "r" # innerBitwidth: Instruction<"MOVSX" # "r" # bitwidth # "r" # innerBitwidth, [Operand0IsDestN<0>]>;

        let operands = (ins gprType:$src1, memLoc:$src2) in
        def MOVSX # "r" # bitwidth # "m" # innerBitwidth: Instruction<"MOVSX" # "r" # bitwidth # "m" # innerBitwidth, [Operand0IsDestN<0>]>;

        // movzx
        if !not(!eq(innerBitwidth, "32")) then{
          let operands = (ins gprType:$src1, innerGprType:$src2) in
          def MOVZX # "r" # bitwidth # "r" # innerBitwidth: Instruction<"MOVZX" # "r" # bitwidth # "r" # innerBitwidth, [Operand0IsDestN<0>]>;

          let operands = (ins gprType:$src1, memLoc:$src2) in
          def MOVZX # "r" # bitwidth # "m" # innerBitwidth: Instruction<"MOVZX" # "r" # bitwidth # "m" # innerBitwidth, [Operand0IsDestN<0>]>;
        }
      }
    }
  }
}

// special instructions independant of bitwidth

// JMP is the cf.br equivalent, is a terminator has block args, but no result. It uses MLIRs built-in successor list to define its successors
// this is kept close to cf.br
def JMP : Instruction<"JMP", [Terminator]>{
  let operands = (ins Variadic<AnyType>:$destOperands);

  let successors = (successor AnySuccessor:$dest);

  // from the cf.br op
  let builders = [
    OpBuilder<(ins "::mlir::Block *":$dest, CArg<"::mlir::ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]
  >];

  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

// conditional jumps as terminator, 2 successors, with TODO invert condition interface
// heavily based on the cf.cond_br, just without the condition
class Jcc<string invertedJumpClassName, string mnemonic> : Instruction<mnemonic, [AttrSizedOperandSegments, Terminator]>{
  let operands = (ins Variadic<AnyType>:$trueDestOperands, Variadic<AnyType>:$falseDestOperands);

  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  
  let extraExtraClassDeclarations = ["amd64::" # invertedJumpClassName # " invert(mlir::OpBuilder&);"];

  let extraExtraClassDefinitions = [[{
    amd64::}] # invertedJumpClassName # [{ $cppClass::invert(mlir::OpBuilder& builder) {
      // inverted destinations
      return builder.create<amd64:: }] # invertedJumpClassName # [{>(getLoc(), getFalseDestOperands(), getTrueDestOperands(), getFalseDest(), getTrueDest());
    }
  }]];
}

multiclass JccAndInverted<string className, string invertedJumpClassName, string mnemonic = className, string invertedJumpMnemonic = invertedJumpClassName> {
  def className             : Jcc<invertedJumpClassName, mnemonic>;
  def invertedJumpClassName : Jcc<className, invertedJumpMnemonic>;
}

defm "" : JccAndInverted<"JZ", "JNZ">;
defm "" : JccAndInverted<"JE", "JNE", "JZ", "JNZ">;
defm "" : JccAndInverted<"JL", "JGE">;
defm "" : JccAndInverted<"JLE", "JG">;
defm "" : JccAndInverted<"JB", "JAE", "JC", "JNC">;
defm "" : JccAndInverted<"JBE", "JA">;


// similar to func.call
def CALL : Instruction<"CALL">{
  let operands = (ins FunctionType:$callee, Variadic<AnyType>:$args);
  let results = (outs gpr64:$ret); // only single return value for now
}

// returns
let results = (outs) in {
  let operandConstraint1 = "{.whichOperand = 0, .reg = FE_AX}" in
  let operands = (ins gpr64:$ret) in 
  def RET : Instruction<"RET", [Terminator]>;

  let operands = (ins) in 
  defm RET : Instr_i<"RET", [Terminator]>;
}

#endif
