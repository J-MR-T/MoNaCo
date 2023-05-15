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
class Operand1IsDestN<int N>
  : ParamNativeOpTrait<"Operand1IsDestN", !cast<string>(N)>;

// TODO replace with static method on op, see AMD64Types.h for details
class OperandNIsConstrainedToReg<int N, string reg>
  : ParamNativeOpTrait<"OperandNIsConstrainedToReg", !cast<string>(N) # "," # reg>;

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
// TODO maybe everything should have the encoding interface? Immediates could also be modeled that way, but might cost performance, because ops are pretty big in comparison to immediates as operands/properties
// TODO create a mem op/encoding interface

class MemOp<string mnemonic, list<Trait> traits = []>: AMD64_Op<mnemonic, traits # [EncodeOpInterface]>{
  code encoding = ?;
  let extraClassDeclaration = [{
    inline FeOp encode() {
      }] # encoding # [{
    }
  }];
}

// TODO for local variables/memref.alloca/llvm.alloca: define a memop-like alloca operation, that gives back a memloc and also implements the encoding interface, it can then be used just as a normal variable

// trying to reduce code duplication with this somewhat generic mem op, which will be instantiated with different bits.
// sadly looks quite hiddeous, but that's better than 5 different memory ops with basically the same code with a few things stripped out
class MemSIBD_subset<string mnemonic, list<Trait> traits = []>: MemOp<mnemonic, traits>{
  bit hasBase = ?; // TODO does it make sense without base? I guess as an rip relative address it does
  bit hasScaleIndex = ?;
  bit hasDisplacement = ?;

  assert !or(hasBase, hasDisplacement), "a memop without a base and without a displacement does not make sense";
  
  let arguments =
    !con(
      // base
      !cond(hasBase: (ins anygprType:$base), true: (ins)),
      // scale + index
      !cond(hasScaleIndex: (ins Property<"uint8_t">:$scale, anygprType:$index), true: (ins)),
      // displacement
      !cond(hasDisplacement: (ins Property<"int64_t">:$displacement), true: (ins))
    );
  let results = (outs memLoc:$memloc);

  // TODO I'm pretty sure the encoding for just displacement won't make much sense, test that later
  let encoding = [{
    return FE_MEM( }]
      # !cond(hasBase: "registerOf(mlir::dyn_cast<mlir::OpResult>(getBase())),", true: "0,")
      # !cond(hasScaleIndex: [{
      getProperties().scale,
      registerOf(mlir::dyn_cast<mlir::OpResult>(getIndex())),
      }], true: "0, 0, ")
      # !cond(hasDisplacement: "getProperties().displacement", true: "0")
      # [{
    );
  }];

  // custom builder to initialize the properties, this is currently the only way to do it
  let builders = [
    OpBuilder<
      // arguments
      !con(
        // base
        !cond(hasBase: (ins "::mlir::Value":$base), true: (ins)),
        // scale + index
        !cond(hasScaleIndex: (ins "uint8_t":$scaleArg, "::mlir::Value":$index), true: (ins)),
        // displacement
        !cond(hasDisplacement: (ins "int64_t":$displacementArg), true: (ins))
      ), [{
      auto& prop = $_state.getOrAddProperties<MemSIBD::Properties>();
      (void) prop; }]
      # !cond(hasScaleIndex: "prop.scale = scaleArg;", true: "")
      # !cond(hasDisplacement: "prop.displacement = displacementArg;", true: "")
      # [{
      build($_builder, $_state}]
      # !cond(hasBase: ", base", true: "")
      # !cond(hasScaleIndex: ", index", true: "")
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
  // basically equivalent to arguments, but arguments are conditionally furthered based on traits, and more info, so these here are the 'raw' operands.
  dag operands = ?;

  // this var is currently unused, but are not hurtful either
  // if the instruction has an immediate, it has the HasImm trait
  defvar hasImm = !not(!empty(!filter(trait, traits, !eq(trait, HasImm))));

  let extraClassDeclaration = "static constexpr FeMnem getFeMnemonic() { return FE_" # mnemonic # "; }" /* declaration, because getFeMnemonic is inline */ #
    "InstructionInfo& instructionInfo();";

  let extraClassDefinition = "InstructionInfo& $cppClass::instructionInfo() { return this->getProperties().instructionInfoImpl; }";

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
multiclass Instr_rmi<string mnemonic, AMD64_GPRegisterType result, AMD64_GPRegisterType op1, AMD64_GPRegisterType op2, list<Trait> traits = [], bit produceMem = 1, bit produceImm = 1> {

  // TODO we could give back the memloc as a result, but that does not make a whole lot of sense. The advantage would be that this could then implement the EncodeOpInterface and forward the encoding of the argument.
  if produceMem then
    let results = (outs) in {
      let operands = (ins memLoc:$src1, op2:$src2) in
      def "mr" : Instruction<mnemonic # "mr", traits>;

      if produceImm then
        let operands = (ins memLoc:$src1) in
        defm "m" : Instr_i<mnemonic # "m", traits>;
    }

  let results = (outs result: $dst) in {
    let operands = (ins op1:$src1, op2:$src2) in 
    def "rr" : Instruction<mnemonic # "rr", traits>;

    if produceMem then
      let operands = (ins op1:$src1, memLoc:$src2) in
      def "rm" : Instruction<mnemonic # "rm", traits>;

    if produceImm then
      let operands = (ins op1:$src1) in
      defm "r" : Instr_i<mnemonic # "r", traits>;
  }

  /*
     this thing above is basically equivalent to this:

  def rr : Instruction<mnemonic # "rr", traits>{
    let operands = (ins op1:$src1, op2:$src2);
    let results = (outs result:$dst);
  }
  def rm : Instruction<mnemonic # "rm", traits>{...}
  def mr ...
  def ri ...
  def mi ...

  */
}

multiclass Instr8_16_32_64_rmi<string mnemonic, list<Trait> traits = [], bit produceMem = 1, bit produceImm = 1> {

  foreach suffix = ["8", "16", "32", "64"] in {
    defvar gprType = !cast<AMD64_GPRegisterType>("gpr"#suffix);
    defm suffix : Instr_rmi<mnemonic#suffix, gprType, gprType, gprType, traits, produceMem, produceImm>;
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
defm ADD : Instr8_16_32_64_rmi<"ADD", [Operand1IsDestN<1>]>;
defm SUB : Instr8_16_32_64_rmi<"SUB", [Operand1IsDestN<1>]>;
defm CMP : Instr8_16_32_64_rmi<"CMP">;
// MOVxxri/MOVxxmi are generated below, as they are special for their custom builders
defm MOV : Instr8_16_32_64_rmi<"MOV", [Operand1IsDestN<1>], /* produceMem */ 1, /* produceImm */ 0>;

// miscellaneous 'special' instructions dependant bitwidth
foreach bitwidth = ["8", "16", "32", "64"] in {
  defvar gprType = !cast<AMD64_GPRegisterType>("gpr"#bitwidth);


  // zero result instructions
  let results = (outs) in {
    let operands = (ins gprType:$src1, memLoc:$src2) in 
    def MUL # bitwidth # "m": Instruction<"MUL" # bitwidth # "m", [Operand1IsDestN<1>, OperandNIsConstrainedToReg<1, "FE_AX">]>;
  }

  // one result instructions
  let results = (outs gprType:$dst) in {
    let operands = (ins gprType:$src1, gprType:$src2) in 
    def MUL # bitwidth # "r" : Instruction<"MUL" # bitwidth # "r", [Operand1IsDestN<1>, OperandNIsConstrainedToReg<1, "FE_AX">]>;

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
    defm MOV # bitwidth # "m" : Instr_i<"MOV" # bitwidth # "m", [Pure]>;
  }
}

// special instructions independant of bitwidth

// JMP is the cf.br equivalent, is a terminator and has neither operands, nor a result. It uses MLIRs built-in successor list to define its successors
// this is kept close to cf.br, but wtihout the operands, as the block args have to be destructed anyway.
def JMP : Instruction<"JMP", [Terminator]>{
  let operands = (ins);

  let successors = (successor AnySuccessor:$dst);

  // from the cf.br
  let builders = [
    OpBuilder<(ins "::mlir::Block *":$dst), [{
      $_state.addSuccessors(dst);
    }]
  >];

  let assemblyFormat = [{
    $dst attr-dict
  }];
}

/*
def ADD8rr : Instruction<"add8rr"> {
  let operands = (ins FeMnem:$mnemonic, gpr8:$src1, gpr8:$src2);
  let results = (outs gpr8:$dst);
}
*/

/*
an extensive and still totally incomplete list of problems
- how to generate all the necessary possible ops, without too much code duplication
  - multiclasses, lets, and defms probably
- how to use fadec symbols here? e.g. FE_ADD8rr (0x30600000000000), how to include this in these definitions, also automatically, without having to map it all out by hand
  - possibly needs a conversion to some kind of tablegen enum, if something like that exists
    - yes that exists and doesn't work, because it generates too much automatically :(
  - 
- 
*/

// example for 8 bit adds

/*
def ADD8rr : AMD64_Base_Op<"add8rr"> {
  let mnemonic = "add";
  let operands = (ins AMD64_i8:$src1, AMD64_i8:$src2);
  let results = (outs AMD64_i8:$dst);

  let fe_mnemonic = 0x30600000000000;
}
*/

#endif
