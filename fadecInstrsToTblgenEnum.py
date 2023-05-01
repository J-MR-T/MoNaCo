mnemonicToI64 = {}

fadecFile = open("fadec-encode-public.inc", "r")

outFile = open("fadec-encode-public.td", "w")

for line in fadecFile:
    line = line.split(sep=" ")

    mnemonic = line[1]
    hexVal = line[2][:-1]

    mnemonicToI64[mnemonic] = hexVal

    outFile.write("def " + mnemonic + ": I64EnumAttrCase<\"" + mnemonic + "\"," + hexVal + ">;\n")

fadecFile.close()

outFile.write("""

def FeMnem : I64EnumAttr<"FeMnem", "AMD64 Instruction mnemonics", [""")

for mnemonic, val in mnemonicToI64.items():
    outFile.write(mnemonic + ", ")

outFile.write("""]>{
  let cppNamespace = "::amd64";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}
""")


outFile.close()

