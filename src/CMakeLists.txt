get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# all possible components:
#llvm_map_components_to_libnames(llvm_libs aarch64asmparser aarch64codegen aarch64desc  aarch64info aarch64utils aggressiveinstcombine all amdgpuasmparser amdgpucodegen amdgpudesc  amdgpuinfo  amdgpuutils analysis armasmparser armcodegen armdesc  arminfo armutils asmparser asmprinter avrasmparser avrcodegen avrdesc  avrinfo binaryformat bitreader bitstreamreader bitwriter bpfasmparser bpfcodegen bpfdesc  bpfinfo cfguard codegen core coroutines  debuginfocodeview debuginfodwarf debuginfomsf debuginfopdb demangle engine executionengine hexagonasmparser hexagoncodegen hexagondesc  hexagoninfo instcombine instrumentation irreader lanaiasmparser lanaicodegen lanaidesc  lanaiinfo mipsasmparser mipscodegen mipsdesc  mipsinfo mirparser msp430asmparser msp430codegen msp430desc  msp430info native nativecodegen nvptxcodegen nvptxdesc nvptxinfo object option powerpcasmparser powerpccodegen powerpcdesc  powerpcinfo riscvasmparser riscvcodegen riscvdesc  riscvinfo sparcasmparser sparccodegen sparcdesc  sparcinfo support symbolize systemzasmparser systemzcodegen systemzdesc  systemzinfo tablegen target textapi transformutils veasmparser vecodegen vectorize vedesc  veinfo webassemblyasmparser webassemblycodegen webassemblydesc  webassemblyinfo webassemblyutils windowsdriver x86asmparser AMD64codegen AMD64desc  AMD64info xcorecodegen xcoredesc  xcoreinfo)
# -lLLVM-jit not found comes from "engine"
llvm_map_components_to_libnames(llvm_libs executionengine support core irreader object native target mcjit)

message(STATUS "LLVM libs: ${llvm_libs}")

set(LIBS
        ${llvm_libs}
        ${dialect_libs}
        ${conversion_libs}
        MLIRAMD64

        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRLLVMDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIROptLib
        rt
        dl
        m
        z
        zstd
        tinfo
        xml2
        ${FADEC_LIB_PATH}
        )

# this solves the linker errors with the target specific initialize methods
set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  )
message(STATUS "LLVM link components: ${LLVM_LINK_COMPONENTS}")
#
file(GLOB sources "*.cpp")
add_llvm_executable(monaco ${sources}
    DEPENDS fadec
)

# this is necessary here too, otherwise the preprocessor doesn't tablegen files in this dir
include_directories(${CMAKE_CURRENT_BINARY_DIR})

llvm_update_compile_flags(monaco)

# Link against the required LLVM libraries for JIT support
target_link_libraries(monaco PRIVATE ${LIBS})

# if its the debug build, add sanitizer flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # TODO add thread sanitizer for clang?
    target_compile_options(monaco PUBLIC -fsanitize=address,undefined,leak -fno-sanitize-recover=all)
    target_link_options(monaco PUBLIC -fsanitize=address,undefined,leak -fno-sanitize-recover=all)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # to make absolutely sure there are no assertions, even if the LLVM build is debug
    target_compile_options(monaco PUBLIC -O3 -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # to make absolutely sure there are no assertions, even if the LLVM build is debug
    target_compile_options(monaco PUBLIC -O2 -DNDEBUG -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer)
endif()

target_compile_options(monaco PRIVATE -Wno-enum-compare)

mlir_check_all_link_libraries(monaco)
