cmake_minimum_required(VERSION 3.13.4)
project(monaco LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ standard to conform to")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  message(STATUS "CMake module path: ${CMAKE_MODULE_PATH}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

set(MONACO_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(MONACO_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${MONACO_SOURCE_DIR}/include)
include_directories(${MONACO_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Configure fadec as a target
macro(configure_fadec fadec_src_path meson_buildtype)
    add_custom_target(
       fadec
       COMMAND meson setup build --buildtype="${meson_buildtype}" && cd build && meson compile
       BYPRODUCTS "${fadec_src_path}/build/libfadec.a"
       WORKING_DIRECTORY "${fadec_src_path}"
    )
    include_directories("${fadec_src_path}")
    include_directories("${fadec_src_path}/build")
endmacro()

set(FADEC_SRC_PATH "${MONACO_SOURCE_DIR}/lib/fadec")
set(FADEC_LIB_PATH "${FADEC_SRC_PATH}/build/libfadec.a")
if (CMAKE_BUILD_TYPE STREQUAL "Release")
	configure_fadec("${FADEC_SRC_PATH}" release) # TODO ensure this is actually replaced on replacing builds
else()
    configure_fadec("${FADEC_SRC_PATH}" debug)
endif()

add_subdirectory(src)

# define dialects etc.
# these are included directly, to not clutter the filesystem with (lib|include)/CMakeLists.txt which just include another subdirectory
add_subdirectory(include/AMD64)
add_subdirectory(lib/AMD64)

