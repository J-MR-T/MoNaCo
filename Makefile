# override this using env var or directly in the Makefile
LLVM_BUILD_DIR=~/programming/Libs/Cpp/llvm-project/build

# just to jot it down somewhere: current llvm commit used: 583d492c6

.phony: release debug makeCMakeBearable clean setup test relWithDebug

all: release

setup:
	mkdir -p build

clean:
	rm -rf build

test:
	$(MAKE) debug
	lit -svj1 tests

release: setup
	[ -f build/isRelease ] || $(MAKE) clean && $(MAKE) setup
	touch build/isRelease
	$(MAKE) cmake_build_type=Release makeCMakeBearable

debug: setup
	[ -f build/isDebug ] || $(MAKE) clean && $(MAKE) setup
	touch build/isDebug
	$(MAKE) cmake_build_type=Debug makeCMakeBearable

relWithDebug: setup
	[ -f build/isRelWithDebug ] || $(MAKE) clean && $(MAKE) setup
	touch build/isRelWithDebug
	$(MAKE) cmake_build_type=RelWithDebugInfo makeCMakeBearable

makeCMakeBearable: setup
	# the - makes it continue, even if the build fails, so that the sed is executed
	-cd build                                                                                                                               && \
	cmake .. -DCMAKE_BUILD_TYPE=$(cmake_build_type) -DLLVM_DIR=$(LLVM_BUILD_DIR)/lib/cmake/llvm -DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir && \
	cmake --build . -j$(shell nproc)                                                                                                        && \
	cd ..
	sed -i 's/-std=gnu++23/-std=c++2b/g' build/compile_commands.json # to make it work for clangd, can't be bothered to try with cmake
