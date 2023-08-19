# override this using env var or directly in the Makefile
LLVM_BUILD_DIR=~/programming/Libs/Cpp/llvm-project/build
LLVM_RELEASE_BUILD_DIR=$(shell dirname $(LLVM_BUILD_DIR))/buildRelease
MONACO_BUILD_DIR=build

# just to jot it down somewhere: current llvm commit used: a403d75be7 (used to be 583d492c6)

.phony: release debug makeCMakeBearable clean setup test relWithDebug

all: release

setup:
	mkdir -p $(MONACO_BUILD_DIR)

clean:
	rm -rf $(MONACO_BUILD_DIR)
	rm -rf lib/fadec/build


test:
	$(MAKE) debug
	lit -svj1 tests

release: setup
	export MONACO_BUILD_DIR
	[ -f $(MONACO_BUILD_DIR)/isRelease ] || $(MAKE) clean && $(MAKE) setup
	touch $(MONACO_BUILD_DIR)/isRelease
	$(MAKE) cmake_build_type=Release LLVM_BUILD_DIR=$(LLVM_RELEASE_BUILD_DIR) makeCMakeBearable

debug: setup
	export MONACO_BUILD_DIR
	[ -f $(MONACO_BUILD_DIR)/isDebug ] || $(MAKE) clean && $(MAKE) setup
	touch $(MONACO_BUILD_DIR)/isDebug
	$(MAKE) cmake_build_type=Debug makeCMakeBearable

relWithDebug: setup
	export MONACO_BUILD_DIR
	[ -f $(MONACO_BUILD_DIR)/isRelWithDebug ] || $(MAKE) clean && $(MAKE) setup
	touch $(MONACO_BUILD_DIR)/isRelWithDebug
	$(MAKE) cmake_build_type=RelWithDebInfo LLVM_BUILD_DIR=$(LLVM_RELEASE_BUILD_DIR) makeCMakeBearable

makeCMakeBearable: setup
	# the - makes it continue, even if the build fails, so that the sed is executed
	-cwd=$(shell pwd)                                                                                                                           && \
	cd $(MONACO_BUILD_DIR)                                                                                                                      && \
	cmake "$$cwd" -DCMAKE_BUILD_TYPE=$(cmake_build_type) -DLLVM_DIR=$(LLVM_BUILD_DIR)/lib/cmake/llvm -DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir && \
	cmake --build . -j$(shell nproc)                                                                                                            && \
	cd "$$cwd"
	sed -i 's/-std=gnu++23/-std=c++2b/g' $(MONACO_BUILD_DIR)/compile_commands.json # to make it work for clangd, can't be bothered to try with cmake
