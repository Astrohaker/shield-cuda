# cuda-usb: SHIELD Tablet (Tegra K1) GPU as a CUDA device for a Linux PC over USB.
#   make            build server (Android), host libcuda.so shim and examples
#   make install-server   push the server to the tablet (connect.sh does this too)
# Bundled toolchains (see toolchain/); override to use installed ones.
CUDA_HOME ?= $(CURDIR)/toolchain/cuda
NDK       ?= $(CURDIR)/toolchain/ndk
NDK_CC    := $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi23-clang
NVCC_K1   := scripts/nvcc-k1
CFLAGS    ?= -O2 -Wall -Wextra -Wno-unused-parameter

BUILD := build
SERVER := $(BUILD)/android/cudausb-server
SHIM   := $(BUILD)/lib/libcuda.so.1
EXAMPLES := $(BUILD)/examples/devicequery $(BUILD)/examples/vecadd $(BUILD)/examples/vecadd.cubin \
            $(BUILD)/examples/launchbench $(BUILD)/examples/launchbench.cubin

KT_CUBIN := python/k1torch/kernels.cubin

all: $(SERVER) $(SHIM) $(BUILD)/lib/libcuda.so $(EXAMPLES) $(KT_CUBIN)

$(SERVER): server/server.c common/protocol.h
	@mkdir -p $(@D)
	$(NDK_CC) $(CFLAGS) -I$(CUDA_HOME)/include $< -o $@ -ldl

$(SHIM): client/libcuda.c common/protocol.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -fPIC -shared -fvisibility=hidden -Wl,-soname,libcuda.so.1 \
		-I$(CUDA_HOME)/include $< -o $@ -lpthread

$(BUILD)/lib/libcuda.so: $(SHIM)
	ln -sf libcuda.so.1 $@

$(BUILD)/examples/%: examples/%.c $(BUILD)/lib/libcuda.so
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(CUDA_HOME)/include $< -o $@ -L$(BUILD)/lib -lcuda

$(BUILD)/examples/%.cubin: examples/%.cu
	@mkdir -p $(@D)
	$(NVCC_K1) -cubin -o $@ $<

$(KT_CUBIN): python/k1torch/kernels.cu
	$(NVCC_K1) -cubin -o $@ $<

install-server: $(SERVER)
	scripts/connect.sh

clean:
	rm -rf $(BUILD)

.PHONY: all install-server clean
