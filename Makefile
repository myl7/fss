SOURCES := $(shell find src include samples -name '*.cuh' -o -name '*.cu')
PRETTIER_SOURCES := $(shell for file in $$(git ls-files '*.md' '*.yaml' '*.yml' '*.json' '*.jsonc' '*.js' '*.jsx' '*.ts' '*.tsx' '*.mjs' '*.cjs' '*.css' '*.html' 2>/dev/null); do if [ -f "$$file" ] && [ ! -L "$$file" ]; then printf '%s ' "$$file"; fi; done)
CPU_ID ?= 0
GPU_ID ?= 0
CUDA_ARCH ?=
CPU_SG := /sys/devices/system/cpu/cpu$(CPU_ID)/cpufreq/scaling_governor
FLAMEGRAPH_DIR ?= ../FlameGraph
FLAMEGRAPH_BENCH ?= BM_DpfEval_Uint_Aes/20
GPU_PROFILE_BENCH ?= BM_DpfEval_Uint/20
GPU_PROFILE_OUT ?= build/nsys_gpu
CMAKE_CUDA_ARCH_FLAGS := $(if $(strip $(CUDA_ARCH)),-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH))
export OMP_NUM_THREADS = 1

.PHONY: format format_check bench_cpu bench_gpu bench_build flamegraph profile_gpu ptx_info

format:
	clang-format -i $(SOURCES)
ifneq ($(strip $(PRETTIER_SOURCES)),)
	prettier --write $(PRETTIER_SOURCES)
endif
format_check:
	clang-format --dry-run --Werror $(SOURCES)
ifneq ($(strip $(PRETTIER_SOURCES)),)
	prettier --check $(PRETTIER_SOURCES)
endif

bench_cpu: bench_build
	cat $(CPU_SG) > /tmp/cpu_sg
	echo performance | sudo tee $(CPU_SG)
	taskset -c $(CPU_ID) ./build/bench_cpu | tee build/bench_cpu.log
	cat /tmp/cpu_sg | sudo tee $(CPU_SG)
bench_gpu: bench_build
	CUDA_VISIBLE_DEVICES=$(GPU_ID) timeout 10 ./build/bench_gpu || true
	CUDA_VISIBLE_DEVICES=$(GPU_ID) ./build/bench_gpu | tee build/bench_gpu.log
bench_build:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_BENCH=ON $(CMAKE_CUDA_ARCH_FLAGS)
	cmake --build build -j

flamegraph:
	cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF -DBUILD_BENCH=ON
	cmake --build build -j
	perf record -g -o build/perf.data ./build/bench_cpu --benchmark_filter=$(FLAMEGRAPH_BENCH)
	perf script -i build/perf.data | "$(FLAMEGRAPH_DIR)/stackcollapse-perf.pl" | "$(FLAMEGRAPH_DIR)/flamegraph.pl" > build/flamegraph.svg

profile_gpu: bench_build
	CUDA_VISIBLE_DEVICES=$(GPU_ID) nsys profile --trace=cuda,nvtx --force-overwrite=true --output=$(GPU_PROFILE_OUT) ./build/bench_gpu --benchmark_filter=$(GPU_PROFILE_BENCH)

ptx_info:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_BENCH=ON $(CMAKE_CUDA_ARCH_FLAGS) -DCMAKE_CUDA_FLAGS="--ptxas-options=-v"
	cmake --build build -j --clean-first 2>&1 | grep "ptxas info" | tee build/ptx_info.log || true
