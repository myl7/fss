SOURCES := $(shell find src include samples -name '*.cuh' -o -name '*.cu')
CPU_ID ?= 0
GPU_ID ?= 0
CUDA_ARCH ?=
JOBS ?= 4
FLAMEGRAPH_DIR ?= ../FlameGraph
FLAMEGRAPH_BENCH ?= BM_DpfEval_Uint_Aes/20
GPU_PROFILE_BENCH ?= BM_DpfEval_Uint_ChaCha/20
GPU_PROFILE_OUT ?= build/nsys_gpu
CMAKE_CUDA_ARCH_FLAGS := $(if $(strip $(CUDA_ARCH)),-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH))
export OMP_NUM_THREADS = 1

.PHONY: format format_check bench_cpu bench_gpu bench_build flamegraph profile_gpu ptx_info

format:
	clang-format -i $(SOURCES)
format_check:
	clang-format --dry-run --Werror $(SOURCES)

bench_cpu:
	python3 third_party/bench.py run --libraries main --platform cpu --cpu $(CPU_ID) --jobs $(JOBS) $(if $(strip $(CUDA_ARCH)),--cuda-arch $(CUDA_ARCH))
bench_gpu:
	python3 third_party/bench.py run --libraries main --platform gpu --cpu $(CPU_ID) --jobs $(JOBS) --gpu $(GPU_ID) $(if $(strip $(CUDA_ARCH)),--cuda-arch $(CUDA_ARCH))
bench_build:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_BENCH=ON $(CMAKE_CUDA_ARCH_FLAGS)
	cmake --build build --parallel $(JOBS)

flamegraph:
	cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF -DBUILD_BENCH=ON
	cmake --build build --parallel $(JOBS)
	perf record -g -o build/perf.data ./build/bench_cpu --benchmark_filter=$(FLAMEGRAPH_BENCH)
	perf script -i build/perf.data | "$(FLAMEGRAPH_DIR)/stackcollapse-perf.pl" | "$(FLAMEGRAPH_DIR)/flamegraph.pl" > build/flamegraph.svg

profile_gpu: bench_build
	CUDA_VISIBLE_DEVICES=$(GPU_ID) nsys profile --trace=cuda,nvtx --force-overwrite=true --output=$(GPU_PROFILE_OUT) ./build/bench_gpu --benchmark_filter=$(GPU_PROFILE_BENCH)

ptx_info:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_BENCH=ON $(CMAKE_CUDA_ARCH_FLAGS) -DCMAKE_CUDA_FLAGS="--ptxas-options=-v"
	cmake --build build --parallel $(JOBS) --clean-first 2>&1 | grep "ptxas info" | tee build/ptx_info.log || true
