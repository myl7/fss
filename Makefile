SOURCES := $(shell find src include samples -name '*.cuh' -o -name '*.cu')
CPU_ID ?= 0
CPU_SG := /sys/devices/system/cpu/cpu$(CPU_ID)/cpufreq/scaling_governor
FLAMEGRAPH_DIR ?= ../FlameGraph
FLAMEGRAPH_BENCH ?= BM_DpfEval_Uint_Aes/20

.PHONY: format format_check bench_cpu bench_gpu bench_build flamegraph ptx_info

format:
	clang-format -i $(SOURCES)
format_check:
	clang-format --dry-run --Werror $(SOURCES)

bench_cpu: bench_build
	cat $(CPU_SG) > /tmp/cpu_sg
	echo performance | sudo tee $(CPU_SG)
	taskset -c $(CPU_ID) ./build/bench_cpu | tee build/bench_cpu.log
	cat /tmp/cpu_sg | sudo tee $(CPU_SG)
bench_gpu: bench_build
	timeout 10 ./build/bench_gpu || true
	./build/bench_gpu | tee build/bench_gpu.log
bench_build:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_BENCH=ON
	cmake --build build -j

flamegraph:
	cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF -DBUILD_BENCH=ON
	cmake --build build -j
	perf record -g -o build/perf.data ./build/bench_cpu --benchmark_filter=$(FLAMEGRAPH_BENCH)
	perf script -i build/perf.data | "$(FLAMEGRAPH_DIR)/stackcollapse-perf.pl" | "$(FLAMEGRAPH_DIR)/flamegraph.pl" > build/flamegraph.svg

ptx_info:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_BENCH=ON -DCMAKE_CUDA_FLAGS="--ptxas-options=-v"
	cmake --build build -j --clean-first 2>&1 | grep "ptxas info" | tee build/ptx_info.log || true
