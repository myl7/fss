SOURCES := $(shell find src include samples -name '*.cuh' -o -name '*.cu')
CPU_ID ?= 0
CPU_SG := /sys/devices/system/cpu/cpu$(CPU_ID)/cpufreq/scaling_governor

.PHONY: format format_check bench_cpu bench_gpu bench_build

format:
	clang-format -i $(SOURCES)
format_check:
	clang-format --dry-run --Werror $(SOURCES)

bench_cpu: bench_build
	cat $(CPU_SG) > /tmp/cpu_sg
	echo performance | sudo tee $(CPU_SG)
	taskset -c $(CPU_ID) ./build/bench_cpu
	cat /tmp/cpu_sg | sudo tee $(CPU_SG)
bench_gpu: bench_build
	timeout 10 ./build/bench_gpu || true
	./build/bench_gpu
bench_build:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_BENCH=ON
	cmake --build build -j
