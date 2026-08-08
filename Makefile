.PHONY: host-build host-check host-bench fixtures fixtures-check tool-tests firmware-bootstrap firmware-build firmware-inspect clean

HOST_BUILD := build/host

host-build:
	cmake -S . -B $(HOST_BUILD) -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build $(HOST_BUILD)

host-check: host-build tool-tests fixtures-check
	ctest --test-dir $(HOST_BUILD) --output-on-failure

host-bench: host-build
	$(HOST_BUILD)/bench/tracker_host_benchmark

fixtures:
	python3 tools/generate_fixtures.py

fixtures-check:
	python3 tools/generate_fixtures.py --check

tool-tests:
	python3 tools/test_benchmark_tools.py

firmware-bootstrap:
	./tools/bootstrap_esp_idf.sh

firmware-build:
	./tools/build_firmware.sh rev1
	./tools/build_firmware.sh rev3

firmware-inspect:
	./tools/inspect_target_elf.sh rev1
	./tools/inspect_target_elf.sh rev3

clean:
	cmake -E rm -rf $(HOST_BUILD) firmware/build
