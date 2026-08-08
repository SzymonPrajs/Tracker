.PHONY: host-build host-check host-bench fixtures fixtures-check tool-tests training-setup training-check training-smoke training-export training-onnx-check training-calibration-smoke training-quantize-smoke training-quantized-compare firmware-bootstrap firmware-build firmware-inspect clean

HOST_BUILD := build/host
TRAIN_PYTHON := .tools/tracker-train/bin/python
QUANT_PYTHON := .tools/tracker-quant/bin/python
TRAIN_ARTIFACTS := training/artifacts/synthetic-example

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

training-setup:
	./training/scripts/setup_mac.sh

training-check:
	PYTHONPATH=training PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(TRAIN_PYTHON) -m pytest -q training/tests

training-smoke:
	PYTHONPATH=training $(TRAIN_PYTHON) training/scripts/train_synthetic.py --smoke --epochs 12 --output $(TRAIN_ARTIFACTS)

training-export:
	PYTHONPATH=training $(TRAIN_PYTHON) training/scripts/export_onnx.py --model-factory tracker_training.model:HCDS31 --checkpoint $(TRAIN_ARTIFACTS)/best-state.pt --output $(TRAIN_ARTIFACTS)/hcds31.onnx --write-reference

training-onnx-check:
	PYTHONPATH=training $(TRAIN_PYTHON) training/scripts/check_onnx.py $(TRAIN_ARTIFACTS)/hcds31.onnx --input-npy $(TRAIN_ARTIFACTS)/hcds31.input.npy --expected-npy $(TRAIN_ARTIFACTS)/hcds31.output.npy

training-calibration-smoke:
	$(TRAIN_PYTHON) training/scripts/prepare_smoke_calibration.py $(TRAIN_ARTIFACTS)/hcds31.input.npy $(TRAIN_ARTIFACTS)/hcds31.calibration.npy

training-quantize-smoke:
	PYTHONPATH=training $(QUANT_PYTHON) training/scripts/quantize_espdl.py $(TRAIN_ARTIFACTS)/hcds31.onnx --output $(TRAIN_ARTIFACTS)/hcds31-int8.espdl --calibration $(TRAIN_ARTIFACTS)/hcds31.calibration.npy --calibration-steps 2 --quantized-output-npy $(TRAIN_ARTIFACTS)/hcds31-int8.output.npy

training-quantized-compare:
	PYTHONPATH=training $(TRAIN_PYTHON) training/scripts/compare_outputs.py $(TRAIN_ARTIFACTS)/hcds31.output.npy $(TRAIN_ARTIFACTS)/hcds31-int8.output.npy --output $(TRAIN_ARTIFACTS)/float-vs-int8.json

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
