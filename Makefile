.PHONY: setup test train model firmware clean

PY := .tools/tracker/bin/python
ARTIFACTS := training/artifacts

setup:
	./training/setup.sh

test:
	cmake -S . -B build -G Ninja
	cmake --build build
	ctest --test-dir build --output-on-failure
	PYTHONPATH=training $(PY) -m pytest -q training/tests

train:
	PYTHONPATH=training $(PY) training/train.py --output $(ARTIFACTS)

model:
	PYTHONPATH=training $(PY) training/export.py $(ARTIFACTS)/model.pt $(ARTIFACTS)/model.onnx
	PYTHONPATH=training $(PY) training/quantize.py $(ARTIFACTS)/model.onnx $(ARTIFACTS)/calibration.npy $(ARTIFACTS)/model.espdl

firmware:
	./tools/build_firmware.sh

clean:
	cmake -E rm -rf build .build
