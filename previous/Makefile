.PHONY: setup test train train-augmented model model-optimized firmware data-all data-preprocess data-status data-open-images data-wider-face data-vgg-hollywood clean

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

train-augmented:
	PYTHONPATH=training $(PY) training/train.py --initialize $(ARTIFACTS)/optimized_model.pt --output training/runs/augmented-v1 --epochs 24 --schedule-epochs 24 --scheduler cosine --lr 0.0015 --patience 0 --save-every-epoch --log-interval 100

model:
	PYTHONPATH=training $(PY) training/export.py $(ARTIFACTS)/model.pt $(ARTIFACTS)/model.onnx
	PYTHONPATH=training $(PY) training/quantize.py $(ARTIFACTS)/model.onnx $(ARTIFACTS)/model.espdl --header include/tracker_model_config.h

model-optimized:
	PYTHONPATH=training $(PY) training/export.py $(ARTIFACTS)/optimized_model.pt $(ARTIFACTS)/optimized_model.onnx
	PYTHONPATH=training $(PY) training/quantize.py $(ARTIFACTS)/optimized_model.onnx $(ARTIFACTS)/optimized_model.espdl --header include/tracker_model_config.h
	cmake -E make_directory firmware/models
	cmake -E copy_if_different $(ARTIFACTS)/optimized_model.espdl firmware/models/hcds31-int8.espdl
	cmake -E copy_if_different $(ARTIFACTS)/optimized_model.quantization.json firmware/models/hcds31-int8.quantization.json

data-status:
	$(PY) training/datasets/pipeline.py status

data-preprocess:
	PYTHONPATH=training $(PY) training/datasets/preprocess.py all

data-all:
	PYTHONPATH=training $(PY) training/datasets/rebuild.py

data-open-images:
	$(PY) training/datasets/pipeline.py prepare open_images_human_head

data-wider-face:
	$(PY) training/datasets/pipeline.py prepare wider_face

data-vgg-hollywood:
	$(PY) training/datasets/pipeline.py prepare vgg_hollywood_heads

firmware:
	./tools/build_firmware.sh

clean:
	cmake -E rm -rf build .build
