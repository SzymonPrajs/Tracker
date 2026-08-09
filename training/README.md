# Training

The baseline uses one path: PyTorch on the Mac, ONNX, then ESP-PPQ INT8 for ESP32-P4.

```sh
./training/setup.sh
make train
make model
```

`make train` learns from small generated multi-head images and writes `model.pt` plus
`calibration.npy`. `make model` writes the fixed `160x288` Q4/Q7 `model.onnx` and
`model.espdl` artifacts. Replace the generated scenes and calibration tensor with real
data later; the model and deployment path do not change.
