#!/usr/bin/env python3
"""Convert the fixed ONNX model to ESP32-P4 INT8."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from esp_ppq.api import espdl_quantize_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    samples = torch.from_numpy(np.load(args.calibration).astype(np.float32))
    loader = [sample[None] for sample in samples]
    graph = espdl_quantize_onnx(
        onnx_import_file=str(args.model),
        espdl_export_file=str(args.output),
        calib_dataloader=loader,
        calib_steps=len(loader),
        input_shape=[1, 3, 160, 288],
        inputs=[loader[0]],
        target="esp32p4",
        num_of_bits=8,
        collate_fn=lambda value: value.float(),
        device="cpu",
        error_report=False,
        export_config=False,
        export_test_values=False,
        verbose=0,
    )
    scale = float(graph.outputs["head"].source_op_config.scale)
    exponent = round(math.log2(scale))
    print(f"{args.output} output_exponent={exponent} offset_q={4 - exponent}")


if __name__ == "__main__":
    main()
