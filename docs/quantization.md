# ESP32-P4 quantization

## Exact target

Quantization is tied to a pinned ESP32-P4 stack. Current ESP-DL guidance uses:

- fixed batch one and static spatial dimensions;
- symmetric power-of-two quantization;
- per-channel quantization for Conv and Gemm on P4;
- per-tensor quantization for other operators;
- round-half-even behavior;
- target-specific `.espdl` artifacts that must not be mixed across chips.

Current P4 per-channel support requires ESP-PPQ 1.2.10 or later and ESP-DL 3.3.1
or later. Recheck and pin exact compatible versions when this pipeline starts.

For exponent `e`, the intended scalar contract is approximately:

```text
q = clip(round_half_even(x / 2^e), -128, 127)
x_hat = q * 2^e
```

The pinned ESP32-P4 target plus ESP-DL/ESP-PPQ versions define quantization and
rounding rules. Emitted `.espdl`, `.info`, and `.json` metadata define the
actual tensor exponents, channel layout, padding, and exported graph. Do not
carry constants from the archived model into the rebuild.

## Build the pipeline

1. Check the current ESP-DL operator table before final float training.
2. Export static batch-one ONNX with raw output branches.
3. Compare PyTorch FP32 and ONNX Runtime on a fixed adversarial pack.
4. Freeze exact camera-byte preprocessing from the completed hardware input
   characterization and freeze output semantics.
5. Build a deterministic representative calibration manifest from training
   data only and record its deployment-stratum proportions.
6. Run default all-INT8 P4 PTQ.
7. Evaluate the returned ESP-PPQ graph with task metrics.
8. Export `.espdl`, `.info`, `.json`, version pins, hashes, and explicit test
   inputs/outputs.

The calibration set mirrors the expected deployment distribution across flat
and warped optics, light, noise, scale, crowds, and certified negatives. Rare
stress cases appear only in justified deployment proportions. A separate
adversarial quantization evaluation pack deliberately balances dark, noisy,
warped, crowded, distant, edge, and negative cases. Neither contains final-test
images.

## Compare quantization-oriented runs

Keep these as separate candidates:

### PTQ baseline

Use the default supported P4 INT8 rules with a representative calibration set.
This is the mandatory reference and may already be sufficient.

### AutoQuant

Use a real task `evaluate_fn` on quantization-validation data, never calibration
examples or the final test; graph SNR alone cannot rank detection behavior.
Require an all-INT8 search configuration when that is the candidate contract.

### TQT

Tune power-of-two thresholds/weights from the PTQ result using the official
ESP-DL/ESP-PPQ path. Treat published iteration counts and learning rates as
starting points, not fixed truths.

### QAT

Warm-start from a float checkpoint trained from scratch, then fine-tune using
fake quantization that matches P4 power-of-two, per-channel/per-tensor, clipping,
and rounding rules. Generic PyTorch fake quantization is not assumed equivalent.
QAT is a candidate, not the only baseline and not a guaranteed improvement.

The candidate set is exactly:

1. fixed default all-INT8 PTQ;
2. AutoQuant-selected all-INT8 PTQ;
3. PTQ initialized with TQT threshold/weight refinement;
4. QAT warm-started from the frozen float checkpoint;
5. explicitly ablated combinations only when a standalone candidate fails.

Inspect exported dispatch/configuration and reject unintended INT16 or float
fallbacks; requesting `num_of_bits=8` alone does not prove all-INT8 execution.
Repeat the selected recipe from its declared beginning. For QAT, reproduce the
float-from-scratch parent and then repeat the frozen QAT fine-tune; fake-quant
training from random initialization is a different named experiment.

## Quantized acceptance chain

Compare exactly the same tensors through:

```text
PyTorch FP32 raw outputs
→ ONNX Runtime FP32 raw outputs
→ ESP-PPQ quantized-graph raw outputs
→ exported .espdl/.info metadata and embedded-golden inspection
→ ESP32-P4 Model::test()
→ custom multi-input board harness
→ real-frame board harness
```

Require multiple explicit gold inputs: clean, dark/noisy, warped, distant,
crowded, edge/truncated, and negative. Because one export may embed only one
explicit input/output pair, broad coverage comes from the custom multi-input
harness or multiple artifacts; `Model::test()` alone is a smoke/alignment check.

Selection uses threshold-free AP/precision-recall, per-stratum error, false
positives, saturation/outlier reports, model and activation memory, and later
board latency. Each candidate's deployment threshold is selected on validation
under the same certified-negative false-positive policy and then frozen. Also
report a common-threshold diagnostic to expose calibration drift. Never tune a
threshold on final test.

No quantized model is accepted until:

- preprocessing is byte-identical to deployment;
- output layout and numeric scales are derived from emitted metadata;
- float-to-INT8 accuracy drop stays within a declared budget for every critical
  stratum and negatives;
- host simulation and board golden outputs match within declared integer
  tolerances;
- the quantization recipe, model, preprocessing, and candidate-specific
  threshold are frozen before the final test is opened;
- board memory and latency pass final physical validation.
