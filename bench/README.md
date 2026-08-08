# Benchmark assets

These tools deliberately separate three evidence classes:

- `host_synthetic`: portable algorithm correctness and host timing only;
- `device_microbenchmark`: a kernel measured on the ESP32-P4;
- `device_pipeline`: the real camera-to-centroid pipeline on the delivered board.

Only `device_pipeline` evidence can support an ESP32-P4 frame-rate claim.

## Host smoke test

```sh
cmake -S bench -B build/bench
cmake --build build/bench
./build/bench/tracker_host_benchmark > build/host-benchmark.ndjson
python3 tools/validate_benchmark.py build/host-benchmark.ndjson
```

The host runner benchmarks only fixed-point heatmap centroid decoding. It does
not emulate PIE, PPA, CSI, PSRAM contention, ESP-DL, or FreeRTOS scheduling.

## Fixtures

Regenerate the deterministic binary fixtures and manifest with:

```sh
python3 tools/generate_fixtures.py
python3 tools/generate_fixtures.py --check
```

## Device collection

Capture benchmark records from standard input:

```sh
python3 tools/collect_benchmark.py --input - --output run.ndjson
```

Or, when the optional `pyserial` package is installed:

```sh
python3 tools/collect_benchmark.py --port /dev/cu.usbmodem0001 --output run.ndjson
```

Summarize raw frame records without turning host timings into device FPS:

```sh
python3 tools/summarize_benchmark.py --output summary.ndjson run.ndjson
python3 tools/validate_benchmark.py --require-gates run.ndjson summary.ndjson
```
