# Data pipeline

Everything in this folder belongs to one readable Python project. There is one
importable library, `data_pipeline`, rather than a workspace of repeatedly
prefixed packages.

```text
data_pipeline/
  config.py       strict resolved configuration
  records.py      canonical image, annotation, and coverage records
  acquire.py      one-source-at-a-time orchestration and cleanup
  images.py       orientation, resize, encoding, and label transforms
  transfer.py     bounded metadata and selected-pixel transfer
  sources/        external dataset adapters only
  packets/        durable packet construction, validation, and reading
tests/             unit, adversarial, and end-to-end tests
```

The public commands all call the same library APIs used by tests:

```bash
uv sync --project Python
uv run --project Python python -m data_pipeline check
uv run --project Python python -m data_pipeline acquire \
  --config configs/data/open_images_smoke.json --dry-run
uv run --project Python python -m data_pipeline acquire \
  --config configs/data/open_images_smoke.json
uv run --project Python python -m data_pipeline validate DATA_PACKET
uv run --project Python python -m data_pipeline inspect DATA_PACKET
```

Acquisition has no keep-raw option. Metadata and selected source pixels live in
one guarded staging directory, and that directory is removed on success or
failure. Only a validated compact packet, its manifest, and a run report remain.
