# Download

The default run produces about 51,000 compact training images:

| Dataset | Images | What it contributes |
|---|---:|---|
| [WIDER FACE](https://shuoyang1213.me/WIDERFACE/) train | 12,880 | face boxes, scale, pose, crowds, and varied scenes |
| [SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) train | 3,405 | full-head boxes in classrooms and internet scenes |
| [CrowdHuman](https://www.crowdhuman.org/) train | 15,000 | head, visible-body, and full-body boxes in crowds |
| [COCO](https://cocodataset.org/dataset/detection-2017.htm) train | 10,000 | person boxes in ordinary indoor and outdoor scenes |
| COCO train | 10,000 | images without a COCO person annotation |

Face, head, visible-person, and full-person boxes remain different label kinds.
COCO negatives are marked negative for `person`; they are not falsely claimed
to be verified head negatives.

## Run

```bash
python3 -m pip install -r python/requirements.txt
python3 python/download.py
```

The terminal shows ordinary progress bars for archive downloads, extraction,
and image conversion. Output is deliberately plain:

```text
data/<dataset>/images/*.webp
data/<dataset>/labels.jsonl
```

Only one source is held in raw form at a time. It is stored in the operating
system's temporary directory and removed when that source finishes, fails, or
is interrupted normally. Compact images preserve aspect ratio, are never
upscaled, and fit within the dimensions in `config/download.toml`.

Useful commands:

```bash
# Download just one source.
python3 python/download.py --only scut_head

# Make a tiny real-data run in another folder.
python3 python/download.py --only coco --limit 20 --data-dir /tmp/tracker-data

# Deliberately replace a finished source.
python3 python/download.py --only coco --force
```

If `labels.jsonl` already exists, that source is left alone. Use `--force` only
when you intentionally want to replace it.
