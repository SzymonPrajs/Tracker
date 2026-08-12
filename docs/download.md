# Download

The focused corpus contains face/head positives and strict face/head negatives:

| Dataset | Images | What it contributes |
|---|---:|---|
| [WIDER FACE](https://shuoyang1213.me/WIDERFACE/) | 12,880 train + 3,226 validation | faces in varied scenes |
| [SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) | 3,405 train + 1,000 test | heads in classrooms and internet scenes |
| [CrowdHuman](https://www.crowdhuman.org/) | 15,000 train + 4,370 validation | heads in busy scenes |
| [Open Images](https://storage.googleapis.com/openimages/web/download_v7.html) | 5,000 train + 1,000 validation | images verified negative for both Human face and Human head |

The model uses only face/head centers. Body boxes are neither downloaded for
supervision nor used by preprocessing. Open Images negatives are accepted only
when its human image-level labels explicitly set both `/m/0dzct` (Human face)
and `/m/04hgtk` (Human head) to zero; a missing label is not treated as a
negative.

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
python3 python/download.py --only open_images --limit 20 --data-dir /tmp/tracker-data

# Deliberately replace a finished source.
python3 python/download.py --only open_images --force

# Keep existing positives and append an official held-out split.
python3 python/download.py --only wider_face --held-out
```

If `labels.jsonl` already exists, that source is left alone. Use `--force` only
when you intentionally want to replace it. `--held-out` upgrades an existing
WIDER FACE, CrowdHuman, or SCUT-HEAD cache without redownloading its training
images.
