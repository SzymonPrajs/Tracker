from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download(url: str, destination: Path, label: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        requests.get(url, stream=True, timeout=60) as response,
        destination.open("wb") as output,
    ):
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=label) as progress:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
                progress.update(len(chunk))
    return destination


def google_drive(file_id: str, destination: Path, label: str) -> Path:
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "Run: python3 -m pip install -r python/requirements.txt"
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"{label}: downloading")
    result = gdown.download(id=file_id, output=str(destination), quiet=False)
    if not result:
        raise RuntimeError(f"Google Drive did not download {label}")
    return destination


def unzip(archive: Path, destination: Path, label: str) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        for member in tqdm(members, desc=f"{label}: extracting", unit="file"):
            source.extract(member, destination)
    return destination


def start_output(output: Path) -> None:
    # labels.jsonl is written last, so its presence means the dataset completed.
    if output.exists() and not (output / "labels.jsonl").exists():
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True, exist_ok=True)
