from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from data_pipeline.errors import PipelineError


@dataclass(frozen=True)
class TransferResult:
    byte_count: int
    sha256: str
    final_url: str


def transfer_file(
    url: str,
    destination: Path,
    *,
    byte_limit: int,
    timeout_seconds: float,
    expected_sha256: str | None = None,
    purpose: str,
) -> TransferResult:
    """Transfer one metadata or pixel file with a hard streamed byte ceiling."""

    if byte_limit <= 0:
        raise PipelineError(f"{purpose}_limit", f"{purpose} byte budget is exhausted")
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in {"https", "http", "file"}:
        raise PipelineError(f"{purpose}_scheme", f"unsupported {purpose} URL scheme: {scheme!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    byte_count = 0
    final_url = url
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "TrackerDataPipeline/0.1"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            final_url = response.geturl()
            if urllib.parse.urlparse(final_url).scheme not in {"https", "http", "file"}:
                raise PipelineError(
                    "redirect_scheme", f"{purpose} redirected to an unsupported scheme"
                )
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > byte_limit:
                raise PipelineError(
                    f"{purpose}_limit",
                    f"declared {purpose} size exceeds the remaining byte budget",
                    declared_bytes=int(declared),
                    remaining_bytes=byte_limit,
                )
            with partial.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    byte_count += len(chunk)
                    if byte_count > byte_limit:
                        raise PipelineError(
                            f"{purpose}_limit",
                            f"{purpose} transfer exceeded the remaining byte budget",
                            remaining_bytes=byte_limit,
                        )
                    output.write(chunk)
                    digest.update(chunk)
        checksum = digest.hexdigest()
        if expected_sha256 is not None and checksum != expected_sha256:
            raise PipelineError(
                "source_checksum",
                f"downloaded {purpose} checksum differs from the manifest",
                expected=expected_sha256,
                actual=checksum,
            )
        partial.replace(destination)
        return TransferResult(byte_count=byte_count, sha256=checksum, final_url=final_url)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def download_metadata(
    url: str, destination: Path, *, byte_limit: int, timeout_seconds: float
) -> TransferResult:
    return transfer_file(
        url,
        destination,
        byte_limit=byte_limit,
        timeout_seconds=timeout_seconds,
        purpose="metadata",
    )


def download_image(
    url: str,
    destination: Path,
    *,
    byte_limit: int,
    timeout_seconds: float,
    expected_sha256: str | None,
) -> TransferResult:
    return transfer_file(
        url,
        destination,
        byte_limit=byte_limit,
        timeout_seconds=timeout_seconds,
        expected_sha256=expected_sha256,
        purpose="image",
    )
