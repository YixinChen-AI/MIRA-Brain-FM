#!/usr/bin/env python3
"""Download and verify the checkpoint described by the release manifest."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from release_utils import (
    load_checkpoint_manifest,
    resolve_path,
    safe_filename,
    sha256_file,
)


PREFIX = "release contract violation:"


def fail(message: str) -> int:
    print(f"{PREFIX} {message}", file=sys.stderr)
    return 1


class HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject any redirect hop that leaves HTTPS."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise ValueError("checkpoint redirect target must use HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_https(url: str):
    opener = urllib.request.build_opener(HTTPSOnlyRedirectHandler())
    return opener.open(url)


def download_checkpoint(
    record: dict[str, Any],
    output_dir: Path,
    *,
    opener: Callable[[str], Any] = open_https,
) -> Path:
    filename = safe_filename(record.get("filename"))
    url = str(record.get("url", ""))
    if urlparse(url).scheme != "https":
        raise ValueError("checkpoint URL must use HTTPS")

    output_dir = resolve_path(output_dir)
    output = output_dir / filename
    if output.is_file():
        if output.stat().st_size == record["bytes"] and sha256_file(output) == record["sha256"]:
            return output
        raise ValueError(f"existing checkpoint does not match manifest: {output}")

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    try:
        with opener(url) as response:
            final_url = response.geturl()
            if urlparse(final_url).scheme != "https":
                raise ValueError("checkpoint redirect final URL must use HTTPS")
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        if temporary.stat().st_size != record["bytes"]:
            raise ValueError("downloaded checkpoint byte count does not match manifest")
        if sha256_file(temporary) != record["sha256"]:
            raise ValueError("downloaded checkpoint sha256 does not match manifest")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("checkpoints/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--release", action="store_true", help="enforce publication state")
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    output_dir = resolve_path(args.output_dir)
    try:
        manifest = load_checkpoint_manifest(manifest_path)
        record = manifest["checkpoints"][0]
    except (OSError, KeyError, ValueError) as exc:
        return fail(f"invalid manifest: {exc}")

    state = record.get("state")
    if args.release and state != "published":
        return fail(f"checkpoint state {state!r} is not published")
    if state == "training_pending":
        return fail("checkpoint state 'training_pending'; public retraining is not complete")
    if state not in {"publication_asset_pending", "published"}:
        return fail(f"unsupported checkpoint state {state!r}")

    try:
        output = download_checkpoint(record, output_dir)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return fail(f"checkpoint download failed: {exc}")

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
