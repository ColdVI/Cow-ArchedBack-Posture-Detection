#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.sources import normalize_and_require_approved_sources


VALID_KINDS = {"video", "image", "image_dir", "youtube", "direct"}


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError(f"Invalid source_id: {value!r}")
    return cleaned


def download_direct(url: str, destination: Path, overwrite: bool) -> Path:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for direct downloads") from exc
    if destination.exists() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(destination)
    return destination


def download_youtube(url: str, output_dir: Path, source_id: str, overwrite: bool) -> Path:
    executable = shutil.which("yt-dlp")
    if executable is None:
        raise RuntimeError("yt-dlp is not installed; install requirements.txt")
    existing = sorted(output_dir.glob(f"{source_id}.*"))
    if existing and not overwrite:
        return existing[0]
    template = str(output_dir / f"{source_id}.%(ext)s")
    command = [
        executable,
        "--no-playlist",
        "--restrict-filenames",
        "--merge-output-format",
        "mp4",
        "-o",
        template,
        url,
    ]
    if overwrite:
        command.insert(1, "--force-overwrites")
    subprocess.run(command, check=True)
    candidates = sorted(output_dir.glob(f"{source_id}.*"))
    if not candidates:
        raise RuntimeError(f"yt-dlp produced no file for {source_id}")
    return candidates[0]


def resolve_local_path(raw: str, sources_path: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = sources_path.resolve().parent / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve explicitly approved local/remote data sources.")
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-sources", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.sources, keep_default_na=False, comment="#")
    # Validate every row before creating directories or touching the network.
    frame = normalize_and_require_approved_sources(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_rows = []
    for _, row in frame.iterrows():
        source_id = safe_id(str(row["source_id"]))
        kind = str(row["kind"]).strip().lower()
        if kind not in VALID_KINDS:
            raise ValueError(f"Unsupported kind {kind!r} for {source_id}")
        if kind in {"video", "image", "image_dir"}:
            resolved = resolve_local_path(str(row["local_path"]), args.sources)
            if not resolved.exists():
                raise FileNotFoundError(f"Local source not found: {resolved}")
        elif kind == "youtube":
            url = str(row["url"]).strip()
            if not url:
                raise ValueError(f"URL is required for {source_id}")
            resolved = download_youtube(url, args.output_dir, source_id, args.overwrite)
        else:
            url = str(row["url"]).strip()
            if not url:
                raise ValueError(f"URL is required for {source_id}")
            suffix = Path(urlparse(url).path).suffix or ".bin"
            resolved = download_direct(url, args.output_dir / f"{source_id}{suffix}", args.overwrite)

        output = row.to_dict()
        output["source_id"] = source_id
        output["resolved_path"] = str(resolved.resolve())
        resolved_rows.append(output)
        print(f"resolved {source_id}: {resolved}")

    result = pd.DataFrame(resolved_rows)
    args.output_sources.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_sources, index=False)
    print(f"wrote {len(result)} sources to {args.output_sources}")


if __name__ == "__main__":
    main()
