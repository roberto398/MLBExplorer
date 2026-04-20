from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


OUTPUT_DIR_NAME = "biomechanics"
PITCHING_OUTPUT = "openbiomechanics_pitching_poi.parquet"
HITTING_OUTPUT = "openbiomechanics_hitting_poi.parquet"
METADATA_OUTPUT = "openbiomechanics_metadata.parquet"
MANIFEST_OUTPUT = "openbiomechanics_manifest.json"


def _candidate_roots(source_dir: Path, discipline: str) -> list[Path]:
    if source_dir.name == "poi":
        return [source_dir]
    if source_dir.name == "data" and source_dir.parent.name == discipline:
        return [source_dir]
    if source_dir.name == discipline:
        return [source_dir / "data", source_dir]
    return [source_dir / discipline / "data", source_dir / discipline]


def _find_poi_files(source_dir: Path, discipline: str) -> list[Path]:
    files: list[Path] = []
    for root in _candidate_roots(source_dir, discipline):
        if not root.exists():
            continue
        poi_dir = root / "poi"
        if poi_dir.exists():
            files.extend(sorted(poi_dir.rglob("*.csv")))
    if files:
        return sorted(dict.fromkeys(files))

    fallback: list[Path] = []
    for root in _candidate_roots(source_dir, discipline):
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            lower_parts = {part.lower() for part in path.parts}
            if "poi" in lower_parts and path.name.lower() != "metadata.csv":
                fallback.append(path)
    return sorted(dict.fromkeys(fallback))


def _find_metadata_files(source_dir: Path, discipline: str) -> list[Path]:
    files: list[Path] = []
    for root in _candidate_roots(source_dir, discipline):
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("metadata.csv")))
    return sorted(dict.fromkeys(files))


def _read_csv_files(paths: list[Path], dataset: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["dataset"] = dataset
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def ingest_openbiomechanics(source_dir: Path, artifacts_dir: Path) -> dict[str, object]:
    output_dir = artifacts_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    pitching_files = _find_poi_files(source_dir, "baseball_pitching")
    hitting_files = _find_poi_files(source_dir, "baseball_hitting")
    pitching_metadata_files = _find_metadata_files(source_dir, "baseball_pitching")
    hitting_metadata_files = _find_metadata_files(source_dir, "baseball_hitting")

    pitching = _read_csv_files(pitching_files, "pitching")
    hitting = _read_csv_files(hitting_files, "hitting")
    metadata = pd.concat(
        [
            _read_csv_files(pitching_metadata_files, "pitching"),
            _read_csv_files(hitting_metadata_files, "hitting"),
        ],
        ignore_index=True,
        sort=False,
    )

    _write_parquet(pitching, output_dir / PITCHING_OUTPUT)
    _write_parquet(hitting, output_dir / HITTING_OUTPUT)
    _write_parquet(metadata, output_dir / METADATA_OUTPUT)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dir": str(source_dir),
        "outputs": {
            PITCHING_OUTPUT: {"rows": int(len(pitching)), "columns": list(map(str, pitching.columns))},
            HITTING_OUTPUT: {"rows": int(len(hitting)), "columns": list(map(str, hitting.columns))},
            METADATA_OUTPUT: {"rows": int(len(metadata)), "columns": list(map(str, metadata.columns))},
        },
        "source_files": {
            "pitching_poi": [str(path) for path in pitching_files],
            "hitting_poi": [str(path) for path in hitting_files],
            "pitching_metadata": [str(path) for path in pitching_metadata_files],
            "hitting_metadata": [str(path) for path in hitting_metadata_files],
        },
        "license_note": "OpenBiomechanics is CC BY-NC-SA 4.0 with additional usage restrictions. Keep admin-only unless separately licensed.",
    }
    (output_dir / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenBiomechanics POI CSVs into compact Kasper artifacts.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Path to the OpenBiomechanics repository or data folder.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Kasper artifacts directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ingest_openbiomechanics(args.source_dir, args.artifacts_dir)
    print(f"Wrote biomechanics artifacts to {args.artifacts_dir / OUTPUT_DIR_NAME}")
    for filename, info in manifest["outputs"].items():
        print(f"- {filename}: {info['rows']} rows, {len(info['columns'])} columns")


if __name__ == "__main__":
    main()
