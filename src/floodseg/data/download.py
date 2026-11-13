"""Download and manifest verification for Sen1Floods11 dataset.

Downloads hand-labeled Sentinel-1, Sentinel-2, and QC label chips from the
official Cloud to Street Google Cloud Storage bucket via public HTTPS.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

GCS_BASE_URL = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLITS_URL = f"{GCS_BASE_URL}/splits/flood_handlabeled"
DATA_URL = f"{GCS_BASE_URL}/data/flood_events/HandLabeled"

SPLIT_FILES = {
    "train": f"{SPLITS_URL}/flood_train_data.csv",
    "val": f"{SPLITS_URL}/flood_valid_data.csv",
    "test": f"{SPLITS_URL}/flood_test_data.csv",
    "bolivia": f"{GCS_BASE_URL}/splits/flood_handlabeled/flood_bolivia_data.csv",
}


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Download a file with streaming and return its SHA-256 checksum."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        # File already exists, compute SHA-256
        return compute_sha256(dest_path)

    temp_path = dest_path.with_suffix(".tmp")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    sha256 = hashlib.sha256()
    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                sha256.update(chunk)

    temp_path.replace(dest_path)
    return sha256.hexdigest()


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def fetch_splits(processed_dir: Path) -> Dict[str, Path]:
    """Download official CSV split files."""
    splits_dir = processed_dir / "splits" / "official"
    splits_dir.mkdir(parents=True, exist_ok=True)
    downloaded_splits = {}

    for split_name, url in SPLIT_FILES.items():
        dest = splits_dir / f"flood_{split_name}_data.csv"
        try:
            download_file(url, dest)
            downloaded_splits[split_name] = dest
            logger.info("Fetched split %s -> %s", split_name, dest)
        except Exception as e:
            logger.warning("Could not download split %s from %s: %s", split_name, url, e)

    return downloaded_splits


def parse_split_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Parse Sen1Floods11 split CSV into structured chip entries."""
    chips = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            s1_filename = row[0].strip()
            label_filename = row[1].strip()

            # e.g., Ghana_103272_S1Hand.tif -> base is Ghana_103272
            base_id = s1_filename.replace("_S1Hand.tif", "").replace("_S1.tif", "")
            parts = base_id.split("_")
            country_event = parts[0]
            chip_number = parts[1] if len(parts) > 1 else ""

            s2_filename = f"{base_id}_S2Hand.tif"

            chips.append(
                {
                    "chip_id": base_id,
                    "country": country_event,
                    "chip_number": chip_number,
                    "s1_file": s1_filename,
                    "s2_file": s2_filename,
                    "label_file": label_filename,
                }
            )
    return chips


def create_loeo_splits(
    all_chips: List[Dict[str, str]],
    output_dir: Path,
    held_out_events: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """Create Leave-One-Event-Out (LOEO) splits to prevent spatial leakage."""
    if held_out_events is None:
        held_out_events = ["India", "Bolivia"]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_val_chips = [c for c in all_chips if c["country"] not in held_out_events]
    held_out_chips = [c for c in all_chips if c["country"] in held_out_events]

    # Split train_val into 80% train, 20% val
    n_train = int(0.8 * len(train_val_chips))
    train_chips = train_val_chips[:n_train]
    val_chips = train_val_chips[n_train:]

    splits = {
        "train": train_chips,
        "val": val_chips,
        "held_out_test": held_out_chips,
    }

    out_paths = {}
    for name, chip_list in splits.items():
        out_csv = output_dir / f"loeo_{name}_data.csv"
        with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for c in chip_list:
                writer.writerow([c["s1_file"], c["label_file"], c["s2_file"]])
        out_paths[name] = out_csv

    logger.info(
        "Created LOEO splits: Train=%d, Val=%d, Held-out Test=%d (Events: %s)",
        len(train_chips),
        len(val_chips),
        len(held_out_chips),
        held_out_events,
    )
    return out_paths


def _download_single_chip(
    chip: Dict[str, str],
    data_dir: Path,
    s1_dir: Path,
    s2_dir: Path,
    label_dir: Path,
) -> Tuple[str, Dict]:
    chip_id = chip["chip_id"]
    res = {}

    s1_url = f"{DATA_URL}/S1Hand/{chip['s1_file']}"
    s1_dest = s1_dir / chip["s1_file"]
    s1_hash = download_file(s1_url, s1_dest)
    res["s1"] = {"path": str(s1_dest.relative_to(data_dir)), "sha256": s1_hash}

    s2_url = f"{DATA_URL}/S2Hand/{chip['s2_file']}"
    s2_dest = s2_dir / chip["s2_file"]
    s2_hash = download_file(s2_url, s2_dest)
    res["s2"] = {"path": str(s2_dest.relative_to(data_dir)), "sha256": s2_hash}

    lbl_url = f"{DATA_URL}/LabelHand/{chip['label_file']}"
    lbl_dest = label_dir / chip["label_file"]
    lbl_hash = download_file(lbl_url, lbl_dest)
    res["label"] = {"path": str(lbl_dest.relative_to(data_dir)), "sha256": lbl_hash}

    return chip_id, res


def download_sen1floods11(
    data_dir: Path,
    max_chips: Optional[int] = None,
    verify_checksums: bool = True,
    max_workers: int = 8,
) -> Dict[str, str]:
    """Download Sen1Floods11 hand-labeled chips and generate checksum manifest."""
    raw_dir = data_dir / "raw" / "sen1floods11"
    s1_dir = raw_dir / "S1Hand"
    s2_dir = raw_dir / "S2Hand"
    label_dir = raw_dir / "LabelHand"

    for d in (s1_dir, s2_dir, label_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Fetch official splits
    processed_dir = data_dir / "processed"
    split_paths = fetch_splits(processed_dir)

    # 2. Gather all unique chips
    all_chips_dict = {}
    for s_name, path in split_paths.items():
        for chip in parse_split_csv(path):
            all_chips_dict[chip["chip_id"]] = chip

    chips_to_download = list(all_chips_dict.values())
    if max_chips is not None:
        chips_to_download = chips_to_download[:max_chips]

    manifest = {}
    manifest_path = processed_dir / "manifest.json"

    logger.info("Downloading %d chips to %s using %d worker threads", len(chips_to_download), raw_dir, max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_single_chip, chip, data_dir, s1_dir, s2_dir, label_dir): chip
            for chip in chips_to_download
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading chips"):
            chip_id, chip_meta = future.result()
            manifest[chip_id] = chip_meta

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote checksum manifest to %s", manifest_path)

    # 3. Create LOEO splits
    create_loeo_splits(list(all_chips_dict.values()), processed_dir / "splits" / "loeo")

    return {"status": "success", "manifest": str(manifest_path), "total_chips": len(manifest)}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Download Sen1Floods11 chips")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--max-chips", type=int, default=None, help="Limit number of chips for testing"
    )
    args = parser.parse_args()

    download_sen1floods11(args.data_dir, max_chips=args.max_chips)
