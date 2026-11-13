"""CLI interface for FloodSeg."""

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="floodseg",
        description="FloodSeg: Flood water segmentation from multimodal satellite imagery",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser(
        "download", help="Download Sen1Floods11 chips and splits"
    )
    download_parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Data directory"
    )
    download_parser.add_argument(
        "--max-chips", type=int, default=None, help="Max chips to download (default: all)"
    )

    # Stats command
    stats_parser = subparsers.add_parser(
        "compute-stats", help="Compute per-band normalization stats on train split"
    )
    stats_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    stats_parser.add_argument(
        "--split", type=str, default="all", choices=["official", "loeo", "all"]
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "download":
        from floodseg.data.download import download_sen1floods11

        download_sen1floods11(args.data_dir, max_chips=args.max_chips)
    elif args.command == "compute-stats":
        from floodseg.data.splits import SPLIT_TYPES
        from floodseg.data.stats import compute_split_stats

        for split_type in SPLIT_TYPES if args.split == "all" else (args.split,):
            compute_split_stats(args.data_dir, split_type)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
