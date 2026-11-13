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

    # Predict command
    pred_parser = subparsers.add_parser(
        "predict", help="Predict flood water for a scene (sliding window) -> GeoTIFFs"
    )
    pred_parser.add_argument(
        "--checkpoint", type=Path, nargs="+", required=True,
        help="One or more best_model.pt files of the same modality (averaged as an ensemble)",
    )
    pred_parser.add_argument("--s1", type=Path, default=None, help="2-band VV/VH GeoTIFF in dB")
    pred_parser.add_argument("--s2", type=Path, default=None, help="13-band S2 L1C TOA x 1e4 GeoTIFF")
    pred_parser.add_argument("--out", type=Path, required=True, help="Output prefix, e.g. outputs/scene")
    pred_parser.add_argument("--threshold", type=float, default=0.5)
    pred_parser.add_argument("--tile", type=int, default=512)
    pred_parser.add_argument("--overlap", type=int, default=128)

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
    elif args.command == "predict":
        from floodseg.predict import predict_scene

        predict_scene(
            args.checkpoint, args.s1, args.s2, args.out,
            threshold=args.threshold, tile=args.tile, overlap=args.overlap,
        )
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
