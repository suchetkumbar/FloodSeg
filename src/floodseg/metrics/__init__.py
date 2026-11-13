"""Metrics, confusion matrix accumulators, and bootstrap estimation."""

from floodseg.metrics.evaluator import (
    SegmentationMetrics,
    bootstrap_ci,
    compute_sample_iou,
    confusion_counts,
    iou_from_counts,
    metrics_from_counts,
    summarize_with_ci,
)

__all__ = [
    "SegmentationMetrics",
    "bootstrap_ci",
    "compute_sample_iou",
    "confusion_counts",
    "iou_from_counts",
    "metrics_from_counts",
    "summarize_with_ci",
]
