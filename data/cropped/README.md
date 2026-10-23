# Derived bounding-box crop dataset

This 13 MB dataset is derived from the published [`data/prepared/`](../prepared/README.md) benchmark. It contains grayscale 96×96 crops for train, validation, held-out test, and synthetic stress splits, plus JSONL manifests with class IDs, source image IDs, split/font metadata, checksums, and crop policy. The split boundaries and source IDs are preserved.

Crops use the ground-truth box expanded by 6%, median-border background normalization, and a white 96×96 canvas. Separated Persian dots are retained. Training uses train crops only; validation is used for checkpoint/threshold selection; held-out test and stress results are reported separately in [`CROP_PIPELINE_RESULTS.md`](../../CROP_PIPELINE_RESULTS.md). The crop dataset is a derived convenience; the source images and labels remain in `data/prepared/`.
