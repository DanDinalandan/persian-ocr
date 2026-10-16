# Measured results

These are local measurements from the completed ROCm runs. Published paper accuracy is not used.

Hardware: AMD Radeon RX 7800 XT; PyTorch 2.11.0+rocm7.2; HIP 7.2.26015; Python 3.12.3.

YOLO completed 70 epochs; CNN completed 86 epochs. Limits: 100 epochs, patience 20, seed 42, batch 16. Best checkpoints were selected on validation data. No hardware batch adjustment was needed.

The clean test split contains 256 images (eight per class), all from the held-out Scheherazade New family. The validation font was Amiri. Rejected valid inputs count as incorrect. Both models use exactly the same test identities.

| Model | Correct / valid | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Valid rejection rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO11n | 256 / 256 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |
| Reduced CNN | 207 / 256 | 80.86% | 80.65% | 80.86% | 79.70% | 79.70% | 12.89% |

Negative inputs are scored separately. Each category has 64 test images; multiple-letter samples contain two independently isolated glyphs.

| Model | Negative category | Rejection accuracy | False acceptance |
| --- | --- | --- | --- |
| YOLO11n | blank | 100.00% | 0 / 64 (0.00%) |
| YOLO11n | multiple | 100.00% | 0 / 64 (0.00%) |
| Reduced CNN | blank | 100.00% | 0 / 64 (0.00%) |
| Reduced CNN | multiple | 65.62% | 22 / 64 (34.38%) |

Deterministic synthetic stress views are separate from the clean test score: 256 blurred images and 256 shadow-gradient images, using the same held-out parents. These are synthetic lighting/blur tests, not camera measurements.

| Model | Blur exact accuracy | Shadow exact accuracy |
| --- | --- | --- |
| YOLO11n | 100.00% | 100.00% |
| Reduced CNN | 80.47% | 15.23% |

Subgroup results on clean test images:

| Model | Subgroup | Images | Exact accuracy |
| --- | --- | --- | --- |
| YOLO11n | dot_similar_beh_peh_teh_seh | 32 | 100.00% |
| YOLO11n | off_center | 153 | 100.00% |
| Reduced CNN | dot_similar_beh_peh_teh_seh | 32 | 100.00% |
| Reduced CNN | off_center | 153 | 78.43% |

YOLO validation-selected policy: candidate floor 0.03, acceptance threshold 0.8, class-agnostic NMS IoU 0.4. CNN acceptance threshold: 0.8. Thresholds were frozen before test evaluation.

Latency is measured on sequential images after four warm-up images, with GPU synchronization, after both training jobs finished. Timing count includes clean, negative and synthetic-stress images. The pipelines differ, so their timings are not interchangeable.

| Timed operations | Images | Mean ms | Median ms | p95 ms |
| --- | --- | --- | --- | --- |
| YOLO forward | 896 | 5.70 | 5.69 | 5.81 |
| YOLO decoded-image pipeline | 896 | 7.13 | 7.15 | 7.36 |
| YOLO file decoding | 896 | 0.92 | 0.89 | 1.21 |
| CNN preprocessing + forward | 896 | 0.89 | 0.85 | 1.17 |

Precision, recall, and F1 are macro averaged across the 32 letter classes. Rejected valid images count as false negatives; rejected inputs are not counted as predictions of any letter. Weighted F1 uses class support.

The full reports contain per-class recalls/supports and confusion matrices with a rejection column: [YOLO](results/test_yolo_rocm.json), [CNN](results/test_cnn_rocm.json). Readable matrices: [YOLO](results/confusion_yolo.png), [CNN](results/confusion_cnn.png). CSV files preserve exact counts.

Limitations: all samples are rendered fonts, with only one family in each held-out split. There are no real camera captures or natural handwriting. Connected-text source crops are quarantined. The CNN uses an image-derived dark-foreground crop and cannot explicitly count letters; dark/shadowed backgrounds can interfere with its crop. Negative inputs cover simple blanks and two isolated glyphs, not the full range of non-letter objects or connected text. Undetected extra letters can still be falsely accepted. Dot-similar and off-center subsets overlap; their scores are not additive.

The reports and confusion matrices linked above were generated from the local ROCm evaluation run. The full training dataset and local integration/audit logs are not part of this inference package.

Published inference weights: `artifacts/yolo11n.pt` and `artifacts/cnn-best.pt`. Setup and inference instructions are in [README](README.md). The app never trains on uploads.
