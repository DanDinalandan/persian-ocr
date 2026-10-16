# Synthetic Persian isolated-letter benchmark

This is the generated dataset used for the model results in the repository README and [`../RESULTS.md`](../../RESULTS.md). It is a synthetic typography benchmark, not a handwriting or camera dataset. The images were rendered from licensed fonts; connected-text source crops were not used.

## Contents

| Split or group | Contents | Count |
| --- | --- | ---: |
| `images/train` | Isolated letters, six font families | 1,536 (48 per class) |
| `images/val` | Isolated letters, Amiri | 256 (8 per class) |
| `images/test` | Isolated letters, Scheherazade New | 256 (8 per class) |
| `negatives/val` | Blank and two-letter images | 128 |
| `negatives/test` | Blank and two-letter images | 128 |
| `stress` | Blur and shadow variants of test letters | 512 |
| `labels/{train,val,test}` | YOLO boxes for positive images | 2,048 |

All positive splits cover the same 32 Persian letter classes. Font families are separated across splits. Test scores in the published results use `images/test` and `negatives/test`; the stress views are scored separately.

## Manifests and integrity

`manifests/` contains JSON Lines files for each positive split, validation and test negatives, and the test stress views. Each record includes the image path relative to the repository root, SHA-256, class, font, and box where applicable. The paths are retained as `data/prepared/...` so they resolve when this repository is cloned. Labels are YOLO normalized class/box rows. Generated Ultralytics cache files are not included.

## Font licenses and provenance

`font_licenses/` contains the SIL Open Font License notices for the eight source font families. `font_source_provenance.json` records their sources and hashes. Review the applicable font notice when redistributing or modifying this dataset. The original Persian OCR source archives were audited but are not included here, and their connected-text images are not part of this benchmark.

## Limits

This small benchmark has one font family in each held-out split and no natural handwriting, camera images, connected words, or pages. Results should not be interpreted as real-world OCR performance. The full preparation/training pipeline and raw source archives remain in the development workspace.
