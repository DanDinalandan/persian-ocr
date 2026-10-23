# Release notes

## GitHub main update: bounding-box crop CNN

- Added a 32-class crop CNN checkpoint trained for normalized 96×96 YOLO bounding-box crops.
- Added crop cleanup and two-stage inference code, plus the validation-selected classifier threshold and detector policy.
- Added the split-preserving derived crop dataset, training metadata, preview, and end-to-end evaluation reports.
- Updated the Colab notebook to run YOLO localization followed by crop CNN classification.
- Held-out synthetic test: crop top-1 256/256; confidence-gated cascade 248/256 (96.88%); blank and two-letter rejection 128/128. Standalone YOLO remains 256/256. See `CROP_PIPELINE_RESULTS.md` for full metrics and limits.

The GitHub repository contains this crop checkpoint. The existing release ZIP and Google Drive assets were not modified by this repository update.
