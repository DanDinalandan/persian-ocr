# Bounding-box crop CNN results

## What changed

The new pipeline keeps YOLO11n as the letter localizer and passes its predicted box through a crop-cleaning step before classification by `CropLetterCNN`. The crop expands the box by 6%, estimates the background from the crop border, flattens near-background texture, preserves detached dots, then centers the grayscale letter on a 96×96 white canvas. The cleaning method does not delete small connected components.

The visual review places the cleaned model input on the left and the source image with its box on the right: [crop preview](crop_preview.png). The generated split-preserving dataset and hashes are under `data/cropped/`; its summary is [crop_dataset.json](crop_dataset.json).

## Training and selection

Training ran on the AMD Radeon RX 7800 XT with ROCm 7.2, using 1,536 training crops and 256 validation crops. Three learning-rate/weight-decay/dropout combinations were trained for up to 80 epochs. Every trial reached 100% validation crop accuracy. The selected trial (`lr=0.001`, `weight_decay=0.0001`, `dropout=0.25`) had the lowest validation loss, 0.0003685, at epoch 76.

The checkpoint is `artifacts/crop_cnn/best.pt`. Trial settings and histories are in `artifacts/crop_cnn/`. YOLO candidate/acceptance settings and the crop-classifier threshold were frozen using validation positives and negatives; the selected crop threshold is 0.99.

## Held-out results

The test split is 256 synthetic letters (eight per class) from the held-out Scheherazade New font. The production-style scores below use YOLO-predicted boxes. The raw top-1 row shows classifier labels before the crop-confidence rejection threshold; final accuracy counts rejected letters as errors.

| Measurement | Correct / total | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing full-image CNN, raw top-1 | 224 / 256 | 87.50% | 87.26% | 87.50% | 84.21% | 84.21% |
| Existing standalone YOLO11n, thresholded | 256 / 256 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Crop CNN top-1 on YOLO boxes, before confidence rejection | 256 / 256 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Full two-stage pipeline, thresholded | 248 / 256 | 96.88% | 96.88% | 96.88% | 96.88% | 96.88% |
| Oracle crop CNN, ground-truth boxes, thresholded | 248 / 256 | 96.88% | 96.88% | 96.88% | 96.88% | 96.88% |

The classifier's top-1 prediction was correct on all 256 valid test images, but the frozen 0.99 confidence threshold rejected eight correctly labeled `vav` examples. Thus accepted-only accuracy is 248/248 (100%), while the deployment metric is 248/256 (96.88%). No threshold was adjusted after viewing test results. The threshold grid tied across its tested values on validation, and the calibration tie-break selected the most conservative threshold.

On the same benchmark, the full-image CNN's saved raw predictions score 224/256 top-1 (87.50%) and its calibrated end-to-end exact accuracy was 207/256 (80.86%). Crop training raises raw top-1 to 256/256 and confidence-gated exact accuracy to 248/256 (96.88%). Standalone YOLO remains at 256/256 with no valid rejections, so this cascade improves on the full-image CNN but does not exceed the YOLO baseline.

Negative test inputs were rejected without false acceptance:

| Category | Rejected | False acceptance |
| --- | ---: | ---: |
| Blank | 64 / 64 | 0 / 64 |
| Two letters | 64 / 64 | 0 / 64 |
| All negatives | 128 / 128 | 0 / 128 |

Synthetic stress views, scored separately with the same frozen policy, reached 248/256 (96.88%) on blur and 247/256 (96.48%) on shadow gradients. These are generated image transformations, not camera tests.

Machine-readable per-class recalls and confusion matrices are in [crop_pipeline_test.json](crop_pipeline_test.json). Validation calibration is in [crop_pipeline_validation.json](crop_pipeline_validation.json) and [crop_pipeline_validation_grid.json](crop_pipeline_validation_grid.json).

## Limits

This remains a small synthetic benchmark with one held-out font and eight images per class. It has no handwriting, real camera photos, connected words, or pages. Ground-truth-box accuracy is an oracle diagnostic; the end-to-end row uses predicted boxes. The conservative confidence threshold caused a measurable loss of coverage on the held-out font, despite correct top-1 labels. These figures do not establish real-world OCR accuracy.
