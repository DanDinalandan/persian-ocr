# Persian isolated letter recognition

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yt22-orb/persian-ocr/blob/main/Persian_Isolated_Letter_Recognizer.ipynb)

A YOLO11n detector for recognizing one isolated Persian letter. The included Colab notebook runs inference on an uploaded image. Download `yolo11n.pt` from the [v1.0.0 release](https://github.com/yt22-orb/persian-ocr/releases/tag/v1.0.0), put it and `labels.json` in a folder in Google Drive, and run the notebook.

## Model scope

The model was trained on synthetic isolated typography. On this small benchmark it reached 256/256 exact accuracy on a held-out synthetic-font test split and rejected 128/128 blank and two-letter negatives. These measurements do not establish performance on natural handwriting, camera images, connected text, or other real-world inputs. There are no camera captures or natural handwriting in the dataset. See [measured results](RESULTS.md) and [attribution](ATTRIBUTION.md).

## Files

- `Persian_Isolated_Letter_Recognizer.ipynb`: Google Colab inference notebook.
- `configs/labels.json`: 32-class label map.
- `configs/decision.json`: validation-selected confidence and NMS policy.
- `artifacts/yolo_metadata.json`: model checkpoint metadata and hash.
- GitHub release assets: YOLO11n checkpoint, reduced CNN comparison checkpoint, and a ZIP package containing both.

## Local setup

Install the pinned Ultralytics version and Pillow/NumPy, then load the checkpoint with `from ultralytics import YOLO; model = YOLO('yolo11n.pt')`. The project’s Streamlit interface and training pipeline are documented in the full project README in the workspace; this release repository is the inference package.
