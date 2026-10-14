# Persian isolated letter recognition

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yt22-orb/persian-ocr/blob/main/Persian_Isolated_Letter_Recognizer.ipynb)

A YOLO11n detector for recognizing one isolated Persian letter. The Colab notebook can pull a dataset from a public GitHub repository or accept an uploaded ZIP, scan its images, and download a CSV of predictions.

## Use a GitHub dataset

Upload images or a ZIP to the `dataset/` folder in a public repository. The notebook defaults to this repository, so you can upload your dataset to its `dataset/` folder and run the notebook. To use another public dataset repository, change `DATASET_REPO` in the notebook to `owner/repository`. Set `DATASET_SUBDIR` if the images are in a different folder. Organizing images into class-named folders records those folder names as reference labels in the output CSV.

For a direct Colab upload, set `DATASET_SOURCE = "colab_upload"` and upload a ZIP of images.

## Model scope

The model was trained on synthetic isolated typography. On this small benchmark it reached 256/256 exact accuracy on a held-out synthetic-font test split and rejected 128/128 blank and two-letter negatives. These measurements do not establish performance on natural handwriting, camera images, connected text, or other real-world inputs. See [measured results](RESULTS.md) and [attribution](ATTRIBUTION.md).

Download `yolo11n.pt` and `labels.json` from the [latest release](https://github.com/yt22-orb/persian-ocr/releases/latest) and place them in the same Google Drive folder as the notebook.
