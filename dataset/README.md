# Dataset upload folder

Upload image files here, or add a ZIP containing images. The Colab notebook clones this public repository and scans this folder recursively. Supported image types: PNG, JPG/JPEG, BMP, WEBP, TIF/TIFF.

For labeled datasets, place images in class-named subfolders, for example `dataset/alef/sample1.png`. The CSV includes the first folder name as `folder_label`; it does not calculate accuracy automatically.

Do not commit private or sensitive images. GitHub's normal file-size limits apply. For larger datasets, store the dataset in a separate public repository or use the notebook's direct Colab ZIP upload option.
