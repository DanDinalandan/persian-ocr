# Persian Letter Scanner UI

The UI follows this flow:

1. Choose **Take a photo** or **Upload image**.
2. Capture/select one isolated Persian letter.
3. Review the full image and choose **Retake** or **Scan**.
4. The processing dialog stays visible while the model runs.
5. A successful scan shows the Persian letter, transliteration, confidence, and **Copy text** action. Model and recognition failures appear as retry/home dialogs.

## Run in VS Code

Open the repository folder in VS Code, then use its integrated terminal:

```powershell
py -m pip install -r requirements-ui.txt
py serve_ui.py
```

Open `http://127.0.0.1:8501` if it does not open automatically.

The server looks for the trained detector at the first matching location:

- the path in the `PERSIAN_OCR_MODEL` environment variable;
- `yolo11n.pt` in the repository root;
- `artifacts/yolo11n.pt`;
- `models/yolo11n.pt`.

The current source archive does not contain `yolo11n.pt`. Until that published checkpoint is added, the interface deliberately shows the **Model can’t load** dialog after Scan.

## UI files

- `ui/index.html` — screens and dialogs
- `ui/styles.css` — reference palette, typography, responsive layout, and animations
- `ui/app.js` — camera/upload/preview/scan/result interactions
- `serve_ui.py` — static server and `/api/predict` model endpoint

Camera access works on localhost in modern browsers. Uploaded images are validated as PNG/JPEG and limited to 10 MB before inference.
