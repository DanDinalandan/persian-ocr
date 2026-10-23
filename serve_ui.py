"""Local web UI and prediction endpoint for the Persian OCR project.

Run from the repository root:

    python serve_ui.py

Then open http://127.0.0.1:8501. The model is loaded lazily on the first scan.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
UI_ROOT = ROOT / "ui"
MAX_BYTES = 10 * 1024 * 1024
HOST = os.environ.get("PERSIAN_OCR_HOST", "127.0.0.1")
PORT = int(os.environ.get("PERSIAN_OCR_PORT", "8501"))

_model: Any = None
_model_error: Exception | None = None
_model_lock = threading.Lock()


def _checkpoint_path() -> Path | None:
    configured = os.environ.get("PERSIAN_OCR_MODEL")
    candidates = [
        Path(configured).expanduser() if configured else None,
        ROOT / "yolo11n.pt",
        ROOT / "artifacts" / "yolo11n.pt",
        ROOT / "models" / "yolo11n.pt",
    ]
    return next((path.resolve() for path in candidates if path and path.is_file()), None)


def _load_model():
    global _model, _model_error
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        checkpoint = _checkpoint_path()
        if checkpoint is None:
            raise FileNotFoundError(
                "The YOLO checkpoint was not found. Add yolo11n.pt to the project root, "
                "artifacts/, or models/, then try again."
            )
        try:
            from ultralytics import YOLO

            _model = YOLO(str(checkpoint))
            _model_error = None
            return _model
        except Exception as exc:  # Model/library errors are returned as a UI-safe message.
            _model_error = exc
            raise RuntimeError("The OCR model could not be loaded.") from exc


def _predict(data: bytes) -> dict[str, Any]:
    import torch

    from persian_ocr.inference import decode_upload, predict

    labels = json.loads((ROOT / "configs" / "labels.json").read_text(encoding="utf-8"))
    policy = json.loads((ROOT / "configs" / "decision.json").read_text(encoding="utf-8"))
    image = decode_upload(data)
    model = _load_model()
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    outcome = predict(model, image, policy, device=device)
    if not outcome.get("accepted"):
        reason = outcome.get("reason", "unrecognized input")
        messages = {
            "zero detections": "No isolated Persian letter was found. Try a clearer, tighter picture.",
            "multiple letters": "More than one letter was found. Show exactly one isolated letter.",
            "low confidence": "The letter is unclear. Improve the lighting or contrast and try again.",
        }
        return {
            "accepted": False,
            "code": "unrecognized",
            "reason": reason,
            "message": messages.get(reason, "Please try again with a clearer picture."),
        }

    candidate = outcome["prediction"]
    label = labels[int(candidate["class_id"])]
    return {
        "accepted": True,
        "letter": label["letter"],
        "transliteration": label["transliteration"],
        "codepoint": label["codepoint"],
        "confidence": float(candidate["confidence"]),
    }


class PersianOCRHandler(SimpleHTTPRequestHandler):
    """Serve the UI and accept raw PNG/JPEG bodies at /api/predict."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/api/predict":
            self._json(HTTPStatus.NOT_FOUND, {"message": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].lower()
        if content_type not in {"image/png", "image/jpeg"}:
            self._json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"accepted": False, "code": "unrecognized", "message": "Only PNG and JPG images are supported."},
            )
            return
        if length <= 0 or length > MAX_BYTES:
            self._json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"accepted": False, "code": "unrecognized", "message": "Upload an image of at most 10 MB."},
            )
            return

        data = self.rfile.read(length)
        try:
            result = _predict(data)
            status = HTTPStatus.OK if result.get("accepted") else HTTPStatus.UNPROCESSABLE_ENTITY
            self._json(status, result)
        except ValueError as exc:
            self._json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"accepted": False, "code": "unrecognized", "message": str(exc)},
            )
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"accepted": False, "code": "model_unavailable", "message": str(exc)},
            )
        except Exception:
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"accepted": False, "code": "scan_failed", "message": "The scan failed unexpectedly. Please try again."},
            )

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    if not UI_ROOT.is_dir():
        raise SystemExit(f"UI directory not found: {UI_ROOT}")
    server = ThreadingHTTPServer((HOST, PORT), PersianOCRHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"Persian Letter Scanner is running at {url}")
    print("Press Ctrl+C to stop.")
    if os.environ.get("PERSIAN_OCR_NO_BROWSER") != "1":
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
