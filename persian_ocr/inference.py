import io
import warnings

import numpy as np
from PIL import Image, ImageOps

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000


def decode_upload(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Upload a PNG or JPEG of at most 10 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as original:
                if original.format not in {'PNG', 'JPEG'}:
                    raise ValueError('Only PNG and JPEG images are supported.')
                if original.width * original.height > MAX_PIXELS or min(original.size) < 16:
                    raise ValueError('Image must be at least 16 pixels per side and at most 16 million pixels.')
                original.load()
                image = ImageOps.exif_transpose(original).convert('RGBA')
        background = Image.new('RGBA', image.size, 'white')
        return Image.alpha_composite(background, image).convert('RGB')
    except (OSError, ImportError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        # Ultralytics wraps Pillow.open and probes optional HEIF support after decode errors.
        # Unsupported/corrupt uploads must still become validation errors if that plugin is absent.
        raise ValueError('Cannot decode this image. Upload a valid PNG or JPEG.') from error


def decision(candidates, policy):
    visible = [c for c in candidates if c['confidence'] >= policy['candidate_floor']]
    if not visible:
        return dict(accepted=False, reason='zero detections', candidates=[])
    if len(visible) != 1:
        return dict(accepted=False, reason='multiple letters', candidates=visible)
    if visible[0]['confidence'] < policy['acceptance_threshold']:
        return dict(accepted=False, reason='low confidence', candidates=visible)
    return dict(accepted=True, reason='accepted', prediction=visible[0], candidates=visible)


def candidates_from_result(result):
    return [dict(class_id=int(c), confidence=float(s), box=[float(x) for x in b])
            for b, s, c in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                result.boxes.cls.cpu().tolist())]


def predict(model, image, policy, device='cpu'):
    # Ultralytics letterboxes at 640 and returns boxes in the original-image coordinates.
    result = model.predict(image, imgsz=640, rect=False, conf=policy['candidate_floor'],
                           iou=policy['nms_iou'], agnostic_nms=True, max_det=32,
                           device=device, verbose=False)[0]
    return decision(candidates_from_result(result), policy)


def cnn_input(image):
    """Deterministic foreground bounds retain all dark components, including dots."""
    import torch
    image = image.convert('L')
    pixels = np.asarray(image)
    ys, xs = np.where(pixels < 180)
    if len(xs):
        x1, y1 = max(0, int(xs.min())-4), max(0, int(ys.min())-4)
        x2, y2 = min(image.width, int(xs.max())+5), min(image.height, int(ys.max())+5)
        image = image.crop((x1, y1, x2, y2))
    side = max(image.size)
    square = Image.new('L', (side, side), 255)
    square.paste(image, ((side-image.width)//2, (side-image.height)//2))
    resized = square.resize((64, 64), Image.Resampling.LANCZOS)
    # Foreground 1, background 0, then fixed [-1, 1] normalization.
    array = ((1-np.asarray(resized, dtype=np.float32)/255)-0.5)/0.5
    return torch.from_numpy(array.copy()).unsqueeze(0)
