from pathlib import Path
import hashlib
import json
import random

ROOT = Path(__file__).resolve().parents[1]
LETTERS = 'ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
TRANSLITERATIONS = ['alef', 'beh', 'peh', 'teh', 'seh', 'jim', 'cheh', 'heh-jimi',
                    'kheh', 'dal', 'zal', 'reh', 'zeh', 'zheh', 'sin', 'shin',
                    'sad', 'zad', 'ta', 'za', 'eyn', 'gheyn', 'feh', 'qaf',
                    'kaf', 'gaf', 'lam', 'mim', 'nun', 'vav', 'heh', 'yeh']
SOURCE_NAMES = ['Alf', 'Beh', 'Peh', 'Teh', 'Seh', 'Jim', 'Che', 'Hej', 'khe',
                'Dal', 'Zal', 'Reh', 'Zeh', 'Zhe', 'Sin', 'Shin', 'Sad', 'Zad',
                'Taa', 'Zaa', 'Ein', 'Ghein', 'Feh', 'Ghaf', 'Kaf', 'Gaf',
                'Lam', 'Mim', 'Nun', 'Vav', 'Heh', 'Yeh']


def labels():
    assert len(LETTERS) == len(TRANSLITERATIONS) == len(SOURCE_NAMES) == 32
    expected = [dict(id=i, letter=c, codepoint=f'U+{ord(c):04X}',
                     transliteration=TRANSLITERATIONS[i]) for i, c in enumerate(LETTERS)]
    path = ROOT / 'configs/labels.json'
    if path.exists():
        actual = json.loads(path.read_text(encoding='utf-8'))
        if actual != expected:
            raise ValueError('Canonical UTF-8 label map disagrees with the verified inventory')
        return actual
    return expected


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest(split, negatives=False):
    suffix = '_negative' if negatives else ''
    return [json.loads(line) for line in
            (ROOT / f'data/prepared/manifests/{split}{suffix}.jsonl').read_text().splitlines()]


def seed_all(seed=42):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    import torch
    if name == 'cpu':
        return torch.device('cpu')
    if name != 'rocm':
        raise ValueError('Device must be rocm or cpu')
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('ROCm requested but HIP PyTorch GPU execution is unavailable; no CPU fallback.')
    return torch.device('cuda:0')  # HIP deliberately uses the torch.cuda API.
