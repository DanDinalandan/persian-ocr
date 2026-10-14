"""Local isolated Persian letter recognition."""
import os
from pathlib import Path

os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parents[1] / '.cache/ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
