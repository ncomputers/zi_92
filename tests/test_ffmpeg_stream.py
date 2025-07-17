import io
import subprocess
import numpy as np
import types
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault('cv2', type('cv2', (), {}))
sys.modules.setdefault('torch', type('torch', (), {}))
sys.modules.setdefault('ultralytics', type('ultralytics', (), {'YOLO': object}))
sys.modules.setdefault('deep_sort_realtime', type('ds', (), {}))
sys.modules['deep_sort_realtime.deepsort_tracker'] = type('t', (), {'DeepSort': object})
sys.modules.setdefault('loguru', type('loguru', (), {'logger': type('l', (), {'info': lambda *a, **k: None})()}))
sys.modules.setdefault('PIL', type('PIL', (), {}))
sys.modules.setdefault('PIL.Image', type('PIL.Image', (), {}))
sys.modules.setdefault('imagehash', type('imagehash', (), {}))

from modules.ffmpeg_stream import FFmpegCameraStream

class DummyPopen:
    def __init__(self, *args, **kwargs):
        self.stdout = io.BytesIO(b"\x01\x02\x03" * 4)
        self._poll = None
    def poll(self):
        return self._poll
    def kill(self):
        self._poll = 0


def test_ffmpeg_stream_read(monkeypatch):
    def fake_popen(cmd, stdout=None, stderr=None, bufsize=None):
        return DummyPopen()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    stream = FFmpegCameraStream("rtsp://test", width=2, height=2)
    ret, frame = stream.read()
    assert ret
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (2, 2, 3)
    stream.release()
