#ffmpeg_stream.py
import subprocess
import numpy as np
from typing import Optional


class FFmpegCameraStream:
    """Camera stream using FFmpeg with NVDEC and mpdecimate."""

    def __init__(self, url: str, width: Optional[int] = None, height: Optional[int] = None):
        self.url = url
        self.width = width
        self.height = height
        if self.width is None or self.height is None:
            self._probe_dimensions()
        self.frame_size = self.width * self.height * 3
        self.proc: Optional[subprocess.Popen] = None
        self._start_process()

    def _probe_dimensions(self) -> None:
        """Probe stream dimensions using ffprobe."""
        try:
            out = subprocess.check_output([
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                self.url,
            ], text=True)
            w, h = out.strip().split("x")
            self.width = int(w)
            self.height = int(h)
        except Exception:
            self.width = self.width or 640
            self.height = self.height or 480

    def _start_process(self) -> None:
        cmd = [
            "ffmpeg",
            "-hwaccel",
            "nvdec",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.url,
            "-vf",
            "mpdecimate,setpts=N/FRAME_RATE/TB",
        ]
        if self.width and self.height:
            cmd += ["-s", f"{self.width}x{self.height}"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10 ** 8,
        )

    def isOpened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def read(self):
        if not self.isOpened():
            self.release()
            self._start_process()
            return False, None
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) < self.frame_size:
            self.release()
            self._start_process()
            return False, None
        frame = np.frombuffer(raw, dtype="uint8").reshape(self.height, self.width, 3)
        return True, frame

    def release(self) -> None:
        if self.proc:
            self.proc.kill()
            self.proc = None
