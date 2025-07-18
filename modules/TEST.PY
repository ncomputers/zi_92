
# 👇 Replace this with your actual RTSP stream URL
#RTSP_URL = "rtsp://rapidmistryudr%40gmail.com:Rapid@18@192.168.1.231:554/stream1"
import sys
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtCore import QTimer
from ffmpeg_stream import FFmpegCameraStream

# 🔁 Replace with your actual RTSP stream
RTSP_URL = "rtsp://rapidmistryudr%40gmail.com:Rapid@18@192.168.1.231:554/stream1"

class VideoWidget(QOpenGLWidget):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.image = None
        self.tensor = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33ms ≈ 30 FPS

    def update_frame(self):
        success, frame = self.stream.read()
        if not success:
            print("⚠️ Frame read failed.")
            return

        # 👉 Push to GPU
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float().cuda() / 255.0
        self.tensor = tensor

        # 🖼️ Convert back to QImage (must be CPU for display)
        frame_cpu = (tensor * 255).byte().permute(1, 2, 0).cpu().numpy()
        h, w, _ = frame_cpu.shape
        self.image = QImage(frame_cpu.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()

        self.update()

    def paintEvent(self, event):
        if self.image:
            painter = QPainter(self)
            painter.drawImage(0, 0, self.image)

class MainWindow(QMainWindow):
    def __init__(self, stream):
        super().__init__()
        self.setWindowTitle("🖥️ GPU Video Viewer")
        self.viewer = VideoWidget(stream)
        self.setCentralWidget(self.viewer)
        self.resize(1280, 720)

def main():
    stream = FFmpegCameraStream(RTSP_URL)
    if not stream.isOpened():
        print("❌ Failed to open stream.")
        return

    app = QApplication(sys.argv)
    window = MainWindow(stream)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
