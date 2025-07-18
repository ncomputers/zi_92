
import logging
import time
from ffmpeg_stream import FFmpegCameraStream

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

# 👇 Replace this with your actual RTSP stream URL
RTSP_URL = "rtsp://your_camera_ip/stream1"

def test_ffmpeg_stream():
    logging.info("🔍 Initializing FFmpegCameraStream...")
    stream = FFmpegCameraStream(RTSP_URL)

    if not stream.isOpened():
        logging.error("❌ Stream not opened! FFmpeg process may have failed.")
        return

    logging.info("✅ Stream initialized. Starting frame read loop...")
    frame_count = 0
    fail_count = 0
    try:
        while frame_count < 100:
            success, frame = stream.read()
            if not success or frame is None:
                logging.warning("⚠️ Frame read failed or stream ended (EOF).")
                fail_count += 1
                time.sleep(0.2)
                continue

            logging.info(f"✅ Frame {frame_count + 1} received, shape={frame.shape}")
            frame_count += 1
            time.sleep(0.1)
    except Exception as e:
        logging.exception(f"💥 Exception occurred during streaming: {e}")
    finally:
        logging.info(f"📊 Finished: {frame_count} frames read, {fail_count} failed.")
        stream.release()

if __name__ == "__main__":
    test_ffmpeg_stream()
