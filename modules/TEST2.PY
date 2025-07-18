
# ✅ Step 3: FFmpeg to decode with NVDEC and drop duplicates
RTSP_URL = "rtsp://rapidmistryudr%40gmail.com:Rapid@18@192.168.1.231:554/stream1"

import logging
import time
import torch
from ffmpeg_stream import FFmpegCameraStream

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

# 🔁 Replace with your actual RTSP stream URL

def test_deduplicated_stream_gpu_only():
    logging.info("🔍 Initializing FFmpegCameraStream with mpdecimate filter...")
    stream = FFmpegCameraStream(RTSP_URL)

    if not stream.isOpened():
        logging.error("❌ Stream not opened! FFmpeg process may have failed.")
        return

    logging.info("✅ Stream initialized. Starting GPU-only deduplicated frame loop...")
    last_frame_time = None
    frame_count = 0
    fail_count = 0

    try:
        while True:
            success, frame = stream.read()
            if not success or frame is None:
                logging.warning("⚠️ Frame read failed or stream ended (EOF). Retrying...")
                fail_count += 1
                time.sleep(0.2)
                continue

            now = time.time()
            delta = f"Δt = {now - last_frame_time:.3f}s" if last_frame_time else "first frame"
            last_frame_time = now

            # Convert to PyTorch tensor (GPU)
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float().cuda() / 255.0

            # You can do inference or other GPU ops here...
            # Example: dummy op
            _ = tensor.mean()

            logging.info(f"✅ Frame {frame_count + 1} received | {delta} | shape={tensor.shape}")
            frame_count += 1
            time.sleep(0.01)
    except Exception as e:
        logging.exception(f"💥 Exception occurred during streaming: {e}")
    finally:
        logging.info(f"📊 Done: {frame_count} frames read, {fail_count} failed.")
        stream.release()

if __name__ == "__main__":
    test_deduplicated_stream_gpu_only()
