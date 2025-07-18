#ppe_worker.py
import time
import json
from modules.profiler import register_thread, profile_predict, log_resource_usage
import threading
from pathlib import Path
import cv2
from loguru import logger
from ultralytics import YOLO
import redis
import torch


class PPEDetector(threading.Thread):
    """Background worker that reads person_logs and performs PPE detection."""

    def __init__(self, cfg: dict, redis_url: str, snap_dir: Path, update_callback=None):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.redis = redis.Redis.from_url(redis_url)
        self.model = YOLO(cfg.get("ppe_model", "mymodel.pt"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logger.info(f"\U0001F9E0 CUDA Enabled: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("\u26A0\uFE0F CUDA not available, using CPU.")
        self.model.model.to(self.device)
        if self.device.type == "cuda":
            self.model.model.half()

        self.last_ts = 0
        self.snap_dir = Path(snap_dir)
        self.running = True
        self.update_callback = update_callback

    @staticmethod
    def _clean_label(name: str) -> str:
        """Normalize labels to lowercase with underscores."""
        return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


    def run(self):
        register_thread("PPEWorker")
        frame_idx = 0
        last_log = time.time()
        skip = int(self.cfg.get("skip_frames", 0))
        while self.running:
            entries = [
                json.loads(e)
                for e in self.redis.zrangebyscore("person_logs", self.last_ts + 1, "+inf")
            ]
            for entry in entries:
                self.last_ts = max(self.last_ts, entry.get("ts", 0))
                if not entry.get("needs_ppe"):
                    continue
                path = entry.get("path")
                if not path:
                    continue
                img_path = Path(path)
                if not img_path.is_absolute():
                    img_path = self.snap_dir / img_path.name
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                frame_idx += 1
                if skip and frame_idx % skip:
                    continue
                res = profile_predict(self.model, "PPEWorker", img, device=self.device, verbose=False)[0]
                scores = {}
                for *xyxy, conf, cls in res.boxes.data.tolist():
                    label = self._clean_label(self.model.names[int(cls)])
                    if conf > scores.get(label, 0):
                        scores[label] = conf
                logger.debug(f"PPE scores: {scores}")
                for item in self.cfg.get("track_ppe", []):
                    key = self._clean_label(item)
                    conf = scores.get(key, 0)
                    status = item if conf >= self.cfg.get("helmet_conf_thresh", 0.5) else f"no_{item}"
                    ts = int(time.time())
                    log = {
                        "ts": ts,
                        "cam_id": entry.get("cam_id"),
                        "track_id": entry.get("track_id"),
                        "status": status,
                        "conf": conf,
                        "path": str(img_path.name),
                    }
                    self.redis.zadd("ppe_logs", {json.dumps(log): ts})
                    if status.startswith('no_'):
                        self.redis.incr(f"{status}_count")
                    if self.update_callback:
                        self.update_callback()
            if time.time() - last_log > 10:
                logger.debug(
                    f"PPEWorker processed {frame_idx} frames in {time.time() - last_log:.1f}s"
                )
                log_resource_usage("PPEWorker")
                frame_idx = 0
                last_log = time.time()
            time.sleep(1)
