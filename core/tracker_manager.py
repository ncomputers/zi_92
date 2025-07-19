#tracker_manager.py
"""Manage PersonTracker instances and related counters."""
from __future__ import annotations
import json
import time
import threading
from datetime import date
import asyncio
from typing import Dict, List

import redis
from loguru import logger

from .config import load_config, save_config, COUNT_GROUPS
from modules.person_tracker import PersonTracker

lock = threading.Lock()


def load_cameras(r: redis.Redis, default_url: str) -> List[dict]:
    data = r.get("cameras")
    if data:
        try:
            cams = json.loads(data)
            for cam in cams:
                t = cam.get("tasks")
                if isinstance(t, dict):
                    lst = []
                    if "counting" in t:
                        if "in" in t["counting"]:
                            lst.append("in_count")
                        if "out" in t["counting"]:
                            lst.append("out_count")
                    lst.extend(t.get("ppe", []))
                    if t.get("full_monitor"):
                        lst.append("full_monitor")
                    cam["tasks"] = lst
                if not isinstance(cam.get("tasks"), list):
                    cam["tasks"] = ["in_count", "out_count"]
                cam.pop("mode", None)
                cam.setdefault("type", "http")
                cam.setdefault("reverse", False)
                cam.setdefault("line_orientation", "vertical")
                cam.setdefault("resolution", "original")
            if len(cams) == 1 and cams[0].get("id") == 1 and cams[0]["url"] != default_url:
                cams[0]["url"] = default_url
                r.set("cameras", json.dumps(cams))
            return cams
        except json.JSONDecodeError:
            pass
    cams = [
        {
            "id": 1,
            "name": "Camera1",
            "url": default_url,
            "tasks": ["in_count", "out_count"],
            "enabled": True,
            "type": "http",
            "reverse": False,
            "line_orientation": "vertical",
            "resolution": "original",
        }
    ]
    r.set("cameras", json.dumps(cams))
    return cams


def save_cameras(cams: List[dict], r: redis.Redis) -> None:
    r.set("cameras", json.dumps(cams))


def start_tracker(cam: dict, cfg: dict, trackers: Dict[int, PersonTracker], r: redis.Redis | None = None) -> PersonTracker:
    tasks = cam.get("tasks", ["in_count", "out_count"])
    if isinstance(tasks, dict):
        lst = []
        if "counting" in tasks:
            if "in" in tasks["counting"]:
                lst.append("in_count")
            if "out" in tasks["counting"]:
                lst.append("out_count")
        lst.extend(tasks.get("ppe", []))
        if tasks.get("full_monitor"):
            lst.append("full_monitor")
        tasks = lst
    if r is None:
        r = redis.Redis.from_url(cfg['redis_url'])
    def _broadcast():
        from .stats import broadcast_stats
        broadcast_stats(trackers, r)

    tr = PersonTracker(
        cam["id"],
        cam["url"],
        cfg.get("object_classes", ["person"]),
        cfg,
        tasks,
        cam.get("type", "http"),
        line_orientation=cam.get("line_orientation", "vertical"),
        reverse=cam.get("reverse", False),
        resolution=cam.get("resolution", "original"),
        rtsp_transport=cam.get("rtsp_transport", "tcp"),
        update_callback=_broadcast,
    )
    trackers[cam["id"]] = tr
    threading.Thread(target=tr.capture_loop, daemon=True).start()
    threading.Thread(target=tr.process_loop, daemon=True).start()
    return tr


def stop_tracker(cam_id: int, trackers: Dict[int, PersonTracker]) -> None:
    tr = trackers.pop(cam_id, None)
    if tr:
        tr.running = False


def reset_counts(trackers: Dict[int, PersonTracker]) -> None:
    for tr in trackers.values():
        tr.in_count = 0
        tr.out_count = 0
        tr.tracks.clear()
        tr.prev_date = date.today()
        tr.redis.mset({tr.key_in: 0, tr.key_out: 0, tr.key_date: tr.prev_date.isoformat()})
    logger.info("Counts reset")


def reset_nohelmet(r: redis.Redis) -> None:
    """Reset the stored no-helmet counter in Redis."""
    r.set("no_helmet_count", 0)
    logger.info("No-helmet counter reset")


def log_counts(r: redis.Redis, trackers: Dict[int, PersonTracker]) -> None:
    ts = int(time.time())
    data = {"ts": ts}
    for g in COUNT_GROUPS.keys():
        in_c = sum(t.in_counts.get(g,0) for t in trackers.values())
        out_c = sum(t.out_counts.get(g,0) for t in trackers.values())
        data[f"in_{g}"] = in_c
        data[f"out_{g}"] = out_c
    entry = json.dumps(data)
    r.zadd("history", {entry: ts})
    r.zremrangebyrank("history", 0, -10001)
    from .stats import broadcast_stats
    broadcast_stats(trackers, r)


async def count_log_loop(r: redis.Redis, trackers: Dict[int, PersonTracker]):
    while True:
        log_counts(r, trackers)
        await asyncio.sleep(60)


last_status: str | None = None

def handle_status_change(status: str, r: redis.Redis) -> None:
    global last_status
    if status == last_status:
        return
    last_status = status
    ts = int(time.time())
    if status == "yellow":
        r.incr("yellow_alert_count")
        entry = {"ts": ts, "cam_id": 0, "track_id": 0, "status": "yellow_alert", "conf": 0, "color": None, "path": None}
        r.zadd("ppe_logs", {json.dumps(entry): ts})
        cfg_data = r.get("config")
        limit = 1000
        if cfg_data:
            try:
                limit = json.loads(cfg_data).get("ppe_log_limit", 1000)
            except Exception:
                pass
        r.zremrangebyrank("ppe_logs", 0, -limit-1)
    elif status == "red":
        r.incr("red_alert_count")
        entry = {"ts": ts, "cam_id": 0, "track_id": 0, "status": "red_alert", "conf": 0, "color": None, "path": None}
        r.zadd("ppe_logs", {json.dumps(entry): ts})
        cfg_data = r.get("config")
        limit = 1000
        if cfg_data:
            try:
                limit = json.loads(cfg_data).get("ppe_log_limit", 1000)
            except Exception:
                pass
        r.zremrangebyrank("ppe_logs", 0, -limit-1)
