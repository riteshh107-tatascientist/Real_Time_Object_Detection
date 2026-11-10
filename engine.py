"""VisionPro detection engine: open-vocabulary YOLOE (or YOLO11) + ByteTrack, counting, masks, recording.

step(frame) takes a BGR image and returns an annotated image (also kept in last_frame / frame_jpg).
One Engine per user session: it owns the model and the tracker state.
"""
import logging
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

import config

log = logging.getLogger("visionpro.engine")
PALETTE = [(255, 170, 0), (0, 220, 150), (255, 90, 120), (120, 140, 255),
           (0, 200, 255), (200, 120, 255), (90, 255, 120), (255, 210, 80)]


# Non-object "scene" labels that exist in the 4,585-name vocabulary and only add clutter.
SCENE_LABELS = frozenset("""autumn, autumn forest, autumn park, city view, close-up, couple photo, dawn, dusk, evening, evening light,
evening sky, evening sun, face close-up, family photo, file photo, group photo, id photo, landscape, morning, morning fog,
morning light, morning sun, mountain landscape, mountain view, night, night sky, night view, old photo, photo, photography,
point, rear view, sea view, scene, scenery, season, street photography, street scene, studio, studio shot, summer,
summer evening, sunrise, sunset, team photo, wedding photo, wedding photography, winter, winter morning, winter scene""".replace("\n", " ").split(", "))
SCENE_LABELS = frozenset(x.strip() for x in SCENE_LABELS)


def name_matches(name, terms):
    """Word-level match: 'pen' matches 'pen' and 'ballpoint pen' but not 'pencil' or 'open'."""
    n = " " + re.sub(r"[^a-z0-9]+", " ", name.lower()).strip() + " "
    return any(" " + t + " " in n for t in terms)


def clean_terms(raw):
    return [re.sub(r"[^a-z0-9]+", " ", t.lower()).strip() for t in raw if t and t.strip()]


class Engine:
    def __init__(self):
        self.cfg = dict(model=config.DEFAULT_MODEL, conf=0.25, imgsz=config.IMGSZ, classes=[],
                        trails=True, line=True, heatmap=False, masks=True, hide_scene=True)
        self.model, self.model_name = None, None
        self.frame_jpg, self.frame_id, self.last_frame = None, 0, None
        self.detections = []
        self.stats = dict(fps=0.0, latency=0.0, objects=0, per_class={}, in_count=0, out_count=0,
                          unique=0, status="starting", recording=False, totals={})
        self.history = deque(maxlen=60)
        self.events = deque(maxlen=40)
        self.trails = defaultdict(lambda: deque(maxlen=30))
        self.side, self.seen, self.last_seen = {}, set(), {}
        self.totals = defaultdict(int)
        self.heat = None
        self.writer, self.writer_size, self.writer_path = None, None, None
        self.on_event = None
        self._frames = 0
        self._t_prev, self._fps = time.time(), 0.0
        self._last_log = {}
        self.lock = threading.RLock()

    # ---------- config ----------
    def set_config(self, new):
        """Validate and apply settings. Returns an error string, or None on success."""
        with self.lock:
            if "model" in new:
                if new["model"] not in config.ALLOWED_MODELS:
                    return "Unknown model."
                if new["model"] != self.cfg["model"]:
                    self.cfg["model"] = new["model"]
                    self.cfg["classes"] = []
            if "conf" in new:
                try:
                    self.cfg["conf"] = min(0.95, max(0.03, float(new["conf"])))
                except (TypeError, ValueError):
                    return "Invalid confidence."
            if "imgsz" in new:
                try:
                    size = int(new["imgsz"])
                except (TypeError, ValueError):
                    return "Invalid input size."
                if size not in (320, 480, 640, 800, 960, 1280):
                    return "Input size must be one of 320, 480, 640, 800, 960, 1280."
                self.cfg["imgsz"] = size
            if "classes" in new:
                raw = new["classes"]
                if not isinstance(raw, (list, tuple)) or len(raw) > 100 or any(
                        not isinstance(c, str) or len(c) > 60 for c in raw):
                    return "Invalid object filter."
                self.cfg["classes"] = clean_terms(raw)
            for k in ("trails", "line", "heatmap", "masks", "hide_scene"):
                if k in new:
                    self.cfg[k] = bool(new[k])
            if new.get("reset"):
                self._reset_tracking(counts=True)
        return None

    def new_scene(self):
        """Forget tracker state (between unrelated images/videos) without touching the counters."""
        with self.lock:
            self._reset_tracking(counts=False)

    def _reset_tracking(self, counts):
        self.trails.clear(); self.side.clear(); self.last_seen.clear(); self.heat = None
        if self.model is not None:
            self.model.predictor = None          # forces ByteTrack to start fresh
        if counts:
            self.stats.update(in_count=0, out_count=0, unique=0, objects=0, per_class={}, totals={})
            self.totals.clear(); self.seen.clear(); self.events.clear(); self.history.clear()
            self.detections = []

    def log(self, msg, kind="info"):
        now = time.time()
        if now - self._last_log.get(msg, 0) < 5 and kind in ("info", "warn"):
            return
        self._last_log[msg] = now
        if len(self._last_log) > 200:
            self._last_log.clear()
        self.events.appendleft(dict(t=time.strftime("%H:%M:%S"), msg=msg, kind=kind))
        if self.on_event:
            try:
                self.on_event(msg, kind)
            except Exception:
                pass

    # ---------- model ----------
    def _ensure_model(self):
        name = self.cfg["model"]
        if self.model is not None and self.model_name == name:
            return
        self.stats["status"] = "loading model"
        try:
            import torch
            torch.set_num_threads(config.TORCH_THREADS)
            from ultralytics import YOLO
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self.model = YOLO(str(config.MODEL_DIR / name))
            self.model_name = name
            self.stats["status"] = "live"
            self.log(f"Model loaded: {name}")
        except Exception as e:
            self.cfg["model"] = self.model_name or config.DEFAULT_MODEL
            self.log(f"Could not load {name}: {e}", "warn")
            if self.model is None:
                raise

    def class_names(self):
        with self.lock:
            self._ensure_model()
            return sorted(set(self.model.names.values()))

    # ---------- one frame ----------
    def step(self, frame):
        with self.lock:
            self._ensure_model()
            t0 = time.time()
            h0, w0 = frame.shape[:2]
            if w0 != config.PROCESS_WIDTH:
                frame = cv2.resize(frame, (config.PROCESS_WIDTH, int(h0 * config.PROCESS_WIDTH / w0)))
            out = self.process(frame)
            self.stats["latency"] = round((time.time() - t0) * 1000, 1)
            now = time.time()
            self._fps = 0.8 * self._fps + 0.2 / max(now - self._t_prev, 1e-3) if self._frames > 1 else 0.0
            self._t_prev = now
            self.stats["fps"] = round(self._fps, 1)
            self._hud(out)
            self.last_frame = out
            if self.writer is not None:
                h, w = out.shape[:2]
                self.writer.write(out if (w, h) == self.writer_size else cv2.resize(out, self.writer_size))
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                self.frame_jpg = buf.tobytes()
                self.frame_id += 1
            return self.frame_jpg

    def process(self, frame):
        cfg = self.cfg
        h, w = frame.shape[:2]
        r = self.model.track(frame, persist=True, conf=cfg["conf"], imgsz=cfg["imgsz"], verbose=False,
                             tracker="bytetrack.yaml")[0]
        terms = cfg["classes"]
        line_y = int(h * 0.55)
        if self.heat is None or self.heat.shape[:2] != (h, w):
            self.heat = np.zeros((h, w), np.float32)
        self.heat *= 0.985
        self._frames += 1

        dets = []                                           # (index, box, track id, name, conf, class id)
        if r.boxes is not None and len(r.boxes):
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            for i, (box, tid) in enumerate(zip(r.boxes, ids)):
                cls = int(box.cls[0])
                name = self.model.names[cls]
                if terms and not name_matches(name, terms):
                    continue
                if cfg["hide_scene"] and not terms and name in SCENE_LABELS:
                    continue
                dets.append((i, box, tid, name, float(box.conf[0]), cls))

        if cfg["masks"] and dets and r.masks is not None:   # segmentation overlay (YOLOE -seg models)
            overlay = frame.copy()
            for i, _, _, _, _, cls in dets:
                try:
                    pts = np.int32(r.masks.xy[i])
                    if len(pts) > 2:
                        cv2.fillPoly(overlay, [pts], PALETTE[cls % len(PALETTE)])
                except (IndexError, ValueError):
                    pass
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)

        per_class, shown = defaultdict(int), []
        for _, box, tid, name, conf, cls in dets:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            col = PALETTE[cls % len(PALETTE)]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            per_class[name] += 1
            shown.append(dict(object=name, confidence=round(conf, 2), track_id=tid))
            cv2.circle(self.heat, (cx, cy), 28, 1.0, -1)
            if tid is not None:
                self.last_seen[tid] = self._frames
                if tid not in self.seen:
                    self.seen.add(tid)
                    self.totals[name] += 1
                    self.log(f"New {name} #{tid}", "new")
                tr = self.trails[tid]
                tr.append((cx, cy))
                if cfg["trails"] and len(tr) > 1:
                    for k in range(1, len(tr)):
                        cv2.line(frame, tr[k - 1], tr[k], col, max(1, k // 8))
                below = cy > line_y
                if cfg["line"] and tid in self.side and self.side[tid] != below:
                    self.stats["in_count" if below else "out_count"] += 1
                    self.log(f"{name} #{tid} crossed {'IN' if below else 'OUT'}", "in" if below else "out")
                self.side[tid] = below
            self._draw_box(frame, x1, y1, x2, y2, col,
                           f"{name}{' #' + str(tid) if tid is not None else ''} {conf:.0%}")

        if self._frames % 300 == 0:                          # bounded memory
            for tid in [t for t, f in self.last_seen.items() if self._frames - f > 300]:
                self.last_seen.pop(tid, None); self.trails.pop(tid, None); self.side.pop(tid, None)
        if cfg["heatmap"]:
            hm = cv2.applyColorMap(np.clip(self.heat * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask = (self.heat > 0.05)[..., None]
            frame[:] = np.where(mask, cv2.addWeighted(frame, 0.55, hm, 0.45, 0), frame)
        if cfg["line"]:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 220, 150), 2)
            cv2.putText(frame, "COUNT LINE", (10, line_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 150), 1)
        self.detections = sorted(shown, key=lambda d: -d["confidence"])[:40]
        self.stats.update(objects=len(dets), per_class=dict(per_class), unique=len(self.seen),
                          totals=dict(self.totals))
        self.history.append(len(dets))
        return frame

    @staticmethod
    def _draw_box(img, x1, y1, x2, y2, col, label):
        L = max(10, min(x2 - x1, y2 - y1) // 4)
        for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(img, (px, py), (px + dx * L, py), col, 3)
            cv2.line(img, (px, py), (px, py + dy * L), col, 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 4 if y1 - th - 10 > 0 else y1 + th + 8
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), col, -1)
        cv2.putText(img, label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 20), 1, cv2.LINE_AA)

    @staticmethod
    def _hud(img):
        cv2.putText(img, "VisionPro | Ritesh Kumar Singh", (12, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ---------- snapshot / recording ----------
    @staticmethod
    def _keep_latest(folder, pattern, keep):
        files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[keep:]:
            try:
                old.unlink()
            except OSError:
                pass

    def snapshot(self):
        if self.frame_jpg is None:
            return None
        config.SNAP_DIR.mkdir(parents=True, exist_ok=True)
        path = config.SNAP_DIR / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path.write_bytes(self.frame_jpg)
        self._keep_latest(config.SNAP_DIR, "snapshot_*.jpg", 30)
        self.log("Snapshot saved")
        return path

    def _close_writer(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = self.writer_size = None
        self.stats["recording"] = False

    def shutdown(self):
        with self.lock:
            self._close_writer()

    def toggle_record(self):
        """Returns (recording, filename). Filename is set when a recording was just finished."""
        with self.lock:
            if self.writer is not None:
                name = self.writer_path.name
                self._close_writer()
                self.log("Recording stopped")
                return False, name
            if self.last_frame is None:
                return False, None
            h, w = self.last_frame.shape[:2]
            config.RECORD_DIR.mkdir(parents=True, exist_ok=True)
            self.writer_path = config.RECORD_DIR / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fps = max(2, min(30, round(self._fps or 5)))
            writer = cv2.VideoWriter(str(self.writer_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                self.log("Recording could not start", "warn")
                return False, None
            self.writer, self.writer_size = writer, (w, h)
            self.stats["recording"] = True
            self._keep_latest(config.RECORD_DIR, "recording_*.mp4", 5)
            self.log("Recording started")
            return True, None
