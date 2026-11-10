"""Engine + database smoke test with the REAL models (no mocks).
Run:  python tests/smoke_test.py     (needs internet once to fetch model weights and a sample image)
"""
import os, sys, tempfile, time, urllib.request
from pathlib import Path

tmp = tempfile.mkdtemp(prefix="vp_test_")
os.environ.update(DATA_DIR=tmp, ADMIN_USERNAME="tester", ADMIN_PASSWORD="test-pass-123", PROCESS_WIDTH="640")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, numpy as np
import config, db
from coco import COCO_NAMES
from engine import Engine, name_matches

results = []
def check(name, cond, extra=""):
    results.append(bool(cond)); print(("PASS" if cond else "FAIL"), "-", name, extra, flush=True)

img_path = Path(tmp) / "bus.jpg"
for url in ("https://ultralytics.com/images/bus.jpg",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"):
    try:
        urllib.request.urlretrieve(url, img_path); break
    except Exception:
        pass
bus = cv2.imread(str(img_path))
check("sample image downloaded", bus is not None)

db.init(); db.ensure_admin("tester", "test-pass-123"); db.start_writer()
eng = Engine(); eng.on_event = db.add_event
check("default model is the open-vocabulary one", eng.cfg["model"].startswith("yoloe"), f"({eng.cfg['model']})")

# ---- open-vocabulary detection ----
eng.step(bus)
found = set(eng.stats["per_class"])
check("detects bus and person", {"bus", "person"} <= found, f"({sorted(found)})")
beyond = [n for n in found if n not in COCO_NAMES]
check("detects objects OUTSIDE the 80 COCO classes", len(beyond) >= 1, f"({beyond})")
check("vocabulary has thousands of classes", len(eng.class_names()) > 4000, f"({len(eng.class_names())})")
check("annotated frame + jpeg produced", eng.last_frame is not None and eng.frame_jpg[:2] == b"\xff\xd8")
check("detection table filled", len(eng.detections) >= 3 and "object" in eng.detections[0])

# ---- name filter ----
check("word match: 'pen' ~ 'ballpoint pen'", name_matches("ballpoint pen", ["pen"]))
check("word match: 'pen' !~ 'pencil' / 'open'", not name_matches("pencil", ["pen"]) and not name_matches("open sign", ["pen"]))
check("rejects unknown model", eng.set_config({"model": "../../etc/passwd"}) is not None)
check("rejects bad detection size", eng.set_config({"imgsz": 12345}) is not None)
check("rejects bad filter", eng.set_config({"classes": "bus"}) is not None and eng.set_config({"classes": ["x" * 80]}) is not None)
check("accepts filter 'bus'", eng.set_config({"classes": ["bus"]}) is None)
eng.new_scene(); eng.step(bus)
check("filter shows only the bus", set(eng.stats["per_class"]) == {"bus"}, f"({eng.stats['per_class']})")
eng.set_config({"classes": []})

# ---- scene-label clutter ----
from engine import SCENE_LABELS
check("scene labels list loaded", "studio shot" in SCENE_LABELS and "night view" in SCENE_LABELS and "pen" not in SCENE_LABELS and "needle" not in SCENE_LABELS)
eng.set_config({"hide_scene": True}); eng.new_scene(); eng.step(bus)
check("no scene labels shown when hidden", not (set(eng.stats["per_class"]) & SCENE_LABELS))

# ---- masks on/off ----
eng.set_config({"masks": False}); eng.new_scene(); eng.step(bus); off = eng.last_frame.copy()
eng.set_config({"masks": True}); eng.new_scene(); eng.step(bus); on = eng.last_frame.copy()
check("segmentation masks change the picture", np.abs(on.astype(int) - off.astype(int)).mean() > 0.5)

# ---- fast COCO mode still works ----
eng.set_config({"model": "yolo11n.pt"}); eng.new_scene(); eng.step(bus)
check("fast COCO model works", eng.stats["objects"] >= 3 and eng.model_name == "yolo11n.pt", f"({eng.stats['per_class']})")
eng.set_config({"model": "yoloe-11s-seg-pf.pt"})

# ---- tracking + IN/OUT counting on a synthetic walk ----
eng.new_scene(); eng.step(bus)
people = [(b.xyxy[0].tolist()) for b in eng.model.track(bus, persist=False, verbose=False, conf=0.4)[0].boxes
          if eng.model.names[int(b.cls[0])] == "person"]
x1, y1, x2, y2 = map(int, sorted(people, key=lambda b: -(b[2]-b[0])*(b[3]-b[1]))[0])
person = bus[y1:y2, x1:x2]
person = cv2.resize(person, (120, int(120 * person.shape[0] / person.shape[1])))
W, H, n_frames = 640, 480, 36
frames_list = []
for i in range(n_frames):
    f = np.full((H, W, 3), 110, np.uint8)
    top = int(10 + i * ((H - person.shape[0] - 20) / (n_frames - 1)))
    f[top:top + person.shape[0], 260:380] = person
    frames_list.append(f)
vid = Path(tmp) / "walk.mp4"
vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
for f in frames_list: vw.write(f)
vw.release()

eng.set_config({"reset": True, "line": True, "conf": 0.3, "classes": ["person"]})
for f in frames_list: eng.step(f)
check("tracking: exactly 1 unique person (raw frames)", eng.stats["unique"] == 1, f"(unique={eng.stats['unique']})")
check("counting: exactly 1 IN, 0 OUT (raw frames)", (eng.stats["in_count"], eng.stats["out_count"]) == (1, 0),
      f"(in={eng.stats['in_count']}, out={eng.stats['out_count']})")
check("per-class totals recorded", eng.stats["totals"].get("person", 0) == 1)

eng.set_config({"reset": True})
cap = cv2.VideoCapture(str(vid)); frames = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    eng.step(fr); frames += 1
cap.release()
check("MP4 file processed frame by frame", frames == n_frames)
check("MP4 file: crossing detected", eng.stats["in_count"] >= 1,
      f"(unique={eng.stats['unique']}, in={eng.stats['in_count']}; compression can split a track)")
eng.set_config({"classes": [], "conf": 0.25})

# ---- snapshot + recording ----
snap = eng.snapshot()
check("snapshot saved", snap is not None and snap.exists() and snap.stat().st_size > 1000)
started, _ = eng.toggle_record()
for _ in range(4): eng.step(bus)
stopped, name = eng.toggle_record()
rec = config.RECORD_DIR / (name or "x")
check("recording written", started and not stopped and rec.exists() and rec.stat().st_size > 1000)
rc = cv2.VideoCapture(str(rec)); ok, _ = rc.read(); rc.release()
check("recording readable by OpenCV", ok)

# ---- database ----
time.sleep(1.5); db._flush()
ev = db.events("", 500)
check("events persisted in SQLite", len(ev) > 0, f"({len(ev)} rows)")
check("IN events persisted", any(e["kind"] == "in" for e in ev))
db.add_sample(eng.stats); db._flush()
check("samples persisted", len(db.samples(60)) == 1)
check("admin login works", db.check_user("tester", "test-pass-123") is not None)
check("wrong password rejected", db.check_user("tester", "nope") is None)
db.ensure_admin("tester", "new-pass-456")
check("password change via env applied", db.check_user("tester", "new-pass-456") is not None and db.check_user("tester", "test-pass-123") is None)
os.remove(config.DB_PATH); db.init()
check("DB auto-recreated if file missing", db.summary()["new"] == 0)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
