"""Central configuration. Everything comes from environment variables (or a local .env file)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _bool(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


BASE = Path(__file__).resolve().parent
ON_RENDER = bool(os.environ.get("RENDER"))
HOSTED = ON_RENDER or str(BASE).startswith("/mount/src") or bool(os.environ.get("SPACE_ID"))

# ---- storage -------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE / "data"))
DB_PATH = Path(os.environ.get("DB_PATH", DATA_DIR / "visionpro.db"))
RECORD_DIR = Path(os.environ.get("RECORD_DIR", DATA_DIR / "recordings"))
SNAP_DIR = Path(os.environ.get("SNAP_DIR", DATA_DIR / "snapshots"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE / "models"))
PERSISTENT_STORAGE = _bool("PERSISTENT_STORAGE", not HOSTED)   # hosted disks are temporary unless you attach one
RETENTION_DAYS = _int("RETENTION_DAYS", 30)

# ---- detection -----------------------------------------------------------
MODEL_LABELS = {
    "yoloe-11s-seg-pf.pt": "Detect everything (4,585 object types)",
    "yoloe-11m-seg-pf.pt": "Detect everything, more accurate (slower)",
    "yolo11n.pt": "Fast mode (80 common objects)",
    "yolo11s.pt": "Fast mode+ (80 common objects)",
}
ALLOWED_MODELS = [m.strip() for m in os.environ.get("ALLOWED_MODELS", ",".join(MODEL_LABELS)).split(",") if m.strip()]
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", ALLOWED_MODELS[0])
IMGSZ = _int("YOLO_IMGSZ", 640)
PROCESS_WIDTH = _int("PROCESS_WIDTH", 640 if HOSTED else 960)
TORCH_THREADS = _int("TORCH_THREADS", os.cpu_count() or 1)
MAX_VIDEO_FRAMES = _int("MAX_VIDEO_FRAMES", 600)
CAM_MAX_WIDTH = _int("CAM_MAX_WIDTH", 640)
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ---- security ------------------------------------------------------------
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "").strip()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
LOGIN_MAX_ATTEMPTS = _int("LOGIN_MAX_ATTEMPTS", 5)
LOGIN_WINDOW_SEC = _int("LOGIN_WINDOW_SEC", 300)


def missing_settings():
    errors = []
    if not ADMIN_USERNAME:
        errors.append("ADMIN_USERNAME is not set.")
    if len(ADMIN_PASSWORD) < 8:
        errors.append("ADMIN_PASSWORD is not set or shorter than 8 characters.")
    if DEFAULT_MODEL not in ALLOWED_MODELS:
        errors.append("DEFAULT_MODEL must be one of ALLOWED_MODELS.")
    return errors


def ensure_dirs():
    for d in (DATA_DIR, RECORD_DIR, SNAP_DIR, MODEL_DIR, DB_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
