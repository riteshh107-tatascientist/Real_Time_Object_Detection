"""VisionPro - AI Vision Studio (Streamlit edition).
Open-vocabulary object detection (4,585 object types) + tracking + counting + segmentation.
Author: Ritesh Kumar Singh (Founder & CEO)
"""
import base64
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="VisionPro | AI Vision Studio", page_icon="👁️", layout="wide",
                   initial_sidebar_state="auto")

try:                                    # Streamlit Cloud "Secrets" -> environment variables
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd

import config
import db
from engine import Engine

camera_component = components.declare_component("vp_camera", path=str(config.BASE / "components" / "camera"))

# ------------------------------------------------------------------ look & feel
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden}
.block-container{padding-top:1.2rem;max-width:1400px}
.hero{display:flex;align-items:center;gap:16px;margin-bottom:6px;animation:rise .8s both}
.logo{flex:none;width:52px;height:52px;border-radius:14px;background:linear-gradient(135deg,#19e3b1,#6c8cff);display:grid;place-items:center;color:#06101a;font:800 24px sans-serif;box-shadow:0 0 28px rgba(25,227,177,.35)}
.hero h1{margin:0;font-size:2rem;background:linear-gradient(90deg,#19e3b1,#6c8cff,#19e3b1);background-size:200% auto;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shine 6s linear infinite}
.hero small{color:#8a98b8}
.scan{height:2px;margin:10px 0 18px;background:linear-gradient(90deg,transparent,#19e3b1,transparent);box-shadow:0 0 14px #19e3b1;animation:glow 3s ease-in-out infinite}
@keyframes shine{to{background-position:200% center}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}}
@keyframes glow{50%{opacity:.35}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes sweep{0%{background-position:160% 0}100%{background-position:-60% 0}}
[data-testid="stMetric"]{background:rgba(20,28,44,.72);border:1px solid rgba(120,150,200,.16);border-radius:14px;padding:10px 14px;transition:.25s}
[data-testid="stMetric"]:hover{border-color:#19e3b1;transform:translateY(-2px)}
[data-testid="stMetricValue"]{font-weight:800}
.stTabs [data-baseweb="tab"]{font-weight:600}
.foot{text-align:center;color:#8a98b8;font-size:.8rem;margin-top:30px}
/* founder portrait: full image (never cropped) with animated studio lighting */
.founder{position:relative;margin:26px auto 10px;width:100%}
.founder .halo{position:absolute;inset:-22px;border-radius:44px;background:conic-gradient(#19e3b1,#6c8cff,#ff5a78,#ffb020,#19e3b1);filter:blur(34px);opacity:.55;animation:spin 10s linear infinite}
.founder .frame{position:relative;border-radius:22px;padding:3px;background:linear-gradient(135deg,#19e3b1,#6c8cff,#ff5a78);box-shadow:0 24px 60px rgba(0,0,0,.65)}
.founder img{display:block;width:100%;height:auto;border-radius:19px;background:#000}
.founder .frame::after{content:"";position:absolute;inset:3px;border-radius:19px;pointer-events:none;background:linear-gradient(115deg,transparent 38%,rgba(255,255,255,.20) 50%,transparent 62%);background-size:250% 100%;animation:sweep 5.5s ease-in-out infinite}
.fname{text-align:center;font-weight:800;font-size:1.3rem;margin-top:16px}
.frole{text-align:center;color:#8a98b8}
.card{background:rgba(20,28,44,.72);border:1px solid rgba(120,150,200,.16);border-radius:16px;padding:16px 18px;margin-bottom:12px}
.card h4{margin:0 0 6px;color:#19e3b1}
@media(max-width:640px){.hero h1{font-size:1.5rem}.logo{width:42px;height:42px;font-size:20px}.block-container{padding-left:.8rem;padding-right:.8rem}}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style>""", unsafe_allow_html=True)


def hero(sub="Real-time AI Vision Studio"):
    st.markdown(f"""<div class="hero"><div class="logo">V</div><div><h1>VisionPro</h1>
    <small>{sub} &nbsp;|&nbsp; Ritesh Kumar Singh, Founder &amp; CEO</small></div></div><div class="scan"></div>""",
                unsafe_allow_html=True)


@st.cache_resource
def founder_b64():
    p = config.BASE / "assets" / "founder.jpg"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


def founder_card(max_width=340):
    b64 = founder_b64()
    if not b64:
        return
    st.markdown(f"""<div style="max-width:{max_width}px;margin:auto"><div class="founder"><div class="halo"></div>
    <div class="frame"><img alt="Ritesh Kumar Singh, Founder and CEO of VisionPro" src="data:image/jpeg;base64,{b64}"></div></div>
    <div class="fname">Ritesh Kumar Singh</div><div class="frole">Founder &amp; CEO, VisionPro</div></div>""",
                unsafe_allow_html=True)


# ------------------------------------------------------------------ configuration check
problems = config.missing_settings()
if problems:
    hero()
    st.error("VisionPro is not configured yet.")
    st.markdown("Set these **secrets / environment variables** (see `.env.example`):\n\n"
                + "\n".join(f"- {p}" for p in problems))
    st.stop()


@st.cache_resource
def boot():
    db.init()
    db.ensure_admin(config.ADMIN_USERNAME, config.ADMIN_PASSWORD)
    db.start_writer()
    return {"attempts": [], "lock": threading.Lock()}


_state = boot()


def locked_out(record=False):
    now = time.time()
    with _state["lock"]:
        _state["attempts"] = [t for t in _state["attempts"] if now - t < config.LOGIN_WINDOW_SEC]
        if record:
            _state["attempts"].append(now)
        return len(_state["attempts"]) >= config.LOGIN_MAX_ATTEMPTS * 2


# ------------------------------------------------------------------ login
if not st.session_state.get("user"):
    hero("Sign in to continue")
    left, right = st.columns([1, 1.15], gap="large", vertical_alignment="center")
    with left:
        founder_card(300)
    with right:
        with st.form("login"):
            st.markdown("#### Welcome back")
            u = st.text_input("Username", autocomplete="username")
            p = st.text_input("Password", type="password", autocomplete="current-password")
            go = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if go:
            if locked_out():
                st.error("Too many failed attempts. Please wait a few minutes.")
            else:
                row = db.check_user(u.strip(), p)
                if row:
                    st.session_state["user"] = row["username"]
                    st.rerun()
                locked_out(record=True)
                st.error("Wrong username or password.")
    st.markdown('<div class="foot">Built by Ritesh Kumar Singh, Founder &amp; CEO</div>', unsafe_allow_html=True)
    st.stop()


# ------------------------------------------------------------------ per-session engine + sidebar
if "eng" not in st.session_state:
    _e = Engine()
    _e.on_event = db.add_event
    st.session_state["eng"] = _e
eng: Engine = st.session_state["eng"]

with st.sidebar:
    st.markdown("### Controls")
    model = st.selectbox("Detection model", config.ALLOWED_MODELS, format_func=lambda m: config.MODEL_LABELS.get(m, m),
                         index=config.ALLOWED_MODELS.index(eng.cfg["model"]))
    conf = st.slider("Confidence (lower finds more)", 3, 95, int(eng.cfg["conf"] * 100), format="%d%%")
    sizes = [320, 480, 640, 800, 960, 1280]
    imgsz = st.select_slider("Detection size (bigger finds small objects)", sizes,
                             value=min(sizes, key=lambda x: abs(x - eng.cfg["imgsz"])))
    only = st.text_input("Only show these objects", placeholder="pen, bottle, book",
                         help="Comma separated. Empty = show everything the model finds.")
    with st.expander("Search object names"):
        q = st.text_input("Type part of a name", key="name_q").strip().lower()
        if q:
            hits = [n for n in eng.class_names() if q in n.lower()][:40]
            st.write(", ".join(hits) if hits else "No match.")
        else:
            st.caption("The Detect-everything models know 4,585 names (pen, needle, laptop, screwdriver...).")
    trails = st.toggle("Motion trails", eng.cfg["trails"])
    line = st.toggle("Counting line", eng.cfg["line"])
    masks = st.toggle("Segmentation masks", eng.cfg["masks"])
    heat = st.toggle("Heatmap", eng.cfg["heatmap"])
    hide_scene = st.toggle("Hide scene labels (night view, studio shot...)", eng.cfg["hide_scene"])
    if st.button("Reset counts", use_container_width=True):
        eng.set_config({"reset": True})
    st.divider()
    st.caption(f"Signed in as **{st.session_state['user']}**")
    if st.button("Log out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

_err = eng.set_config({"model": model, "conf": conf / 100, "imgsz": imgsz,
                       "classes": [t for t in only.split(",") if t.strip()],
                       "trails": trails, "line": line, "masks": masks, "heatmap": heat, "hide_scene": hide_scene})
if _err:
    st.sidebar.error(_err)


# ------------------------------------------------------------------ helpers
def maybe_sample():
    if time.time() - st.session_state.get("_last_sample", 0) > 5 and eng.frame_id:
        st.session_state["_last_sample"] = time.time()
        db.add_sample(eng.stats)


def show_metrics(container):
    s = eng.stats
    items = [("Objects in frame", s["objects"]), ("Unique tracked", s["unique"]), ("Crossed IN", s["in_count"]),
             ("Crossed OUT", s["out_count"]), ("FPS", s["fps"]), ("Latency (ms)", s["latency"])]
    with container.container():
        for row in (items[:3], items[3:]):
            for col, (label, val) in zip(st.columns(3), row):
                col.metric(label, val)
        found = ", ".join(f"{n} x{c}" for n, c in sorted(s["per_class"].items(), key=lambda kv: -kv[1]))
        st.caption("Detected now: " + (found or "nothing yet. Try a lower confidence or a bigger detection size."))


def detections_table():
    if eng.detections:
        with st.expander(f"Detection details ({len(eng.detections)})"):
            st.dataframe(pd.DataFrame(eng.detections), hide_index=True, use_container_width=True)


def to_h264(src: Path) -> Path:
    """Browsers cannot play OpenCV's mp4v codec, so transcode to H.264 when ffmpeg is available."""
    try:
        import imageio_ffmpeg
        dst = src.with_name("annotated_h264.mp4")
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(src), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart", str(dst)],
                           capture_output=True, timeout=600)
        return dst if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0 else src
    except Exception:
        return src


def process_video(path, skip, preview, bar, metrics_box):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("That file could not be opened as a video.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or config.MAX_VIDEO_FRAMES
    target = max(1, min(total, config.MAX_VIDEO_FRAMES))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    eng.set_config({"reset": True})
    raw = Path(tempfile.mkdtemp(prefix="vp_")) / "annotated_raw.mp4"
    writer, i, out = None, 0, None
    try:
        while i < target:
            ok, frame = cap.read()
            if not ok:
                break
            if out is None or i % skip == 0:           # analyse every Nth frame, reuse the result in between
                eng.step(frame)
                out = eng.last_frame
                maybe_sample()
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(out)
            i += 1
            if i == 1 or i % (3 * skip) == 0:
                preview.image(out, channels="BGR", output_format="JPEG", use_container_width=True)
                bar.progress(min(i / target, 1.0), text=f"Processing frame {i} / {target}")
                show_metrics(metrics_box)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if i == 0:
        raise ValueError("The video contains no readable frames.")
    show_metrics(metrics_box)
    bar.progress(1.0, text=f"Done. {i} frames processed.")
    final = to_h264(raw)
    return dict(data=final.read_bytes(), frames=i, truncated=i >= config.MAX_VIDEO_FRAMES and total > config.MAX_VIDEO_FRAMES,
                totals=dict(eng.stats["totals"]), unique=eng.stats["unique"],
                inn=eng.stats["in_count"], out=eng.stats["out_count"], h264=final.name.endswith("h264.mp4"))


def csv_bytes(rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    if "msg" in df:
        df["msg"] = df["msg"].map(lambda v: "'" + v if v[:1] in ("=", "+", "-", "@") else v)
    return df.to_csv(index=False).encode()


@st.fragment
def live_view():
    """Live camera. The browser captures frames (front or back camera) and sends them over Streamlit's own
    connection, so no WebRTC/TURN server is needed. Only this fragment reruns for each frame."""
    side = st.radio("Camera", ["Front", "Back"], horizontal=True, key="cam_side",
                    help="On a phone, Back uses the rear camera.")
    st.session_state["ack"] = st.session_state.get("ack", 0) + 1
    val = camera_component(facing="environment" if side == "Back" else "user", ack=st.session_state["ack"],
                           max_width=config.CAM_MAX_WIDTH, delay_ms=40, key="live_cam", default=None)
    if isinstance(val, dict):
        if val.get("error"):
            st.error(f"Camera problem: {val['error']}. Allow camera access in your browser (HTTPS is required).")
        elif val.get("img") and val.get("n") != st.session_state.get("cam_n"):
            st.session_state["cam_n"] = val["n"]
            try:
                raw = base64.b64decode(val["img"].split(",", 1)[1])
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if len(raw) < 3_000_000 else None
                if frame is not None:
                    eng.step(frame)
                    st.session_state["live_jpg"] = eng.frame_jpg
                    maybe_sample()
            except Exception as e:
                eng.log(f"Frame error: {e}", "warn")
        elif val.get("stopped"):
            st.session_state["cam_n"] = val.get("n")
    jpg = st.session_state.get("live_jpg")
    if not jpg:
        st.info("Tap **Start camera**, allow access, and detection starts. Works on phones (front and back camera).")
        return
    st.image(jpg, use_container_width=True)
    show_metrics(st.container())
    detections_table()
    b1, b2 = st.columns(2)
    b1.download_button("📸 Download this frame", jpg, "visionpro_frame.jpg", "image/jpeg", use_container_width=True)
    if b2.button("⏹ Stop recording" if eng.stats["recording"] else "⏺ Record", use_container_width=True):
        _, name = eng.toggle_record()
        if name:
            st.session_state["rec"] = (config.RECORD_DIR / name).read_bytes()
            st.session_state["rec_name"] = name
    if st.session_state.get("rec"):
        st.download_button("⬇ Download recording", st.session_state["rec"], st.session_state.get("rec_name", "recording.mp4"),
                           "video/mp4", use_container_width=True)


# ------------------------------------------------------------------ pages
hero()
tab_live, tab_video, tab_photo, tab_hist, tab_about = st.tabs(
    ["🎥 Live camera", "📹 Video studio", "📸 Photo detect", "📊 History", "👤 Founder & About"])

with tab_live:
    live_view()

with tab_video:
    st.caption(f"Upload a video (max {st.get_option('server.maxUploadSize')} MB, first {config.MAX_VIDEO_FRAMES} frames). "
               "You get an annotated video, IN/OUT counts and a summary of everything found.")
    up = st.file_uploader("Video file", type=[e.strip(".") for e in sorted(config.VIDEO_EXTS)])
    skip = st.select_slider("Analyse every Nth frame (higher = faster)", [1, 2, 3, 4, 5], value=2)
    if up and st.button("▶ Process video", type="primary"):
        tmp = Path(tempfile.mkdtemp(prefix="vp_up_")) / f"input{Path(up.name).suffix.lower()}"
        tmp.write_bytes(up.getvalue())
        bar = st.progress(0.0, text="Starting...")
        v_metrics, preview = st.empty(), st.empty()
        try:
            with st.spinner("Loading model and processing..."):
                st.session_state["video_result"] = process_video(tmp, skip, preview, bar, v_metrics)
        except Exception as e:
            st.session_state.pop("video_result", None)
            st.error(f"Could not process this video: {e}")
        finally:
            tmp.unlink(missing_ok=True)
    res = st.session_state.get("video_result")
    if res:
        st.success(f"Processed {res['frames']} frames." + (" (Truncated to the frame limit.)" if res["truncated"] else ""))
        a, b, c = st.columns(3)
        a.metric("Unique objects tracked", res["unique"]); b.metric("Crossed IN", res["inn"]); c.metric("Crossed OUT", res["out"])
        if res["totals"]:
            st.bar_chart(pd.Series(res["totals"], name="objects").sort_values(ascending=False))
        if res["h264"]:
            st.video(res["data"])
        else:
            st.info("Preview unavailable (no H.264 encoder). Download the file to watch it.")
        st.download_button("⬇ Download annotated video", res["data"], "visionpro_annotated.mp4", "video/mp4")

with tab_photo:
    st.caption("Take a photo or upload an image. On a phone, the upload button also opens the camera (use the back camera for objects).")
    shot = st.camera_input("Camera")
    img_up = st.file_uploader("...or upload / take a photo", type=["jpg", "jpeg", "png", "webp"])
    data = shot.getvalue() if shot else (img_up.getvalue() if img_up else None)
    if data:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error("That image could not be read.")
        else:
            with st.spinner("Detecting..."):
                eng.new_scene()
                eng.step(frame)
            st.image(eng.last_frame, channels="BGR", output_format="JPEG", use_container_width=True)
            st.write(", ".join(f"**{n}** x{c}" for n, c in eng.stats["per_class"].items())
                     or "Nothing found. Lower the confidence or raise the detection size in the sidebar.")
            detections_table()
            st.download_button("⬇ Download annotated image", eng.frame_jpg, "visionpro_detection.jpg", "image/jpeg")
            maybe_sample()

with tab_hist:
    if not config.PERSISTENT_STORAGE:
        st.warning("Storage on this deployment is temporary: history resets when the app restarts or redeploys. "
                   "Attach persistent storage and set PERSISTENT_STORAGE=true to keep it.")
    f1, f2, f3 = st.columns([1, 1, 1])
    kind = f1.selectbox("Event type", ["all", "new", "in", "out", "info", "warn"])
    mins = f2.selectbox("Chart window", [60, 360, 1440], format_func=lambda m: f"Last {m // 60} h")
    f3.write("")
    if f3.button("🔄 Refresh", use_container_width=True):
        st.rerun()
    summ = db.summary()
    for row in ([("Objects tracked", summ["new"]), ("Crossed IN", summ["inn"]), ("Crossed OUT", summ["out"])],
                [("Peak in frame", summ["peak"]), ("Average FPS", summ["avg_fps"])]):
        for col, (label, val) in zip(st.columns(3), row):
            col.metric(label, val)
    samples = db.samples(mins)
    if samples:
        sdf = pd.DataFrame(samples)
        sdf["time"] = pd.to_datetime(sdf["ts"], unit="s")
        st.line_chart(sdf.set_index("time")["objects"], height=220)
    events = db.events("" if kind == "all" else kind, 300)
    if events:
        edf = pd.DataFrame(events)
        edf["time"] = pd.to_datetime(edf["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(edf[["time", "kind", "msg"]], use_container_width=True, hide_index=True, height=300)
    else:
        st.info("No events yet. Run a video or the camera and they will appear here.")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Export events CSV", csv_bytes(db.events("" if kind == "all" else kind, 100000), ["ts", "kind", "msg"]),
                       "visionpro_events.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇ Export stats CSV", csv_bytes(db.samples(60 * 24 * 30), ["ts", "objects", "in_count", "out_count", "fps"]),
                       "visionpro_stats.csv", "text/csv", use_container_width=True)

with tab_about:
    c_img, c_txt = st.columns([1, 1.5], gap="large", vertical_alignment="center")
    with c_img:
        founder_card(340)
    with c_txt:
        st.markdown("## Ritesh Kumar Singh")
        st.markdown("**Founder & CEO, VisionPro** - an AI computer-vision studio that finds, tracks and counts "
                    "real-world objects from a camera, a photo or a video.")
        links = [(k, os.environ.get(v)) for k, v in (("LinkedIn", "CONTACT_LINKEDIN"), ("GitHub", "CONTACT_GITHUB"))]
        cols = st.columns(3)
        for col, (label, url) in zip(cols, links):
            if url:
                col.link_button(label, url, use_container_width=True)
        if os.environ.get("CONTACT_EMAIL"):
            cols[2].link_button("Email", "mailto:" + os.environ["CONTACT_EMAIL"], use_container_width=True)
    st.markdown("### What VisionPro does")
    g1, g2 = st.columns(2)
    g1.markdown("""<div class="card"><h4>Detect almost anything</h4>Open-vocabulary YOLOE recognises 4,585 object types
    (pen, needle, bottle, book, shoes, laptop...) instead of a fixed list of 80, with pixel-accurate segmentation masks.</div>
    <div class="card"><h4>Track and count</h4>ByteTrack gives every object a persistent ID, so people, vehicles or items are counted once as they
    cross a line (IN / OUT), with motion trails and a heatmap.</div>""", unsafe_allow_html=True)
    g2.markdown("""<div class="card"><h4>Works on phones</h4>Live camera with front and back lens, photo capture and video upload,
    all in the browser. No app install and no special streaming server.</div>
    <div class="card"><h4>Analytics built in</h4>Every event is stored in SQLite with charts, retention and CSV export, behind a secure login.</div>""",
                unsafe_allow_html=True)
    st.markdown("### Real-world uses")
    st.markdown("- **Retail:** count visitors in and out, find popular zones with the heatmap\n"
                "- **Inventory and warehouse:** list and count objects in a shelf photo or video\n"
                "- **Traffic and parking:** count vehicles crossing a line\n"
                "- **Accessibility and education:** name the objects in front of a camera")
    with st.expander("How it works (system design)"):
        st.markdown("""
1. **Capture:** the browser reads the camera (`getUserMedia`, front or back) and sends compressed frames over the Streamlit connection.
2. **Detect:** YOLOE (YOLO11-based, prompt-free vocabulary of 4,585 classes) returns boxes, classes and masks on CPU.
3. **Track:** ByteTrack links detections across frames, so each object keeps an ID.
4. **Count and analyse:** a virtual line turns ID movement into IN/OUT counts; events go through a queue into SQLite.
5. **Serve:** Streamlit fragments update only the live panel per frame; the app runs in Docker or on Streamlit Cloud.

**Tech stack:** Python, PyTorch (CPU), Ultralytics YOLOE / YOLO11, ByteTrack, OpenCV, Streamlit, SQLite, Docker.
""")

st.markdown('<div class="foot">&copy; 2026 VisionPro by <b>Ritesh Kumar Singh</b>, Owner &amp; CEO</div>', unsafe_allow_html=True)
