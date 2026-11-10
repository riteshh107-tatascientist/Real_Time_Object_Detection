"""SQLite storage for users, detection events and periodic stat samples.

Writes from the detection thread go through a queue so inference is never blocked by disk I/O.
The database file and its tables are created automatically if they are missing.
"""
import atexit
import hashlib
import hmac
import logging
import os
import queue
import sqlite3
import threading
import time
from contextlib import closing

import config

log = logging.getLogger("visionpro.db")
_q = queue.Queue(maxsize=5000)
_writer = None
_stop = threading.Event()


def generate_password_hash(password, iterations=240_000):
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2${iterations}${salt.hex()}${h.hex()}"


def check_password_hash(stored, password):
    try:
        _, it, salt, h = stored.split("$")
        calc = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(calc.hex(), h)
    except (ValueError, AttributeError):
        return False


def conn():
    c = sqlite3.connect(str(config.DB_PATH), timeout=10)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    c.execute("PRAGMA busy_timeout=5000")
    return c


def init():
    """Create folders, tables and indexes. Safe to call on every start."""
    config.ensure_dirs()
    with closing(conn()) as c, c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw TEXT NOT NULL, role TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, msg TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY, ts REAL NOT NULL, objects INT, in_count INT, out_count INT, fps REAL);
        CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);
        CREATE INDEX IF NOT EXISTS ix_samples_ts ON samples(ts);""")
    log.info("Database ready at %s (persistent=%s)", config.DB_PATH, config.PERSISTENT_STORAGE)


def ensure_admin(username, password):
    """Single-admin model: the admin always matches the ADMIN_USERNAME / ADMIN_PASSWORD env vars."""
    with closing(conn()) as c, c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            c.execute("INSERT INTO users(username,pw,role) VALUES(?,?,'admin')",
                      (username, generate_password_hash(password)))
        elif not check_password_hash(row["pw"], password):
            c.execute("UPDATE users SET pw=? WHERE username=?", (generate_password_hash(password), username))
        c.execute("DELETE FROM users WHERE username<>?", (username,))  # no stale credentials survive a rename


def check_user(username, password):
    with closing(conn()) as c:
        r = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return r if r and check_password_hash(r["pw"], password) else None


# ---------- background writer ----------
def add_event(msg, kind):
    try:
        _q.put_nowait(("e", (time.time(), kind, msg)))
    except queue.Full:
        pass


def add_sample(s):
    try:
        _q.put_nowait(("s", (time.time(), s["objects"], s["in_count"], s["out_count"], s["fps"])))
    except queue.Full:
        pass


def _flush():
    ev, sm = [], []
    while True:
        try:
            kind, row = _q.get_nowait()
        except queue.Empty:
            break
        (ev if kind == "e" else sm).append(row)
    if not (ev or sm):
        return
    try:
        with closing(conn()) as c, c:
            c.executemany("INSERT INTO events(ts,kind,msg) VALUES(?,?,?)", ev)
            c.executemany("INSERT INTO samples(ts,objects,in_count,out_count,fps) VALUES(?,?,?,?,?)", sm)
    except sqlite3.Error as e:
        log.warning("History write failed: %s", e)


def _run():
    last_prune = 0
    while not _stop.is_set():
        _stop.wait(1.0)
        _flush()
        if time.time() - last_prune > 3600:
            last_prune = time.time()
            prune()


def start_writer():
    global _writer
    if _writer is None:
        _writer = threading.Thread(target=_run, name="db-writer", daemon=True)
        _writer.start()
        atexit.register(stop_writer)


def stop_writer():
    _stop.set()
    _flush()


def prune(days=None):
    cutoff = time.time() - (days or config.RETENTION_DAYS) * 86400
    try:
        with closing(conn()) as c, c:
            c.execute("DELETE FROM events WHERE ts<?", (cutoff,))
            c.execute("DELETE FROM samples WHERE ts<?", (cutoff,))
    except sqlite3.Error as e:
        log.warning("Prune failed: %s", e)


# ---------- queries ----------
def events(kind="", limit=200):
    q, a = "SELECT ts,kind,msg FROM events", []
    if kind:
        q += " WHERE kind=?"
        a.append(kind)
    with closing(conn()) as c:
        return [dict(r) for r in c.execute(q + " ORDER BY ts DESC LIMIT ?", a + [limit])]


def samples(minutes=60):
    with closing(conn()) as c:
        return [dict(r) for r in c.execute(
            "SELECT ts,objects,in_count,out_count,fps FROM samples WHERE ts>? ORDER BY ts",
            (time.time() - minutes * 60,))]


def summary():
    with closing(conn()) as c:
        k = {r["kind"]: r["n"] for r in c.execute("SELECT kind,COUNT(*) n FROM events GROUP BY kind")}
        p = c.execute("SELECT COALESCE(MAX(objects),0) m, COALESCE(AVG(fps),0) f FROM samples").fetchone()
    return dict(new=k.get("new", 0), inn=k.get("in", 0), out=k.get("out", 0),
                peak=p["m"], avg_fps=round(p["f"], 1))
