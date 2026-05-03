"""Rocksmith Web — FastAPI backend serving highway viewer + library."""

import asyncio
import json
import logging
import os
import sys
import tempfile
import shutil
from pathlib import Path

from logging_setup import configure_logging
configure_logging()

log = logging.getLogger("slopsmith.server")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from psarc import unpack_psarc, read_psarc_entries
from song import load_song, phrase_to_wire, arrangement_string_count
from audio import find_wem_files, convert_wem
from tunings import tuning_name
import sloppak as sloppak_mod

import concurrent.futures
import inspect
import re
import sqlite3
import threading
import time
import traceback
import uuid
import warnings
import xml.etree.ElementTree as ET

from fastapi import Request

app = FastAPI(title="Rocksmith Web")

# Plugins that maintain session stores can register a cleanup callback here.
# The demo-mode janitor calls every registered hook once per hour so stale
# sessions are swept without the core needing to know plugin internals.
_DEMO_JANITOR_HOOKS: list = []
_DEMO_JANITOR_HOOKS_LOCK = threading.Lock()
_DEMO_JANITOR_STARTED = False
_DEMO_JANITOR_STOP = threading.Event()
_DEMO_JANITOR_THREAD: threading.Thread | None = None


def register_demo_janitor_hook(fn) -> None:
    """Register a zero-argument callable to be invoked hourly by the demo
    janitor.  Plugins call this from their ``setup(app, context)`` when they
    want to participate in session cleanup under demo mode.

    The callable must accept no required arguments.  Async (coroutine)
    functions are rejected: the janitor runs in a plain thread and cannot
    await coroutines.
    """
    if not callable(fn):
        raise TypeError(
            f"register_demo_janitor_hook expects a callable, got {type(fn).__name__!r}"
        )
    # Reject coroutine functions — check both the callable itself and its
    # __call__ method so objects with an async __call__ (e.g. class instances,
    # functools.partial wrappers around async functions) are also caught.
    _call = getattr(fn, "__call__", None)
    if inspect.iscoroutinefunction(fn) or (
        _call is not None and inspect.iscoroutinefunction(_call)
    ):
        raise TypeError(
            "register_demo_janitor_hook does not accept async functions; "
            "the janitor runs in a plain thread and cannot await coroutines"
        )
    # Validate that the callable accepts zero required arguments so it won't
    # crash at sweep time (hourly, far from the registration site).
    try:
        sig = inspect.signature(fn)
    except ValueError:
        # inspect.signature() raises ValueError for built-in C callables whose
        # signature cannot be determined.  Accept them as-is; if they fail at
        # runtime the janitor will catch and log the exception.
        pass
    else:
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]
        if required:
            raise TypeError(
                f"register_demo_janitor_hook expects a zero-argument callable; "
                f"{fn!r} has {len(required)} required parameter(s): "
                + ", ".join(p.name for p in required)
            )
    with _DEMO_JANITOR_HOOKS_LOCK:
        _DEMO_JANITOR_HOOKS.append(fn)


def _run_janitor_hook(hook) -> None:
    """Run a single janitor hook inline, swallowing and logging any exception.

    If the hook returns an awaitable (e.g. a coroutine slipped through the
    async-function guard), the coroutine is closed immediately to avoid
    ``RuntimeWarning: coroutine was never awaited`` noise, and a warning is
    emitted so the plugin author knows to fix their hook.
    """
    try:
        result = hook()
    except Exception:
        traceback.print_exc()
        return
    if inspect.iscoroutine(result):
        # A coroutine slipped through the async-function guard (e.g. via a
        # wrapper/partial).  Close it to suppress "coroutine never awaited",
        # then warn so the plugin author knows to fix their hook.
        try:
            result.close()
        except Exception:
            traceback.print_exc()
        warnings.warn(
            f"janitor hook {hook!r} returned a coroutine; "
            "hooks must be plain synchronous callables — "
            "register_demo_janitor_hook does not accept async functions",
            RuntimeWarning,
            stacklevel=1,
        )
    elif inspect.isawaitable(result):
        # Future/Task: no .close() method; just warn and leave it alone.
        warnings.warn(
            f"janitor hook {hook!r} returned an awaitable (Future/Task); "
            "hooks must be plain synchronous callables",
            RuntimeWarning,
            stacklevel=1,
        )


_DEMO_BLOCKED: list[tuple[str, re.Pattern]] = [
    ("POST",   re.compile(r"^/api/settings$")),
    ("POST",   re.compile(r"^/api/settings/import$")),
    ("POST",   re.compile(r"^/api/rescan$")),
    ("POST",   re.compile(r"^/api/rescan/full$")),
    ("POST",   re.compile(r"^/api/favorites/toggle$")),
    ("POST",   re.compile(r"^/api/loops$")),
    ("DELETE", re.compile(r"^/api/loops/[^/]+$")),
    ("POST",   re.compile(r"^/api/song/.*/meta$")),
    ("POST",   re.compile(r"^/api/song/.*/art/upload$")),
    ("GET",    re.compile(r"^/api/plugins/updates$")),
    ("POST",   re.compile(r"^/api/plugins/[^/]+/update$")),
    ("POST",   re.compile(r"^/api/plugins/editor/save$")),
    ("POST",   re.compile(r"^/api/plugins/editor/build$")),
    ("POST",   re.compile(r"^/api/plugins/editor/upload-art$")),
    ("POST",   re.compile(r"^/api/plugins/editor/upload-audio$")),
    ("POST",   re.compile(r"^/api/plugins/editor/youtube-audio$")),
    ("POST",   re.compile(r"^/api/plugins/editor/import-gp$")),
    ("POST",   re.compile(r"^/api/plugins/editor/import-midi$")),
    ("POST",   re.compile(r"^/api/plugins/lyrics_karaoke/align$")),
    ("POST",   re.compile(r"^/api/plugins/lyrics_karaoke/generate-pitch$")),
    ("POST",   re.compile(r"^/api/plugins/lyrics_karaoke/save-lyrics$")),
    ("POST",   re.compile(r"^/api/plugins/lyrics_sync/align$")),
    ("POST",   re.compile(r"^/api/plugins/lyrics_sync/save$")),
    ("POST",   re.compile(r"^/api/plugins/studio/sessions/[^/]+/extract-drums$")),
]


@app.middleware("http")
async def _demo_mode_guard(request: Request, call_next):
    if os.environ.get("SLOPSMITH_DEMO_MODE") == "1":
        path = request.url.path
        for method, pattern in _DEMO_BLOCKED:
            if request.method == method and pattern.match(path):
                return JSONResponse({"error": "demo mode: read-only"}, status_code=403)
        response = await call_next(request)
        if request.method == "GET" and path == "/" and "slopsmith_demo_session" not in request.cookies:
            forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
            is_secure = request.url.scheme == "https" or forwarded_proto.lower() == "https"
            response.set_cookie(
                "slopsmith_demo_session", str(uuid.uuid4()),
                max_age=86400, httponly=True, samesite="lax",
                secure=is_secure,
            )
        return response
    return await call_next(request)

from asgi_correlation_id import CorrelationIdMiddleware

# validator=None accepts any non-empty inbound X-Request-ID value, including
# opaque proxy-generated hex strings, not just RFC-4122 UUIDs.
app.add_middleware(CorrelationIdMiddleware, validator=None)

STATIC_DIR = Path(__file__).parent / "static"
try:
    STATIC_DIR.mkdir(exist_ok=True)
except OSError:
    pass  # Read-only in packaged installs

# Distinguish "env not set / empty" from "explicitly set". Path("") collapses
# to Path(".") so we can't recover that signal after the cast — capture the
# raw env-var string up front and let _get_dlc_dir() consult both. This way
# `DLC_DIR=.` remains a valid opt-in for cwd while `DLC_DIR=""` (or unset)
# falls through to the config.json fallback.
_DLC_DIR_ENV = os.environ.get("DLC_DIR", "").strip()
DLC_DIR = Path(_DLC_DIR_ENV) if _DLC_DIR_ENV else Path("")
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", str(Path.home() / ".local" / "share" / "rocksmith-cdlc")))

# Writable cache directories (use CONFIG_DIR, not STATIC_DIR which may be read-only)
ART_CACHE_DIR = CONFIG_DIR / "art_cache"
AUDIO_CACHE_DIR = CONFIG_DIR / "audio_cache"
SLOPPAK_CACHE_DIR = CONFIG_DIR / "sloppak_cache"


# ── SQLite metadata cache ─────────────────────────────────────────────────────

class MetadataDB:
    def __init__(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = str(CONFIG_DIR / "web_library.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                filename TEXT PRIMARY KEY,
                mtime REAL,
                size INTEGER,
                title TEXT,
                artist TEXT,
                album TEXT,
                year TEXT,
                duration REAL,
                tuning TEXT,
                arrangements TEXT,
                has_lyrics INTEGER DEFAULT 0,
                format TEXT DEFAULT 'psarc',
                stem_count INTEGER DEFAULT 0,
                stem_ids TEXT DEFAULT '[]',
                tuning_name TEXT DEFAULT '',
                tuning_sort_key INTEGER DEFAULT 0
            )
        """)
        # Idempotent migrations for installs that predate each column.
        for ddl in (
            "ALTER TABLE songs ADD COLUMN format TEXT DEFAULT 'psarc'",
            "ALTER TABLE songs ADD COLUMN stem_count INTEGER DEFAULT 0",
            # slopsmith#129: per-stem filter needs the id list, not just count.
            "ALTER TABLE songs ADD COLUMN stem_ids TEXT DEFAULT '[]'",
            # slopsmith#69 + #22: denormalized canonical tuning name + numeric
            # sort key (sum of offsets). The existing `tuning` text column
            # stays — these are caches, repopulated on rescan.
            "ALTER TABLE songs ADD COLUMN tuning_name TEXT DEFAULT ''",
            "ALTER TABLE songs ADD COLUMN tuning_sort_key INTEGER DEFAULT 0",
        ):
            try:
                self.conn.execute(ddl)
            except sqlite3.OperationalError:
                pass
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist COLLATE NOCASE)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_title ON songs(title COLLATE NOCASE)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_tuning_name ON songs(tuning_name COLLATE NOCASE)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_tuning_sort_key ON songs(tuning_sort_key)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_songs_year ON songs(year)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS favorites (filename TEXT PRIMARY KEY)")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS loops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                name TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()
        self._lock = threading.Lock()

    def is_favorite(self, filename: str) -> bool:
        return self.conn.execute("SELECT 1 FROM favorites WHERE filename = ?", (filename,)).fetchone() is not None

    def toggle_favorite(self, filename: str) -> bool:
        """Toggle favorite status. Returns new state."""
        with self._lock:
            if self.is_favorite(filename):
                self.conn.execute("DELETE FROM favorites WHERE filename = ?", (filename,))
                self.conn.commit()
                return False
            else:
                self.conn.execute("INSERT OR IGNORE INTO favorites VALUES (?)", (filename,))
                self.conn.commit()
                return True

    def favorite_set(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT filename FROM favorites").fetchall()}

    def get(self, filename: str, mtime: float, size: int) -> dict | None:
        row = self.conn.execute(
            "SELECT mtime, size, title, artist, album, year, duration, tuning, arrangements, has_lyrics, "
            "format, stem_count, stem_ids, tuning_name, tuning_sort_key "
            "FROM songs WHERE filename = ?", (filename,)
        ).fetchone()
        if row and row[0] == mtime and row[1] == size and row[2]:
            return {
                "title": row[2], "artist": row[3], "album": row[4],
                "year": row[5], "duration": row[6], "tuning": row[7],
                "arrangements": json.loads(row[8]) if row[8] else [],
                "has_lyrics": bool(row[9]),
                "format": row[10] or "psarc",
                "stem_count": int(row[11] or 0),
                "stem_ids": json.loads(row[12]) if row[12] else [],
                "tuning_name": row[13] or "",
                "tuning_sort_key": int(row[14] or 0),
            }
        return None

    def put(self, filename: str, mtime: float, size: int, meta: dict):
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO songs "
                "(filename, mtime, size, title, artist, album, year, duration, tuning, arrangements, "
                "has_lyrics, format, stem_count, stem_ids, tuning_name, tuning_sort_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (filename, mtime, size, meta.get("title", ""), meta.get("artist", ""),
                 meta.get("album", ""), meta.get("year", ""), meta.get("duration", 0),
                 meta.get("tuning", ""), json.dumps(meta.get("arrangements", [])),
                 1 if meta.get("has_lyrics") else 0,
                 meta.get("format", "psarc"),
                 int(meta.get("stem_count", 0) or 0),
                 json.dumps(meta.get("stem_ids", []) or []),
                 meta.get("tuning_name", "") or "",
                 int(meta.get("tuning_sort_key", 0) or 0)),
            )
            self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM songs WHERE title != ''").fetchone()[0]

    def delete_missing(self, current_filenames: set[str]):
        """Remove DB entries for files no longer on disk."""
        with self._lock:
            db_files = {r[0] for r in self.conn.execute("SELECT filename FROM songs").fetchall()}
            stale = db_files - current_filenames
            if stale:
                self.conn.executemany("DELETE FROM songs WHERE filename = ?", [(f,) for f in stale])
                self.conn.commit()
            return len(stale)

    def _estd_set(self) -> set[str]:
        """Get set of filenames that have a retuned variant (_EStd_ or _DropD_) in the DB."""
        rows = self.conn.execute(
            "SELECT filename FROM songs WHERE filename LIKE '%\\_EStd\\_%' ESCAPE '\\' "
            "OR filename LIKE '%\\_DropD\\_%' ESCAPE '\\'"
        ).fetchall()
        originals = set()
        for (fname,) in rows:
            originals.add(fname.replace("_EStd_", "_").replace("_DropD_", "_"))
        return originals

    # Manifest-allowed filter values. Whitelisted before binding so a
    # malformed query string can't push arbitrary text through to SQL —
    # parameters are bound, but capping the input space is still cheap
    # defense-in-depth (see slopsmith#129).
    _ALLOWED_ARRANGEMENT_NAMES = {"Lead", "Rhythm", "Bass", "Combo"}
    # Stem ids match the bare strings sloppak manifests use today —
    # `full`, `guitar`, `bass`, `drums`, `vocals`, `piano`, `other`. The
    # frontend filter UI omits `full` (it's the always-on fallback mix
    # and would match every sloppak), but the server-side whitelist
    # keeps it so a hand-rolled API client can still ask for it.
    _ALLOWED_STEM_IDS = {"full", "guitar", "bass", "drums", "vocals", "piano", "other"}

    def _build_where(self, q: str = "", favorites_only: bool = False,
                     format_filter: str = "",
                     arrangements_has: list[str] | None = None,
                     arrangements_lacks: list[str] | None = None,
                     stems_has: list[str] | None = None,
                     stems_lacks: list[str] | None = None,
                     has_lyrics: int | None = None,
                     tunings: list[str] | None = None) -> tuple[str, list]:
        """Shared WHERE-clause builder for query_page / query_artists /
        query_stats. Returns (where_sql, params). Leading 'WHERE' is
        included so callers paste it directly. See slopsmith#129/#69.
        """
        where = "WHERE title != ''"
        params: list = []
        if favorites_only:
            where += " AND filename IN (SELECT filename FROM favorites)"
        if format_filter:
            where += " AND format = ?"
            params.append(format_filter)
        if q:
            where += " AND (title LIKE ? COLLATE NOCASE OR artist LIKE ? COLLATE NOCASE OR album LIKE ? COLLATE NOCASE)"
            params += [f"%{q}%"] * 3
        # arrangements_has: OR within axis (any-of). Uses JSON1's
        # json_each which yields one row per arrangement, then matches
        # the `name` field. The whole subquery is wrapped in EXISTS so
        # we don't multiply rows in the outer SELECT.
        arr_has = [a for a in (arrangements_has or []) if a in self._ALLOWED_ARRANGEMENT_NAMES]
        if arr_has:
            placeholders = ",".join(["?"] * len(arr_has))
            where += (" AND EXISTS (SELECT 1 FROM json_each(songs.arrangements) "
                      f"WHERE json_extract(value, '$.name') IN ({placeholders}))")
            params += arr_has
        arr_lacks = [a for a in (arrangements_lacks or []) if a in self._ALLOWED_ARRANGEMENT_NAMES]
        if arr_lacks:
            placeholders = ",".join(["?"] * len(arr_lacks))
            where += (" AND NOT EXISTS (SELECT 1 FROM json_each(songs.arrangements) "
                      f"WHERE json_extract(value, '$.name') IN ({placeholders}))")
            params += arr_lacks
        stems_h = [s for s in (stems_has or []) if s in self._ALLOWED_STEM_IDS]
        if stems_h:
            placeholders = ",".join(["?"] * len(stems_h))
            where += (" AND EXISTS (SELECT 1 FROM json_each(songs.stem_ids) "
                      f"WHERE value IN ({placeholders}))")
            params += stems_h
        stems_l = [s for s in (stems_lacks or []) if s in self._ALLOWED_STEM_IDS]
        if stems_l:
            placeholders = ",".join(["?"] * len(stems_l))
            where += (" AND NOT EXISTS (SELECT 1 FROM json_each(songs.stem_ids) "
                      f"WHERE value IN ({placeholders}))")
            params += stems_l
        if has_lyrics in (0, 1):
            where += " AND has_lyrics = ?"
            params.append(has_lyrics)
        if tunings:
            # Keep the input cap conservative (32) so a hostile caller
            # can't blow out the parameter list. Real tuning sets in the
            # wild number in the low double digits.
            tn = [t for t in tunings if isinstance(t, str) and t][:32]
            if tn:
                placeholders = ",".join(["?"] * len(tn))
                where += f" AND tuning_name COLLATE NOCASE IN ({placeholders})"
                params += tn
        return where, params

    def query_page(self, q: str = "", page: int = 0, size: int = 24,
                   sort: str = "artist", direction: str = "asc",
                   favorites_only: bool = False,
                   format_filter: str = "",
                   arrangements_has: list[str] | None = None,
                   arrangements_lacks: list[str] | None = None,
                   stems_has: list[str] | None = None,
                   stems_lacks: list[str] | None = None,
                   has_lyrics: int | None = None,
                   tunings: list[str] | None = None) -> tuple[list[dict], int]:
        """Server-side paginated search. Returns (songs, total_count)."""
        where, params = self._build_where(
            q=q, favorites_only=favorites_only, format_filter=format_filter,
            arrangements_has=arrangements_has, arrangements_lacks=arrangements_lacks,
            stems_has=stems_has, stems_lacks=stems_lacks,
            has_lyrics=has_lyrics, tunings=tunings,
        )

        sort_map = {
            "artist": "artist COLLATE NOCASE", "artist-desc": "artist COLLATE NOCASE DESC",
            "title": "title COLLATE NOCASE", "title-desc": "title COLLATE NOCASE DESC",
            "recent": "mtime DESC",
            # Tuning sort uses musical distance from E Standard
            # (slopsmith#22 — was alphabetical). `tuning_sort_key` is
            # the sum of per-string offsets, so |sort_key| is the
            # magnitude of the down/up-tune. ABS ascending puts E
            # Standard (0) first, then ±2 (Drop D, F Standard), then
            # ±6 (Eb Standard, F# Standard), and so on. Within a
            # magnitude tier we break ties by signed key ASC so the
            # negative (down-tuned) variant comes before the positive
            # (up-tuned) one — Eb Standard before F Standard, matching
            # how Rocksmith groups its tuning list. Final tiebreak by
            # name keeps the order fully deterministic.
            #
            # Leading term pushes pre-migration / unscanned rows to
            # the bottom — without it ABS(0) collides with E
            # Standard's 0 and unindexed rows would sort first.
            # COALESCE on every column the clause references guards
            # against NULL values — SQLite's literal-constant ADD
            # COLUMN does backfill on most versions, but raw SQL
            # inserts that bypass `put()`, edge-case migration paths,
            # or future code that writes None could still leave NULLs
            # behind, and a NULL `tuning_name` in `(tuning_name = '')`
            # evaluates to NULL itself (which sorts ahead of 0 in
            # ASC), defeating the push-to-bottom intent.
            "tuning": (
                "(COALESCE(tuning_name, '') = '') ASC, "
                "ABS(COALESCE(tuning_sort_key, 0)), "
                "COALESCE(tuning_sort_key, 0) ASC, "
                "COALESCE(tuning_name, '') COLLATE NOCASE"
            ),
            # Year sort (slopsmith#128). Empty-year rows pushed to the
            # bottom for both directions; otherwise CAST so '2010' >
            # '2005' rather than alphabetic.
            "year": "(year = '') ASC, CAST(year AS INTEGER) ASC",
            "year-desc": "(year = '') ASC, CAST(year AS INTEGER) DESC",
        }
        order = sort_map.get(sort, "artist COLLATE NOCASE")
        # Legacy `dir=desc` toggle: only safe to append on simple sort
        # clauses that don't already encode a direction. Compound /
        # multi-term entries above (tuning, year, year-desc) bake their
        # ASC/DESC into the clause, so a global ` DESC` append would
        # produce invalid SQL like `CAST(year AS INTEGER) ASC DESC`.
        # Skip the append in that case — clients flipping direction on
        # those sorts use the explicit `-desc` sort key instead.
        if direction == "desc" and " ASC" not in order and " DESC" not in order:
            order += " DESC"

        total = self.conn.execute(f"SELECT COUNT(*) FROM songs {where}", params).fetchone()[0]
        rows = self.conn.execute(
            f"SELECT filename, title, artist, album, year, duration, tuning, arrangements, has_lyrics, mtime, "
            f"format, stem_count, stem_ids, tuning_name "
            f"FROM songs {where} ORDER BY {order} LIMIT ? OFFSET ?",
            params + [size, page * size]
        ).fetchall()

        estd = self._estd_set()
        favs = self.favorite_set()
        songs = []
        for r in rows:
            songs.append({
                "filename": r[0], "title": r[1], "artist": r[2], "album": r[3],
                "year": r[4], "duration": r[5], "tuning": r[6],
                "arrangements": json.loads(r[7]) if r[7] else [],
                "has_lyrics": bool(r[8]), "mtime": r[9],
                "format": r[10] or "psarc",
                "stem_count": int(r[11] or 0),
                "stem_ids": json.loads(r[12]) if r[12] else [],
                "tuning_name": r[13] or "",
                "has_estd": r[0] in estd, "favorite": r[0] in favs,
            })
        return songs, total

    def query_artists(self, letter: str = "", q: str = "",
                      favorites_only: bool = False,
                      page: int = 0, size: int = 50,
                      format_filter: str = "",
                      arrangements_has: list[str] | None = None,
                      arrangements_lacks: list[str] | None = None,
                      stems_has: list[str] | None = None,
                      stems_lacks: list[str] | None = None,
                      has_lyrics: int | None = None,
                      tunings: list[str] | None = None) -> tuple[list[dict], int]:
        """Get artists grouped by letter with their albums and songs. Returns (artists, total_artists)."""
        where, params = self._build_where(
            q=q, favorites_only=favorites_only, format_filter=format_filter,
            arrangements_has=arrangements_has, arrangements_lacks=arrangements_lacks,
            stems_has=stems_has, stems_lacks=stems_lacks,
            has_lyrics=has_lyrics, tunings=tunings,
        )
        if letter == "#":
            where += " AND artist NOT GLOB '[A-Za-z]*'"
        elif letter:
            where += " AND UPPER(SUBSTR(artist, 1, 1)) = ?"
            params.append(letter.upper())

        # Get paginated distinct artists
        total_artists = self.conn.execute(
            f"SELECT COUNT(DISTINCT artist COLLATE NOCASE) FROM songs {where}", params
        ).fetchone()[0]

        artist_rows = self.conn.execute(
            f"SELECT DISTINCT artist COLLATE NOCASE as a FROM songs {where} ORDER BY a LIMIT ? OFFSET ?",
            params + [size, page * size]
        ).fetchall()
        artist_names = [r[0] for r in artist_rows]

        if not artist_names:
            return [], total_artists

        # Fetch songs for these artists only
        placeholders = ",".join(["?"] * len(artist_names))
        song_where = f"{where} AND artist COLLATE NOCASE IN ({placeholders})"
        song_params = params + artist_names

        rows = self.conn.execute(
            f"SELECT filename, title, artist, album, year, duration, tuning, arrangements, has_lyrics, "
            f"format, stem_count, stem_ids, tuning_name "
            f"FROM songs {song_where} ORDER BY artist COLLATE NOCASE, album COLLATE NOCASE, title COLLATE NOCASE",
            song_params
        ).fetchall()

        # Group into artist -> album -> songs
        from collections import OrderedDict
        estd = self._estd_set()
        favs = self.favorite_set()
        artists = OrderedDict()
        for r in rows:
            artist = r[2] or "Unknown Artist"
            album = r[3] or "Unknown Album"
            akey = artist.lower()
            if akey not in artists:
                artists[akey] = {"name": artist, "albums": OrderedDict()}
            bkey = album.lower()
            if bkey not in artists[akey]["albums"]:
                artists[akey]["albums"][bkey] = {"name": album, "songs": []}
            artists[akey]["albums"][bkey]["songs"].append({
                "filename": r[0], "title": r[1], "artist": r[2], "album": r[3],
                "year": r[4], "duration": r[5], "tuning": r[6],
                "arrangements": json.loads(r[7]) if r[7] else [],
                "has_lyrics": bool(r[8]),
                "format": r[9] or "psarc",
                "stem_count": int(r[10] or 0),
                "stem_ids": json.loads(r[11]) if r[11] else [],
                "tuning_name": r[12] or "",
                "has_estd": r[0] in estd,
                "favorite": r[0] in favs,
            })

        # Pick most common name variant per artist/album
        result = []
        for akey, aval in artists.items():
            albums = []
            for bkey, bval in aval["albums"].items():
                albums.append({"name": bval["name"], "songs": bval["songs"]})
            result.append({"name": aval["name"], "album_count": len(albums),
                           "song_count": sum(len(a["songs"]) for a in albums), "albums": albums})
        return result, total_artists

    def query_stats(self, favorites_only: bool = False,
                    q: str = "", format_filter: str = "",
                    arrangements_has: list[str] | None = None,
                    arrangements_lacks: list[str] | None = None,
                    stems_has: list[str] | None = None,
                    stems_lacks: list[str] | None = None,
                    has_lyrics: int | None = None,
                    tunings: list[str] | None = None) -> dict:
        """Aggregate stats for the letter bar. Accepts the same filter
        params as query_page so the letter counts stay synchronized
        with the grid when filters are active."""
        where, params = self._build_where(
            q=q, favorites_only=favorites_only, format_filter=format_filter,
            arrangements_has=arrangements_has, arrangements_lacks=arrangements_lacks,
            stems_has=stems_has, stems_lacks=stems_lacks,
            has_lyrics=has_lyrics, tunings=tunings,
        )
        total = self.conn.execute(f"SELECT COUNT(*) FROM songs {where}", params).fetchone()[0]
        # NOCASE collation here mirrors `query_artists` and the per-
        # letter `COUNT(DISTINCT artist COLLATE NOCASE)` below — without
        # it, an artist stored under two different casings would inflate
        # `total_artists` against the letter-bar breakdown the UI
        # renders next to it.
        artist_count = self.conn.execute(
            f"SELECT COUNT(DISTINCT artist COLLATE NOCASE) FROM songs {where}", params
        ).fetchone()[0]
        rows = self.conn.execute(
            f"SELECT UPPER(SUBSTR(artist, 1, 1)) as letter, COUNT(DISTINCT artist COLLATE NOCASE) "
            f"FROM songs {where} GROUP BY letter", params
        ).fetchall()
        letters = {}
        for letter, count in rows:
            if letter and letter.isalpha():
                letters[letter] = count
            else:
                letters["#"] = letters.get("#", 0) + count
        return {"total_songs": total, "total_artists": artist_count, "letters": letters}


meta_db = MetadataDB()


def _get_dlc_dir() -> Path | None:
    # Only consider DLC_DIR if the env var was non-empty. `Path("")` collapses
    # to `.` and reports `.is_dir() == True`, which would silently shadow the
    # config.json fallback. Checking the raw env string preserves
    # `DLC_DIR=.` as a valid opt-in for cwd while keeping unset/empty out.
    if _DLC_DIR_ENV and DLC_DIR.is_dir():
        return DLC_DIR
    config_file = CONFIG_DIR / "config.json"
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text())
            raw = str(cfg.get("dlc_dir", "")).strip()
            if raw:
                p = Path(raw)
                if p.is_dir():
                    return p
        except Exception:
            pass
    return None


# ── Background metadata scan ──────────────────────────────────────────────────

def _extract_meta_fast(psarc_path: Path) -> dict:
    """Extract metadata from a PSARC using in-memory reading (no disk I/O)."""
    files = read_psarc_entries(str(psarc_path), ["*.json", "*.xml", "*vocals*.sng"])

    title = artist = album = year = ""
    duration = 0.0
    tuning = "E Standard"
    # Track the offsets for the tuning we ultimately keep so we can
    # compute tuning_sort_key (#22) without re-deriving it from the
    # name. Defaults to E Standard offsets.
    tuning_offsets: list[int] = [0] * 6
    _tuning_from_guitar = False
    arrangements = []
    has_lyrics = False
    arr_index = 0

    # Parse manifest JSONs for metadata + arrangement info
    for path, data in sorted(files.items()):
        if not path.lower().endswith(".json"):
            continue
        try:
            jdata = json.loads(data)
            entries = jdata.get("Entries") or {}
            for k, v in entries.items():
                attrs = v.get("Attributes") or {}
                arr_name = attrs.get("ArrangementName", "")
                if arr_name in ("Vocals", "ShowLights", "JVocals"):
                    continue
                if not title:
                    title = attrs.get("SongName", "")
                    artist = attrs.get("ArtistName", "")
                    album = attrs.get("AlbumName", "")
                    yr = attrs.get("SongYear")
                    year = str(yr) if yr else ""
                    sl = attrs.get("SongLength")
                    if sl:
                        try: duration = float(sl)
                        except (ValueError, TypeError): pass
                if arr_name:
                    # Get tuning - prefer guitar arrangements over bass
                    tun = attrs.get("Tuning")
                    if tun and isinstance(tun, dict):
                        offsets = [tun.get(f"string{i}", 0) for i in range(6)]
                        tun_name = tuning_name(offsets)
                        is_guitar = arr_name in ("Lead", "Rhythm", "Combo")
                        if tuning == "E Standard" or (is_guitar and not _tuning_from_guitar):
                            tuning = tun_name
                            tuning_offsets = offsets
                            if is_guitar:
                                _tuning_from_guitar = True
                    notes = attrs.get("NotesHard", 0) or attrs.get("NotesMedium", 0) or 0
                    arrangements.append({"index": arr_index, "name": arr_name, "notes": notes})
                    arr_index += 1
        except Exception:
            continue

    # Check XMLs for vocals (CDLC), or fall back to vocals SNG (official DLC)
    for path, data in files.items():
        if path.lower().endswith(".xml"):
            try:
                root = ET.fromstring(data)
                if root.tag == "vocals":
                    has_lyrics = True
                    break
            except Exception:
                continue
        elif path.lower().endswith(".sng") and "vocals" in path.lower():
            has_lyrics = True
            break

    # Sort arrangements: Lead > Combo > Rhythm > Bass
    priority = {"Lead": 0, "Combo": 1, "Rhythm": 2, "Bass": 3}
    arrangements.sort(key=lambda a: priority.get(a["name"], 99))
    for i, a in enumerate(arrangements):
        a["index"] = i

    return {
        "title": title, "artist": artist, "album": album, "year": year,
        "duration": duration, "tuning": tuning,
        "arrangements": arrangements, "has_lyrics": has_lyrics,
        # PSARCs have no stems; emit an empty list so the column round-
        # trips uniformly with sloppaks (slopsmith#129).
        "stem_ids": [],
        # Cached tuning fields (slopsmith#22 / #69). The text `tuning`
        # column above stays the source of truth for display; these are
        # the indexable / filterable forms.
        "tuning_name": tuning,
        "tuning_sort_key": sum(tuning_offsets),
    }


def _extract_meta_sloppak(path: Path) -> dict:
    """Extract metadata for a sloppak (file or directory)."""
    meta = sloppak_mod.extract_meta(path)
    offsets = meta.pop("tuning_offsets", None) or [0] * 6
    name = tuning_name(offsets)
    meta["tuning"] = name
    meta["tuning_name"] = name
    meta["tuning_sort_key"] = sum(offsets)
    meta["format"] = "sloppak"
    # `extract_meta` already populates `stem_ids` (slopsmith#129);
    # default to empty for older callers / mocks.
    meta.setdefault("stem_ids", [])
    return meta


def _extract_meta_for_file(psarc_path: Path) -> dict:
    """Extract metadata — dispatches on extension; PSARC path tries fast then falls back."""
    if sloppak_mod.is_sloppak(psarc_path):
        return _extract_meta_sloppak(psarc_path)
    try:
        meta = _extract_meta_fast(psarc_path)
        if meta["title"]:
            return meta
    except Exception:
        pass
    # Fallback: full extraction (handles SNG-only official DLC etc.)
    tmp = tempfile.mkdtemp(prefix="rs_scan_")
    try:
        unpack_psarc(str(psarc_path), tmp)
        song = load_song(tmp)
        tuning = "E Standard"
        tuning_offsets: list[int] = [0] * 6
        if song.arrangements and song.arrangements[0].tuning:
            tuning_offsets = list(song.arrangements[0].tuning)
            tuning = tuning_name(tuning_offsets)
        arrangements = [
            {"index": i, "name": a.name,
             "notes": len(a.notes) + sum(len(c.notes) for c in a.chords)}
            for i, a in enumerate(song.arrangements)
        ]
        has_lyrics = False
        for xf in Path(tmp).rglob("*.xml"):
            try:
                if ET.parse(str(xf)).getroot().tag == "vocals":
                    has_lyrics = True
                    break
            except Exception:
                pass
        return {
            "title": song.title, "artist": song.artist,
            "album": song.album, "year": str(song.year) if song.year else "",
            "duration": song.song_length, "tuning": tuning,
            "arrangements": arrangements, "has_lyrics": has_lyrics,
            "stem_ids": [],
            "tuning_name": tuning,
            "tuning_sort_key": sum(tuning_offsets),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_SCAN_STATUS_INIT = {"running": False, "stage": "idle", "total": 0, "done": 0, "current": "", "error": None}
_scan_status = dict(_SCAN_STATUS_INIT)

_STARTUP_STATUS_INIT = {
    "running": True,
    "phase": "booting",
    "message": "Starting Slopsmith server...",
    "current_plugin": "",
    "loaded": 0,
    "total": 0,
    "error": None,
}
_startup_status = dict(_STARTUP_STATUS_INIT)
_startup_status_lock = threading.Lock()


def _set_startup_status(**updates):
    global _startup_status
    with _startup_status_lock:
        next_status = dict(_startup_status)
        next_status.update(updates)
        _startup_status = next_status


def _get_startup_status():
    with _startup_status_lock:
        return dict(_startup_status)


def _background_scan():
    """Scan all PSARCs and cache metadata on startup. Uses thread pool for parallelism."""
    global _scan_status
    _scan_status = {**_SCAN_STATUS_INIT, "running": True, "stage": "listing"}

    dlc = _get_dlc_dir()
    if not dlc:
        _scan_status = {**_SCAN_STATUS_INIT, "stage": "idle", "error": "DLC folder not configured"}
        print("Scan: no DLC folder configured", flush=True)
        return

    # Listing can fail on macOS without Full Disk Access, or on Docker if the
    # path isn't shared. Report the failure explicitly rather than silently
    # appearing to scan nothing.
    try:
        # Skip RS1 compatibility mega-PSARCs (multi-song, not individually playable)
        psarcs = [f for f in sorted(dlc.rglob("*.psarc"))
                  if f.is_file()
                  and "rs1compatibility" not in f.name.lower()]
        # Sloppaks: match both file (zip) and directory form by suffix.
        sloppaks = [f for f in sorted(dlc.rglob("*.sloppak"))
                    if sloppak_mod.is_sloppak(f)]
    except PermissionError as e:
        msg = (f"Permission denied reading {dlc}. "
               "On macOS: grant Full Disk Access to the app in System Settings → Privacy & Security. "
               "With Docker: share this path in Docker Desktop → Settings → Resources → File Sharing.")
        print(f"Scan failed: {msg} ({e})", flush=True)
        _scan_status = {**_SCAN_STATUS_INIT, "stage": "error", "error": msg}
        return
    except OSError as e:
        print(f"Scan failed listing {dlc}: {e}", flush=True)
        _scan_status = {**_SCAN_STATUS_INIT, "stage": "error", "error": f"Unable to list {dlc}: {e}"}
        return

    all_songs = psarcs + sloppaks
    print(f"Scan: listed {len(psarcs)} PSARCs and {len(sloppaks)} sloppaks in {dlc}", flush=True)

    def _rel(f: Path) -> str:
        # Store the path relative to the DLC root so sub-folders (e.g.
        # dlc/sloppak/foo.sloppak produced by the converter) resolve back
        # correctly later. PSARCs always live directly in dlc/, so this
        # reduces to f.name for them.
        try:
            return f.relative_to(dlc).as_posix()
        except ValueError:
            return f.name

    current_files = {_rel(f) for f in all_songs}

    # Clean up stale DB entries
    stale = meta_db.delete_missing(current_files)
    if stale:
        print(f"Removed {stale} stale DB entries", flush=True)

    # Figure out which need scanning
    to_scan = []
    for f in all_songs:
        stat = f.stat()
        if not meta_db.get(_rel(f), stat.st_mtime, stat.st_size):
            to_scan.append((f, stat))

    if not to_scan:
        _scan_status = {**_SCAN_STATUS_INIT, "stage": "complete"}
        print(f"Scan: nothing new to scan ({len(all_songs)} songs, all cached)", flush=True)
        return

    _scan_status = {**_SCAN_STATUS_INIT, "running": True, "stage": "scanning", "total": len(to_scan)}
    print(f"Library: {len(psarcs)} PSARCs + {len(sloppaks)} sloppaks, {len(all_songs) - len(to_scan)} cached, {len(to_scan)} to scan", flush=True)

    def _scan_one(item):
        f, stat = item
        # Per-file log so users running the server / desktop can see live
        # activity and distinguish a stuck scan from a slow one.
        print(f"  scanning {f.name}", flush=True)
        meta = _extract_meta_for_file(f)
        return _rel(f), stat.st_mtime, stat.st_size, meta

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_scan_one, item): item[0].name for item in to_scan}
        for future in concurrent.futures.as_completed(futures):
            fname = futures[future]
            try:
                name, mtime, size, meta = future.result()
                meta_db.put(name, mtime, size, meta)
            except Exception as e:
                print(f"  Failed: {fname}: {e}", flush=True)
            _scan_status["done"] += 1
            _scan_status["current"] = fname

    print(f"Scan complete: {len(to_scan)} songs cached", flush=True)
    _scan_status = {**_SCAN_STATUS_INIT, "stage": "complete"}


# ── Register plugin API endpoints (lightweight, before app starts) ───────────
from plugins import load_plugins, register_plugin_api
register_plugin_api(app)

# Plugin loading deferred to startup event (see below) to avoid blocking
# server startup when many plugins are installed.


@app.on_event("startup")
async def startup_events():
    # Safety net: re-apply the structlog pipeline in case the server was
    # started directly via `uvicorn server:app` (without main.py).  When
    # running via `python main.py`, configure_logging() was already called
    # before uvicorn.run(..., log_config=None), so uvicorn never calls its
    # own dictConfig() and this call is effectively a no-op.  When running
    # the uvicorn CLI directly, uvicorn applies LOGGING_CONFIG before the
    # ASGI startup hook fires, overwriting the uvicorn* handlers; this call
    # restores them for all messages after "Waiting for application startup".
    configure_logging()

    loop = asyncio.get_running_loop()

    _set_startup_status(
        running=True,
        phase="starting",
        message="Core server ready. Starting plugin loader...",
        error=None,
    )

    plugin_context = {
        "config_dir": CONFIG_DIR,
        "get_dlc_dir": _get_dlc_dir,
        "extract_meta": _extract_meta_for_file,
        "meta_db": meta_db,
        "get_sloppak_cache_dir": lambda: SLOPPAK_CACHE_DIR,
        "register_demo_janitor_hook": register_demo_janitor_hook,
    }

    # Load plugins asynchronously so HTTP routes and the desktop window can
    # come up immediately while heavy plugin imports/install steps continue.
    _sync_mode = os.environ.get("SLOPSMITH_SYNC_STARTUP", "").lower() in {"1", "true", "yes", "on"}

    def _load_plugins_background():
        try:
            def _on_progress(event: dict):
                total = int(event.get("total") or 0)
                loaded = int(event.get("loaded") or 0)
                plugin_id = event.get("plugin_id") or ""
                message = event.get("message") or "Loading plugins..."
                phase = event.get("phase") or "plugins-loading"
                update: dict = dict(
                    running=True,
                    phase=phase,
                    message=message,
                    current_plugin=plugin_id,
                    loaded=loaded,
                    total=total,
                )
                # Only forward the error field when the event carries a real
                # (non-null) error string.  This preserves any previously
                # reported plugin error across the many subsequent non-error
                # progress events that follow — without this guard the final
                # `complete` status would almost always show `error: null`
                # even when a plugin failed requirements or route setup.
                if event.get("error") is not None:
                    update["error"] = event["error"]
                _set_startup_status(**update)

            def _route_setup_on_main(fn):
                """Schedule plugin route registration on the event-loop thread.

                FastAPI/Starlette router mutation is not thread-safe, so the
                actual setup() call is normally marshalled back onto the event
                loop via call_soon_threadsafe.  The background thread blocks
                until the registration completes, raises, or a 60 s timeout
                elapses.

                In synchronous startup mode (_sync_mode=True) this function is
                called directly from the event-loop thread, so marshalling via
                call_soon_threadsafe + fut.result() would deadlock (the loop
                cannot drain the queued callback while it is blocked here).
                In that case fn() is invoked inline instead.

                On timeout (async mode only), startup continues normally.  Any
                exception that eventually arrives is logged via a done-callback
                so it is never silently dropped.
                """
                if _sync_mode:
                    # Already on the event-loop thread — call directly.
                    fn()
                    return

                fut: concurrent.futures.Future = concurrent.futures.Future()

                def _do():
                    try:
                        fn()
                        fut.set_result(None)
                    except Exception as exc:
                        fut.set_exception(exc)

                loop.call_soon_threadsafe(_do)
                try:
                    fut.result(timeout=60)
                except concurrent.futures.TimeoutError:
                    _pid = getattr(fn, "_plugin_id", "unknown")
                    print(f"[Plugin] WARNING: route registration for '{_pid}' timed out after 60 s", flush=True)

                    def _log_deferred(f: concurrent.futures.Future):
                        try:
                            exc = f.exception()
                        except concurrent.futures.CancelledError:
                            return
                        if exc is not None:
                            print(f"[Plugin] ERROR: deferred route registration for '{_pid}' raised: {exc}", flush=True)

                    fut.add_done_callback(_log_deferred)
                    raise  # propagate to load_plugins() so it emits plugin-error and skips "Loaded routes"

            _set_startup_status(
                running=True,
                phase="plugins-loading",
                message="Loading plugins...",
                current_plugin="",
                loaded=0,
                total=0,
                error=None,
            )
            load_plugins(app, plugin_context, progress_cb=_on_progress,
                         route_setup_fn=_route_setup_on_main)
            status = _get_startup_status()
            _set_startup_status(
                running=False,
                phase="complete",
                message="Startup complete",
                current_plugin="",
                loaded=status.get("loaded", 0),
                total=max(status.get("total", 0), status.get("loaded", 0)),
                error=status.get("error"),
            )
        except Exception as e:
            _set_startup_status(
                running=False,
                phase="error",
                message="Plugin startup failed",
                error=str(e),
            )
            import traceback
            traceback.print_exc()

    if _sync_mode:
        # Caller requested synchronous startup (e.g. test environment).
        # Run the loader inline so startup is complete before the server's
        # startup handler returns — no polling or timing workarounds needed.
        _load_plugins_background()
    else:
        threading.Thread(target=_load_plugins_background, daemon=True).start()

    global _DEMO_JANITOR_STARTED, _DEMO_JANITOR_THREAD
    if os.environ.get("SLOPSMITH_DEMO_MODE") == "1" and not _DEMO_JANITOR_STARTED:
        _DEMO_JANITOR_STARTED = True
        _DEMO_JANITOR_STOP.clear()
        def _janitor():
            while not _DEMO_JANITOR_STOP.wait(timeout=3600):
                with _DEMO_JANITOR_HOOKS_LOCK:
                    hooks = list(_DEMO_JANITOR_HOOKS)
                for hook in hooks:
                    _run_janitor_hook(hook)
        _DEMO_JANITOR_THREAD = threading.Thread(target=_janitor, daemon=True, name="demo-janitor")
        _DEMO_JANITOR_THREAD.start()

    # Start background metadata scan
    startup_scan()


@app.on_event("shutdown")
def shutdown_events():
    """Stop the demo-mode janitor thread (if running) on server shutdown."""
    global _DEMO_JANITOR_STARTED, _DEMO_JANITOR_THREAD
    if _DEMO_JANITOR_STARTED:
        _DEMO_JANITOR_STOP.set()
        thread = _DEMO_JANITOR_THREAD
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                import warnings
                warnings.warn(
                    "demo-janitor thread did not stop within 5 s; "
                    "a registered hook may be blocking",
                    RuntimeWarning,
                    stacklevel=1,
                )
                # Leave _DEMO_JANITOR_STARTED True so a new janitor is not
                # spawned by a subsequent startup while the old one is alive.
                return
            _DEMO_JANITOR_THREAD = None
        _DEMO_JANITOR_STARTED = False
        with _DEMO_JANITOR_HOOKS_LOCK:
            _DEMO_JANITOR_HOOKS.clear()


def startup_scan():
    """Start background metadata scan and periodic rescan on server start."""
    thread = threading.Thread(target=_background_scan, daemon=True)
    thread.start()
    # Periodic rescan every 5 minutes
    rescan_thread = threading.Thread(target=_periodic_rescan, daemon=True)
    rescan_thread.start()


def _periodic_rescan():
    """Check for new files every 5 minutes."""
    time.sleep(300)  # Wait 5 minutes after startup
    while True:
        if not _scan_status["running"]:
            _background_scan()
        time.sleep(300)


@app.get("/api/version")
def get_version():
    env_version = os.environ.get("APP_VERSION", "").strip()
    if env_version:
        return {"version": env_version}
    version_file = Path(__file__).parent / "VERSION"
    version = "unknown"
    if version_file.exists():
        try:
            version = version_file.read_text().strip()
        except (OSError, UnicodeDecodeError):
            pass
    return {"version": version}


@app.get("/api/scan-status")
def scan_status():
    return _scan_status


@app.get("/api/startup-status")
def startup_status():
    return _get_startup_status()


@app.post("/api/rescan")
def trigger_rescan():
    """Manually trigger a library rescan."""
    if _scan_status["running"]:
        return {"message": "Scan already in progress"}
    thread = threading.Thread(target=_background_scan, daemon=True)
    thread.start()
    return {"message": "Rescan started"}


@app.post("/api/rescan/full")
def trigger_full_rescan():
    """Clear cache and rescan everything."""
    if _scan_status["running"]:
        return {"message": "Scan already in progress"}
    with meta_db._lock:
        meta_db.conn.execute("DELETE FROM songs")
        meta_db.conn.commit()
    thread = threading.Thread(target=_background_scan, daemon=True)
    thread.start()
    return {"message": "Full rescan started"}


# ── Library API ───────────────────────────────────────────────────────────────

def _split_csv(raw: str) -> list[str]:
    """Parse a comma-separated query-string list. Empty / whitespace-only
    entries are dropped so `arrangements_has=` (no value) and
    `arrangements_has=,` both mean 'no filter'."""
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_has_lyrics(raw: str) -> int | None:
    """Tri-state parse for has_lyrics. `1` → require, `0` → exclude,
    anything else (including empty) → no filter."""
    if raw == "1":
        return 1
    if raw == "0":
        return 0
    return None


@app.get("/api/library")
def list_library(q: str = "", page: int = 0, size: int = 24, sort: str = "artist",
                 dir: str = "asc", favorites: int = 0, format: str = "",
                 arrangements_has: str = "", arrangements_lacks: str = "",
                 stems_has: str = "", stems_lacks: str = "",
                 has_lyrics: str = "", tunings: str = ""):
    """Paginated library search, queried from SQLite."""
    size = min(size, 100)
    fmt = format if format in ("psarc", "sloppak") else ""
    songs, total = meta_db.query_page(
        q=q, page=page, size=size, sort=sort,
        direction=dir, favorites_only=bool(favorites), format_filter=fmt,
        arrangements_has=_split_csv(arrangements_has),
        arrangements_lacks=_split_csv(arrangements_lacks),
        stems_has=_split_csv(stems_has),
        stems_lacks=_split_csv(stems_lacks),
        has_lyrics=_parse_has_lyrics(has_lyrics),
        tunings=_split_csv(tunings),
    )
    return {"songs": songs, "total": total, "page": page, "size": size}


@app.get("/api/library/artists")
def list_artists(letter: str = "", q: str = "", favorites: int = 0, page: int = 0, size: int = 50,
                 format: str = "",
                 arrangements_has: str = "", arrangements_lacks: str = "",
                 stems_has: str = "", stems_lacks: str = "",
                 has_lyrics: str = "", tunings: str = ""):
    """Get artists grouped by letter with albums and songs (for tree view)."""
    fmt = format if format in ("psarc", "sloppak") else ""
    artists, total = meta_db.query_artists(
        letter=letter, q=q, favorites_only=bool(favorites),
        page=page, size=min(size, 100), format_filter=fmt,
        arrangements_has=_split_csv(arrangements_has),
        arrangements_lacks=_split_csv(arrangements_lacks),
        stems_has=_split_csv(stems_has),
        stems_lacks=_split_csv(stems_lacks),
        has_lyrics=_parse_has_lyrics(has_lyrics),
        tunings=_split_csv(tunings),
    )
    return {"artists": artists, "total_artists": total, "page": page, "size": size}


@app.get("/api/library/stats")
def library_stats(favorites: int = 0, q: str = "", format: str = "",
                  arrangements_has: str = "", arrangements_lacks: str = "",
                  stems_has: str = "", stems_lacks: str = "",
                  has_lyrics: str = "", tunings: str = ""):
    """Aggregate stats for the UI. Accepts the same filter params as
    /api/library so the letter bar mirrors the active grid filter set."""
    fmt = format if format in ("psarc", "sloppak") else ""
    return meta_db.query_stats(
        favorites_only=bool(favorites), q=q, format_filter=fmt,
        arrangements_has=_split_csv(arrangements_has),
        arrangements_lacks=_split_csv(arrangements_lacks),
        stems_has=_split_csv(stems_has),
        stems_lacks=_split_csv(stems_lacks),
        has_lyrics=_parse_has_lyrics(has_lyrics),
        tunings=_split_csv(tunings),
    )


@app.get("/api/library/tuning-names")
def list_tuning_names():
    """Distinct tuning names present in the library, with per-tuning
    counts. Powers the tuning multi-select. Sorted by `tuning_sort_key`
    so names appear in the same musical order the sort uses
    (slopsmith#22) — E Standard first, then nearest neighbors."""
    rows = meta_db.conn.execute(
        "SELECT tuning_name, MIN(tuning_sort_key), COUNT(*) "
        # NULL/empty-name rows are excluded entirely from the picker —
        # users can't usefully filter by an unknown tuning. Once they
        # rescan, the rows acquire a name and join the list.
        "FROM songs WHERE title != '' AND COALESCE(tuning_name, '') != '' "
        "GROUP BY tuning_name COLLATE NOCASE "
        # Same ordering as `sort=tuning` in `query_page`: distance
        # from E Standard first, then signed-key ASC so the down-
        # tuned variant precedes the up-tuned one at equal distance
        # (Eb Standard before F Standard at distance 6). Final
        # alphabetical tiebreak keeps the order deterministic.
        # COALESCE around the sort_key guards against NULL the same
        # way the main tuning sort does.
        "ORDER BY ABS(COALESCE(MIN(tuning_sort_key), 0)), "
        "COALESCE(MIN(tuning_sort_key), 0) ASC, "
        "tuning_name COLLATE NOCASE"
    ).fetchall()
    return {
        "tunings": [
            {"name": name, "sort_key": int(sk or 0), "count": count}
            for name, sk, count in rows
        ],
    }


@app.post("/api/favorites/toggle")
def toggle_favorite(data: dict):
    """Toggle a song's favorite status."""
    filename = data.get("filename", "")
    if not filename:
        return {"error": "No filename"}
    new_state = meta_db.toggle_favorite(filename)
    return {"favorite": new_state}


# ── Loops API ────────────────────────────────────────────────────────────────

@app.get("/api/loops")
def list_loops(filename: str):
    rows = meta_db.conn.execute(
        "SELECT id, name, start_time, end_time FROM loops WHERE filename = ? ORDER BY start_time",
        (filename,)
    ).fetchall()
    return [{"id": r[0], "name": r[1], "start": r[2], "end": r[3]} for r in rows]


@app.post("/api/loops")
def save_loop(data: dict):
    filename = data.get("filename", "")
    name = data.get("name", "").strip()
    start = data.get("start")
    end = data.get("end")
    if not filename or start is None or end is None:
        return {"error": "Missing fields"}
    if not name:
        count = meta_db.conn.execute(
            "SELECT COUNT(*) FROM loops WHERE filename = ?", (filename,)
        ).fetchone()[0]
        name = f"Loop {count + 1}"
    with meta_db._lock:
        meta_db.conn.execute(
            "INSERT INTO loops (filename, name, start_time, end_time) VALUES (?, ?, ?, ?)",
            (filename, name, float(start), float(end))
        )
        meta_db.conn.commit()
    return {"ok": True, "name": name}


@app.delete("/api/loops/{loop_id}")
def delete_loop(loop_id: int):
    with meta_db._lock:
        meta_db.conn.execute("DELETE FROM loops WHERE id = ?", (loop_id,))
        meta_db.conn.commit()
    return {"ok": True}


# ── Settings API ──────────────────────────────────────────────────────────────

def _default_settings():
    """Fallback settings returned when config.json is missing or
    unreadable. Also used to seed a fresh cfg on first-run POSTs so a
    single-key write (e.g. the difficulty slider) can't silently wipe
    defaults that subsequent GETs would have exposed."""
    # Same `_DLC_DIR_ENV` truthy check as `_get_dlc_dir`: an empty env
    # var collapses to `Path(".")` whose `.is_dir()` is True, so without
    # the explicit guard we'd surface `"."` to /api/settings — and any
    # partial-update POST would then persist that into config.json,
    # silently undoing the env-var fix on the next load.
    return {"dlc_dir": str(DLC_DIR) if (_DLC_DIR_ENV and DLC_DIR.is_dir()) else ""}


def _load_config(config_file):
    """Read and parse config.json. Returns the parsed dict, or None if
    the file is missing, unreadable, invalid JSON, or parses to a
    non-dict (e.g. the file contains `[]` or `42`). Callers treat None
    as "fall back to defaults". Shared between GET and POST so both
    handle bad files the same way."""
    if not config_file.exists():
        return None
    try:
        parsed = json.loads(config_file.read_text())
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


@app.get("/api/settings")
def get_settings():
    cfg = _load_config(CONFIG_DIR / "config.json")
    return cfg if cfg is not None else _default_settings()


@app.post("/api/settings")
def save_settings(data: dict):
    # Partial-update: merge only keys present in the request body so
    # single-key POSTs (like the difficulty slider's oninput) don't
    # clobber unrelated settings on disk.
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_file = CONFIG_DIR / "config.json"
    # Seed defaults when config.json is missing, unreadable, or parses
    # to a non-dict (e.g. `[]`, `42`). Without the non-dict guard, the
    # next `cfg["..."] = ...` assignment would raise TypeError and 500
    # the public endpoint. Seeding also ensures single-key POSTs (the
    # difficulty slider's fire-and-forget write) don't produce a config
    # file missing the dlc_dir fallback GET would have surfaced.
    cfg = _load_config(config_file)
    if cfg is None:
        cfg = _default_settings()

    messages = []
    if "dlc_dir" in data:
        dlc_path = data["dlc_dir"]
        # null / missing is no-op (preserve on-disk value). Only an
        # explicit empty string means "clear". Non-string values are
        # rejected so Path(...) can't be surprised by non-str JSON.
        if dlc_path is None:
            pass
        elif not isinstance(dlc_path, str):
            return {"error": "dlc_dir must be a string path or empty"}
        elif dlc_path == "":
            cfg["dlc_dir"] = ""
        else:
            if Path(dlc_path).is_dir():
                cfg["dlc_dir"] = dlc_path
                count = sum(1 for f in Path(dlc_path).iterdir() if f.suffix == ".psarc")
                messages.append(f"DLC folder: {count} .psarc files found")
            else:
                return {"error": f"DLC directory not found: {dlc_path}"}

    # Both of these are consumed downstream as strings (e.g.
    # demucs_server_url.rstrip('/') in lib/sloppak_convert.py), so
    # reject non-string shapes here. Matches the dlc_dir pattern above:
    # null is no-op, empty string clears, non-string is a structured
    # error that preserves the on-disk value.
    for key in ("default_arrangement", "demucs_server_url"):
        if key in data:
            raw = data[key]
            if raw is None:
                pass
            elif not isinstance(raw, str):
                return {"error": f"{key} must be a string or empty"}
            else:
                cfg[key] = raw
    if "master_difficulty" in data:
        # Coerce defensively — public endpoint, so `null`, `""`, or a
        # non-numeric string shouldn't 500 the request. float() accepts
        # both integer and float-shaped strings; anything else returns
        # a structured error like the dlc_dir branch above.
        raw = data["master_difficulty"]
        # Reject bool explicitly: Python makes bool a subclass of int, so
        # True/False would otherwise coerce to 1/0 and persist as a valid
        # difficulty. Caller almost certainly means "bad input".
        if isinstance(raw, bool):
            return {"error": "master_difficulty must be a number between 0 and 100"}
        try:
            cfg["master_difficulty"] = max(0, min(100, int(float(raw))))
        except (TypeError, ValueError, OverflowError):
            # OverflowError covers int(float("inf")) / int(float("1e309"))
            # which Python raises distinctly from ValueError.
            return {"error": "master_difficulty must be a number between 0 and 100"}

    if "av_offset_ms" in data:
        # Audio-output pipeline latency compensation. Positive values
        # mean audio is running ahead of visuals; the highway adds
        # this to its render clock to catch the visuals up. Clamped
        # to ±1000 ms to mirror the client-side slider — a direct
        # POST shouldn't be able to persist `1e9`. Same defensive
        # coercion shape as master_difficulty above (reject bool,
        # cover OverflowError, structured 4xx-style return on bad
        # input rather than 500).
        raw = data["av_offset_ms"]
        if isinstance(raw, bool):
            return {"error": "av_offset_ms must be a number between -1000 and 1000"}
        try:
            cfg["av_offset_ms"] = max(-1000.0, min(1000.0, float(raw)))
        except (TypeError, ValueError, OverflowError):
            return {"error": "av_offset_ms must be a number between -1000 and 1000"}

    config_file.write_text(json.dumps(cfg, indent=2))
    return {"message": ". ".join(messages) if messages else "Settings saved"}


# ── Settings export/import (slopsmith#113) ───────────────────────────────────

# Bumped only when the bundle JSON shape changes incompatibly. Importer
# refuses anything but this exact value — version mismatches are warned
# but not blocked, schema mismatches ARE blocked.
SETTINGS_BUNDLE_SCHEMA = 1


def _running_version() -> str:
    """Same lookup chain `/api/version` uses, factored out so the export
    bundle records what shipped this file. Kept as a helper so future
    changes (e.g. baked-in version) only have to touch one site."""
    env_version = os.environ.get("APP_VERSION", "").strip()
    if env_version:
        return env_version
    version_file = Path(__file__).parent / "VERSION"
    if version_file.exists():
        try:
            return version_file.read_text().strip()
        except (OSError, UnicodeDecodeError):
            pass
    return "unknown"


def _validate_server_config_types(cfg: dict) -> str | None:
    """Type-and-range gate for the server_config block of an import
    bundle, mirroring the per-key checks in `POST /api/settings`. The
    importer writes config.json verbatim, so without this gate a
    hand-edited bundle could persist a non-string `demucs_server_url`
    (which downstream code calls `.rstrip('/')` on and crashes) or an
    out-of-range `master_difficulty` (which bypasses the slider's
    clamp). Returns None on success, an error string on the first
    violation. Filesystem-existence checks (e.g. dlc_dir is_dir) are
    NOT performed here — restoring a bundle on a different machine
    legitimately may reference paths that don't exist locally yet,
    and the `POST /api/settings` interactive endpoint is the right
    place for that ergonomic check, not the bulk-restore path.
    Unknown keys are passed through so future settings (and per-plugin
    keys that may be added later) round-trip without code changes
    here."""
    if "dlc_dir" in cfg:
        v = cfg["dlc_dir"]
        if v is not None and not isinstance(v, str):
            return "server_config.dlc_dir must be a string"
    for key in ("default_arrangement", "demucs_server_url"):
        if key in cfg:
            v = cfg[key]
            if v is not None and not isinstance(v, str):
                return f"server_config.{key} must be a string"
    if "master_difficulty" in cfg:
        v = cfg["master_difficulty"]
        # bool is an int subclass — reject explicitly so True/False
        # don't quietly persist as 1/0 difficulty values.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return "server_config.master_difficulty must be a number between 0 and 100"
        if not (0 <= v <= 100):
            return "server_config.master_difficulty must be between 0 and 100"
    if "av_offset_ms" in cfg:
        v = cfg["av_offset_ms"]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return "server_config.av_offset_ms must be a number between -1000 and 1000"
        if not (-1000 <= v <= 1000):
            return "server_config.av_offset_ms must be between -1000 and 1000"
    return None


class _UndeclaredFile(ValueError):
    """Raised when a relpath would otherwise be safe but isn't covered by
    the plugin's manifest allowlist. Distinct from the generic
    `ValueError` so the import handler can warn-and-skip this case
    without resorting to message-string matching (which would silently
    change behavior on a future error-text refactor)."""


def _matches_allowlist(relpath: str, allowed: list[str]) -> bool:
    """Return True if `relpath` is covered by an entry in the manifest's
    `_export_paths`. Entries ending in `/` are directory rules
    (strict prefix-match); other entries are exact-file rules. Both
    `relpath` and `allowed` are POSIX strings already normalized
    through `_normalize_export_paths` on the loader side. Caller is
    expected to pass an already-normalized relpath — `_validate_relpath`
    enforces this so a bundle can't satisfy a prefix rule with a
    string that later normalizes to a different target."""
    for allow in allowed:
        if allow.endswith("/"):
            # Strict prefix match only. We deliberately reject
            # `relpath == prefix.rstrip("/")` — a directory entry
            # never authorizes writing AT the directory itself, and
            # accepting that would let phase 2 try to `os.replace()`
            # over an existing directory and crash mid-apply.
            if relpath.startswith(allow):
                return True
        elif relpath == allow:
            return True
    return False


def _validate_relpath(relpath: str, allowed: list[str], config_dir: Path) -> Path:
    """Resolve `relpath` to an absolute path under `config_dir`, raising
    on anything that smells like path-traversal, an absolute path, or
    a manifest-undeclared file. Layered defenses:

      1. String-level: reject backslash, drive letter, absolute, and
         any `.` / `..` segment in the *raw* input — BEFORE any
         normalization. Critically, this catches the
         `allowed_dir/../config.json` shape: the raw string starts
         with `allowed_dir/`, so a naive prefix-match would accept
         it; if we then normalized first, the `..` would collapse
         away and the segment guard would have nothing to reject. By
         refusing pre-normalization any input containing a `.` or
         `..` segment, we make it impossible for a normalize-then-
         resolve pass to "launder" a hostile prefix into a different
         target.
      2. Allowlist match against the now-known-clean relpath.
         Allowlist-miss raises `_UndeclaredFile` (a `ValueError`
         subclass) so the caller can distinguish "manifest changed
         between export and import" from "this looks like an attack"
         without string-matching the error message.
      3. Realpath check: after resolving under config_dir, the target
         must still live inside config_dir. This catches symlinks-
         under-config_dir attacks where someone planted a symlink
         pointing out and tried to import a file "under" it.
      4. Symlink rejection: even when a symlink (or symlinked
         directory component) resolves to a path that *still* lives
         inside config_dir, importing through it would let an
         allowlisted relpath redirect the write to a different
         in-config file — bypassing the manifest's intent. We probe
         every path component from `config_dir` down to the target
         using `lstat`, refusing if any link is set on the chain.
         This matches the documented "symlinks are never followed on
         import" guarantee.

    Returns the resolved absolute path (caller writes there in phase 2).
    """
    if not isinstance(relpath, str) or not relpath or relpath != relpath.strip():
        raise ValueError(f"illegal relpath: {relpath!r}")
    # Reject backslashes outright — manifest entries are POSIX, and
    # accepting `foo\bar` here on a platform whose Path treats `\` as
    # a separator would let a hostile bundle smuggle traversal past
    # the part-by-part check below.
    if "\\" in relpath:
        raise ValueError(f"relpath uses non-POSIX separator: {relpath!r}")
    # Absolute / drive-letter check before splitting.
    if relpath.startswith("/") or (len(relpath) >= 2 and relpath[1] == ":"):
        raise ValueError(f"relpath must be relative: {relpath!r}")
    raw_parts = relpath.split("/")
    # Empty parts catch `foo//bar` and a trailing `/`. `.` / `..` catch
    # both leading and embedded forms (`./x`, `a/./b`, `allow/../escape`).
    if any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError(f"relpath contains illegal segment: {relpath!r}")
    # Defense-in-depth: any leading `.` segment (e.g. dotfile-disguised
    # paths like `.git/config`) is also rejected — config_dir isn't a
    # place plugins should be writing dotfiles, and accepting them here
    # would let one plugin claim a global filename like `.npmrc`.
    if raw_parts[0].startswith("."):
        raise ValueError(f"relpath starts with dotfile segment: {relpath!r}")

    if not _matches_allowlist(relpath, allowed):
        raise _UndeclaredFile(
            f"relpath not declared in plugin manifest: {relpath!r}"
        )

    target = (config_dir / relpath).resolve()
    config_root = config_dir.resolve()
    # `target == config_root` would mean the relpath resolved to the
    # config dir itself, which can't be a file write target — reject.
    if target == config_root:
        raise ValueError(f"relpath resolves to config_dir itself: {relpath!r}")
    if config_root not in target.parents:
        raise ValueError(f"relpath escapes config_dir: {relpath!r}")

    # Walk every component from config_dir down to (but not including)
    # the target file, refusing if any is a symlink. The target itself
    # is checked too — a symlinked file inside config_dir could still
    # redirect the write to another in-config file, defeating the
    # manifest's allowlist intent. `lstat` is the right primitive: it
    # reports the link itself rather than the link's destination, so a
    # broken or self-referential symlink won't slip through. Missing
    # intermediate dirs are fine — `_atomic_write_file` mkdirs them
    # under config_dir, and a path that doesn't exist yet trivially
    # isn't a symlink.
    probe = config_dir
    for part in relpath.split("/"):
        probe = probe / part
        try:
            st = os.lstat(probe)
        except FileNotFoundError:
            # Component doesn't exist yet → can't be a symlink. Any
            # remaining components also don't exist, so we're done.
            break
        import stat as _stat
        if _stat.S_ISLNK(st.st_mode):
            raise ValueError(
                f"relpath traverses or targets a symlink: {relpath!r}"
            )
    return target


def _encode_file(abs_path: Path) -> dict:
    """Encode a single file for the export bundle. JSON files that parse
    cleanly use the `json` encoding so the bundle stays diff-friendly;
    everything else (sqlite, NAM models, IRs, binary blobs) falls back
    to base64. Symlinks are skipped at the caller — we never reach this
    helper for them."""
    import base64
    raw = abs_path.read_bytes()
    if abs_path.suffix.lower() == ".json":
        try:
            return {"encoding": "json", "data": json.loads(raw.decode("utf-8"))}
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Fall through to base64 — file claimed `.json` but isn't
            # valid JSON; preserve bytes verbatim rather than refusing.
            pass
    return {"encoding": "base64", "data": base64.b64encode(raw).decode("ascii")}


def _decode_entry(entry: dict) -> bytes:
    """Inverse of `_encode_file`. Raises ValueError on malformed entries
    so phase 1 of the importer can refuse the whole bundle without
    having written anything."""
    import base64
    if not isinstance(entry, dict):
        raise ValueError(f"file entry must be an object, got {type(entry).__name__}")
    encoding = entry.get("encoding")
    data = entry.get("data")
    if encoding == "base64":
        if not isinstance(data, str):
            raise ValueError("base64 entry: 'data' must be a string")
        try:
            return base64.b64decode(data, validate=True)
        except Exception as e:
            raise ValueError(f"base64 entry: invalid payload ({e})")
    if encoding == "json":
        # We re-serialize the parsed value with stable formatting. Round
        # trips with the original byte stream aren't guaranteed (key
        # order, whitespace), but the file's *meaning* is preserved.
        try:
            return json.dumps(data, indent=2).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise ValueError(f"json entry: cannot re-serialize ({e})")
    raise ValueError(f"unknown encoding: {encoding!r}")


def _walk_export_paths(allowed: list[str], config_dir: Path) -> dict:
    """Expand a plugin's `_export_paths` against disk and return a
    `{relpath: encoded_entry}` dict. Missing files are silently skipped
    (intentional — manifests can list optional files). Symlinks are
    skipped with no entry. Directories are walked recursively; their
    contained files surface as POSIX-joined relpaths.

    Symlink policy is "skipped and never followed" at every depth:
    `os.walk(..., followlinks=False)` ensures we don't *recurse* into
    symlinked subdirectories, but we additionally drop any symlinked
    entry from `dirnames` (so its name isn't even reported to the
    caller, even though the walker wouldn't descend) and skip files
    whose path is itself a symlink. Without those extra filters, a
    planted symlink directory under an allowed prefix could leak data
    from outside `config_dir` into the export bundle.
    """
    out: dict[str, dict] = {}
    for entry in allowed:
        is_dir = entry.endswith("/")
        rel = entry.rstrip("/")
        abs_target = config_dir / rel
        if abs_target.is_symlink():
            continue
        if is_dir:
            if not abs_target.is_dir():
                continue
            collected: list[Path] = []
            for dirpath, dirnames, filenames in os.walk(
                str(abs_target), followlinks=False
            ):
                # Strip symlinked subdirs from `dirnames` in-place so
                # the walker neither yields their names nor descends.
                dirnames[:] = [
                    d for d in dirnames
                    if not os.path.islink(os.path.join(dirpath, d))
                ]
                for fname in filenames:
                    full = os.path.join(dirpath, fname)
                    if os.path.islink(full) or not os.path.isfile(full):
                        continue
                    collected.append(Path(full))
            # Sort for deterministic bundle output (test fixtures and
            # diffs both rely on stable ordering).
            for child in sorted(collected):
                # POSIX-joined relpath relative to config_dir keeps the
                # bundle cross-platform — Windows-authored bundles can
                # be applied on Linux and vice versa.
                child_rel = child.relative_to(config_dir).as_posix()
                out[child_rel] = _encode_file(child)
        else:
            if not abs_target.is_file():
                continue
            out[rel] = _encode_file(abs_target)
    return out


def _atomic_write_file(target: Path, payload: bytes):
    """Write `payload` to `target` via a uniquely-named sibling temp file
    + os.replace. `os.replace` is atomic on both POSIX and Win32 —
    readers see either the old file or the new one, never a half-written
    state.

    The temp name is generated by `tempfile.mkstemp` so two concurrent
    imports (or two workers sharing the same config volume) can't race
    on the same `<target>.tmp.import` path and clobber each other's
    in-flight writes. On any failure between mkstemp and the successful
    `os.replace`, we remove the temp file so a failed import doesn't
    leave `.tmp.import` litter under config_dir."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=target.name + ".",
        suffix=".tmp.import",
    )
    tmp = Path(tmp_name)
    # Hand fd to os.fdopen inside its own try, so a failure to wrap
    # the descriptor (rare — typically EMFILE / ENOMEM) doesn't leak
    # the raw fd. On Windows an open fd would also keep the temp file
    # locked and undeletable. Once `with` enters, the fdopen'd file
    # owns close responsibility.
    try:
        f = os.fdopen(fd, "wb")
    except Exception:
        os.close(fd)
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    try:
        with f:
            f.write(payload)
        os.replace(tmp, target)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


@app.get("/api/settings/export")
def export_settings():
    """Build a settings bundle covering server config + opted-in plugin
    server-side files. Frontend layers in `local_storage` before
    triggering the download. See slopsmith#113."""
    import datetime
    from plugins import LOADED_PLUGINS, PLUGINS_LOCK

    config_file = CONFIG_DIR / "config.json"
    server_config = _load_config(config_file)
    if server_config is None:
        server_config = _default_settings()

    plugin_blocks: dict[str, dict] = {}
    with PLUGINS_LOCK:
        plugins_snapshot = list(LOADED_PLUGINS)
    for p in plugins_snapshot:
        allowed = p.get("_export_paths") or []
        plugin_blocks[p["id"]] = {"files": _walk_export_paths(allowed, CONFIG_DIR)}

    # Capture the timestamp once so the bundle's `exported_at` and the
    # download filename's date prefix can't disagree if the request
    # crosses midnight UTC between the two formats.
    now = datetime.datetime.now(datetime.timezone.utc)
    bundle = {
        "schema": SETTINGS_BUNDLE_SCHEMA,
        "exported_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slopsmith_version": _running_version(),
        "server_config": server_config,
        "plugin_server_configs": plugin_blocks,
    }
    filename = f"slopsmith-settings-{now.strftime('%Y-%m-%d')}.json"
    return JSONResponse(
        bundle,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/settings/import")
def import_settings(bundle: dict):
    """Apply a previously exported settings bundle. Validates the entire
    bundle in phase 1 (no disk writes); only on full success does
    phase 2 commit each file via temp+rename. The frontend reads
    `local_storage` itself — server ignores it. See slopsmith#113."""
    from plugins import LOADED_PLUGINS, PLUGINS_LOCK

    if not isinstance(bundle, dict):
        return JSONResponse({"ok": False, "error": "bundle must be a JSON object"}, status_code=400)

    # ── Phase 1: validate everything before touching disk ────────────
    schema = bundle.get("schema")
    if schema != SETTINGS_BUNDLE_SCHEMA:
        return JSONResponse(
            {
                "ok": False,
                "error": f"unsupported schema {schema!r}; this server speaks schema {SETTINGS_BUNDLE_SCHEMA}",
            },
            status_code=400,
        )

    server_config = bundle.get("server_config")
    if not isinstance(server_config, dict):
        return JSONResponse(
            {"ok": False, "error": "server_config must be an object"},
            status_code=400,
        )
    cfg_err = _validate_server_config_types(server_config)
    if cfg_err is not None:
        return JSONResponse(
            {"ok": False, "error": cfg_err},
            status_code=400,
        )

    plugin_blocks = bundle.get("plugin_server_configs") or {}
    if not isinstance(plugin_blocks, dict):
        return JSONResponse(
            {"ok": False, "error": "plugin_server_configs must be an object"},
            status_code=400,
        )

    warnings: list[str] = []
    bundle_version = bundle.get("slopsmith_version")
    running = _running_version()
    if bundle_version and bundle_version != running:
        warnings.append(
            f"version mismatch: bundle {bundle_version!r} vs running {running!r}; importing anyway"
        )

    with PLUGINS_LOCK:
        by_id = {p["id"]: p for p in LOADED_PLUGINS}

    # Stage every (display_relpath, target_abs_path, payload) tuple before
    # writing. The relpath is what we surface in the `partial` field on a
    # mid-apply failure — absolute paths would leak the deployment's
    # config_dir layout, while the relpath is the same identifier the
    # bundle itself used and is portable across machines.
    staged: list[tuple[str, Path, bytes]] = []
    applied_plugins: list[str] = []
    for plugin_id, block in plugin_blocks.items():
        if not isinstance(plugin_id, str) or not plugin_id:
            return JSONResponse(
                {"ok": False, "error": f"invalid plugin id key: {plugin_id!r}"},
                status_code=400,
            )
        plugin = by_id.get(plugin_id)
        if plugin is None:
            warnings.append(f"plugin {plugin_id!r} not loaded; skipping its files")
            continue
        if not isinstance(block, dict):
            return JSONResponse(
                {"ok": False, "error": f"plugin {plugin_id!r}: block must be an object"},
                status_code=400,
            )
        files = block.get("files") or {}
        if not isinstance(files, dict):
            return JSONResponse(
                {"ok": False, "error": f"plugin {plugin_id!r}: files must be an object"},
                status_code=400,
            )
        allowed = plugin.get("_export_paths") or []
        skipped_for_plugin: list[str] = []
        applied_for_plugin = False
        for relpath, file_entry in files.items():
            try:
                target = _validate_relpath(relpath, allowed, CONFIG_DIR)
            except _UndeclaredFile:
                # Manifest-allowlist miss is a normal outcome of a
                # plugin update between export and import — warn-and-
                # skip so the rest of the bundle still applies.
                skipped_for_plugin.append(relpath)
                continue
            except ValueError as e:
                # Path-traversal / absolute-path / illegal-segment /
                # backslash / dotfile errors are hard failures: we
                # never want to apply a bundle that contains those,
                # even partially. Caught AFTER `_UndeclaredFile`
                # because that's a `ValueError` subclass — Python
                # would otherwise route it through this branch.
                return JSONResponse(
                    {
                        "ok": False,
                        "error": f"plugin {plugin_id!r}, file {relpath!r}: {e}",
                    },
                    status_code=400,
                )
            try:
                payload = _decode_entry(file_entry)
            except ValueError as e:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": f"plugin {plugin_id!r}, file {relpath!r}: {e}",
                    },
                    status_code=400,
                )
            # Display key prefixes the plugin id so a partial-failure
            # report is unambiguous when two plugins happen to declare
            # files with the same relpath.
            display = f"{plugin_id}/{relpath}"
            staged.append((display, target, payload))
            applied_for_plugin = True
        if skipped_for_plugin:
            warnings.append(
                f"plugin {plugin_id!r}: skipped {len(skipped_for_plugin)} file(s) "
                f"no longer declared in manifest: {skipped_for_plugin}"
            )
        if applied_for_plugin:
            applied_plugins.append(plugin_id)

    # ── Phase 2: commit ──────────────────────────────────────────────
    written: list[str] = []
    try:
        for display, target, payload in staged:
            _atomic_write_file(target, payload)
            written.append(display)
        # Server config last so a write failure on a plugin file
        # doesn't leave config.json mismatched against the (untouched)
        # plugin state. Full-replace: caller is responsible for the
        # whole dict — this is restore semantics, not partial-update.
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_file(
            CONFIG_DIR / "config.json",
            json.dumps(server_config, indent=2).encode("utf-8"),
        )
    except OSError as e:
        # Phase-1 validation should have caught all foreseeable
        # failures; an OSError here means disk-level trouble (ENOSPC,
        # permission). We can't roll back already-replaced files
        # because we didn't snapshot them — surface what got written
        # (as relpaths, not absolute server paths) so the user knows
        # the state is partial without leaking deployment layout.
        return JSONResponse(
            {
                "ok": False,
                "error": f"write failed mid-apply: {e}",
                "partial": written,
            },
            status_code=500,
        )

    return {
        "ok": True,
        "warnings": warnings,
        "applied": {
            "server_config": True,
            "plugins": applied_plugins,
        },
    }


# ── Plugin-provided routes are registered at startup via plugins/__init__.py ─
# (CustomsForge, Ultimate Guitar, etc. are loaded from plugins/ directory)



@app.websocket("/ws/retune")
async def ws_retune(websocket: WebSocket, filename: str, target: str = "E Standard"):
    """Retune a song to a target tuning with real-time progress."""
    import asyncio
    await websocket.accept()

    dlc = _get_dlc_dir()
    if not dlc:
        await websocket.send_json({"error": "DLC folder not configured"})
        await websocket.close()
        return

    psarc_path = dlc / filename
    if not psarc_path.exists():
        await websocket.send_json({"error": "File not found"})
        await websocket.close()
        return

    # Retune only operates on PSARC containers — sloppak is an open format
    # and doesn't share the SNG/encryption pipeline retune.py depends on.
    if filename.lower().endswith(".sloppak") or sloppak_mod.is_sloppak(psarc_path):
        await websocket.send_json({"error": "Retune is not supported for .sloppak files"})
        await websocket.close()
        return

    progress_queue = asyncio.Queue()

    def _do_retune():
        from retune import retune_to_standard, get_tuning

        def report(stage, pct):
            progress_queue.put_nowait({"stage": stage, "progress": pct})

        try:
            report("Checking tuning...", 5)
            offsets, uniform = get_tuning(str(psarc_path))

            # Determine target offsets
            if target == "Drop D":
                target_offsets = [-2, 0, 0, 0, 0, 0]
            else:
                target_offsets = [0, 0, 0, 0, 0, 0]

            # Check if already at target
            if offsets == target_offsets:
                progress_queue.put_nowait({"error": f"Already in {target}"})
                return

            # For uniform tunings (all same offset), shift everything to 0
            # For drop tunings, check if the shift is uniform
            shift = [target_offsets[i] - offsets[i] for i in range(6)]
            if len(set(shift)) != 1:
                progress_queue.put_nowait({"error": f"Cannot uniformly retune {offsets} to {target} — shift varies per string"})
                return

            semitones = shift[0]
            report("Extracting PSARC...", 10)

            import builtins
            _orig_print = builtins.print
            def _progress_print(*args, **kwargs):
                msg = " ".join(str(a) for a in args)
                if "Processing" in msg: report(msg, 30)
                elif "Decoded" in msg: report(msg, 45)
                elif "Shifted" in msg: report(msg, 60)
                elif "Updated tuning" in msg: report(msg, 70)
                elif "Recompiling" in msg: report(msg, 80)
                elif "Repacking" in msg: report(msg, 90)
                elif "Created" in msg: report(msg, 95)
                _orig_print(*args, **kwargs)

            builtins.print = _progress_print
            try:
                # Set custom output path based on target
                suffix = "_EStd" if target == "E Standard" else "_DropD"
                p = Path(psarc_path)
                stem = p.stem.replace("_p", "")
                out_path = str(p.parent / f"{stem}{suffix}_p.psarc")
                result = retune_to_standard(str(psarc_path), output_path=out_path)
            finally:
                builtins.print = _orig_print

            # Cache metadata for new file
            new_path = Path(result)
            if new_path.exists():
                try:
                    meta = _extract_meta_for_file(new_path)
                    stat = new_path.stat()
                    meta_db.put(new_path.name, stat.st_mtime, stat.st_size, meta)
                except Exception:
                    pass

            progress_queue.put_nowait({
                "done": True, "progress": 100,
                "stage": "Complete!",
                "filename": new_path.name,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            progress_queue.put_nowait({"error": str(e)})

    loop = asyncio.get_event_loop()
    build_task = loop.run_in_executor(None, _do_retune)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                await websocket.send_json(msg)
                if msg.get("done") or msg.get("error"):
                    break
            except asyncio.TimeoutError:
                if build_task.done():
                    break
    except WebSocketDisconnect:
        pass

    await websocket.close()


@app.get("/api/song/{filename:path}/art")
async def get_song_art(filename: str):
    """Extract and serve album art from a PSARC as PNG."""
    import asyncio
    dlc = _get_dlc_dir()
    if not dlc:
        return JSONResponse({"error": "not configured"}, 404)

    psarc_path = dlc / filename
    if not psarc_path.exists():
        return JSONResponse({"error": "not found"}, 404)

    # Sloppak path: pull cover.jpg from the source dir (manifest-declared or default).
    if sloppak_mod.is_sloppak(psarc_path):
        try:
            src = sloppak_mod.resolve_source_dir(filename, dlc, SLOPPAK_CACHE_DIR)
            manifest = sloppak_mod.load_manifest(psarc_path)
            cover_rel = str(manifest.get("cover") or "cover.jpg")
            cover_path = (src / cover_rel).resolve()
            # Prevent escape and fall back to default name if missing.
            try:
                cover_path.relative_to(src.resolve())
            except ValueError:
                return JSONResponse({"error": "forbidden"}, 403)
            if cover_path.exists() and cover_path.is_file():
                mt = {
                    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp",
                }.get(cover_path.suffix.lower(), "image/jpeg")
                return FileResponse(str(cover_path), media_type=mt)
        except Exception:
            pass
        return JSONResponse({"error": "no art"}, 404)

    # Check cache first
    art_cache = ART_CACHE_DIR
    art_cache.mkdir(parents=True, exist_ok=True)
    safe_name = filename.replace("/", "_").replace(" ", "_")
    cached = art_cache / f"{safe_name}.png"
    if cached.exists():
        return FileResponse(str(cached), media_type="image/png")

    def _extract_art():
        tmp = tempfile.mkdtemp(prefix="rs_art_")
        try:
            unpack_psarc(str(psarc_path), tmp)
            dds_files = sorted(Path(tmp).rglob("*.dds"), key=lambda p: p.stat().st_size, reverse=True)
            if not dds_files:
                return None
            from PIL import Image
            img = Image.open(dds_files[0]).convert("RGB")
            img.save(str(cached), "PNG")
            return str(cached)
        except Exception:
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    result = await asyncio.get_event_loop().run_in_executor(None, _extract_art)
    if result:
        return FileResponse(result, media_type="image/png")
    return JSONResponse({"error": "no art"}, 404)


@app.post("/api/song/{filename:path}/meta")
def update_song_meta(filename: str, data: dict):
    """Update song metadata in the cache."""
    with meta_db._lock:
        updates = []
        params = []
        for field in ("title", "artist", "album", "year"):
            if field in data:
                updates.append(f"{field} = ?")
                params.append(data[field])
        if not updates:
            return {"error": "No fields to update"}
        params.append(filename)
        meta_db.conn.execute(
            f"UPDATE songs SET {', '.join(updates)} WHERE filename = ?", params
        )
        meta_db.conn.commit()
    return {"ok": True}


@app.post("/api/song/{filename:path}/art/upload")
async def upload_song_art_b64(filename: str, data: dict):
    """Upload custom album art as base64 PNG/JPG."""
    import base64
    b64 = data.get("image", "")
    if not b64:
        return {"error": "No image data"}
    # Strip data URL prefix if present
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        img_data = base64.b64decode(b64)
    except Exception:
        return {"error": "Invalid base64"}

    art_cache = ART_CACHE_DIR
    art_cache.mkdir(parents=True, exist_ok=True)
    safe_name = filename.replace("/", "_").replace(" ", "_")
    cached = art_cache / f"{safe_name}.png"

    # Convert to PNG if needed
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img.save(str(cached), "PNG")
    except Exception as e:
        return {"error": f"Invalid image: {e}"}

    return {"ok": True}


@app.get("/api/song/{filename:path}")
async def get_song_info(filename: str):
    """Return song metadata, from cache or by extracting PSARC."""
    import asyncio
    dlc = _get_dlc_dir()
    if not dlc:
        return JSONResponse({"error": "DLC folder not configured"}, 404)

    psarc_path = dlc / filename
    if not psarc_path.exists():
        return JSONResponse({"error": "File not found"}, 404)

    stat = psarc_path.stat()
    cached = meta_db.get(filename, stat.st_mtime, stat.st_size)
    if cached:
        return cached

    # Extract in thread pool
    def _extract():
        meta = _extract_meta_for_file(psarc_path)
        meta_db.put(filename, stat.st_mtime, stat.st_size, meta)
        return meta

    meta = await asyncio.get_event_loop().run_in_executor(None, _extract)
    return meta


# ── Highway WebSocket ─────────────────────────────────────────────────────────

# Cache extracted PSARCs to avoid re-extraction on arrangement switch
_extract_cache = {}  # filename -> (tmp_dir, song, timestamp)
_extract_cache_lock = threading.Lock()


def _get_or_extract(filename, psarc_path):
    """Return cached extraction or extract fresh."""
    import time
    with _extract_cache_lock:
        cached = _extract_cache.get(filename)
        if cached:
            tmp, song, ts = cached
            if Path(tmp).exists() and (time.time() - ts) < 300:  # 5 min cache
                return tmp, song, False  # False = not new
            else:
                shutil.rmtree(tmp, ignore_errors=True)
                del _extract_cache[filename]

    tmp = tempfile.mkdtemp(prefix="rs_web_")
    unpack_psarc(str(psarc_path), tmp)
    song = load_song(tmp)

    with _extract_cache_lock:
        # Clean old entries if cache gets too big
        if len(_extract_cache) > 10:
            oldest = min(_extract_cache, key=lambda k: _extract_cache[k][2])
            old_tmp = _extract_cache.pop(oldest)[0]
            shutil.rmtree(old_tmp, ignore_errors=True)
        import time as _t
        _extract_cache[filename] = (tmp, song, _t.time())

    return tmp, song, True  # True = freshly extracted


@app.get("/api/sloppak/{filename:path}/file/{rel_path:path}")
def serve_sloppak_file(filename: str, rel_path: str):
    """Serve a file from inside a sloppak (stems, cover, etc.)."""
    src = sloppak_mod.get_cached_source_dir(filename)
    if src is None:
        dlc = _get_dlc_dir()
        if not dlc:
            return JSONResponse({"error": "not configured"}, 404)
        try:
            src = sloppak_mod.resolve_source_dir(filename, dlc, SLOPPAK_CACHE_DIR)
        except Exception:
            return JSONResponse({"error": "not found"}, 404)
    # Prevent path traversal.
    target = (src / rel_path).resolve()
    try:
        target.relative_to(src.resolve())
    except ValueError:
        return JSONResponse({"error": "forbidden"}, 403)
    if not target.exists() or not target.is_file():
        return JSONResponse({"error": "not found"}, 404)
    ext = target.suffix.lower()
    mt = {
        ".ogg": "audio/ogg", ".opus": "audio/ogg", ".oga": "audio/ogg",
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".json": "application/json",
    }.get(ext)
    return FileResponse(str(target), media_type=mt) if mt else FileResponse(str(target))


@app.websocket("/ws/highway/{filename:path}")
async def highway_ws(websocket: WebSocket, filename: str, arrangement: int = -1):
    """Stream song data for the highway renderer over WebSocket."""
    await websocket.accept()

    dlc = _get_dlc_dir()
    if not dlc:
        await websocket.send_json({"error": "DLC folder not configured"})
        await websocket.close()
        return

    psarc_path = dlc / filename
    if not psarc_path.exists():
        await websocket.send_json({"error": "File not found"})
        await websocket.close()
        return

    is_slop = sloppak_mod.is_sloppak(psarc_path)
    tmp = None
    owns_tmp = False
    loaded_slop = None  # LoadedSloppak when is_slop
    _keepalive_active = True

    async def _send_keepalives():
        while _keepalive_active:
            try:
                await asyncio.sleep(3)
                if _keepalive_active:
                    await websocket.send_json({"type": "loading", "stage": "Loading..."})
            except Exception:
                break

    try:
        await websocket.send_json({"type": "loading", "stage": "Extracting..."})
        keepalive_task = asyncio.create_task(_send_keepalives())

        try:
            loop = asyncio.get_event_loop()
            if is_slop:
                SLOPPAK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                loaded_slop = await loop.run_in_executor(
                    None,
                    lambda: sloppak_mod.load_song(filename, dlc, SLOPPAK_CACHE_DIR),
                )
                song = loaded_slop.song
                tmp = str(loaded_slop.source_dir)
                owns_tmp = False
            else:
                tmp, song, owns_tmp = await loop.run_in_executor(None, lambda: _get_or_extract(filename, psarc_path))
        finally:
            _keepalive_active = False
            keepalive_task.cancel()

        if not song.arrangements:
            await websocket.send_json({"error": "No arrangements found"})
            await websocket.close()
            return

        # Pick arrangement: explicit request > user preference > most notes
        best = -1
        if 0 <= arrangement < len(song.arrangements):
            best = arrangement
        else:
            # Check user's default arrangement preference
            pref = ""
            config_file = CONFIG_DIR / "config.json"
            if config_file.exists():
                try:
                    pref = json.loads(config_file.read_text()).get("default_arrangement", "")
                except Exception:
                    pass
            if pref:
                for i, a in enumerate(song.arrangements):
                    if a.name == pref:
                        best = i
                        break
        if best < 0:
            # Fallback: most notes
            best = 0
            best_count = 0
            for i, a in enumerate(song.arrangements):
                c = len(a.notes) + sum(len(ch.notes) for ch in a.chords)
                if c > best_count:
                    best_count = c
                    best = i
        arr = song.arrangements[best]

        # Convert audio with unique filename (check cache first)
        audio_url = None
        audio_error: str | None = None  # Surfaced in song_info when audio_url is None
        stems_payload: list[dict] = []
        audio_id = Path(filename).stem.replace(" ", "_")

        if is_slop:
            # Stems are served via the sloppak file endpoint; the first stem
            # (or explicit default) is the core <audio> source. The stems
            # plugin replaces it with a mixed graph when active.
            from urllib.parse import quote
            q_fn = quote(filename, safe="")
            for s in loaded_slop.stems:
                url = f"/api/sloppak/{q_fn}/file/{quote(s['file'])}"
                stems_payload.append({"id": s["id"], "url": url, "default": s["default"]})
            if stems_payload:
                audio_url = stems_payload[0]["url"]
            else:
                audio_error = "This sloppak has no playable stems."
        else:
            AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Check if audio already cached (writable cache dir or legacy static dir)
            for ext in [".mp3", ".ogg", ".wav"]:
                for cache_dir in [AUDIO_CACHE_DIR, STATIC_DIR]:
                    cached_audio = cache_dir / f"audio_{audio_id}{ext}"
                    if cached_audio.exists() and cached_audio.stat().st_size > 1000:
                        audio_url = f"/audio/audio_{audio_id}{ext}"
                        break
                if audio_url:
                    break

        if not audio_url and not is_slop:
            await websocket.send_json({"type": "loading", "stage": "Converting audio..."})
            wem_files = find_wem_files(tmp)
            if not wem_files:
                audio_error = "No WEM audio files were found inside this PSARC."
            else:
                try:
                    audio_path = convert_wem(wem_files[0], os.path.join(tmp, "audio"))
                    ext = Path(audio_path).suffix
                    audio_dest = AUDIO_CACHE_DIR / f"audio_{audio_id}{ext}"
                    shutil.copy2(audio_path, audio_dest)
                    audio_url = f"/audio/audio_{audio_id}{ext}"
                except Exception as e:
                    print(f"Audio conversion failed: {e}")
                    import traceback
                    traceback.print_exc()
                    audio_error = f"Audio conversion failed: {e}"

            # Clean up old audio cache files (keep max 100)
            try:
                audio_files = [f for f in AUDIO_CACHE_DIR.iterdir()
                               if f.name.startswith("audio_") and f.suffix in (".mp3", ".ogg", ".wav")]
                if len(audio_files) > 100:
                    audio_files.sort(key=lambda f: f.stat().st_atime)
                    for f in audio_files[:len(audio_files) - 100]:
                        f.unlink(missing_ok=True)
            except Exception:
                pass

        # Send song metadata
        arr_list = [{"index": i, "name": a.name, "notes": len(a.notes) + sum(len(c.notes) for c in a.chords)}
                    for i, a in enumerate(song.arrangements)]
        await websocket.send_json({
            "type": "song_info",
            "title": song.title,
            "artist": song.artist,
            "duration": song.song_length,
            "arrangement": arr.name,
            "arrangement_index": best,
            "arrangements": arr_list,
            "audio_url": audio_url,
            "audio_error": audio_error,
            "tuning": arr.tuning,
            # Number of strings on the active arrangement
            # (slopsmith-plugin-3dhighway#7). RS XML / PSARC sources
            # always emit `tuning` as length 6 with zero-padding for
            # unused string slots, so `len(arr.tuning)` is unreliable
            # there; sloppak / GP-imported sources may instead carry
            # a trimmed list. arrangement_string_count() combines a
            # notes-derived lower bound, a name-based fallback (4 for
            # "bass" arrangements), and the tuning length (when it
            # disagrees with the RS-XML padded 6) into a single
            # reliable signal. Plugins should size string-indexed UI
            # / geometry against THIS rather than assuming 6 or
            # using `tuning.length` directly.
            "stringCount": arrangement_string_count(arr),
            "capo": arr.capo,
            "format": "sloppak" if is_slop else "psarc",
            "stems": stems_payload,
        })

        # Send beats
        beats = [{"time": b.time, "measure": b.measure} for b in song.beats]
        await websocket.send_json({"type": "beats", "data": beats})

        # Send sections
        sections = [{"name": s.name, "time": s.start_time} for s in song.sections]
        await websocket.send_json({"type": "sections", "data": sections})

        # Send anchors
        anchors = [{"time": a.time, "fret": a.fret, "width": a.width} for a in arr.anchors]
        await websocket.send_json({"type": "anchors", "data": anchors})

        # Send chord templates. Include `fingers` alongside `name` /
        # `frets` so plugin overlays consuming highway.getChordTemplates()
        # can render full chord boxes (Rocksmith-style fingering
        # diagrams), not just chord names. Each fingering entry is
        # per-string: -1 = unused, 0 = open string, n > 0 = finger
        # number. RS XML sources populate real values; GP imports
        # currently emit all -1 (no finger data available pre-import).
        templates = []
        for ct in arr.chord_templates:
            templates.append({
                "name": ct.name,
                "fingers": ct.fingers,
                "frets": ct.frets,
            })
        await websocket.send_json({"type": "chord_templates", "data": templates})

        # Send lyrics if available
        import xml.etree.ElementTree as ET
        lyrics = []
        if is_slop:
            lyrics = list(song.lyrics or [])
        else:
            for xml_path in sorted(Path(tmp).rglob("*.xml")):
                try:
                    root = ET.parse(xml_path).getroot()
                    if root.tag == "vocals":
                        for v in root.findall("vocal"):
                            lyrics.append({
                                "t": round(float(v.get("time", "0")), 3),
                                "d": round(float(v.get("length", "0")), 3),
                                "w": v.get("lyric", ""),
                            })
                        break
                except Exception:
                    pass
            if not lyrics:
                # SNG-only PSARC (official DLC) — decode vocals SNG directly.
                try:
                    from lib.sng_vocals import parse_vocals_sng
                    for sng_path in sorted(Path(tmp).rglob("*vocals*.sng")):
                        plat = "mac" if "/macos/" in str(sng_path).replace("\\", "/").lower() else "pc"
                        try:
                            lyrics = parse_vocals_sng(str(sng_path), plat)
                        except Exception:
                            lyrics = []
                        if lyrics:
                            break
                except ImportError:
                    pass
        if lyrics:
            await websocket.send_json({"type": "lyrics", "data": lyrics})

        # Send tone changes (PSARC only; sloppak has no tone XML)
        tone_changes = []
        if is_slop:
            xml_paths = []
        else:
            xml_paths = sorted(Path(tmp).rglob("*.xml"))

        # Build tone ID→name map from manifest JSON matching selected arrangement
        tone_id_map = {}  # {0: "Tone_A_name", 1: "Tone_B_name", ...}
        arr_name_lower = arr.name.lower() if arr else ""
        for jf in sorted(Path(tmp).rglob("*.json")):
            try:
                # Prefer manifest matching selected arrangement
                if arr_name_lower and arr_name_lower not in jf.stem.lower():
                    continue
                jdata = json.loads(jf.read_text())
                for entry in (jdata.get("Entries") or {}).values():
                    attrs = entry.get("Attributes") or {}
                    for idx, key in enumerate(["Tone_A", "Tone_B", "Tone_C", "Tone_D"]):
                        val = attrs.get(key, "")
                        if val:
                            tone_id_map[idx] = val
                    if tone_id_map:
                        break
            except Exception:
                continue
            if tone_id_map:
                break
        # Fallback: try any manifest if arrangement-specific one not found
        if not tone_id_map:
            for jf in sorted(Path(tmp).rglob("*.json")):
                try:
                    jdata = json.loads(jf.read_text())
                    for entry in (jdata.get("Entries") or {}).values():
                        attrs = entry.get("Attributes") or {}
                        for idx, key in enumerate(["Tone_A", "Tone_B", "Tone_C", "Tone_D"]):
                            val = attrs.get(key, "")
                            if val:
                                tone_id_map[idx] = val
                        if tone_id_map:
                            break
                except Exception:
                    continue
                if tone_id_map:
                    break

        # Parse XMLs — prefer the one matching selected arrangement, fall back to any
        # Try arrangement-matching XML first, then fall back to any
        def _xml_matches_arr(xp):
            return arr_name_lower and arr_name_lower in xp.stem.lower()
        sorted_xml = sorted(xml_paths, key=lambda xp: (0 if _xml_matches_arr(xp) else 1, xp.name))
        for xml_path in sorted_xml:
            try:
                root = ET.parse(xml_path).getroot()
                if root.tag != "song":
                    continue
                tones_el = root.find("tones")
                if tones_el is not None:
                    for t in tones_el.findall("tone"):
                        tc_time = t.get("time")
                        tc_name = t.get("name", "")
                        tc_id = t.get("id", "")
                        # Resolve "N/A" or empty names using tone ID map
                        if (not tc_name or tc_name == "N/A") and tc_id:
                            tc_name = tone_id_map.get(int(tc_id), f"Tone {tc_id}")
                        if tc_time and tc_name:
                            tone_changes.append({
                                "t": round(float(tc_time), 3),
                                "name": tc_name,
                            })
                    if tone_changes:
                        tonebase = root.find("tonebase")
                        base_name = tonebase.text if tonebase is not None and tonebase.text else ""
                        # If base name not in XML, use Tone_A from tone_id_map (same arrangement)
                        if not base_name:
                            base_name = tone_id_map.get(0, "")
                        await websocket.send_json({
                            "type": "tone_changes",
                            "base": base_name,
                            "data": sorted(tone_changes, key=lambda x: x["t"]),
                        })
                        break
            except Exception:
                pass

        # Send notes in chunks
        notes = []
        for n in arr.notes:
            notes.append({
                "t": round(n.time, 3), "s": n.string, "f": n.fret,
                "sus": round(n.sustain, 3),
                "sl": n.slide_to, "slu": n.slide_unpitch_to,
                "bn": round(n.bend, 1) if n.bend else 0,
                "ho": n.hammer_on, "po": n.pull_off,
                "hm": n.harmonic, "hp": n.harmonic_pinch,
                "pm": n.palm_mute, "mt": n.mute,
                "tr": n.tremolo, "ac": n.accent, "tp": n.tap,
            })
        # Send in chunks of 500
        for i in range(0, len(notes), 500):
            await websocket.send_json({
                "type": "notes",
                "data": notes[i:i+500],
                "total": len(notes),
            })

        # Send chords
        chords = []
        for c in arr.chords:
            chord_notes = [{
                "s": cn.string, "f": cn.fret,
                "sus": round(cn.sustain, 3),
                "bn": round(cn.bend, 1) if cn.bend else 0,
                "sl": cn.slide_to, "slu": cn.slide_unpitch_to,
                "ho": cn.hammer_on, "po": cn.pull_off,
                "hm": cn.harmonic, "hp": cn.harmonic_pinch,
                "pm": cn.palm_mute, "mt": cn.mute,
                "tr": cn.tremolo, "ac": cn.accent, "tp": cn.tap,
            } for cn in c.notes]
            chords.append({
                "t": round(c.time, 3),
                "id": c.chord_id,
                "hd": c.high_density,
                "notes": chord_notes,
            })
        for i in range(0, len(chords), 500):
            await websocket.send_json({
                "type": "chords",
                "data": chords[i:i+500],
                "total": len(chords),
            })

        # Per-phrase difficulty data for the master-difficulty slider
        # (slopsmith#48). Only sent when the source chart had multiple
        # `<level>` tiers — single-level charts (GP converter, older
        # sloppaks without phrase data) produce arr.phrases=None, and the
        # frontend treats the missing message as "slider disabled".
        # Consumers that don't know about this message type ignore it.
        #
        # Chunked at phrase granularity (20 phrases per frame) because
        # each phrase nests per-level note/chord lists — a single frame
        # could otherwise exceed proxy/WS size limits on large songs.
        # Chunk boundary is per-phrase (not per-level) so the frontend
        # reassembles whole phrase ladders.
        if arr.phrases:
            total = len(arr.phrases)
            for i in range(0, total, 20):
                await websocket.send_json({
                    "type": "phrases",
                    "data": [phrase_to_wire(p) for p in arr.phrases[i:i + 20]],
                    "total": total,
                })

        await websocket.send_json({"type": "ready"})

        # Keep connection alive for control messages
        try:
            while True:
                msg = await websocket.receive_text()
                data = json.loads(msg)
                if data.get("action") == "change_arrangement":
                    pass
        except WebSocketDisconnect:
            pass

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
        except Exception:
            pass

    finally:
        pass  # Don't clean up — cached for arrangement switching


# ── Audio serving ─────────────────────────────────────────────────────────────


@app.get("/audio/{filename:path}")
def serve_audio(filename: str):
    """Serve audio files from the writable audio cache directory."""
    for d in [AUDIO_CACHE_DIR, STATIC_DIR]:
        audio_file = d / filename
        if audio_file.exists():
            return FileResponse(str(audio_file))
    return JSONResponse({"error": "not found"}, status_code=404)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))
