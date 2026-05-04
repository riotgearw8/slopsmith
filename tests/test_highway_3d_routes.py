"""Tests for plugins/highway_3d/routes.py — upload, serve, and delete
endpoints for the 3D Highway background video slot.

Each test drives the routes directly via a minimal FastAPI app so no
full server import is needed and tests stay isolated from each other.
"""

import io
import sys
import asyncio
import importlib
from pathlib import Path

import pytest
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def setup_routes(tmp_path):
    """Import routes.py from the bundled plugin directory and call setup()
    against a fresh FastAPI app with a tmp config_dir.  Returns a tuple of
    (TestClient, upload_dir, routes_module) so tests can inspect the
    filesystem and patch module-level constants."""
    routes_path = (
        Path(__file__).parent.parent / "plugins" / "highway_3d" / "routes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "highway_3d_routes_test_module", routes_path
    )
    routes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(routes)

    app = FastAPI()
    context = {"config_dir": str(tmp_path)}
    routes.setup(app, context)

    upload_dir = tmp_path / "plugin_uploads" / "highway_3d"
    client = TestClient(app, raise_server_exceptions=True)
    try:
        yield client, upload_dir, routes, app
    finally:
        client.close()
        # Remove the test module from sys.modules so repeated runs don't
        # collide.
        sys.modules.pop("highway_3d_routes_test_module", None)


# ── Upload ────────────────────────────────────────────────────────────────────

def test_upload_mp4_creates_slot_file(setup_routes):
    client, upload_dir, _, __ = setup_routes
    data = b"\x00\x01\x02" * 100
    r = client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("clip.mp4", io.BytesIO(data), "video/mp4")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "current.mp4"
    assert body["size"] == len(data)
    assert (upload_dir / "current.mp4").read_bytes() == data


def test_upload_webm_creates_slot_file(setup_routes):
    client, upload_dir, _, __ = setup_routes
    data = b"\xef\xbf\xbd" * 50
    r = client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("clip.webm", io.BytesIO(data), "video/webm")},
    )
    assert r.status_code == 200
    assert (upload_dir / "current.webm").read_bytes() == data


def test_upload_replaces_previous_slot(setup_routes):
    client, upload_dir, _, __ = setup_routes
    first = b"first" * 10
    second = b"second" * 10
    client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("a.mp4", io.BytesIO(first), "video/mp4")},
    )
    client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("b.mp4", io.BytesIO(second), "video/mp4")},
    )
    assert (upload_dir / "current.mp4").read_bytes() == second


def test_upload_ext_change_removes_old_ext(setup_routes):
    """Uploading an .mp4 after a .webm (or vice-versa) cleans up the old slot."""
    client, upload_dir, _, __ = setup_routes
    client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("a.webm", io.BytesIO(b"webm"), "video/webm")},
    )
    assert (upload_dir / "current.webm").exists()
    client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("b.mp4", io.BytesIO(b"mp4data"), "video/mp4")},
    )
    assert (upload_dir / "current.mp4").exists()
    # Old .webm slot removed by the cleanup loop in _do_upload.
    assert not (upload_dir / "current.webm").exists()


def test_upload_rejects_disallowed_extension(setup_routes):
    client, _, __, ___ = setup_routes
    r = client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("clip.avi", io.BytesIO(b"x"), "video/x-msvideo")},
    )
    assert r.status_code == 400


def test_upload_rejects_disallowed_mime_with_allowed_ext(setup_routes):
    """If the browser sends an explicit disallowed MIME for a valid extension,
    the MIME check should reject it."""
    client, _, __, ___ = setup_routes
    r = client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("clip.mp4", io.BytesIO(b"x"), "text/html")},
    )
    assert r.status_code == 400


def test_upload_accepts_octet_stream_mime(setup_routes):
    """application/octet-stream is treated as 'browser couldn't determine
    content-type' and falls through to the extension whitelist."""
    client, upload_dir, _, __ = setup_routes
    data = b"video_bytes"
    r = client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("clip.mp4", io.BytesIO(data), "application/octet-stream")},
    )
    assert r.status_code == 200
    assert (upload_dir / "current.mp4").exists()


def test_upload_rejects_empty_file(setup_routes):
    client, _, __, ___ = setup_routes
    r = client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("clip.mp4", io.BytesIO(b""), "video/mp4")},
    )
    assert r.status_code == 400


def test_upload_oversized_returns_413_and_no_part_file(setup_routes):
    """An upload that exceeds the size cap returns 413. With the Content-Length
    pre-parse guard (Depends), honest clients are rejected before the multipart
    body is buffered. The streaming chunk-count cap remains as fallback.
    The temp .part file must never be left behind in either path."""
    client, upload_dir, routes_module, _ = setup_routes
    # Temporarily lower the cap to a tiny value so the test body stays small.
    original_cap = routes_module.MAX_VIDEO_BYTES
    routes_module.MAX_VIDEO_BYTES = 10
    try:
        oversized = b"x" * 11  # one byte over the patched cap
        r = client.post(
            "/api/plugins/highway_3d/files",
            files={"file": ("clip.mp4", io.BytesIO(oversized), "video/mp4")},
        )
    finally:
        routes_module.MAX_VIDEO_BYTES = original_cap
    assert r.status_code == 413
    # No .part temp files should have been left behind.
    part_files = list(upload_dir.glob("upload-*.part")) if upload_dir.exists() else []
    assert part_files == [], f"Leaked temp files: {part_files}"


def test_upload_content_length_preparse_guard(setup_routes):
    """Content-Length header triggers 413 before FastAPI buffers the body.

    The pre-parse Depends(_pre_parse_size_guard) reads the Content-Length
    header before FastAPI calls request.form() / python-multipart. This test
    verifies rejection happens on the header value alone by sending a request
    whose Content-Length header exceeds the cap but whose actual body is tiny
    (not a real multipart form). If the guard fires, we get 413 before any
    attempt to parse the form."""
    _, upload_dir, routes_module, app_ref = setup_routes
    original_cap = routes_module.MAX_VIDEO_BYTES
    routes_module.MAX_VIDEO_BYTES = 1024  # 1 KB cap
    try:
        async def _run():
            transport = httpx.ASGITransport(app=app_ref)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                # Content-Length claims 2 KB (above the 1 KB cap); actual
                # body is 10 bytes. The guard must fire on the header alone.
                r = await ac.post(
                    "/api/plugins/highway_3d/files",
                    content=b"x" * 10,
                    headers={
                        "content-length": str(2048),
                        "content-type": "multipart/form-data; boundary=testbound",
                    },
                )
                return r

        r = asyncio.run(_run())
    finally:
        routes_module.MAX_VIDEO_BYTES = original_cap
    assert r.status_code == 413
    # No disk I/O: the guard fired before any body parsing → no .part files.
    part_files = list(upload_dir.glob("upload-*.part")) if upload_dir.exists() else []
    assert part_files == [], f"Leaked temp files: {part_files}"


# ── GET ───────────────────────────────────────────────────────────────────────

def test_get_uploaded_file(setup_routes):
    client, upload_dir, _, __ = setup_routes
    data = b"some video bytes"
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / "current.mp4").write_bytes(data)
    r = client.get("/api/plugins/highway_3d/files/current.mp4")
    assert r.status_code == 200
    assert r.content == data
    assert r.headers["content-type"].startswith("video/mp4")
    assert r.headers.get("cache-control") == "no-cache"
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_get_returns_404_for_missing_slot(setup_routes):
    client, _, __, ___ = setup_routes
    r = client.get("/api/plugins/highway_3d/files/current.mp4")
    assert r.status_code == 404


def test_get_rejects_non_slot_filename(setup_routes):
    client, upload_dir, _, __ = setup_routes
    # Write a file the regex won't match.
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / "other.mp4").write_bytes(b"data")
    r = client.get("/api/plugins/highway_3d/files/other.mp4")
    assert r.status_code == 404


def test_get_rejects_traversal_attempt(setup_routes):
    client, _, __, ___ = setup_routes
    r = client.get("/api/plugins/highway_3d/files/../../../etc/passwd")
    # FastAPI normalizes the URL before routing, so this either 404s
    # (route not matched) or raises a validation error — either way,
    # not a 200.
    assert r.status_code != 200


# ── DELETE ────────────────────────────────────────────────────────────────────

def test_delete_removes_current_files(setup_routes):
    client, upload_dir, _, __ = setup_routes
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / "current.mp4").write_bytes(b"data")
    r = client.delete("/api/plugins/highway_3d/files")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "current.mp4" in body["deleted"]
    assert not (upload_dir / "current.mp4").exists()


def test_delete_with_both_exts_removes_all(setup_routes):
    client, upload_dir, _, __ = setup_routes
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / "current.mp4").write_bytes(b"mp4")
    (upload_dir / "current.webm").write_bytes(b"webm")
    r = client.delete("/api/plugins/highway_3d/files")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert set(body["deleted"]) == {"current.mp4", "current.webm"}
    assert not (upload_dir / "current.mp4").exists()
    assert not (upload_dir / "current.webm").exists()


def test_delete_empty_dir_returns_ok(setup_routes):
    """DELETE with nothing to remove still returns ok=True with empty lists."""
    client, _, __, ___ = setup_routes
    r = client.delete("/api/plugins/highway_3d/files")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["deleted"] == []
    assert body["leftover"] == []


# ── Upload → GET round-trip ───────────────────────────────────────────────────

def test_upload_then_get_round_trip(setup_routes):
    client, _, __, ___ = setup_routes
    data = b"\x00\xff\xab\xcd" * 32
    upload_r = client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("video.mp4", io.BytesIO(data), "video/mp4")},
    )
    assert upload_r.status_code == 200
    url_path = upload_r.json()["url"]
    # url_path is e.g. "/api/plugins/highway_3d/files/current.mp4"
    filename = url_path.rsplit("/", 1)[-1]
    get_r = client.get(f"/api/plugins/highway_3d/files/{filename}")
    assert get_r.status_code == 200
    assert get_r.content == data


# ── Single-slot invariant ─────────────────────────────────────────────────────

def test_sequential_different_ext_leaves_one_slot(setup_routes):
    """Two sequential uploads of different extensions: only the latest survives."""
    client, upload_dir, _, __ = setup_routes
    client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("a.mp4", io.BytesIO(b"mp4data"), "video/mp4")},
    )
    client.post(
        "/api/plugins/highway_3d/files",
        files={"file": ("b.webm", io.BytesIO(b"webmdata"), "video/webm")},
    )
    # Only the webm should survive; mp4 must have been cleaned up.
    assert (upload_dir / "current.webm").exists()
    assert not (upload_dir / "current.mp4").exists()
    slot_files = list(upload_dir.glob("current.*"))
    assert len(slot_files) == 1, f"Expected exactly one slot file, found: {slot_files}"


def test_concurrent_different_ext_leaves_one_slot(setup_routes):
    """Two truly concurrent uploads of different extensions must leave exactly
    one slot file on disk — the asyncio.Lock in routes.py serialises the
    commit + cleanup step so both uploads can't win simultaneously."""
    client, upload_dir, _, app_ref = setup_routes

    async def _run():
        transport = httpx.ASGITransport(app=app_ref)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            r_mp4, r_webm = await asyncio.gather(
                ac.post(
                    "/api/plugins/highway_3d/files",
                    files={"file": ("a.mp4", b"mp4data" * 500, "video/mp4")},
                ),
                ac.post(
                    "/api/plugins/highway_3d/files",
                    files={"file": ("b.webm", b"webmdata" * 500, "video/webm")},
                ),
            )
        return r_mp4, r_webm

    r_mp4, r_webm = asyncio.run(_run())
    assert r_mp4.status_code == 200
    assert r_webm.status_code == 200

    slot_files = list(upload_dir.glob("current.*"))
    assert len(slot_files) == 1, (
        f"Single-slot invariant violated: found {[f.name for f in slot_files]}"
    )
