"""Tests for GET /api/startup-status — shape, field types, and the
_set_startup_status / _get_startup_status state helpers introduced in
the async plugin-loading PR (slopsmith#115).
"""

import importlib
import sys
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with CONFIG_DIR isolated in a per-test tmp_path.

    SLOPSMITH_SYNC_STARTUP=1 makes the plugin-loader run synchronously
    inside startup_events() so startup is complete before TestClient.__enter__
    returns — no threading races, no polling.  load_plugins is still stubbed
    to a no-op so the "load" takes microseconds and startup_scan is also
    suppressed to avoid unrelated background I/O during tests.
    """
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SLOPSMITH_SYNC_STARTUP", "1")
    sys.modules.pop("server", None)
    server = importlib.import_module("server")
    # Stub out the two background callables that call _set_startup_status.
    # Patching at the function level (not threading.Thread) leaves TestClient
    # and AnyIO free to create real threads for their own internal use.
    monkeypatch.setattr(server, "load_plugins", lambda *a, **kw: None)
    monkeypatch.setattr(server, "startup_scan", lambda: None)
    with TestClient(server.app) as test_client:
        # With SLOPSMITH_SYNC_STARTUP the loader ran inline during startup, so
        # the status must already be complete.  Poll briefly as a safety net in
        # case something unexpected deferred the update.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not server._get_startup_status().get("running", True):
                break
            time.sleep(0.01)
        last_status = server._get_startup_status()
        assert not last_status.get("running", True), (
            f"Background startup thread did not complete within 5 s; "
            f"last status: {last_status}"
        )
        try:
            yield test_client, server
        finally:
            conn = getattr(getattr(server, "meta_db", None), "conn", None)
            if conn is not None:
                conn.close()


# ── /api/startup-status endpoint ─────────────────────────────────────────────

def test_startup_status_returns_200(client):
    tc, _ = client
    r = tc.get("/api/startup-status")
    assert r.status_code == 200


def test_startup_status_response_has_expected_keys(client):
    tc, _ = client
    data = tc.get("/api/startup-status").json()
    for key in ("running", "phase", "message", "current_plugin", "loaded", "total", "error"):
        assert key in data, f"Missing key '{key}' in /api/startup-status response"


def test_startup_status_field_types(client):
    tc, _ = client
    data = tc.get("/api/startup-status").json()
    assert isinstance(data["running"], bool)
    assert isinstance(data["phase"], str)
    assert isinstance(data["message"], str)
    assert isinstance(data["current_plugin"], str)
    assert isinstance(data["loaded"], int)
    assert isinstance(data["total"], int)
    # error is either None (JSON null) or a string
    assert data["error"] is None or isinstance(data["error"], str)


# ── _set_startup_status / _get_startup_status helpers ────────────────────────

def test_set_get_startup_status_round_trip(client):
    """_set_startup_status partial-updates the state; _get_startup_status
    returns a snapshot dict."""
    _, server = client
    server._set_startup_status(running=False, phase="complete", message="done",
                               current_plugin="", loaded=3, total=3, error=None)
    status = server._get_startup_status()
    assert status["running"] is False
    assert status["phase"] == "complete"
    assert status["loaded"] == 3
    assert status["total"] == 3
    assert status["error"] is None


def test_set_startup_status_partial_update_does_not_clobber_other_keys(client):
    """A partial _set_startup_status call must not lose previously-set keys."""
    _, server = client
    server._set_startup_status(running=True, phase="plugins-loading", message="loading",
                               current_plugin="myplugin", loaded=1, total=5, error=None)
    # Only update message.
    server._set_startup_status(message="installing requirements")
    status = server._get_startup_status()
    assert status["message"] == "installing requirements"
    assert status["phase"] == "plugins-loading"
    assert status["current_plugin"] == "myplugin"
    assert status["loaded"] == 1
    assert status["total"] == 5


def test_startup_status_endpoint_reflects_set_status(client):
    """The HTTP endpoint must reflect what was last written via _set_startup_status."""
    tc, server = client
    server._set_startup_status(running=False, phase="complete", message="All done",
                               current_plugin="", loaded=7, total=7, error=None)
    data = tc.get("/api/startup-status").json()
    assert data["running"] is False
    assert data["phase"] == "complete"
    assert data["loaded"] == 7
    assert data["total"] == 7
