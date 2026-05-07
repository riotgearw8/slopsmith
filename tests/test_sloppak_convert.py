"""Tests for lib/sloppak_convert.py pure helpers (sanitize_stem, _arrangement_id)
and the demucs subprocess wiring (_run_demucs PATH-pinning + error capture).
"""

import os
import subprocess
from types import SimpleNamespace

import pytest

import sloppak_convert
from sloppak_convert import sanitize_stem, _arrangement_id


# ── sanitize_stem ────────────────────────────────────────────────────────────
# Regex replaces [^A-Za-z0-9._-]+ with "_", strips leading/trailing "_",
# falls back to "song" for empty result.

SANITIZE_CASES = [
    ("cleanname", "cleanname"),                          # passthrough
    ("song_v2.mp3", "song_v2.mp3"),                      # dot and underscore preserved
    ("safe-name.ogg", "safe-name.ogg"),                  # hyphen preserved
    ("my track", "my_track"),                            # space -> underscore
    ("my   track", "my_track"),                          # run of spaces collapses to one _
    ("path/to/file", "path_to_file"),                    # slashes -> underscore
    ("weird!@#name", "weird_name"),                      # punctuation run collapses to one _
    ("  spaced  ", "spaced"),                            # leading/trailing _ stripped
    ("__both__", "both"),                                # chained _ stripped at ends
    ("___", "song"),                                     # all-underscore input -> fallback
    ("", "song"),                                        # empty input -> fallback
    ("!!!", "song"),                                     # all-forbidden-chars -> fallback
    ("ünicöde", "nic_de"),                               # non-ASCII chars collapse to "_" via [^A-Za-z0-9._-]
]


@pytest.mark.parametrize("raw,expected", SANITIZE_CASES)
def test_sanitize_stem(raw, expected):
    assert sanitize_stem(raw) == expected


# ── _arrangement_id ──────────────────────────────────────────────────────────
# Regex replaces [^a-z0-9]+ with "_" in the lowercased input, strips "_",
# falls back to "arr" for empty. On collision, appends 2/3/… and mutates
# the `used` set in place.

def test_arrangement_id_first_call_passes_through():
    used = set()
    assert _arrangement_id("Lead", used) == "lead"
    assert used == {"lead"}


def test_arrangement_id_deduplicates_on_collision():
    used = {"lead"}
    assert _arrangement_id("Lead", used) == "lead2"
    assert used == {"lead", "lead2"}


def test_arrangement_id_chains_through_multiple_collisions():
    used = {"lead", "lead2"}
    assert _arrangement_id("Lead", used) == "lead3"
    assert used == {"lead", "lead2", "lead3"}


def test_arrangement_id_lowercases_and_strips_punctuation():
    used = set()
    assert _arrangement_id("Part Bass-01", used) == "part_bass_01"


def test_arrangement_id_empty_input_falls_back_to_arr():
    used = set()
    assert _arrangement_id("", used) == "arr"
    assert used == {"arr"}


def test_arrangement_id_all_punctuation_falls_back_to_arr():
    used = set()
    assert _arrangement_id("!!!", used) == "arr"
    assert used == {"arr"}


def test_arrangement_id_mutates_the_used_set_in_place():
    used = set()
    _arrangement_id("Rhythm", used)
    _arrangement_id("Rhythm", used)
    _arrangement_id("Rhythm", used)
    assert used == {"rhythm", "rhythm2", "rhythm3"}


# ── _run_demucs PATH pinning + error capture ────────────────────────────────
# On Windows desktop builds, demucs's audio loader spawns ffprobe before
# ffmpeg. The bundled binaries live in resources/bin/, two parents up from
# this lib/ file. We pin them onto the child's PATH so demucs always finds
# them, and we fold stdout into RuntimeError messages because demucs writes
# its loader errors to stdout.

def _stub_subprocess_run(captured: dict, *, returncode=0, stdout="", stderr=""):
    def fake_run(cmd, env=None, capture_output=False, text=False, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["capture_output"] = capture_output
        captured["text"] = text
        # Guard against regressions: _run_demucs must capture both streams in
        # text mode, otherwise the RuntimeError tails come back as bytes or
        # are dropped entirely (re-introducing the empty-error-output bug
        # this PR fixes).
        assert capture_output, "_run_demucs must call subprocess.run with capture_output=True"
        assert text, "_run_demucs must call subprocess.run with text=True"
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    return fake_run


def test_run_demucs_prepends_bundled_bin_to_path(tmp_path, monkeypatch):
    """When the desktop bundle layout is detected, _run_demucs prepends
    resources/bin/ to the child env's PATH. The detection signature is
    a vgmstream-cli marker file inside the candidate bin dir."""
    fake_resources = tmp_path / "resources"
    fake_lib = fake_resources / "slopsmith" / "lib"
    fake_lib.mkdir(parents=True)
    fake_bin = fake_resources / "bin"
    fake_bin.mkdir()
    (fake_bin / "vgmstream-cli").write_text("")  # desktop-bundle marker

    monkeypatch.setattr(sloppak_convert, "__file__", str(fake_lib / "sloppak_convert.py"))
    monkeypatch.setenv("PATH", "/preexisting/path")
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))

    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run(captured))

    out_dir = tmp_path / "out"
    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    # _run_demucs raises afterwards because the result dir doesn't exist
    # under the stub — we only care about the env that was passed to run.
    with pytest.raises(RuntimeError):
        sloppak_convert._run_demucs(full_ogg, out_dir, model="htdemucs_6s")

    env = captured["env"]
    assert env is not None
    path_parts = env["PATH"].split(os.pathsep)
    assert path_parts[0] == str(fake_bin)
    assert "/preexisting/path" in path_parts


def test_run_demucs_prepends_bundled_bin_does_not_introduce_empty_pathsep(tmp_path, monkeypatch):
    """When the parent PATH is empty/missing, the prepend must not produce
    a trailing pathsep — that implicitly injects the current directory
    into the search path on some platforms."""
    fake_resources = tmp_path / "resources"
    fake_lib = fake_resources / "slopsmith" / "lib"
    fake_lib.mkdir(parents=True)
    fake_bin = fake_resources / "bin"
    fake_bin.mkdir()
    (fake_bin / "vgmstream-cli").write_text("")

    monkeypatch.setattr(sloppak_convert, "__file__", str(fake_lib / "sloppak_convert.py"))
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))

    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run(captured))

    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    with pytest.raises(RuntimeError):
        sloppak_convert._run_demucs(full_ogg, tmp_path / "out", model="htdemucs_6s")

    assert captured["env"]["PATH"] == str(fake_bin)
    assert not captured["env"]["PATH"].endswith(os.pathsep)


def test_run_demucs_preserves_windows_path_var_casing(tmp_path, monkeypatch):
    """On Windows os.environ.copy() returns a dict containing `Path`
    (the OS's native casing) rather than `PATH`. Blindly writing
    `env["PATH"] = ...` leaves the original `Path` key untouched and
    spawns a subprocess with both keys in its env block — Windows
    resolution is then implementation-defined. The implementation must
    reuse whatever PATH-equivalent key was already there."""
    fake_resources = tmp_path / "resources"
    fake_lib = fake_resources / "slopsmith" / "lib"
    fake_lib.mkdir(parents=True)
    fake_bin = fake_resources / "bin"
    fake_bin.mkdir()
    (fake_bin / "vgmstream-cli.exe").write_text("")

    monkeypatch.setattr(sloppak_convert, "__file__", str(fake_lib / "sloppak_convert.py"))
    # Remove every existing PATH-equivalent and seed only `Path` (Windows shape).
    for key in [k for k in os.environ if k.upper() == "PATH"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("Path", "C:\\Windows\\system32;C:\\Windows")
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))

    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run(captured))

    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    with pytest.raises(RuntimeError):
        sloppak_convert._run_demucs(full_ogg, tmp_path / "out", model="htdemucs_6s")

    env = captured["env"]
    # Exactly one PATH-equivalent key must exist, and it must reuse the
    # original `Path` casing rather than introduce a sibling `PATH`.
    path_keys = [k for k in env if k.upper() == "PATH"]
    assert path_keys == ["Path"], f"expected only 'Path', got {path_keys!r}"
    assert env["Path"].split(os.pathsep)[0] == str(fake_bin)
    assert "C:\\Windows\\system32" in env["Path"]


def test_run_demucs_prepends_bundled_bin_with_windows_marker(tmp_path, monkeypatch):
    """vgmstream-cli.exe (Windows desktop bundle) also satisfies the marker check."""
    fake_resources = tmp_path / "resources"
    fake_lib = fake_resources / "slopsmith" / "lib"
    fake_lib.mkdir(parents=True)
    fake_bin = fake_resources / "bin"
    fake_bin.mkdir()
    (fake_bin / "vgmstream-cli.exe").write_text("")

    monkeypatch.setattr(sloppak_convert, "__file__", str(fake_lib / "sloppak_convert.py"))
    monkeypatch.setenv("PATH", "/preexisting/path")
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))

    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run(captured))

    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    with pytest.raises(RuntimeError):
        sloppak_convert._run_demucs(full_ogg, tmp_path / "out", model="htdemucs_6s")

    assert captured["env"]["PATH"].split(os.pathsep)[0] == str(fake_bin)


def test_run_demucs_no_op_when_bundled_bin_missing(tmp_path, monkeypatch):
    """No resources/bin/ → PATH is left untouched (dev / non-desktop case)."""
    fake_lib = tmp_path / "no-bundle" / "lib"
    fake_lib.mkdir(parents=True)
    monkeypatch.setattr(sloppak_convert, "__file__", str(fake_lib / "sloppak_convert.py"))
    monkeypatch.setenv("PATH", "/preexisting/path")
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))

    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run(captured))

    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    with pytest.raises(RuntimeError):
        sloppak_convert._run_demucs(full_ogg, tmp_path / "out", model="htdemucs_6s")

    assert captured["env"]["PATH"] == "/preexisting/path"


def test_run_demucs_no_op_when_bin_lacks_desktop_marker(tmp_path, monkeypatch):
    """Docker case: parents[2] resolves to /, /bin exists with system
    binaries, but vgmstream-cli is absent — must not prepend."""
    fake_resources = tmp_path / "resources"
    fake_lib = fake_resources / "slopsmith" / "lib"
    fake_lib.mkdir(parents=True)
    fake_bin = fake_resources / "bin"
    fake_bin.mkdir()
    # Simulate /bin: ffmpeg/ffprobe present, vgmstream-cli absent.
    (fake_bin / "ffmpeg").write_text("")
    (fake_bin / "ffprobe").write_text("")

    monkeypatch.setattr(sloppak_convert, "__file__", str(fake_lib / "sloppak_convert.py"))
    monkeypatch.setenv("PATH", "/preexisting/path")
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))

    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run(captured))

    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    with pytest.raises(RuntimeError):
        sloppak_convert._run_demucs(full_ogg, tmp_path / "out", model="htdemucs_6s")

    assert captured["env"]["PATH"] == "/preexisting/path"


def test_run_demucs_failure_includes_stdout_in_error(tmp_path, monkeypatch):
    """demucs prints loader errors to stdout — they must reach the RuntimeError."""
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))
    captured: dict = {}
    monkeypatch.setattr(
        subprocess, "run",
        _stub_subprocess_run(
            captured,
            returncode=1,
            stdout="Could not load file song.ogg.\nFFmpeg is not installed.",
            stderr="",
        ),
    )

    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    with pytest.raises(RuntimeError) as excinfo:
        sloppak_convert._run_demucs(full_ogg, tmp_path / "out", model="htdemucs_6s")

    msg = str(excinfo.value)
    assert "code 1" in msg
    assert "FFmpeg is not installed" in msg
    assert "Could not load file" in msg


def test_run_demucs_failure_with_no_output_yields_sentinel(tmp_path, monkeypatch):
    """Empty stdout+stderr should still produce a meaningful error tail."""
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))
    captured: dict = {}
    monkeypatch.setattr(
        subprocess, "run",
        _stub_subprocess_run(captured, returncode=1, stdout="", stderr=""),
    )

    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    with pytest.raises(RuntimeError) as excinfo:
        sloppak_convert._run_demucs(full_ogg, tmp_path / "out", model="htdemucs_6s")

    assert "(no output)" in str(excinfo.value)


# ── _run_demucs torchaudio.save shim bootstrap ──────────────────────────────
# torchaudio>=2.11 routes .save() through save_with_torchcodec, which
# requires torchcodec. _run_demucs spawns the demucs subprocess via
# `python -c <bootstrap>` so that torchaudio.save is monkey-patched to
# soundfile.write before demucs imports. These tests pin the wire-level
# shape of that bootstrap so accidental refactors don't drop the shim.

def test_run_demucs_uses_dash_c_bootstrap_with_extra_paths_arg(tmp_path, monkeypatch):
    """cmd shape: [python, -c, <bootstrap>, <extra_paths_json>, demucs args...]."""
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run(captured))

    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    with pytest.raises(RuntimeError):
        sloppak_convert._run_demucs(full_ogg, tmp_path / "out", model="htdemucs_6s")

    cmd = captured["cmd"]
    # python interpreter, -c flag, bootstrap script, JSON-encoded extra_paths,
    # then the demucs CLI args.
    assert cmd[1] == "-c"
    bootstrap = cmd[2]
    # extra_paths arg is JSON; round-trip parses to a list of strings.
    import json as _json
    parsed = _json.loads(cmd[3])
    assert isinstance(parsed, list) and all(isinstance(p, str) for p in parsed)
    # demucs args follow.
    assert cmd[4:7] == ["-n", "htdemucs_6s", "-o"]
    # Bootstrap reads sys.argv[1] (the JSON), strips it, then runs demucs.
    assert "sys.argv[1]" in bootstrap
    assert "runpy.run_module" in bootstrap


def test_run_demucs_bootstrap_installs_torchaudio_save_shim(tmp_path, monkeypatch):
    """Bootstrap must monkey-patch torchaudio.save -> soundfile.write so the
    demucs subprocess does not depend on torchcodec being importable. This
    test pins the patch wiring so a refactor that drops the shim is caught."""
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run(captured))

    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    with pytest.raises(RuntimeError):
        sloppak_convert._run_demucs(full_ogg, tmp_path / "out", model="htdemucs_6s")

    bootstrap = captured["cmd"][2]
    # Imports the right libs.
    assert "import torchaudio" in bootstrap
    assert "import soundfile" in bootstrap or "soundfile as _sf" in bootstrap
    # Defines a replacement and assigns it.
    assert "def _ta_save" in bootstrap
    assert "_ta.save = _ta_save" in bootstrap
    # Honors channels_first kwarg (so callers passing False are not corrupted).
    assert "channels_first" in bootstrap
    # Soundfile is what actually does the write.
    assert "_sf.write" in bootstrap


def test_run_demucs_bootstrap_runs_demucs_main(tmp_path, monkeypatch):
    """Bootstrap must end by handing control to demucs's __main__ via runpy
    with run_name='__main__', otherwise demucs's CLI argparse never fires."""
    monkeypatch.setenv("CONFIG_DIR", str(tmp_path / "cfg"))
    captured: dict = {}
    monkeypatch.setattr(subprocess, "run", _stub_subprocess_run(captured))

    full_ogg = tmp_path / "song.ogg"
    full_ogg.write_bytes(b"")
    with pytest.raises(RuntimeError):
        sloppak_convert._run_demucs(full_ogg, tmp_path / "out", model="htdemucs_6s")

    bootstrap = captured["cmd"][2]
    assert "runpy.run_module" in bootstrap
    assert "demucs" in bootstrap
    assert "run_name='__main__'" in bootstrap or 'run_name="__main__"' in bootstrap
