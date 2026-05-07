# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Diagnostic bundle export (#166). New "Export Diagnostics" + "Preview Bundle" buttons in Settings produce a single redacted zip combining server logs (tail of `LOG_FILE`), system info (Python/OS/version), hardware probe (CPU model + cores + freq + RAM, GPU via `nvidia-smi`/`rocm-smi`/`system_profiler`, container/Electron/bare runtime detection), full plugin inventory with git SHA + remote URL (read directly from `.git/HEAD` so it works in minimal runtime images without `git` installed) + orphan/failed-to-load detection, the browser console transcript (all levels: log/info/warn/error/debug + window.onerror + unhandledrejection, 500-entry ring buffer), browser hardware (WebGL/WebGPU adapter info, navigator + userAgentData), filtered localStorage, and per-plugin contributed diagnostics. Top-level `manifest.json` lists every file with its versioned schema id (`system.hardware.v1`, `client.console.v1`, etc.) so AI agents can dispatch by schema. Redaction is on by default: DLC paths, song filenames (`<song:HASH8>` stable per-bundle), IPv4/IPv6 addresses, bearer tokens, and `key=`/`token=` query strings are replaced. Plugins opt their backend diagnostics in via a new `diagnostics` manifest field (`server_files` allowlist mirroring `settings.server_files` semantics, plus an optional `callable: "<module>:<function>"` resolved lazily via `load_sibling`). Frontend plugins push diagnostics via `window.slopsmith.diagnostics.contribute(plugin_id, payload)`. Three new endpoints: `POST /api/diagnostics/export`, `GET /api/diagnostics/preview`, `GET /api/diagnostics/hardware`. Full bundle format spec in `docs/diagnostics-bundle-spec.md`.
- Structured logging bootstrap (phase 1 of #155). Three new environment variables control server log output: `LOG_LEVEL` (default `INFO`), `LOG_FORMAT` (`text` for coloured console, `json` for one-JSON-object-per-line suitable for Loki/ELK/Promtail), and `LOG_FILE` (optional path, rotated at 10 MB with 5 backups). HTTP responses now include a `X-Request-ID` correlation header (via `CorrelationIdMiddleware`); the same request ID appears as `request_id` in structured log lines emitted via the stdlib `logging` / `structlog` APIs during that request. Code that still uses `print()` or `traceback.print_exc()` is not yet covered — those calls will be migrated in phase 2 of #155.
- **Lyrics Karaoke plugin** — end-to-end karaoke setup for Sloppak songs in one workflow. The setup screen shows a per-song checklist (vocals stem / synced lyrics / per-syllable pitch) and a single "Build Karaoke" button that runs whatever's missing: Whisper alignment of pasted lyric text against the vocals stem, then `librosa.pyin` pitch extraction. Both artifacts persist inside the Sloppak (`lyrics.json`, `vocal_pitch.json`). In the player, a "Karaoke" toggle swaps the text-lyrics overlay for a horizontal pitch ribbon (one bar per syllable, vertically positioned by pitch, sweeping playhead).
- Settings export/import (#113). Two buttons on the Settings page bundle server config, browser localStorage, and opted-in plugin server-side files into a single versioned JSON file for backup, migration, or sharing a calibrated setup. Server-side import is all-or-nothing for safety-critical failures: phase-1 validates the entire bundle (schema, path-traversal, encoding) before any disk writes; phase-2 commits each file via temp+rename. Plugin-state mismatches between export and import are handled leniently: files referenced for a plugin that isn't loaded are skipped with a warning, files referenced for a plugin whose manifest no longer declares them are skipped with a warning, and localStorage is merged (not cleared) so first-run defaults from plugins installed after the export are preserved. Path-traversal, absolute paths, schema mismatch, and decode failures remain hard refusals. Plugins opt their server-side files in by declaring `settings.server_files` in `plugin.json` (list of relpaths under `CONFIG_DIR`; trailing `/` denotes a directory).
- Library filtering by parts present or missing (#129, #69). New right-side Filters drawer (single button next to the format/sort row, with active-filter count badge and dismissible chips below) lets you require or exclude arrangements (Lead/Rhythm/Bass/Combo), specific stems on Sloppaks (drums/bass/vocals/piano/other), lyrics, and tuning. Multi-select within an axis is OR (Lead OR Rhythm); cross-axis is AND. State persists across reloads. New endpoint `GET /api/library/tuning-names` returns distinct tunings present in the library, ordered by musical distance.
- Sort library by year (#128). Two new options in the sort dropdown: "Year (newest)" and "Year (oldest)". Songs without a year are pushed to the bottom for both directions.

### Changed
- Tuning sort is now ordered by musical distance from E Standard (#22) instead of alphabetical: E Standard first, then Drop D / F Standard at distance 2, then Eb Standard / F# Standard at distance 6, etc. Matches Rocksmith's grouping. Within a magnitude tier, down-tuned variants come before up-tuned, then alphabetical.
- Settings page restructured into separate "Slopsmith" (core) and "Plugins" sections, with each plugin's settings rendered as a collapsible panel (collapsed by default). "Plugin Updates" moved into the Plugins section.
- **Lyrics Sync** is now a redirect stub. Its alignment + save endpoints moved into the new Lyrics Karaoke plugin alongside the pitch extraction. Existing nav entries and bookmarks land on a "moved" page that auto-redirects to the merged plugin.

### Fixed
- Demucs stem split failing on Windows desktop with `OSError: Could not load this library: libtorchcodec_core4.dll` or `ImportError: TorchCodec is required for save_with_torchcodec`. The demucs subprocess now bootstraps a `torchaudio.save` → `soundfile.write` shim before importing demucs, sidestepping the torchcodec dependency entirely. The override stays in place across torchaudio versions — soundfile's WAV writes are behaviorally equivalent for demucs's float32 outputs.

### Migration notes
- The library filters depend on three new columns (`stem_ids`, `tuning_name`, `tuning_sort_key`) that are populated as songs are scanned. If filters look empty after upgrading, run **Settings → Full Rescan** to repopulate; alternatively the periodic background rescan picks them up over time.

## [0.2.4] - 2026-04-22

### Added
- Version badge in navbar (`/api/version` endpoint + `VERSION` file)
- `CHANGELOG.md` and semantic versioning
- Step Mode plugin
- `gp2midi` improvements and expanded test coverage
- Note Detection plugin factory-pattern refactor with multi-instance/splitscreen support
- Per-panel note detection in Split Screen plugin with M/L/R channel routing for multi-input interfaces

### Fixed
- `SLOPPAK_CACHE_DIR` moved to `CONFIG_DIR` for AppImage compatibility
- Improved error message when plugin requirements fail to install
