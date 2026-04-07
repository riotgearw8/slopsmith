# Slopsmith

A self-contained web application for browsing, playing, and practicing Rocksmith 2014 Custom DLC (CDLC). Runs entirely in Docker — no local dependencies required.

![Library](docs/library.png)
![Player](docs/player.png)

## Features

### Library Browser
- **Grid View** — album art cards with arrangement badges, tuning, lyrics indicator
- **Artist/Album Tree View** — hierarchical browser with letter filter (A-Z), expandable artist and album groups
- **Search** — filter by song title, artist, or album name
- **Sort** — by artist, title, recently added, or tuning
- **Favorites** — mark songs with a heart, browse favorites in a dedicated view
- **Edit Metadata** — update song title, artist, album, and album art directly from the library
- **Retune to E Standard** — pitch-shift songs in Eb/D/C#/C Standard to E Standard with one click

### Note Highway Player
A real-time canvas-based note highway that renders Rocksmith arrangements as they would appear in the game.

**Note rendering:**
- Fret-positioned notes with string colors (red, orange, blue, orange, green, purple)
- Open string bars spanning the highway
- Chord brackets connecting chord notes with chord name labels
- Sustain tails that stay visible until the sustain finishes

**Techniques:**
- Bends with curved arrows and labels (1/2, full, 1-1/2, 2)
- Unison bends with dashed connector and "U" label
- Slides (diagonal arrow)
- Hammer-ons / Pull-offs / Taps (H/P/T labels)
- Palm mutes (PM label)
- Tremolo (wavy line)
- Accents (> marker)
- Harmonics (diamond shape)
- Pinch harmonics (diamond + PH label)

**Additional features:**
- Synced lyrics display (phrase-based, multi-row, karaoke highlighting) — toggleable
- Dynamic anchor zoom — fret range adjusts smoothly, looks ahead at upcoming notes
- Arrangement switcher — switch between Lead, Rhythm, Bass during playback
- Speed control — continuous slider from 0.25x to 1.50x
- Volume control

### Practice Tools
- **A-B Looping** — set start (A) and end (B) points to repeat a section
- **Saved Loops** — name and save multiple loop sections per song, persisted across sessions
- **4-Count Click** — tempo-matched metronome count-in (1-2-3-4) before each loop repetition
- **Rewind Effect** — highway smoothly rewinds to the loop start point

### CDLC Creation
- **Create from Guitar Pro Tab** — search Ultimate Guitar for GP3/GP4/GP5 tabs and convert them to playable CDLC with MIDI audio (available as a plugin)

### Compatibility
- Supports both **custom CDLC** (from CustomsForge, etc.) and **official Rocksmith DLC**
- Official DLC: automatically converts SNG binary files to XML via built-in RsCli tool
- Reads arrangement names from manifest JSON (accurate Lead/Rhythm/Bass identification)

### Scalability
- **In-memory PSARC scanning** — reads metadata without writing to disk
- **Parallel scanning** — 8-thread metadata extraction
- **Server-side pagination and search** — SQLite-backed, handles 80,000+ songs
- **Non-blocking scan** — browse already-scanned songs while import continues in background

## Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

### Run

1. Clone the repository:
   ```bash
   git clone https://github.com/byrongamatos/slopsmith.git
   cd slopsmith
   ```

2. Set your DLC folder path and start:
   ```bash
   DLC_PATH=/path/to/your/Rocksmith2014/dlc docker compose up -d
   ```

3. Open http://localhost:8000 in your browser.

On first launch, the app scans your DLC folder and imports metadata. A progress banner shows at the bottom of the screen. The library is usable while the scan runs.

### Configuration

- **DLC Folder** — set in Settings or via the `DLC_PATH` environment variable
- **Default Arrangement** — choose Lead, Rhythm, or Bass as the default when opening songs (Settings)

### Docker Compose Example

```yaml
services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      # Mount your Rocksmith DLC folder
      - /path/to/Rocksmith2014/dlc:/dlc
      # Persistent config, cache, and favorites
      - slopsmith-config:/config
      # Optional: mount plugins for development
      # - ./plugins:/app/plugins
    environment:
      - DLC_DIR=/dlc
      - CONFIG_DIR=/config

volumes:
  slopsmith-config:
```

## Plugins

Slopsmith supports a plugin system for extending functionality. Plugins can add navigation links, screens, settings sections, and API routes — all discovered automatically at startup.

### Installing a Plugin

Drop the plugin folder into the `plugins/` directory (or mount it as a Docker volume):

```
plugins/
  my_plugin/
    plugin.json
    routes.py
    screen.html
    screen.js
    settings.html  (optional)
```

Then restart the container. The plugin's nav link, screen, and settings will appear automatically.

### Plugin Structure

Each plugin requires a `plugin.json` manifest:

```json
{
  "id": "my_plugin",
  "name": "My Plugin",
  "version": "1.0.0",
  "nav": {
    "label": "My Feature",
    "screen": "my-screen"
  },
  "screen": "screen.html",
  "script": "screen.js",
  "settings": { "html": "settings.html" },
  "routes": "routes.py"
}
```

| Field      | Required | Description                                                      |
|------------|----------|------------------------------------------------------------------|
| id         | Yes      | Unique identifier, used in API paths                             |
| name       | Yes      | Display name                                                     |
| nav        | No       | Navigation link with label and screen ID                         |
| screen     | No       | HTML file for the plugin screen content                          |
| script     | No       | JavaScript file loaded after the screen is injected              |
| settings   | No       | Object with html field pointing to a settings HTML fragment      |
| routes     | No       | Python module with a setup(app, context) function for API routes |

### Plugin API Routes

The `routes.py` module must export a `setup(app, context)` function:

```python
def setup(app, context):
    config_dir = context["config_dir"]    # Path to config directory
    get_dlc_dir = context["get_dlc_dir"]  # Function returning DLC Path
    meta_db = context["meta_db"]          # MetadataDB instance
    extract_meta = context["extract_meta"] # Function to extract PSARC metadata

    @app.get("/api/plugins/my_plugin/search")
    def search(q: str):
        # Your logic here
        return {"results": []}
```

Routes are registered under `/api/plugins/{plugin_id}/` to avoid conflicts.

### Plugin Frontend

- `screen.html` — HTML fragment (no `<html>` or `<body>` tags). Injected into a `<div class="screen">` container.
- `screen.js` — JavaScript loaded after the HTML. Has access to all core functions (`showScreen()`, `esc()`, `formatTime()`, etc.).
- `settings.html` — HTML fragment injected into the Settings page.

### Available Plugins

| Plugin | Description | Install |
|--------|-------------|---------|
| [Create from Tab](https://github.com/byrongamatos/slopsmith-plugin-ug) | Search Ultimate Guitar for GP tabs and convert to playable CDLC | `git clone ...slopsmith-plugin-ug.git ultimate_guitar` |
| [Import Tab](https://github.com/byrongamatos/slopsmith-plugin-tabimport) | Drag and drop Guitar Pro files to create CDLC | `git clone ...slopsmith-plugin-tabimport.git tab_import` |
| [Practice Journal](https://github.com/byrongamatos/slopsmith-plugin-practice) | Auto-track practice time, speed, loops. Dashboard with charts | `git clone ...slopsmith-plugin-practice.git practice_journal` |
| [Setlist Builder](https://github.com/byrongamatos/slopsmith-plugin-setlist) | Create ordered playlists with sequential playback | `git clone ...slopsmith-plugin-setlist.git setlist` |
| [Metronome](https://github.com/byrongamatos/slopsmith-plugin-metronome) | Audible click and visual beat flash synced to song tempo | `git clone ...slopsmith-plugin-metronome.git metronome` |
| [Tone Player](https://github.com/byrongamatos/slopsmith-plugin-tones) | View amp/pedal/cab signal chains with Rocksmith gear artwork | `git clone ...slopsmith-plugin-tones.git tones` |
| [Fretboard View](https://github.com/byrongamatos/slopsmith-plugin-fretboard) | Live fretboard overlay showing active notes in real-time | `git clone ...slopsmith-plugin-fretboard.git fretboard` |
| [Tab View](https://github.com/byrongamatos/slopsmith-plugin-tabview) | Scrolling guitar tablature notation via alphaTab | `git clone ...slopsmith-plugin-tabview.git tab_view` |
| [MIDI Amp Control](https://github.com/byrongamatos/slopsmith-plugin-midi) | Auto-switch amp/modeler presets via MIDI on tone changes | `git clone ...slopsmith-plugin-midi.git midi_amp` |

Install any plugin by cloning it into your `plugins/` directory and restarting:

```bash
cd plugins
git clone https://github.com/byrongamatos/slopsmith-plugin-ug.git ultimate_guitar
docker compose restart
```

## Tech Stack

- **Backend**: Python / FastAPI / SQLite / WebSocket
- **Frontend**: Vanilla JS / Canvas 2D / Tailwind CSS (CDN)
- **PSARC**: Custom AES-CFB-128 decryptor with in-memory reading
- **SNG Compiler**: F# CLI tool wrapping [Rocksmith2014.NET](https://github.com/iminashi/Rocksmith2014.NET)
- **Audio**: vgmstream (WEM decode) / FFmpeg / FluidSynth (MIDI render) / rubberband (pitch shift)
- **Docker**: Self-contained image with all dependencies

## License

MIT
