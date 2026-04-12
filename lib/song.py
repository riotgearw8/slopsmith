"""Rocksmith 2014 arrangement XML parser and song data models."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import subprocess
import xml.etree.ElementTree as ET


@dataclass
class Note:
    time: float
    string: int
    fret: int
    sustain: float = 0.0
    slide_to: int = -1
    slide_unpitch_to: int = -1
    bend: float = 0.0
    hammer_on: bool = False
    pull_off: bool = False
    harmonic: bool = False
    harmonic_pinch: bool = False
    palm_mute: bool = False
    mute: bool = False
    tremolo: bool = False
    accent: bool = False
    link_next: bool = False
    tap: bool = False


@dataclass
class ChordTemplate:
    name: str
    fingers: list[int]
    frets: list[int]


@dataclass
class Chord:
    time: float
    chord_id: int
    notes: list[Note] = field(default_factory=list)
    high_density: bool = False


@dataclass
class Anchor:
    time: float
    fret: int
    width: int = 4


@dataclass
class Beat:
    time: float
    measure: int  # -1 for non-downbeat


@dataclass
class Section:
    name: str
    number: int
    start_time: float


@dataclass
class HandShape:
    chord_id: int
    start_time: float
    end_time: float


@dataclass
class Arrangement:
    name: str
    tuning: list[int] = field(default_factory=lambda: [0] * 6)
    capo: int = 0
    notes: list[Note] = field(default_factory=list)
    chords: list[Chord] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)
    hand_shapes: list[HandShape] = field(default_factory=list)
    chord_templates: list[ChordTemplate] = field(default_factory=list)


@dataclass
class Song:
    title: str = ""
    artist: str = ""
    album: str = ""
    year: int = 0
    song_length: float = 0.0
    offset: float = 0.0
    beats: list[Beat] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    arrangements: list[Arrangement] = field(default_factory=list)
    audio_path: str = ""
    # Optional lyrics, one entry per syllable: {"t": float, "d": float, "w": str}
    lyrics: list[dict] = field(default_factory=list)


# ── Wire format serialization (shared between highway_ws and sloppak loader) ──
#
# These helpers produce/consume the same JSON shape the highway WebSocket streams
# to the client. They are the authoritative definition of the `.sloppak`
# arrangement file format — see `arrangements/*.json` inside a sloppak.

def note_to_wire(n: Note) -> dict:
    return {
        "t": round(n.time, 3), "s": n.string, "f": n.fret,
        "sus": round(n.sustain, 3),
        "sl": n.slide_to, "slu": n.slide_unpitch_to,
        "bn": round(n.bend, 1) if n.bend else 0,
        "ho": n.hammer_on, "po": n.pull_off,
        "hm": n.harmonic, "hp": n.harmonic_pinch,
        "pm": n.palm_mute, "mt": n.mute,
        "tr": n.tremolo, "ac": n.accent, "tp": n.tap,
    }


def chord_note_to_wire(cn: Note) -> dict:
    # Chord notes omit their own time (the chord carries it).
    d = note_to_wire(cn)
    d.pop("t", None)
    return d


def chord_to_wire(c: Chord) -> dict:
    return {
        "t": round(c.time, 3),
        "id": c.chord_id,
        "hd": c.high_density,
        "notes": [chord_note_to_wire(cn) for cn in c.notes],
    }


def note_from_wire(d: dict, time: float | None = None) -> Note:
    return Note(
        time=float(d.get("t", time if time is not None else 0.0)),
        string=int(d.get("s", 0)),
        fret=int(d.get("f", 0)),
        sustain=float(d.get("sus", 0.0)),
        slide_to=int(d.get("sl", -1)),
        slide_unpitch_to=int(d.get("slu", -1)),
        bend=float(d.get("bn", 0.0)),
        hammer_on=bool(d.get("ho", False)),
        pull_off=bool(d.get("po", False)),
        harmonic=bool(d.get("hm", False)),
        harmonic_pinch=bool(d.get("hp", False)),
        palm_mute=bool(d.get("pm", False)),
        mute=bool(d.get("mt", False)),
        tremolo=bool(d.get("tr", False)),
        accent=bool(d.get("ac", False)),
        tap=bool(d.get("tp", False)),
    )


def chord_from_wire(d: dict) -> Chord:
    t = float(d.get("t", 0.0))
    return Chord(
        time=t,
        chord_id=int(d.get("id", 0)),
        high_density=bool(d.get("hd", False)),
        notes=[note_from_wire(cn, time=t) for cn in d.get("notes", [])],
    )


def arrangement_to_wire(arr: Arrangement) -> dict:
    """Serialize an Arrangement into a JSON-ready dict matching the wire format."""
    return {
        "name": arr.name,
        "tuning": list(arr.tuning),
        "capo": arr.capo,
        "notes": [note_to_wire(n) for n in arr.notes],
        "chords": [chord_to_wire(c) for c in arr.chords],
        "anchors": [{"time": a.time, "fret": a.fret, "width": a.width} for a in arr.anchors],
        "handshapes": [
            {"chord_id": h.chord_id, "start_time": h.start_time, "end_time": h.end_time}
            for h in arr.hand_shapes
        ],
        "templates": [
            {"name": ct.name, "fingers": list(ct.fingers), "frets": list(ct.frets)}
            for ct in arr.chord_templates
        ],
    }


def arrangement_from_wire(d: dict) -> Arrangement:
    """Parse a wire-format arrangement dict back into an Arrangement dataclass."""
    return Arrangement(
        name=d.get("name", ""),
        tuning=list(d.get("tuning", [0] * 6)),
        capo=int(d.get("capo", 0)),
        notes=[note_from_wire(n) for n in d.get("notes", [])],
        chords=[chord_from_wire(c) for c in d.get("chords", [])],
        anchors=[
            Anchor(time=float(a.get("time", 0)), fret=int(a.get("fret", 0)),
                   width=int(a.get("width", 4)))
            for a in d.get("anchors", [])
        ],
        hand_shapes=[
            HandShape(chord_id=int(h.get("chord_id", 0)),
                      start_time=float(h.get("start_time", 0)),
                      end_time=float(h.get("end_time", 0)))
            for h in d.get("handshapes", [])
        ],
        chord_templates=[
            ChordTemplate(name=ct.get("name", ""),
                          fingers=list(ct.get("fingers", [-1] * 6)),
                          frets=list(ct.get("frets", [-1] * 6)))
            for ct in d.get("templates", [])
        ],
    )


def _float(elem, attr, default=0.0):
    v = elem.get(attr)
    return float(v) if v is not None else default


def _int(elem, attr, default=0):
    v = elem.get(attr)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return int(float(v))


def _bool(elem, attr):
    v = elem.get(attr)
    return v is not None and v != "0"


def _parse_note(n) -> Note:
    return Note(
        time=_float(n, "time"),
        string=_int(n, "string"),
        fret=_int(n, "fret"),
        sustain=_float(n, "sustain"),
        slide_to=_int(n, "slideTo", -1),
        slide_unpitch_to=_int(n, "slideUnpitchTo", -1),
        bend=_float(n, "bend"),
        hammer_on=_bool(n, "hammerOn"),
        pull_off=_bool(n, "pullOff"),
        harmonic=_bool(n, "harmonic"),
        harmonic_pinch=_bool(n, "harmonicPinch"),
        palm_mute=_bool(n, "palmMute"),
        mute=_bool(n, "mute"),
        tremolo=_bool(n, "tremolo"),
        accent=_bool(n, "accent"),
        link_next=_bool(n, "linkNext"),
        tap=_bool(n, "tap"),
    )


def parse_arrangement(xml_path: str) -> Arrangement:
    """Parse a Rocksmith arrangement XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Name
    arr_name = ""
    el = root.find("arrangement")
    if el is not None and el.text:
        arr_name = el.text

    # Tuning
    tuning = [0] * 6
    el = root.find("tuning")
    if el is not None:
        for i in range(6):
            tuning[i] = _int(el, f"string{i}")

    # Capo
    capo = 0
    el = root.find("capo")
    if el is not None and el.text:
        try:
            capo = int(el.text)
        except ValueError:
            pass

    # Chord templates
    chord_templates = []
    container = root.find("chordTemplates")
    if container is not None:
        for ct in container.findall("chordTemplate"):
            chord_templates.append(
                ChordTemplate(
                    name=ct.get("chordName", ""),
                    fingers=[_int(ct, f"finger{i}", -1) for i in range(6)],
                    frets=[_int(ct, f"fret{i}", -1) for i in range(6)],
                )
            )

    # Merge notes per-phrase: each phrase has its own maxDifficulty, and the full
    # chart is built by taking each phrase's notes from its max difficulty level.
    # For single-level XMLs (e.g. from GP converter), skip merging and use the one level directly.
    levels_el = root.find("levels")
    phrases_el = root.find("phrases")
    phrase_iters_el = root.find("phraseIterations")

    all_levels = {}
    if levels_el is not None:
        for level in levels_el.findall("level"):
            all_levels[_int(level, "difficulty")] = level

    notes = []
    chords = []
    anchors = []
    hand_shapes = []

    def _collect_from_level(level, t_start, t_end):
        """Collect notes/chords/anchors/handshapes from a level within a time range."""
        container = level.find("notes")
        if container is not None:
            for n in container.findall("note"):
                t = _float(n, "time")
                if t_start <= t < t_end:
                    notes.append(_parse_note(n))

        container = level.find("chords")
        if container is not None:
            for c in container.findall("chord"):
                t = _float(c, "time")
                if t_start <= t < t_end:
                    chord_notes = [_parse_note(cn) for cn in c.findall("chordNote")]
                    cid = _int(c, "chordId")
                    if not chord_notes and cid < len(chord_templates):
                        ct = chord_templates[cid]
                        for s in range(6):
                            if ct.frets[s] >= 0:
                                chord_notes.append(Note(time=t, string=s, fret=ct.frets[s]))
                    chords.append(
                        Chord(
                            time=t, chord_id=cid,
                            notes=chord_notes,
                            high_density=_bool(c, "highDensity"),
                        )
                    )

        container = level.find("anchors")
        if container is not None:
            for a in container.findall("anchor"):
                t = _float(a, "time")
                if t_start <= t < t_end:
                    anchors.append(Anchor(
                        time=t, fret=_int(a, "fret"), width=_int(a, "width", 4),
                    ))

        container = level.find("handShapes")
        if container is not None:
            for hs in container.findall("handShape"):
                t = _float(hs, "startTime")
                if t_start <= t < t_end:
                    hand_shapes.append(HandShape(
                        chord_id=_int(hs, "chordId"),
                        start_time=_float(hs, "startTime"),
                        end_time=_float(hs, "endTime"),
                    )
                )

    # If there's only one level, use it directly (no per-phrase merge needed)
    if len(all_levels) == 1:
        _collect_from_level(list(all_levels.values())[0], 0.0, 99999.0)
    # Merge per-phrase if we have phrase data and multiple levels
    elif phrases_el is not None and phrase_iters_el is not None and all_levels:
        phrase_list = phrases_el.findall("phrase")
        iterations = phrase_iters_el.findall("phraseIteration")
        for i, it in enumerate(iterations):
            pid = _int(it, "phraseId")
            if pid >= len(phrase_list):
                continue
            max_diff = _int(phrase_list[pid], "maxDifficulty")
            level = all_levels.get(max_diff)
            if level is None:
                # Fall back to closest available level
                for d in range(max_diff, -1, -1):
                    if d in all_levels:
                        level = all_levels[d]
                        break
            if level is None:
                continue
            t_start = _float(it, "time")
            t_end = _float(iterations[i + 1], "time") if i + 1 < len(iterations) else 99999.0
            _collect_from_level(level, t_start, t_end)
    elif all_levels:
        # Fallback: use the level with most notes
        best_level = max(
            all_levels.values(),
            key=lambda lv: (
                int(lv.find("notes").get("count", "0")) if lv.find("notes") is not None else 0
            ) + (
                int(lv.find("chords").get("count", "0")) if lv.find("chords") is not None else 0
            ),
        )
        _collect_from_level(best_level, 0.0, 99999.0)

    notes.sort(key=lambda n: n.time)
    chords.sort(key=lambda c: c.time)
    anchors.sort(key=lambda a: a.time)
    hand_shapes.sort(key=lambda h: h.start_time)

    return Arrangement(
        name=arr_name,
        tuning=tuning,
        capo=capo,
        notes=notes,
        chords=chords,
        anchors=anchors,
        hand_shapes=hand_shapes,
        chord_templates=chord_templates,
    )


def _convert_sng_to_xml(extracted_dir: str):
    """If no arrangement XMLs exist but SNG files do, convert them via RsCli."""
    d = Path(extracted_dir)
    # Check if we already have arrangement XMLs (not just showlights/vocals)
    xml_files = list(d.rglob("*.xml"))
    has_arrangement_xml = False
    for xf in xml_files:
        try:
            root = ET.parse(xf).getroot()
            if root.tag == "song":
                el = root.find("arrangement")
                if el is not None and el.text:
                    low = el.text.lower().strip()
                    if low not in ("vocals", "showlights", "jvocals"):
                        has_arrangement_xml = True
                        break
                else:
                    has_arrangement_xml = True
                    break
        except Exception:
            continue

    if has_arrangement_xml:
        return  # Already have XMLs

    # Find SNG files (skip vocals)
    sng_files = list(d.rglob("*.sng"))
    if not sng_files:
        return

    rscli = os.environ.get("RSCLI_PATH", "")
    if not rscli or not Path(rscli).exists():
        # Try common locations (bundled, system, local)
        candidates = [
            Path(__file__).parent.parent / "tools" / "rscli" / "RsCli",
            Path(os.environ.get("PATH_BIN", "")) / "rscli" / "RsCli",
            Path("/opt/rscli/RsCli"),
            Path("./rscli/RsCli"),
        ]
        # Also check electron app's resources/bin/rscli
        if "RESOURCESPATH" in os.environ:
            candidates.insert(0, Path(os.environ["RESOURCESPATH"]) / "bin" / "rscli" / "RsCli")
        for p in candidates:
            if p.exists():
                rscli = str(p)
                break
    if not rscli:
        print("RsCli not found, cannot convert SNG to XML")
        return

    # Detect platform from directory structure
    platform = "pc"
    for sng in sng_files:
        parts = str(sng).lower()
        if "/macos/" in parts or "/mac/" in parts:
            platform = "mac"
            break

    arr_dir = d / "songs" / "arr"
    arr_dir.mkdir(parents=True, exist_ok=True)

    for sng_path in sng_files:
        stem = sng_path.stem
        if "vocals" in stem.lower():
            continue
        xml_out = arr_dir / f"{stem}.xml"
        try:
            result = subprocess.run(
                [rscli, "sng2xml", str(sng_path), str(xml_out), platform],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                print(f"sng2xml failed for {stem}: {result.stderr}")
        except Exception as e:
            print(f"sng2xml error for {stem}: {e}")


def load_song(extracted_dir: str) -> Song:
    """Load a song from an extracted PSARC directory."""
    # Convert SNG files to XML if needed (official DLC)
    _convert_sng_to_xml(extracted_dir)

    song = Song()
    xml_files = sorted(Path(extracted_dir).rglob("*.xml"))

    # Build manifest lookup: xml_stem (lowercase) -> ArrangementName
    _manifest_names = {}
    for jf in Path(extracted_dir).rglob("*.json"):
        try:
            data = json.loads(jf.read_text())
            entries = data.get("Entries") or {}
            for k, v in entries.items():
                attrs = v.get("Attributes") or {}
                arr_name = attrs.get("ArrangementName", "")
                if arr_name and arr_name not in ("Vocals", "ShowLights", "JVocals"):
                    # Match by JSON filename stem (same as XML stem)
                    _manifest_names[jf.stem.lower()] = arr_name
        except Exception:
            continue

    metadata_loaded = False
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            continue

        if root.tag != "song":
            continue

        # Skip vocals and showlights
        el = root.find("arrangement")
        if el is not None and el.text:
            low = el.text.lower().strip()
            if low in ("vocals", "showlights", "jvocals"):
                continue

        # Metadata from first valid arrangement
        if not metadata_loaded:
            for tag, attr in [
                ("title", "title"),
                ("artistName", "artist"),
                ("albumName", "album"),
            ]:
                el = root.find(tag)
                if el is not None and el.text:
                    setattr(song, attr, el.text)

            el = root.find("albumYear")
            if el is not None and el.text:
                try:
                    song.year = int(el.text)
                except ValueError:
                    pass

            el = root.find("songLength")
            if el is not None and el.text:
                song.song_length = float(el.text)

            el = root.find("offset")
            if el is not None and el.text:
                song.offset = float(el.text)

            # Beats
            container = root.find("ebeats")
            if container is not None:
                for eb in container.findall("ebeat"):
                    song.beats.append(
                        Beat(time=_float(eb, "time"), measure=_int(eb, "measure", -1))
                    )

            # Sections
            container = root.find("sections")
            if container is not None:
                for s in container.findall("section"):
                    song.sections.append(
                        Section(
                            name=s.get("name", ""),
                            number=_int(s, "number"),
                            start_time=_float(s, "startTime"),
                        )
                    )

            metadata_loaded = True

        # Parse arrangement
        arrangement = parse_arrangement(str(xml_path))

        # Try to get the correct name from the manifest JSON
        manifest_name = _manifest_names.get(xml_path.stem.lower())
        if manifest_name:
            arrangement.name = manifest_name
        else:
            # Fallback: map internal XML names to display names
            _name_map = {
                "part real_guitar": "Lead",
                "part real_guitar_22": "Rhythm",
                "part real_bass": "Bass",
                "part real_guitar_bonus": "Bonus Lead",
                "part real_bass_22": "Bass 2",
            }
            low = arrangement.name.lower().strip()
            if low in _name_map:
                arrangement.name = _name_map[low]
            elif not arrangement.name or low.startswith("part "):
                # Infer from filename
                fname = xml_path.stem.lower()
                if "lead" in fname:
                    arrangement.name = "Lead"
                elif "rhythm" in fname:
                    arrangement.name = "Rhythm"
                elif "bass" in fname:
                    arrangement.name = "Bass"
                elif "combo" in fname:
                    arrangement.name = "Combo"
                else:
                    arrangement.name = xml_path.stem

        song.arrangements.append(arrangement)

    # Sort: Lead > Combo > Rhythm > Bass > other
    priority = {"lead": 0, "combo": 1, "rhythm": 2, "bass": 3}
    song.arrangements.sort(key=lambda a: priority.get(a.name.lower(), 99))

    # Fallback: read metadata from manifest JSON files (official DLC)
    if not song.title or not song.artist:
        _load_manifest_metadata(song, extracted_dir)

    return song


def _load_manifest_metadata(song: Song, extracted_dir: str):
    """Read song metadata from manifest JSON files (used for official DLC)."""
    d = Path(extracted_dir)
    for jf in d.rglob("*.json"):
        try:
            data = json.loads(jf.read_text())
            # Manifest JSON has: Entries -> {key} -> Attributes
            entries = data.get("Entries") or data.get("entries") or {}
            if entries:
                for key, val in entries.items():
                    attrs = val.get("Attributes") or val.get("attributes") or {}
                    if not song.title and attrs.get("SongName"):
                        song.title = attrs["SongName"]
                    if not song.artist and attrs.get("ArtistName"):
                        song.artist = attrs["ArtistName"]
                    if not song.album and attrs.get("AlbumName"):
                        song.album = attrs["AlbumName"]
                    if not song.year and attrs.get("SongYear"):
                        try:
                            song.year = int(attrs["SongYear"])
                        except (ValueError, TypeError):
                            pass
                    if not song.song_length and attrs.get("SongLength"):
                        try:
                            song.song_length = float(attrs["SongLength"])
                        except (ValueError, TypeError):
                            pass
                    if song.title and song.artist:
                        return
            # Also check flat structure (individual arrangement manifests)
            attrs = data.get("Attributes") or data.get("attributes") or {}
            if attrs:
                if not song.title and attrs.get("SongName"):
                    song.title = attrs["SongName"]
                if not song.artist and attrs.get("ArtistName"):
                    song.artist = attrs["ArtistName"]
                if not song.album and attrs.get("AlbumName"):
                    song.album = attrs["AlbumName"]
                if not song.year and attrs.get("SongYear"):
                    try:
                        song.year = int(attrs["SongYear"])
                    except (ValueError, TypeError):
                        pass
                if not song.song_length and attrs.get("SongLength"):
                    try:
                        song.song_length = float(attrs["SongLength"])
                    except (ValueError, TypeError):
                        pass
                if song.title and song.artist:
                    return
        except Exception:
            continue
