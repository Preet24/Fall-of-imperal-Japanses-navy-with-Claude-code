"""Extract the loss tables from JANAC.pdf into CSV files.

The report is a scanned typewritten document with a noisy OCR text layer, so
rows are rebuilt from word coordinates rather than from plain text:

* summary pages (8-12)        -> Tables I-V, losses by sinking agent
* naval pages (15-44)         -> naval vessels sunk, chronological
* merchant pages (45-115)     -> merchant vessels sunk, chronological

Every output row keeps the raw OCR text next to the parsed values plus an
``issues`` column. A QA report checks the parsed rows against the monthly
subtotals and table totals printed in the report itself.

Usage:
    python extract_janac.py --pages 8-17 --out data/test_run --debug
    python extract_janac.py                      # all pages -> data/raw
"""

import argparse
import csv
import difflib
import json
import re
import statistics
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
KNOWN_FLAGS = ["United States", "Great Britain", "Netherlands", "Japan", "Australia",
               "New Zealand", "China", "Russia", "India", "Unknown"]
KNOWN_ASSESSMENTS = ["Sunk", "Probably sunk"]
# Vessel types as spelled in the report; OCR variants within ~85% similarity snap to these.
KNOWN_VESSEL_TYPES = [
    "Battleship", "Aircraft Carrier", "Light Aircraft Carrier", "Escort Carrier", "Seaplane Tender",
    "Heavy Cruiser", "Light Cruiser", "Old Light Cruiser", "Converted Light Cruiser", "Destroyer",
    "Old Destroyer", "Frigate", "Submarine", "Submarine Chaser", "Minesweeper", "Minelayer",
    "Repair Ship", "Gunboat", "Patrol Boat", "Torpedo Boat", "Netlayer", "Transport", "Tanker",
    "Cargo", "Passenger-Cargo", "Converted Seaplane Tender", "Converted Gunboat",
    "Aircraft Transport", "Submarine Tender", "Destroyer Transport", "Landing Ship",
    "Ammunition Ship", "Supply Ship", "Training Cruiser", "Target Ship",
    "Special Submarine Chaser", "Special Minelayer", "Special Minesweeper", "Tank Landing Ship",
    "High Speed Transport", "Escort Aircraft Carrier", "Torpedo Boat", "River Gunboat", "Yacht",
    "Salvage Vessel", "Miscellaneous Auxiliary", "Surveying Ship", "Motor Torpedo Boat",
    "District Craft", "Tug"]
# Agent names as spelled in the report (summary tables and chronological lists).
KNOWN_AGENTS = ["Submarine", "Surface Craft", "Aircraft", "Army Aircraft",
                "Navy Carrier-Based Aircraft", "Navy Land-Based Aircraft",
                "Marine Land-Based Aircraft", "Navy-Marine Land-Based Aircraft",
                "Navy Shore-Battery", "Shore Battery", "Marine Casualty", "Mine", "Army Mine",
                "Navy Mine", "Sabotage", "Unknown"]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]

# Characters the OCR commonly substitutes for digits.
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1",
                           "i": "1", "|": "1", "!": "1", "]": "1", "[": "1", "Z": "2", "z": "2", "S": "5",
                           "s": "5", "B": "8", "&": "8", "$": "8", "g": "9"})
DASH_CHARS = set("-–—=_~")

# Header anchors of the chronological tables, as offsets from the "Location"
# header (measured on pages 15 and 61); used when OCR garbles a header word.
OFFSETS_FROM_LOCATION = {"name": -294, "vtype": -173, "tonnage": -61,
                         "flag": 70, "agent": 149, "assessment": 227}

LOSS_FIELDS = ["page", "y", "date", "year", "month", "day", "vessel_name", "vessel_type",
               "tonnage", "tonnage_estimated", "location", "lat", "lon", "flag_of_agent",
               "agent_type", "agent_probable", "assessment", "issues",
               "raw_date", "raw_name", "raw_tonnage", "raw_location", "raw_flag",
               "raw_agent", "raw_assessment"]
SUMMARY_COLS = ["naval_no", "naval_tonnage", "merchant_no", "merchant_tonnage",
                "total_no", "total_tonnage"]
SUMMARY_FIELDS = (["table", "page", "y", "level", "group", "sinking_agent", "footnote"]
                  + SUMMARY_COLS + ["issues"] + [f"raw_{c}" for c in SUMMARY_COLS])


# --------------------------------------------------------------------------- words

@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    y: float = 0.0     # deskewed vertical centre
    tag: str = ""      # column/role assigned by a parser (drawn in debug images)

    @property
    def xc(self):
        return (self.x0 + self.x1) / 2

    @property
    def yc(self):
        return (self.y0 + self.y1) / 2


@dataclass
class Line:
    words: list

    @property
    def y(self):
        return statistics.fmean(w.y for w in self.words)


def estimate_slope(words):
    """Page skew (dy/dx) from words at the left and right ends of the same typed line."""
    left = [w for w in words if w.x0 < 300]
    right = [w for w in words if w.x0 > 540]
    if not left or not right:
        return 0.0
    slope = 0.0
    for _ in range(3):
        samples = []
        for a in left:
            ya = a.yc - slope * a.xc
            b = min(right, key=lambda r: abs(r.yc - slope * r.xc - ya))
            if abs(b.yc - slope * b.xc - ya) < 3.0:
                samples.append((b.yc - a.yc) / (b.xc - a.xc))
        if len(samples) < 5:
            break
        slope = statistics.median(samples)
    return slope


_OCR = None


def ocr_phrases(page, cache_dir, dpi=300):
    """RapidOCR phrases as [x0, y0, x1, y1, text, confidence] in PDF points, cached as JSON."""
    global _OCR
    path = cache_dir / f"page_{page.number + 1:03d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if _OCR is None:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
        _OCR = (RapidOCR(), np)
    engine, np = _OCR
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.stride)[:, :pix.w]
    # Pages are upright; the text-direction classifier flips some names 180° (MIZUHO -> OHOZIN).
    result, _ = engine(img, use_cls=False)
    scale = 72 / dpi
    phrases = []
    for box, text, conf in result or []:
        xs, ys = [p[0] * scale for p in box], [p[1] * scale for p in box]
        phrases.append([min(xs), min(ys), max(xs), max(ys),
                        unicodedata.normalize("NFKC", text), round(float(conf), 3)])
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(phrases), encoding="utf-8")
    return phrases


# RapidOCR often drops the space between words: "December10", "7Vessels", "I170",
# "NO9", "23-45N,155-35W". Split points inside such tokens:
GLUE = re.compile(r"(?<=[A-Za-z]{3})(?=\d)"       # December10 -> December 10
                  r"|(?<=\d)(?=[A-Z][a-z])"        # 7Vessels   -> 7 Vessels
                  r"|(?<=^I)(?=\d)|(?<=^RO)(?=\d)"  # I170, RO35 -> I 170, RO 35
                  r"|(?<=NO)(?=\d)"                 # NO9        -> NO 9
                  r"|(?<=,)(?=\d{1,3}[-~–—])")      # 45N,155-35W -> 45N, 155-35W


def split_phrase(x0, y0, x1, y1, text):
    """Typewriter text is monospaced, so word positions are proportional to character index."""
    # Leading non-ASCII junk is how RapidOCR reads the summary tables' footnote stars
    # ("米米#Allied"); keep it as a "#" marker. Other non-ASCII is stray noise.
    text = re.sub(r"^[^\x20-\x7e]+", lambda m: "#" * len(m.group()), text)
    text = re.sub(r"[^\x20-\x7e]", " ", text)
    n = len(text)
    words = []
    for m in re.finditer(r"\S+", text):
        cuts = sorted({0, len(m.group())} | {g.start() for g in GLUE.finditer(m.group())})
        for a, b in zip(cuts, cuts[1:]):
            s, e = m.start() + a, m.start() + b
            words.append(Word(x0 + (x1 - x0) * s / n, y0, x0 + (x1 - x0) * e / n, y1, text[s:e]))
    return words


def page_words(page, source="pdf", cache_dir=None):
    if source == "ocr":
        words = [w for x0, y0, x1, y1, text, _ in ocr_phrases(page, cache_dir)
                 for w in split_phrase(x0, y0, x1, y1, text)]
    else:
        words = [Word(*w[:5]) for w in page.get_text("words") if w[4].strip()]
    slope = estimate_slope(words)
    mid = page.rect.width / 2
    for w in words:
        w.y = w.yc - slope * (w.xc - mid)
    return words, slope


def group_lines(words, tol=3.8, reach=200):
    """Chain each word to its nearest left-hand neighbour on the same typed line.

    Scans are warped as well as skewed, so a row can drift by 5+ pt across the
    page; linking neighbouring words only relies on the drift being small locally.
    """
    parent = list(range(len(words)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, w in enumerate(words):
        # nearest word to the left that sits on the same line (boxes may overlap a little)
        # (a box overlapping by more than a third of the narrower word is the same
        # column on another line, e.g. a tonnage and the subtotal below it)
        left = [(max(0.0, w.x0 - v.x1), j) for j, v in enumerate(words)
                if j != i and v.x0 < w.x0 and w.x0 - v.x1 < reach and abs(v.y - w.y) <= tol
                and v.x1 - w.x0 <= min(v.x1 - v.x0, w.x1 - w.x0) / 3]
        if left:
            parent[root(i)] = root(min(left)[1])
    groups = {}
    for i, w in enumerate(words):
        groups.setdefault(root(i), []).append(w)
    lines = [Line(sorted(g, key=lambda w: w.x0)) for g in groups.values()]
    return sorted(lines, key=lambda ln: ln.y)


def similar(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ------------------------------------------------------------------ value parsing

def parse_int(text):
    t = re.sub(r"[,.\s]", "", text.strip().strip("_")).translate(DIGIT_FIX)
    return int(t) if t.isdigit() else None


def is_dash(text):
    return bool(text) and set(text) <= DASH_CHARS


def parse_tonnage(raw):
    """'8,000e' -> (8000, True). Returns (None, False) when unreadable."""
    t = raw.replace(" ", "")
    estimated = len(t) > 1 and t[-1] in "ecg@"          # the "e" marker, as OCR misreads it
    # The report writes 1,000+ with a comma, so "8340" is "834e" with the e read as 0.
    if not estimated and re.fullmatch(r"\d{3}0", t):
        estimated = True
        t = t[:-1] + "e"
    if estimated:
        t = t[:-1]
    return parse_int(t), estimated


# OCR misreadings of the hemisphere letter. Longitude can never be N/S, so an
# S or 5 there must be a garbled E.
LAT_HEMI = {"M": "N", "H": "N", "K": "N", "»": "N", "W": "N", "!": "N", "1": "N", "11": "N",
            "$": "S", "5": "S", "8": "S", "3": "S"}
# The typewriter's 4 comes out as L or /; ] is a 1; C is a 0.
COORD_DIGIT_FIX = str.maketrans({"A": "4", "L": "4", "/": "4", "J": "1", "]": "1", "C": "0"})
LON_HEMI = {"S": "E", "5": "E", "B": "E", "F": "E", "T": "E", "R": "E", "3": "E", "8": "E",
            "G": "E", "V": "W", "11": "W", "M": "W"}
# Japanese losses stop around 25 S, so an unreadable hemisphere above that latitude is N.
NORTH_ONLY_ABOVE = 25
# RapidOCR reads the typewriter E as B throughout; that one is not worth a review flag.
ROUTINE_HEMI_FIXES = {"B"}


def parse_coord(token, hemispheres, limit, fixes):
    """'138-14E' -> (138.2333, corrected?)."""
    # minutes: two characters (OCR may garble one: "1C"), else a single digit ("8-9S");
    # the hemisphere letter is required first, and only then allowed to be missing
    m = re.match(r"^(.{1,3}?)[-~–—^+]+(.{2}|\d)(.{1,2}?)\W*$", token)
    result = m and _coord_value(m, hemispheres, limit, fixes)
    if result and result[0] is not None:
        return result
    m = re.match(r"^(.{1,3}?)[-~–—^+]+(.{2})()\W*$", token)
    return _coord_value(m, hemispheres, limit, fixes) if m else (None, False)


def _coord_value(m, hemispheres, limit, fixes):
    deg = parse_int(m.group(1).translate(COORD_DIGIT_FIX))
    mins = parse_int(m.group(2).translate(COORD_DIGIT_FIX))
    hemi = m.group(3).upper()
    if len(hemi) == 2 and hemi[-1] in hemispheres:
        hemi = hemi[-1]                                   # junk before the letter: "1N"
    if hemi == "" and hemispheres == "EW" and deg is not None and deg >= 90:
        hemi = "?E"                                       # letter lost; Japanese losses at 90-180 are E
    corrected = hemi not in hemispheres and not (hemispheres == "EW" and hemi in ROUTINE_HEMI_FIXES)
    hemi = "E" if hemi == "?E" else fixes.get(hemi, hemi)
    if hemi not in hemispheres and hemispheres == "NS" and deg is not None and deg > NORTH_ONLY_ABOVE:
        hemi = "N"
    if deg is None or mins is None or hemi not in hemispheres or deg > limit or mins >= 60:
        return None, False
    value = deg + mins / 60
    return round(-value if hemi in "SW" else value, 4), corrected


def parse_location(raw):
    """'34-26N, 138-14E' -> (34.4333, 138.2333, issue). Named areas have no coords and no issue."""
    if not re.search(r"\d", raw):
        return None, None, None
    raw = re.sub(r"(\d)\.(?=\d)", r"\1", raw)                  # "1.27-51E" -> "127-51E"
    raw = re.sub(r",\.", ", ", raw)                             # "24-26N,.120-26E"
    raw = re.sub(r"\b(\d{1,3})(\d{2})([NSEW])\b", r"\1-\2\3", raw)   # "1900N" -> "19-00N"
    raw = re.sub(r"(\d),(\d{2}[NS])", r"\1-\2", raw)          # "16,52N" -> "16-52N"
    raw = re.sub(r"(\d{2}[NS])(\d)", r"\1 \2", raw)           # "3-04S150-42E" -> "3-04S 150-42E"
    parts = [p for p in re.split(r"[,;\s]+", raw.strip()) if p]
    if len(parts) > 2:                       # stray OCR specks: "1-22N,1 172-47E"
        parts = [p for p in parts if re.search(r"\d[-~–—^]\d", p)]
    if len(parts) == 2:
        lat, fix_a = parse_coord(parts[0], "NS", 90, LAT_HEMI)
        lon, fix_b = parse_coord(parts[1], "EW", 180, LON_HEMI)
        if lat is not None and lon is not None:
            return lat, lon, "location_hemisphere_guessed" if fix_a or fix_b else None
    return None, None, "location_unreadable"


def clean_vessel_type(raw):
    # digits inside words are OCR slips ("01d Destroyer"); then snap near-misses ("Deatroyer")
    text = " ".join(t.translate(str.maketrans("01", "Ol")) if not t.isdigit() else t
                    for t in raw.split())
    return snap(text, KNOWN_VESSEL_TYPES, 0.7)       # exact names always win; "Prigato" -> Frigate


def clean_name(raw):
    raw = re.sub(r"(?i)PATROL\s*BOAT", "PATROL BOAT ", raw)        # RapidOCR glues these
    raw = re.sub(r"(?i)SUBMARINE\s*CHASER", "SUBMARINE CHASER ", raw)
    raw = re.sub(r"(?i)(BOAT|CHASER|SWEEPER|LAYER|MARU)\s*NO\b", r"\1 NO", raw)
    toks = re.sub(r"[^A-Za-z0-9\s-]", " ", raw).upper().split()
    if len(toks) == 2 and toks[0] in ("1", "L", "T") and toks[1].isdigit():
        toks[0] = "I"                                           # submarine "1 30" -> "I 30"
    if len(toks) == 1 and re.fullmatch(r"1\d{1,3}", toks[0]):
        toks = ["I", toks[0][1:]]                               # "1164" -> "I 164"
    if len(toks) == 1 and re.fullmatch(r"R[O0]\d{1,3}", toks[0]):
        toks = ["RO", toks[0][2:]]                              # "R0100" -> "RO 100"
    for i, tok in enumerate(toks):
        if i > 0 and len(tok) == 4 and tok[1:] == "ARU":       # VARU, NARU -> MARU
            toks[i] = "MARU"
        if (i >= 1 and i == len(toks) - 2 and toks[-1].isdigit() and len(tok) == 2
                and tok[1] in "O0" and tok != "RO"):
            toks[i] = "NO"                                      # HO 5, N0 5 -> NO 5
    if "MARU" not in toks and toks:
        # MARU read as "ARU"/"LIARU" or glued on: "TOEI ARU", "KOMAKIYARU" -> "KOMAKI MARU"
        for i, tok in enumerate(toks):
            if i > 0 and re.fullmatch(r"[A-Z]{0,2}ARU", tok):
                toks[i] = "MARU"
                break
            if i == len(toks) - 1 or (i + 2 == len(toks) and toks[-1] == "NO"):
                if len(tok) > 6 and re.search(r"[^M]ARU$", tok):
                    toks[i:i + 1] = [tok[:-4], "MARU"]
                    break
        if ("MARU" not in toks and len(toks) >= 2 and 3 <= len(toks[-1]) <= 5
                and re.search(r"AR|RU", toks[-1]) and similar(toks[-1], "MARU") >= 0.5):
            toks[-1] = "MARU"                                    # "NIKKYO FAR"
    toks = ["UNKNOWN" if similar(t, "UNKNOWN") >= 0.7 else t for t in toks]
    if len(toks) >= 2 and toks[-2] == "NO" and not toks[-1].isdigit():
        fixed = toks[-1].replace("L", "1").translate(DIGIT_FIX)  # "NO 1L2" -> "NO 112"
        if fixed.isdigit():
            toks[-1] = fixed
    name = " ".join(toks)
    # dots, commas and quotes are harmless OCR specks; other junk or lower case is not
    raw = re.sub(r"[!\"']", " ", raw)                            # specks before a fixed "MARU"
    suspect = bool(re.search(r"[^A-Za-z0-9\s.,'-]", raw)) or bool(re.search(r"[a-z]", raw)) and not (
        toks and toks[-1].isdigit() and re.search(r"[a-z]", raw.split()[-1])) and not (
        "MARU" in toks and not re.search(r"[a-z]", raw.replace("Li", "")))
    return name, suspect


def snap(text, vocabulary, cutoff=0.8):
    hit = difflib.get_close_matches(text, vocabulary, n=1, cutoff=cutoff)
    return hit[0] if hit else text


def join_frags(col, frags):
    texts = [t for _, t in sorted(frags)]
    out = texts[0]
    for t in texts[1:]:
        if out.endswith("-"):
            out += t
        else:
            out += ("; " if col in ("agent", "flag") else " ") + t
    return out


# ------------------------------------------------------------- chronological lists

def name_column_start(words, lo, hi):
    """Left edge of the vessel-name column: where several ALL-CAPS words line up."""
    xs = sorted(w.x0 for w in words
                if lo < w.x0 < hi and re.fullmatch(r"[A-Z][A-Z0-9-]+", w.text))
    for x in xs:
        if sum(x <= v <= x + 3 for v in xs) >= 3:
            return x
    return None


def chrono_layout(words, height):
    """Column boundaries (minimum word x0 per column) from the page's header row."""
    top = [w for w in words if w.y < height * 0.3]
    loc = max(top, key=lambda w: similar(w.text, "Location"), default=None)
    if loc is None or similar(loc.text, "Location") < 0.75:
        return None
    band = [w for w in top if abs(w.y - loc.y) < 8]
    header_bottom = max(w.y1 for w in band)

    def find(target, cond=lambda w: True, cutoff=0.7):
        cands = [w for w in band if cond(w) and similar(w.text, target) >= cutoff]
        return max(cands, key=lambda w: similar(w.text, target)).x0 if cands else None

    a = {"location": loc.x0,
         "name": find("Name"),
         "vtype": find("Type", lambda w: w.x0 < loc.x0),
         "tonnage": find("Tonnage"),
         "flag": find("Flag"),
         "agent": find("Type", lambda w: w.x0 > loc.x0),
         "assessment": find("Assessment", cutoff=0.6)}
    for k, off in OFFSETS_FROM_LOCATION.items():
        if a[k] is None:
            a[k] = loc.x0 + off
    bounds = {"name": a["name"] - 24, "vtype": a["vtype"] - 30, "tonnage": a["tonnage"] - 25,
              "location": a["location"] - 20, "flag": a["flag"] - 8,
              "agent": a["agent"] - 15, "assessment": a["assessment"] - 12}
    data = [w for w in words if w.y0 > header_bottom]
    start = name_column_start(data, bounds["name"] - 30, bounds["vtype"])
    if start is not None:
        bounds["name"] = start - 4
    return bounds, header_bottom


def digit_edits(a, b):
    """Levenshtein distance between two digit strings."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def looks_numeric(text):
    core = re.sub(r"[,.]", "", text).translate(DIGIT_FIX)
    return bool(core) and sum(c.isdigit() or c == "?" for c in core) >= len(core) / 2


def chrono_column(w, b):
    x = w.x0
    if x >= b["assessment"]:
        return "assessment"
    if x >= b["agent"]:
        return "agent"
    if x >= b["flag"]:
        return "flag"
    if x >= b["location"]:
        return "location"
    if x >= b["tonnage"] and looks_numeric(w.text):
        return "tonnage"
    if x >= b["vtype"]:
        return "vtype"
    if x >= b["name"]:
        return "name"
    return "date"


@dataclass
class Item:
    kind: str          # main | cont | subtotal | skip
    y: float
    cells: dict
    frags: dict = field(default_factory=dict)


def is_subtotal_label(text):
    """'8 Vessels', OCR'd as '8 Veasela', '8Vessels', '4 Vessel 8' (from '4Vessel8') ..."""
    m = re.match(r"^\s*\d\S{0,2}?\s*([A-Za-z].*)$", text)
    if not m:
        return False
    word = re.sub(r"\s", "", m.group(1))
    return 5 <= len(word) <= 9 and similar(word, "Vessels") >= 0.6


def classify(cells):
    joined = " ".join(cells.values())
    if "probable agent" in joined.lower():
        return "skip"
    if set(cells) <= {"agent", "assessment"} and re.fullmatch(r"[0-9ivxl]+", joined.strip()):
        return "skip"                                          # page number
    if not cells.get("name") and is_subtotal_label(cells.get("vtype", "")):
        return "subtotal"
    # "120  419,112" with the word "Vessels" lost by OCR
    if (not cells.get("name") and not cells.get("date") and not cells.get("location")
            and re.fullmatch(r"\d{1,3}", cells.get("vtype", "").strip()) and cells.get("tonnage")):
        return "subtotal"
    has_name = len(re.sub(r"\W", "", cells.get("name", ""))) >= 2
    has_data = any(cells.get(c) for c in ("date", "vtype", "tonnage", "location"))
    if (has_name and has_data) or (cells.get("tonnage") and cells.get("location")):
        return "main"
    # a name with nothing else on its line is the second line of a long name
    # ("SPECIAL SUBMARINE CHASER" / "NO 30")
    if set(cells) == {"date"}:
        return "skip"
    return "cont"


class ChronoParser:
    def __init__(self, section, complete):
        self.section = section
        self.rows = []
        self.checks = []
        self.notes = []
        self.year = self.month = self.last_day = None
        self.since_subtotal = []
        self.partial = not complete          # run started mid-section

    def parse_page(self, pno, words, height):
        layout = chrono_layout(words, height)
        if layout is None:
            return 0
        bounds, header_bottom = layout
        data = []
        for w in words:
            if w.y0 > header_bottom:
                w.tag = chrono_column(w, bounds)
                data.append(w)
            else:
                w.tag = "header"

        items = []
        for ln in group_lines(data):
            cells = {}
            for w in ln.words:
                cells.setdefault(w.tag, []).append(w.text)
            cells = {k: " ".join(v) for k, v in cells.items()}
            it = Item(classify(cells), ln.y, cells)
            if it.kind == "main":
                it.frags = {c: [(it.y, t)] for c, t in cells.items()}
            elif it.kind == "subtotal":
                # a wrapped agent line can sit level with the subtotal; split it off
                extra = {c: cells.pop(c) for c in ("flag", "agent", "assessment") if c in cells}
                if extra:
                    items.append(Item("cont", ln.y, extra))
            items.append(it)

        self.merge_split_rows(items)
        self.attach_continuations(pno, items)
        n = 0
        for it in items:
            if it.kind == "main":
                row = self.make_row(pno, it)
                self.rows.append(row)
                self.since_subtotal.append(row)
                n += 1
            elif it.kind == "subtotal":
                self.check_subtotal(pno, it)
        return n

    @staticmethod
    def merge_split_rows(items):
        """On warped pages one row can come out as two lines: date + name, then type +
        tonnage + position slightly lower. Merge the nameless half into the row above."""
        for i in range(1, len(items)):
            it, prev = items[i], items[i - 1]
            if (it.kind == "main" and prev.kind == "main" and not it.cells.get("name")
                    and not it.cells.get("date") and not prev.cells.get("tonnage")
                    and not prev.cells.get("location") and it.y - prev.y < 12):
                for c, t in it.cells.items():
                    prev.cells[c] = (prev.cells[c] + " " + t) if c in prev.cells else t
                    prev.frags.setdefault(c, []).append((it.y, t))
                it.kind = "skip"

    def attach_continuations(self, pno, items):
        """Multi-line cells always stack downward from the record's main line, so a
        continuation line belongs to the record above it. Lines above the first record
        on a page continue the last record of the previous page."""
        mains = [i for i, it in enumerate(items) if it.kind == "main"]
        subtotals = [i for i, it in enumerate(items) if it.kind == "subtotal"]
        for i, it in enumerate(items):
            if it.kind != "cont":
                continue
            p = next((j for j in reversed(mains) if j < i), None)
            if p is not None and not any(p < s < i for s in subtotals):
                for c, t in it.cells.items():
                    items[p].frags.setdefault(c, []).append((it.y, t))
            elif p is None and not any(s < i for s in subtotals) and self.rows:
                self.extend_previous_page_row(pno, it)
            else:
                self.notes.append(f"page {pno}: orphan line at y={it.y:.0f}: {it.cells}")

    def extend_previous_page_row(self, pno, it):
        row = self.rows[-1]
        for c, field_, raw_field in (("agent", "agent_type", "raw_agent"),
                                     ("flag", "flag_of_agent", "raw_flag")):
            if c in it.cells:
                sep = "" if row[raw_field].endswith("-") else "; "
                row[raw_field] += sep + it.cells[c]
                vocab = KNOWN_AGENTS if c == "agent" else KNOWN_FLAGS
                row[field_] = "; ".join(snap(p.strip(), vocab, 0.75) for p in
                                        re.sub(r"\*", "", row[raw_field]).split(";") if p.strip())
        row["issues"] = ";".join(filter(None, [row["issues"], "continued_on_next_page"]))
        self.notes.append(f"page {pno}: top line {it.cells} added to {row['vessel_name']} (page {row['page']})")

    def read_date(self, raw):
        issues = []
        m = y = d = None
        # "September-1942" -> "September - 1942"; "121942" (day glued to year) -> "12 1942"
        raw = re.sub(r"(?<=[A-Za-z])-|-(?=19\d\d)", " - ", raw)
        raw = re.sub(r"\b(\d{1,2})[.,]?(194\d)\b", r"\1 \2", raw)     # also "1.1944"
        no_day = False
        for tok in raw.split():
            if is_dash(tok):              # the report prints "-" when the day is not known
                no_day = True
                continue
            letters = re.sub(r"[^A-Za-z]", "", tok)
            if len(letters) >= 3 and not re.search(r"\d", tok):
                hit = difflib.get_close_matches(letters.capitalize(), MONTHS, n=1, cutoff=0.6)
                if hit:
                    m = MONTHS.index(hit[0]) + 1
                else:
                    issues.append("date_unreadable")
                continue
            v = parse_int(tok.replace("-", ""))
            if v is None:
                issues.append("date_unreadable")
            elif 1940 <= v <= 1946:
                y = v
            elif 1 <= v <= 31:
                d = v
            else:
                issues.append("date_unreadable")

        if m is None and y is not None:          # month name lost in OCR
            # Pages repeat the current month at the top, so the day order decides:
            # a day that continues the sequence stays in the current month.
            if self.month is None:
                m = 12
            elif d is not None and self.last_day is not None and d >= self.last_day:
                m = self.month
            else:
                m = self.month % 12 + 1
            issues.append("month_inferred")
        if m is not None:
            if y is None and self.year is not None:
                y = self.year + (1 if self.month and m < self.month else 0)
            if (y, m) != (self.year, self.month):
                self.last_day = None
            self.year, self.month = y, m
        if self.month is None or self.year is None:
            issues.append("month_unknown")
        if d is None:
            issues.append("day_not_given" if no_day else "day_missing")
        elif self.last_day and d < self.last_day:
            issues.append("day_out_of_order")
        if d is not None:
            self.last_day = d
        return self.year, self.month, d, issues

    def make_row(self, pno, it):
        f = {c: join_frags(c, frags) for c, frags in it.frags.items()}
        year, month, day, issues = self.read_date(f.get("date", ""))
        iso = ""
        if year and month and day:
            try:
                iso = date(year, month, day).isoformat()
            except ValueError:
                issues.append("date_invalid")

        name, suspect = clean_name(f.get("name", ""))
        if suspect:
            issues.append("name_ocr_suspect")
        tonnage, estimated = parse_tonnage(f.get("tonnage", ""))
        if tonnage is None:
            issues.append("tonnage_unreadable" if f.get("tonnage") else "tonnage_missing")
        lat, lon, loc_issue = parse_location(f.get("location", ""))
        if loc_issue:
            issues.append(loc_issue)
        if not f.get("location"):
            issues.append("location_missing")

        raw_agent = f.get("agent", "")
        flags = "; ".join(snap(p.strip(), KNOWN_FLAGS) for p in f.get("flag", "").split(";") if p.strip())
        raw_assess = f.get("assessment", "").strip()
        assessment = ("Probably sunk" if raw_assess.lower().startswith("prob")
                      else "Sunk" if re.fullmatch(r"(?i)sun\W?\w?\W?", raw_assess)     # "Sunl:"
                      # "Sunls", "Suric", "Sahk", "Curk", "Sanr" - a 4-5 letter word near "Sunk"
                      else "Sunk" if 3 <= len(raw_assess) <= 5 and similar(raw_assess, "Sunk") >= 0.4
                      else snap(raw_assess, KNOWN_ASSESSMENTS, 0.7))
        if assessment not in KNOWN_ASSESSMENTS:
            issues.append("assessment_unknown")

        return {
            "page": pno, "y": round(it.y), "date": iso, "year": year, "month": month, "day": day,
            "vessel_name": name, "vessel_type": clean_vessel_type(f.get("vtype", "").strip()),
            "tonnage": tonnage, "tonnage_estimated": estimated,
            "location": f.get("location", "").strip(), "lat": lat, "lon": lon,
            "flag_of_agent": flags,
            # the "probable agent" star; RapidOCR reads it as "#" (17 marks = Table V's 17)
            "agent_type": "; ".join(snap(p.strip(), KNOWN_AGENTS, 0.75)
                                    for p in re.sub(r"[*#]", "", raw_agent).split(";") if p.strip()),
            "agent_probable": bool(re.search(r"[*#]", raw_agent)),
            "assessment": assessment, "issues": ";".join(dict.fromkeys(issues)),
            "raw_date": f.get("date", ""), "raw_name": f.get("name", ""),
            "raw_tonnage": f.get("tonnage", ""), "raw_location": f.get("location", ""),
            "raw_flag": f.get("flag", ""), "raw_agent": raw_agent,
            "raw_assessment": f.get("assessment", ""),
        }

    @staticmethod
    def tonnage_repair(rows, gap):
        """Find the one row whose tonnage, corrected by `gap`, matches what OCR saw up to
        a single-digit slip. Returns (row, value) or None.

        1. The month's last tonnage is underlined, which garbles its OCR badly; accept the
           implied value there if it merely resembles the OCR text.
        2. Elsewhere OCR drops, adds or swaps one digit (7,812 -> 781; 4,000 -> 40,000):
           accept only if exactly one row's implied value is one digit edit away.
        """
        def seen(r):
            return re.sub(r"\D", "", r["raw_tonnage"].translate(DIGIT_FIX))

        def implied(r):
            return (r["tonnage"] or 0) + gap

        last = rows[-1]
        v, s = implied(last), seen(last)
        if 0 < v < 80000 and (last["tonnage"] is None or similar(str(v), s) >= 0.5
                              or (s and str(v).startswith(s))):
            return last, v
        hits = [(r, implied(r)) for r in rows
                if 0 < implied(r) < 80000 and digit_edits(str(implied(r)), seen(r)) <= 1]
        return hits[0] if len(hits) == 1 else None

    def check_subtotal(self, pno, it):
        rows = self.since_subtotal
        stated_count = parse_int(it.cells.get("vtype", "").split()[0])
        stated_ton, _ = parse_tonnage(it.cells.get("tonnage", ""))
        parsed_ton = sum(r["tonnage"] for r in rows if r["tonnage"] is not None)
        if (not self.partial and rows and stated_count == len(rows)
                and stated_ton is not None and stated_ton != parsed_ton):
            fix = self.tonnage_repair(rows, stated_ton - parsed_ton)
            if fix:
                row, value = fix
                row["tonnage"] = value
                row["issues"] = ";".join(i for i in row["issues"].split(";")
                                         if i and not i.startswith("tonnage_"))
                row["issues"] = ";".join(filter(None, [row["issues"], "tonnage_from_subtotal"]))
                parsed_ton = stated_ton
        last = rows[-1] if rows else {}
        for r in rows:
            r["block"] = len(self.checks)          # which subtotal this row is counted in
        self.checks.append({
            "section": self.section, "page": pno,
            "month": f"{last.get('year')}-{last.get('month') or 0:02d}" if rows else "?",
            "stated_count": stated_count, "parsed_count": len(rows),
            "count_ok": stated_count == len(rows),
            "stated_tonnage": stated_ton, "parsed_tonnage": parsed_ton,
            "tonnage_ok": stated_ton == parsed_ton,
            "rows_unreadable_tonnage": sum(r["tonnage"] is None for r in rows),
            "partial": self.partial,
        })
        if not self.partial:
            check = self.checks[-1]
            flag = (None if check["count_ok"] and check["tonnage_ok"] else
                    "month_count_mismatch" if not check["count_ok"] else "month_tonnage_mismatch")
            if flag:
                for r in rows:
                    r["issues"] = ";".join(filter(None, [r["issues"], flag]))
        self.since_subtotal = []
        self.partial = False


# ------------------------------------------------------------------ summary tables

def is_num_token(w, zone):
    t = w.text
    if re.search(r"\d", t) or is_dash(t):
        return True
    # short OCR junk inside the number columns (e.g. "ll" for 11)
    return w.x0 >= zone and len(t) <= 2 and t.lower() not in {"of", "by", "in", "an"}


class SummaryParser:
    def __init__(self):
        self.rows = []
        self.checks = []
        self.notes = []

    def parse_page(self, pno, words, table_id):
        tons = sorted((w for w in words if similar(w.text, "Tonnage") >= 0.7), key=lambda w: w.y)[:3]
        nos = sorted((w for w in words if re.fullmatch(r"[NHM][oO0][.,]?", w.text)), key=lambda w: w.y)[:3]
        if len(tons) < 3 or not nos:
            return 0
        tons.sort(key=lambda w: w.x0)
        nos = self.complete_no_headers(pno, tons, nos)
        # (column, reference x, compare by) - counts are centred, tonnages right-aligned
        cols = []
        for no, ton, prefix in zip(nos, tons, ("naval", "merchant", "total")):
            cols += [(f"{prefix}_no", no.xc, "center"), (f"{prefix}_tonnage", ton.x1, "right")]
        header_bottom = max(w.y1 for w in tons + nos)
        zone = nos[0].x0 - 15

        data = []
        for w in words:
            if w.y0 > header_bottom + 1:
                data.append(w)
            else:
                w.tag = "header"

        def column_of(w):
            return min(cols, key=lambda c: abs((w.xc if c[2] == "center" else w.x1) - c[1]))[0]

        # Each table row has exactly one label line, so labels anchor the rows: every
        # number goes to the nearest label line, at most one number per column per row.
        label_words = [w for w in data if not is_num_token(w, zone)]
        num_words = [w for w in data if is_num_token(w, zone)]
        for w in label_words:
            w.tag = "label"
        for w in num_words:
            w.tag = "num"
        rows_ = [Line(ln.words) for ln in group_lines(label_words, reach=400)]
        slots = {}                                      # (row index, column) -> word
        orphans, lost = [], {}                          # lost: id(word) -> rows it lost
        queue = sorted(num_words, key=lambda w: w.y)
        while queue:
            w = queue.pop(0)
            col = column_of(w)
            cands = sorted((abs(r.y - w.y), i) for i, r in enumerate(rows_)
                           if i not in lost.get(id(w), ()) and abs(r.y - w.y) <= 7)
            if not cands:
                orphans.append(w)
                continue
            d, i = cands[0]
            other = slots.get((i, col))
            if other is None:
                slots[(i, col)] = w
            elif d < abs(rows_[i].y - other.y):          # closer word takes the slot
                slots[(i, col)] = w
                lost.setdefault(id(other), set()).add(i)
                queue.append(other)
            else:
                lost.setdefault(id(w), set()).add(i)
                queue.append(w)
        lines = [Line(r.words + [w for (i, _), w in slots.items() if i == k]) for k, r in enumerate(rows_)]
        lines += [Line([w]) for w in orphans]
        lines.sort(key=lambda ln: ln.y)

        parsed = []
        pending, pending_x = "", None
        for ln in lines:
            ln.words.sort(key=lambda w: w.x0)
            label_ws = [w for w in ln.words if w.tag == "label"]
            num_ws = [w for w in ln.words if w.tag == "num"]
            marks = ""
            while label_ws and not re.search(r"[A-Za-z]{2}", label_ws[0].text):
                marks += label_ws.pop(0).text
            if label_ws:                               # markers glued on: "#United", "**Allied"
                lead = re.match(r"^[^A-Za-z]*", label_ws[0].text).group()
                if lead:
                    marks += lead
                    w0 = label_ws[0]
                    x0 = w0.x0 + (w0.x1 - w0.x0) * len(lead) / len(w0.text)   # monospace
                    label_ws[0] = Word(x0, w0.y0, w0.x1, w0.y1, w0.text[len(lead):], w0.y)
            label = " ".join(w.text for w in label_ws).strip()
            if re.search(r"see table|detailed analysis", label, re.I) or (
                    not num_ws and re.fullmatch(r"[ivxlIVXL]+", label)):
                continue
            values = {}
            for w in num_ws:
                col = column_of(w)
                values[col] = values.get(col, "") + w.text
            if not label and values:
                if parsed:                         # number pushed off its line by an underline
                    for c, v in values.items():
                        parsed[-1]["values"].setdefault(c, v)
                continue
            if not values and not label.endswith(":"):
                pending, pending_x = pending + label + " ", pending_x or label_ws[0].x0
                continue
            x = pending_x or (label_ws[0].x0 if label_ws else zone)
            parsed.append({"label": (pending + label).strip(), "x": x, "marks": marks,
                           "y": ln.y, "values": values})
            pending, pending_x = "", None
            if label.upper().startswith("TOTAL"):
                break

        if not parsed:
            return 0
        base = min(p["x"] for p in parsed)
        group = ""
        page_rows = []
        for p in parsed:
            is_total = p["label"].upper().startswith("TOTAL")
            # footnote stars (OCR: "#") mark top-level rows whatever their indent
            starred = bool(re.search(r"[*#]", p["marks"]))
            level = 1 if p["x"] - base > 8 and not is_total and not starred else 0
            name = p["label"].rstrip(":").strip()
            if level == 0:
                group = "" if is_total else name
            row = {"table": table_id, "page": pno, "y": round(p["y"]), "level": level,
                   "group": group if level else "", "sinking_agent": name,
                   "footnote": len(re.findall(r"[*#]", p["marks"])) or None}
            issues = []
            for c in SUMMARY_COLS:
                raw = p["values"].get(c, "")
                row[f"raw_{c}"] = raw
                if not raw:
                    row[c] = None if p["values"] else ""
                elif is_dash(raw):
                    row[c] = 0
                else:
                    row[c] = parse_int(raw)
                    if row[c] is None:
                        issues.append(f"{c}_unreadable")
            row["issues"] = issues
            page_rows.append(row)

        self.check_table(table_id, page_rows)
        for row in page_rows:
            row["issues"] = ";".join(row["issues"])
        self.rows += page_rows
        return len(page_rows)

    def complete_no_headers(self, pno, tons, nos):
        """Each "No." header sits a fixed distance left of its "Tonnage"; rebuild any OCR lost."""
        pairs = {}
        for no in nos:
            right = [t for t in tons if 0 < t.x0 - no.x0 < 90]
            if right:
                pairs[id(min(right, key=lambda t: t.x0 - no.x0))] = no
        if len(pairs) == 3:
            return [pairs[id(t)] for t in tons]
        offset = statistics.median(t.x0 - pairs[id(t)].x0 for t in tons if id(t) in pairs) if pairs else 42
        width = statistics.median(n.x1 - n.x0 for n in nos)
        out = []
        for t in tons:
            if id(t) in pairs:
                out.append(pairs[id(t)])
            else:
                out.append(Word(t.x0 - offset, t.y0, t.x0 - offset + width, t.y1, "No.", t.y))
                self.notes.append(f"page {pno}: 'No.' header missing left of x={t.x0:.0f}; position inferred")
        return out

    def check_table(self, table_id, rows):
        """The tables are full of redundant sums (naval + merchant = total, children add up
        to their group, top-level rows add up to TOTAL). Use them to fill unreadable cells
        and to correct a misread cell when two independent sums agree on its value, then
        record which sums hold."""
        def active(r):                       # group headers without numbers take no part
            return any(r[c] != "" for c in SUMMARY_COLS)

        def kids_of(r):
            return [k for k in rows if k["level"] == 1 and k["group"] == r["sinking_agent"] and active(k)]

        cons = []                            # (check, item, column, target cell, part cells)
        for r in filter(active, rows):
            for kind in ("no", "tonnage"):
                cons.append(("naval+merchant=total", r["sinking_agent"], kind, (r, f"total_{kind}"),
                             [(r, f"naval_{kind}"), (r, f"merchant_{kind}")]))
            if r["level"] == 0 and kids_of(r):
                for c in SUMMARY_COLS:
                    cons.append(("children=parent", r["sinking_agent"], c, (r, c),
                                 [(k, c) for k in kids_of(r)]))
        total = next((r for r in rows if r["sinking_agent"].upper().startswith("TOTAL")), None)
        if total:
            tops = []
            for r in rows:
                if r["level"] == 0 and r is not total:
                    tops += [r] if active(r) else kids_of(r)
            for c in SUMMARY_COLS:
                cons.append(("rows=TOTAL", "TOTAL", c, (total, c), [(t, c) for t in tops]))

        def val(cell):
            return cell[0][cell[1]]

        def key(cell):
            return id(cell[0]), cell[1]

        def implied(cell, tgt, parts):       # value of `cell` that would make the sum hold
            rest = sum(val(p) for p in parts if key(p) != key(cell))
            return rest if key(cell) == key(tgt) else val(tgt) - rest

        changed = True
        while changed:
            changed = False
            for _, _, _, tgt, parts in cons:                 # one unknown in a sum: solve it
                unknown = [c for c in [tgt] + parts if val(c) is None]
                if len(unknown) == 1:
                    v = implied(unknown[0], tgt, parts)
                    if v >= 0:
                        unknown[0][0][unknown[0][1]] = v
                        unknown[0][0]["issues"].append(f"{unknown[0][1]}_inferred")
                        changed = True
            if changed:
                continue
            proposals = {}                                   # misread cell: >=2 failing sums agree
            for _, _, _, tgt, parts in cons:
                cells = [tgt] + parts
                if any(val(c) is None for c in cells) or val(tgt) == sum(val(p) for p in parts):
                    continue
                for c in cells:
                    proposals.setdefault(key(c), []).append((implied(c, tgt, parts), c))
            for props in proposals.values():
                values = {v for v, _ in props}
                if len(props) >= 2 and len(values) == 1 and min(values) >= 0:
                    cell = props[0][1]
                    cell[0][cell[1]] = props[0][0]
                    cell[0]["issues"].append(f"{cell[1]}_corrected")
                    changed = True
                    break

        for check, item, col, tgt, parts in cons:
            if all(val(c) is not None for c in [tgt] + parts):
                parsed = sum(val(p) for p in parts)
                self.checks.append({"table": table_id, "check": check, "item": item, "column": col,
                                    "expected": val(tgt), "parsed": parsed, "ok": val(tgt) == parsed})


# ------------------------------------------------------------------------- driver

def section_starts(doc):
    starts = {"summary": 8, "naval": 15, "merchant": 45}
    for _, title, page in doc.get_toc():
        t = title.lower()
        if "summary" in t:
            starts["summary"] = page
        elif "merchant" in t:
            starts["merchant"] = page
        elif "naval" in t:
            starts["naval"] = page
    return starts


def section_of(pno, starts):
    for name in ("merchant", "naval", "summary"):
        if pno >= starts[name]:
            return name
    return None


def parse_page_spec(spec, n_pages):
    pages = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        pages.update(range(int(lo), int(hi or lo) + 1))
    return sorted(p for p in pages if 1 <= p <= n_pages)


TAG_COLORS = {"date": (0.85, 0, 0), "name": (0, 0, 0.85), "vtype": (0, 0.55, 0),
              "tonnage": (0.9, 0.45, 0), "location": (0.55, 0, 0.55), "flag": (0, 0.55, 0.55),
              "agent": (0.6, 0.5, 0), "assessment": (0.35, 0.35, 0.35),
              "label": (0, 0, 0.85), "num": (0.9, 0.45, 0), "header": (0.7, 0.7, 0.7)}


def save_debug_images(pdf, pages_words, out_dir):
    """Draw each word's box coloured by its assigned column. Works on an in-memory copy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf) as doc:
        for pno, words in pages_words:
            page = doc[pno - 1]
            for w in words:
                page.draw_rect(pymupdf.Rect(w.x0, w.y0, w.x1, w.y1),
                               color=TAG_COLORS.get(w.tag, (1, 0, 0)), width=0.8)
            page.get_pixmap(dpi=110).save(out_dir / f"page_{pno:03d}.png")


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def build_report(args, page_stats, parsers, summary, total_s):
    out = [f"# JANAC extraction QA report", "",
           f"Source: `{args.source}`. Pages: `{args.pages or 'all'}`, total time {total_s:.2f} s, "
           f"average {1000 * total_s / max(len(page_stats), 1):.0f} ms/page.", "",
           "## Pages", "", "| page | section | words | skew | rows | ms |", "|---|---|---|---|---|---|"]
    for s in page_stats:
        section = s["section"] if s["has_table"] else f"{s['section']} (no table)"
        out.append(f"| {s['page']} | {section} | {s['words']} | {s['slope']:+.4f} "
                   f"| {s['rows']} | {s['ms']:.0f} |")

    for p in parsers.values():
        rows = p.rows
        if not rows:
            continue
        n = len(rows)
        issue_counts = Counter(i for r in rows for i in r["issues"].split(";") if i)
        located = sum(r["lat"] is not None or bool(r["location"] and not re.search(r"\d", r["location"]))
                      for r in rows)
        out += ["", f"## {p.section.title()} vessels ({n} rows)", "",
                "| field | parsed OK |", "|---|---|",
                f"| full date | {pct(sum(bool(r['date']) for r in rows), n)} |",
                f"| tonnage | {pct(sum(r['tonnage'] is not None for r in rows), n)} |",
                f"| location (coords or named area) | {pct(located, n)} |",
                f"| name without OCR junk | {pct(sum('name_ocr_suspect' not in r['issues'] for r in rows), n)} |",
                f"| assessment recognised | {pct(sum(r['assessment'] in KNOWN_ASSESSMENTS for r in rows), n)} |",
                f"| rows with no issues | {pct(sum(not r['issues'] for r in rows), n)} |",
                "", "Issue counts: " + (", ".join(f"`{k}` {v}" for k, v in issue_counts.most_common()) or "none")]
        if p.checks:
            ok_c = sum(c["count_ok"] for c in p.checks)
            ok_t = sum(c["tonnage_ok"] for c in p.checks)
            out += ["", f"### Monthly subtotal checks: counts {ok_c}/{len(p.checks)}, "
                        f"tonnage {ok_t}/{len(p.checks)}", "",
                    "| month | page | stated n | parsed n | stated t | parsed t | unreadable t | note |",
                    "|---|---|---|---|---|---|---|---|"]
            for c in p.checks:
                note = "partial (run started mid-month)" if c["partial"] else (
                    "" if c["count_ok"] and c["tonnage_ok"] else "**MISMATCH**")
                out.append(f"| {c['month']} | {c['page']} | {c['stated_count']} | {c['parsed_count']} "
                           f"| {c['stated_tonnage']} | {c['parsed_tonnage']} "
                           f"| {c['rows_unreadable_tonnage']} | {note} |")
        for col in ("vessel_type", "agent_type", "flag_of_agent", "assessment"):
            vc = Counter(r[col] for r in rows).most_common(25)
            out += ["", f"`{col}` values: " + ", ".join(f"{k or '∅'} ({v})" for k, v in vc)]
        flagged = [r for r in rows if r["issues"]]
        if flagged:
            out += ["", "### Rows with issues", "", "| page | y | date | name | raw tonnage | raw location | issues |",
                    "|---|---|---|---|---|---|---|"]
            for r in flagged[:80]:
                out.append(f"| {r['page']} | {r['y']} | {r['date'] or r['raw_date']} | {r['raw_name']} "
                           f"| {r['raw_tonnage']} | {r['raw_location']} | {r['issues']} |")
        if p.notes:
            out += ["", "Notes:", *[f"- {n}" for n in p.notes]]

    if summary.rows:
        checks = summary.checks
        out += ["", f"## Summary tables ({len(summary.rows)} rows)", "",
                f"Arithmetic checks passed: {sum(c['ok'] for c in checks)}/{len(checks)}", "",
                "| table | check | item | column | expected | parsed |", "|---|---|---|---|---|---|"]
        for c in checks:
            if not c["ok"]:
                out.append(f"| {c['table']} | {c['check']} | {c['item']} | {c['column']} "
                           f"| {c['expected']} | {c['parsed']} |")
        unread = [r for r in summary.rows if "unreadable" in r["issues"]]
        out += ["", f"Rows with unreadable values: {len(unread)}"]
        for r in unread:
            out.append(f"- Table {r['table']} `{r['sinking_agent']}`: {r['issues']}")
        if summary.notes:
            out += ["", "Notes:", *[f"- {n}" for n in summary.notes]]
    return "\n".join(out) + "\n"


def run(pdf, page_spec=None, source="ocr", cache=HERE / "data" / "ocr_cache"):
    """Parse the given pages. Returns (chrono parsers, summary parser, page stats, page words, seconds)."""
    t_start = time.perf_counter()
    doc = pymupdf.open(pdf)
    starts = section_starts(doc)
    pages = parse_page_spec(page_spec, len(doc)) if page_spec else list(range(starts["summary"], len(doc) + 1))
    pages = [p for p in pages if section_of(p, starts)]

    parsers = {s: ChronoParser(s, complete=starts[s] in pages) for s in ("naval", "merchant")}
    summary = SummaryParser()
    page_stats, page_word_lists = [], []
    for pno in pages:
        t0 = time.perf_counter()
        section = section_of(pno, starts)
        page = doc[pno - 1]
        words, slope = page_words(page, source, cache)
        if section == "summary":
            n = summary.parse_page(pno, words, ROMAN[pno - starts["summary"]])
        else:
            n = parsers[section].parse_page(pno, words, page.rect.height)
        page_stats.append({"page": pno, "section": section, "has_table": n > 0,
                           "words": len(words), "slope": slope, "rows": n,
                           "ms": 1000 * (time.perf_counter() - t0)})
        page_word_lists.append((pno, words))
    doc.close()
    return parsers, summary, page_stats, page_word_lists, time.perf_counter() - t_start


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=HERE / "JANAC.pdf")
    ap.add_argument("--pages", help='1-based PDF pages, e.g. "8-17" or "15,20-25" (default: all)')
    ap.add_argument("--out", type=Path, default=HERE / "data" / "raw")
    ap.add_argument("--source", choices=["ocr", "pdf"], default="ocr",
                    help="ocr: re-OCR page images with RapidOCR (slow first run, cached); "
                         "pdf: use the PDF's own, noisier text layer")
    ap.add_argument("--cache", type=Path, default=HERE / "data" / "ocr_cache",
                    help="where RapidOCR results are cached")
    ap.add_argument("--debug", action="store_true", help="save per-page images of column assignments")
    args = ap.parse_args()

    parsers, summary, page_stats, debug_pages, total_s = run(args.pdf, args.pages, args.source, args.cache)
    pages = [s["page"] for s in page_stats]

    args.out.mkdir(parents=True, exist_ok=True)
    for s, p in parsers.items():
        if p.rows:
            write_csv(args.out / f"{s}_losses.csv", LOSS_FIELDS, p.rows)
    checks = [c for p in parsers.values() for c in p.checks]
    if checks:
        write_csv(args.out / "subtotal_checks.csv", list(checks[0]), checks)
    if summary.rows:
        write_csv(args.out / "summary_tables.csv", SUMMARY_FIELDS, summary.rows)
    report = build_report(args, page_stats, parsers, summary, total_s)
    (args.out / "qa_report.md").write_text(report, encoding="utf-8")
    if args.debug:
        save_debug_images(args.pdf, debug_pages, args.out / "debug")

    print(f"{len(pages)} pages in {total_s:.2f} s -> {args.out}")
    for s, p in parsers.items():
        if p.rows:
            ok = sum(c["count_ok"] for c in p.checks)
            print(f"  {s}: {len(p.rows)} rows, {sum(not r['issues'] for r in p.rows)} clean, "
                  f"subtotal counts matched {ok}/{len(p.checks)}")
    if summary.rows:
        print(f"  summary: {len(summary.rows)} rows, arithmetic checks "
              f"{sum(c['ok'] for c in summary.checks)}/{len(summary.checks)}")
    print(f"  report: {args.out / 'qa_report.md'}")


if __name__ == "__main__":
    main()
