"""Shared logic for the per-page summary-table scripts (page_08_table_I.py ... page_12_table_V.py).

Each script:
1. OCRs its page with extract_janac (RapidOCR, cached) and parses the table;
2. compares every cell with the hand-verified transcription in tests/fixtures/table_<ROMAN>.csv
   and uses the transcribed value wherever OCR disagrees, recording what it changed;
3. re-checks every sum the table implies on the final values;
4. saves the dataframe to data/summary/table_<ROMAN>.csv.

The transcriptions were read from the page images at 140 dpi and must satisfy the
table's own sums; a sum that fails in the final table is either listed in
SOURCE_ERRORS (a typo in the 1947 original) or makes the script fail.
"""

import csv
import difflib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import extract_janac as ex  # noqa: E402

OUT_DIR = ROOT / "data" / "summary"
FIXTURES = ROOT / "tests" / "fixtures"
TITLES = {
    "I": "Total Japanese naval and merchant vessels sunk during World War II",
    "II": "Japanese naval and merchant vessels sunk by United States forces",
    "III": "Japanese naval and merchant vessels sunk by Allied forces excluding United States",
    "IV": "Japanese vessels sunk by combinations of US and Allied forces, and miscellaneous agents",
    "V": 'Vessels assessed sunk by "probable" sinking agents (already included in Tables I-IV)',
}
# Sums that do not hold in the printed report itself: (table, check, item, column) -> note
SOURCE_ERRORS = {
    ("IV", "rows=TOTAL", "TOTAL", "merchant_tonnage"):
        "printed TOTAL merchant tonnage 492,589 is 2 more than its rows (492,587); kept as printed",
    ("IV", "naval+merchant=total", "TOTAL", "tonnage"):
        "printed TOTAL merchant tonnage 492,589 is 2 more than its rows (492,587); kept as printed",
}
COLS = ex.SUMMARY_COLS


def read_fixture(table_id):
    with open(FIXTURES / f"table_{table_id}.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["level"] = int(r["level"])
        for c in COLS:
            r[c] = int(r[c])
    return rows


def align(truth, ocr):
    """Match each transcribed row to the OCR row with the most similar label, in page order."""
    pairs, j = [], 0
    for t in truth:
        window = list(enumerate(ocr))[j:j + 4]
        best = max(window, key=lambda io: ex.similar(io[1]["sinking_agent"], t["sinking_agent"]),
                   default=None)
        if best and ex.similar(best[1]["sinking_agent"], t["sinking_agent"]) >= 0.6:
            pairs.append((t, best[1]))
            j = best[0] + 1
        else:
            pairs.append((t, None))
    return pairs


def sum_checks(df, table_id):
    """Every sum the table implies, evaluated on the final values."""
    rows = df.to_dict("records")
    checks = []

    def add(check, item, col, expected, actual):
        note = SOURCE_ERRORS.get((table_id, check, item, col), "")
        checks.append({"check": check, "item": item, "column": col, "expected": expected,
                       "actual": actual, "ok": expected == actual, "source_error": note})

    for r in rows:
        for kind in ("no", "tonnage"):
            add("naval+merchant=total", r["sinking_agent"], kind, r[f"total_{kind}"],
                r[f"naval_{kind}"] + r[f"merchant_{kind}"])
    for i, r in enumerate(rows):
        if r["level"] != 0:
            continue
        kids = []
        for k in rows[i + 1:]:
            if k["level"] == 0:
                break
            kids.append(k)
        if kids:
            for c in COLS:
                add("children=parent", r["sinking_agent"], c, r[c], sum(k[c] for k in kids))
    total = next((r for r in rows if r["sinking_agent"] == "TOTAL"), None)
    if total:
        tops = [r for r in rows if r["level"] == 0 and r is not total]
        for c in COLS:
            add("rows=TOTAL", "TOTAL", c, total[c], sum(r[c] for r in tops))
    return pd.DataFrame(checks)


def build_table(page):
    with ex.pymupdf.open(ROOT / "JANAC.pdf") as doc:
        starts = ex.section_starts(doc)
    table_id = ex.ROMAN[page - starts["summary"]]
    _, summary, _, _, _ = ex.run(ROOT / "JANAC.pdf", str(page), "ocr", ROOT / "data" / "ocr_cache")
    truth = read_fixture(table_id)

    out, cells_ok, cells = [], 0, 0
    group = ""
    for i, (t, o) in enumerate(align(truth, summary.rows), 1):
        if t["level"] == 0:
            group = "" if t["sinking_agent"] == "TOTAL" else t["sinking_agent"]
        fixes = []
        for c in COLS:
            cells += 1
            got = o[c] if o else None
            if got == t[c]:
                cells_ok += 1
            else:
                fixes.append(f"{c}:{'' if got in (None, '') else got}->{t[c]}")
        out.append({
            "table": table_id, "page": page, "row": i, "level": t["level"],
            "group": group if t["level"] else "", "sinking_agent": t["sinking_agent"],
            **{c: t[c] for c in COLS},
            "ocr_label": o["sinking_agent"] if o else "",
            "ocr_matched": not fixes,
            "corrected_cells": "; ".join(fixes) if o else "row not found by OCR",
            "ocr_issues": o["issues"] if o else "",
        })
    df = pd.DataFrame(out)

    checks = sum_checks(df, table_id)
    bad = checks[~checks.ok & (checks.source_error == "")]
    notes = checks[~checks.ok & (checks.source_error != "")]
    df["note"] = ""
    for _, n in notes.iterrows():
        df.loc[df.sinking_agent == n["item"], "note"] = n["source_error"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"table_{table_id}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"Table {table_id} (page {page}): {TITLES[table_id]}")
    print(f"  rows: {len(df)} | OCR got {cells_ok}/{cells} cells right "
          f"({100 * cells_ok / cells:.1f}%), {cells - cells_ok} corrected from the transcription")
    print(f"  sum checks: {checks.ok.sum()}/{len(checks)} hold"
          + (f"; {len(notes)} fail in the printed report itself (noted)" if len(notes) else ""))
    for _, r in df[~df.ocr_matched].iterrows():
        print(f"    fixed {r['sinking_agent']}: {r['corrected_cells']}")
    print(f"  saved {path.relative_to(ROOT)}")
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit(f"Table {table_id}: {len(bad)} sum check(s) fail - check the transcription")
    return df
