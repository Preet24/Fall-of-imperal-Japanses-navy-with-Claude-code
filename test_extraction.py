"""Page-by-page check of extract_janac.py before running it on the whole PDF.

Each page gets a verdict:

* PASS        table found, the report's own totals reconcile, every row readable
* WARN        as PASS, but some values were repaired or guessed (flagged in `issues`)
* FAIL        table missing, totals don't reconcile, or rows with unreadable fields
* NO TABLE    title/blank page with nothing to extract (not counted)

Pages with a hand-checked fixture in tests/fixtures/ (page_NNN.csv for ship
lists, table_<ROMAN>.csv for summary tables) are also compared field by field.

Usage:
    python test_extraction.py --pages 8-18
    python test_extraction.py --pages 15-44 --source pdf
"""

import argparse
import csv
import sys
from pathlib import Path

import extract_janac as ex

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "tests" / "fixtures"

# Row issues that leave a required field empty or unknown.
BLOCKING = {"date_unreadable", "day_missing", "month_unknown", "date_invalid",
            "tonnage_unreadable", "tonnage_missing", "location_unreadable", "location_missing",
            "assessment_unknown", "month_count_mismatch", "month_tonnage_mismatch"}
# Row issues where a value exists but was repaired or guessed.
WARNING = {"tonnage_from_subtotal", "location_hemisphere_guessed", "month_inferred",
           "day_out_of_order", "name_ocr_suspect", "continued_on_next_page"}
# Anything else (e.g. day_not_given: the report itself prints "-" for the day) is informational.
SHIP_FIELDS = ["date", "vessel_name", "vessel_type", "tonnage", "location",
               "flag_of_agent", "agent_type", "assessment"]


def read_fixture(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def compare_ships(rows, truth):
    """Field-by-field comparison of a page's parsed rows with its fixture (in page order)."""
    errors, compared = [], 0
    if len(rows) != len(truth):
        errors.append(f"row count {len(rows)} != {len(truth)} in fixture")
    for i, (r, t) in enumerate(zip(rows, truth), 1):
        for f in SHIP_FIELDS:
            compared += 1
            if f == "location":
                want = ex.parse_location(t[f])[:2]
                got = (r["lat"], r["lon"]) if r["lat"] is not None else (None, None)
                if want == (None, None):      # named area: compare the text
                    ok = r["location"].lower() == t[f].lower()
                else:
                    ok = got == want
                shown = r["location"]
            elif f == "tonnage":
                ok, shown = str(r[f]) == t[f], r[f]
            elif f == "vessel_name":
                # spacing inside names is unreliable in the typescript ("OK INOSHIMA");
                # names get normalised against reference lists later
                ok, shown = r[f].replace(" ", "") == t[f].replace(" ", ""), r[f]
            else:
                ok, shown = str(r[f]).strip() == t[f], r[f]
            if not ok:
                errors.append(f"row {i} {f}: got {shown!r}, expected {t[f]!r}")
    return errors, compared


def compare_summary(rows, truth):
    errors, compared = [], 0
    if len(rows) != len(truth):
        errors.append(f"row count {len(rows)} != {len(truth)} in fixture")
    for r, t in zip(rows, truth):
        if ex.similar(r["sinking_agent"], t["sinking_agent"]) < 0.8:
            errors.append(f"label: got {r['sinking_agent']!r}, expected {t['sinking_agent']!r}")
        for c in ex.SUMMARY_COLS:
            compared += 1
            if str(r[c]) != t[c]:
                errors.append(f"{t['sinking_agent']} {c}: got {r[c]!r}, expected {t[c]}")
    return errors, compared


def judge_ship_page(stat, parser):
    rows = [r for r in parser.rows if r["page"] == stat["page"]]
    checks = [c for c in parser.checks if c["page"] == stat["page"] and not c["partial"]]
    fails, warns = [], []
    for c in checks:
        if not c["count_ok"]:
            fails.append(f"{c['month']}: {c['parsed_count']} rows parsed, subtotal says {c['stated_count']}")
        elif not c["tonnage_ok"]:
            fails.append(f"{c['month']}: tonnage {c['parsed_tonnage']} vs subtotal {c['stated_tonnage']}")
    for r in rows:
        issues = set(filter(None, r["issues"].split(";")))
        label = f"y={r['y']} {r['vessel_name'] or r['raw_name']!r}"
        if issues & BLOCKING - {"month_count_mismatch", "month_tonnage_mismatch"}:
            fails.append(f"{label}: {', '.join(sorted(issues & BLOCKING))}")
        elif issues & WARNING:
            warns.append(f"{label}: {', '.join(sorted(issues & WARNING))}")
    fixture = FIXTURES / f"page_{stat['page']:03d}.csv"
    accuracy = None
    if fixture.exists():
        errs, n = compare_ships(rows, read_fixture(fixture))
        accuracy = (n - len(errs), n)
        fails += [f"fixture: {e}" for e in errs]
    return rows, checks, fails, warns, accuracy


def judge_summary_page(stat, summary):
    rows = [r for r in summary.rows if r["page"] == stat["page"]]
    table = rows[0]["table"] if rows else "?"
    checks = [c for c in summary.checks if c["table"] == table]
    fails = [f"{c['check']} {c['item']} {c['column']}: {c['parsed']} vs {c['expected']}"
             for c in checks if not c["ok"]]
    warns = []
    for r in rows:
        issues = [i for i in r["issues"].split(";") if i]
        unreadable = [i for i in issues if i.endswith("_unreadable")
                      and i.replace("_unreadable", "_inferred") not in issues]
        inferred = [i for i in issues if i.endswith(("_inferred", "_corrected"))]
        if unreadable:
            fails.append(f"{r['sinking_agent']}: {', '.join(unreadable)}")
        if inferred:
            warns.append(f"{r['sinking_agent']}: {', '.join(inferred)}")
    fixture = FIXTURES / f"table_{table}.csv"
    accuracy = None
    if fixture.exists():
        errs, n = compare_summary(rows, read_fixture(fixture))
        accuracy = (n - len(errs), n)
        fails += [f"fixture: {e}" for e in errs]
    return rows, checks, fails, warns, accuracy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", default="8-18")
    ap.add_argument("--source", choices=["ocr", "pdf"], default="ocr")
    ap.add_argument("--pdf", type=Path, default=HERE / "JANAC.pdf")
    ap.add_argument("--cache", type=Path, default=HERE / "data" / "ocr_cache")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every warning, not just failures")
    args = ap.parse_args()

    parsers, summary, page_stats, _, seconds = ex.run(args.pdf, args.pages, args.source, args.cache)

    results = []
    for stat in page_stats:
        if not stat["has_table"]:
            verdict = "NO TABLE" if stat["words"] < 30 else "FAIL"
            fails = [] if verdict == "NO TABLE" else ["table header not found"]
            results.append((stat, verdict, 0, "", fails, [], None))
            continue
        if stat["section"] == "summary":
            rows, checks, fails, warns, acc = judge_summary_page(stat, summary)
            check_txt = f"{sum(c['ok'] for c in checks)}/{len(checks)} sums"
        else:
            rows, checks, fails, warns, acc = judge_ship_page(stat, parsers[stat["section"]])
            check_txt = f"{sum(c['count_ok'] and c['tonnage_ok'] for c in checks)}/{len(checks)} subtotals"
        verdict = "FAIL" if fails else "WARN" if warns else "PASS"
        results.append((stat, verdict, len(rows), check_txt, fails, warns, acc))

    print(f"\nSource: {args.source}, pages {args.pages}, {seconds:.1f} s\n")
    print(f"{'page':>4}  {'section':<8} {'verdict':<8} {'rows':>4}  {'checks':<14} {'fixture':<12} detail")
    for stat, verdict, n, check_txt, fails, warns, acc in results:
        acc_txt = f"{acc[0]}/{acc[1]} ok" if acc else ""
        detail = f"{len(fails)} problem(s)" if fails else f"{len(warns)} warning(s)" if warns else ""
        print(f"{stat['page']:>4}  {stat['section']:<8} {verdict:<8} {n:>4}  {check_txt:<14} {acc_txt:<12} {detail}")

    for stat, verdict, *_, fails, warns, _ in results:
        shown = fails + (warns if args.verbose else [])
        if shown:
            print(f"\npage {stat['page']} ({verdict}):")
            for line in shown:
                print(f"  - {line}")

    judged = [r for r in results if r[1] != "NO TABLE"]
    passed = sum(r[1] in ("PASS", "WARN") for r in judged)
    warned = sum(r[1] == "WARN" for r in judged)
    print(f"\nSuccessful pages: {passed}/{len(judged)} "
          f"({warned} with warnings; {len(results) - len(judged)} page(s) without a table skipped)")
    return 0 if passed == len(judged) else 1


if __name__ == "__main__":
    sys.exit(main())
