"""Shared logic for naval_pages.py and merchant_pages.py.

Pages are OCR'd with RapidOCR (cached in data/ocr_cache/) and parsed by extract_janac.py;
every month is checked against the "N Vessels / tonnage" subtotal the report prints at
its end, and cells OCR could not read are filled from a per-list table of hand-verified
fixes (flagged "<column>_manual" in `issues`).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import extract_janac as ex  # noqa: E402

COLUMNS = ["page", "block", "date", "year", "month", "day", "vessel_name", "vessel_type", "tonnage",
           "tonnage_estimated", "location", "lat", "lon", "flag_of_agent", "agent_type",
           "agent_probable", "assessment", "issues",
           "raw_date", "raw_name", "raw_tonnage", "raw_location", "raw_flag", "raw_agent",
           "raw_assessment"]
# Values that were repaired, guessed or hand-transcribed rather than read; usable, worth a look.
REPAIRED = {"tonnage_from_subtotal", "location_hemisphere_guessed", "month_inferred",
            "continued_on_next_page", "day_out_of_order"}
# The report itself leaves these out; not an extraction problem.
SOURCE_GAPS = {"day_not_given"}


def apply_manual_fixes(df, fixes):
    """fixes: {(page, vessel_name or raw OCR name): {column: value}} transcribed from the page image.
    A "_where" entry ({column: value}) picks one ship when a page has two of the same name."""
    for (page, name), fix in fixes.items():
        fix = dict(fix)
        where = fix.pop("_where", {})
        hit = (df.page == page) & ((df.vessel_name == name) | (df.raw_name.str.strip() == name))
        for col, value in where.items():
            hit &= df[col].astype(str).str.strip() == str(value)
        if hit.sum() != 1:
            raise SystemExit(f"manual fix for {name} (page {page}) matched {hit.sum()} rows")
        for col, value in fix.items():
            df.loc[hit, col] = value
            if col == "location":
                lat, lon, _ = ex.parse_location(value)
                df.loc[hit, ["lat", "lon"]] = [lat, lon]
            if col == "day":
                y, m = df.loc[hit, ["year", "month"]].iloc[0]
                df.loc[hit, "date"] = f"{int(y)}-{int(m):02d}-{int(value):02d}"
        fixed = tuple(c.split("_")[0] for c in fix)
        kept = [i for i in df.loc[hit, "issues"].iloc[0].split(";") if i and not i.startswith(fixed)]
        df.loc[hit, "issues"] = ";".join(kept + [f"{c}_manual" for c in fix])
    return df


def recheck_subtotals(df, checks, source_errors):
    """Re-run the monthly checks after manual fixes; months in source_errors are typos in
    the printed subtotal itself (kept as printed, noted, not counted as failures)."""
    if not len(checks):
        return checks
    sums = df.groupby("block").agg(n=("vessel_name", "size"), t=("tonnage", "sum"))
    checks = checks.copy()
    checks["parsed_count"] = [int(sums.n.get(i, 0)) for i in checks.index]
    checks["parsed_tonnage"] = [int(sums.t.get(i, 0)) for i in checks.index]
    checks["count_ok"] = checks.parsed_count == checks.stated_count
    checks["tonnage_ok"] = checks.parsed_tonnage == checks.stated_tonnage
    checks["source_error"] = checks.month.map(source_errors).fillna("")
    for i, c in checks.iterrows():
        rows = df.block == i
        drop = {"month_count_mismatch"} if c.count_ok else set()
        drop |= {"month_tonnage_mismatch"} if c.tonnage_ok else set()
        if c.source_error:
            drop |= {"month_count_mismatch", "month_tonnage_mismatch"}
        df.loc[rows, "issues"] = df.loc[rows, "issues"].map(
            lambda s: ";".join(x for x in s.split(";") if x and x not in drop))
    return checks


def run_list(section, default_pages, manual_fixes, doc, source_errors=None):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", default=default_pages, help=f'PDF pages, e.g. "45-54" (default {default_pages})')
    args = ap.parse_args()

    parsers, _, _, _, seconds = ex.run(ROOT / "JANAC.pdf", args.pages, "ocr", ROOT / "data" / "ocr_cache")
    parser = parsers[section]
    df = pd.DataFrame(parser.rows, columns=COLUMNS)
    df["issues"] = df.issues.fillna("")
    df = apply_manual_fixes(df, {k: v for k, v in manual_fixes.items() if k[0] in set(df.page)})
    checks = recheck_subtotals(df, pd.DataFrame(parser.checks), source_errors or {})

    out_dir = ROOT / "data" / section
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.pages.replace(",", "_")
    df.to_csv(out_dir / f"{section}_losses_{tag}.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(out_dir / f"{section}_subtotals_{tag}.csv", index=False, encoding="utf-8-sig")

    issues = df.issues.str.split(";").apply(lambda xs: {x for x in xs if x})
    repaired = lambda s: s & REPAIRED or any(i.endswith("_manual") for i in s)
    df["status"] = ["problem" if {i for i in s if not i.endswith("_manual")} - REPAIRED - SOURCE_GAPS
                    else "repaired" if repaired(s) else "clean" for s in issues]

    print(f"{section.title()} vessels, pages {args.pages}: {len(df)} ships, "
          f"{int(df.tonnage.fillna(0).sum()):,} t, in {seconds:.1f} s\n")
    per_page = df.groupby("page").status.value_counts().unstack(fill_value=0)
    for col in ("clean", "repaired", "problem"):
        if col not in per_page:
            per_page[col] = 0
    per_page = per_page[["clean", "repaired", "problem"]]
    per_page["ships"] = per_page.sum(axis=1)
    print(per_page.to_string(), "\n")

    if len(checks):
        full = checks[~checks.partial]
        print(f"Monthly subtotals: ship counts match {full.count_ok.sum()}/{len(full)}, "
              f"tonnage matches {full.tonnage_ok.sum()}/{len(full)}")
        for _, c in full[~(full.count_ok & full.tonnage_ok)].iterrows():
            note = f"  <- error in the report: {c['source_error']}" if c["source_error"] else ""
            print(f"  {c['month']} (page {c['page']}): {c['parsed_count']} ships / {c['parsed_tonnage']:,} t parsed, "
                  f"report says {c['stated_count']} / {c['stated_tonnage']:,} t{note}")
        print(f"Sum of printed monthly subtotals: {int(full.stated_count.sum()):,} ships, "
              f"{int(full.stated_tonnage.sum()):,} t")
    filled = lambda col: df[col].notna() & (df[col].astype(str) != "")
    print(f"Fields read: date {filled('date').mean():.1%}, tonnage {filled('tonnage').mean():.1%}, "
          f"position {(filled('lat') | ~df.location.str.contains(r'\d', na=False)).mean():.1%}, "
          f"agent {filled('agent_type').mean():.1%}, probable-agent marks {int(df.agent_probable.sum())}")
    counts = pd.Series([i for s in issues for i in s], dtype=str).value_counts()
    if len(counts):
        print("Issue counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    problems = df[df.status == "problem"]
    if len(problems):
        print(f"\nRows needing review ({len(problems)}):")
        print(problems[["page", "date", "vessel_name", "raw_date", "raw_tonnage", "raw_location", "issues"]]
              .to_string(index=False))
    print(f"\nSaved {(out_dir / f'{section}_losses_{tag}.csv').relative_to(ROOT)}")
    return df, checks
