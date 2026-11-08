"""
How far from Tokyo were Japanese ships sunk, month by month? (Analysis A1)

The story this script tests: in 1942 Japan lost ships at the far edges of its
new empire (the Solomons, New Guinea, the Indies); by 1945 it was losing them
in its own harbours and straits. If that is true, the typical distance from
Tokyo of a sinking should fall over the war, and the share of sinkings in
Japan's home waters should rise.

For every ship with a printed position, the great-circle distance to Tokyo is
worked out. Then:
  1. Monthly summary: median distance and the 25th-75th percentile range,
     for all ships and for warships and merchant ships separately.
  2. Ship-level trend: quantile regression of distance on time (the 25th,
     50th and 75th percentiles), with Spearman's rank correlation.
  3. Monthly trend test: Mann-Kendall (trend-free pre-whitening) on the
     monthly median, with Sen's slope.
  4. Change points in the monthly median (exact segmentation, BIC), with a
     bootstrap confidence interval for each phase's median distance.
  5. Home waters: the share of sinkings within HOME_RADIUS_KM of Tokyo,
     modelled with logistic regression on time (standard errors clustered
     by month).

It uses the same filters as sinking_map.py (agent, weapon, vessel, ship type,
tonnage); leave them out to include all ships. All agents are included by
default: the question is where Japan lost ships, whoever sank them.

Input:
    data/naval/naval_losses_15-44.csv
    data/merchant/merchant_losses_45-115.csv
    (read through sinking_map.load_ships())

Output (named after the filters, like the other scripts):
    figures/distance/distance_from_tokyo_<filters>.png
    data/distance/monthly_distance_<filters>.csv   one row per month and series
    data/distance/distance_tests_<filters>.csv     every test result, one row each
    data/distance/distance_phases_<filters>.csv    phases between change points

The method is explained in analysis_A1_B_C_explained.md.

Run:
    python distance_from_tokyo.py                         # all ships
    python distance_from_tokyo.py --weapon submarine      # where submarines struck
    python distance_from_tokyo.py --vessel merchant --agent us
"""

import argparse
import math
import sys
import warnings

import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no window is needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import ruptures as rpt
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import sinking_map as smap
import table_plot_helpers as helpers
from sinking_animation import sinking_dates
from sinking_map import OptionError, VESSEL_COLOURS
from table_plot_helpers import BACKGROUND, GRID, INK, INK_SOFT, tint

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
FIG_DIR = smap.PROJECT_DIR / "figures" / "distance"
DATA_DIR = smap.PROJECT_DIR / "data" / "distance"

# Tokyo (the Imperial Palace), the centre of the Japanese state. Distance
# from Tokyo is a simple, readable stand-in for "distance from home". An
# alternative would be the distance to the nearest point of the home
# islands' coast, which would make Shimonoseki (about 850 km from Tokyo)
# count as 0; Tokyo was chosen because readers can picture it.
TOKYO_LAT, TOKYO_LON = 35.68, 139.77
EARTH_RADIUS_KM = 6371.0

# "Home waters": within this distance of Tokyo. 1,000 km covers the seas
# around Honshu, Shikoku, Kyushu and southern Hokkaido, the Korea Strait and
# the Inland Sea, but not Formosa (about 2,100 km) or the Bonins (just over
# 1,000 km).
HOME_RADIUS_KM = 1000

# Every month of the war (first sinking 8 Dec 1941, last 15 Aug 1945).
MONTHS = pd.period_range("1941-12", "1945-08", freq="M")
WAR_START = pd.Timestamp("1941-12-07")

# A monthly median from very few ships is mostly noise. Months with fewer
# ships than this (in a given series) are shown faintly but not used in that
# series' trend test or change points.
MIN_SHIPS_PER_MONTH = 5

# Change points: the same settings as us_loss_time_series.py. Each phase
# lasts at least 6 months; at most 3 breaks; BIC chooses how many.
MIN_SEGMENT = 6
MAX_BREAKS = 3

QUANTILES = [0.25, 0.5, 0.75]
BOOTSTRAP_SAMPLES = 2000
RANDOM_SEED = 12345
AVERAGE_MONTH_DAYS = 365.25 / 12
SERIES = {"all": "All ships", "naval": "Warships", "merchant": "Merchant ships"}
ALL_COLOUR = INK   # the combined series: ink, as it is the sum of the two kinds


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def distance_to_tokyo(lat, lon):
    """Great-circle distance in km from each (lat, lon) to Tokyo (haversine formula).

    Longitudes may be on the 0-360 scale used by sinking_map.load_ships();
    the formula only uses differences of longitude, so that makes no
    difference.
    """
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = math.radians(TOKYO_LAT), math.radians(TOKYO_LON)
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * math.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def ships_with_distance(selected):
    """Keep ships with a position and add date, month, time and distance columns.

    38 ships have only a text location ("off Japan") and are dropped here;
    the footnote reports how many. "t" is time in average-length months since
    7 December 1941, so regression slopes read as "km per month". Ships
    without a printed day are put on the 15th (see sinking_dates()).
    """
    ships = selected.dropna(subset=["lat", "lon"]).copy()
    ships["sunk"] = sinking_dates(ships)
    ships["month_period"] = ships["sunk"].dt.to_period("M")
    ships["t"] = (ships["sunk"] - WAR_START).dt.days / AVERAGE_MONTH_DAYS
    ships["distance_km"] = distance_to_tokyo(ships["lat"].to_numpy(), ships["lon"].to_numpy())
    ships["home"] = (ships["distance_km"] <= HOME_RADIUS_KM).astype(int)
    return ships


def monthly_summary(ships):
    """Median, quartiles, count and home-waters share per month, for each series.

    Returns a long table: month, series, n, median_km, q25_km, q75_km,
    home_share. Months with no ship in a series get n = 0 and empty
    statistics.
    """
    rows = []
    for series in SERIES:
        part = ships if series == "all" else ships[ships["vessel"] == series]
        by_month = dict(tuple(part.groupby("month_period")))
        for month in MONTHS:
            group = by_month.get(month, part.iloc[0:0])
            d = group["distance_km"]
            rows.append({"month": str(month), "series": series, "n": len(d),
                         "median_km": d.median() if len(d) else np.nan,
                         "q25_km": d.quantile(0.25) if len(d) else np.nan,
                         "q75_km": d.quantile(0.75) if len(d) else np.nan,
                         "home_share": group["home"].mean() if len(d) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def quantile_trends(ships):
    """Quantile regression of distance on time at the 25th, 50th and 75th percentiles.

    Why quantile regression rather than ordinary least squares: distances
    are skewed (a few sinkings near Ceylon or Alaska are 5,000+ km away) and
    the question is about the TYPICAL sinking, i.e. the median. Quantile
    regression estimates how a chosen percentile changes with time without
    assuming normal errors, and is not pulled by the far-away few. Fitting
    the 25th and 75th percentiles as well shows whether the whole
    distribution moved, or only its middle.

    Standard errors use statsmodels' default kernel estimate of the error
    density (Hall-Sheather bandwidth), which is robust to the spread
    changing over time. They do treat ships as independent, which ships sunk
    in the same raid are not, so the intervals are somewhat too narrow.
    """
    rows = []
    for q in QUANTILES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # iteration warnings from the solver
            fit = smf.quantreg("distance_km ~ t", ships).fit(q=q, max_iter=5000)
        low, high = fit.conf_int().loc["t"]
        rows.append({"test": f"Quantile regression, {int(q * 100)}th percentile of distance on time",
                     "estimate": fit.params["t"], "ci_low": low, "ci_high": high,
                     "p_value": fit.pvalues["t"], "unit": "km per month", "n": len(ships)})
    return rows


def spearman_trend(ships):
    """Spearman's rank correlation between sinking date and distance (ship level).

    A simple, assumption-light check: does a later sinking tend to be closer
    to Tokyo? Spearman uses ranks, so it measures any monotonic relationship
    and is not affected by the skew of distances.
    """
    rho, p = stats.spearmanr(ships["t"], ships["distance_km"])
    return {"test": "Spearman rank correlation, date vs distance", "estimate": rho,
            "ci_low": np.nan, "ci_high": np.nan, "p_value": p, "unit": "rho", "n": len(ships)}


def monthly_medians(summary, series):
    """Return one series' monthly median distances, NaN where n < MIN_SHIPS_PER_MONTH."""
    part = summary[summary["series"] == series].reset_index(drop=True)
    return part["median_km"].where(part["n"] >= MIN_SHIPS_PER_MONTH).to_numpy()


def mann_kendall(medians, series):
    """Mann-Kendall (trend-free pre-whitening) on the monthly medians, with Sen's slope.

    The same test as in us_loss_time_series.py (see us_time_series_explained.md).
    Months below MIN_SHIPS_PER_MONTH are dropped first; the pre-whitening
    step then treats the remaining months as evenly spaced, which is a small
    approximation when a few months are missing.
    """
    values = medians[~np.isnan(medians)]
    test = mk.trend_free_pre_whitening_modification_test(values)
    return {"test": f"Mann-Kendall (TFPW) on monthly median distance, {SERIES[series].lower()}",
            "estimate": test.slope, "ci_low": np.nan, "ci_high": np.nan,
            "p_value": test.p, "unit": "km per month (Sen's slope)", "n": len(values)}


def change_points(values):
    """Exact change-point search on the monthly medians, BIC choosing 0-MAX_BREAKS breaks.

    The same method as us_loss_time_series.change_points (ruptures.Dynp with
    an L2 cost, phases of at least MIN_SEGMENT months, BIC = n log(RSS/n) +
    (2k + 1) log n), applied to the raw medians: the noise of a median
    distance does not grow with its level, so no transformation is needed.
    Months with too few ships are filled from their neighbours (linear
    interpolation) so the months stay evenly spaced.

    Returns (breaks, bics): the first month of each new phase, and the BIC
    for each number of breaks.
    """
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    z = (filled / 1000).reshape(-1, 1)        # in 1,000 km; the unit doesn't change the result
    n = len(z)
    algo = rpt.Dynp(model="l2", min_size=MIN_SEGMENT, jump=1).fit(z)
    best, best_breaks, bics = math.inf, [], []
    for k in range(MAX_BREAKS + 1):
        ends = [n] if k == 0 else algo.predict(n_bkps=k)
        starts = [0] + ends[:-1]
        rss = sum(float(((z[a:b] - z[a:b].mean()) ** 2).sum()) for a, b in zip(starts, ends))
        bic = n * math.log(max(rss, 1e-12) / n) + (2 * k + 1) * math.log(n)
        bics.append(bic)
        if bic < best:
            best, best_breaks = bic, ends[:-1]
    return best_breaks, bics


def phase_table(ships, breaks):
    """Median distance per phase, with a cluster-bootstrap 95% confidence interval.

    Ships sunk in the same month (often in the same raid) are not
    independent, so the bootstrap resamples whole MONTHS within the phase,
    with replacement, and takes all their ships (a cluster bootstrap); the
    interval is the 2.5th-97.5th percentile of the resampled medians.
    Resampling single ships would give an interval that is too narrow.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    edges = [0] + list(breaks) + [len(MONTHS)]
    rows = []
    for phase, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        inside = ships[ships["month_period"].between(MONTHS[a], MONTHS[b - 1])]
        groups = [g["distance_km"].to_numpy() for _, g in inside.groupby("month_period")]
        boot = [np.median(np.concatenate([groups[i] for i in
                                          rng.integers(0, len(groups), len(groups))]))
                for _ in range(BOOTSTRAP_SAMPLES)]
        d = inside["distance_km"]
        rows.append({"phase": phase, "start": str(MONTHS[a]), "end": str(MONTHS[b - 1]),
                     "ships": len(d), "median_km": d.median(),
                     "ci_low_km": np.percentile(boot, 2.5),
                     "ci_high_km": np.percentile(boot, 97.5),
                     "home_share": inside["home"].mean()})
    return pd.DataFrame(rows)


def home_waters_trend(ships):
    """Logistic regression: does the chance that a sinking is in home waters rise over time?

    Model: logit P(home) = b0 + b1 * t, with t in months. exp(12 * b1) is the
    odds ratio per year. Standard errors are clustered by month: ships sunk
    in the same month (often in the same raid) are not independent, and
    ignoring that would make the p-value far too small.
    """
    groups = ships["month_period"].astype(str).factorize()[0]
    fit = smf.logit("home ~ t", ships).fit(disp=0, cov_type="cluster",
                                           cov_kwds={"groups": groups})
    low, high = fit.conf_int().loc["t"]
    return {"test": f"Logistic regression, P(within {HOME_RADIUS_KM:,} km of Tokyo) on time",
            "estimate": math.exp(12 * fit.params["t"]),
            "ci_low": math.exp(12 * low), "ci_high": math.exp(12 * high),
            "p_value": fit.pvalues["t"], "unit": "odds ratio per year", "n": len(ships)}


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def draw_figure(ships, summary, breaks, phases, chosen, n_without_position, out_path):
    """Draw the distance-over-time chart and the home-waters share under it.

    Top panel, from back to front:
      - one faint dot per ship (date, distance), coloured by kind of vessel,
        so the reader sees the raw data the summary lines come from;
      - a grey band: the middle half (25th-75th percentile) of each month;
      - the monthly median for all ships (thick, ink) and for warships and
        merchant ships (thin, blue / orange);
      - dotted lines at the change points, and a dashed horizontal line at
        each phase's median.
    The y axis is linear in km, with a dashed line at the home-waters radius.
    Bottom panel: the share of sinkings within that radius, per month.
    """
    fig, (ax, home_ax) = plt.subplots(2, 1, figsize=(16, 10.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.12})
    x = MONTHS.to_timestamp() + pd.Timedelta(days=14)
    top = 7000

    for kind in ["merchant", "naval"]:
        part = ships[ships["vessel"] == kind]
        ax.scatter(part["sunk"], part["distance_km"].clip(upper=top - 50), s=10,
                   color=VESSEL_COLOURS[kind], alpha=0.35, linewidths=0, zorder=1)
    all_rows = summary[summary["series"] == "all"]
    ax.fill_between(x, all_rows["q25_km"], all_rows["q75_km"], color=GRID, alpha=0.9, zorder=2)
    for series, colour, width in [("naval", VESSEL_COLOURS["naval"], 1.6),
                                  ("merchant", VESSEL_COLOURS["merchant"], 1.6),
                                  ("all", ALL_COLOUR, 3.0)]:
        ax.plot(x, monthly_medians(summary, series), color=colour, linewidth=width,
                zorder=4 if series == "all" else 3)

    for b in breaks:
        when = MONTHS[b].to_timestamp()
        ax.axvline(when, color=INK_SOFT, linestyle=":", linewidth=1.4, zorder=5)
        ax.text(when, top * 0.985, f" {MONTHS[b].strftime('%b %Y')}", fontsize=10.5,
                color=INK_SOFT, va="top", zorder=6)
    for _, phase in phases.iterrows():
        start = pd.Period(phase["start"], "M").to_timestamp()
        end = (pd.Period(phase["end"], "M") + 1).to_timestamp()
        ax.hlines(phase["median_km"], start, end, color=ALL_COLOUR, linewidth=1.2,
                  linestyle="--", zorder=4)
        ax.text(start + (end - start) / 2, phase["median_km"] + 150,
                f"phase median {phase['median_km']:,.0f} km", ha="center", fontsize=10.5,
                color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=BACKGROUND, edgecolor="none",
                          alpha=0.85))
    ax.axhline(HOME_RADIUS_KM, color=INK_SOFT, linestyle="--", linewidth=1, zorder=3)
    ax.text(x[0], HOME_RADIUS_KM - 80, f"  home waters: within {HOME_RADIUS_KM:,} km of Tokyo",
            fontsize=10, color=INK_SOFT, va="top", zorder=6)
    ax.set_ylim(0, top)
    ax.set_ylabel("Distance from Tokyo (km)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    handles = [Line2D([], [], color=ALL_COLOUR, linewidth=3, label="Monthly median, all ships"),
               Line2D([], [], color=VESSEL_COLOURS["naval"], linewidth=1.6,
                      label="Monthly median, warships"),
               Line2D([], [], color=VESSEL_COLOURS["merchant"], linewidth=1.6,
                      label="Monthly median, merchant ships"),
               Patch(color=GRID, label="Middle half of each month's sinkings"),
               Line2D([], [], color=ALL_COLOUR, linewidth=1.2, linestyle="--",
                      label="Phase median (between change points)"),
               Line2D([], [], marker="o", linestyle="", color=INK_SOFT, alpha=0.4, markersize=5,
                      label="One ship (blue warship, orange merchant)")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1, 0.93), ncol=2,
              frameon=True, framealpha=0.92, edgecolor=GRID, fontsize=10.5)

    home = all_rows["home_share"].where(all_rows["n"] >= MIN_SHIPS_PER_MONTH) * 100
    home_ax.bar(x, home, width=24, color=tint(ALL_COLOUR, 0.65), zorder=2)
    home_ax.set_ylim(0, 100)
    home_ax.set_ylabel(f"% within {HOME_RADIUS_KM:,} km")
    home_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    home_ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    home_ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    home_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    home_ax.set_xlim(MONTHS[0].to_timestamp(), (MONTHS[-1] + 1).to_timestamp())

    fig.suptitle("The noose tightens: how far from Tokyo Japanese ships were sunk",
                 x=0.07, ha="left", y=0.985, fontsize=18, fontweight="bold", color=INK)
    fig.text(0.07, 0.938, smap.selection_text(chosen), fontsize=12.5, color=INK_SOFT)
    fig.text(0.07, 0.012,
             f"Source: JANAC (1947). {len(ships):,} ships with a printed position "
             f"({n_without_position} with only a text location are left out). Great-circle "
             f"distance to Tokyo; ships beyond {top:,} km (Indian Ocean, Atlantic) are drawn at "
             f"the top edge. Medians shown for months with at least {MIN_SHIPS_PER_MONTH} ships.",
             fontsize=9.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.915, bottom=0.07)
    fig.savefig(out_path, dpi=170, facecolor=BACKGROUND)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def analyse(agent=None, weapon=None, vessel=None, ship_type=None,
            min_tonnage=None, max_tonnage=None):
    """Run every step for the selected ships and save the chart and tables.

    Filters work as in sinking_map.make_map(). Raises OptionError for a
    wrong value, and also when fewer than 30 ships with a position match
    (too few for the trend statistics).
    """
    chosen = smap.check_filters(agent=agent, weapon=weapon, vessel=vessel, ship_type=ship_type,
                                min_tonnage=min_tonnage, max_tonnage=max_tonnage)
    helpers.set_readable_style()
    selected = smap.select_ships(smap.load_ships(), chosen)
    ships = ships_with_distance(selected)
    if len(ships) < 30:
        raise OptionError(f"Only {len(ships)} ships with a position match "
                          f"{smap.selection_text(chosen)}; at least 30 are needed.")

    summary = monthly_summary(ships)
    tests = quantile_trends(ships) + [spearman_trend(ships)]
    for series in SERIES:
        medians = monthly_medians(summary, series)
        if np.sum(~np.isnan(medians)) >= 2 * MIN_SEGMENT:
            tests.append(mann_kendall(medians, series))
    tests.append(home_waters_trend(ships))

    medians = monthly_medians(summary, "all")
    breaks, bics = change_points(medians)
    phases = phase_table(ships, breaks)
    tests.append({"test": "Change points in monthly median distance (all ships)",
                  "estimate": len(breaks), "unit": "breaks chosen by BIC",
                  "n": int(np.sum(~np.isnan(medians))),
                  "detail": "BIC " + ", ".join(f"k={k}: {b:.1f}" for k, b in enumerate(bics))
                            + " | breaks at " + ", ".join(str(MONTHS[b]) for b in breaks)})

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filters = smap.file_name(chosen).replace("sinkings_", "").replace(".png", "")
    summary.to_csv(DATA_DIR / f"monthly_distance_{filters}.csv", index=False, float_format="%.4g")
    pd.DataFrame(tests).to_csv(DATA_DIR / f"distance_tests_{filters}.csv", index=False,
                               float_format="%.4g")
    phases.to_csv(DATA_DIR / f"distance_phases_{filters}.csv", index=False, float_format="%.4g")
    out_path = FIG_DIR / f"distance_from_tokyo_{filters}.png"
    draw_figure(ships, summary, breaks, phases, chosen, len(selected) - len(ships), out_path)

    with pd.option_context("display.width", 250, "display.max_columns", 12,
                           "display.max_colwidth", 80):
        print(pd.DataFrame(tests).drop(columns="detail", errors="ignore").to_string(index=False))
        print(tests[-1]["detail"])
        print(phases.round(2).to_string(index=False))
    print(f"Saved {out_path}")
    return tests, phases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distance from Tokyo of Japanese ship losses over time (trend, change "
                    "points, home-waters share). Leave a filter out to include all options.")
    smap.add_filter_arguments(parser)
    parser.add_argument("--options", action="store_true", help="list filter values and stop")
    args = parser.parse_args()
    if args.options:
        print(smap.describe_options())
        sys.exit(0)
    try:
        analyse(**smap.filter_arguments(args))
    except OptionError as error:
        print(error, file=sys.stderr)
        sys.exit(2)
