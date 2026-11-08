"""
Time-series analysis of Japanese ship losses to the United States, month by
month, December 1941 - August 1945: long-term trend, change points and
seasonality, for warships and merchant ships, by weapon.

Why the United States only: the US is credited with about 90% of all
Japanese losses. Every other nation has too few sinkings per month for a
time series (most months would be 0), so a trend or season could not be
told apart from chance.

What is analysed. For each kind of vessel (naval, merchant) there is one
series for all US sinkings and one per weapon that has at least
MIN_SERIES_SHIPS losses (submarine, aircraft, surface craft, mines). Each
series is analysed twice: number of ships lost, and tonnage lost. Tonnage is
never added across the two kinds, because naval tonnage is standard
displacement and merchant tonnage is gross register tons (GRT).

Shared sinkings count as a fraction. A ship with n agents listed gives 1/n of
itself (and 1/n of its tonnage) to each agent. So a ship sunk by a US
submarine and US carrier aircraft adds 0.5 to "submarine" and 0.5 to
"aircraft"; a ship sunk by US aircraft and an Australian ship adds 0.5 to the
US. This way the weapon series add up to the "all" series, and no ship is
counted twice. Monthly counts can therefore be numbers like 12.5.

Months have different lengths, and the first and last months of the war
are partial (24 days in December 1941, 15 in August 1945), so everything is
analysed as losses per average-length month (see EXPOSURE).

The methods, one per question:
  1. What does the trend look like?  The trend part of an STL decomposition
     (13-month smoothing) draws a smooth line through each series. Used to
     describe, not to test. STL's own seasonal part is not used: with under
     four years it can't tell a season from a campaign.
  2. Is there a steady rise or fall?  Mann-Kendall trend test, in the
     trend-free pre-whitening version that allows for autocorrelation (this
     month's losses are like last month's), with Sen's slope (the typical
     change per month).
  3. When did the loss rate change?  Change-point detection (ruptures,
     exact dynamic programming). The number of change points (0-3) is
     chosen by BIC, so a change point is only kept when it improves the fit
     by more than it costs.
  4. How fast did losses grow or shrink in each phase?  A regression with a
     separate level and slope for each phase between change points:
     negative binomial for ship counts, linear on log(1 + tonnage) for
     tonnage. Gives "% change per month" with a 95% confidence interval.
  5. Is there a season?  The same regression plus one yearly wave (a sine
     and a cosine of the calendar month). A likelihood-ratio test says
     whether the wave improves the fit beyond the trend.
  6. Do the weapons differ?  One pooled regression per kind of vessel tests
     whether the weapons share the same trend shape, and the same season.

Read the season results with care: the war lasted 45 months, under four
years, so each calendar month is seen only three or four times. A month
can look "seasonal" because a big campaign happened to fall in it (Leyte
in October 1944). And these are RAW losses: they also depend on how many
Japanese ships were still afloat, which the report does not give. Fewer
merchant losses in 1945 partly means there were few merchant ships left.

Input (built by ship_lists/naval_pages.py and ship_lists/merchant_pages.py,
read through sinking_map.load_ships()):
    data/naval/naval_losses_15-44.csv
    data/merchant/merchant_losses_45-115.csv
    data/summary/table_II.csv   (US totals from the report, used as a check)

Output:
    data/time_series/us_monthly_losses.csv     month x vessel x series: ships, tonnage
    data/time_series/us_series_tests.csv       one row per series and measure:
                                               Mann-Kendall, change points, season test
    data/time_series/us_trend_segments.csv     one row per phase: % change per month
    data/time_series/us_weapon_comparison.csv  pooled tests: do weapons differ?
    figures/time_series/us_naval_ships.png       \
    figures/time_series/us_naval_tonnage.png      |  monthly losses, trend, change
    figures/time_series/us_merchant_ships.png     |  points and seasonal profile
    figures/time_series/us_merchant_tonnage.png  /   per series
    figures/time_series/us_weapon_shares.png     share of each weapon over time

Run (takes about half a minute):
    python us_loss_time_series.py
"""

import math
import time
import warnings

import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no window is needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import ruptures as rpt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.seasonal import STL

import sinking_map as smap
import table_plot_helpers as helpers
from sinking_map import VESSEL_COLOURS, VESSEL_OPTIONS
from table_plot_helpers import (AQUA, BACKGROUND, GREY, GRID, INK, INK_SOFT, PINK,
                                VIOLET, YELLOW, tint)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIG_DIR = smap.PROJECT_DIR / "figures" / "time_series"
DATA_DIR = smap.PROJECT_DIR / "data" / "time_series"
TABLE_II_FILE = smap.PROJECT_DIR / "data" / "summary" / "table_II.csv"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# The agent analysed (an option of sinking_map's agent filter).
AGENT = "us"

# Every month of the war, so months with no losses count as 0 instead of
# being missing. The report's first loss is in December 1941, its last in
# August 1945: 45 months.
MONTHS = pd.period_range("1941-12", "1945-08", freq="M")

# Exposure: how many days of war each month had. The first and last months
# are partial: the first sinking is on 8 December 1941 (the attacks of 7
# December Hawaii time were 8 December in Asia) and the last on 15 August
# 1945, the day Japan announced its surrender. So December 1941 had 24 days
# of war and August 1945 only 15. Every other month counts its real length,
# so February's 28 days don't look like a seasonal dip.
# All analysis is done on losses per average-length month (30.44 days):
# August 1945's 63 ships in 15 days become about 128 per month. Without this,
# the last month's half-length would look like a sudden collapse.
ACTIVE_DAYS = {pd.Period("1941-12", "M"): 24, pd.Period("1945-08", "M"): 15}
AVERAGE_MONTH_DAYS = 365.25 / 12
EXPOSURE = np.array([ACTIVE_DAYS.get(m, m.days_in_month) for m in MONTHS]) / AVERAGE_MONTH_DAYS

# STL trend window in months (must be odd and longer than the 12-month
# season). STL's default for a 12-month season is about 23 months, which on a
# 45-month war smooths away everything but a straight line (the mining
# campaign of 1945 became a ramp starting in 1942). 13 months follows the
# real turns of the war while still smoothing out single-month spikes.
STL_TREND_MONTHS = 13

# Weapons get their own series in this order. Anything else the US is
# credited with (shore batteries: 2 warships; sabotage: half a ship; cause
# unknown: 4 merchant ships) is pooled as "other": it is far too small for a
# series, but it stays in the "all" series and in the weapon-share chart.
WEAPON_ORDER = ["submarine", "aircraft", "surface", "mine"]
OTHER = "other"

# A weapon gets its own series only with at least this many (weighted) ships
# lost in the whole war, i.e. about 2 a month on average. Below that most
# months are 0 and the models can't separate trend from noise. With 100,
# this keeps: naval submarine, aircraft, surface; merchant submarine,
# aircraft, mine. It drops naval mines (20 ships) and merchant surface (16).
MIN_SERIES_SHIPS = 100

# STL and the season test use a 12-month cycle (a calendar year).
SEASON_PERIOD = 12

# Change points: each phase must last at least MIN_SEGMENT months, and there
# are at most MAX_BREAKS change points. Six months stops the method from
# cutting out a single big month (e.g. the Truk raid, Feb 1944) as its own
# "phase"; 3 breaks (4 phases) is plenty for a 45-month war and keeps each
# phase long enough to estimate a slope.
MIN_SEGMENT = 6
MAX_BREAKS = 3

ALPHA = 0.05            # significance level, used for words like "rising"
LJUNG_BOX_LAG = 6       # residual autocorrelation is checked up to 6 months back
SPLINE_DF = 4           # flexibility of the smooth trend in the pooled weapon model

# Check against the report: Table II's TOTAL row is ships sunk by the US
# ALONE (no other nation involved). Our data must give the same numbers for
# ships whose every agent is the US. Read from data/summary/table_II.csv.
# Naval matches exactly. Merchant is short by this known amount (ships, t):
#   - KENKOKU (p54, 954 GRT): the report prints no agent for it, so the ship
#     list has it as "Unknown", but Table II counts it as a US sinking (our
#     Unknown total is exactly one ship / 954 t above Table IV's).
#   - 100 GRT more: our Japan-credited ships total 100 t above Table I and
#     our US ships 100 t below, so one tonnage or agent is misread somewhere
#     in a way that leaves the monthly subtotals right. Not found yet.
# 1 ship in 2,117 and 0.01% of the tonnage: too small to change any result.
KNOWN_DIFFERENCE = {"naval": (0, 0), "merchant": (1, 1054)}

# Weapon colours for the share chart. Blue and orange are left out on
# purpose: in every other chart of this project they mean naval and
# merchant, and reusing them for weapons would mislead. Grey is kept for
# "other", as everywhere in the project.
WEAPON_COLOURS = {"submarine": VIOLET, "aircraft": AQUA, "surface": PINK,
                  "mine": YELLOW, OTHER: GREY}
WEAPON_LABELS = {"all": "All US sinkings", "submarine": "Submarines",
                 "aircraft": "Aircraft", "surface": "Surface warships",
                 "mine": "Mines", OTHER: "Other"}
MEASURE_LABELS = {"ships": "ships", "tonnage": "tonnage"}
TONNAGE_UNITS = {"naval": "tons", "merchant": "GRT"}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Building the monthly series
# ---------------------------------------------------------------------------
def us_credits(ships):
    """List every US credit as one row with its share of the ship.

    A ship with n credited agents gives 1/n of itself to each (see the
    module docstring). Rows for other nations are dropped after the share is
    worked out, so a ship shared by the US and Australia leaves 0.5 here.
    Weapons outside WEAPON_ORDER are renamed "other".
    """
    rows = []
    for _, ship in ships.iterrows():
        share = 1 / len(ship["credits"])
        for agent, weapon in ship["credits"]:
            if agent != AGENT:
                continue
            rows.append({"month": pd.Period(year=int(ship["year"]), month=int(ship["month"]),
                                            freq="M"),
                         "vessel": ship["vessel"],
                         "weapon": weapon if weapon in WEAPON_ORDER else OTHER,
                         "ships": share,
                         "tonnage": ship["tonnage"] * share})
    return pd.DataFrame(rows)


def check_against_report(ships):
    """Compare ships sunk by the US alone with Table II's TOTAL row.

    This checks that the agent columns were read and grouped correctly
    before anything is built on them. Shared US + other-nation sinkings are
    not in Table II (they are in Table III), so only ships whose every
    credit is the US are counted. The explained gap in KNOWN_DIFFERENCE is
    allowed for; anything else is printed as a warning rather than stopping,
    so a small OCR difference doesn't block the analysis.
    """
    report = pd.read_csv(TABLE_II_FILE)
    total = report[report["sinking_agent"] == "TOTAL"].iloc[0]
    us_only = ships[ships["credits"].apply(lambda c: all(a == AGENT for a, _ in c))]
    for vessel in VESSEL_OPTIONS:
        mine = us_only[us_only["vessel"] == vessel]
        ours = (len(mine), int(mine["tonnage"].sum()))
        theirs = (int(total[f"{vessel}_no"]), int(total[f"{vessel}_tonnage"]))
        gap = (theirs[0] - ours[0], theirs[1] - ours[1])
        status = ("matches" if gap == (0, 0)
                  else "matches, apart from the known gap (see KNOWN_DIFFERENCE),"
                  if gap == KNOWN_DIFFERENCE[vessel] else "WARNING: differs from")
        print(f"US-only {vessel}: {ours[0]:,} ships, {ours[1]:,} t {status} "
              f"Table II ({theirs[0]:,} ships, {theirs[1]:,} t)")


def monthly_losses(credits):
    """Sum the credits per month into a long table: month, vessel, series, ships, tonnage.

    "series" is "all" or a weapon. Every month of MONTHS appears for every
    series, with 0 where nothing was lost, because a missing month would
    otherwise be skipped by the models instead of counted as a quiet month.
    """
    parts = []
    for vessel in VESSEL_OPTIONS:
        of_kind = credits[credits["vessel"] == vessel]
        for series in ["all"] + WEAPON_ORDER + [OTHER]:
            rows = of_kind if series == "all" else of_kind[of_kind["weapon"] == series]
            table = (rows.groupby("month")[["ships", "tonnage"]].sum()
                     .reindex(MONTHS, fill_value=0.0))
            table.index.name = "month"
            # active_days is kept in the CSV so the per-month rates used in
            # the analysis can be recomputed from it (see EXPOSURE).
            table["active_days"] = EXPOSURE * AVERAGE_MONTH_DAYS
            parts.append(table.reset_index().assign(vessel=vessel, series=series))
    return pd.concat(parts, ignore_index=True)[["month", "vessel", "series", "ships",
                                               "tonnage", "active_days"]]


def series_to_analyse(monthly, vessel):
    """Return "all" plus the weapons with at least MIN_SERIES_SHIPS losses for one vessel kind."""
    totals = monthly[monthly["vessel"] == vessel].groupby("series")["ships"].sum()
    return ["all"] + [w for w in WEAPON_ORDER if totals.get(w, 0) >= MIN_SERIES_SHIPS]


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
def stabilise(values, measure):
    """Transform a series so its month-to-month noise is about the same size throughout.

    Counts: square root. For counts, the noise grows with the level (a month
    averaging 60 losses swings far more than one averaging 4); after a square
    root it is roughly constant. Tonnage: log(1 + tonnage), because tonnage
    swings are proportional to the level and one big ship (YAMATO, 63,000 t)
    would otherwise dominate. "1 +" keeps months with 0 tonnage usable.
    Change-point detection assumes even noise, which is why it needs this.
    """
    return np.sqrt(values) if measure == "ships" else np.log1p(values)


def change_points(values, measure):
    """Find the months where the level of a series changed, choosing how many by BIC.

    ruptures.Dynp tries every way of cutting the (stabilised) series into
    k + 1 phases of at least MIN_SEGMENT months and returns the cut with the
    smallest within-phase spread. That is done for k = 0 ... MAX_BREAKS, and
    BIC picks k: it rewards a closer fit but charges for every extra phase,
    so a break is only kept when the series really changes level there.
    Dynp (exact search) is used rather than a faster approximate method
    because 45 months is tiny and the exact answer costs nothing.

    Returns (breaks, bics): the index of the first month of each new phase,
    e.g. [19, 33], and the BIC for each number of breaks 0 ... MAX_BREAKS
    (saved in the results so the choice can be checked).
    """
    z = stabilise(np.asarray(values, dtype=float), measure).reshape(-1, 1)
    n = len(z)
    if np.allclose(z, z[0]):          # a flat series has nothing to cut
        return [], []
    algo = rpt.Dynp(model="l2", min_size=MIN_SEGMENT, jump=1).fit(z)
    best_bic, best_breaks, bics = math.inf, [], []
    for k in range(MAX_BREAKS + 1):
        ends = [n] if k == 0 else algo.predict(n_bkps=k)
        starts = [0] + ends[:-1]
        rss = sum(float(((z[a:b] - z[a:b].mean()) ** 2).sum()) for a, b in zip(starts, ends))
        # Gaussian BIC: n*log(RSS/n) + (number of parameters)*log(n). The
        # parameters are k+1 phase means and k break positions.
        bic = n * math.log(max(rss, 1e-12) / n) + (2 * k + 1) * math.log(n)
        bics.append(bic)
        if bic < best_bic:
            best_bic, best_breaks = bic, ends[:-1]
    return best_breaks, bics


def phases(breaks, n):
    """Turn break positions into (start, end) month-index pairs, end not included."""
    edges = [0] + list(breaks) + [n]
    return list(zip(edges[:-1], edges[1:]))


def design_matrix(n, breaks, seasonal):
    """Build the regression inputs: a level and a slope for each phase, plus an optional season.

    For phase s the columns are:
        level_s  1 inside the phase, 0 elsewhere
        slope_s  months since the phase began, inside the phase; 0 elsewhere
    so each phase has its own straight line (on the log scale) and the line
    may jump at a change point. This is "segmented regression".

    With seasonal=True, sin and cos of the calendar month are added. Any
    smooth yearly wave, whatever month it peaks in, is a mix of these two,
    so one wave costs only 2 parameters; 11 month-of-year dummies would use
    up far too much of a 45-month series.
    """
    t = np.arange(n)
    columns = {}
    for s, (a, b) in enumerate(phases(breaks, n)):
        inside = (t >= a) & (t < b)
        columns[f"level_{s}"] = inside.astype(float)
        columns[f"slope_{s}"] = np.where(inside, t - a, 0.0)
    if seasonal:
        calendar_month = MONTHS.month.to_numpy()[:n]
        columns["sin"] = np.sin(2 * np.pi * calendar_month / SEASON_PERIOD)
        columns["cos"] = np.cos(2 * np.pi * calendar_month / SEASON_PERIOD)
    return pd.DataFrame(columns)


def fit_model(y, X, measure, exposure):
    """Fit one regression: negative binomial for ship counts, OLS on log(1 + t) for tonnage.

    y is the raw monthly losses and exposure the length of each month (see
    EXPOSURE). The count model takes the exposure directly: it models
    losses per average month and expects proportionally fewer in a short
    month. For tonnage, y is first turned into a per-average-month rate.

    Why negative binomial: losses are counts, and they vary far more from
    month to month than a Poisson model allows (losses come in bunches:
    one convoy attack can sink six ships). The negative binomial has an
    extra "alpha" parameter for that extra spread. The counts here can be
    fractions (shared sinkings count 1/n); the negative binomial
    likelihood works with those too.

    Why OLS on log(1 + tonnage): tonnage is not a count, and it is very
    skewed (one battleship weighs as much as twenty frigates). On the log
    scale it is close to normal, and a slope b means about (e^b - 1) x 100
    percent change per month.

    The negative binomial is fitted from the Poisson model's answer as a
    starting point, and that Poisson model is fitted as a GLM (by IRLS,
    iteratively reweighted least squares). The default start (all zeros)
    with Newton steps overflowed to NaN on the merchant series, whose
    100+ losses a month are far from that guess; IRLS gets there safely.

    Returns the fitted result and whether the fit converged.
    """
    if measure == "tonnage":
        return sm.OLS(np.log1p(y / exposure), X).fit(), True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        poisson = sm.GLM(y, X, family=sm.families.Poisson(), exposure=exposure).fit()
        start = np.append(poisson.params.to_numpy(), 0.1)   # 0.1 = starting alpha
        result = sm.NegativeBinomial(y, X, exposure=exposure).fit(
            start_params=start, method="bfgs", maxiter=5000, disp=0)
    converged = (result.mle_retvals.get("converged", True)
                 and np.isfinite(result.params).all()
                 and not any(issubclass(w.category, ConvergenceWarning) for w in caught))
    return result, converged


def fitted_rate(result, X, measure):
    """Return the model's fitted losses per average-length month (ships or tonnage).

    exposure=1 asks the count model for a full average month, so the line
    is on the same scale as the bars in the charts. For tonnage the fit is
    on log(1 + rate), so it is turned back with exp(...) - 1. That gives a
    typical (geometric-mean-like) month rather than the average month, so
    the dashed line sits a little below the spiky bars. That is expected.
    """
    if measure == "tonnage":
        return np.expm1(result.predict(X))
    return result.predict(X, exposure=np.ones(len(X)))


def log_distance(rate, fit, measure):
    """How far each month's rate is above (+) or below (-) the fit, on the log scale.

    Counts use log((rate + 0.5) / (fit + 0.5)); the 0.5 keeps a month with
    no losses from becoming minus infinity. Tonnage is already modelled on
    log(1 + rate), so it uses the difference on that scale. Averaged per
    calendar month, this is the seasonal profile in the charts.
    """
    if measure == "tonnage":
        return np.log1p(rate) - np.log1p(fit)
    return np.log((rate + 0.5) / (fit + 0.5))


def residuals(result, y, X, measure, exposure):
    """Pearson residuals for the negative binomial, plain residuals for OLS.

    Pearson residuals divide each miss by the expected spread for that
    month (mean + alpha * mean^2), so busy and quiet months count equally
    in the autocorrelation check.
    """
    if measure == "tonnage":
        return result.resid
    mean = result.predict(X, exposure=exposure)
    alpha = result.params["alpha"]
    return (y - mean) / np.sqrt(mean + alpha * mean ** 2)


def season_shape(result):
    """Turn the fitted sin/cos into (peak month, peak-to-trough ratio).

    b_sin*sin(x) + b_cos*cos(x) is one wave with height A = sqrt(b_sin^2 +
    b_cos^2) that peaks where x = atan2(b_sin, b_cos). On the log scale the
    peak is A above average and the trough A below, so the peak month has
    exp(2A) times the losses of the trough month, with the trend held fixed.
    """
    b_sin, b_cos = result.params["sin"], result.params["cos"]
    angle = math.atan2(b_sin, b_cos) % (2 * math.pi)
    peak = round(angle * SEASON_PERIOD / (2 * math.pi)) % SEASON_PERIOD
    return MONTH_NAMES[(peak or SEASON_PERIOD) - 1], math.exp(2 * math.hypot(b_sin, b_cos))


def analyse_series(values, measure):
    """Run methods 1-5 on one monthly series and return everything needed for tables and charts.

    values are the raw monthly losses. STL, Mann-Kendall and change points
    work on the rate per average month (values / EXPOSURE); the regressions
    take the raw values with the exposure (see fit_model).

    Phases with no loss at all (e.g. mines before the first mine sinking)
    are left out of the regressions: a phase of zeros has a level of minus
    infinity on the log scale, which the fit cannot reach. Those months are
    still in the chart and in the Mann-Kendall test.
    """
    y = np.asarray(values, dtype=float)
    rate = y / EXPOSURE
    n = len(y)
    out = {"rate": rate}

    # 1. STL, used only for its trend line (a smooth description). Its
    # seasonal part is NOT used: with under 4 cycles it just repeats
    # whichever campaign fell in each calendar month (for mines it made a
    # "May-July season" out of the 1945 mining campaign). The season is
    # tested by the regression instead. robust=True stops single huge
    # months (a carrier battle) from bending the trend.
    stl = STL(pd.Series(rate, index=MONTHS.to_timestamp()), period=SEASON_PERIOD,
              trend=STL_TREND_MONTHS, robust=True).fit()
    out["stl_trend"] = np.clip(stl.trend.to_numpy(), 0, None)   # a loss rate can't be negative

    # 2. Mann-Kendall with trend-free pre-whitening (TFPW, Yue et al. 2002).
    # The plain test assumes each month is independent; losses run in
    # streaks, which makes the plain test find "significant" trends too
    # easily. TFPW removes the trend, strips out the month-to-month carry-over
    # (lag-1 autocorrelation), puts the trend back and then tests. The
    # Hamed-Rao correction was tried first, but on these short, steep series
    # its variance estimate can turn negative and give no p-value at all
    # (it did for merchant aircraft tonnage); where both work they agree.
    test = mk.trend_free_pre_whitening_modification_test(rate)
    out.update(mk_trend=test.trend, mk_p=test.p, sen_slope=test.slope)

    # 3. Change points.
    breaks, bics = change_points(rate, measure)
    out["breaks"] = breaks
    out["bic_by_breaks"] = "; ".join(f"k={k}: {b:.1f}" for k, b in enumerate(bics))

    # 4. Segmented trend model (without season) and 5. the same with season.
    keep = np.ones(n, dtype=bool)
    for a, b in phases(breaks, n):
        if y[a:b].sum() == 0:
            keep[a:b] = False
    X_trend = design_matrix(n, breaks, seasonal=False)
    X_season = design_matrix(n, breaks, seasonal=True)
    # Drop the columns of empty phases (they are all 0 on the kept months).
    used = [c for c in X_trend.columns if X_trend.loc[keep, c].abs().sum() > 0]
    X_trend, X_season = X_trend[used], X_season[used + ["sin", "cos"]]
    exposure = EXPOSURE[keep]
    trend_fit, ok_trend = fit_model(y[keep], X_trend[keep], measure, exposure)
    season_fit, ok_season = fit_model(y[keep], X_season[keep], measure, exposure)

    fit = np.full(n, np.nan)
    fit[keep] = fitted_rate(trend_fit, X_trend[keep], measure)
    out["model_fit"] = fit

    # Seasonal profile for the chart: the average distance of each calendar
    # month from the trend, in percent. The trend here is the trend part of
    # the SEASON model (its prediction with the wave switched off, sin = cos
    # = 0), not the trend-only model: adding the wave shifts the fitted
    # trend a little, and the bars must be measured from the same trend as
    # the wave drawn over them, or the two don't match.
    no_wave = X_season[keep].assign(sin=0.0, cos=0.0)
    season_trend = fitted_rate(season_fit, no_wave, measure)
    distance = pd.Series(log_distance(rate[keep], season_trend, measure),
                         index=MONTHS.month[keep])
    out["season_profile"] = 100 * np.expm1(
        distance.groupby(level=0).mean().reindex(range(1, 13)).to_numpy())
    # The fitted yearly wave, in percent above or below the trend, on a fine
    # grid of calendar months (1 = January ... 12 = December, the same
    # numbers design_matrix() uses) for a smooth curve.
    grid = np.linspace(0.5, 12.5, 121)
    wave = season_fit.params["sin"] * np.sin(2 * np.pi * grid / SEASON_PERIOD) \
        + season_fit.params["cos"] * np.cos(2 * np.pi * grid / SEASON_PERIOD)
    out["season_wave"] = (grid, 100 * np.expm1(wave))

    # Growth per phase: e^slope - 1 is the proportional change per month.
    segments = []
    for s, (a, b) in enumerate(phases(breaks, n)):
        row = {"phase": s + 1, "start": str(MONTHS[a]), "end": str(MONTHS[b - 1]),
               "months": b - a, "total": y[a:b].sum(),
               "mean_per_month": y[a:b].sum() / EXPOSURE[a:b].sum()}
        name = f"slope_{s}"
        if name in trend_fit.params.index:
            low, high = trend_fit.conf_int().loc[name]
            row.update(change_pct_per_month=100 * math.expm1(trend_fit.params[name]),
                       ci_low_pct=100 * math.expm1(low), ci_high_pct=100 * math.expm1(high),
                       p_value=trend_fit.pvalues[name])
        segments.append(row)
    out["segments"] = segments

    # Season test: likelihood ratio. Twice the gain in log-likelihood from
    # adding the wave follows a chi-square with 2 degrees of freedom (sin and
    # cos) if there is no real season.
    lr = max(0.0, 2 * (season_fit.llf - trend_fit.llf))
    out["season_lr"] = lr
    out["season_p"] = stats.chi2.sf(lr, df=2)
    out["season_peak"], out["season_ratio"] = season_shape(season_fit)
    out["converged"] = ok_trend and ok_season
    out["months_modelled"] = int(keep.sum())

    # Model diagnostics, saved so the model choices can be checked:
    #   ships:   the negative binomial's dispersion alpha, and a test of
    #            alpha = 0 (i.e. would a Poisson model do?). The LR statistic
    #            against the Poisson fit is compared with a 50:50 mix of 0
    #            and chi-square(1), because alpha = 0 is on the edge of the
    #            allowed range (alpha >= 0); that halves the usual p-value.
    #   tonnage: R-squared of the trend model, and the Jarque-Bera test of
    #            whether the residuals look normal (OLS p-values assume so).
    if measure == "ships":
        poisson = sm.GLM(y[keep], X_trend[keep], family=sm.families.Poisson(),
                         exposure=exposure).fit()
        lr_alpha = max(0.0, 2 * (trend_fit.llf - poisson.llf))
        out.update(nb_alpha=trend_fit.params["alpha"],
                   overdispersion_p=0.5 * stats.chi2.sf(lr_alpha, df=1),
                   r_squared=np.nan, jarque_bera_p=np.nan)
    else:
        out.update(nb_alpha=np.nan, overdispersion_p=np.nan,
                   r_squared=trend_fit.rsquared,
                   jarque_bera_p=stats.jarque_bera(season_fit.resid).pvalue)
    out["aic_trend"], out["aic_season"] = trend_fit.aic, season_fit.aic

    # If the residuals still follow each other month to month, the model
    # missed part of the pattern and its p-values are too optimistic. This
    # is reported, not corrected, so readers can judge.
    resid = residuals(season_fit, y[keep], X_season[keep], measure, exposure)
    out["ljung_box_p"] = float(acorr_ljungbox(resid, lags=[LJUNG_BOX_LAG])["lb_pvalue"].iloc[0])
    return out


def compare_weapons(monthly, vessel, weapons, measure):
    """Test whether the weapons of one vessel kind share a trend shape and a season.

    All weapon series are fitted in one model with a smooth trend (a
    natural cubic spline with SPLINE_DF degrees of freedom). Two
    likelihood-ratio tests:
      - trend: one shared trend shape (each weapon only higher or lower)
        against a separate shape per weapon.
      - season: one shared yearly wave against a separate wave per weapon
        (both with separate trend shapes).
    A spline is used here instead of change points because each weapon has
    its own change points, and the test needs one common time axis.
    Returns a list of result rows.
    """
    data = monthly[(monthly["vessel"] == vessel) & monthly["series"].isin(weapons)].copy()
    data["t"] = data["month"].apply(lambda m: MONTHS.get_loc(m))
    data["sin"] = np.sin(2 * np.pi * data["month"].dt.month / SEASON_PERIOD)
    data["cos"] = np.cos(2 * np.pi * data["month"].dt.month / SEASON_PERIOD)
    # Same treatment of month length as in fit_model: exposure for counts,
    # a per-average-month rate for tonnage.
    data["exposure"] = EXPOSURE[data["t"].to_numpy()]
    data["y"] = (data[measure] if measure == "ships"
                 else np.log1p(data[measure] / data["exposure"]))
    spline = f"cr(t, df={SPLINE_DF}, constraints='center')"
    formulas = {
        "shared_trend": f"y ~ C(series) + {spline}",
        "separate_trend": f"y ~ C(series) * {spline}",
        "shared_season": f"y ~ C(series) * {spline} + sin + cos",
        "separate_season": f"y ~ C(series) * {spline} + C(series):(sin + cos) + sin + cos",
    }
    fits = {}
    with warnings.catch_warnings():
        # Mines are 0 for most of 1942-43, which pushes their fitted level
        # very low; the optimiser warns about it but the comparison holds.
        warnings.simplefilter("ignore")
        for name, formula in formulas.items():
            if measure == "ships":
                fits[name] = smf.negativebinomial(formula, data, exposure=data["exposure"]
                                                  ).fit(disp=0, maxiter=1000)
            else:
                fits[name] = smf.ols(formula, data).fit()

    rows = []
    for test, (small, big) in {"trend shape differs by weapon": ("shared_trend", "separate_trend"),
                               "season differs by weapon": ("shared_season", "separate_season")
                               }.items():
        lr = max(0.0, 2 * (fits[big].llf - fits[small].llf))
        df = len(fits[big].params) - len(fits[small].params)
        rows.append({"vessel": vessel, "measure": measure, "weapons": ", ".join(weapons),
                     "test": test, "lr_statistic": lr, "df": df,
                     "p_value": stats.chi2.sf(lr, df)})
    return rows


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def save_figure(fig, out_path):
    """Save a figure, retrying for a moment if Windows refuses to open the file.

    Windows sometimes reports "Errno 22 Invalid argument" when a PNG is being
    replaced while an image viewer, the search indexer or antivirus has it
    open. The same save works a second later, so try up to 3 times (as
    sinking_map.draw_map does) instead of losing the whole run.
    """
    for attempt in range(3):
        try:
            fig.savefig(out_path, dpi=170, facecolor=BACKGROUND)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)
    plt.close(fig)


def trend_words(result):
    """Describe the Mann-Kendall result in a few words for a panel title."""
    if result["mk_p"] >= ALPHA:
        return f"no steady trend (Mann-Kendall p = {result['mk_p']:.2f})"
    direction = "rising" if result["sen_slope"] > 0 else "falling"
    return f"{direction} overall (Mann-Kendall p = {result['mk_p']:.3f})"


def value_formatter(measure):
    """Tick labels: plain numbers for ships, 50k / 1.2M for tonnage."""
    def fmt(value, _position):
        if measure == "ships":
            return f"{value:g}"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:g}M"
        return f"{value / 1_000:g}k" if abs(value) >= 1_000 else f"{value:g}"
    return plt.FuncFormatter(fmt)


def draw_series_figure(monthly, results, vessel, measure, out_path):
    """Draw one figure: a row per series, timeline on the left, seasonal profile on the right.

    Left panel, from back to front:
      - bars: the monthly losses (pale fleet colour), the raw data;
      - solid dark line: the STL trend, a smooth description of the level;
      - dashed coloured line: the segmented regression (method 4);
      - dotted verticals: the change points, labelled with their month.
    Right panel: for each calendar month, how far above or below the trend
    model that month was on average (bars, in %), with the fitted yearly
    wave of the season test drawn over it (line) and the test's p-value in
    the title. If the bars zig-zag with no shape and the wave is flat,
    there is no season.

    All values are losses per average-length month (see EXPOSURE), so the
    short first and last months are comparable with the rest. Each row has
    its own y scale, because the series differ hugely in size (100+
    merchant ships a month against a few warships).
    """
    names = [s for s in results if s[0] == vessel and s[2] == measure]
    colour = VESSEL_COLOURS[vessel]
    rows = len(names)
    fig, axes = plt.subplots(rows, 2, figsize=(16, 2.9 * rows + 1.9),
                             gridspec_kw={"width_ratios": [3.4, 1], "wspace": 0.14,
                                          "hspace": 0.62})
    axes = np.atleast_2d(axes)
    x = MONTHS.to_timestamp() + pd.Timedelta(days=14)   # bar in the middle of its month

    for row, key in enumerate(names):
        series = key[1]
        result = results[key]
        total = monthly[(monthly["vessel"] == vessel)
                        & (monthly["series"] == series)][measure].sum()
        rate = result["rate"]
        ax, season_ax = axes[row]

        ax.bar(x, rate, width=24, color=tint(colour, 0.55), zorder=2)
        ax.plot(x, result["stl_trend"], color=INK, linewidth=2, zorder=4)
        ax.plot(x, result["model_fit"], color=colour, linewidth=2, linestyle="--", zorder=5)
        top = max(rate.max(), np.nanmax(result["stl_trend"]),
                  np.nanmax(result["model_fit"])) * 1.12 or 1
        for b in result["breaks"]:
            when = MONTHS[b].to_timestamp()
            ax.axvline(when, color=INK_SOFT, linestyle=":", linewidth=1.4, zorder=3)
            ax.text(when, top * 0.97, f" {MONTHS[b].strftime('%b %Y')}", fontsize=10,
                    color=INK_SOFT, va="top", ha="left", zorder=6)
        ax.set_ylim(0, top)
        ax.set_xlim(MONTHS[0].to_timestamp(), (MONTHS[-1] + 1).to_timestamp())
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.yaxis.set_major_formatter(value_formatter(measure))
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_title(f"{WEAPON_LABELS[series]}: {total:,.0f} "
                     f"{'ships' if measure == 'ships' else TONNAGE_UNITS[vessel]}"
                     f" — {trend_words(result)}", loc="left", fontsize=12.5, color=INK)

        grid, wave = result["season_wave"]
        season_ax.bar(range(1, 13), result["season_profile"], color=tint(colour, 0.55),
                      width=0.7, zorder=2)
        season_ax.plot(grid, wave, color=colour, linewidth=2.2, zorder=4)
        season_ax.axhline(0, color=INK_SOFT, linewidth=0.8, zorder=3)
        season_ax.set_xticks(range(1, 13), [m[0] for m in MONTH_NAMES], fontsize=9.5)
        season_ax.set_xlim(0.4, 12.6)
        # Fit both the bars and the whole wave (matplotlib's automatic limits
        # cut off the wave in one panel).
        values = np.concatenate([result["season_profile"], wave, [0]])
        span = np.nanmax(values) - np.nanmin(values)
        season_ax.set_ylim(np.nanmin(values) - 0.08 * span, np.nanmax(values) + 0.08 * span)
        season_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
        season_ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        verdict = (f"peak {result['season_peak']}, p = {result['season_p']:.3f}"
                   if result["season_p"] < ALPHA else f"no clear season, p = {result['season_p']:.2f}")
        season_ax.set_title(verdict, loc="left", fontsize=11, color=INK_SOFT)

    unit = "ships" if measure == "ships" else f"tonnage ({TONNAGE_UNITS[vessel]})"
    fig.suptitle(f"Japanese {VESSEL_OPTIONS[vessel].split(' (')[0].lower()} sunk by the "
                 f"United States per month — {unit}", x=0.06, ha="left", y=0.995,
                 fontsize=17, fontweight="bold", color=INK)
    height = fig.get_figheight()
    handles = [Patch(color=tint(colour, 0.55), label="Losses per month"),
               Line2D([], [], color=INK, linewidth=2, label="STL trend (13-month smoothing)"),
               Line2D([], [], color=colour, linewidth=2, linestyle="--",
                      label="Segmented regression (trend per phase)"),
               Line2D([], [], color=INK_SOFT, linestyle=":", linewidth=1.4,
                      label="Change point"),
               Line2D([], [], color=colour, linewidth=2.2,
                      label="Right: fitted yearly wave; bars = average % above/below trend")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 1 - 0.62 / height),
               ncol=5, frameon=False, fontsize=10.5)
    fig.text(0.06, 0.12 / height,
             "Source: JANAC (1947). Shared sinkings count 1/n per agent. Losses per average-length "
             "month: Dec 1941 (24 days of war) and Aug 1945 (15 days) are scaled up to a full month.\n"
             "Raw losses: they also depend on how many Japanese ships were still afloat, which the "
             "report does not give. The war lasted under 4 years, so each calendar month is seen "
             "only 3-4 times.", fontsize=9.5, color=INK_SOFT, va="bottom")
    fig.subplots_adjust(left=0.06, right=0.98, top=1 - 1.45 / height, bottom=0.95 / height)
    save_figure(fig, out_path)


def draw_share_figure(monthly, out_path):
    """Draw the share of US sinkings by weapon over time, one panel per vessel kind.

    Shares are taken over a centred 3-month window (sum of 3 months'
    losses per weapon / sum of all losses), because single early months
    have only a handful of losses and a month with 2 ships would jump
    between 0% and 100%. Stacked areas are used because the shares add up
    to 100%, so the shift from one weapon to another reads as one band
    growing while another shrinks.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.5), sharex=True,
                             gridspec_kw={"hspace": 0.28})
    x = MONTHS.to_timestamp() + pd.Timedelta(days=14)
    order = WEAPON_ORDER + [OTHER]
    for ax, vessel in zip(axes, VESSEL_OPTIONS):
        table = (monthly[(monthly["vessel"] == vessel) & monthly["series"].isin(order)]
                 .pivot(index="month", columns="series", values="ships")
                 .reindex(columns=order, fill_value=0))
        smooth = table.rolling(3, center=True, min_periods=1).sum()
        shares = smooth.div(smooth.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        ax.stackplot(x, [shares[w] for w in order], colors=[WEAPON_COLOURS[w] for w in order],
                     labels=[WEAPON_LABELS[w] for w in order], edgecolor=BACKGROUND,
                     linewidth=0.6)
        ax.set_ylim(0, 100)
        ax.set_xlim(x[0], x[-1])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_title(f"{VESSEL_OPTIONS[vessel]} — {table.to_numpy().sum():,.0f} ships",
                     loc="left", fontsize=13, color=INK)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=5, frameon=False,
                   fontsize=11, reverse=True)
    fig.suptitle("Which US weapons sank Japanese ships, share of ships per month",
                 x=0.06, ha="left", y=0.985, fontsize=17, fontweight="bold", color=INK)
    fig.text(0.06, 0.015, "Source: JANAC (1947). Shares over a centred 3-month window; "
             "shared sinkings count 1/n per agent. Other = shore batteries, sabotage, "
             "cause unknown.", fontsize=9.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.08)
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def main():
    """Build the monthly series, run every method, and save the tables and charts."""
    helpers.set_readable_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ships = smap.load_ships()
    check_against_report(ships)
    monthly = monthly_losses(us_credits(ships))
    monthly.to_csv(DATA_DIR / "us_monthly_losses.csv", index=False)

    results, test_rows, segment_rows, comparison_rows = {}, [], [], []
    for vessel in VESSEL_OPTIONS:
        names = series_to_analyse(monthly, vessel)
        for measure in MEASURE_LABELS:
            for series in names:
                values = monthly[(monthly["vessel"] == vessel)
                                 & (monthly["series"] == series)][measure].to_numpy()
                result = analyse_series(values, measure)
                results[(vessel, series, measure)] = result
                test_rows.append({
                    "vessel": vessel, "series": series, "measure": measure,
                    "total": values.sum(),
                    "mk_trend": result["mk_trend"], "mk_p": result["mk_p"],
                    "sen_slope_per_month": result["sen_slope"],
                    "change_points": "; ".join(str(MONTHS[b]) for b in result["breaks"]),
                    "season_lr": result["season_lr"], "season_p": result["season_p"],
                    "season_peak_month": result["season_peak"],
                    "season_peak_to_trough": result["season_ratio"],
                    "ljung_box_p": result["ljung_box_p"],
                    "months_modelled": result["months_modelled"],
                    "nb_alpha": result["nb_alpha"],
                    "overdispersion_p": result["overdispersion_p"],
                    "r_squared": result["r_squared"],
                    "jarque_bera_p": result["jarque_bera_p"],
                    "aic_trend": result["aic_trend"], "aic_season": result["aic_season"],
                    "bic_by_breaks": result["bic_by_breaks"],
                    "model_converged": result["converged"]})
                segment_rows += [{"vessel": vessel, "series": series, "measure": measure, **seg}
                                 for seg in result["segments"]]
            comparison_rows += compare_weapons(monthly, vessel, names[1:], measure)
            draw_series_figure(monthly, results, vessel, measure,
                               FIG_DIR / f"us_{vessel}_{measure}.png")

    tests = pd.DataFrame(test_rows)
    # Benjamini-Hochberg q-values: 16 series are tested at once, so about one
    # "significant" result in 20 would appear by chance alone. The q-value is
    # the p-value adjusted for that; use q < 0.05 for firm conclusions.
    tests["mk_q"] = multipletests(tests["mk_p"], method="fdr_bh")[1]
    tests["season_q"] = multipletests(tests["season_p"], method="fdr_bh")[1]
    tests.to_csv(DATA_DIR / "us_series_tests.csv", index=False, float_format="%.4g")
    pd.DataFrame(segment_rows).to_csv(DATA_DIR / "us_trend_segments.csv", index=False,
                                      float_format="%.4g")
    pd.DataFrame(comparison_rows).to_csv(DATA_DIR / "us_weapon_comparison.csv", index=False,
                                         float_format="%.4g")
    draw_share_figure(monthly, FIG_DIR / "us_weapon_shares.png")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tests[["vessel", "series", "measure", "total", "mk_trend", "mk_q",
                     "change_points", "season_p", "season_q", "season_peak_month",
                     "ljung_box_p", "model_converged"]].round(3).to_string(index=False))
        print(pd.DataFrame(comparison_rows).round(4).to_string(index=False))
    print(f"Saved tables to {DATA_DIR} and charts to {FIG_DIR}")


if __name__ == "__main__":
    main()
