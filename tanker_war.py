"""
The tanker war: when, where and by what Japan's oil tankers were sunk. (Analysis T)

Japan went to war for the oil of the Netherlands East Indies, and that oil
could only reach Japan by tanker. The story to test: tanker losses were
small until late 1943, then exploded in 1944 as Allied submarines (and from
late 1944 carrier and land-based aircraft) concentrated on them, until by
mid-1945 the southern oil route was closed and the surviving tankers were
small coastal ships near Japan.

262 merchant ships of the report's type "Tanker" (1.31 million GRT) are
analysed:
  T1. Monthly tanker losses (ships and GRT) by weapon, with change points in
      the monthly GRT (same method as us_loss_time_series.py).
  T2. Did tankers become a LARGER SHARE of merchant losses? Each month's
      tanker count out of all merchant ships sunk is modelled as
      beta-binomial with a share that is constant within phases; the number
      and dates of the phase breaks are chosen by BIC, and "no change at
      all" is tested with a sup-likelihood-ratio test calibrated by a
      parametric bootstrap.
  T3. Were tankers more likely than other merchant ships to fall to
      submarines, and did that change? Odds ratios (tanker vs other merchant
      ship, submarine kill vs other cause) in four periods, the
      Mantel-Haenszel common odds ratio, the Cochran-Mantel-Haenszel test
      and the Breslow-Day test of equal odds ratios.
  T4. Where tankers died compared with other merchant ships: route group
      (from shipping_routes.py) x tanker / other, chi-square with a
      permutation p-value and adjusted residuals.
  T5. Did the tankers sunk get smaller? Median regression of log GRT on
      time (as % per year) and Spearman's rank correlation.
  T6. Submarines vs aircraft: logistic regression of "sunk by aircraft"
      (vs submarine) on time for tankers, with the crossover month (when
      aircraft became the more likely killer) and a month-cluster
      bootstrap interval for it.

Conventions shared with the other analysis scripts: shared sinkings are
split equally between weapons (1/n) in the charts; "sunk by a submarine" /
"by aircraft" in the tests means every credit is that weapon; standard
errors and bootstraps are clustered by month, because ships sunk in the same
month (often the same convoy) are not independent. Merchant tonnage is gross
register tons (GRT) throughout.

Input:
    data/merchant/merchant_losses_45-115.csv   (via shipping_routes.merchant_ships())

Output:
    figures/tankers/tanker_monthly.png         T1 monthly ships, GRT and cumulative GRT
    figures/tankers/tanker_share.png           T2 tanker share of merchant losses
    figures/tankers/tanker_weapons.png         T3 odds ratios and T6 crossover
    figures/tankers/tanker_where_size.png      T4 route groups and T5 size
    data/tankers/tanker_monthly.csv            monthly tanker and merchant totals
    data/tankers/tanker_phases.csv             T1 GRT phases and T2 share phases
    data/tankers/tanker_odds_by_period.csv     T3 per-period odds ratios
    data/tankers/tanker_route_groups.csv       T4 counts, expected counts, residuals
    data/tankers/tanker_tests.csv              every test, one row each

The method is explained in routes_tankers_explained.md.

Run:
    python tanker_war.py
"""

import itertools
import math
import warnings

import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no window is needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import optimize, special, stats
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2

import shipping_routes as routes
import table_plot_helpers as helpers
from distance_from_tokyo import change_points
from shipping_routes import (GROUP_COLOURS, GROUP_LABELS, GROUP_ORDER, MONTHS, WAR_START,
                             WEAPON_COLOURS, WEAPON_LABELS, WEAPON_ORDER, save_figure)
from table_plot_helpers import BACKGROUND, GRID, INK, INK_SOFT, ORANGE, tint
from weapon_ship_analysis import adjusted_residuals, chi_square, cramers_v

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
FIG_DIR = routes.smap.PROJECT_DIR / "figures" / "tankers"
DATA_DIR = routes.smap.PROJECT_DIR / "data" / "tankers"

RANDOM_SEED = 12345
BOOTSTRAP_SAMPLES = 2000        # month-cluster bootstraps of cheap statistics
LR_BOOTSTRAP_SAMPLES = 299      # parametric bootstrap of the sup-LR test (T2, slower)
MIN_SEGMENT = 6                 # shortest phase, in months (as in the other scripts)
MAX_SHARE_BREAKS = 2            # T2 tries 0, 1 or 2 breaks in the tanker share
ALPHA = 0.05

# T3 periods. Calendar years would leave 1942 with only 3 tankers sunk (all
# by submarines), so 1942 and 1943 are pooled; 1944, the year of the big
# tanker losses, is split into halves.
PERIODS = [("1942-43", "1941-12", "1943-12"), ("1944 H1", "1944-01", "1944-06"),
           ("1944 H2", "1944-07", "1944-12"), ("1945", "1945-01", "1945-08")]

TANKER_COLOUR = ORANGE   # tankers in the share chart (the subject of this script)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load():
    """All merchant ships (with area, route group and weapon shares) and a tanker flag.

    Built on shipping_routes.merchant_ships(), so the route groups and the
    weapon rules are exactly those of the routes analysis.
    """
    ships = routes.merchant_ships()
    ships["tanker"] = (ships["ship_type"] == "tanker").astype(int)
    ships["aircraft_only"] = (ships["share_aircraft"] == 1).astype(int)
    return ships


def monthly_table(ships):
    """Per month: merchant ships and GRT sunk, tanker ships and GRT, and tanker GRT per weapon.

    Every war month is present (none had zero merchant losses: the quietest
    had 9), so the series is evenly spaced for the change-point search.
    """
    tankers = ships[ships["tanker"] == 1]
    table = pd.DataFrame(index=MONTHS)
    table["merchant_ships"] = ships.groupby("month").size()
    table["merchant_grt"] = ships.groupby("month")["tonnage"].sum()
    table["tanker_ships"] = tankers.groupby("month").size()
    table["tanker_grt"] = tankers.groupby("month")["tonnage"].sum()
    for weapon in WEAPON_ORDER:
        table[f"tanker_ships_{weapon}"] = tankers.groupby("month")[f"share_{weapon}"].sum()
        table[f"tanker_grt_{weapon}"] = (tankers[f"share_{weapon}"] * tankers["tonnage"]).groupby(
            tankers["month"]).sum()
    table = table.fillna(0)
    table["tanker_share_ships"] = table["tanker_ships"] / table["merchant_ships"]
    table["tanker_share_grt"] = table["tanker_grt"] / table["merchant_grt"]
    table.index.name = "month"
    return table


# ---------------------------------------------------------------------------
# T1: change points in monthly tanker GRT
# ---------------------------------------------------------------------------
def grt_phases(monthly):
    """Change points in log(1 + monthly tanker GRT), BIC choosing 0-3 breaks.

    Uses distance_from_tokyo.change_points (ruptures.Dynp, L2 cost, phases of
    at least 6 months, BIC). The log makes the noise of a busy month and a
    quiet month comparable, the same reason us_loss_time_series.py
    transforms its series; the +1 keeps the 4 months with no tanker loss
    defined.
    """
    values = np.log1p(monthly["tanker_grt"].to_numpy())
    breaks, bics = change_points(values)
    edges = [0] + list(breaks) + [len(MONTHS)]
    rows = []
    for phase, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        part = monthly.iloc[a:b]
        rows.append({"measure": "tanker GRT per month", "phase": phase,
                     "start": str(MONTHS[a]), "end": str(MONTHS[b - 1]), "months": b - a,
                     "tanker_ships": part["tanker_ships"].sum(),
                     "tanker_grt": part["tanker_grt"].sum(),
                     "mean_grt_per_month": part["tanker_grt"].mean()})
    test = {"test": "T1 Change points in log(1 + monthly tanker GRT)", "estimate": len(breaks),
            "unit": "breaks chosen by BIC", "n": len(MONTHS),
            "detail": "BIC " + ", ".join(f"k={k}: {b:.1f}" for k, b in enumerate(bics))
                      + " | breaks at " + ", ".join(str(MONTHS[b]) for b in breaks)}
    return pd.DataFrame(rows), breaks, test


# ---------------------------------------------------------------------------
# T2: beta-binomial segmented model of the tanker share
# ---------------------------------------------------------------------------
def betabinom_loglik(y, n, p, rho):
    """Log-likelihood of counts y out of n under a beta-binomial with mean p and ICC rho.

    The beta-binomial lets the monthly share vary more than a binomial would
    (overdispersion): with a = p (1 - rho) / rho and b = (1 - p)(1 - rho) / rho,
    E[y] = n p and Var[y] = n p (1 - p) (1 + (n - 1) rho). rho is the
    correlation between two ships sunk in the same month; it captures
    tankers being lost together (a tanker convoy caught at once). Written
    with log-gamma functions for numerical stability.
    """
    a = p * (1 - rho) / rho
    b = (1 - p) * (1 - rho) / rho
    return float(np.sum(special.gammaln(n + 1) - special.gammaln(y + 1)
                        - special.gammaln(n - y + 1)
                        + special.betaln(y + a, n - y + b) - special.betaln(a, b)))


def fit_segments(y, n, breaks):
    """Maximum-likelihood fit of a beta-binomial with one share per phase and a common rho.

    Parameters are optimised on the logit scale (so they stay inside 0-1);
    start values are each phase's pooled share and rho = 0.05. Returns
    (log-likelihood, shares per phase, rho).
    """
    edges = [0] + list(breaks) + [len(y)]
    segment = np.repeat(np.arange(len(edges) - 1), np.diff(edges))
    pooled = [max(min(y[a:b].sum() / n[a:b].sum(), 0.99), 0.01)
              for a, b in zip(edges[:-1], edges[1:])]
    start = np.r_[special.logit(pooled), special.logit(0.05)]

    def negative(theta):
        p = special.expit(theta[:-1])[segment]
        rho = special.expit(theta[-1])
        return -betabinom_loglik(y, n, p, rho)

    fit = optimize.minimize(negative, start, method="L-BFGS-B",
                            bounds=[(-12, 12)] * len(start))
    return -fit.fun, special.expit(fit.x[:-1]), special.expit(fit.x[-1])


def best_breaks(y, n, k):
    """Exhaustive search for the k break months that maximise the beta-binomial likelihood.

    Every phase must be at least MIN_SEGMENT months long. With 45 months
    that is 34 candidates for one break and a few hundred pairs for two,
    small enough to try them all, so the best split is exact, not a
    greedy approximation.
    """
    if k == 0:
        return fit_segments(y, n, []) + ([],)
    best = None
    for breaks in itertools.combinations(range(MIN_SEGMENT, len(y) - MIN_SEGMENT + 1), k):
        if any(b - a < MIN_SEGMENT for a, b in zip(breaks[:-1], breaks[1:])):
            continue
        ll, shares, rho = fit_segments(y, n, breaks)
        if best is None or ll > best[0]:
            best = (ll, shares, rho, list(breaks))
    return best


def share_phases(monthly, rng):
    """T2: phases of the tanker share of merchant ships sunk, and the tests.

    Model choice: BIC = -2 logL + q log(T), with T = 45 months and q = number
    of phase shares + 1 (rho) + number of breaks (a break date is an
    estimated parameter too, the same count as the change-point BIC in the
    other scripts).

    Test of "the share never changed": the sup-LR statistic
    LR = 2 (logL_1break - logL_0breaks), maximised over the break date. Its
    null distribution is NOT chi-square (the break date only exists under
    the alternative: Davies' problem), so it is calibrated by a parametric
    bootstrap: simulate monthly tanker counts from the fitted no-change
    beta-binomial with the observed monthly totals, redo the whole search,
    and see how often the simulated LR reaches the observed one.

    Overdispersion: binomial (rho = 0) vs beta-binomial, LR test with the
    boundary-corrected p-value 0.5 x P(chi-square_1 > LR), as in
    us_loss_time_series.py.

    Each phase's share gets a month-cluster bootstrap 95% interval.
    """
    y = monthly["tanker_ships"].to_numpy().astype(float)
    n = monthly["merchant_ships"].to_numpy().astype(float)
    T = len(y)
    fits, bics = {}, []
    for k in range(MAX_SHARE_BREAKS + 1):
        ll, shares, rho, breaks = best_breaks(y, n, k)
        fits[k] = (ll, shares, rho, breaks)
        q = (k + 1) + 1 + k
        bics.append(-2 * ll + q * math.log(T))
    k_best = int(np.argmin(bics))
    ll, shares, rho, breaks = fits[k_best]

    lr_observed = 2 * (fits[1][0] - fits[0][0])
    p0, rho0 = fits[0][1][0], fits[0][2]
    a0, b0 = p0 * (1 - rho0) / rho0, (1 - p0) * (1 - rho0) / rho0
    exceed = 0
    for _ in range(LR_BOOTSTRAP_SAMPLES):
        y_sim = stats.betabinom.rvs(n.astype(int), a0, b0, random_state=rng).astype(float)
        lr_sim = 2 * (best_breaks(y_sim, n, 1)[0] - fit_segments(y_sim, n, [])[0])
        exceed += lr_sim >= lr_observed
    p_suplr = (1 + exceed) / (1 + LR_BOOTSTRAP_SAMPLES)

    # Binomial fit with the same breaks: rho fixed near 0.
    edges = [0] + breaks + [T]
    binom_ll = sum(stats.binom.logpmf(y[a:b], n[a:b], y[a:b].sum() / n[a:b].sum()).sum()
                   for a, b in zip(edges[:-1], edges[1:]))
    lr_rho = max(0.0, 2 * (ll - binom_ll))

    rows = []
    for phase, (a, b) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        months = np.arange(a, b)
        boot = []
        for _ in range(BOOTSTRAP_SAMPLES):
            pick = rng.choice(months, len(months))
            boot.append(y[pick].sum() / n[pick].sum())
        rows.append({"measure": "tanker share of merchant ships sunk", "phase": phase,
                     "start": str(MONTHS[a]), "end": str(MONTHS[b - 1]), "months": b - a,
                     "tanker_ships": y[a:b].sum(), "merchant_ships": n[a:b].sum(),
                     "share_fitted": shares[phase - 1],
                     "share_ci_low": np.percentile(boot, 2.5),
                     "share_ci_high": np.percentile(boot, 97.5)})
    tests = [{"test": "T2 Sup-LR test, any change in the tanker share (beta-binomial, 1 break)",
              "statistic": lr_observed, "p_value": p_suplr, "unit": "LR (parametric bootstrap p)",
              "n": T, "detail": f"{LR_BOOTSTRAP_SAMPLES} simulations; break at "
                                f"{MONTHS[fits[1][3][0]]}"},
             {"test": "T2 Breaks in the tanker share chosen by BIC", "estimate": k_best,
              "unit": "breaks", "n": T,
              "detail": "BIC " + ", ".join(f"k={k}: {b:.1f}" for k, b in enumerate(bics))
                        + " | breaks at " + ", ".join(str(MONTHS[b]) for b in breaks)},
             {"test": "T2 Overdispersion, beta-binomial vs binomial (same phases)",
              "statistic": lr_rho, "estimate": rho, "unit": "rho (intra-month correlation)",
              "p_value": 0.5 * stats.chi2.sf(lr_rho, 1), "n": T}]
    return pd.DataFrame(rows), breaks, tests


# ---------------------------------------------------------------------------
# T3: were tankers singled out by submarines?
# ---------------------------------------------------------------------------
def period_of(month):
    """Return the T3 period label of a month (see PERIODS)."""
    for label, start, end in PERIODS:
        if pd.Period(start, "M") <= month <= pd.Period(end, "M"):
            return label
    return None


def submarine_odds(ships):
    """Per-period and pooled odds ratios: tanker vs other merchant ship, submarine kill vs other.

    In each period a 2x2 table:
                       sunk by submarine only   any other cause
        tanker                  a                      b
        other merchant          c                      d
    OR = (a d) / (b c). OR > 1: a tanker was more likely than another
    merchant ship to have been sunk by a submarine (rather than by
    aircraft, mines, ...). The Mantel-Haenszel common OR pools the periods
    while holding period fixed, so it is not confounded by the weapon mix
    and the tanker share both changing over time. Breslow-Day tests H0: the
    OR is the same in every period, i.e. whether the submarines' focus on
    tankers (relative to other weapons) changed.
    """
    data = ships.assign(period=ships["month"].apply(period_of))
    tables, rows = [], []
    for label, _, _ in PERIODS:
        part = data[data["period"] == label]
        table = np.array([[((part["tanker"] == 1) & (part["submarine_only"] == 1)).sum(),
                           ((part["tanker"] == 1) & (part["submarine_only"] == 0)).sum()],
                          [((part["tanker"] == 0) & (part["submarine_only"] == 1)).sum(),
                           ((part["tanker"] == 0) & (part["submarine_only"] == 0)).sum()]])
        tables.append(table)
        t2 = Table2x2(table)
        low, high = t2.oddsratio_confint()
        rows.append({"period": label, "tanker_sub": table[0, 0], "tanker_other": table[0, 1],
                     "other_sub": table[1, 0], "other_other": table[1, 1],
                     "odds_ratio": t2.oddsratio, "ci_low": low, "ci_high": high,
                     "p_value": t2.test_nominal_association().pvalue})
    strat = StratifiedTable(tables)
    low, high = strat.oddsratio_pooled_confint()
    rows.append({"period": "Mantel-Haenszel pooled", "odds_ratio": strat.oddsratio_pooled,
                 "ci_low": low, "ci_high": high, "p_value": strat.test_null_odds().pvalue})
    bd = strat.test_equal_odds()
    tests = [{"test": "T3 Cochran-Mantel-Haenszel, tanker x submarine kill given period",
              "statistic": strat.test_null_odds().statistic,
              "p_value": strat.test_null_odds().pvalue, "estimate": strat.oddsratio_pooled,
              "ci_low": low, "ci_high": high, "unit": "Mantel-Haenszel common odds ratio",
              "n": int(sum(t.sum() for t in tables))},
             {"test": "T3 Breslow-Day, equal odds ratios in all periods",
              "statistic": bd.statistic, "df": len(tables) - 1, "p_value": bd.pvalue,
              "unit": "chi-square", "n": int(sum(t.sum() for t in tables))}]
    return pd.DataFrame(rows), tests


# ---------------------------------------------------------------------------
# T4: where tankers died
# ---------------------------------------------------------------------------
def route_groups(ships, rng):
    """Route group x (tanker, other merchant) table, chi-square and adjusted residuals.

    H0: tankers were lost in the same route groups, in the same proportions,
    as other merchant ships. Computed with the permutation chi-square of
    weapon_ship_analysis.py.
    """
    placed = ships[ships["group"].notna()]
    table = pd.crosstab(placed["group"], placed["tanker"].map({1: "tanker", 0: "other merchant"}))
    table = table.reindex(index=GROUP_ORDER)[["tanker", "other merchant"]]
    result = chi_square(table, rng)
    v, v_corr = cramers_v(result["x2"], result["n"], *table.shape)
    residuals = adjusted_residuals(table, result["expected"])
    expected = pd.DataFrame(result["expected"], index=table.index, columns=table.columns)
    out = pd.concat({"observed": table, "expected": expected, "adjusted_residual": residuals},
                    axis=1)
    test = {"test": "T4 Chi-square, route group x tanker/other merchant",
            "statistic": result["x2"], "df": result["dof"], "p_value": result["p_permutation"],
            "estimate": v_corr, "unit": "bias-corrected Cramer's V (p: permutation)",
            "n": result["n"], "detail": f"asymptotic p {result['p_asymptotic']:.3g}"}
    return out, test


# ---------------------------------------------------------------------------
# T5: tanker size over time
# ---------------------------------------------------------------------------
def size_trend(tankers):
    """Median regression of log(GRT) on time, and Spearman's rho.

    log(GRT) because tanker sizes are strongly skewed (503 to 19,262 GRT)
    and bimodal: big ocean tankers of 5,000-10,000 GRT and small wartime
    coastal tankers under 1,000 GRT. On the log scale the slope reads as a
    percentage change: 100 x (exp(12 b) - 1) % per year in the median size.
    Median regression is not pulled by the two clumps the way a mean would be.
    """
    data = tankers.assign(log_grt=np.log(tankers["tonnage"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.quantreg("log_grt ~ t", data).fit(q=0.5, max_iter=5000)
    low, high = fit.conf_int().loc["t"]
    rho, p = stats.spearmanr(data["t"], data["tonnage"])
    return [{"test": "T5 Median regression, log(GRT) of tankers sunk on time",
             "estimate": 100 * (math.exp(12 * fit.params["t"]) - 1),
             "ci_low": 100 * (math.exp(12 * low) - 1), "ci_high": 100 * (math.exp(12 * high) - 1),
             "p_value": fit.pvalues["t"], "unit": "% change in median GRT per year",
             "n": len(data)},
            {"test": "T5 Spearman, sinking date vs tanker GRT", "estimate": rho, "p_value": p,
             "unit": "rho", "n": len(data)}], fit


# ---------------------------------------------------------------------------
# T6: aircraft overtake submarines
# ---------------------------------------------------------------------------
def crossover(tankers, rng):
    """Logistic regression of 'aircraft' vs 'submarine' on time, and the crossover date.

    Tankers sunk by aircraft only or by submarines only (229 of 262). Model
    logit P(aircraft) = b0 + b1 t. The crossover, when an aircraft kill
    became more likely than a submarine kill, is where the logit is 0:
    t* = -b0 / b1. Its 95% interval comes from a month-cluster bootstrap
    (whole months resampled, model refitted, t* recomputed), which avoids
    both the delta method's normal approximation for a ratio and the
    assumption of independent ships. Standard errors of b1 are clustered by
    month.
    """
    data = tankers[(tankers["aircraft_only"] == 1) | (tankers["submarine_only"] == 1)].copy()
    groups = data["month"].astype(str).factorize()[0]
    fit = smf.logit("aircraft_only ~ t", data).fit(disp=0, cov_type="cluster",
                                                   cov_kwds={"groups": groups})
    b0, b1 = fit.params["Intercept"], fit.params["t"]
    t_star = -b0 / b1
    by_month = [g for _, g in data.groupby("month")]
    boot = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = pd.concat([by_month[i] for i in rng.integers(0, len(by_month), len(by_month))])
        if sample["aircraft_only"].nunique() < 2:
            continue
        X = sm.add_constant(sample["t"].to_numpy())
        try:
            params = sm.Logit(sample["aircraft_only"].to_numpy(), X).fit(disp=0).params
        except Exception:   # perfect separation in a rare resample: skip it
            continue
        if params[1] > 0:
            boot.append(-params[0] / params[1])
    low, high = np.percentile(boot, [2.5, 97.5])
    low_b, high_b = fit.conf_int().loc["t"]
    tests = [{"test": "T6 Logistic, P(aircraft rather than submarine) for tankers, per year",
              "estimate": math.exp(12 * b1), "ci_low": math.exp(12 * low_b),
              "ci_high": math.exp(12 * high_b), "p_value": fit.pvalues["t"],
              "unit": "odds ratio per year (cluster-robust)", "n": len(data)},
             {"test": "T6 Crossover date, aircraft as likely as submarines",
              "estimate": t_star, "ci_low": low, "ci_high": high,
              "unit": "months since 7 Dec 1941 (CI: month-cluster bootstrap)", "n": len(data),
              "detail": f"{routes.t_to_date(t_star).date()} (95% CI "
                        f"{routes.t_to_date(low).date()} to {routes.t_to_date(high).date()}); "
                        f"{len(boot)} bootstrap refits"}]
    return tests, fit, data


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def month_x():
    """Mid-month dates for plotting monthly values."""
    return MONTHS.to_timestamp() + pd.Timedelta(days=14)


def format_time_axis(ax):
    """January and July ticks, labelled 'Jan 1944', across the whole war."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(MONTHS[0].to_timestamp(), (MONTHS[-1] + 1).to_timestamp())


def stacked_weapon_bars(ax, monthly, measure, scale=1.0):
    """Stack one bar per month by weapon (1/n shares), in WEAPON_ORDER from the bottom."""
    x = month_x()
    bottom = np.zeros(len(monthly))
    for weapon in WEAPON_ORDER:
        values = monthly[f"tanker_{measure}_{weapon}"].to_numpy() / scale
        if values.sum() == 0:
            continue
        ax.bar(x, values, bottom=bottom, width=26, color=WEAPON_COLOURS[weapon],
               edgecolor=BACKGROUND, linewidth=0.5, label=WEAPON_LABELS[weapon], zorder=2)
        bottom += values


def draw_monthly(monthly, phases, breaks, test, out_path):
    """T1: tanker ships and GRT sunk per month by weapon, and cumulative GRT.

    Three panels on one time axis, each with its own y scale (counts,
    thousands of GRT, millions of GRT): separate panels instead of a second
    y axis, so no scale can be misread against another. Dotted lines mark
    the GRT change points; a dashed line in the GRT panel shows each phase's
    mean monthly GRT.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.4, 0.9], "hspace": 0.12})
    stacked_weapon_bars(axes[0], monthly, "ships")
    axes[0].set_ylabel("Tankers sunk")
    stacked_weapon_bars(axes[1], monthly, "grt", scale=1000)
    axes[1].set_ylabel("Thousand GRT")
    for _, phase in phases.iterrows():
        start = pd.Period(phase["start"], "M").to_timestamp()
        end = (pd.Period(phase["end"], "M") + 1).to_timestamp()
        axes[1].hlines(phase["mean_grt_per_month"] / 1000, start, end, color=INK,
                       linestyle="--", linewidth=1.3, zorder=3)
        axes[1].text(start + (end - start) / 2, phase["mean_grt_per_month"] / 1000 + 4,
                     f"{phase['mean_grt_per_month'] / 1000:,.0f}k GRT/month", ha="center",
                     fontsize=10, color=INK, zorder=4,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=BACKGROUND,
                               edgecolor="none", alpha=0.85))
    cumulative = monthly["tanker_grt"].cumsum() / 1e6
    axes[2].fill_between(month_x(), cumulative, color=tint(TANKER_COLOUR, 0.6), step="mid")
    axes[2].plot(month_x(), cumulative, color=TANKER_COLOUR, linewidth=2, drawstyle="steps-mid")
    axes[2].set_ylabel("Million GRT,\ncumulative")
    axes[2].text(month_x()[-1], cumulative.iloc[-1], f" {cumulative.iloc[-1]:.2f}M ",
                 ha="right", va="bottom", fontsize=11, color=INK)
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        for b in breaks:
            ax.axvline(MONTHS[b].to_timestamp(), color=INK_SOFT, linestyle=":", linewidth=1.3)
    for b in breaks:
        axes[0].text(MONTHS[b].to_timestamp(), axes[0].get_ylim()[1] * 0.97,
                     f" {MONTHS[b].strftime('%b %Y')}", va="top", fontsize=10, color=INK_SOFT)
    format_time_axis(axes[2])
    # Legend in the empty top-left of the first panel, under the break
    # labels, so it does not cover the "Jan 1943" label.
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.88), ncol=1, frameon=False,
                   fontsize=10.5)
    fig.suptitle("The tanker war: Japanese tankers sunk, month by month",
                 x=0.06, ha="left", y=0.985, fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 0.945, f"{int(monthly['tanker_ships'].sum())} tankers, "
             f"{monthly['tanker_grt'].sum() / 1e6:.2f} million GRT. Bars split by weapon "
             f"(joint kills shared equally). Dotted lines: change points in monthly GRT "
             f"({test['detail'].split('|')[1].strip()}); dashed: each phase's mean.",
             fontsize=11.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.05)
    save_figure(fig, out_path)


def draw_share(monthly, phases, tests, out_path):
    """T2: tanker share of merchant ships sunk (top) and of merchant GRT sunk (bottom).

    Top: one dot per month, area proportional to the number of merchant
    ships sunk that month (a share out of 9 ships is noisier than one out of
    120), with the fitted beta-binomial phase shares as steps and their
    bootstrap intervals as bands. Bottom: the GRT share as bars, descriptive
    only: tankers are larger than the average merchant ship, so their GRT
    share is higher than their ship share.
    """
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True,
                                  gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.1})
    x = month_x()
    ax.scatter(x, monthly["tanker_share_ships"] * 100, s=monthly["merchant_ships"] * 1.6,
               color=TANKER_COLOUR, alpha=0.45, edgecolors="white", linewidths=0.6, zorder=3)
    for _, phase in phases.iterrows():
        start = pd.Period(phase["start"], "M").to_timestamp()
        end = (pd.Period(phase["end"], "M") + 1).to_timestamp()
        ax.fill_between([start, end], phase["share_ci_low"] * 100, phase["share_ci_high"] * 100,
                        color=tint(INK_SOFT, 0.8), zorder=1)
        ax.hlines(phase["share_fitted"] * 100, start, end, color=INK, linewidth=2.2, zorder=4)
        ax.text(start + (end - start) / 2, phase["share_ci_high"] * 100 + 1.5,
                f"{phase['share_fitted'] * 100:.1f}% "
                f"({phase['share_ci_low'] * 100:.1f}–{phase['share_ci_high'] * 100:.1f})",
                ha="center", fontsize=11, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=BACKGROUND, edgecolor="none",
                          alpha=0.85))
    ax.set_ylabel("Tankers, % of merchant\nships sunk that month")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylim(0, max(45, monthly["tanker_share_ships"].max() * 100 + 5))
    handles = [Line2D([], [], marker="o", linestyle="", color=TANKER_COLOUR, alpha=0.5,
                      markersize=9, label="One month (dot area = merchant ships sunk)"),
               Line2D([], [], color=INK, linewidth=2.2, label="Phase share (beta-binomial fit)"),
               Patch(color=tint(INK_SOFT, 0.8), label="95% month-cluster bootstrap interval")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10.5)

    ax2.bar(x, monthly["tanker_share_grt"] * 100, width=24, color=tint(TANKER_COLOUR, 0.35),
            zorder=2)
    ax2.set_ylabel("Tankers, % of\nmerchant GRT sunk")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    format_time_axis(ax2)

    sup = next(t for t in tests if t["test"].startswith("T2 Sup-LR"))
    rho = next(t for t in tests if t["test"].startswith("T2 Overdispersion"))
    fig.suptitle("Tankers became a growing share of Japan's merchant losses",
                 x=0.06, ha="left", y=0.985, fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 0.935, f"Sup-LR test of no change: LR = {sup['statistic']:.1f}, parametric "
             f"bootstrap p = {sup['p_value']:.3f}. Overdispersion ρ = {rho['estimate']:.3f}. "
             "Phases chosen by BIC (each at least 6 months).",
             fontsize=11.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.06)
    save_figure(fig, out_path)


def draw_weapons(odds, cross_fit, cross_data, cross_tests, out_path):
    """T3 forest plot of odds ratios (left) and T6 aircraft-vs-submarine curve (right).

    Left: odds ratio (log scale, so 0.5 and 2 are equally far from 1) of a
    submarine kill for tankers vs other merchant ships, per period and
    pooled. Right: the modelled probability that a tanker kill was by
    aircraft rather than by a submarine, with the observed quarterly shares
    as dots and the crossover date marked.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(17, 6.8),
                                      gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.28})
    rows = odds.iloc[::-1].reset_index(drop=True)
    for y, row in rows.iterrows():
        pooled = row["period"].startswith("Mantel")
        colour = INK if pooled else TANKER_COLOUR
        left.plot([row["ci_low"], row["ci_high"]], [y, y], color=colour, linewidth=2)
        left.plot(row["odds_ratio"], y, marker="D" if pooled else "o", color=colour,
                  markersize=10, markeredgecolor="white")
        label = f"{row['odds_ratio']:.2f} ({row['ci_low']:.2f}–{row['ci_high']:.2f})"
        left.text(row["ci_high"] * 1.08, y, label, va="center", fontsize=10.5, color=INK)
    left.axvline(1, color=INK_SOFT, linestyle="--", linewidth=1)
    left.set_xscale("log")
    left.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
    left.set_xlim(0.15, 12)
    labels = [r["period"] if r["period"].startswith("Mantel") else
              f"{r['period']}  ({int(r['tanker_sub'] + r['tanker_other'])} tankers)"
              for _, r in rows.iterrows()]
    left.set_yticks(range(len(rows)), labels)
    left.set_xlabel("Odds ratio: tanker vs other merchant ship\nof being sunk by a submarine")
    left.set_title("Did submarines single out tankers?", loc="left", fontsize=13.5, color=INK)
    left.grid(axis="x", color=GRID, linewidth=0.6)

    grid = pd.DataFrame({"t": np.linspace(cross_data["t"].min(), cross_data["t"].max(), 200)})
    dates = [routes.t_to_date(t) for t in grid["t"]]
    right.plot(dates, cross_fit.predict(grid) * 100, color=INK, linewidth=2.4, zorder=3)
    quarterly = cross_data.assign(q=cross_data["month"].apply(lambda m: m.asfreq("Q")))
    shares = quarterly.groupby("q")["aircraft_only"].agg(["mean", "size"])
    right.scatter([q.to_timestamp() + pd.Timedelta(days=45) for q in shares.index],
                  shares["mean"] * 100, s=shares["size"] * 6, color=WEAPON_COLOURS["aircraft"],
                  alpha=0.6, edgecolors="white", zorder=2)
    t_star = cross_tests[1]
    start, end = routes.t_to_date(t_star["ci_low"]), routes.t_to_date(t_star["ci_high"])
    right.axvspan(start, end, color=tint(INK_SOFT, 0.85), zorder=0)
    right.axvline(routes.t_to_date(t_star["estimate"]), color=INK_SOFT, linestyle=":")
    right.axhline(50, color=INK_SOFT, linestyle="--", linewidth=1)
    right.text(routes.t_to_date(t_star["estimate"]), 96,
               f" crossover {routes.t_to_date(t_star['estimate']).strftime('%d %b %Y')}\n"
               f" 95% CI {start.strftime('%b %Y')} – {end.strftime('%b %Y')}",
               va="top", fontsize=10.5, color=INK)
    right.set_ylim(0, 100)
    right.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    right.set_ylabel("% of tanker kills by aircraft\n(aircraft or submarine kills only)")
    right.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    right.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    right.grid(axis="y", color=GRID, linewidth=0.6)
    right.set_title(f"Aircraft overtake submarines as the tanker killer (odds ×"
                    f"{cross_tests[0]['estimate']:.1f} per year)", loc="left", fontsize=13.5,
                    color=INK)
    fig.suptitle("Who sank the tankers", x=0.06, ha="left", y=0.985, fontsize=18,
                 fontweight="bold", color=INK)
    fig.text(0.06, 0.012, "Left: 'sunk by a submarine' means every credit is a submarine; "
             "95% intervals: Woolf (log odds ratio) for each period, Mantel-Haenszel for the "
             "pool.\nRight: logistic regression on time (line), observed quarterly shares "
             "(dots, area = tankers), crossover interval from a month-cluster bootstrap "
             "(grey band).", fontsize=9.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.87, bottom=0.15)
    save_figure(fig, out_path)


def draw_where_size(tankers, route_table, size_fit, size_tests, route_test, out_path):
    """T4: tanker GRT by route group and year (left); T5: tanker size over time (right).

    Left: stacked bars per year, coloured by route group; the route-group
    residuals (tanker vs other merchant) are written under the bars.
    Right: every tanker as a dot (GRT on a log axis, coloured by its main
    weapon) with the fitted median line, showing the shift from big ocean
    tankers to small coastal ones.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(17, 7),
                                      gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.22})
    years = [1942, 1943, 1944, 1945]
    data = tankers[tankers["group"].notna()].assign(
        year=tankers["month"].apply(lambda m: max(m.year, 1942)))
    bottom = np.zeros(len(years))
    for group in GROUP_ORDER:
        values = data[data["group"] == group].groupby("year")["tonnage"].sum().reindex(
            years, fill_value=0).to_numpy() / 1000
        left.bar([str(y) for y in years], values, bottom=bottom, color=GROUP_COLOURS[group],
                 edgecolor=BACKGROUND, linewidth=1, label=GROUP_LABELS[group])
        bottom += values
    left.set_ylabel("Tanker GRT sunk (thousands)")
    left.grid(axis="y", color=GRID, linewidth=0.6)
    left.legend(loc="upper left", frameon=False, fontsize=10.5)
    residuals = route_table["adjusted_residual"]["tanker"]
    left.set_title("Where tankers were sunk", loc="left", fontsize=13.5, color=INK)
    left.text(0, -0.14, "Tankers vs other merchant ships, adjusted residuals: "
              + ", ".join(f"{GROUP_LABELS[g].split(':')[0]} {residuals[g]:+.1f}"
                          for g in GROUP_ORDER)
              + f"\nχ² = {route_test['statistic']:.1f}, permutation p = "
                f"{route_test['p_value']:.4f}, V = {route_test['estimate']:.2f}",
              transform=left.transAxes, fontsize=9.5, color=INK_SOFT, va="top")

    main_weapon = tankers[[f"share_{w}" for w in WEAPON_ORDER]].idxmax(axis=1).str[6:]
    for weapon in WEAPON_ORDER:
        part = tankers[main_weapon == weapon]
        if part.empty:
            continue
        right.scatter(part["sunk"], part["tonnage"], s=26, color=WEAPON_COLOURS[weapon],
                      alpha=0.7, edgecolors="white", linewidths=0.4, label=WEAPON_LABELS[weapon],
                      zorder=2)
    # The trend is drawn as half-year medians, not as the fitted regression
    # line: sizes form two clumps (ocean tankers ~5,000-10,000 GRT, coastal
    # tankers under 1,000), so a straight line on the log scale would cut
    # between them and, extended back to early 1942 where there are only 3
    # tankers, suggest 30,000-GRT ships that never existed. The regression
    # result is still reported in the title.
    halves = tankers.assign(half=tankers["sunk"].dt.year.astype(str) + "H"
                            + np.where(tankers["sunk"].dt.month <= 6, "1", "2"))
    medians = halves.groupby("half").agg(grt=("tonnage", "median"), n=("tonnage", "size"),
                                         when=("sunk", "median"))
    medians = medians[medians["n"] >= 5]
    right.plot(medians["when"], medians["grt"], color=INK, linewidth=2.2, marker="s",
               markersize=7, label="Half-year median (halves with ≥ 5 tankers)", zorder=3)
    right.set_yscale("log")
    right.set_yticks([500, 1000, 2000, 5000, 10000, 20000],
                     ["500", "1,000", "2,000", "5,000", "10,000", "20,000"])
    right.set_ylabel("GRT (log scale)")
    right.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    right.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    right.grid(axis="y", color=GRID, linewidth=0.6)
    right.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=10)
    s = size_tests[0]
    right.set_title(f"Tankers sunk got smaller: median size {s['estimate']:+.0f}% per year "
                    f"(95% CI {s['ci_low']:+.0f} to {s['ci_high']:+.0f}%)", loc="left",
                    fontsize=13.5, color=INK)
    fig.suptitle("The tanker war: where, and how big", x=0.06, ha="left", y=0.985,
                 fontsize=18, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.17)
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def main():
    """Run T1-T6, save every table and chart, and print the main numbers."""
    helpers.set_readable_style()
    rng = np.random.default_rng(RANDOM_SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ships = load()
    tankers = ships[ships["tanker"] == 1].copy()
    monthly = monthly_table(ships)
    monthly.to_csv(DATA_DIR / "tanker_monthly.csv", float_format="%.4g")

    grt_phase_table, grt_breaks, grt_test = grt_phases(monthly)
    share_phase_table, _, share_tests = share_phases(monthly, rng)
    odds, odds_tests = submarine_odds(ships)
    route_table, route_test = route_groups(ships, rng)
    size_tests, size_fit = size_trend(tankers)
    cross_tests, cross_fit, cross_data = crossover(tankers, rng)

    tests = [grt_test] + share_tests + odds_tests + [route_test] + size_tests + cross_tests
    pd.concat([grt_phase_table, share_phase_table]).to_csv(DATA_DIR / "tanker_phases.csv",
                                                           index=False, float_format="%.4g")
    odds.to_csv(DATA_DIR / "tanker_odds_by_period.csv", index=False, float_format="%.4g")
    route_table.to_csv(DATA_DIR / "tanker_route_groups.csv", float_format="%.4g")
    pd.DataFrame(tests).to_csv(DATA_DIR / "tanker_tests.csv", index=False, float_format="%.4g")

    draw_monthly(monthly, grt_phase_table, grt_breaks, grt_test, FIG_DIR / "tanker_monthly.png")
    draw_share(monthly, share_phase_table, share_tests, FIG_DIR / "tanker_share.png")
    draw_weapons(odds, cross_fit, cross_data, cross_tests, FIG_DIR / "tanker_weapons.png")
    draw_where_size(tankers, route_table, size_fit, size_tests, route_test,
                    FIG_DIR / "tanker_where_size.png")

    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.max_colwidth", 100):
        print(pd.DataFrame(tests).to_string(index=False))
        print(grt_phase_table.to_string(index=False))
        print(share_phase_table.to_string(index=False))
        print(odds.round(3).to_string(index=False))
        print(route_table.round(2).to_string())
    print(f"Saved figures to {FIG_DIR} and tables to {DATA_DIR}")


if __name__ == "__main__":
    main()
