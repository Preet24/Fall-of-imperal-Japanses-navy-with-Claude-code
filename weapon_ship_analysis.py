"""
Who killed what: which weapons sank which types of Japanese ship, and were
some weapons sinking bigger ships than others? (Analyses B1 and B2)

B1 - weapon x ship type. For warships and merchant ships separately, a
     contingency table of ship type (rows) by weapon (columns):
       - chi-square test of independence, with a permutation p-value
         (several cells are small, so the chi-square approximation is not
         trusted on its own);
       - Cramer's V (bias-corrected) as the effect size;
       - adjusted standardised residuals, to see WHICH type-weapon pairs
         are over- or under-represented (Bonferroni-corrected);
       - correspondence analysis, a map that places types near the weapons
         that sank them;
       - a log-linear test of type x weapon within years, to check the
         association isn't just weapons and ship types both changing over
         the war.

B2 - ship size by weapon. For each fleet:
       - Kruskal-Wallis test of tonnage across weapons, with epsilon-squared;
       - Dunn's post-hoc pairwise tests (Holm-corrected) and Cliff's delta;
       - a regression of log tonnage on weapon, adjusted for ship type and
         year (HC3 robust errors), to check whether the size differences are
         just a different mix of ship types or periods.

Which ships are analysed. A ship is included when every weapon credited
with it is the same one of: submarine, aircraft, surface warship, mine.
Left out (counts are printed and saved):
  - ships sunk by two different weapons (e.g. submarine + aircraft), since
    a ship can only sit in one column of a contingency table;
  - losses credited to Japan itself (accidents and Japan's own mines): the
    question is which ENEMY weapons sank which ships;
  - shore batteries, sabotage and unknown causes (too few to analyse).
All Allied nations are included; the US is about 90% of them.

Tonnage: naval tons (standard displacement) and merchant GRT are different
units, so B2 never compares across fleets. Estimated tonnages (printed with
an "e") are used as printed.

Input:  data/naval/naval_losses_15-44.csv, data/merchant/merchant_losses_45-115.csv
Output:
    figures/weapon_ship/weapon_by_type_{naval,merchant}.png    B1 heatmap + CA map
    figures/weapon_ship/tonnage_by_weapon_{naval,merchant}.png B2 distributions + effects
    data/weapon_ship/contingency_{naval,merchant}.csv          observed, expected, residuals
    data/weapon_ship/correspondence_{naval,merchant}.csv       CA coordinates
    data/weapon_ship/dunn_{naval,merchant}.csv                 pairwise tests + Cliff's delta
    data/weapon_ship/tonnage_regression.csv                    weapon effects, raw and adjusted
    data/weapon_ship/tests.csv                                 every overall test, one row each

The method is explained in analysis_A1_B_C_explained.md.

Run:
    python weapon_ship_analysis.py
"""

import math

import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no window is needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats

import sinking_map as smap
import table_plot_helpers as helpers
from sinking_map import SHIP_TYPE_OF, VESSEL_OPTIONS
from table_plot_helpers import BACKGROUND, GRID, INK, INK_SOFT, tint
from us_loss_time_series import WEAPON_COLOURS, WEAPON_LABELS

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
FIG_DIR = smap.PROJECT_DIR / "figures" / "weapon_ship"
DATA_DIR = smap.PROJECT_DIR / "data" / "weapon_ship"

# The four weapons analysed, in the project's usual order.
WEAPONS = ["submarine", "aircraft", "surface", "mine"]
# Reference weapon for the regression: submarines, the largest group in both
# fleets, so every other weapon is compared with the one that sank the most.
REFERENCE = "submarine"

PERMUTATIONS = 9999
ALPHA = 0.05
RANDOM_SEED = 12345

# Readable names for the ship-type options.
TYPE_LABELS = {"battleship": "Battleship", "carrier": "Carrier", "cruiser": "Cruiser",
               "destroyer": "Destroyer", "submarine": "Submarine", "escort": "Escort",
               "mine_warfare": "Mine warfare", "landing_ship": "Landing ship / fast transport",
               "naval_auxiliary": "Naval auxiliary", "cargo": "Cargo",
               "passenger_cargo": "Passenger-cargo", "tanker": "Tanker",
               "transport": "Transport", "converted_auxiliary": "Converted auxiliary",
               "other_merchant": "Other merchant"}
UNITS = {"naval": "tons", "merchant": "GRT"}

# Diverging colours for the residual heatmap: slate blue = fewer than
# expected, brick red = more, near-white = as expected. The weapon colours
# (violet, aqua, pink, yellow) and the fleet colours (blue, orange) are not
# reused, because here colour means "more or fewer than chance", not a
# weapon or a fleet.
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "residuals", ["#3b6e8f", "#f4f3ee", "#b5452a"])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def single_weapon(credits):
    """Return the one weapon that sank a ship, or a reason the ship is left out.

    Returns (weapon, None) for an included ship, or (None, reason).
    """
    agents = {agent for agent, _ in credits}
    weapons = {weapon for _, weapon in credits}
    if "japan" in agents:
        return None, "credited to Japan (accident or own mine)"
    if len(weapons) > 1:
        return None, "sunk by two or more different weapons"
    weapon = weapons.pop()
    if weapon not in WEAPONS:
        return None, "shore battery, sabotage or unknown cause"
    return weapon, None


def analysis_ships():
    """Load all ships, keep those with one analysable weapon, and count the rest by reason."""
    ships = smap.load_ships()
    result = ships["credits"].apply(single_weapon)
    ships["weapon"] = result.str[0]
    ships["excluded_because"] = result.str[1]
    excluded = ships["excluded_because"].value_counts().rename_axis("reason").reset_index()
    kept = ships[ships["weapon"].notna()].copy()
    kept["log_tonnage"] = np.log(kept["tonnage"])
    return kept, excluded


# ---------------------------------------------------------------------------
# B1: contingency table, chi-square, residuals, correspondence analysis
# ---------------------------------------------------------------------------
def contingency(ships, vessel):
    """Ship type (rows, in the option order) x weapon (columns) table of ship counts."""
    part = ships[ships["vessel"] == vessel]
    types = [t for t in dict.fromkeys(SHIP_TYPE_OF[vessel].values())
             if t in set(part["ship_type"])]
    if vessel == "merchant" and "other_merchant" in set(part["ship_type"]) \
            and "other_merchant" not in types:
        types.append("other_merchant")
    table = pd.crosstab(part["ship_type"], part["weapon"]).reindex(
        index=types, columns=WEAPONS, fill_value=0)
    # A weapon that sank none of this fleet's ships would be an all-zero
    # column, which makes the expected counts 0; drop it.
    return table.loc[:, table.sum() > 0]


def chi_square(table, rng):
    """Chi-square test of independence with an asymptotic and a permutation p-value.

    Statistic: X^2 = sum (O - E)^2 / E, with E = row total x column total / n.
    Under independence X^2 ~ chi-square with (r-1)(c-1) df, but only
    approximately, and the approximation is poor when many expected counts
    are below 5. The permutation p-value avoids that: weapon labels are
    shuffled among the ships PERMUTATIONS times (which keeps both margins
    fixed, as the test assumes), X^2 is recomputed each time, and
    p = (1 + number of shuffled X^2 >= observed) / (1 + PERMUTATIONS).
    """
    observed = table.to_numpy()
    n = observed.sum()
    expected = np.outer(observed.sum(1), observed.sum(0)) / n
    x2 = float(((observed - expected) ** 2 / expected).sum())
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    # Rebuild one row per ship (row code, column code) and shuffle the columns.
    rows = np.repeat(np.repeat(np.arange(observed.shape[0]), observed.shape[1]),
                     observed.ravel())
    cols = np.repeat(np.tile(np.arange(observed.shape[1]), observed.shape[0]),
                     observed.ravel())
    k = observed.shape[1]
    exceed = 0
    for _ in range(PERMUTATIONS):
        shuffled = np.bincount(rows * k + rng.permutation(cols),
                               minlength=observed.size).reshape(observed.shape)
        exceed += ((shuffled - expected) ** 2 / expected).sum() >= x2 - 1e-9
    return {"x2": x2, "dof": dof, "p_asymptotic": stats.chi2.sf(x2, dof),
            "p_permutation": (1 + exceed) / (1 + PERMUTATIONS), "n": int(n),
            "cells_expected_below_5": int((expected < 5).sum()),
            "cells": int(expected.size), "expected": expected}


def cramers_v(x2, n, r, c):
    """Cramer's V and its bias-corrected version (Bergsma, 2013).

    V = sqrt(X^2 / (n (min(r, c) - 1))) runs from 0 (independent) to 1
    (each type sunk by only one weapon). Plain V is biased upwards in
    small or sparse tables; the corrected version subtracts the expected
    value of X^2/n under independence and corrects r and c the same way.
    """
    phi2 = x2 / n
    v = math.sqrt(phi2 / (min(r, c) - 1))
    phi2_corr = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    v_corr = math.sqrt(phi2_corr / min(r_corr - 1, c_corr - 1))
    return v, v_corr


def adjusted_residuals(table, expected):
    """Adjusted standardised residuals (Haberman, 1973).

    r_ij = (O - E) / sqrt(E (1 - row_i/n) (1 - col_j/n)). Under
    independence each is approximately N(0, 1), so |r| > 1.96 is unusual
    for one cell. Many cells are looked at, so the threshold used for
    "significant" is Bonferroni-corrected: z_(1 - alpha / (2 x cells)).
    """
    observed = table.to_numpy()
    n = observed.sum()
    row_share = observed.sum(1, keepdims=True) / n
    col_share = observed.sum(0, keepdims=True) / n
    residual = (observed - expected) / np.sqrt(expected * (1 - row_share) * (1 - col_share))
    return pd.DataFrame(residual, index=table.index, columns=table.columns)


def correspondence_analysis(table):
    """Correspondence analysis by SVD of the standardised residual matrix.

    With P = table / n, row masses r and column masses c, the matrix
    S = D_r^(-1/2) (P - r c') D_c^(-1/2) is decomposed as S = U Sigma V'.
    Principal coordinates: rows F = D_r^(-1/2) U Sigma, columns
    G = D_c^(-1/2) V Sigma. The squared singular values are the "principal
    inertias"; their sum is X^2 / n, and each one's share says how much of
    the association that dimension shows. Plotting rows and columns in the
    same space (a "symmetric map") puts a ship type near the weapons it was
    sunk by more often than chance.
    """
    p = table.to_numpy() / table.to_numpy().sum()
    r, c = p.sum(1), p.sum(0)
    s = (p - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    u, sigma, vt = np.linalg.svd(s, full_matrices=False)
    rows = (u * sigma) / np.sqrt(r)[:, None]
    cols = (vt.T * sigma) / np.sqrt(c)[:, None]
    inertia = sigma ** 2
    share = inertia / inertia.sum()
    coords = pd.concat([
        pd.DataFrame({"kind": "ship type", "name": table.index, "dim1": rows[:, 0],
                      "dim2": rows[:, 1], "mass": r}),
        pd.DataFrame({"kind": "weapon", "name": table.columns, "dim1": cols[:, 0],
                      "dim2": cols[:, 1], "mass": c})], ignore_index=True)
    return coords, share


def draw_contingency_figure(table, residuals, threshold, coords, share, result, v_corr,
                            vessel, out_path):
    """Heatmap of adjusted residuals (left) and the correspondence-analysis map (right).

    Heatmap: each cell shows the number of ships and, below it, the
    residual; the colour is the residual (brick = more than independence
    predicts, slate = fewer). Cells beyond the Bonferroni threshold are
    outlined. The colour scale is symmetric and capped at +/- 8 so the very
    large residuals don't wash out the rest.
    CA map: weapons as coloured squares (their usual colours), ship types as
    grey dots sized by how many ships they hold; axis labels give each
    dimension's share of the association.
    """
    fig, (heat_ax, ca_ax) = plt.subplots(1, 2, figsize=(17, 0.55 * len(table) + 5.2),
                                         gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.28})
    cap = 8
    heat_ax.imshow(residuals.clip(-cap, cap).to_numpy(), cmap=RESIDUAL_CMAP, vmin=-cap,
                   vmax=cap, aspect="auto")
    for i, ship_type in enumerate(table.index):
        for j, weapon in enumerate(table.columns):
            res = residuals.iloc[i, j]
            heat_ax.text(j, i - 0.12, f"{table.iloc[i, j]:,}", ha="center", va="center",
                         fontsize=12, fontweight="bold",
                         color="white" if abs(res) > 5.5 else INK)
            heat_ax.text(j, i + 0.22, f"{res:+.1f}", ha="center", va="center", fontsize=9.5,
                         color="white" if abs(res) > 5.5 else INK_SOFT)
            if abs(res) > threshold:
                heat_ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                edgecolor=INK, linewidth=1.6))
    heat_ax.set_xticks(range(len(table.columns)), [WEAPON_LABELS[w] for w in table.columns],
                       fontsize=11)
    heat_ax.set_yticks(range(len(table.index)), [TYPE_LABELS[t] for t in table.index],
                       fontsize=11)
    heat_ax.xaxis.tick_top()
    heat_ax.tick_params(length=0)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    heat_ax.set_title(f"Ships sunk (top) and adjusted residual (below); outlined: |residual| > "
                      f"{threshold:.2f} (Bonferroni)", loc="left", fontsize=11, color=INK_SOFT,
                      pad=34)

    types = coords[coords["kind"] == "ship type"]
    weapons = coords[coords["kind"] == "weapon"]
    ca_ax.axhline(0, color=GRID, linewidth=1, zorder=0)
    ca_ax.axvline(0, color=GRID, linewidth=1, zorder=0)
    ca_ax.scatter(types["dim1"], types["dim2"], s=30 + 1500 * types["mass"], color=INK_SOFT,
                  alpha=0.5, linewidths=0, zorder=2)
    # Ship types whose points almost coincide would print their labels on
    # top of each other; each label that is close to one already placed is
    # moved down a line instead.
    placed = []
    x_span, y_span = np.ptp(coords["dim1"]), np.ptp(coords["dim2"])
    for _, row in types.sort_values("mass", ascending=False).iterrows():
        near = sum(abs(row["dim1"] - px) < 0.12 * x_span and abs(row["dim2"] - py) < 0.06 * y_span
                   for px, py in placed)
        ca_ax.annotate(TYPE_LABELS[row["name"]], (row["dim1"], row["dim2"]),
                       xytext=(6, 4 - 14 * near), textcoords="offset points", fontsize=10,
                       color=INK_SOFT)
        placed.append((row["dim1"], row["dim2"]))
    for _, row in weapons.iterrows():
        ca_ax.scatter(row["dim1"], row["dim2"], marker="s", s=170,
                      color=WEAPON_COLOURS[row["name"]], edgecolors=INK, linewidths=0.8, zorder=3)
        ca_ax.annotate(WEAPON_LABELS[row["name"]], (row["dim1"], row["dim2"]), xytext=(8, -14),
                       textcoords="offset points", fontsize=11.5, fontweight="bold", color=INK)
    ca_ax.set_xlabel(f"Dimension 1 ({share[0]:.0%} of the association)")
    ca_ax.set_ylabel(f"Dimension 2 ({share[1]:.0%} of the association)" if len(share) > 1 else "")
    ca_ax.grid(False)
    ca_ax.set_title("Correspondence analysis: types sit near the weapons that sank them",
                    loc="left", fontsize=11, color=INK_SOFT)
    # Keep some room around the labels.
    xs = coords["dim1"]
    ys = coords["dim2"]
    ca_ax.set_xlim(xs.min() - 0.25 * np.ptp(xs), xs.max() + 0.35 * np.ptp(xs))
    ca_ax.set_ylim(ys.min() - 0.2 * np.ptp(ys), ys.max() + 0.2 * np.ptp(ys))

    fig.suptitle(f"Which weapons sank which {VESSEL_OPTIONS[vessel].split(' (')[0].lower()}",
                 x=0.04, ha="left", y=0.99, fontsize=18, fontweight="bold", color=INK)
    p_text = (f"p < {1 / (1 + PERMUTATIONS):.4f}" if result["p_permutation"] <= 1 / (1 + PERMUTATIONS)
              else f"p = {result['p_permutation']:.4f}")
    fig.text(0.04, 1 - 0.95 / fig.get_figheight(),
             f"Chi-square = {result['x2']:,.0f}, df = {result['dof']}, permutation {p_text} "
             f"({PERMUTATIONS:,} shuffles); Cramér's V (bias-corrected) = {v_corr:.2f}; "
             f"{result['n']:,} ships sunk by a single weapon.",
             fontsize=12, color=INK_SOFT)
    fig.text(0.04, 0.012, "Source: JANAC (1947). All Allied agents; ships sunk by two different "
             "weapons, losses credited to Japan (accidents, own mines) and minor causes are left "
             "out. Residual = (observed - expected) / its standard error under independence.",
             fontsize=9.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.14, right=0.98, top=1 - 2.1 / fig.get_figheight(),
                        bottom=0.75 / fig.get_figheight())
    fig.savefig(out_path, dpi=170, facecolor=BACKGROUND)
    plt.close(fig)


def conditional_on_year(ships, vessel):
    """Log-linear test: is ship type still associated with weapon WITHIN each year?

    Weapons were used at different times (mines mostly in 1945) and the mix
    of ship types afloat changed too, so a type-weapon association could be
    nothing more than both changing over time. This test holds the year
    fixed. The 3-way table of counts (type x weapon x year) is modelled with
    Poisson log-linear models:
        H0: log mu = [type x year] + [weapon x year]           (type and weapon
            independent given year: conditional independence)
        H1: log mu = [type x year] + [weapon x year] + [type x weapon]
    The likelihood-ratio statistic is the drop in deviance, compared with
    chi-square on (types - 1)(weapons - 1) df. It is the log-linear version
    of a stratified (Cochran-Mantel-Haenszel-type) test, and allows the
    type-weapon association to be any pattern, not only a trend.
    """
    part = ships[ships["vessel"] == vessel]
    counts = (part.groupby(["ship_type", "weapon", "year"]).size()
              .reindex(pd.MultiIndex.from_product(
                  [sorted(part["ship_type"].unique()), sorted(part["weapon"].unique()),
                   sorted(part["year"].unique())], names=["ship_type", "weapon", "year"]),
                  fill_value=0).rename("n").reset_index())
    counts["year"] = counts["year"].astype(str)
    h0 =smf.glm("n ~ C(ship_type) * C(year) + C(weapon) * C(year)", counts,
                 family=sm.families.Poisson()).fit()
    h1 = smf.glm("n ~ C(ship_type) * C(year) + C(weapon) * C(year) + C(ship_type):C(weapon)",
                 counts, family=sm.families.Poisson()).fit()
    lr = h0.deviance - h1.deviance
    df = int(round(h0.df_resid - h1.df_resid))
    return {"analysis": "B1", "fleet": vessel,
            "test": "Log-linear test of type x weapon given year (conditional independence)",
            "statistic": lr, "df": df, "p_value": stats.chi2.sf(lr, df),
            "p_asymptotic": stats.chi2.sf(lr, df), "effect_size": np.nan,
            "effect_size_name": "", "n": int(counts["n"].sum()),
            "note": f"residual deviance H0 = {h0.deviance:.1f} on {int(h0.df_resid)} df; "
                    f"H1 = {h1.deviance:.1f} on {int(h1.df_resid)} df"}


def analyse_contingency(ships, vessel, rng):
    """Run B1 for one fleet: test, effect size, residuals, CA; save tables and the figure."""
    table = contingency(ships, vessel)
    result = chi_square(table, rng)
    v, v_corr = cramers_v(result["x2"], result["n"], *table.shape)
    residuals = adjusted_residuals(table, result["expected"])
    threshold = stats.norm.ppf(1 - ALPHA / (2 * table.size))
    coords, share = correspondence_analysis(table)

    long = pd.DataFrame({
        "ship_type": np.repeat(table.index, table.shape[1]),
        "weapon": np.tile(table.columns, table.shape[0]),
        "observed": table.to_numpy().ravel(),
        "expected": result["expected"].ravel(),
        "adjusted_residual": residuals.to_numpy().ravel()})
    long["significant_bonferroni"] = long["adjusted_residual"].abs() > threshold
    long.to_csv(DATA_DIR / f"contingency_{vessel}.csv", index=False, float_format="%.4g")
    coords.assign(dim_share_1=share[0], dim_share_2=share[1] if len(share) > 1 else np.nan) \
        .to_csv(DATA_DIR / f"correspondence_{vessel}.csv", index=False, float_format="%.4g")
    draw_contingency_figure(table, residuals, threshold, coords, share, result, v_corr, vessel,
                            FIG_DIR / f"weapon_by_type_{vessel}.png")

    return [{"analysis": "B1", "fleet": vessel, "test": "Chi-square test of independence "
             "(ship type x weapon)", "statistic": result["x2"], "df": result["dof"],
             "p_value": result["p_permutation"], "p_asymptotic": result["p_asymptotic"],
             "effect_size": v_corr, "effect_size_name": "Cramer's V (bias-corrected)",
             "n": result["n"],
             "note": f"plain V = {v:.3f}; {result['cells_expected_below_5']} of "
                     f"{result['cells']} cells have expected count < 5; CA dimension 1 = "
                     f"{share[0]:.1%}, dimension 2 = {share[1]:.1%} of inertia; Bonferroni "
                     f"residual threshold = {threshold:.2f}"}], long


# ---------------------------------------------------------------------------
# B2: tonnage by weapon
# ---------------------------------------------------------------------------
def cliffs_delta(a, b):
    """Cliff's delta: P(a > b) - P(a < b) for a random ship from each group.

    Computed from the Mann-Whitney U statistic: delta = 2U / (n_a n_b) - 1.
    It runs from -1 to 1; 0 means neither group tends to be bigger.
    Rough guide (Romano et al., 2006): |delta| < 0.15 negligible, < 0.33
    small, < 0.47 medium, otherwise large.
    """
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def size_tests(ships, vessel):
    """Kruskal-Wallis across weapons, then Dunn's pairwise tests with Cliff's delta.

    Kruskal-Wallis is a rank test: H0 is that tonnage has the same
    distribution for every weapon. It assumes nothing about the shape of
    the tonnage distribution, which is strongly right-skewed. epsilon^2 =
    H / (n - 1) is its effect size (the share of rank variance explained by
    weapon). Dunn's test then compares each pair of weapons using the same
    pooled ranks; the p-values are Holm-corrected for the number of pairs.
    """
    part = ships[ships["vessel"] == vessel]
    groups = {w: part.loc[part["weapon"] == w, "tonnage"].to_numpy() for w in WEAPONS
              if (part["weapon"] == w).sum() >= 5}
    h, p = stats.kruskal(*groups.values())
    n = sum(len(g) for g in groups.values())
    dunn = sp.posthoc_dunn(part[part["weapon"].isin(groups)], val_col="tonnage",
                           group_col="weapon", p_adjust="holm")
    pairs = []
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append({"fleet": vessel, "weapon_a": a, "weapon_b": b,
                          "median_a": np.median(groups[a]), "median_b": np.median(groups[b]),
                          "dunn_p_holm": dunn.loc[a, b],
                          "cliffs_delta": cliffs_delta(groups[a], groups[b])})
    test = {"analysis": "B2", "fleet": vessel,
            "test": "Kruskal-Wallis, tonnage across weapons", "statistic": h,
            "df": len(groups) - 1, "p_value": p, "p_asymptotic": p,
            "effect_size": h / (n - 1), "effect_size_name": "epsilon-squared", "n": n,
            "note": "; ".join(f"{w}: n={len(g)}, median={np.median(g):,.0f}"
                              for w, g in groups.items())}
    return test, pd.DataFrame(pairs), groups


def size_regression(ships, vessel):
    """OLS of log tonnage on weapon: raw, then adjusted for ship type and year.

    log(tonnage) = b0 + weapon effects [+ ship type effects + year effects].
    exp(b) - 1 is the % difference in typical (geometric-mean) tonnage
    against submarines. The adjusted model asks: for ships of the SAME type
    sunk in the SAME year, did the weapons still sink different sizes? If the
    raw gap shrinks a lot after adjustment, it was mostly a different mix of
    ship types or periods. HC3 standard errors are robust to the spread
    differing between groups.
    """
    part = ships[(ships["vessel"] == vessel)].copy()
    part = part[part["weapon"].isin(part["weapon"].value_counts()[lambda s: s >= 5].index)]
    part["weapon"] = pd.Categorical(part["weapon"],
                                    [w for w in WEAPONS if w in set(part["weapon"])])
    rows = []
    for label, formula in [("raw", f"log_tonnage ~ C(weapon, Treatment('{REFERENCE}'))"),
                           ("adjusted for ship type and year",
                            f"log_tonnage ~ C(weapon, Treatment('{REFERENCE}')) + C(ship_type) "
                            f"+ C(year)")]:
        fit = smf.ols(formula, part).fit(cov_type="HC3")
        ci = fit.conf_int()
        for name in fit.params.index:
            if not name.startswith("C(weapon"):
                continue
            weapon = name.split("[T.")[1].rstrip("]")
            rows.append({"fleet": vessel, "model": label, "weapon": weapon,
                         "vs": REFERENCE, "pct_difference": 100 * math.expm1(fit.params[name]),
                         "ci_low_pct": 100 * math.expm1(ci.loc[name, 0]),
                         "ci_high_pct": 100 * math.expm1(ci.loc[name, 1]),
                         "p_value": fit.pvalues[name], "r_squared": fit.rsquared,
                         "n": int(fit.nobs)})
    return pd.DataFrame(rows)


def draw_size_figure(groups, pairs, regression, test, vessel, out_path):
    """Tonnage distributions by weapon (left) and the regression effects (right).

    Left: a box plot per weapon on a log tonnage axis (tonnage spans
    100 to 63,000), with every ship as a jittered dot so the raw data shows,
    and the median written above each box.
    Right: a forest plot of each weapon's % difference in typical tonnage
    against submarines, raw (open circle) and adjusted for ship type and
    year (filled), with 95% confidence intervals. A dashed line at 0 % means
    "the same size as submarine victims".
    """
    fig, (box_ax, forest_ax) = plt.subplots(1, 2, figsize=(16, 7.4),
                                            gridspec_kw={"width_ratios": [1.25, 1],
                                                         "wspace": 0.25})
    rng = np.random.default_rng(RANDOM_SEED)
    names = list(groups)
    box = box_ax.boxplot([groups[w] for w in names], positions=range(len(names)), widths=0.55,
                         patch_artist=True, showfliers=False, medianprops={"color": INK,
                                                                          "linewidth": 2})
    for patch, weapon in zip(box["boxes"], names):
        patch.set_facecolor(tint(WEAPON_COLOURS[weapon], 0.6))
        patch.set_edgecolor(WEAPON_COLOURS[weapon])
    for i, weapon in enumerate(names):
        values = groups[weapon]
        box_ax.scatter(i + rng.uniform(-0.22, 0.22, len(values)), values, s=6,
                       color=WEAPON_COLOURS[weapon], alpha=0.35, linewidths=0, zorder=3)
        box_ax.text(i, values.max() * 1.6, f"median {np.median(values):,.0f}\nn = {len(values):,}",
                    ha="center", fontsize=10.5, color=INK)
    box_ax.set_yscale("log")
    box_ax.set_xticks(range(len(names)), [WEAPON_LABELS[w] for w in names], fontsize=11.5)
    box_ax.set_ylabel(f"Tonnage of each ship sunk ({UNITS[vessel]}, log scale)")
    box_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    box_ax.grid(axis="y", color=GRID, linewidth=0.6)
    box_ax.set_ylim(top=max(g.max() for g in groups.values()) * 6)
    box_ax.set_title(f"Kruskal-Wallis H = {test['statistic']:.1f}, df = {test['df']}, "
                     f"p = {test['p_value']:.2g}; ε² = {test['effect_size']:.3f}",
                     loc="left", fontsize=11, color=INK_SOFT)

    weapons = [w for w in names if w != REFERENCE]
    for i, weapon in enumerate(weapons):
        for offset, model, filled in [(-0.12, "raw", False),
                                      (0.12, "adjusted for ship type and year", True)]:
            row = regression[(regression["weapon"] == weapon) & (regression["model"] == model)]
            if row.empty:
                continue
            row = row.iloc[0]
            y = i + offset
            forest_ax.hlines(y, row["ci_low_pct"], row["ci_high_pct"],
                             color=WEAPON_COLOURS[weapon], linewidth=2.2)
            forest_ax.scatter(row["pct_difference"], y, s=70, zorder=3,
                              color=WEAPON_COLOURS[weapon] if filled else BACKGROUND,
                              edgecolors=WEAPON_COLOURS[weapon], linewidths=2)
            forest_ax.text(row["ci_high_pct"], y, f"  {row['pct_difference']:+.0f}%",
                           va="center", fontsize=10, color=INK)
    forest_ax.axvline(0, color=INK_SOFT, linestyle="--", linewidth=1)
    forest_ax.set_yticks(range(len(weapons)), [WEAPON_LABELS[w] for w in weapons], fontsize=11.5)
    forest_ax.invert_yaxis()
    forest_ax.set_xlabel(f"Difference in typical tonnage vs {WEAPON_LABELS[REFERENCE].lower()} (%)")
    forest_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    forest_ax.grid(axis="x", color=GRID, linewidth=0.6)
    forest_ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="-", color=INK_SOFT, markerfacecolor=BACKGROUND,
               label="Raw"),
        Line2D([], [], marker="o", linestyle="-", color=INK_SOFT,
               label="Adjusted for ship type and year")],
        loc="lower right", frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=10.5)
    forest_ax.set_title("Regression of log tonnage, 95% CI (HC3)", loc="left", fontsize=11,
                        color=INK_SOFT)

    fig.suptitle(f"Did some weapons sink bigger {VESSEL_OPTIONS[vessel].split(' (')[0].lower()}?",
                 x=0.05, ha="left", y=0.985, fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.012, f"Source: JANAC (1947). Ships sunk by a single weapon, all Allied "
             f"agents. Tonnage in {UNITS[vessel]} as printed (estimates included). Typical "
             f"tonnage = geometric mean, from the regression on log tonnage.",
             fontsize=9.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.12)
    fig.savefig(out_path, dpi=170, facecolor=BACKGROUND)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def main():
    """Run B1 and B2 for both fleets and save every table and chart."""
    helpers.set_readable_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    ships, excluded = analysis_ships()
    print("Ships analysed by fleet:", ships["vessel"].value_counts().to_dict())
    print(excluded.to_string(index=False))

    tests, all_pairs, all_regressions = [], [], []
    for vessel in VESSEL_OPTIONS:
        b1, cells = analyse_contingency(ships, vessel, rng)
        tests += b1 + [conditional_on_year(ships, vessel)]
        b2, pairs, groups = size_tests(ships, vessel)
        tests.append(b2)
        regression = size_regression(ships, vessel)
        pairs.to_csv(DATA_DIR / f"dunn_{vessel}.csv", index=False, float_format="%.4g")
        all_pairs.append(pairs)
        all_regressions.append(regression)
        draw_size_figure(groups, pairs, regression, b2, vessel,
                         FIG_DIR / f"tonnage_by_weapon_{vessel}.png")
        print(f"\n{vessel}: strongest residuals")
        print(cells.reindex(cells["adjusted_residual"].abs().sort_values(ascending=False).index)
              .head(8).round(2).to_string(index=False))

    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(DATA_DIR / "tests.csv", index=False, float_format="%.4g")
    excluded.to_csv(DATA_DIR / "excluded_ships.csv", index=False)
    regressions = pd.concat(all_regressions, ignore_index=True)
    regressions.to_csv(DATA_DIR / "tonnage_regression.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_colwidth", 140):
        print(tests_df.drop(columns="note").round(4).to_string(index=False))
        for note in tests_df["note"]:
            print("  ", note)
        print(pd.concat(all_pairs).round(3).to_string(index=False))
        print(regressions.round(3).to_string(index=False))
    print(f"Saved tables to {DATA_DIR} and charts to {FIG_DIR}")


if __name__ == "__main__":
    main()
