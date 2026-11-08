"""
Sinking events and how concentrated the losses were. (Analyses C1 and C2)

C1 - events. Ships sunk close together in time and space are grouped into
     one "event": a convoy massacre, a carrier raid on a harbour, a battle.
     Two ships are linked when they were sunk within DAY_GAP days and
     KM_GAP km of each other; an event is a chain of linked ships (single
     linkage, i.e. the connected components of the link graph). The 20
     biggest events are listed and charted, and the grouping is re-run with
     looser and stricter thresholds to show how much the result depends on
     them.

C2 - concentration. Were losses spread evenly over the war, or packed into
     a few disastrous days?
       - Lorenz-type concentration curve and Gini coefficient of losses per
         day (ships; naval tons; merchant GRT), and the share of all losses
         on the worst 1%, 5% and 10% of days;
       - a permutation test of whether losses are MORE bunched than the
         monthly totals alone would produce: each ship is moved to a random
         day of its own month (keeping every month's total), many times, to
         build the distribution of the Gini under "no bunching within
         months";
       - the event-size distribution, comparing a geometric model (events
         grow ship by ship with a constant chance of one more) with a
         heavy-tailed discrete power law (zeta), by Vuong's test.

All agents are included (Allied and Japanese accidents/mines): the question
is how Japan's losses were distributed, whatever caused them.

Input:  data/naval/naval_losses_15-44.csv, data/merchant/merchant_losses_45-115.csv
Output:
    figures/events/events_top20.png         the 20 biggest events
    figures/events/events_timeline.png      every event over time
    figures/events/concentration.png        concentration curves + event sizes
    data/events/events.csv                  one row per event (all thresholds' base run)
    data/events/ship_events.csv             each ship with its event number
    data/events/event_sensitivity.csv       results for other thresholds
    data/events/concentration_tests.csv     Gini, shares, permutation and Vuong tests

The method is explained in analysis_A1_B_C_explained.md.

Run:
    python sinking_events.py
"""

import math
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no window is needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import optimize, sparse, special, stats
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import haversine_distances

import sinking_map as smap
import table_plot_helpers as helpers
from sinking_clusters import medoid, place_name
from sinking_map import AGENT_OPTIONS, VESSEL_COLOURS
from table_plot_helpers import BACKGROUND, GREY, GRID, INK, INK_SOFT, tint
from us_loss_time_series import WEAPON_COLOURS, WEAPON_LABELS

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
FIG_DIR = smap.PROJECT_DIR / "figures" / "events"
DATA_DIR = smap.PROJECT_DIR / "data" / "events"
EARTH_RADIUS_KM = 6371.0

# Linking thresholds for the main run. 1 day allows for attacks that ran
# past midnight and for dates recorded a day apart; 100 km is about the
# reach of one carrier strike on a harbour, or of a convoy and its pursuer
# over a day. Other values are tried in SENSITIVITY.
DAY_GAP = 1
KM_GAP = 100
SENSITIVITY = [(0, 50), (1, 50), (1, 100), (2, 200), (3, 300)]

TOP_EVENTS = 20
PERMUTATIONS = 999
RANDOM_SEED = 12345

# The war's first and last days with a sinking (8 Dec 1941 in Asian time,
# 15 Aug 1945). Every day in between counts, including days with no loss.
WAR_FIRST_DAY = pd.Timestamp("1941-12-08")
WAR_LAST_DAY = pd.Timestamp("1945-08-15")

AGENT_NAMES = {"us": "US", "britain": "British", "netherlands": "Dutch",
               "australia": "Australian", "new_zealand": "NZ", "india": "Indian",
               "china": "Chinese", "russia": "Soviet", "japan": "Japanese", "unknown": "unknown"}
EXTRA_WEAPON_LABELS = {"shore_battery": "Shore battery", "sabotage": "Sabotage",
                       "accident": "Accident", "unknown": "Unknown cause"}


# ---------------------------------------------------------------------------
# C1: finding events
# ---------------------------------------------------------------------------
def load():
    """Load all ships with their exact date (NaT when the report gives no day)."""
    ships = smap.load_ships()
    ships["exact_date"] = pd.to_datetime(ships["date"], errors="coerce")
    return ships


def find_events(ships, day_gap, km_gap):
    """Number every ship's event: linked ships share a number.

    Ships with both a position and an exact date can be linked; two are
    linked when |date difference| <= day_gap AND distance <= km_gap. The
    events are the connected components of that link graph (single
    linkage), so a convoy attacked over three days by a moving submarine
    pack forms one event as long as each sinking is close to the next.
    Single linkage can chain far apart ships together through
    intermediate ones; the sensitivity run shows how much that matters.

    Ships without a position or exact day (44 in all) cannot be linked and
    each forms its own one-ship event.
    """
    linkable = ships["exact_date"].notna() & ships["lat"].notna()
    idx = np.flatnonzero(linkable.to_numpy())
    points = np.radians(ships.loc[linkable, ["lat", "lon"]].to_numpy())
    km = haversine_distances(points) * EARTH_RADIUS_KM
    days = ships.loc[linkable, "exact_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    linked = (np.abs(days[:, None] - days[None, :]) <= day_gap) & (km <= km_gap)
    _, labels = connected_components(sparse.csr_matrix(linked), directed=False)

    event = np.full(len(ships), -1)
    event[idx] = labels
    next_id = labels.max() + 1
    for i in np.flatnonzero(~linkable.to_numpy()):
        event[i] = next_id
        next_id += 1
    return event


def credit_shares(credit_lists, position):
    """Share of all credits (weighted 1/n per ship) held by each agent (position 0) or weapon (1)."""
    totals = Counter()
    for credits in credit_lists:
        for credit in credits:
            totals[credit[position]] += 1 / len(credits)
    whole = sum(totals.values())
    return {key: value / whole for key, value in totals.most_common()}


def summarise_events(ships):
    """One row per event: dates, ships, tonnage by fleet, place, main agent and weapon, names.

    Events are ranked by the number of ships (the unit that is the same for
    warships and merchant ships); ties go to the earlier event. Tonnage is
    reported separately for warships (tons) and merchant ships (GRT),
    never added.
    """
    rows = []
    for event, group in ships.groupby("event"):
        dated = group["exact_date"].dropna()
        start = dated.min() if len(dated) else pd.Timestamp(year=int(group["year"].iloc[0]),
                                                             month=int(group["month"].iloc[0]),
                                                             day=15)
        # The event's position is its medoid: the real sinking site with the
        # smallest total distance to the others. The average position was
        # tried first, but in a chained event spread along a coast it can
        # fall between places; it named the Kure raid of July 1945 after the
        # Shimonoseki Strait.
        positioned = group.dropna(subset=["lat", "lon"])
        centre = medoid(positioned) if len(positioned) else None
        lat = centre["lat"] if centre is not None else np.nan
        lon = centre["lon"] if centre is not None else np.nan
        agents = credit_shares(group["credits"], 0)
        weapons = credit_shares(group["credits"], 1)
        top_agent, top_weapon = next(iter(agents)), next(iter(weapons))
        names = group.sort_values("tonnage", ascending=False)["vessel_name"].tolist()
        rows.append({
            "event": event, "start": start.date(), "end": (dated.max() if len(dated) else start).date(),
            "ships": len(group),
            "naval_ships": int((group["vessel"] == "naval").sum()),
            "merchant_ships": int((group["vessel"] == "merchant").sum()),
            "naval_tons": int(group.loc[group["vessel"] == "naval", "tonnage"].sum()),
            "merchant_grt": int(group.loc[group["vessel"] == "merchant", "tonnage"].sum()),
            "lat": lat, "lon": lon if pd.isna(lon) or lon <= 180 else lon - 360,
            "place": place_name(lat, lon) if len(positioned) else "position not given",
            "main_agent": top_agent, "main_agent_share": agents[top_agent],
            "main_weapon": top_weapon, "main_weapon_share": weapons[top_weapon],
            "names": ", ".join(names[:6]) + (f" and {len(names) - 6} more" if len(names) > 6 else "")})
    events = pd.DataFrame(rows).sort_values(["ships", "start"], ascending=[False, True])
    events.insert(0, "rank", range(1, len(events) + 1))
    return events


def sensitivity(ships):
    """Re-run the grouping for each (days, km) in SENSITIVITY and summarise the result.

    Reported for each setting: number of events, number of multi-ship
    events, share of ships in multi-ship events, the largest event and the
    median size of the top 20. If the main conclusions (most ships sink in
    groups; the biggest events are the same raids) hold across settings,
    they don't depend on the exact thresholds.
    """
    rows = []
    for day_gap, km_gap in SENSITIVITY:
        event = find_events(ships, day_gap, km_gap)
        sizes = pd.Series(event).value_counts()
        biggest = sizes.index[0]
        members = ships[event == biggest]
        centre = medoid(members.dropna(subset=["lat", "lon"]))
        rows.append({"day_gap": day_gap, "km_gap": km_gap, "events": len(sizes),
                     "multi_ship_events": int((sizes > 1).sum()),
                     "share_of_ships_in_multi_ship_events": sizes[sizes > 1].sum() / len(ships),
                     "largest_event_ships": int(sizes.iloc[0]),
                     "largest_event_start": members["exact_date"].min().date(),
                     "largest_event_place": place_name(centre["lat"], centre["lon"]),
                     "median_top20_size": float(sizes.iloc[:TOP_EVENTS].median())})
    return pd.DataFrame(rows)


def event_label(row, with_names=False):
    """Short text for an event: date, place, who and how, ships and tonnage."""
    when = pd.Timestamp(row["start"]).strftime("%d %b %Y")
    if row["end"] != row["start"]:
        when += "–" + pd.Timestamp(row["end"]).strftime("%d %b")
    weapon = WEAPON_LABELS.get(row["main_weapon"], EXTRA_WEAPON_LABELS.get(row["main_weapon"],
                                                                          row["main_weapon"]))
    tonnage = []
    if row["naval_ships"]:
        tonnage.append(f"{row['naval_tons']:,} t naval")
    if row["merchant_ships"]:
        tonnage.append(f"{row['merchant_grt']:,} GRT merchant")
    text = (f"{when} · {row['place']} · {AGENT_NAMES[row['main_agent']]} "
            f"{weapon.lower()} · {' + '.join(tonnage)}")
    return text + (f"\n{row['names']}" if with_names else "")


def draw_top_events(events, out_path):
    """Horizontal bars of the 20 biggest events: warships (blue) + merchant ships (orange).

    Bars are ranked by number of ships, the biggest at the top. Each bar is
    labelled with the date, the nearest named place, the main agent and
    weapon, and the tonnage of each fleet.
    """
    top = events.head(TOP_EVENTS).iloc[::-1]
    fig, ax = plt.subplots(figsize=(16, 11))
    y = np.arange(len(top))
    ax.barh(y, top["naval_ships"], color=VESSEL_COLOURS["naval"], height=0.68)
    ax.barh(y, top["merchant_ships"], left=top["naval_ships"], color=VESSEL_COLOURS["merchant"],
            height=0.68)
    for yi, (_, row) in zip(y, top.iterrows()):
        ax.text(row["ships"] + 0.4, yi, f"{row['ships']}  " + event_label(row), va="center",
                fontsize=10.5, color=INK)
    ax.set_yticks(y, [f"#{r}" for r in top["rank"]], fontsize=10.5)
    ax.set_xlim(0, top["ships"].max() * 2.45)   # room for the text labels right of the bars
    ax.set_xlabel("Ships sunk in the event")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.legend(handles=[Patch(color=VESSEL_COLOURS["naval"], label="Warships"),
                       Patch(color=VESSEL_COLOURS["merchant"], label="Merchant ships")],
              loc="lower right", frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=11)
    fig.suptitle(f"The {TOP_EVENTS} worst days: Japan's biggest single sinking events",
                 x=0.05, ha="left", y=0.98, fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.935, f"An event = ships sunk within {DAY_GAP} day and {KM_GAP} km of each "
             f"other (chained). Place = nearest named location; agent and weapon = largest share "
             f"of the credits.", fontsize=12, color=INK_SOFT)
    fig.text(0.05, 0.012, "Source: JANAC (1947). All agents. Naval tonnage in tons (standard "
             "displacement), merchant in GRT; not added together.", fontsize=9.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.07)
    fig.savefig(out_path, dpi=170, facecolor=BACKGROUND)
    plt.close(fig)


def draw_timeline(events, out_path):
    """Every multi-ship event as a dot: date (x), ships (y, log), coloured by main weapon.

    One-ship "events" (single sinkings) are summarised as a grey strip at
    the bottom rather than drawn individually, so the eye goes to the
    clusters of loss. Up to the 10 biggest events are labelled; a label that
    cannot be placed without overlapping another is left out (every big
    event is listed in events_top20.png anyway).
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    multi = events[events["ships"] > 1]
    single = events[events["ships"] == 1]
    ax.scatter(pd.to_datetime(single["start"]), np.full(len(single), 0.85), marker="|", s=60,
               color=GREY, alpha=0.35, linewidths=0.8)
    colours = [WEAPON_COLOURS.get(w, GREY) for w in multi["main_weapon"]]
    ax.scatter(pd.to_datetime(multi["start"]), multi["ships"], s=12 + 9 * multi["ships"],
               c=colours, alpha=0.75, edgecolors=BACKGROUND, linewidths=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.7, events["ships"].max() * 1.6)
    ax.set_xlim(pd.Timestamp("1941-11-15"), pd.Timestamp("1945-09-15"))
    # Label the 10 biggest events. Labels of 1945 events point left, so they
    # don't run off the right edge. Each label is checked against the ones
    # already drawn (in screen coordinates) and moved down a step until it
    # overlaps none, so the crowded mid-1945 labels stay readable.
    renderer = fig.canvas.get_renderer()
    placed = []
    for _, row in events.head(10).iterrows():
        when = pd.Timestamp(row["start"])
        short = f"{when.strftime('%b %Y')} {row['place']} ({row['ships']})"
        late = when >= pd.Timestamp("1945-03-01")
        # Try the preferred side first (left for 1945, right otherwise),
        # then the other side, each at up to 8 steps down.
        tries = [(side, step) for side in ([-1, 1] if late else [1, -1]) for step in range(8)]
        for side, step in tries:
            label = ax.annotate(short, (when, row["ships"]), xytext=(10 * side, 6 - 15 * step),
                                textcoords="offset points", ha="left" if side > 0 else "right",
                                fontsize=10, color=INK, zorder=5,
                                arrowprops=dict(arrowstyle="-", color=INK_SOFT, linewidth=0.6)
                                if step else None,
                                bbox=dict(boxstyle="round,pad=0.15", facecolor=BACKGROUND,
                                          edgecolor="none", alpha=0.85))
            box = label.get_window_extent(renderer)
            inside = ax.get_window_extent(renderer).contains(box.x0, box.y0) and \
                ax.get_window_extent(renderer).contains(box.x1, box.y1)
            if inside and not any(box.overlaps(other) for other in placed):
                placed.append(box)
                break
            label.remove()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Ships sunk in the event (log scale)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=9, color=WEAPON_COLOURS[w],
                      label=WEAPON_LABELS[w]) for w in ["submarine", "aircraft", "surface", "mine"]]
    handles += [Line2D([], [], marker="o", linestyle="", markersize=9, color=GREY,
                       label="Other / accident"),
                Line2D([], [], marker="|", linestyle="", markersize=12, color=GREY,
                       label=f"Single sinking ({len(single):,})")]
    ax.legend(handles=handles, loc="upper left", ncol=3, frameon=True, framealpha=0.9,
              edgecolor=GRID, fontsize=10.5, title="Main weapon of the event")
    fig.suptitle("Japanese losses came in bursts: every multi-ship sinking event",
                 x=0.05, ha="left", y=0.98, fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.012, f"Source: JANAC (1947). Event = ships sunk within {DAY_GAP} day and "
             f"{KM_GAP} km of each other. Dot area grows with the number of ships.",
             fontsize=9.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08)
    fig.savefig(out_path, dpi=170, facecolor=BACKGROUND)
    plt.close(fig)


# ---------------------------------------------------------------------------
# C2: concentration over days
# ---------------------------------------------------------------------------
def war_days(ships):
    """All calendar days of the war (index 0 = 8 Dec 1941), extended if a date falls outside."""
    dated = ships["exact_date"].dropna()
    first = min(WAR_FIRST_DAY, dated.min())
    last = max(WAR_LAST_DAY, dated.max())
    return pd.date_range(first, last, freq="D")


def gini(values):
    """Gini coefficient of non-negative values (0 = perfectly even, ~1 = all on one day).

    G = 2 sum_i i x_(i) / (n sum x) - (n + 1) / n, with x sorted ascending
    and i = 1..n. Days with no loss count as zeros, which is the point:
    a war where most days are quiet and a few are catastrophic has a high G.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    return 2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n


def concentration_curve(values):
    """Cumulative share of the total held by the worst days first: (share of days, share of losses)."""
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    return np.arange(1, len(x) + 1) / len(x), np.cumsum(x) / x.sum()


def top_share(values, fraction):
    """Share of the total that fell on the worst `fraction` of days."""
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    k = max(1, int(round(fraction * len(x))))
    return x[:k].sum() / x.sum()


def daily_totals(day_index, weights, n_days):
    """Sum weights per day index (0..n_days-1)."""
    return np.bincount(day_index, weights=weights, minlength=n_days)


def bunching_test(dated, days, rng):
    """Permutation test: are losses more concentrated on particular days than monthly totals imply?

    H0: within each month, a ship's day of sinking is uniform over that
    month's war days, independent of the other ships. The monthly totals,
    and so the whole long-term trend, are kept exactly. Under H0 each ship
    is given a random day of its own month PERMUTATIONS times, and the Gini
    is recomputed each time; p = (1 + number of null Ginis >= observed) /
    (1 + PERMUTATIONS). A small p means losses cluster on particular days
    (raids, convoy battles) beyond what the month-to-month rise and fall of
    the war explains.

    The test is run for ship counts, naval tonnage and merchant GRT. The
    null concentration curves for ship counts are kept for the chart.
    """
    n_days = len(days)
    day_index = ((dated["exact_date"] - days[0]).dt.days).to_numpy()
    month = dated["exact_date"].dt.to_period("M")
    # For each ship, the first index and the number of war days of its month.
    month_first = np.array([max(0, (p.start_time - days[0]).days) for p in month])
    month_last = np.array([min(n_days - 1, (p.end_time.normalize() - days[0]).days) for p in month])
    month_len = month_last - month_first + 1

    measures = {"ships": np.ones(len(dated)),
                "naval tons": np.where(dated["vessel"] == "naval", dated["tonnage"], 0.0),
                "merchant GRT": np.where(dated["vessel"] == "merchant", dated["tonnage"], 0.0)}
    observed = {name: daily_totals(day_index, w, n_days) for name, w in measures.items()}
    null_gini = {name: [] for name in measures}
    grid = np.linspace(0, 1, 201)
    null_curves = []
    for _ in range(PERMUTATIONS):
        random_day = month_first + np.floor(rng.random(len(dated)) * month_len).astype(int)
        for name, w in measures.items():
            totals = daily_totals(random_day, w, n_days)
            null_gini[name].append(gini(totals))
            if name == "ships":
                share_days, share_loss = concentration_curve(totals)
                null_curves.append(np.interp(grid, share_days, share_loss))
    rows = []
    for name, totals in observed.items():
        g = gini(totals)
        null = np.array(null_gini[name])
        rows.append({"measure": name, "days": n_days, "total": totals.sum(),
                     "zero_loss_days_share": float((totals == 0).mean()),
                     "gini": g, "null_gini_mean": null.mean(),
                     "null_gini_2.5pct": np.percentile(null, 2.5),
                     "null_gini_97.5pct": np.percentile(null, 97.5),
                     "p_value_bunching": (1 + (null >= g).sum()) / (1 + PERMUTATIONS),
                     "top_1pct_days_share": top_share(totals, 0.01),
                     "top_5pct_days_share": top_share(totals, 0.05),
                     "top_10pct_days_share": top_share(totals, 0.10)})
    return pd.DataFrame(rows), observed, grid, np.array(null_curves)


def event_size_models(sizes):
    """Compare a geometric and a discrete power-law (zeta) model of event sizes (Vuong's test).

    Both have support {1, 2, 3, ...} and one parameter, fitted by maximum
    likelihood:
      geometric: P(X = x) = p (1 - p)^(x - 1), MLE p = 1 / mean. It is what
          you get if, once an event has started, each further ship is lost
          with the same chance: big events are exponentially rare.
      zeta (discrete power law): P(X = x) = x^(-a) / zeta(a), a > 1. Big
          events are only polynomially rare: a heavy tail.
    Vuong's test compares non-nested models: with d_i the difference of the
    two log-likelihoods for observation i, V = sqrt(n) mean(d) / sd(d) is
    approximately N(0, 1) when both fit equally well. V > 0 favours the
    power law, V < 0 the geometric. Both models have one parameter, so no
    correction term is needed.
    """
    x = np.asarray(sizes, dtype=float)
    n = len(x)
    p = 1 / x.mean()
    ll_geo = np.log(p) + (x - 1) * np.log(1 - p)

    def neg_ll(a):
        return a * np.log(x).sum() + n * np.log(special.zeta(a, 1))
    a_hat = optimize.minimize_scalar(neg_ll, bounds=(1.01, 8), method="bounded").x
    ll_zeta = -a_hat * np.log(x) - np.log(special.zeta(a_hat, 1))
    d = ll_zeta - ll_geo
    v = math.sqrt(n) * d.mean() / d.std(ddof=1)
    return {"events": n, "geometric_p": p, "zeta_exponent": a_hat,
            "loglik_geometric": ll_geo.sum(), "loglik_zeta": ll_zeta.sum(),
            "vuong_statistic": v, "vuong_p_two_sided": 2 * stats.norm.sf(abs(v)),
            "favoured": "zeta (heavy tail)" if v > 0 else "geometric"}, p, a_hat


def draw_concentration(observed, grid, null_curves, tests, sizes, p_geo, a_zeta, out_path):
    """Two panels: concentration curves of daily losses, and the event-size distribution.

    Left: for each measure, the share of the war's losses (y) held by the
    worst share of days (x), worst days first. A perfectly even war would be
    the diagonal. The grey band is the middle 95% of the null curves for
    ship counts (each ship moved to a random day of its month): the gap
    between the black curve and the band is bunching on particular days.
    Right: the share of events with at least x ships (log-log), with the
    fitted geometric (dashed) and power-law (dotted) curves.
    """
    fig, (curve_ax, size_ax) = plt.subplots(1, 2, figsize=(16, 7.6),
                                            gridspec_kw={"wspace": 0.22})
    low, high = np.percentile(null_curves, [2.5, 97.5], axis=0)
    curve_ax.fill_between(grid * 100, low * 100, high * 100, color=GRID, alpha=0.9,
                          label="Ships if each loss fell on a random day of its month (95%)")
    styles = {"ships": (INK, 2.6, "Ships"),
              "naval tons": (VESSEL_COLOURS["naval"], 1.8, "Naval tonnage"),
              "merchant GRT": (VESSEL_COLOURS["merchant"], 1.8, "Merchant tonnage (GRT)")}
    for name, totals in observed.items():
        colour, width, label = styles[name]
        row = tests[tests["measure"] == name].iloc[0]
        share_days, share_loss = concentration_curve(totals)
        curve_ax.plot(share_days * 100, share_loss * 100, color=colour, linewidth=width,
                      label=f"{label}: Gini {row['gini']:.2f}, worst 5% of days = "
                            f"{row['top_5pct_days_share']:.0%}")
    curve_ax.plot([0, 100], [0, 100], color=INK_SOFT, linestyle="--", linewidth=1,
                  label="Perfectly even losses")
    curve_ax.axvline(5, color=INK_SOFT, linestyle=":", linewidth=1)
    curve_ax.set_xlim(0, 100)
    curve_ax.set_ylim(0, 101)
    curve_ax.set_xlabel("Share of war days, worst first (%)")
    curve_ax.set_ylabel("Share of all losses (%)")
    curve_ax.grid(color=GRID, linewidth=0.6)
    curve_ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor=GRID, fontsize=10)
    curve_ax.set_title("Losses were packed into a few days", loc="left", fontsize=13, color=INK)

    x = np.sort(np.asarray(sizes))
    values = np.arange(1, x.max() + 1)
    ccdf = np.array([(x >= v).mean() for v in values])
    size_ax.scatter(values[ccdf > 0], ccdf[ccdf > 0], s=28, color=INK, zorder=3,
                    label="Observed share of events with ≥ x ships")
    geo_ccdf = (1 - p_geo) ** (values - 1)
    zeta_ccdf = special.zeta(a_zeta, values) / special.zeta(a_zeta, 1)
    size_ax.plot(values, geo_ccdf, color=INK_SOFT, linestyle="--", linewidth=1.8,
                 label=f"Geometric fit (p = {p_geo:.2f})")
    size_ax.plot(values, zeta_ccdf, color=INK_SOFT, linestyle=":", linewidth=2.2,
                 label=f"Power-law fit (exponent {a_zeta:.2f})")
    size_ax.set_xscale("log")
    size_ax.set_yscale("log")
    size_ax.set_ylim(ccdf[ccdf > 0].min() / 3, 1.5)
    size_ax.set_xlabel("Ships in the event, x (log scale)")
    size_ax.set_ylabel("Share of events with at least x ships (log scale)")
    size_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    size_ax.grid(color=GRID, linewidth=0.6)
    size_ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor=GRID, fontsize=10)
    size_ax.set_title("A heavy tail of very large events?", loc="left", fontsize=13, color=INK)

    fig.suptitle("How concentrated were Japan's losses?", x=0.05, ha="left", y=0.98,
                 fontsize=18, fontweight="bold", color=INK)
    fig.text(0.05, 0.012, "Source: JANAC (1947). All agents; ships with an exact date "
             "(left) and events of ships with a position and date (right). Days with no loss "
             "count as zeros.", fontsize=9.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.11)
    fig.savefig(out_path, dpi=170, facecolor=BACKGROUND)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def main():
    """Run C1 and C2 and save every table and chart."""
    helpers.set_readable_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    ships = load()
    ships["event"] = find_events(ships, DAY_GAP, KM_GAP)
    events = summarise_events(ships)
    events.to_csv(DATA_DIR / "events.csv", index=False, float_format="%.4g")
    ships[["page", "vessel_name", "vessel", "date", "tonnage", "lat", "lon", "event"]] \
        .merge(events[["event", "rank"]], on="event").to_csv(DATA_DIR / "ship_events.csv",
                                                             index=False)
    sens = sensitivity(ships)
    sens.to_csv(DATA_DIR / "event_sensitivity.csv", index=False, float_format="%.4g")
    draw_top_events(events, FIG_DIR / "events_top20.png")
    draw_timeline(events, FIG_DIR / "events_timeline.png")

    dated = ships[ships["exact_date"].notna()]
    days = war_days(ships)
    tests, observed, grid, null_curves = bunching_test(dated, days, rng)
    linkable = ships["exact_date"].notna() & ships["lat"].notna()
    sizes = ships.loc[linkable, "event"].value_counts().to_numpy()
    model, p_geo, a_zeta = event_size_models(sizes)
    size_summary = {"share_events_single": float((sizes == 1).mean()),
                    "share_ships_in_events_2plus": float(sizes[sizes >= 2].sum() / sizes.sum()),
                    "share_ships_in_events_5plus": float(sizes[sizes >= 5].sum() / sizes.sum()),
                    "share_ships_in_events_10plus": float(sizes[sizes >= 10].sum() / sizes.sum()),
                    "largest_event": int(sizes.max())}
    out = tests.copy()
    out.to_csv(DATA_DIR / "concentration_tests.csv", index=False, float_format="%.4g")
    pd.DataFrame([{**model, **size_summary}]).to_csv(DATA_DIR / "event_size_models.csv",
                                                    index=False, float_format="%.4g")
    draw_concentration(observed, grid, null_curves, tests, sizes, p_geo, a_zeta,
                       FIG_DIR / "concentration.png")

    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_colwidth", 70):
        print(events.head(TOP_EVENTS)[["rank", "start", "end", "ships", "naval_ships",
                                       "merchant_ships", "naval_tons", "merchant_grt", "place",
                                       "main_agent", "main_weapon"]].to_string(index=False))
        print(sens.round(3).to_string(index=False))
        print(tests.round(4).to_string(index=False))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in model.items()})
        print({k: round(v, 4) for k, v in size_summary.items()})
    print(f"Ships: {len(ships):,}; linkable {int(linkable.sum()):,}; with exact date {len(dated):,}")
    print(f"Saved tables to {DATA_DIR} and charts to {FIG_DIR}")


if __name__ == "__main__":
    main()
