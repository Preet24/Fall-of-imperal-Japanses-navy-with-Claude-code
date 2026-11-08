"""
Table I of the JANAC report: Japanese ships sunk in WWII, by who sank them.

This script draws four PNG figures from the hand-verified transcription of
Table I (tests/fixtures/table_I.csv):

    figures/table_I/table_I_bar_ships.png     bar charts, number of ships
    figures/table_I/table_I_bar_tonnage.png   bar charts, tonnage
    figures/table_I/table_I_pie_ships.png     pie charts, number of ships
    figures/table_I/table_I_pie_tonnage.png   pie charts, tonnage

Every figure has three panels side by side: Naval ships, Merchant ships and
All ships (naval + merchant). Each bar / slice is one "sinking agent"
(United States, Allied, ...), and its label shows both the raw number and the
percentage of that panel's total.

A note on tonnage: the report measures warships in *standard displacement*
(weight, long tons) and merchant ships in *gross register tons* (internal
volume). They are different units. The report itself adds them together in
its "Total" column, so the "All ships" tonnage panel is kept for completeness,
but it is labelled as a mixed-unit figure and should be read as approximate.

Run:
    python table_1_plots.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Paths are built relative to this file, so the script works no matter which
# folder it is launched from (a hard-coded absolute path breaks if the project
# folder is ever moved or renamed).
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "tests" / "fixtures" / "table_I.csv"
OUT_DIR = PROJECT_DIR / "figures" / "table_I"

# ---------------------------------------------------------------------------
# Labels and colours
# ---------------------------------------------------------------------------
# The CSV uses the report's wording. Some of it is long or ambiguous on a chart
# ("Marine Casualties" sounds like it means US Marines, but in the report it
# means accidents: groundings, collisions, storms, explosions). These shorter
# labels say what each category actually is. The dictionary order is also the
# order in which categories appear on every chart, so the reader always finds
# a category in the same place.
SHORT_LABELS = {
    "United States Forces": "United States",
    "Allied Forces": "Other Allies",
    "United States and Allied Forces": "US + Allies (shared)",
    "Marine Casualties": "Accidents (marine casualty)",
    "Japanese Mines": "Japan's own mines",
    "Unknown Forces": "Unknown",
}

# One fixed colour per category, used in all four figures, so "orange" means
# "Other Allies" everywhere. The colours are the first six slots of a
# categorical palette, in that palette's fixed order. That order was checked
# with a colour-blindness validator: neighbouring colours stay distinguishable
# for protan, deutan and tritan vision (worst neighbouring difference ΔE 9.1,
# target >= 8). Three of the colours (aqua, yellow, pink) are light against a
# white background, so every bar and slice also carries a text label; colour is
# never the only way to identify a category.
COLOURS = {
    "United States": "#2a78d6",                # blue
    "Other Allies": "#eb6834",                 # orange
    "US + Allies (shared)": "#1baf7a",         # aqua
    "Accidents (marine casualty)": "#eda100",  # yellow
    "Japan's own mines": "#e87ba4",            # pink
    "Unknown": "#008300",                      # green
}

# Text and chart-frame colours. Text is always dark grey/black, never the
# category colour: coloured text is hard to read, especially in yellow.
INK = "#0b0b0b"         # titles and values
INK_SOFT = "#52514e"    # subtitles, axis labels, notes
GRID = "#e1e0d9"        # faint gridlines, so they don't compete with the data
BACKGROUND = "#fcfcfb"  # near-white figure background

# The three panels drawn in every figure: (panel title, CSV column prefix).
# "total" is the report's own naval + merchant column.
PANELS = [
    ("Naval ships", "naval"),
    ("Merchant ships", "merchant"),
    ("All ships (naval + merchant)", "total"),
]

# Unit shown under each panel title for the tonnage figures.
TONNAGE_UNITS = {
    "naval": "standard displacement tons",
    "merchant": "gross register tons",
    "total": "mixed units: standard + gross tons",
}


def set_readable_style():
    """Global matplotlib settings so every label is easy to read in a PNG.

    Matplotlib's defaults (10 pt text) become small once a 3-panel figure is
    shrunk into a report or slide, so the base size is raised to 12 pt and the
    titles to 14-18 pt. The top and right frame lines are removed because they
    carry no information and only add clutter.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "text.color": INK,
        "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT,
        "ytick.color": INK,
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
    })


def load_table(path):
    """Read Table I and keep only the six top-level categories.

    The CSV is hierarchical:
      level 0 = a main category (e.g. "Allied Forces")
      level 1 = a breakdown of the level-0 row above it (e.g. "British")
      plus a final "TOTAL" row.
    Using level 0 and level 1 together would count the same ships twice, and
    TOTAL is the sum of everything, so only level-0 rows (minus TOTAL) are
    plotted. TOTAL is still returned so the percentages can be checked against
    the report's own grand total.
    """
    table = pd.read_csv(path)

    total_row = table[table["sinking_agent"] == "TOTAL"].iloc[0]
    top = table[(table["level"] == 0) & (table["sinking_agent"] != "TOTAL")].copy()

    # Replace the report's wording with the short chart labels, and put the
    # rows in the fixed category order defined in SHORT_LABELS.
    top["label"] = top["sinking_agent"].map(SHORT_LABELS)
    top = top.set_index("label").loc[list(SHORT_LABELS.values())]

    # Sanity check: the six categories must add up to the printed TOTAL row.
    # If they don't, the CSV has been edited wrongly and every percentage
    # would be off, so stop with a clear message instead of drawing charts.
    for prefix in ("naval", "merchant", "total"):
        for measure in ("no", "tonnage"):
            col = f"{prefix}_{measure}"
            if top[col].sum() != total_row[col]:
                raise ValueError(
                    f"{col}: categories sum to {top[col].sum():,} "
                    f"but the TOTAL row says {total_row[col]:,}"
                )
    return top


def format_count(value):
    """Thousands separators for readability: 1822210 -> '1,822,210'."""
    return f"{value:,.0f}"


def percent_text(share):
    """Percentage text with enough precision for the smallest categories.

    Several categories are under 1% (e.g. 7 of 686 naval ships), and rounding
    them to "0%" or "1%" would hide the difference between them, so values
    below 1% get two decimals and the rest get one.
    """
    if share == 0:
        return "0%"
    return f"{share:.2f}%" if share < 1 else f"{share:.1f}%"


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------
def plot_bars(table, measure, title, out_path):
    """One figure with three horizontal bar charts: naval, merchant, all.

    Why horizontal bars: the category names are long ("Accidents (marine
    casualty)"). On a horizontal bar chart they sit on the y-axis and read
    left-to-right; on a vertical chart they would have to be rotated or
    squeezed.

    Why each panel has its own x-axis: naval and merchant totals differ a lot
    (686 vs 2,346 ships), and the tonnage panels use different units. A shared
    axis would make the naval bars look tiny and would wrongly suggest the
    units can be compared. The panels are side by side, not overlaid on one
    chart with two axes, which is easy to misread.

    Why the bars are labelled with "value (percent)": the United States is
    about 90% of every panel, so the other bars are very short. The labels
    give their exact values, which the bar length alone can't show.
    """
    column = "no" if measure == "count" else "tonnage"
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)

    for ax, (panel_title, prefix) in zip(axes, PANELS):
        values = table[f"{prefix}_{column}"]
        shares = values / values.sum() * 100

        # Category order is reversed on the y-axis so that the first category
        # (United States) is at the top: matplotlib draws y position 0 at the
        # bottom.
        y = range(len(values))[::-1]
        colours = [COLOURS[label] for label in values.index]
        ax.barh(list(y), values.values, height=0.62, color=colours,
                edgecolor=BACKGROUND, linewidth=2)

        # Label every bar just past its end. Text stays dark (INK), not the
        # bar colour, so it is readable even beside the light yellow bars.
        for yi, value, share in zip(y, values.values, shares.values):
            ax.text(value, yi, f"  {format_count(value)}  ({percent_text(share)})",
                    va="center", ha="left", fontsize=11.5, color=INK)

        # Leave 45% spare width to the right of the longest bar so its label
        # fits inside the panel instead of running off the edge.
        ax.set_xlim(0, values.max() * 1.45)

        # Panel title, with the unit of measurement underneath.
        unit = "ships" if measure == "count" else TONNAGE_UNITS[prefix]
        ax.set_title(f"{panel_title}\n", loc="left")
        ax.text(0, 1.02, f"Total: {format_count(values.sum())} {unit}",
                transform=ax.transAxes, fontsize=11.5, color=INK_SOFT)

        # Faint vertical gridlines help read the bar lengths; no horizontal
        # gridlines, as each bar already has its own label.
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

        # Large tonnage values: show the axis in thousands or millions so the
        # tick labels don't overlap.
        ax.xaxis.set_major_formatter(FuncFormatter(axis_formatter))
        ax.set_xlabel("Number of ships" if measure == "count" else "Tons")

    # Category names on the y-axis of the first panel only; the others share
    # that axis.
    axes[0].set_yticks(range(len(table))[::-1])
    axes[0].set_yticklabels(table.index)

    fig.suptitle(title, x=0.01, ha="left")
    fig.text(0.01, 0.01,
             "Source: JANAC (1947), Japanese Naval and Merchant Shipping Losses "
             "During World War II by All Causes, Table I. "
             "Percentages are of each panel's total.",
             fontsize=10, color=INK_SOFT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def axis_formatter(value, _position):
    """Axis tick labels: 2,000,000 -> '2M', 250,000 -> '250k', 500 -> '500'."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Pie charts
# ---------------------------------------------------------------------------
# Slices smaller than this get no label on the pie itself; a label crammed
# into a sliver 2 degrees wide would overlap its neighbours. Their exact values
# are always shown in the legend under the pie.
MIN_SHARE_FOR_SLICE_LABEL = 5.0


def plot_pies(table, measure, title, out_path):
    """One figure with three pie charts: naval, merchant, all.

    A pie answers "what share of the whole?", which suits Table I because the
    categories add up exactly to the report's total. It is weak at comparing
    small slices, though: here five categories share only about 10% of the
    pie. So each pie gets its own legend underneath that lists every category
    with its number and percentage. The legend, not the slice size, is how the
    reader gets the small values, and no category is identified by colour
    alone.
    """
    column = "no" if measure == "count" else "tonnage"
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    for ax, (panel_title, prefix) in zip(axes, PANELS):
        values = table[f"{prefix}_{column}"]
        shares = values / values.sum() * 100

        # Categories with zero ships (e.g. no warships were lost to Japan's own
        # mines) are left out of the pie, since a 0-degree slice can't be
        # drawn. They are still listed in the legend as "0 (0%)" so the reader
        # sees that the category exists and was zero.
        nonzero = values[values > 0]

        # The pie starts at 12 o'clock and runs clockwise, the order people
        # read a clock face, so the slices follow the category order used
        # everywhere else. The white edge (2 px) separates neighbouring slices
        # so the thin slivers stay visible.
        wedges, _ = ax.pie(
            nonzero.values,
            colors=[COLOURS[label] for label in nonzero.index],
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": BACKGROUND, "linewidth": 2},
        )

        # Label only the slices that are big enough to hold a label; the
        # percentage is written in white inside the slice.
        for wedge, label in zip(wedges, nonzero.index):
            share = shares[label]
            if share < MIN_SHARE_FOR_SLICE_LABEL:
                continue
            # Place the text at the middle angle of the slice, 60% of the way
            # from the centre to the edge (the widest part of the slice that
            # is still clear of the rim).
            angle = np.radians((wedge.theta1 + wedge.theta2) / 2)
            x, y = 0.6 * np.cos(angle), 0.6 * np.sin(angle)
            ax.text(x, y, percent_text(share), ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white")

        unit = "ships" if measure == "count" else TONNAGE_UNITS[prefix]
        ax.set_title(f"{panel_title}\nTotal: {format_count(values.sum())} {unit}",
                     fontsize=14)
        ax.set_aspect("equal")  # keep the pie round, not stretched into an oval

        # The legend lists every category in the fixed order, including those
        # with no slice, using a coloured square for the ones that are drawn.
        handles, labels = [], []
        for label in table.index:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOURS[label]))
            labels.append(f"{label}:  {format_count(values[label])}  "
                          f"({percent_text(shares[label])})")
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03),
                  frameon=False, fontsize=11.5, handlelength=1.2,
                  handleheight=1.2)

    fig.suptitle(title, x=0.01, ha="left")
    fig.text(0.01, 0.01,
             "Source: JANAC (1947), Table I. Slices under "
             f"{MIN_SHARE_FOR_SLICE_LABEL:g}% are labelled only in the legend.",
             fontsize=10, color=INK_SOFT)
    # tight_layout() ignores legends placed outside the axes, so here the
    # margins are set by hand instead: the bottom 30% of the figure is kept
    # free for the three legends, which would otherwise be cut off.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.30, wspace=0.25)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    set_readable_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = load_table(DATA_FILE)
    print(table[["naval_no", "naval_tonnage", "merchant_no",
                 "merchant_tonnage", "total_no", "total_tonnage"]])

    plot_bars(table, "count",
              "Japanese ships sunk in WWII, by who sank them: number of ships",
              OUT_DIR / "table_I_bar_ships.png")
    plot_bars(table, "tonnage",
              "Japanese ships sunk in WWII, by who sank them: tonnage",
              OUT_DIR / "table_I_bar_tonnage.png")
    plot_pies(table, "count",
              "Share of Japanese ships sunk in WWII, by who sank them: number of ships",
              OUT_DIR / "table_I_pie_ships.png")
    plot_pies(table, "tonnage",
              "Share of Japanese ships sunk in WWII, by who sank them: tonnage",
              OUT_DIR / "table_I_pie_tonnage.png")

    print(f"Saved 4 figures to {OUT_DIR}")
