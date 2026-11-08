"""
Table II of the JANAC report: Japanese ships sunk by US forces, by weapon.

Table I showed that the United States sank about 90% of all Japanese ships
lost in WWII. Table II takes just those US sinkings (611 warships and 2,117
merchant ships) and splits them by the *type of weapon* that sank them:
submarines, surface warships, Army aircraft, Navy/Marine aircraft, mines,
and combinations of these.

This script draws four PNG figures from the hand-verified transcription of
Table II (tests/fixtures/table_II.csv):

    figures/table_II/table_II_bar_ships.png     bar charts, number of ships
    figures/table_II/table_II_bar_tonnage.png   bar charts, tonnage
    figures/table_II/table_II_pie_ships.png     pie charts, number of ships
    figures/table_II/table_II_pie_tonnage.png   pie charts, tonnage
    figures/table_II/table_II_pie_broad_ships.png    pie charts, 6 broad weapon
    figures/table_II/table_II_pie_broad_tonnage.png  groups (see BROAD_GROUPS)
    figures/table_II/table_II_detail_bar_ships.png    detailed categories
    figures/table_II/table_II_detail_bar_tonnage.png  (e.g. carrier-based
    figures/table_II/table_II_detail_pie_ships.png    aircraft, air-dropped
    figures/table_II/table_II_detail_pie_tonnage.png  mines) inside the broad
                                                      groups; see
                                                      DETAIL_CATEGORIES

The first four figures use the report's own categories (8 after merging the
two smallest). The last two answer a simpler question, "which *kind* of
weapon did the damage?", by combining them into 6 broad groups: aircraft,
surface warships, submarines, mines, combinations and other.

Every figure has three panels side by side: Naval ships, Merchant ships and
All ships (naval + merchant). Each bar / slice is one weapon type, and its
label shows both the raw number and the percentage of that panel's total.

A note on tonnage: the report measures warships in *standard displacement*
(weight, long tons) and merchant ships in *gross register tons* (internal
volume). They are different units. The report itself adds them together in
its "Total" column, so the "All ships" tonnage panel is kept for completeness,
but it is labelled as a mixed-unit figure and should be read as approximate.

The layout and styling deliberately match table_1_plots.py, so the figures
for the two tables look like one set.

Run:
    python table_2_plots.py
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
# folder it is launched from.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "tests" / "fixtures" / "table_II.csv"
OUT_DIR = PROJECT_DIR / "figures" / "table_II"

# ---------------------------------------------------------------------------
# Categories: from the report's 9 rows to 8 chart categories
# ---------------------------------------------------------------------------
# Table II has 9 top-level (level 0) rows. A chart can only use about 8
# colours before neighbouring colours become hard to tell apart, especially
# for colour-blind readers. So the two smallest rows are merged into one
# "Other" category:
#   - Navy Shore Batteries: 2 warships sunk by coastal guns
#   - Unknown Agents:       4 merchant ships, weapon not identified
# Together they are 6 of 2,728 ships (0.2%), so merging them hides nothing
# important, and the legend says exactly what "Other" contains.
#
# The dictionary maps the report's wording (left) to a short chart label
# (right). Several labels are reworded to say what the category really is:
#   - "Surface Craft" means US warships (destroyers, cruisers, battleships)
#     sinking ships with guns and torpedoes, so it becomes "Surface warships".
#   - 257 of the 266 mine sinkings were from mines dropped by Army Air Forces
#     bombers (see the level-1 rows of the CSV), so the label says
#     "mostly air-dropped".
#   - "Aircraft and Other Agents" is a mixed bag: aircraft working together
#     with ships, submarines or mines, plus 12 sinkings by unidentified
#     aircraft.
# The dictionary order is the order the categories appear in on every chart:
# the same order as the printed table, so the charts can be checked against
# the report line by line.
CATEGORY_LABELS = {
    "Submarines": "Submarines",
    "Surface Craft": "Surface warships",
    "Army Aircraft": "Army aircraft",
    "Navy-Marine Aircraft": "Navy & Marine aircraft",
    "Army, Navy, and Marine Aircraft in Combination": "Army + Navy/Marine aircraft together",
    "Mines": "Mines (mostly air-dropped)",
    "Aircraft and Other Agents": "Aircraft + ships/subs/mines together",
    "Navy Shore Batteries": "Other (shore guns, unknown)",
    "Unknown Agents": "Other (shore guns, unknown)",
}

# One fixed colour per category, used in all four figures, so "blue" always
# means "Submarines". The first seven are the first seven slots of a
# categorical palette, in its fixed order. That order was run through a
# colour-blindness validator: every pair of neighbouring colours stays
# distinguishable for red-, green- and blue-weak vision (worst neighbouring
# difference ΔE 9.1, target >= 8).
# "Other" is a neutral grey on purpose. It is a catch-all, not a real weapon
# type, and grey tells the reader "this is the leftover".
# Three colours (aqua, yellow, pink) are pale on a white background, so every
# bar and slice also has a text label: colour is never the only way to tell
# categories apart.
COLOURS = {
    "Submarines": "#2a78d6",                            # blue
    "Surface warships": "#eb6834",                      # orange
    "Army aircraft": "#1baf7a",                         # aqua
    "Navy & Marine aircraft": "#eda100",                # yellow
    "Army + Navy/Marine aircraft together": "#e87ba4",  # pink
    "Mines (mostly air-dropped)": "#008300",            # green
    "Aircraft + ships/subs/mines together": "#4a3aa7",  # violet
    "Other (shore guns, unknown)": "#898781",           # grey
}

# ---------------------------------------------------------------------------
# Broad weapon groups (for the extra "broad" pie charts)
# ---------------------------------------------------------------------------
# The report splits aircraft into three level-0 rows (Army, Navy & Marine,
# and the two working together). That is useful detail, but it hides how much
# aircraft did *in total*. These groups combine the report's level-0 rows
# into 6 kinds of weapon. Left: the report's wording; right: the broad group.
#
# Two decisions worth knowing about:
#   - "Army, Navy, and Marine Aircraft in Combination" goes to "Aircraft",
#     not "Combinations": every weapon involved was an aircraft, just from
#     different services. "Combinations" is kept for sinkings where
#     *different kinds* of weapon worked together (e.g. aircraft + submarine).
#   - "Aircraft and Other Agents" goes to "Combinations", except for one of
#     its level-1 rows, "Unknown Aircraft" (12 merchant ships). Those were
#     sunk by aircraft alone, the report just doesn't know whose, so they are
#     moved to "Aircraft" in load_broad_table() below.
BROAD_GROUPS = {
    "Submarines": "Submarines",
    "Surface Craft": "Surface warships",
    "Army Aircraft": "Aircraft",
    "Navy-Marine Aircraft": "Aircraft",
    "Army, Navy, and Marine Aircraft in Combination": "Aircraft",
    "Mines": "Mines",
    "Aircraft and Other Agents": "Combinations",
    "Navy Shore Batteries": "Other (shore guns, unknown)",
    "Unknown Agents": "Other (shore guns, unknown)",
}

# The level-1 row that is moved out of "Combinations" into "Aircraft".
UNKNOWN_AIRCRAFT_ROW = "Unknown Aircraft"

# Colours for the 6 broad groups. Groups that also exist in the detailed
# charts keep the same colour (Submarines blue, Surface warships orange,
# Mines green, Other grey), so a reader moving between the figures isn't
# confused. "Aircraft" takes aqua, the colour of "Army aircraft" in the
# detailed charts. Yellow (Navy & Marine aircraft) was tried first, but the
# validator failed it: yellow next to orange is hard to tell apart even with
# normal colour vision. "Combinations" takes violet, as its main part
# ("Aircraft + ships/subs/mines together") does in the detailed charts.
# This order passed the colour-blindness validator.
# The dictionary order is also the slice order.
BROAD_COLOURS = {
    "Submarines": "#2a78d6",                   # blue
    "Surface warships": "#eb6834",             # orange
    "Aircraft": "#1baf7a",                     # aqua
    "Mines": "#008300",                        # green
    "Combinations": "#4a3aa7",                 # violet
    "Other (shore guns, unknown)": "#898781",  # grey
}

# ---------------------------------------------------------------------------
# Detailed weapon categories inside the broad groups
# ---------------------------------------------------------------------------
# The broad pies above merge, for example, carrier-based and land-based
# aircraft into "Aircraft". The detail charts keep that detail and show it
# inside the broad groups. They use the most detailed rows of the CSV: the
# level-1 rows, plus the level-0 rows that have no breakdown (Submarines,
# Surface Craft, Army Aircraft, Navy Shore Batteries, Unknown Agents).
# Each entry is: report wording -> (broad group, detailed label).
#
# Table II has 30 such rows. Most of the 13 combination rows are 1-3 ships
# each, so they are summarised by *what the aircraft worked with* (surface
# ships, submarines, mines or sabotage), which is the interesting part.
# That gives 17 detailed categories.
#
# The broad group of each detail follows BROAD_GROUPS, including the
# "Unknown Aircraft" move, and the script checks that the details add up to
# exactly the broad groups.
#   - The four "Army, Navy, and Marine Aircraft in Combination" rows become
#     one detail, "Army + Navy/Marine aircraft together".
#   - Mines are split by who laid them. "Laid by Army Air Forces" means
#     dropped from Army bombers (257 of the 266 mine sinkings).
_OTHER = "Other (shore guns, unknown)"
DETAIL_CATEGORIES = {
    "Submarines": ("Submarines", "Submarines"),
    "Surface Craft": ("Surface warships", "Surface warships"),
    "Army Aircraft": ("Aircraft", "Army aircraft"),
    "Carrier-Based": ("Aircraft", "Navy carrier-based aircraft"),
    "Land-Based": ("Aircraft", "Navy & Marine land-based aircraft"),
    "Army and Navy Carrier-Based Aircraft": ("Aircraft", "Army + Navy/Marine aircraft together"),
    "Army and Navy-Marine Land-Based Aircraft": ("Aircraft", "Army + Navy/Marine aircraft together"),
    "Navy Carrier-Based and Navy-Marine Land-Based Aircraft": ("Aircraft", "Army + Navy/Marine aircraft together"),
    "Army, Navy Carrier-Based, and Navy-Marine Land-Based Aircraft": ("Aircraft", "Army + Navy/Marine aircraft together"),
    "Unknown Aircraft": ("Aircraft", "Unidentified aircraft"),
    "Laid by Army Air Forces": ("Mines", "Dropped by Army bombers"),
    "Laid by Navy Land-Based Aircraft": ("Mines", "Dropped by Navy aircraft"),
    "Laid by Surface Craft": ("Mines", "Laid by surface ships"),
    "Laid by Submarines": ("Mines", "Laid by submarines"),
    "Army Aircraft, Navy-Marine Land-Based Aircraft and Surface Craft": ("Combinations", "Aircraft + surface ships"),
    "Army Aircraft and Surface Craft": ("Combinations", "Aircraft + surface ships"),
    "Army Aircraft, Navy Carrier-Based Aircraft, Navy-Marine Land-Based Aircraft, and Surface Craft": ("Combinations", "Aircraft + surface ships"),
    "Navy Carrier-Based Aircraft and Surface Craft": ("Combinations", "Aircraft + surface ships"),
    "Navy-Marine Land-Based Aircraft and Surface Craft": ("Combinations", "Aircraft + surface ships"),
    "Army Aircraft and Submarines": ("Combinations", "Aircraft + submarines"),
    "Navy Carrier-Based Aircraft and Submarines": ("Combinations", "Aircraft + submarines"),
    "Navy-Marine Land-Based Aircraft and Submarines": ("Combinations", "Aircraft + submarines"),
    "Army Aircraft and Army Mines": ("Combinations", "Aircraft + mines"),
    "Army Aircraft, Army Mine, and Navy Carrier-Based Aircraft": ("Combinations", "Aircraft + mines"),
    "Navy-Marine Land-Based Aircraft and Navy Mines": ("Combinations", "Aircraft + mines"),
    "Army Aircraft and Sabotage": ("Combinations", "Aircraft + sabotage"),
    "Navy Shore Batteries": (_OTHER, "Navy shore batteries"),
    "Unknown Agents": (_OTHER, "Unknown agents"),
}

# Fills too pale for white text. Percentages written inside these pie slices
# use dark text instead, so they stay readable (white on yellow is not).
PALE_FILLS = {"#1baf7a", "#eda100", "#e87ba4", "#898781"}

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

# The six numeric columns of the CSV: (naval, merchant, total) x (count, tons).
VALUE_COLUMNS = [f"{prefix}_{measure}"
                 for prefix in ("naval", "merchant", "total")
                 for measure in ("no", "tonnage")]

SOURCE_NOTE = ("Source: JANAC (1947), Japanese Naval and Merchant Shipping Losses "
               "During World War II by All Causes, Table II (US forces only). ")


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
    """Read Table II and reduce it to the 8 chart categories.

    The CSV is hierarchical:
      level 0 = a main category (e.g. "Mines")
      level 1 = a breakdown of the level-0 row above it
                (e.g. "Laid by Army Air Forces")
      plus a final "TOTAL" row.
    Using level 0 and level 1 together would count the same ships twice, and
    TOTAL is the sum of everything, so only the level-0 rows (minus TOTAL) are
    plotted. TOTAL is kept to check the numbers against the report.
    """
    table = pd.read_csv(path)

    total_row = table[table["sinking_agent"] == "TOTAL"].iloc[0]
    top = table[(table["level"] == 0) & (table["sinking_agent"] != "TOTAL")].copy()

    # Every level-0 row must have a chart label. If the CSV ever gains a row
    # that the dictionary doesn't know about, stop here: otherwise that row
    # would silently drop out of the charts and the percentages would be wrong.
    unknown = set(top["sinking_agent"]) - set(CATEGORY_LABELS)
    if unknown:
        raise ValueError(f"No chart label for these Table II rows: {unknown}")

    # Give each row its chart label, then add up rows that share a label.
    # This is the step that merges "Navy Shore Batteries" and
    # "Unknown Agents" into "Other". sort=False keeps the rows in the order
    # they first appear, i.e. the report's order.
    top["label"] = top["sinking_agent"].map(CATEGORY_LABELS)
    grouped = top.groupby("label", sort=False)[VALUE_COLUMNS].sum()

    # Put the categories in the fixed order of the COLOURS dictionary.
    grouped = grouped.loc[list(COLOURS)]

    check_against_total(grouped, total_row)
    return grouped


def load_broad_table(path):
    """Read Table II and combine it into the 6 broad weapon groups.

    Works like load_table(), but uses BROAD_GROUPS instead of
    CATEGORY_LABELS, plus one extra step: the "Unknown Aircraft" ships are
    moved from "Combinations" to "Aircraft" (see the comment above
    BROAD_GROUPS for why).
    """
    table = pd.read_csv(path)

    total_row = table[table["sinking_agent"] == "TOTAL"].iloc[0]
    top = table[(table["level"] == 0) & (table["sinking_agent"] != "TOTAL")].copy()

    # Same guard as in load_table(): every level-0 row must belong to a group.
    unknown = set(top["sinking_agent"]) - set(BROAD_GROUPS)
    if unknown:
        raise ValueError(f"No broad group for these Table II rows: {unknown}")

    # Add up the level-0 rows within each broad group.
    top["group"] = top["sinking_agent"].map(BROAD_GROUPS)
    grouped = top.groupby("group", sort=False)[VALUE_COLUMNS].sum()

    # Move the "Unknown Aircraft" ships: subtract their numbers from
    # "Combinations" and add them to "Aircraft". Taking them from the CSV
    # (rather than typing in 12 ships / 30,866 tons) keeps the script correct
    # if the transcription is ever corrected. The grand total doesn't change,
    # since the ships only move between two groups.
    unknown_aircraft = table.loc[
        (table["level"] == 1) & (table["sinking_agent"] == UNKNOWN_AIRCRAFT_ROW),
        VALUE_COLUMNS,
    ].iloc[0]
    grouped.loc["Combinations"] -= unknown_aircraft
    grouped.loc["Aircraft"] += unknown_aircraft

    # Put the groups in the fixed order of BROAD_COLOURS (= slice order).
    grouped = grouped.loc[list(BROAD_COLOURS)]

    check_against_total(grouped, total_row)
    return grouped


def check_against_total(grouped, total_row):
    """Stop if the categories don't add up to the report's TOTAL row.

    The TOTAL row is 611 warships, 2,117 merchant ships, and their tonnages.
    If the categories add up to anything else, a row was lost, counted twice
    or mistyped, and every percentage on the charts would be wrong. So stop
    with a clear message instead of drawing misleading charts.
    """
    for col in VALUE_COLUMNS:
        if grouped[col].sum() != total_row[col]:
            raise ValueError(
                f"{col}: categories sum to {grouped[col].sum():,} "
                f"but the TOTAL row says {total_row[col]:,}"
            )


def format_count(value):
    """Thousands separators for readability: 1822210 -> '1,822,210'."""
    return f"{value:,.0f}"


def percent_text(share):
    """Percentage text with enough precision for the smallest categories.

    Several categories are under 1% (e.g. 2 of 611 warships), and rounding
    them to "0%" or "1%" would hide the difference between them, so values
    below 1% get two decimals and the rest get one. An exact zero is "0%".
    """
    if share == 0:
        return "0%"
    return f"{share:.2f}%" if share < 1 else f"{share:.1f}%"


def axis_formatter(value, _position):
    """Axis tick labels: 2,000,000 -> '2M', 250,000 -> '250k', 500 -> '500'."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Bar charts
# ---------------------------------------------------------------------------
def plot_bars(table, measure, title, out_path):
    """One figure with three horizontal bar charts: naval, merchant, all.

    Why horizontal bars: the category names are long ("Army + Navy/Marine
    aircraft together"). On a horizontal bar chart they sit on the y-axis and
    read left-to-right; on a vertical chart they would have to be rotated.

    Why each panel has its own x-axis: naval and merchant totals differ a lot
    (611 vs 2,117 ships), and the tonnage panels use different units. A shared
    axis would squash the naval bars and suggest the units can be compared.

    Why the bars are labelled with "value (percent)": some categories are very
    small (2 or 4 ships) and their bars are barely visible. The label gives
    the exact value, which bar length alone can't show.
    """
    column = "no" if measure == "count" else "tonnage"
    # 8 categories need more height than Table I's 6, so each bar keeps
    # roughly the same thickness and the labels don't crowd each other.
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5), sharey=True)

    for ax, (panel_title, prefix) in zip(axes, PANELS):
        values = table[f"{prefix}_{column}"]
        shares = values / values.sum() * 100

        # Reverse the y positions so the first category (Submarines) is drawn
        # at the top: matplotlib puts y position 0 at the bottom.
        y = range(len(values))[::-1]
        colours = [COLOURS[label] for label in values.index]
        ax.barh(list(y), values.values, height=0.62, color=colours,
                edgecolor=BACKGROUND, linewidth=2)

        # Label every bar just past its end. Text stays dark (INK), not the
        # bar colour, so it is readable even beside the pale yellow bars.
        for yi, value, share in zip(y, values.values, shares.values):
            ax.text(value, yi, f"  {format_count(value)}  ({percent_text(share)})",
                    va="center", ha="left", fontsize=11.5, color=INK)

        # Leave 50% spare width to the right of the longest bar so its label
        # fits inside the panel instead of running off the edge.
        ax.set_xlim(0, values.max() * 1.5)

        # Panel title, with the total and its unit underneath.
        unit = "ships" if measure == "count" else TONNAGE_UNITS[prefix]
        ax.set_title(f"{panel_title}\n", loc="left")
        ax.text(0, 1.02, f"Total: {format_count(values.sum())} {unit}",
                transform=ax.transAxes, fontsize=11.5, color=INK_SOFT)

        # Faint vertical gridlines help read bar lengths; no horizontal
        # gridlines, as each bar already has its own label.
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

        # Show large tonnage values in thousands / millions on the axis so the
        # tick labels don't overlap.
        ax.xaxis.set_major_formatter(FuncFormatter(axis_formatter))
        ax.set_xlabel("Number of ships" if measure == "count" else "Tons")

    # Category names go on the first panel's y-axis only; the other two panels
    # share that axis (sharey=True), so repeating the names would waste space.
    axes[0].set_yticks(range(len(table))[::-1])
    axes[0].set_yticklabels(table.index)

    fig.suptitle(title, x=0.01, ha="left")
    fig.text(0.01, 0.01,
             SOURCE_NOTE + "Percentages are of each panel's total. "
             "'Other' = Navy shore batteries (2 warships) + unknown agents (4 merchant ships).",
             fontsize=10, color=INK_SOFT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pie charts
# ---------------------------------------------------------------------------
# Slices smaller than this get no label on the pie itself, because a label
# squeezed into a thin sliver would overlap its neighbours. Their exact values
# are always shown in the legend under the pie.
MIN_SHARE_FOR_SLICE_LABEL = 5.0


def plot_pies(table, measure, title, out_path, colours=COLOURS, extra_note=""):
    """One figure with three pie charts: naval, merchant, all.

    The same function draws both pie sets. `colours` says which colour each
    category gets: COLOURS for the 8 detailed categories (the default),
    BROAD_COLOURS for the 6 broad groups. `extra_note` adds a sentence to the
    footnote, e.g. to explain how the broad groups were formed.

    A pie answers "what share of the whole?", which suits Table II because the
    categories add up exactly to the report's total for US forces. Pies are
    weak at comparing similar-sized or very small slices, though, so each pie
    has its own legend underneath that lists every category with its number
    and percentage. The legend, not the slice size, is how the reader gets
    the exact values, and no category is identified by colour alone.
    """
    column = "no" if measure == "count" else "tonnage"
    fig, axes = plt.subplots(1, 3, figsize=(21, 11))

    for ax, (panel_title, prefix) in zip(axes, PANELS):
        values = table[f"{prefix}_{column}"]
        shares = values / values.sum() * 100

        # Categories with zero (e.g. no merchant ships were sunk by shore guns)
        # are left out of the pie, since a 0-degree slice can't be drawn.
        # They are still listed in the legend as "0 (0%)".
        nonzero = values[values > 0]

        # The pie starts at 12 o'clock and runs clockwise, the way a clock is
        # read, so the slices follow the same category order as the bars and
        # the legend. A 2 px edge in the background colour separates
        # neighbouring slices so thin ones stay visible.
        wedges, _ = ax.pie(
            nonzero.values,
            colors=[colours[label] for label in nonzero.index],
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": BACKGROUND, "linewidth": 2},
        )

        # Write the percentage inside each slice that is big enough to hold it.
        for wedge, label in zip(wedges, nonzero.index):
            share = shares[label]
            if share < MIN_SHARE_FOR_SLICE_LABEL:
                continue
            # Place the text at the slice's middle angle, 65% of the way from
            # the centre to the rim: far enough out that a narrow slice is
            # wide enough there to hold the text.
            angle = np.radians((wedge.theta1 + wedge.theta2) / 2)
            x, y = 0.65 * np.cos(angle), 0.65 * np.sin(angle)
            # White text on dark slices, near-black text on pale ones.
            text_colour = INK if colours[label] in PALE_FILLS else "white"
            ax.text(x, y, percent_text(share), ha="center", va="center",
                    fontsize=15, fontweight="bold", color=text_colour)

        unit = "ships" if measure == "count" else TONNAGE_UNITS[prefix]
        ax.set_title(f"{panel_title}\nTotal: {format_count(values.sum())} {unit}",
                     fontsize=14)
        ax.set_aspect("equal")  # keep the pie round, not stretched into an oval

        # The legend lists every category in the fixed order, including those
        # with no slice in this pie, with its exact number and percentage.
        handles, labels = [], []
        for label in table.index:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=colours[label]))
            labels.append(f"{label}:  {format_count(values[label])}  "
                          f"({percent_text(shares[label])})")
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03),
                  frameon=False, fontsize=11.5, handlelength=1.2,
                  handleheight=1.2)

    fig.suptitle(title, x=0.01, ha="left")
    fig.text(0.01, 0.01,
             SOURCE_NOTE + f"Slices under {MIN_SHARE_FOR_SLICE_LABEL:g}% are "
             "labelled only in the legend." + extra_note,
             fontsize=10, color=INK_SOFT)
    # tight_layout() ignores legends placed outside the axes, so the margins
    # are set by hand: the bottom third of the figure is kept free for the
    # three 8-line legends, which would otherwise be cut off.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.33, wspace=0.25)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    set_readable_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = load_table(DATA_FILE)
    print(table)

    plot_bars(table, "count",
              "Japanese ships sunk by US forces in WWII, by weapon: number of ships",
              OUT_DIR / "table_II_bar_ships.png")
    plot_bars(table, "tonnage",
              "Japanese ships sunk by US forces in WWII, by weapon: tonnage",
              OUT_DIR / "table_II_bar_tonnage.png")
    plot_pies(table, "count",
              "Share of Japanese ships sunk by US forces in WWII, by weapon: number of ships",
              OUT_DIR / "table_II_pie_ships.png")
    plot_pies(table, "tonnage",
              "Share of Japanese ships sunk by US forces in WWII, by weapon: tonnage",
              OUT_DIR / "table_II_pie_tonnage.png")

    # Broad weapon groups: aircraft, surface warships, submarines, mines,
    # combinations and other.
    broad = load_broad_table(DATA_FILE)
    print(broad)

    # Footnote saying what each broad group contains, so the pie can be read
    # without opening this script. It is long, so it starts on a new line.
    broad_note = ("\nAircraft = Army, Navy & Marine aircraft, alone or together, incl. "
                  "unidentified aircraft.  Combinations = aircraft working with ships, "
                  "submarines or mines.  Other = Navy shore batteries + unknown agents.")
    plot_pies(broad, "count",
              "Share of Japanese ships sunk by US forces in WWII, by kind of weapon: number of ships",
              OUT_DIR / "table_II_pie_broad_ships.png",
              colours=BROAD_COLOURS, extra_note=broad_note)
    plot_pies(broad, "tonnage",
              "Share of Japanese ships sunk by US forces in WWII, by kind of weapon: tonnage",
              OUT_DIR / "table_II_pie_broad_tonnage.png",
              colours=BROAD_COLOURS, extra_note=broad_note)

    # Detailed categories inside the broad groups: a grouped bar chart and a
    # two-ring donut chart, for ships and tonnage. The detail charts are drawn
    # by the shared module table_plot_helpers.py (also used by Tables III-V),
    # so this script only supplies the categories. It is imported here,
    # where it's needed, because the rest of this script doesn't use it.
    import table_plot_helpers as helpers

    raw_table, _ = helpers.read_table(DATA_FILE)
    helpers.check_breakdowns(raw_table)   # each breakdown must add up to its main row
    details = helpers.group_details(helpers.detail_rows(raw_table), DETAIL_CATEGORIES,
                                    list(BROAD_COLOURS))
    # The details must add up to exactly the broad groups of the pies above.
    helpers.check_details_match_groups(details, broad)
    print(details)

    helpers.plot_detail_all(
        details, BROAD_COLOURS,
        "Japanese ships sunk by US forces in WWII, by detailed weapon",
        SOURCE_NOTE + "Percentages are of each panel's total.",
        OUT_DIR, "table_II")

    print(f"Saved 10 figures to {OUT_DIR}")
