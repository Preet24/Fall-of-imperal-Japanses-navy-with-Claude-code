"""
Shared code for the summary-table plotting scripts (Tables III, IV and V).

table_1_plots.py and table_2_plots.py were written first and each holds its
own complete plotting code. Tables III, IV and V need exactly the same kind
of charts, so instead of copying that code three more times, it lives here
once and the three scripts import it:

    table_3_plots.py   Table III: ships sunk by the other Allies
    table_4_plots.py   Table IV:  shared credits, accidents, Japan's own
                                  mines, unidentified forces
    table_5_plots.py   Table V:   sinkings credited only "probably"

Each table script only decides *what* to plot: which categories, which
labels, which colours, and why. This module decides *how* to plot it, with
the same layout and style as Tables I and II:

  - three panels per figure: Naval ships, Merchant ships, All ships;
  - horizontal bar charts labelled "value (percent)";
  - pie charts with a full legend underneath;
  - one figure for number of ships, one for tonnage;
  - optionally, "detail" charts that show detailed categories (e.g. Navy
    carrier-based aircraft, sabotage) inside their broad groups (e.g.
    Aircraft, Other): grouped bar charts and two-ring donut charts.

table_2_plots.py also imports this module, for its detail charts only.

This file is not run on its own; it only provides functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
# The categorical palette, as named colours. Each table script picks its
# colours from here, in an order that has been checked with a
# colour-blindness validator (the check that matters is between colours that
# sit next to each other in the chart). Naming the colours makes the table
# scripts readable ("BLUE" instead of "#2a78d6").
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
PINK = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
GREY = "#898781"   # only for "Other" / leftover categories, never a real one

# Fills too pale for white text. Percentages written inside these pie slices
# use dark text instead, so they stay readable (white on yellow is not).
PALE_FILLS = {AQUA, YELLOW, PINK, GREY}

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

# Unit shown under each panel title for the tonnage figures. Warships are
# measured in standard displacement (a weight) and merchant ships in gross
# register tons (a volume), so the "All ships" tonnage mixes two units; the
# report adds them anyway, and the label warns the reader.
TONNAGE_UNITS = {
    "naval": "standard displacement tons",
    "merchant": "gross register tons",
    "total": "mixed units: standard + gross tons",
}

# The six numeric columns of every table CSV:
# (naval, merchant, total) x (number of ships, tonnage).
VALUE_COLUMNS = [f"{prefix}_{measure}"
                 for prefix in ("naval", "merchant", "total")
                 for measure in ("no", "tonnage")]

# Pie slices smaller than this (in percent) get no label on the pie itself,
# because a label squeezed into a thin sliver would overlap its neighbours.
# Their exact values are always in the legend under the pie. Tables I and II
# use 5%; here it is 7%, because at 5-6% a bold label like "5.5%" was seen to
# spill over the edges of its slice in the Table III charts.
MIN_SHARE_FOR_SLICE_LABEL = 7.0


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


# ---------------------------------------------------------------------------
# Reading and grouping the tables
# ---------------------------------------------------------------------------
def read_table(path):
    """Read a summary-table CSV and record which main row each row belongs to.

    The CSVs are hierarchical:
      level 0 = a main row (e.g. "British Forces")
      level 1 = a breakdown of the level-0 row above it (e.g. "Submarines"
                under "British Forces")
    and most end with a "TOTAL" row.

    The level-1 rows don't say which main row they belong to; that is only
    implied by their position. So a "parent" column is added: each row gets
    the name of the nearest level-0 row above it (for a level-0 row, its own
    name). This is done by blanking the name on level-1 rows and filling the
    gaps downwards with the last level-0 name (pandas' ffill).

    The TOTAL row (if any) is returned separately, so it can't be plotted or
    summed by mistake. Table V has no TOTAL row, so None is returned for it.
    """
    table = pd.read_csv(path)

    is_total = table["sinking_agent"] == "TOTAL"
    total_row = table[is_total].iloc[0] if is_total.any() else None
    table = table[~is_total].copy()

    table["parent"] = table["sinking_agent"].where(table["level"] == 0).ffill()
    return table, total_row


def detail_rows(table):
    """Return the most detailed rows, each ship appearing exactly once.

    For a main row that has a breakdown (e.g. "British Forces" -> Submarines,
    Mines, ...), its level-1 rows are used. A main row with no breakdown
    (e.g. "Chinese Aircraft") is used as it is. Taking both a main row and its
    breakdown would count the same ships twice.

    This is what's needed to regroup a table by *weapon* across all forces:
    the weapon is only named on the detailed rows.
    """
    has_children = set(table.loc[table["level"] == 1, "parent"])
    keep = (table["level"] == 1) | ~table["sinking_agent"].isin(has_children)
    return table[keep]


def group_rows(rows, key_column, mapping, order):
    """Add up rows into chart categories.

    rows        the CSV rows to group
    key_column  which column to look up in `mapping`, e.g. "sinking_agent"
                for the level-0 name, or a combined key built by the caller
    mapping     dict: value of key_column -> chart category label
    order       list of the chart categories, in the order they are drawn

    Two guards protect the charts from silently wrong numbers:
      - every row must have an entry in `mapping`; an unknown row would
        otherwise just disappear from the charts;
      - every mapped category must be in `order`, and vice versa, so none
        is dropped or left empty by a typo.
    """
    unmapped = set(rows[key_column]) - set(mapping)
    if unmapped:
        raise ValueError(f"No chart category for these rows: {sorted(unmapped)}")
    used = set(mapping[key] for key in rows[key_column])
    if used != set(order):
        raise ValueError(f"Categories {sorted(used)} don't match the order list {order}")

    labelled = rows.assign(category=rows[key_column].map(mapping))
    grouped = labelled.groupby("category")[VALUE_COLUMNS].sum()
    return grouped.loc[order]


def check_totals(grouped, expected, known_differences=None):
    """Stop if the chart categories don't add up to the table's total.

    grouped            the table returned by group_rows()
    expected           the total to compare with: the CSV's TOTAL row, or a
                       dict of totals for tables that print no TOTAL row
    known_differences  dict: column -> (printed total - sum of rows) for
                       errors that are in the printed report itself. They
                       are allowed, but printed as a warning so they are
                       never forgotten.

    If the categories don't add up, a row was lost, counted twice or mistyped,
    and every percentage on the charts would be wrong. So the script stops
    with a clear message rather than draw misleading charts.
    """
    known_differences = known_differences or {}
    for col in expected.keys():
        if col not in VALUE_COLUMNS:
            continue
        difference = expected[col] - grouped[col].sum()
        if difference == known_differences.get(col, 0):
            if difference:
                print(f"Note: {col} printed total is {expected[col]:,}, "
                      f"rows add up to {grouped[col].sum():,} (a known error in the report).")
            continue
        raise ValueError(
            f"{col}: categories sum to {grouped[col].sum():,} "
            f"but the expected total is {expected[col]:,}"
        )


def check_breakdowns(table):
    """Stop if any breakdown doesn't add up to its main row.

    Each level-0 row that has level-1 rows under it is the sum of them, e.g.
    British Forces (28 warships) = Submarines 14 + Surface Craft 8 + ... in
    Table III. If a breakdown doesn't match, a number was mistyped in the
    transcription, and charts built from the detailed rows (the "by weapon"
    charts) would disagree with charts built from the main rows.

    This matters most for Table V, which prints no TOTAL row: there, this is
    the only check that the transcribed numbers hang together.
    """
    children = table[table["level"] == 1].groupby("parent")[VALUE_COLUMNS].sum()
    parents = table[table["level"] == 0].set_index("sinking_agent")[VALUE_COLUMNS]
    for parent, child_sums in children.iterrows():
        mismatched = child_sums != parents.loc[parent]
        if mismatched.any():
            raise ValueError(
                f"Breakdown of '{parent}' doesn't add up in: "
                f"{list(mismatched[mismatched].index)}"
            )


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------
def format_count(value):
    """Thousands separators for readability: 1822210 -> '1,822,210'."""
    return f"{value:,.0f}"


def percent_text(share):
    """Percentage text with enough precision for the smallest categories.

    Values below 1% get two decimals, so small categories aren't all rounded
    to the same "0%" or "1%"; the rest get one. An exact zero is "0%".
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
def plot_bars(table, measure, colours, title, note, out_path):
    """One figure with three horizontal bar charts: naval, merchant, all.

    table     categories as rows, VALUE_COLUMNS as columns (row order = the
              order the bars are drawn, top to bottom)
    measure   "count" (number of ships) or "tonnage"
    colours   dict: category -> colour
    title     figure title
    note      footnote text (source and any explanations)

    Why horizontal bars: category names are long, and on the y-axis they read
    left-to-right instead of being rotated under vertical bars.

    Why each panel has its own x-axis: naval and merchant totals differ a lot,
    and the tonnage panels use different units, so a shared axis would squash
    one panel and wrongly suggest the units are comparable.

    Why every bar is labelled "value (percent)": in these tables some
    categories are a single ship, and their bars are barely visible. The label
    gives the exact value, which bar length alone can't show.
    """
    column = "no" if measure == "count" else "tonnage"
    # Figure height grows with the number of categories, so bars and labels
    # keep roughly the same size whether a table has 4 categories or 7.
    height = 3.0 + 0.6 * len(table)
    fig, axes = plt.subplots(1, 3, figsize=(21, height), sharey=True)

    for ax, (panel_title, prefix) in zip(axes, PANELS):
        values = table[f"{prefix}_{column}"]
        shares = values / values.sum() * 100

        # Reverse the y positions so the first category is drawn at the top:
        # matplotlib puts y position 0 at the bottom.
        y = range(len(values))[::-1]
        ax.barh(list(y), values.values, height=0.62,
                color=[colours[label] for label in values.index],
                edgecolor=BACKGROUND, linewidth=2)

        # Label every bar just past its end, in dark text (not the bar colour)
        # so it's readable even beside the pale bars.
        for yi, value, share in zip(y, values.values, shares.values):
            ax.text(value, yi, f"  {format_count(value)}  ({percent_text(share)})",
                    va="center", ha="left", fontsize=11.5, color=INK)

        # Leave 55% spare width to the right of the longest bar so its label
        # fits inside the panel. max(..., 1) avoids a zero-width axis if a
        # panel were all zeros.
        ax.set_xlim(0, max(values.max(), 1) * 1.55)

        unit = "ships" if measure == "count" else TONNAGE_UNITS[prefix]
        ax.set_title(f"{panel_title}\n", loc="left")
        ax.text(0, 1.02, f"Total: {format_count(values.sum())} {unit}",
                transform=ax.transAxes, fontsize=11.5, color=INK_SOFT)

        # Faint vertical gridlines help read bar lengths; no horizontal
        # gridlines, as each bar already has its own label.
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

        # Ship counts are whole numbers. These tables have counts as small as
        # 1 or 2, where matplotlib would otherwise put ticks at 0.5, 1.5, ...
        # (half a ship makes no sense), so integer ticks are forced.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(axis_formatter))
        ax.set_xlabel("Number of ships" if measure == "count" else "Tons")

    # Category names on the first panel only; the panels share the y-axis.
    axes[0].set_yticks(range(len(table))[::-1])
    axes[0].set_yticklabels(table.index)

    fig.suptitle(title, x=0.01, ha="left")
    fig.text(0.01, 0.01, note, fontsize=10, color=INK_SOFT)
    # Reserve room at the bottom for the footnote: about 0.45 inch, plus
    # 0.22 inch for each extra line. Dividing by the figure height turns
    # inches into the 0-1 fraction that tight_layout() expects.
    footnote_inches = 0.45 + 0.22 * note.count("\n")
    fig.tight_layout(rect=(0, footnote_inches / height, 1, 1 - 0.5 / height))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pie charts
# ---------------------------------------------------------------------------
def plot_pies(table, measure, colours, title, note, out_path):
    """One figure with three pie charts: naval, merchant, all.

    Same arguments as plot_bars().

    A pie answers "what share of the whole?", which suits these tables
    because the categories add up to the table's total. Pies are weak at
    comparing similar or very small slices, though, so each pie has its own
    legend underneath listing every category with its number and percentage.
    The legend, not the slice size, gives the exact values, and no category is
    identified by colour alone.
    """
    column = "no" if measure == "count" else "tonnage"
    # The legend under each pie has one line per category, so the figure
    # gets taller (and the legend area bigger) as categories are added.
    height = 8.0 + 0.4 * len(table)
    fig, axes = plt.subplots(1, 3, figsize=(21, height))

    for ax, (panel_title, prefix) in zip(axes, PANELS):
        values = table[f"{prefix}_{column}"]
        shares = values / values.sum() * 100

        # Zero categories can't be drawn as a slice, so they are left out of
        # the pie. They are still in the legend as "0 (0%)", so the reader
        # sees the category exists and was zero.
        nonzero = values[values > 0]

        # Start at 12 o'clock and go clockwise, the way a clock is read, so
        # the slices follow the same order as the bars and the legend. A 2 px
        # edge in the background colour separates neighbouring slices.
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
            # A pie of a single category is one full circle; its label goes in
            # the centre. Otherwise the label sits at the slice's middle angle,
            # 65% of the way from the centre to the rim, where even a narrow
            # slice is wide enough for the text.
            if share == 100:
                x, y = 0.0, 0.0
            else:
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
        handles = [plt.Rectangle((0, 0), 1, 1, color=colours[label])
                   for label in table.index]
        labels = [f"{label}:  {format_count(values[label])}  ({percent_text(shares[label])})"
                  for label in table.index]
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03),
                  frameon=False, fontsize=11.5, handlelength=1.2, handleheight=1.2)

    fig.suptitle(title, x=0.01, ha="left")
    fig.text(0.01, 0.01, note + f"  Slices under {MIN_SHARE_FOR_SLICE_LABEL:g}% "
             "are labelled only in the legend.", fontsize=10, color=INK_SOFT)
    # tight_layout() ignores legends placed outside the axes, so the margins
    # are set by hand: the bottom part of the figure is kept free for the
    # legends. Its size is worked out from the number of legend lines.
    legend_inches = 0.9 + 0.32 * len(table) + 0.2 * note.count("\n")
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - 1.4 / height,
                        bottom=legend_inches / height, wspace=0.25)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Detailed categories inside broad groups
# ---------------------------------------------------------------------------
# The broad weapon charts (e.g. "Aircraft") hide useful detail: carrier-based
# vs land-based aircraft, sabotage, mines dropped from the air, and so on.
# The charts below show both levels at once: each detailed category is drawn
# inside its broad group.
#
# A colour problem: Table II has 17 detailed categories, but only about 8
# colours can be told apart reliably (fewer for colour-blind readers). So
# colour shows the *broad group* (all aircraft categories are aqua, all mine
# categories green, ...), and the detailed category is identified by its text
# label. In the donut's outer ring, neighbouring details of the same group
# alternate between the group colour and a lighter tint of it, so their
# borders are visible.

def parent_key(rows):
    """A key naming each row together with its main row: "parent / row".

    Some tables use the same row name under different main rows, e.g. Table V
    has "Submarines" under both "United States Forces" and "British Forces".
    When the charts must keep those apart, rows are looked up by this
    combined key ("British Forces / Submarines") instead of by name alone.
    A main row without a breakdown is its own parent, e.g.
    "Japanese Mines / Japanese Mines".
    """
    return rows["parent"] + " / " + rows["sinking_agent"]


def group_details(rows, detail_map, group_order, key=None):
    """Add up rows into detailed categories, each belonging to a broad group.

    rows         the CSV rows to use (normally helpers.detail_rows(table))
    detail_map   dict: row key -> (broad group, detail label)
    group_order  the broad groups, in drawing order
    key          how rows are looked up in detail_map: None (the default)
                 uses the row name ("sinking_agent"); pass parent_key to use
                 "parent / row" when the same name appears under several
                 main rows

    Returns one row per detailed category with the VALUE_COLUMNS plus a
    "group" column, sorted by group (in group_order) and, within a group, in
    the order the details first appear in detail_map.

    The same guards as group_rows(): every row must be in detail_map, and
    the groups used must be exactly group_order.
    """
    keys = rows["sinking_agent"] if key is None else key(rows)
    unmapped = set(keys) - set(detail_map)
    if unmapped:
        raise ValueError(f"No detailed category for these rows: {sorted(unmapped)}")
    used_groups = {detail_map[k][0] for k in keys}
    if used_groups != set(group_order):
        raise ValueError(f"Groups {sorted(used_groups)} don't match {group_order}")

    labelled = rows.assign(
        group=keys.map(lambda k: detail_map[k][0]),
        detail=keys.map(lambda k: detail_map[k][1]),
    )
    details = labelled.groupby(["group", "detail"], sort=False)[VALUE_COLUMNS].sum()

    # Sort: groups in group_order; within a group, details in the order they
    # are listed in detail_map (which the table scripts write in a sensible
    # reading order).
    detail_rank = {label: i for i, (_, label) in enumerate(detail_map.values())}
    details = details.reset_index()
    details["group_rank"] = details["group"].map(group_order.index)
    details["detail_rank"] = details["detail"].map(detail_rank)
    details = details.sort_values(["group_rank", "detail_rank"])
    return details.set_index("detail")[["group"] + VALUE_COLUMNS]


def check_details_match_groups(details, broad):
    """Stop if the detailed categories don't add up to the broad groups.

    The detailed charts and the broad charts must tell the same story: e.g.
    the aircraft details must add up to exactly the "Aircraft" group. A
    mismatch would mean a detail was put in the wrong group.
    """
    sums = details.groupby("group")[VALUE_COLUMNS].sum().loc[broad.index]
    if not (sums == broad[VALUE_COLUMNS]).all().all():
        raise ValueError("Detailed categories don't add up to the broad groups:\n"
                         f"{sums - broad[VALUE_COLUMNS]}")


def tint(hex_colour, amount):
    """Mix a colour with white: amount 0 = unchanged, 1 = white.

    Used to make lighter shades of a group colour for the donut's outer ring.
    """
    rgb = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)])
    mixed = rgb + (255 - rgb) * amount
    return "#" + "".join(f"{int(round(c)):02x}" for c in mixed)


def text_colour_on(hex_colour):
    """Dark or white text, whichever is more readable on this fill colour.

    Uses the colour's perceived brightness (luminance): bright fills such as
    yellow or the light tints get near-black text; dark fills get white.
    This replaces the fixed PALE_FILLS list for the tinted shades, which
    aren't in that list.
    """
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if luminance > 0.45 else "white"


def plot_detail_bars(details, measure, group_colours, title, note, out_path):
    """Three-panel bar chart of detailed categories, grouped under headers.

    For every broad group there is a bold header row showing the group's
    subtotal ("Aircraft  251 (41.1%)"), followed by one bar per detailed
    category in that group. So the chart shows both levels: the header rows
    are the broad picture, the bars are the detail.

    If a group has a single detail with the same name as the group (e.g.
    "Submarines"), a header row would just repeat the bar, so that group is
    drawn as one bar with a bold label instead.

    All bars of a group share the group's colour; the detailed categories are
    told apart by their labels.
    """
    column = "no" if measure == "count" else "tonnage"

    # Build the list of rows to draw, top to bottom:
    # ("header", group, None) for a group header, or
    # ("bar", group, detail) for a detailed category.
    rows = []
    for group in group_colours:
        group_details_ = details[details["group"] == group]
        if len(group_details_) == 1 and group_details_.index[0] == group:
            rows.append(("bar", group, group))   # single detail = the group itself
            continue
        rows.append(("header", group, None))
        rows.extend(("bar", group, detail) for detail in group_details_.index)

    height = 2.2 + 0.36 * len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(21, height), sharey=True)
    y_positions = list(range(len(rows)))[::-1]   # first row at the top

    for ax, (panel_title, prefix) in zip(axes, PANELS):
        values = details[f"{prefix}_{column}"]
        group_totals = details.groupby("group")[f"{prefix}_{column}"].sum()
        panel_total = values.sum()
        x_max = max(values.max(), 1)

        for y, (kind, group, detail) in zip(y_positions, rows):
            if kind == "header":
                # Header row: no bar, just the group's subtotal in bold.
                # The leading spaces keep the number clear of the row label
                # on the first panel's y-axis.
                subtotal = group_totals[group]
                ax.text(0, y, f"  {format_count(subtotal)}  ({percent_text(subtotal / panel_total * 100)})",
                        va="center", ha="left", fontsize=11.5, fontweight="bold", color=INK)
                continue
            value = values[detail]
            ax.barh(y, value, height=0.7, color=group_colours[group],
                    edgecolor=BACKGROUND, linewidth=1.5)
            ax.text(value, y, f"  {format_count(value)}  ({percent_text(value / panel_total * 100)})",
                    va="center", ha="left", fontsize=11, color=INK)

        ax.set_xlim(0, x_max * 1.6)
        # Fix the vertical range to the rows. Matplotlib sizes the y-axis to
        # fit the bars only, so a header row at the top (which has no bar)
        # would end up on the edge, overlapping the "Total" line above.
        ax.set_ylim(-0.7, len(rows) - 0.3)
        unit = "ships" if measure == "count" else TONNAGE_UNITS[prefix]
        ax.set_title(f"{panel_title}\n", loc="left")
        ax.text(0, 1.01, f"Total: {format_count(panel_total)} {unit}",
                transform=ax.transAxes, fontsize=11.5, color=INK_SOFT)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(axis_formatter))
        ax.set_xlabel("Number of ships" if measure == "count" else "Tons")

    # Row labels on the first panel: group headers in bold black, details in
    # regular dark grey, so the two levels are easy to tell apart.
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([group if kind == "header" or detail == group else detail
                             for kind, group, detail in rows])
    for tick_label, (kind, group, detail) in zip(axes[0].get_yticklabels(), rows):
        if kind == "header" or detail == group:
            tick_label.set_fontweight("bold")
        else:
            tick_label.set_color(INK_SOFT)

    fig.suptitle(title, x=0.01, ha="left")
    full_note = note + ("\nBold rows are broad groups (with their subtotal); the bars "
                        "under them are its detailed categories, in the group's colour.")
    fig.text(0.01, 0.01, full_note, fontsize=10, color=INK_SOFT)
    footnote_inches = 0.45 + 0.22 * full_note.count("\n")
    fig.tight_layout(rect=(0, footnote_inches / height, 1, 1 - 0.5 / height))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_detail_donuts(details, measure, group_colours, title, note, out_path):
    """Three two-ring pie charts: broad groups inside, details outside.

    Inner ring: the broad groups (e.g. Aircraft, Mines).
    Outer ring: the detailed categories, each directly outside its group, in
    shades of the group's colour.
    Both rings start at 12 o'clock and run clockwise in the same order, and
    the details of a group add up to the group, so each detail lines up with
    its group exactly.

    Percentages are written on slices of at least MIN_SHARE_FOR_SLICE_LABEL.
    With up to 17 details, a legend with numbers under each of the three pies
    would be too long to read, so there is one shared legend (names only)
    under the figure, laid out with one column per group. The exact numbers
    are in the detailed bar chart.
    """
    column = "no" if measure == "count" else "tonnage"

    # Outer-ring colour of each detail: alternate between the group colour
    # and a 45% tint of it within the group, so neighbouring slices of the
    # same group are separated by more than the thin white gap.
    shade = {}
    for group, colour in group_colours.items():
        for i, detail in enumerate(details.index[details["group"] == group]):
            shade[detail] = colour if i % 2 == 0 else tint(colour, 0.45)

    # Legend layout: one column per group, header first, then its details.
    # Shorter columns are padded with blank entries so every column starts
    # at the top (matplotlib fills legend columns top-to-bottom).
    columns = []
    for group, colour in group_colours.items():
        entries = [(colour, group, True)]
        group_detail_names = list(details.index[details["group"] == group])
        if group_detail_names != [group]:   # skip "Submarines" under "Submarines"
            entries += [(shade[d], d, False) for d in group_detail_names]
        columns.append(entries)
    longest = max(len(c) for c in columns)

    # Details that are the only detail of their group (used to avoid
    # labelling the same percentage on both rings).
    group_sizes = details["group"].value_counts()
    single_detail_groups = {d: group_sizes[g] == 1 for d, g in details["group"].items()}

    # Figure height = pie area + legend area. The pies are limited by the
    # panel width (7 inches each), so a pie area taller than about 7.5 inches
    # would only add empty space above the legend.
    legend_inches = 0.6 + 0.3 * longest
    height = 7.5 + legend_inches + 0.2 * note.count("\n")
    fig, axes = plt.subplots(1, 3, figsize=(21, height))

    for ax, (panel_title, prefix) in zip(axes, PANELS):
        values = details[f"{prefix}_{column}"]
        total = values.sum()
        groups = details.groupby("group", sort=False)[f"{prefix}_{column}"].sum()
        groups = groups[groups > 0]
        nonzero = values[values > 0]

        # Inner ring (a full pie of radius 0.62) and outer ring (a band from
        # radius 0.65 to 1.0). The small gap between them separates the levels.
        inner, _ = ax.pie(groups.values, radius=0.62, startangle=90, counterclock=False,
                          colors=[group_colours[g] for g in groups.index],
                          wedgeprops={"edgecolor": BACKGROUND, "linewidth": 2})
        outer, _ = ax.pie(nonzero.values, radius=1.0, startangle=90, counterclock=False,
                          colors=[shade[d] for d in nonzero.index],
                          wedgeprops={"width": 0.35, "edgecolor": BACKGROUND, "linewidth": 1.5})

        # Percent labels: inner ring at 60% of its radius, outer ring in the
        # middle of its band (radius 0.825). Text colour follows the fill.
        for wedges, labels, colours, radius in [
            (inner, groups.index, group_colours, 0.62 * 0.6),
            (outer, nonzero.index, shade, 0.825),
        ]:
            for wedge, label in zip(wedges, labels):
                share = (groups if wedges is inner else nonzero)[label] / total * 100
                if share < MIN_SHARE_FOR_SLICE_LABEL:
                    continue
                # A detail that is its group's only detail (e.g. "Submarines")
                # has the same percentage as the inner slice next to it, so
                # it isn't labelled twice.
                if wedges is outer and single_detail_groups.get(label):
                    continue
                if share == 100:
                    x, y = (0.0, 0.0) if wedges is inner else (0.0, radius)
                else:
                    angle = np.radians((wedge.theta1 + wedge.theta2) / 2)
                    x, y = radius * np.cos(angle), radius * np.sin(angle)
                ax.text(x, y, percent_text(share), ha="center", va="center",
                        fontsize=12.5, fontweight="bold",
                        color=text_colour_on(colours[label]))

        unit = "ships" if measure == "count" else TONNAGE_UNITS[prefix]
        ax.set_title(f"{panel_title}\nTotal: {format_count(total)} {unit}", fontsize=14)
        ax.set_aspect("equal")

    # One shared legend for the figure, column per group. Group names are
    # bold so they read as headings over their details.
    handles, texts, bold = [], [], []
    for entries in columns:
        padded = entries + [(None, "", False)] * (longest - len(entries))
        for colour, text, is_header in padded:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=colour or BACKGROUND,
                                         alpha=1 if colour else 0))
            texts.append(text)
            bold.append(is_header)
    # The legend sits just above the footnote, whose height depends on its
    # number of lines (about 0.2 inch each).
    footnote_inches = 0.3 + 0.2 * (note.count("\n") + 1)
    legend = fig.legend(handles, texts, loc="lower center", ncol=len(columns),
                        bbox_to_anchor=(0.5, footnote_inches / height), frameon=False,
                        fontsize=11, handlelength=1.2, handleheight=1.2,
                        columnspacing=2.0)
    for text, is_bold in zip(legend.get_texts(), bold):
        if is_bold:
            text.set_fontweight("bold")

    fig.suptitle(title, x=0.01, ha="left")
    # The footnote starts a new line for the ring explanation, so it never runs
    # off the right edge of the figure.
    fig.text(0.01, 0.01, note + "\nInner ring: broad groups. Outer ring: detailed "
             "categories (exact numbers in the detailed bar chart). Slices under "
             f"{MIN_SHARE_FOR_SLICE_LABEL:g}% are not labelled.",
             fontsize=10, color=INK_SOFT)
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - 1.4 / height,
                        bottom=(legend_inches + footnote_inches) / height, wspace=0.15)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_detail_all(details, group_colours, title_start, note, out_dir, file_start):
    """Draw the detailed bar chart and donut chart, for ships and tonnage.

    Saved as <file_start>_detail_bar_ships.png, _detail_bar_tonnage.png,
    _detail_pie_ships.png and _detail_pie_tonnage.png in out_dir.
    """
    for measure, measure_name, file_measure in [("count", "number of ships", "ships"),
                                                ("tonnage", "tonnage", "tonnage")]:
        plot_detail_bars(details, measure, group_colours,
                         f"{title_start}: {measure_name}", note,
                         out_dir / f"{file_start}_detail_bar_{file_measure}.png")
        plot_detail_donuts(details, measure, group_colours,
                           f"Share of {title_start}: {measure_name}", note,
                           out_dir / f"{file_start}_detail_pie_{file_measure}.png")


def plot_all(table, colours, title_start, note, out_dir, file_start):
    """Draw the four standard figures for one set of categories.

    Bars and pies, each for number of ships and for tonnage, saved as
    <file_start>_bar_ships.png, _bar_tonnage.png, _pie_ships.png and
    _pie_tonnage.png in out_dir.
    """
    for measure, measure_name, file_measure in [("count", "number of ships", "ships"),
                                                ("tonnage", "tonnage", "tonnage")]:
        plot_bars(table, measure, colours, f"{title_start}: {measure_name}", note,
                  out_dir / f"{file_start}_bar_{file_measure}.png")
        plot_pies(table, measure, colours, f"Share of {title_start}: {measure_name}", note,
                  out_dir / f"{file_start}_pie_{file_measure}.png")
