"""
Line graphs of Japanese ship losses per month, December 1941 - August 1945.

The x axis is time (one point per month). The y axis is either the number of
ships sunk that month or the tonnage sunk that month. Each kind of vessel
(warships, merchant ships) gets its own line.

It uses the same three filters, with the same options and defaults, as
sinking_map.py and sinking_animation.py, plus one more:

    agent    - the nation credited with the sinking (us, britain, australia, ...)
    weapon   - the kind of weapon (aircraft, submarine, mine, ...)
    vessel   - naval (warships) or merchant ships
    measure  - ships (number of ships) or tonnage

Leave a filter out and it includes all of its options; leaving out measure
makes both graphs. A wrong value stops the program with a message that names
the filter and lists its options. A selection that matches no ship makes no
graph, and says so.

Unlike the maps, these graphs count EVERY matching ship, including the ones
whose position is only text ("off Japan"): a sinking only needs a month to be
counted here, and every ship in the report has one.

Input (the same as sinking_map.py):
    data/naval/naval_losses_15-44.csv
    data/merchant/merchant_losses_45-115.csv

Output: PNGs in figures/line_graphs/, named after the measure and the filters, e.g.
    figures/line_graphs/losses_ships_agent-all_weapon-all_vessel-all_type-all_tons-all.png
    figures/line_graphs/losses_tonnage_agent-us_weapon-submarine_vessel-merchant_type-all_tons-all.png
(The type-/tons- parts come from sinking_map.file_name(); this script has
no ship-type or tonnage filter yet, so they are always "all".)

Run:
    python loss_line_graph.py                                     # both graphs, all ships
    python loss_line_graph.py --measure tonnage --agent us --weapon submarine
    python loss_line_graph.py --measure ships --vessel naval --weapon aircraft
    python loss_line_graph.py --options                           # list every valid value

or from Python:
    from loss_line_graph import make_line_graphs
    make_line_graphs(agent="britain", measure="ships")
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no window is needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

import sinking_map as smap
import table_plot_helpers as helpers
from sinking_map import OptionError, VESSEL_COLOURS, VESSEL_OPTIONS
from table_plot_helpers import BACKGROUND, GRID, INK, INK_SOFT

OUT_DIR = smap.PROJECT_DIR / "figures" / "line_graphs"

# The fourth filter. Like the other three: leave it out to get all options,
# i.e. both graphs.
MEASURE_OPTIONS = {
    "ships": "Number of ships sunk per month",
    "tonnage": "Tonnage sunk per month",
}

# Every month of the war, so months with no losses show as 0 instead of the
# line jumping straight over them. The report's first loss is in December
# 1941 and its last in August 1945.
MONTHS = pd.period_range("1941-12", "1945-08", freq="M")

# Tonnage units. They are different measures (see the footnote), which is
# why tonnage is never added across the two kinds of vessel.
TONNAGE_UNITS = {"naval": "tons (standard displacement)",
                 "merchant": "gross register tons (GRT)"}

# The total line for "ships" (warships + merchant ships together). Counting
# ships is the same unit for both kinds, so a total is meaningful. It is
# drawn in near-black ink, not a palette colour, because it is not a third
# kind of vessel but the sum of the other two.
TOTAL_COLOUR = INK


class EmptySelection(ValueError):
    """Raised when the filters match no ship at all."""


def check_measure(measure):
    """Turn the measure filter into a list of options, or raise OptionError.

    Works like sinking_map.check_filters(): None means all options, a
    string means one option, and a wrong value gives the same kind of
    message, naming the filter and listing its options.
    """
    if measure is None or (not isinstance(measure, str) and len(measure) == 0):
        return list(MEASURE_OPTIONS)
    values = [measure] if isinstance(measure, str) else measure
    keys = [smap.normalise(v) for v in values]
    keys = ["ships" if k in ("number", "count", "ship") else k for k in keys]  # aliases
    wrong = [v for v, k in zip(values, keys) if k not in MEASURE_OPTIONS]
    if wrong:
        raise OptionError("Wrong filter value(s):\n" + "\n".join(
            f"  '{v}' is not a valid measure. Options for measure: "
            f"{', '.join(MEASURE_OPTIONS)}" for v in wrong)
            + "\nLeave a filter out to include all of its options.")
    return [k for k in MEASURE_OPTIONS if k in keys]


def monthly_table(selected, measure):
    """Sum the selected ships per month for each kind of vessel.

    Returns a table with one row per month of the war (MONTHS) and one
    column per kind of vessel: the number of ships, or their tonnage.
    The month comes from the year and month columns, which every ship has,
    even the six warships whose day is not given in the report.
    """
    month = pd.PeriodIndex.from_fields(year=selected["year"], month=selected["month"], freq="M")
    values = selected["tonnage"] if measure == "tonnage" else pd.Series(1, index=selected.index)
    table = (values.groupby([month, selected["vessel"]]).sum().unstack(fill_value=0)
             .reindex(index=MONTHS, fill_value=0))
    return table.reindex(columns=[k for k in VESSEL_OPTIONS if k in table.columns],
                         fill_value=0)


def thousands(value, _position):
    """Tick label for tonnage: 250,000 -> '250k', 1,500,000 -> '1.5M'."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def style_axis(ax, measure):
    """Grid, ticks and labels shared by every panel.

    Only horizontal gridlines: the eye reads values across from the y axis,
    and vertical lines every month would compete with the data.
    """
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    if measure == "ships":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # no "2.5 ships"
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def place_end_labels(ax, x_end, labels):
    """Write each line's name just right of its last point, without overlaps.

    Labels at the line ends (direct labelling) save the reader going back
    and forth to a legend. But lines often end at the same value (e.g. all
    at 0 in August 1945), which would print the names on top of each other.
    So the labels are sorted by height and pushed apart until neighbours are
    at least MIN_GAP of the axis height apart. Call this after the y limits
    are final, because the gap is measured in axis units.

    labels: list of (text, colour, last value).
    """
    min_gap = 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    placed = []
    for text, colour, y in sorted(labels, key=lambda item: item[2]):
        if placed and y - placed[-1][2] < min_gap:
            y = placed[-1][2] + min_gap
        placed.append((text, colour, y))
    for text, colour, y in placed:
        ax.text(x_end + 12, y, text, color=colour, fontsize=12, fontweight="bold",
                va="center", ha="left")


def draw_line(ax, x, values, colour, label, measure, width=2.2):
    """Draw one line and mark its peak month; return its end-label details.

    The peak month is marked with a dot and its value, the single number
    most people look for. The line's name is written later by
    place_end_labels(), once all lines are drawn, so names can't overlap.
    """
    ax.plot(x, values, color=colour, linewidth=width, marker="o", markersize=3.5,
            label=label, zorder=3)
    peak = values.to_numpy().argmax()
    if values.iloc[peak] > 0:
        ax.scatter([x[peak]], [values.iloc[peak]], s=60, color=colour, zorder=4,
                   edgecolors="white", linewidths=1.2)
        when = MONTHS[peak].strftime("%b %Y")
        number = f"{values.iloc[peak]:,.0f}"
        # A box in the background colour behind the text, so another line
        # passing behind the label (e.g. the total line) doesn't cross it.
        ax.annotate(f"Peak: {number} ({when})", (x[peak], values.iloc[peak]),
                    xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=11, color=INK_SOFT, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=BACKGROUND,
                              edgecolor="none", alpha=0.9))
    return (label, colour, values.iloc[-1])


def draw_graph(selected, chosen, measure, out_path):
    """Draw the line graph for one measure and save it as a PNG.

    Ships: one panel. Warships and merchant ships are both counted in
    ships, so they share one axis, with a third line for their total.

    Tonnage: one panel per kind of vessel, one above the other with the same
    time axis. Naval tonnage is standard displacement and merchant tonnage is
    gross register tons, which are different units, so they get separate y
    axes and are never added up. Separate panels (not one chart with two
    y axes) because two y scales on one chart make it look as if the lines
    can be compared by height when they can't.
    """
    table = monthly_table(selected, measure)
    kinds = [k for k in chosen["vessel"] if table[k].sum() > 0] if not table.empty else []
    x = mdates.date2num(MONTHS.to_timestamp() + pd.Timedelta(days=14))  # mid-month

    if measure == "ships":
        fig, ax = plt.subplots(figsize=(16, 7.5))
        axes = [ax]
        end_labels = [[]]             # one list of end labels per panel
        for kind in kinds:
            end_labels[0].append(draw_line(ax, x, table[kind], VESSEL_COLOURS[kind],
                                           VESSEL_OPTIONS[kind].split(" (")[0], measure))
        if len(kinds) == 2:
            end_labels[0].append(draw_line(ax, x, table[kinds].sum(axis=1), TOTAL_COLOUR,
                                           "All ships", measure, width=1.6))
        ax.set_ylabel("Ships sunk per month", fontsize=13, color=INK_SOFT)
    else:
        fig, axes = plt.subplots(len(kinds), 1, figsize=(16, 3.4 + 3.6 * len(kinds)),
                                 sharex=True, squeeze=False)
        axes = list(axes[:, 0])
        end_labels = []
        for ax, kind in zip(axes, kinds):
            end_labels.append([draw_line(ax, x, table[kind], VESSEL_COLOURS[kind],
                                         VESSEL_OPTIONS[kind].split(" (")[0], measure)])
            ax.set_ylabel(f"{TONNAGE_UNITS[kind].split(' (')[0].capitalize()}\nper month",
                          fontsize=13, color=INK_SOFT)

    for ax, labels in zip(axes, end_labels):
        style_axis(ax, measure)
        ax.set_xlim(x[0] - 20, x[-1] + 20)
        ax.margins(y=0.15)            # room above the peak label
        ax.set_ylim(bottom=0)
        place_end_labels(ax, x[-1], labels)
    axes[-1].tick_params(axis="x", labelsize=12)

    # Title and selection line are placed in inches from the top, so the
    # spacing is the same for the one-panel and two-panel figures.
    height = fig.get_figheight()
    what = "Number of Japanese ships sunk" if measure == "ships" else "Japanese tonnage sunk"
    fig.text(0.06, 1 - 0.2 / height, f"{what} per month, 1941–1945", va="top",
             fontsize=19, fontweight="bold", color=INK)
    fig.text(0.06, 1 - 0.62 / height, smap.selection_text(chosen), va="top",
             fontsize=13, color=INK_SOFT)

    totals = []
    for kind in kinds:
        total = table[kind].sum()
        unit = "ships" if measure == "ships" else TONNAGE_UNITS[kind].split(" (")[0]
        totals.append(f"{VESSEL_OPTIONS[kind].split(' (')[0]}: {total:,.0f} {unit}")
    note = (f"Source: JANAC (1947) ship lists. Total over the war: {'; '.join(totals)}. "
            "Every matching ship is counted, including those with no map position.")
    if measure == "tonnage":
        note += ("\nNaval tonnage is standard displacement, merchant tonnage is gross register "
                 "tons: different units, so each has its own panel. Some tonnages are the "
                 "report's estimates.")
    else:
        note += "\nShips sunk jointly by several agents or weapons are counted once."
    fig.text(0.06, 0.012, note, fontsize=10.5, color=INK_SOFT, va="bottom")

    # Fixed bands in inches: 1.35 at the top for the title, selection line
    # and peak labels; 1.05 at the bottom for the date labels and footnote.
    # The right margin leaves room for the line labels.
    fig.subplots_adjust(left=0.06, right=0.88, bottom=1.05 / height,
                        top=1 - 1.35 / height, hspace=0.25)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_line_graphs(agent=None, weapon=None, vessel=None, measure=None):
    """Check the filters, draw one graph per measure, and return the PNG paths.

    agent, weapon and vessel work exactly as in sinking_map.make_map(); measure
    is "ships", "tonnage", a list of both, or None for both. Raises
    OptionError for a wrong value and EmptySelection when no ship matches.
    """
    # Check all four filters before stopping, so one message lists every
    # wrong value (e.g. a wrong agent AND a wrong measure), as in sinking_map.
    problems = []
    try:
        chosen = smap.check_filters(agent=agent, weapon=weapon, vessel=vessel)
    except OptionError as error:
        problems += str(error).splitlines()[1:-1]
    try:
        measures = check_measure(measure)
    except OptionError as error:
        problems += str(error).splitlines()[1:-1]
    if problems:
        raise OptionError("Wrong filter value(s):\n" + "\n".join(problems)
                          + "\nLeave a filter out to include all of its options.")
    helpers.set_readable_style()

    selected = smap.select_ships(smap.load_ships(), chosen)
    if selected.empty:
        raise EmptySelection(f"No ship matches {smap.selection_text(chosen)}; no graph made.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for m in measures:
        out_path = OUT_DIR / smap.file_name(chosen).replace("sinkings_", f"losses_{m}_")
        draw_graph(selected, chosen, m, out_path)
        paths.append(out_path)
        print(f"Saved {out_path}")
    return paths


if __name__ == "__main__":
    # Same filter arguments as sinking_map.py, plus --measure.
    parser = argparse.ArgumentParser(
        description="Line graph of Japanese ship losses per month, filtered by agent, weapon "
                    "and vessel kind. Leave a filter out to include all options.")
    parser.add_argument("--agent", nargs="+", help="who sank them, e.g. us australia")
    parser.add_argument("--weapon", nargs="+", help="weapon, e.g. aircraft submarine")
    parser.add_argument("--vessel", nargs="+", help="naval and/or merchant")
    parser.add_argument("--measure", nargs="+", help="ships and/or tonnage (default: both)")
    parser.add_argument("--options", action="store_true", help="list all valid values and stop")
    args = parser.parse_args()

    if args.options:
        print(smap.describe_options())
        print("measure:\n" + "\n".join(f"  {k:<14} {v}" for k, v in MEASURE_OPTIONS.items()))
        sys.exit(0)
    try:
        make_line_graphs(agent=args.agent, weapon=args.weapon, vessel=args.vessel,
                         measure=args.measure)
    except (OptionError, EmptySelection) as error:
        print(error, file=sys.stderr)
        sys.exit(2)
