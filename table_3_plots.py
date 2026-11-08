"""
Table III of the JANAC report: Japanese ships sunk by the other Allies.

Table I showed that Allied forces other than the United States sank 45
Japanese warships and 73 merchant ships on their own (about 4% of all
losses). Table III breaks those 118 ships down by nation (British, Dutch,
Australian, New Zealand, Chinese, Soviet) and, within each nation, by weapon.

This script draws eight PNG figures from the hand-verified transcription of
Table III (tests/fixtures/table_III.csv), in figures/table_III/:

  By nation (who sank them):
    table_III_bar_ships.png      table_III_bar_tonnage.png
    table_III_pie_ships.png      table_III_pie_tonnage.png
  By kind of weapon (all nations together):
    table_III_weapon_bar_ships.png   table_III_weapon_bar_tonnage.png
    table_III_weapon_pie_ships.png   table_III_weapon_pie_tonnage.png

Every figure has three panels: Naval ships, Merchant ships and All ships.
Each bar / slice shows the raw number and its percentage of the panel total.
The plotting itself is shared with Tables IV and V and lives in
table_plot_helpers.py; this script decides which categories to plot and why.

Note: the numbers here are small (the biggest is 42 ships), so a single ship
moves a percentage by one or more points. Read the percentages together with
the counts next to them.

Run:
    python table_3_plots.py
"""

from pathlib import Path

import table_plot_helpers as helpers
from table_plot_helpers import AQUA, BLUE, GREEN, GREY, ORANGE, PINK, VIOLET, YELLOW

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Built relative to this file, so the script works from any folder.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "tests" / "fixtures" / "table_III.csv"
OUT_DIR = PROJECT_DIR / "figures" / "table_III"

# ---------------------------------------------------------------------------
# Chart 1: by nation
# ---------------------------------------------------------------------------
# Table III has 9 main (level 0) rows. Six are single nations. The other
# three are sinkings shared by two Allies: Netherlands + Australian aircraft
# (2 ships), Netherlands + Indian surface craft (1 ship), and Australian +
# Indian surface craft (1 ship). "Indian" means the Royal Indian Navy. These
# three are merged into one "Two Allies together" category, for two reasons:
#   - they are 4 ships in total, and three 1-2 ship categories would add three
#     colours to the charts for almost no information;
#   - 9 categories is more than the 8 colours that stay distinguishable for
#     colour-blind readers.
# Left: the report's wording; right: the chart label.
NATION_LABELS = {
    "British Forces": "British",
    "Netherlands Forces": "Dutch",
    "Australian Forces": "Australian",
    "New Zealand Forces": "New Zealand",
    "Chinese Aircraft": "Chinese (aircraft)",
    "Russian Forces": "Soviet",
    "Netherlands and Australian Aircraft": "Two Allies together",
    "Netherlands and Indian Surface Craft": "Two Allies together",
    "Australian and Indian Surface Craft": "Two Allies together",
}

# One colour per nation, in the palette's fixed order, which passed the
# colour-blindness validator for neighbouring colours (bars next to each
# other and pie slices next to each other). The dictionary order is also the
# drawing order: the report's own order, so the charts can be checked
# against the printed table line by line.
NATION_COLOURS = {
    "British": BLUE,
    "Dutch": ORANGE,
    "Australian": AQUA,
    "New Zealand": YELLOW,
    "Chinese (aircraft)": PINK,
    "Soviet": GREEN,
    "Two Allies together": VIOLET,
}

# ---------------------------------------------------------------------------
# Chart 2: by kind of weapon
# ---------------------------------------------------------------------------
# The weapon is named only on the detailed rows (e.g. "Submarines" under
# "British Forces"), or in the name of main rows that have no breakdown
# (e.g. "Chinese Aircraft"). helpers.detail_rows() picks exactly those rows,
# so each ship is counted once. They are then grouped into 5 kinds of weapon.
#
# Choices made here:
#   - All aircraft go into one "Aircraft" group (Navy carrier-based, Navy
#     land-based, Army, and plain "Aircraft"), because Table III's numbers
#     are too small to split them meaningfully.
#   - "Other" collects the rare cases: shore batteries (2 Dutch sinkings),
#     sabotage (2 British), one British aircraft + surface craft attack, and
#     one Soviet sinking of unknown cause. Together 6 ships.
WEAPON_GROUPS = {
    "Submarines": "Submarines",
    "Surface Craft": "Surface warships",
    "Netherlands and Indian Surface Craft": "Surface warships",
    "Australian and Indian Surface Craft": "Surface warships",
    "Navy Carrier-Based Aircraft": "Aircraft",
    "Navy Land-Based Aircraft": "Aircraft",
    "Army Aircraft": "Aircraft",
    "Aircraft": "Aircraft",
    "Chinese Aircraft": "Aircraft",
    "Netherlands and Australian Aircraft": "Aircraft",
    "Mines": "Mines",
    "Army Aircraft and Surface Craft": "Other (shore guns, sabotage, mixed, unknown)",
    "Sabotage": "Other (shore guns, sabotage, mixed, unknown)",
    "Shore Batteries": "Other (shore guns, sabotage, mixed, unknown)",
    "Unknown": "Other (shore guns, sabotage, mixed, unknown)",
}

# Same colours as the broad weapon groups of Table II (submarines blue,
# surface warships orange, aircraft aqua, mines green, other grey), so the
# reader can compare the US and Allied weapon mix figure to figure. This set
# passed the colour-blindness validator. Grey marks "Other" as a leftover.
WEAPON_COLOURS = {
    "Submarines": BLUE,
    "Surface warships": ORANGE,
    "Aircraft": AQUA,
    "Mines": GREEN,
    "Other (shore guns, sabotage, mixed, unknown)": GREY,
}

# ---------------------------------------------------------------------------
# Chart 3: detailed weapon categories inside the broad weapon groups
# ---------------------------------------------------------------------------
# Same detailed rows as chart 2, but the detail is kept instead of being
# merged: which kind of aircraft, sabotage vs shore guns, and so on. Each
# entry is: report wording -> (broad group, detailed label). The broad group
# is always the one WEAPON_GROUPS gives, which the script checks.
#
# Choices made here:
#   - Joint surface-craft sinkings (Dutch + Indian, Australian + Indian) are
#     ordinary surface-warship sinkings, so they join "Surface warships".
#   - Rows that just say "Aircraft" (Dutch, Australian, New Zealand, Soviet,
#     Chinese, and Dutch + Australian) don't say which service flew them; they
#     become "Aircraft, type not given" rather than being guessed.
# The order within each group is the reading order on the charts.
OTHER = "Other (shore guns, sabotage, mixed, unknown)"
DETAIL_CATEGORIES = {
    "Submarines": ("Submarines", "Submarines"),
    "Surface Craft": ("Surface warships", "Surface warships"),
    "Netherlands and Indian Surface Craft": ("Surface warships", "Surface warships"),
    "Australian and Indian Surface Craft": ("Surface warships", "Surface warships"),
    "Navy Carrier-Based Aircraft": ("Aircraft", "Navy carrier-based aircraft"),
    "Navy Land-Based Aircraft": ("Aircraft", "Navy land-based aircraft"),
    "Army Aircraft": ("Aircraft", "Army aircraft"),
    "Aircraft": ("Aircraft", "Aircraft, type not given"),
    "Chinese Aircraft": ("Aircraft", "Aircraft, type not given"),
    "Netherlands and Australian Aircraft": ("Aircraft", "Aircraft, type not given"),
    "Mines": ("Mines", "Mines"),
    "Shore Batteries": (OTHER, "Shore batteries"),
    "Sabotage": (OTHER, "Sabotage"),
    "Army Aircraft and Surface Craft": (OTHER, "Aircraft + surface craft together"),
    "Unknown": (OTHER, "Unknown"),
}

SOURCE_NOTE = ("Source: JANAC (1947), Japanese Naval and Merchant Shipping Losses "
               "During World War II by All Causes, Table III (Allied forces other than "
               "the US, acting without US forces). Percentages are of each panel's total.")


if __name__ == "__main__":
    helpers.set_readable_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table, total_row = helpers.read_table(DATA_FILE)

    # Each nation's weapon breakdown must add up to that nation's row, or the
    # two sets of charts below would disagree; the script stops if not.
    helpers.check_breakdowns(table)

    # --- By nation: use the main (level 0) rows only. ---------------------
    main_rows = table[table["level"] == 0]
    by_nation = helpers.group_rows(main_rows, "sinking_agent", NATION_LABELS,
                                   list(NATION_COLOURS))
    # The nations must add up to the report's TOTAL row (45 warships,
    # 73 merchant ships, and their tonnages), or the script stops.
    helpers.check_totals(by_nation, total_row)
    print(by_nation, "\n")

    helpers.plot_all(
        by_nation, NATION_COLOURS,
        "Japanese ships sunk by Allies other than the US, by nation",
        SOURCE_NOTE + "\n'Two Allies together' = Dutch + Australian aircraft (2 ships), "
        "Dutch + Indian surface craft (1), Australian + Indian surface craft (1).",
        OUT_DIR, "table_III")

    # --- By kind of weapon: use the most detailed rows. --------------------
    by_weapon = helpers.group_rows(helpers.detail_rows(table), "sinking_agent",
                                   WEAPON_GROUPS, list(WEAPON_COLOURS))
    # Regrouping must not lose or add a single ship: same check as above.
    helpers.check_totals(by_weapon, total_row)
    print(by_weapon)

    helpers.plot_all(
        by_weapon, WEAPON_COLOURS,
        "Japanese ships sunk by Allies other than the US, by kind of weapon",
        SOURCE_NOTE + "\nAircraft = Navy carrier- and land-based, Army, and unspecified "
        "aircraft. Other = shore batteries (2 ships), sabotage (2), aircraft + "
        "surface craft (1), unknown (1).",
        OUT_DIR, "table_III_weapon")

    # --- Detailed weapon categories inside the broad groups. ---------------
    # Every detail must sit in the same broad group that WEAPON_GROUPS gives
    # its row, so the detailed and broad charts can't disagree.
    for key, (group, _) in DETAIL_CATEGORIES.items():
        if WEAPON_GROUPS[key] != group:
            raise ValueError(f"'{key}' is in group '{group}' here but "
                             f"'{WEAPON_GROUPS[key]}' in WEAPON_GROUPS")
    details = helpers.group_details(helpers.detail_rows(table), DETAIL_CATEGORIES,
                                    list(WEAPON_COLOURS))
    helpers.check_details_match_groups(details, by_weapon)
    print(details)

    helpers.plot_detail_all(
        details, WEAPON_COLOURS,
        "Japanese ships sunk by Allies other than the US, by detailed weapon",
        SOURCE_NOTE, OUT_DIR, "table_III_weapon")

    print(f"\nSaved 12 figures to {OUT_DIR}")
