"""
Table IV of the JANAC report: losses not credited to one side alone.

Tables II and III cover ships sunk by US forces alone and by other Allies
alone. Table IV covers everything else, 30 warships and 156 merchant ships:
  - sinkings shared between US and Allied forces;
  - marine casualties, i.e. accidents (groundings, collisions, storms,
    explosions), not enemy action;
  - ships that hit Japan's own (defensive) minefields;
  - ships sunk by forces that could not be identified.
Each of these four groups is one top-level (level 0) row of Table IV, and
also one row of Table I.

This script draws four PNG figures from the hand-verified transcription of
Table IV (tests/fixtures/table_IV.csv), in figures/table_IV/:

    table_IV_bar_ships.png     table_IV_bar_tonnage.png
    table_IV_pie_ships.png     table_IV_pie_tonnage.png
  and the same four categories with their details inside them (which
  Allies shared a credit; what the unidentified forces were):
    table_IV_detail_bar_ships.png     table_IV_detail_bar_tonnage.png
    table_IV_detail_pie_ships.png     table_IV_detail_pie_tonnage.png

Every figure has three panels: Naval ships, Merchant ships and All ships.
Each bar / slice shows the raw number and its percentage of the panel total.
The plotting itself is shared with Tables III and V and lives in
table_plot_helpers.py; this script decides which categories to plot and why.

Known error in the report: Table IV's printed TOTAL for merchant tonnage is
492,589, but its rows add up to 492,587 (2 tons less). The charts use the
rows, so percentages add up to 100%; the difference is allowed explicitly
below and printed as a note when the script runs.

Run:
    python table_4_plots.py
"""

from pathlib import Path

import table_plot_helpers as helpers
from table_plot_helpers import AQUA, GREEN, PINK, YELLOW

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Built relative to this file, so the script works from any folder.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "tests" / "fixtures" / "table_IV.csv"
OUT_DIR = PROJECT_DIR / "figures" / "table_IV"

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
# The four level-0 rows are plotted as they are; there are few enough that no
# merging is needed. Their level-1 breakdowns (e.g. which US and Allied
# forces shared a sinking) are 1-6 ships each, too small to chart usefully.
#
# The labels are the same as for these rows in the Table I charts, so the
# reader recognises them. "Marine Casualties" is relabelled because it
# sounds like it means US Marines; in the report it means accidents.
CATEGORY_LABELS = {
    "United States and Allied Forces": "US + Allies (shared)",
    "Marine Casualties": "Accidents (marine casualty)",
    "Japanese Mines": "Japan's own mines",
    "Unidentified Forces": "Unknown",
}

# The same colours these four categories have in the Table I charts (aqua,
# yellow, pink, green), so a category looks the same in both sets of
# figures. In this order they also passed the colour-blindness validator,
# including the pie's wrap-around pair (green next to aqua). The dictionary
# order is the drawing order: the report's own order.
COLOURS = {
    "US + Allies (shared)": AQUA,
    "Accidents (marine casualty)": YELLOW,
    "Japan's own mines": PINK,
    "Unknown": GREEN,
}

# ---------------------------------------------------------------------------
# Detailed categories inside the four groups
# ---------------------------------------------------------------------------
# The detail charts show what is inside each of the four categories, using
# the level-1 rows of Table IV. Rows are looked up by "main row / row"
# (helpers.parent_key), so it is always clear which main row a detail comes
# from. Each entry is: that key -> (chart category, detailed label).
#
# Choices made here:
#   - The 9 shared US + Allied rows are 1-4 ships each and name exact
#     weapons ("US Army Aircraft and Australian Aircraft", ...). They are
#     summarised by *which Allies* shared the credit with the US, the point
#     of this category: 5 partner combinations.
#   - Accidents and Japan's own mines have no breakdown in the report, so
#     each is a single detail with the same name as its category. The
#     charts then show them as one bar / one slice, without a header row.
#   - "Unidentified Forces" splits into aircraft, a submarine, and fully
#     unknown cause.
_SHARED = "United States and Allied Forces / "
_UNKNOWN = "Unidentified Forces / "
DETAIL_CATEGORIES = {
    _SHARED + "United States Army Aircraft and Australian Aircraft":
        ("US + Allies (shared)", "US + Australian"),
    _SHARED + "United States Army Aircraft and Surface Craft and Australian and Netherlands Aircraft":
        ("US + Allies (shared)", "US + Australian + Dutch"),
    _SHARED + "United States Navy Carrier-Based Aircraft and British Aircraft":
        ("US + Allies (shared)", "US + British"),
    _SHARED + "United States Army Aircraft and British Aircraft":
        ("US + Allies (shared)", "US + British"),
    _SHARED + "United States and British Surface Craft":
        ("US + Allies (shared)", "US + British"),
    _SHARED + "United States Surface Craft and British Aircraft":
        ("US + Allies (shared)", "US + British"),
    _SHARED + "United States Army Aircraft and Surface Craft and British Aircraft":
        ("US + Allies (shared)", "US + British"),
    _SHARED + "United States Army Aircraft and British and Netherlands Aircraft":
        ("US + Allies (shared)", "US + British + Dutch"),
    _SHARED + "United States Navy-Marine Land-Based Aircraft and New Zealand Surface Craft":
        ("US + Allies (shared)", "US + New Zealand"),
    "Marine Casualties / Marine Casualties":
        ("Accidents (marine casualty)", "Accidents (marine casualty)"),
    "Japanese Mines / Japanese Mines":
        ("Japan's own mines", "Japan's own mines"),
    _UNKNOWN + "Aircraft": ("Unknown", "Unidentified aircraft"),
    _UNKNOWN + "Submarines": ("Unknown", "Unidentified submarine"),
    _UNKNOWN + "Unknown": ("Unknown", "Cause unknown"),
}

# The printing error in the report (see the module docstring), as
# "printed TOTAL minus the sum of the rows". check_totals() allows exactly
# this difference and nothing else, so any new mistake would still stop the
# script. The "total_tonnage" column is not affected: its printed TOTAL
# (566,387) does match its rows.
KNOWN_SOURCE_ERRORS = {"merchant_tonnage": 492_589 - 492_587}

SOURCE_NOTE = ("Source: JANAC (1947), Japanese Naval and Merchant Shipping Losses "
               "During World War II by All Causes, Table IV. Percentages are of each "
               "panel's total.\nAccidents = groundings, collisions, storms, explosions. "
               "Merchant tonnage: the report prints a TOTAL of 492,589; its rows add "
               "up to 492,587, which is used here.")


if __name__ == "__main__":
    helpers.set_readable_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table, total_row = helpers.read_table(DATA_FILE)

    # The level-1 rows aren't plotted, but checking that they add up to their
    # main rows still catches typing errors in the transcription.
    helpers.check_breakdowns(table)

    # Use the main (level 0) rows only: the level-1 rows are a breakdown of
    # them, and using both would count the same ships twice.
    main_rows = table[table["level"] == 0]
    categories = helpers.group_rows(main_rows, "sinking_agent", CATEGORY_LABELS,
                                    list(COLOURS))

    # The categories must add up to the TOTAL row (30 warships, 156 merchant
    # ships, and their tonnages), apart from the known 2-ton error.
    helpers.check_totals(categories, total_row, KNOWN_SOURCE_ERRORS)
    print(categories)

    helpers.plot_all(
        categories, COLOURS,
        "Japanese ships lost to shared credits, accidents, own mines and unknown causes",
        SOURCE_NOTE, OUT_DIR, "table_IV")

    # --- Detailed categories inside the four groups. -----------------------
    # detail_rows() takes the breakdown rows where a main row has them, and
    # the main row itself where it doesn't (accidents, Japan's own mines),
    # so every ship is counted once. The details must add up to exactly the
    # four categories plotted above, or the script stops.
    details = helpers.group_details(helpers.detail_rows(table), DETAIL_CATEGORIES,
                                    list(COLOURS), key=helpers.parent_key)
    helpers.check_details_match_groups(details, categories)
    print(details)

    helpers.plot_detail_all(
        details, COLOURS,
        "Japanese ships lost to shared credits, accidents, own mines and unknown causes, in detail",
        SOURCE_NOTE, OUT_DIR, "table_IV")

    print(f"\nSaved 8 figures to {OUT_DIR}")
