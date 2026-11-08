"""
Table V of the JANAC report: sinkings credited only "probably".

For most sinkings the committee was sure who or what sank the ship. For 17
warships and 79 merchant ships it was not, and the credit is only a
"probable" one (in the ship lists, these are marked with * after the agent).
Table V lists those uncertain credits, by force and weapon.

Important: Table V is NOT extra ships. These 96 ships are already counted in
Tables I-IV; Table V only says which of those credits are uncertain. So it
measures how reliable the other tables are, e.g. "3.1% of US-credited
merchant sinkings are only probable".

This script draws eight PNG figures from the hand-verified transcription of
Table V (tests/fixtures/table_V.csv), in figures/table_V/:

  By who was (probably) responsible:
    table_V_bar_ships.png      table_V_bar_tonnage.png
    table_V_pie_ships.png      table_V_pie_tonnage.png
  By kind of weapon:
    table_V_weapon_bar_ships.png   table_V_weapon_bar_tonnage.png
    table_V_weapon_pie_ships.png   table_V_weapon_pie_tonnage.png
  Weapon groups with their detailed weapons inside (chart 3):
    table_V_weapon_detail_bar_ships.png   table_V_weapon_detail_bar_tonnage.png
    table_V_weapon_detail_pie_ships.png   table_V_weapon_detail_pie_tonnage.png
  "Who" categories with their details inside (chart 4: the US by weapon,
  the other Allies by nation):
    table_V_detail_bar_ships.png   table_V_detail_bar_tonnage.png
    table_V_detail_pie_ships.png   table_V_detail_pie_tonnage.png

Every figure has three panels: Naval ships, Merchant ships and All ships.
Each bar / slice shows the raw number and its percentage of the panel total.
The plotting itself is shared with Tables III and IV and lives in
table_plot_helpers.py; this script decides which categories to plot and why.

Table V has no printed TOTAL row, unlike Tables I-IV. Its totals (17 naval,
79 merchant) come from adding up its rows. The check that the transcription
is consistent is that each force's breakdown adds up to that force's row
(helpers.check_breakdowns).

Run:
    python table_5_plots.py
"""

from pathlib import Path

import table_plot_helpers as helpers
from table_plot_helpers import AQUA, BLUE, GREEN, GREY, ORANGE, PINK, VIOLET, YELLOW

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Built relative to this file, so the script works from any folder.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "tests" / "fixtures" / "table_V.csv"
OUT_DIR = PROJECT_DIR / "figures" / "table_V"

# ---------------------------------------------------------------------------
# Chart 1: by who was (probably) responsible
# ---------------------------------------------------------------------------
# Table V has 8 main (level 0) rows. Four of them are small Allied rows of
# 1-3 ships each (British forces, Netherlands aircraft, Russian aircraft,
# Australian mines). They are merged into "Other Allies", matching the
# grouping of Table I, so the two sets of charts can be compared directly.
# Left: the report's wording; right: the chart label.
WHO_LABELS = {
    "United States Forces": "United States",
    "British Forces": "Other Allies",
    "Netherlands Aircraft": "Other Allies",
    "Russian Aircraft": "Other Allies",
    "Australian Mines": "Other Allies",
    "Unknown Aircraft": "Unknown aircraft",
    "Marine Casualties": "Accidents (marine casualty)",
    "Japanese Mines": "Japan's own mines",
}

# Colours. United States, Other Allies, Accidents and Japan's own mines keep
# their Table I colours (blue, orange, yellow, pink). "Unknown aircraft" gets
# violet rather than Table I's green for "Unknown": in this set green would
# sit next to orange, and the colour-blindness validator failed that pair
# (red-weak readers can't tell them apart). The order below passed.
# The dictionary order is the drawing order.
WHO_COLOURS = {
    "United States": BLUE,
    "Other Allies": ORANGE,
    "Unknown aircraft": VIOLET,
    "Accidents (marine casualty)": YELLOW,
    "Japan's own mines": PINK,
}

# ---------------------------------------------------------------------------
# Chart 2: by kind of weapon
# ---------------------------------------------------------------------------
# The weapon is named on the detailed rows (e.g. "Submarines" under "United
# States Forces") or in the name of main rows with no breakdown (e.g.
# "Netherlands Aircraft"). helpers.detail_rows() picks exactly those rows, so
# each ship is counted once.
#
# Choices made here:
#   - All aircraft go into one "Aircraft" group, as in the Table III weapon
#     charts; the numbers are too small (1-4 ships) to split them.
#   - Mines laid by the Allies (US Army and Navy mines, British and
#     Australian mines) are "Allied mines". Japan's own mines are kept out of
#     that group: they are not an Allied weapon, and mixing them in would
#     inflate the Allied mining record. They go with accidents into
#     "Non-combat (own mines, accidents)", the losses Japan suffered without
#     an enemy attacking.
#   - "Unknown Aircraft" appears twice in the CSV (as a breakdown row under
#     US forces, and as its own main row); both are aircraft, so the one
#     mapping entry covers both.
WEAPON_GROUPS = {
    "Submarines": "Submarines",
    "Surface Craft": "Surface warships",
    "Army Aircraft": "Aircraft",
    "Navy Carrier-Based Aircraft": "Aircraft",
    "Navy-Marine Land-Based Aircraft": "Aircraft",
    "Unknown Aircraft": "Aircraft",
    "Netherlands Aircraft": "Aircraft",
    "Russian Aircraft": "Aircraft",
    "Army Mine": "Allied mines",
    "Navy Mine": "Allied mines",
    "Mines": "Allied mines",
    "Australian Mines": "Allied mines",
    "Japanese Mines": "Non-combat (own mines, accidents)",
    "Marine Casualties": "Non-combat (own mines, accidents)",
}

# Same colours as the weapon groups of Table III (submarines blue, surface
# warships orange, aircraft aqua, mines green), so the figures can be
# compared. Grey marks the non-combat group as "not a weapon". This set
# passed the colour-blindness validator.
WEAPON_COLOURS = {
    "Submarines": BLUE,
    "Surface warships": ORANGE,
    "Aircraft": AQUA,
    "Allied mines": GREEN,
    "Non-combat (own mines, accidents)": GREY,
}

# ---------------------------------------------------------------------------
# Chart 3: detailed weapon categories inside the broad weapon groups
# ---------------------------------------------------------------------------
# Same detailed rows as chart 2, but keeping the detail: which service's
# aircraft, whose mines, own mines vs accidents. Each entry is:
# report wording -> (broad group, detailed label). The broad group is always
# the one WEAPON_GROUPS gives, which the script checks.
#
# Choices made here:
#   - "Netherlands Aircraft" and "Russian Aircraft" don't say which service
#     flew them, so they are "Aircraft, type not given".
#   - British "Mines" and "Australian Mines" are merged into one detail
#     (1 ship each), as the report doesn't say how either was laid.
#   - US "Army Mine" sinkings were from mines dropped by Army Air Forces
#     bombers (see Table II), hence the label.
NON_COMBAT = "Non-combat (own mines, accidents)"
DETAIL_CATEGORIES = {
    "Submarines": ("Submarines", "Submarines"),
    "Surface Craft": ("Surface warships", "Surface warships"),
    "Army Aircraft": ("Aircraft", "Army aircraft"),
    "Navy Carrier-Based Aircraft": ("Aircraft", "Navy carrier-based aircraft"),
    "Navy-Marine Land-Based Aircraft": ("Aircraft", "Navy & Marine land-based aircraft"),
    "Unknown Aircraft": ("Aircraft", "Unidentified aircraft"),
    "Netherlands Aircraft": ("Aircraft", "Aircraft, type not given"),
    "Russian Aircraft": ("Aircraft", "Aircraft, type not given"),
    "Army Mine": ("Allied mines", "US Army mines (air-dropped)"),
    "Navy Mine": ("Allied mines", "US Navy mines"),
    "Mines": ("Allied mines", "British & Australian mines"),
    "Australian Mines": ("Allied mines", "British & Australian mines"),
    "Japanese Mines": (NON_COMBAT, "Japan's own mines"),
    "Marine Casualties": (NON_COMBAT, "Accidents (marine casualty)"),
}

# ---------------------------------------------------------------------------
# Chart 4: detail inside the "who" categories of chart 1
# ---------------------------------------------------------------------------
# Chart 3 above already splits the *weapon* groups into detail. This one
# splits the *who* categories of chart 1: the United States by weapon, and
# "Other Allies" by nation (and weapon, where the report gives it).
#
# "Submarines" appears under both "United States Forces" and "British
# Forces", so rows are looked up by "main row / row" (helpers.parent_key)
# to keep US and British submarines apart. Each entry is:
# that key -> (who category from chart 1, detailed label). The script checks
# that each detail's category is the one WHO_LABELS gives its main row.
#
# Unknown aircraft, accidents and Japan's own mines have no breakdown, so
# each is a single detail with the same name as its category; the charts
# then show it as one bar / slice without a header row.
_US = "United States Forces / "
DETAIL_WHO = {
    _US + "Submarines": ("United States", "Submarines"),
    _US + "Surface Craft": ("United States", "Surface warships"),
    _US + "Army Aircraft": ("United States", "Army aircraft"),
    _US + "Navy Carrier-Based Aircraft": ("United States", "Navy carrier-based aircraft"),
    _US + "Navy-Marine Land-Based Aircraft": ("United States", "Navy & Marine land-based aircraft"),
    _US + "Unknown Aircraft": ("United States", "Unidentified US aircraft"),
    _US + "Army Mine": ("United States", "Army mines (air-dropped)"),
    _US + "Navy Mine": ("United States", "Navy mines"),
    "British Forces / Submarines": ("Other Allies", "British submarines"),
    "British Forces / Mines": ("Other Allies", "British mines"),
    "Netherlands Aircraft / Netherlands Aircraft": ("Other Allies", "Dutch aircraft"),
    "Russian Aircraft / Russian Aircraft": ("Other Allies", "Soviet aircraft"),
    "Australian Mines / Australian Mines": ("Other Allies", "Australian mines"),
    "Unknown Aircraft / Unknown Aircraft": ("Unknown aircraft", "Unknown aircraft"),
    "Marine Casualties / Marine Casualties":
        ("Accidents (marine casualty)", "Accidents (marine casualty)"),
    "Japanese Mines / Japanese Mines": ("Japan's own mines", "Japan's own mines"),
}

SOURCE_NOTE = ("Source: JANAC (1947), Japanese Naval and Merchant Shipping Losses "
               "During World War II by All Causes, Table V. These ships are already "
               "counted in Tables I-IV; Table V marks which credits are only probable.\n"
               "Percentages are of each panel's total. Table V prints no TOTAL row; "
               "totals are the sum of its rows.")


if __name__ == "__main__":
    helpers.set_readable_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # total_row is None here: Table V has no TOTAL row.
    table, _ = helpers.read_table(DATA_FILE)

    # With no TOTAL row to compare against, this is the main consistency
    # check: the breakdowns under US and British forces must add up to their
    # main rows. The script stops if they don't.
    helpers.check_breakdowns(table)

    # The reference total is the sum of the main (level 0) rows: 17 naval,
    # 79 merchant ships. Both groupings below are checked against it, so
    # merging rows can't lose or add a ship.
    main_rows = table[table["level"] == 0]
    reference_total = main_rows[helpers.VALUE_COLUMNS].sum()

    # --- By who was responsible: main rows only. --------------------------
    by_who = helpers.group_rows(main_rows, "sinking_agent", WHO_LABELS,
                                list(WHO_COLOURS))
    helpers.check_totals(by_who, reference_total)
    print(by_who, "\n")

    helpers.plot_all(
        by_who, WHO_COLOURS,
        "Japanese ships whose sinking credit is only probable, by who",
        SOURCE_NOTE + "  Other Allies = British (3 ships), Soviet aircraft (3), "
        "Dutch aircraft (1), Australian mines (1).",
        OUT_DIR, "table_V")

    # --- By kind of weapon: most detailed rows. ---------------------------
    by_weapon = helpers.group_rows(helpers.detail_rows(table), "sinking_agent",
                                   WEAPON_GROUPS, list(WEAPON_COLOURS))
    helpers.check_totals(by_weapon, reference_total)
    print(by_weapon)

    helpers.plot_all(
        by_weapon, WEAPON_COLOURS,
        "Japanese ships whose sinking credit is only probable, by kind of weapon",
        SOURCE_NOTE + "  Allied mines = US Army/Navy, British and Australian mines.",
        OUT_DIR, "table_V_weapon")

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
        "Japanese ships whose sinking credit is only probable, by detailed weapon",
        SOURCE_NOTE, OUT_DIR, "table_V_weapon")

    # --- Detail inside the "who" categories. --------------------------------
    # Each detail's category must be the one WHO_LABELS gives its main row
    # (the part of the key before " / "), so chart 4 can't disagree with
    # chart 1.
    for key, (who, _) in DETAIL_WHO.items():
        main_row = key.split(" / ")[0]
        if WHO_LABELS[main_row] != who:
            raise ValueError(f"'{key}' is under '{who}' here but "
                             f"'{WHO_LABELS[main_row]}' in WHO_LABELS")
    who_details = helpers.group_details(helpers.detail_rows(table), DETAIL_WHO,
                                        list(WHO_COLOURS), key=helpers.parent_key)
    helpers.check_details_match_groups(who_details, by_who)
    print(who_details)

    helpers.plot_detail_all(
        who_details, WHO_COLOURS,
        "Japanese ships whose sinking credit is only probable, by who, in detail",
        SOURCE_NOTE, OUT_DIR, "table_V")

    print(f"\nSaved 16 figures to {OUT_DIR}")
