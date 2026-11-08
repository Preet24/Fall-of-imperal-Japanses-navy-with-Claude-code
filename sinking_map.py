"""
Map of where Japanese ships were sunk in WWII, filtered by who sank them, with
what weapon, and whether they were warships or merchant ships.

Every ship in the JANAC ship lists with a printed latitude/longitude is drawn
as one dot on a map of the Pacific and Indian Oceans. Five filters choose
which ships are drawn:

    agent      - the nation credited with the sinking (us, britain, australia, ...)
    weapon     - the kind of weapon (aircraft, submarine, mine, ...)
    vessel     - naval (warships) or merchant ships
    ship_type  - the Japanese ship's type (destroyer, carrier, tanker, cargo, ...)
    min_tonnage / max_tonnage - the range of the ship's tonnage

Each filter takes one or more values (the tonnage limits take one number
each). Leave a filter out and it includes all of its options. A wrong value
stops the program with a message that names the filter and lists its valid
options.

These filters are shared with sinking_animation.py and sinking_clusters.py,
which import check_filters(), select_ships() and add_filter_arguments() from
here.

Input (built by ship_lists/naval_pages.py and ship_lists/merchant_pages.py):
    data/naval/naval_losses_15-44.csv
    data/merchant/merchant_losses_45-115.csv
    data/basemap/ne_50m_land/ne_50m_land.shp   (Natural Earth coastlines)

Output: one PNG per run in figures/maps/, named after the filters, e.g.
    figures/maps/sinkings_agent-all_weapon-all_vessel-all_type-all_tons-all.png
    figures/maps/sinkings_agent-us_weapon-submarine_vessel-merchant_type-tanker_tons-all.png
    figures/maps/sinkings_agent-all_weapon-all_vessel-all_type-all_tons-5000-up.png

Run from the command line:
    python sinking_map.py                                    # everything
    python sinking_map.py --agent us --weapon aircraft       # US aircraft, both fleets
    python sinking_map.py --agent britain netherlands --vessel naval
    python sinking_map.py --ship-type destroyer cruiser      # two ship types
    python sinking_map.py --min-tonnage 5000                 # big ships only
    python sinking_map.py --ship-type tanker --min-tonnage 1000 --max-tonnage 5000
    python sinking_map.py --options                          # list every valid value
    python sinking_map.py --all-combinations                 # every combination, no empty maps

or from Python:
    from sinking_map import make_map, make_all_maps
    make_map(agent="australia", weapon=["mine", "aircraft"])
    make_map(ship_type="carrier", min_tonnage=10000)
    saved, skipped = make_all_maps()

--all-combinations / make_all_maps() draws one map for each single agent (or
all agents) x each single weapon (or all) x each vessel kind (or both): 297
combinations. Combinations with no ship to draw are skipped, not saved blank.
--ship-type and the tonnage limits are not looped over; if given, they apply
to every map of the run.
"""

import argparse
import functools
import math
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no window is needed
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from shapely.geometry import box

import table_plot_helpers as helpers
from table_plot_helpers import BACKGROUND, BLUE, GRID, INK, INK_SOFT, ORANGE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Built relative to this file, so the script works from any folder.
PROJECT_DIR = Path(__file__).resolve().parent
NAVAL_FILE = PROJECT_DIR / "data" / "naval" / "naval_losses_15-44.csv"
MERCHANT_FILE = PROJECT_DIR / "data" / "merchant" / "merchant_losses_45-115.csv"
LAND_FILE = PROJECT_DIR / "data" / "basemap" / "ne_50m_land" / "ne_50m_land.shp"
LAND_URL = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
OUT_DIR = PROJECT_DIR / "figures" / "maps"

# ---------------------------------------------------------------------------
# Filter options
# ---------------------------------------------------------------------------
# Each filter has a fixed list of options. The dictionary order is the order
# the options are listed in error messages and joined in file names, so the
# same selection always gives the same file name.

# Agent: the report's "Flag of Agent" -> the option name you type.
# Japan is an option because the report credits some losses to Japan itself:
# accidents (marine casualties) and ships that hit Japan's own minefields.
AGENT_OPTIONS = {
    "us": "United States",
    "britain": "Great Britain",
    "netherlands": "Netherlands",
    "australia": "Australia",
    "new_zealand": "New Zealand",
    "india": "India",
    "china": "China",
    "russia": "Russia",
    "japan": "Japan",
    "unknown": "Unknown",
}

# Weapon: broad kinds of weapon. The report names about 15 agent types
# ("Navy Carrier-Based Aircraft", "Army Mine", ...); weapon_of() below sorts
# each one into one of these. All aircraft are one option because the
# question here is "what kind of weapon", not "which service flew it".
WEAPON_OPTIONS = {
    "aircraft": "Aircraft (carrier, land-based, Army, unidentified)",
    "submarine": "Submarines",
    "surface": "Surface warships",
    "mine": "Mines (Allied or Japanese)",
    "shore_battery": "Shore batteries",
    "sabotage": "Sabotage",
    "accident": "Accidents (marine casualty)",
    "unknown": "Cause unknown",
}

VESSEL_OPTIONS = {
    "naval": "Naval vessels (warships)",
    "merchant": "Merchant vessels",
}

# Ship type: the Japanese ship's own type, from the report's "Type of Vessel"
# column. The report uses about 50 types, many with only 1-5 ships (one Tug,
# one District Craft, ...). One option per report type would give a very long
# option list and many near-empty maps, so the types are grouped into the
# classes below. The description lists the report types in each class.
# Naval classes come first, then merchant ones, matching the vessel filter.
SHIP_TYPE_OPTIONS = {
    "battleship": "Battleships (naval)",
    "carrier": "Aircraft carriers: fleet, light and escort (naval)",
    "cruiser": "Cruisers: heavy, light, training (naval)",
    "destroyer": "Destroyers, including old destroyers (naval)",
    "submarine": "Submarines (naval)",
    "escort": "Escorts: frigates, submarine chasers, torpedo boats, gunboats (naval)",
    "mine_warfare": "Minesweepers and minelayers (naval)",
    "landing_ship": "Tank landing ships and high-speed transports (naval)",
    "naval_auxiliary": "Naval auxiliaries: repair, seaplane tender, salvage, survey, "
                       "tug, yacht, ... (naval)",
    "cargo": "Cargo ships and colliers (merchant)",
    "passenger_cargo": "Passenger-cargo ships (merchant)",
    "tanker": "Tankers (merchant)",
    "transport": "Transports and ferries (merchant)",
    "converted_auxiliary": "Merchant ships taken into navy service: converted gunboats, "
                           "seaplane tenders, cruisers, aircraft transports, ... (merchant)",
    "other_merchant": "Other merchant craft: salvage, dredges, trawler, unknown (merchant)",
}

# Report type -> ship-type option, kept apart for the two lists because a few
# names appear in both with different meanings. For example "Aircraft
# Carrier" in the naval list is a real carrier (e.g. AKAGI), but in the
# merchant list it is a merchant ship carrying aircraft (e.g. KAMOGAWA MARU),
# so it goes to converted_auxiliary. In the same way the merchant "Motor
# Torpedo Boat" entries are 1,800-4,900-ton tenders for torpedo boats, not
# torpedo boats.
SHIP_TYPE_OF = {
    "naval": {
        "Battleship": "battleship",
        "Aircraft Carrier": "carrier", "Light Aircraft Carrier": "carrier",
        "Escort Aircraft Carrier": "carrier",
        "Heavy Cruiser": "cruiser", "Light Cruiser": "cruiser",
        "Training Cruiser": "cruiser",
        "Destroyer": "destroyer", "Old Destroyer": "destroyer",
        "Submarine": "submarine",
        "Frigate": "escort", "Submarine Chaser": "escort",
        "Special Submarine Chaser": "escort", "Torpedo Boat": "escort",
        "Gunboat": "escort", "River Gunboat": "escort",
        "Minesweeper": "mine_warfare", "Special Minesweeper": "mine_warfare",
        "Minelayer": "mine_warfare", "Special Minelayer": "mine_warfare",
        "Tank Landing Ship": "landing_ship", "High Speed Transport": "landing_ship",
        "Repair Ship": "naval_auxiliary", "Seaplane Tender": "naval_auxiliary",
        "Salvage Vessel": "naval_auxiliary", "Surveying Ship": "naval_auxiliary",
        "Tug": "naval_auxiliary", "Submarine Tender": "naval_auxiliary",
        "District Craft": "naval_auxiliary", "Miscellaneous Auxiliary": "naval_auxiliary",
        "Yacht": "naval_auxiliary",
    },
    "merchant": {
        "Cargo": "cargo", "CARGO": "cargo", "Collier": "cargo", "Supply Ship": "cargo",
        "Passenger-Cargo": "passenger_cargo",
        "Tanker": "tanker",
        "Transport": "transport", "Ferry": "transport",
        "Converted Gunboat": "converted_auxiliary",
        "Converted Seaplane Tender": "converted_auxiliary",
        "Converted Light Cruiser": "converted_auxiliary",
        "Converted Minesweeper": "converted_auxiliary",
        "Converted Water carrier": "converted_auxiliary",
        "Converted Aircraft Ferry": "converted_auxiliary",
        "Aircraft Carrier": "converted_auxiliary",
        "Motor Torpedo Boat": "converted_auxiliary",
        "Submarine Chaser": "converted_auxiliary",
        "Salvage Vessel": "other_merchant", "Dredge": "other_merchant",
        "Trawler": "other_merchant",
    },
}

# Other spellings people are likely to type. They are matched to the option
# on the right, so "--weapon aeroplane" works the same as "--weapon aircraft".
# Error messages list only the official options above, to keep them short.
ALIASES = {
    "agent": {"usa": "us", "united_states": "us", "america": "us",
              "uk": "britain", "great_britain": "britain", "british": "britain",
              "dutch": "netherlands", "australian": "australia",
              "nz": "new_zealand", "soviet": "russia", "ussr": "russia"},
    "weapon": {"aeroplane": "aircraft", "airplane": "aircraft", "plane": "aircraft",
               "submarines": "submarine", "surface_craft": "surface",
               "warship": "surface", "mines": "mine", "marine_casualty": "accident"},
    "vessel": {"navy": "naval", "warship": "naval", "merchant_ship": "merchant"},
    "ship_type": {"battleships": "battleship", "aircraft_carrier": "carrier",
                  "carriers": "carrier", "cruisers": "cruiser",
                  "destroyers": "destroyer", "submarines": "submarine", "sub": "submarine",
                  "frigate": "escort", "escorts": "escort", "submarine_chaser": "escort",
                  "minesweeper": "mine_warfare", "minelayer": "mine_warfare",
                  "mine": "mine_warfare", "landing": "landing_ship",
                  "auxiliary": "naval_auxiliary", "freighter": "cargo",
                  "passenger": "passenger_cargo", "tankers": "tanker", "oiler": "tanker",
                  "transports": "transport", "converted": "converted_auxiliary",
                  "other": "other_merchant"},
}

# The filters that take a list of named options. The tonnage filter is a
# number range, not a list, so it is checked separately (check_tonnage).
OPTIONS = {"agent": AGENT_OPTIONS, "weapon": WEAPON_OPTIONS, "vessel": VESSEL_OPTIONS,
           "ship_type": SHIP_TYPE_OPTIONS}

# ---------------------------------------------------------------------------
# OCR fixes needed for filtering
# ---------------------------------------------------------------------------
# A few agent cells in the ship lists are OCR misreads. They are fixed here so
# the ships land in the right filter. Each was checked against the page image.
# (They should eventually move into the MANUAL_FIXES of the ship-list scripts;
# until then this keeps the map correct.)
FLAG_FIXES = {"States": "United States"}       # "United" was cut off (2 rows)
AGENT_FIXES = {
    "Lirorart": "Aircraft",                    # SPECIAL SUBMARINE CHASER NO 23, p27
    "l'ine": "Mine",                           # MORIOKA MARU, p46
    "Unkmowh": "Unknown",                      # SHINKO MARU, p109
    "Unknowm Aircraft": "Unknown Aircraft",
    "Unknow Aircraft": "Unknown Aircraft",
    "Driknown Aircraft": "Unknown Aircraft",   # TATSUSUGI MARU, p115
}
# Wrapped names like "Navy Land-" / "Based Aircraft" were sometimes joined with
# "; ", which would make them look like two agents. Glue them back first.
WRAP_FIXES = {"Land; Based": "Land-Based", "Carrier; Based": "Carrier-Based"}
# KINKAI MARU (p50): the agent cell came out empty; the scan says Submarine.
ROW_AGENT_FIXES = {(50, "KINKAI MARU"): "Submarine"}

# ---------------------------------------------------------------------------
# Map layout
# ---------------------------------------------------------------------------
# Longitudes are drawn from 0 to 360 instead of -180 to 180, so the Pacific is
# in one piece: Hawaii (155 W) becomes 205 and sits east of Japan, where it
# belongs, instead of jumping to the far left edge.
#
# The window is fixed (not fitted to each selection) so maps of different
# selections can be laid side by side and compared. It covers every position
# in the report from the Bay of Bengal (83 E) to Alaska (135 W = 225), and
# from 30 S to 60 N. Only two ships fall outside: RO 501 and I 52, sunk in the
# Atlantic on their way to and from Germany. They are counted in the footnote.
LON_RANGE = (70, 232)
LAT_RANGE = (-30, 60)
# Dot colour shows the kind of vessel (the third filter). Blue and orange are
# the first two colours of the project palette and pass the colour-blindness
# check next to each other.
VESSEL_COLOURS = {"naval": BLUE, "merchant": ORANGE}
LAND_COLOUR = "#e8e6df"   # a quiet grey-beige, so the dots stand out
SEA_COLOUR = BACKGROUND   # the figure background: water is the "empty" area
# Dot size (points^2) and opacity depend on how many ships are drawn. About
# 3,000 dots overlap heavily around Japan and the Philippines, so small
# half-transparent dots let dense areas show as darker patches instead of one
# solid blob. A selection of a few dozen ships doesn't overlap, and the same
# small faint dots would be hard to find, so it gets bigger, solid dots.
# Each entry: (up to this many ships, dot size, opacity).
DOT_STYLES = [(100, 45, 0.9), (500, 25, 0.75), (float("inf"), 14, 0.6)]


class OptionError(ValueError):
    """Raised when a filter value is not one of that filter's options."""


# ---------------------------------------------------------------------------
# Checking the filters
# ---------------------------------------------------------------------------
def normalise(value):
    """Lower-case a typed value and use underscores, e.g. 'New Zealand' -> 'new_zealand'.

    This lets people type options the way they read them, in any case, with
    spaces or hyphens.
    """
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def check_tonnage(name, value):
    """Turn a min/max tonnage into a number, or return (None, problem text).

    None means "no limit". Text such as "5,000" (typed with a thousands
    comma) is accepted, because that is how the report prints tonnages.
    Returns (number, None) when the value is fine and (None, message) when
    it is not, so check_filters() can report it together with the others.
    """
    if value is None:
        return None, None
    try:
        number = float(str(value).replace(",", ""))
    except ValueError:
        number = None
    if number is None or math.isnan(number) or number < 0:
        return None, (f"  '{value}' is not a valid {name}. "
                      f"Options for {name}: a number of tons, 0 or more (e.g. 1000)")
    return number, None


def check_filters(agent=None, weapon=None, vessel=None, ship_type=None,
                  min_tonnage=None, max_tonnage=None):
    """Turn each filter into a list of valid options, or raise OptionError.

    A filter that is None or empty means "all options". A single string is
    treated as a one-item list. Every filter is checked before raising, so
    one message lists all the wrong values at once, each with the filter's
    name and its valid options, instead of making you fix them one by one.

    min_tonnage and max_tonnage are numbers (None = no limit). They are
    returned together as chosen["tonnage"] = (min, max).
    """
    filters = {"agent": agent, "weapon": weapon, "vessel": vessel, "ship_type": ship_type}
    chosen, problems = {}, []
    low, problem_low = check_tonnage("min_tonnage", min_tonnage)
    high, problem_high = check_tonnage("max_tonnage", max_tonnage)
    problems += [p for p in (problem_low, problem_high) if p]
    if low is not None and high is not None and low > high:
        problems.append(f"  min_tonnage ({low:,.0f}) is larger than max_tonnage "
                        f"({high:,.0f}); no ship could match.")
    chosen["tonnage"] = (low, high)
    for name, values in filters.items():
        options = OPTIONS[name]
        if values is None or (not isinstance(values, str) and len(values) == 0):
            chosen[name] = list(options)
            continue
        if isinstance(values, str):
            values = [values]
        picked = []
        for value in values:
            key = normalise(value)
            key = ALIASES[name].get(key, key)
            if key in options:
                picked.append(key)
            else:
                problems.append(f"  '{value}' is not a valid {name}. "
                                f"Options for {name}: {', '.join(options)}")
        # Keep the official option order, without repeats, so the file name
        # is the same whatever order the values were typed in.
        chosen[name] = [key for key in options if key in picked]
    if problems:
        raise OptionError("Wrong filter value(s):\n" + "\n".join(problems)
                          + "\nLeave a filter out to include all of its options.")
    return chosen


def describe_options():
    """Return a readable list of every filter and its options (for --options)."""
    lines = []
    for name, options in OPTIONS.items():
        lines.append(f"{name}:")
        lines += [f"  {key:<20} {label}" for key, label in options.items()]
    lines += ["min_tonnage / max_tonnage:",
              "  a number of tons, 0 or more; leave out for no limit. Naval ships are",
              "  measured in standard tons and merchant ships in gross register tons."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading the ships
# ---------------------------------------------------------------------------
def weapon_of(agent_type):
    """Sort one of the report's agent types into a broad weapon option.

    The report's wording is matched by keywords, in this order:
      - "Marine Casualty" first: it means an accident, not the Marines.
      - "Aircraft" before "Unknown", so "Unknown Aircraft" counts as aircraft:
        it is known to be an aircraft, just not whose.
      - "Mine" covers Army, Navy, Allied and Japanese mines.
    An agent type that matches nothing stops the program, so a new spelling
    can never drop ships from the map without anyone noticing.
    """
    if agent_type == "Marine Casualty":
        return "accident"
    for keyword, weapon in [("Aircraft", "aircraft"), ("Submarine", "submarine"),
                            ("Surface Craft", "surface"), ("Mine", "mine"),
                            ("Shore", "shore_battery"), ("Sabotage", "sabotage"),
                            ("Unknown", "unknown")]:
        if keyword in agent_type:
            return weapon
    raise ValueError(f"Don't know which weapon '{agent_type}' is; add it to weapon_of()")


def ship_type_of(row):
    """Sort one ship's report type into a ship-type option (see SHIP_TYPE_OF).

    The lookup uses the vessel kind as well as the type, because a few type
    names mean different ships in the naval and merchant lists. KENKOKU (p54)
    has no type printed; it is put in other_merchant. Any other type that is
    not in SHIP_TYPE_OF stops the program, so a new spelling can never drop
    ships silently.
    """
    if pd.isna(row["vessel_type"]):
        return "other_merchant"
    try:
        return SHIP_TYPE_OF[row["vessel"]][row["vessel_type"]]
    except KeyError:
        raise ValueError(f"Don't know the ship type of {row['vessel']} type "
                         f"'{row['vessel_type']}'; add it to SHIP_TYPE_OF") from None


def credits_of(row):
    """List the (agent, weapon) pairs credited with sinking one ship.

    Most ships have one agent. Shared sinkings list several, e.g. flag
    "United States; Australia" with type "Army Aircraft; Aircraft": the two
    lists run in parallel, so the pairs are (us, aircraft) and (australia,
    aircraft). When only one flag is printed it applies to every weapon, e.g.
    MIKUMA: "United States" with carrier aircraft and Marine land-based
    aircraft.

    Keeping pairs (not two separate lists) matters for filtering: a ship sunk
    by US aircraft plus a British submarine must NOT match "British aircraft".
    """
    flags = FLAG_FIXES.get(row["flag_of_agent"], row["flag_of_agent"])
    agents = ROW_AGENT_FIXES.get((row["page"], row["vessel_name"]), row["agent_type"])
    # KENKOKU (p54) has no agent at all in the report; it has no position
    # either, so it never reaches the map, but it still needs a value here.
    flags = "Unknown" if pd.isna(flags) else flags
    agents = "Unknown" if pd.isna(agents) else agents
    for broken, fixed in WRAP_FIXES.items():
        agents = agents.replace(broken, fixed)

    flag_list = flags.split("; ")
    agent_list = [AGENT_FIXES.get(a, a) for a in agents.split("; ")]
    if len(flag_list) == 1:
        flag_list = flag_list * len(agent_list)
    if len(flag_list) != len(agent_list):
        raise ValueError(f"{row['vessel_name']} (p{row['page']}): "
                         f"{len(flag_list)} flags but {len(agent_list)} agent types")

    flag_to_key = {flag: key for key, flag in AGENT_OPTIONS.items()}
    return [(flag_to_key[flag], weapon_of(agent))
            for flag, agent in zip(flag_list, agent_list)]


def load_ships():
    """Read both ship lists into one table with 'vessel', 'ship_type' and 'credits' columns.

    Longitudes west of 180 (negative numbers) get 360 added, to match the
    0-360 map window (see LON_RANGE). Ships without a parsed position (38 in
    all, whose location is printed as text such as "off Japan" or "Yellow
    Sea") are kept, so the footnote can say how many of the selected ships
    could not be drawn.
    """
    naval = pd.read_csv(NAVAL_FILE).assign(vessel="naval")
    merchant = pd.read_csv(MERCHANT_FILE).assign(vessel="merchant")
    ships = pd.concat([naval, merchant], ignore_index=True)
    ships["credits"] = ships.apply(credits_of, axis=1)
    ships["ship_type"] = ships.apply(ship_type_of, axis=1)
    ships["lon"] = ships["lon"].where(ships["lon"] >= 0, ships["lon"] + 360)
    return ships


def kinds_shown(chosen):
    """List the vessel kinds ('naval', 'merchant') that can appear in the selection.

    A kind is shown when it is chosen in the vessel filter AND at least one
    chosen ship type belongs to it. Without the second test, picking
    --ship-type destroyer would still put "Merchant vessels: 0" in the
    legend, although no destroyer can ever be a merchant ship. A kind that is
    possible but has no ship (e.g. agent india, merchant) is still shown
    with 0, because that zero is real information.
    """
    return [kind for kind in chosen["vessel"]
            if set(SHIP_TYPE_OF[kind].values()) & set(chosen["ship_type"])]


def select_ships(ships, chosen):
    """Keep the ships that match all the filters.

    A ship matches when it is of a chosen vessel kind AND of a chosen ship
    type AND its tonnage is inside the chosen range AND at least one of its
    (agent, weapon) credits has a chosen agent together with a chosen weapon.
    So a ship sunk jointly by US and Australian aircraft appears both on a
    "us" map and on an "australia" map: each nation did take part.

    The tonnage limits include their end points (min <= tonnage <= max). Each
    ship is compared using its own printed number: standard tons for naval
    ships, gross register tons for merchant ships. The two units are not the
    same, so "1,000 to 5,000 tons" picks slightly different sizes of ship in
    the two fleets. Estimated tonnages (printed with an "e") are used as
    printed.
    """
    agents, weapons = set(chosen["agent"]), set(chosen["weapon"])
    credit_matches = ships["credits"].apply(
        lambda credits: any(a in agents and w in weapons for a, w in credits))
    low, high = chosen["tonnage"]
    in_range = ships["tonnage"].between(-math.inf if low is None else low,
                                        math.inf if high is None else high)
    # Every ship has a tonnage today, but a ship without one should still be
    # kept when no tonnage limit was asked for (the default "all").
    if low is None and high is None:
        in_range = in_range | ships["tonnage"].isna()
    return ships[credit_matches & in_range
                 & ships["vessel"].isin(chosen["vessel"])
                 & ships["ship_type"].isin(chosen["ship_type"])]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_land():
    """Read the Natural Earth land polygons and shift them to the 0-360 window.

    The shapefile uses -180..180. The western half (America, and the Pacific
    islands east of the date line) is cut off and moved 360 degrees east, so
    it lines up with the shifted ship positions.

    lru_cache keeps the result after the first call, so drawing hundreds of
    maps (make_all_maps) reads and clips the shapefile only once.
    """
    if not LAND_FILE.exists():
        raise FileNotFoundError(
            f"Basemap not found: {LAND_FILE}\nDownload {LAND_URL} and unzip it "
            f"into {LAND_FILE.parent}")
    land = gpd.read_file(LAND_FILE)
    east = land.clip(box(0, -90, 180, 90))
    west = land.clip(box(-180, -90, 0, 90)).translate(xoff=360)
    shifted = gpd.GeoSeries(pd.concat([east.geometry, west], ignore_index=True))
    # Keep only the land inside the map window (plus a margin, so coastlines
    # don't stop visibly at the edge). Europe, Africa and the Americas would
    # be drawn off-screen for nothing; cutting them makes each draw much
    # faster, which matters for the 1,200 frames of sinking_animation.py.
    window = box(LON_RANGE[0] - 5, LAT_RANGE[0] - 5, LON_RANGE[1] + 5, LAT_RANGE[1] + 5)
    return shifted.clip(window)


def degree_label(value, axis):
    """Format a tick as 120°E / 160°W / 20°N / 10°S.

    Longitudes above 180 are on the 0-360 scale, so 200 is printed as 160°W.
    """
    if axis == "lon":
        value = value - 360 if value > 180 else value
        hemisphere = "E" if 0 < value < 180 else "W" if value < 0 else ""
    else:
        hemisphere = "N" if value > 0 else "S" if value < 0 else ""
    return f"{abs(value):g}°{hemisphere}"


def map_aspect_ratio():
    """Width / height of the map window once it is drawn (see style_map_axes)."""
    stretch = 1 / math.cos(math.radians(sum(LAT_RANGE) / 2))
    return (LON_RANGE[1] - LON_RANGE[0]) / ((LAT_RANGE[1] - LAT_RANGE[0]) * stretch)


def style_map_axes(ax):
    """Draw the land and set up the map window, ticks and grid on ax.

    Shared by the still maps here and by sinking_animation.py, so both show
    exactly the same map.

    The map is equal-angle (plate carrée), stretched by 1/cos(latitude) at
    the middle of the window so shapes near the equator look about right. A
    proper projection library (cartopy) is not installed and not needed at
    this scale.
    """
    ax.set_facecolor(SEA_COLOUR)
    load_land().plot(ax=ax, color=LAND_COLOUR, edgecolor="#b9b7af", linewidth=0.4, zorder=1)
    ax.set_xlim(*LON_RANGE)
    ax.set_ylim(*LAT_RANGE)
    ax.set_aspect(1 / math.cos(math.radians(sum(LAT_RANGE) / 2)))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: degree_label(v, "lon")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: degree_label(v, "lat")))
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_color("#b9b7af")


def selection_text(chosen):
    """Say in words what was selected, e.g. 'Agent: US, Britain'.

    A filter that has every option selected is shown as 'all', which is
    shorter and clearer than listing ten nations.
    """
    parts = []
    for name, options in OPTIONS.items():
        keys = chosen[name]
        text = "all" if len(keys) == len(options) else ", ".join(keys)
        parts.append(f"{name.replace('_', ' ').capitalize()}: {text}")
    parts.append(f"Tonnage: {tonnage_text(chosen['tonnage'])}")
    return "   |   ".join(parts)


def tonnage_text(tonnage):
    """Say the tonnage range in words: 'all', '1,000–5,000', '≥ 1,000' or '≤ 500'."""
    low, high = tonnage
    if low is None and high is None:
        return "all"
    if high is None:
        return f"≥ {low:,.0f}"
    if low is None:
        return f"≤ {high:,.0f}"
    return f"{low:,.0f}–{high:,.0f}"


def tonnage_file_part(tonnage):
    """Write the tonnage range for a file name: 'all', '1000-5000', '1000-up' or 'upto-500'.

    Words are used for an open end instead of symbols such as '>=', which
    Windows does not allow in file names.
    """
    low, high = tonnage
    if low is None and high is None:
        return "all"
    if high is None:
        return f"{low:g}-up"
    if low is None:
        return f"upto-{high:g}"
    return f"{low:g}-{high:g}"


def file_name(chosen):
    """Build the PNG name from the filters.

    For example sinkings_agent-us_weapon-all_vessel-naval_type-destroyer_tons-1000-up.png.
    Several values for one filter are joined with '+'. A filter with every
    option selected is written 'all', so leaving a filter out and listing
    all of its options give the same file. "ship_type" is shortened to
    "type" and the tonnage range is written as "tons-...", to keep names short.
    """
    parts = []
    for name, options in OPTIONS.items():
        keys = chosen[name]
        label = "type" if name == "ship_type" else name
        parts.append(f"{label}-" + ("all" if len(keys) == len(options) else "+".join(keys)))
    parts.append("tons-" + tonnage_file_part(chosen["tonnage"]))
    return "sinkings_" + "_".join(parts) + ".png"


def draw_map(selected, chosen, out_path):
    """Draw the selected ships on the Pacific map and save it as a PNG.

    Merchant ships are drawn first and warships on top, because there are
    fewer warships and they would disappear under the merchant dots. Each
    dot is one ship at its printed position; the legend gives the number of
    ships drawn for each kind of vessel.
    """
    on_map = selected.dropna(subset=["lat", "lon"])
    inside = on_map["lon"].between(*LON_RANGE) & on_map["lat"].between(*LAT_RANGE)
    no_position = len(selected) - len(on_map)
    outside = int((~inside).sum())

    width = 16
    height = width / map_aspect_ratio() + 2.2   # + room for title and footnote
    fig, ax = plt.subplots(figsize=(width, height))
    style_map_axes(ax)

    # Pick the dot style for the number of ships actually drawn (DOT_STYLES).
    dot_size, dot_alpha = next((size, alpha) for limit, size, alpha in DOT_STYLES
                               if inside.sum() <= limit)
    handles = []
    for vessel in ["merchant", "naval"]:
        if vessel not in kinds_shown(chosen):
            continue
        ships = on_map[inside & (on_map["vessel"] == vessel)]
        ax.scatter(ships["lon"], ships["lat"], s=dot_size, color=VESSEL_COLOURS[vessel],
                   alpha=dot_alpha, edgecolors="white", linewidths=0.3, zorder=3)
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=9,
                              color=VESSEL_COLOURS[vessel],
                              label=f"{VESSEL_OPTIONS[vessel]}: {len(ships):,}"))
    # The legend lists warships first, the same order as the vessel filter.
    ax.legend(handles=handles[::-1], loc="lower left", frameon=True, framealpha=0.9,
              edgecolor=GRID, fontsize=12, title="Ships drawn", title_fontsize=12)

    if selected.empty:
        ax.text(sum(LON_RANGE) / 2, sum(LAT_RANGE) / 2, "No ships match this selection",
                ha="center", va="center", fontsize=18, color=INK_SOFT, zorder=4)

    ax.set_title(f"Where Japanese ships were sunk, 1941–1945\n", loc="left",
                 fontsize=18, fontweight="bold", color=INK)
    ax.text(0, 1.015, selection_text(chosen), transform=ax.transAxes,
            fontsize=13, color=INK_SOFT)

    # Footnote: say what is NOT on the map and how the filters work, so a
    # reader doesn't take the dot count as the full number of ships.
    notes = [f"Source: JANAC (1947) ship lists. {len(selected):,} ships match; "
             f"{len(on_map) - outside:,} are drawn."]
    if no_position:
        notes.append(f"{no_position} have only a text location (e.g. 'off Japan') and "
                     "can't be drawn.")
    if outside:
        notes.append(f"{outside} lie outside this map (e.g. in the Atlantic).")
    notes.append("Ships sunk jointly appear on the map of each agent and weapon involved. "
                 "Positions are OCR-read from the report; a few may be misread.")
    fig.text(0.01, 0.01, "\n".join([" ".join(notes[:-1]), notes[-1]]),
             fontsize=10, color=INK_SOFT, va="bottom")

    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.10)
    # The fixed map aspect leaves empty bands above and below the axes;
    # bbox_inches="tight" trims them while keeping the title and footnote.
    # When hundreds of maps are written in a row (make_all_maps), Windows
    # occasionally refuses to open a new file for a moment ("Errno 22
    # Invalid argument"), most likely while antivirus or the search indexer
    # is scanning the folder. The same save works a second later, so try up
    # to 3 times before giving up, instead of losing the whole batch.
    for attempt in range(3):
        try:
            fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(),
                        bbox_inches="tight", pad_inches=0.25)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)
    plt.close(fig)


def drawable_count(selected):
    """Count the selected ships that would actually appear as dots.

    A ship needs a parsed position inside the map window. A selection can
    match ships and still draw nothing, e.g. when its only ship has a text
    location such as "off Japan"; such a map is as empty as one with no match.
    """
    on_map = selected.dropna(subset=["lat", "lon"])
    return int((on_map["lon"].between(*LON_RANGE)
                & on_map["lat"].between(*LAT_RANGE)).sum())


def add_filter_arguments(parser):
    """Add the six filter options (--agent ... --max-tonnage) to an argparse parser.

    Shared by sinking_map.py, sinking_animation.py and sinking_clusters.py,
    so the filters are typed the same way in all three.

    The values are checked by check_filters(), not by argparse's own
    "choices" or type=float, so the error message is the same from the
    command line and from Python, and aliases such as "aeroplane" work.
    """
    parser.add_argument("--agent", nargs="+", help="who sank them, e.g. us australia")
    parser.add_argument("--weapon", nargs="+", help="weapon, e.g. aircraft submarine")
    parser.add_argument("--vessel", nargs="+", help="naval and/or merchant")
    parser.add_argument("--ship-type", nargs="+",
                        help="Japanese ship type, e.g. destroyer tanker (see --options)")
    parser.add_argument("--min-tonnage", help="smallest tonnage to include (default: no limit)")
    parser.add_argument("--max-tonnage", help="largest tonnage to include (default: no limit)")


def filter_arguments(args):
    """Pick the six filter values out of parsed arguments, as keyword arguments."""
    return {"agent": args.agent, "weapon": args.weapon, "vessel": args.vessel,
            "ship_type": args.ship_type, "min_tonnage": args.min_tonnage,
            "max_tonnage": args.max_tonnage}


def make_map(agent=None, weapon=None, vessel=None, ship_type=None,
             min_tonnage=None, max_tonnage=None, ships=None):
    """Check the filters, draw the map, and return the path of the saved PNG.

    agent, weapon, vessel and ship_type each take None (all options), one
    option as a string, or a list of options. min_tonnage and max_tonnage
    take a number, or None for no limit. A wrong value raises OptionError,
    which names the filter and lists its options.

    ships is the table from load_ships(). Leave it out and it is read from
    the CSVs; make_all_maps() passes it in so the CSVs are read only once
    for hundreds of maps.
    """
    chosen = check_filters(agent=agent, weapon=weapon, vessel=vessel, ship_type=ship_type,
                           min_tonnage=min_tonnage, max_tonnage=max_tonnage)
    helpers.set_readable_style()
    ships = load_ships() if ships is None else ships
    selected = select_ships(ships, chosen)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / file_name(chosen)
    draw_map(selected, chosen, out_path)
    print(f"{len(selected):,} ships match ({selection_text(chosen)})")
    print(f"Saved {out_path}")
    return out_path


def make_all_maps(include_all=True, ship_type=None, min_tonnage=None, max_tonnage=None):
    """Draw one map for every combination of agent, weapon and vessel.

    ship_type, min_tonnage and max_tonnage are NOT looped over: they are
    applied as they are to every map (default: all types, any tonnage).
    Looping over the 15 ship types as well would make about 4,800
    combinations. To get every combination for destroyers only, call
    make_all_maps(ship_type="destroyer").

    Each filter takes either one single option or, when include_all is True,
    also "all" of its options. That gives (10 + 1) agents x (8 + 1) weapons
    x (2 + 1) vessels = 297 combinations; without "all" it is 10 x 8 x 2 =
    160. Multi-value mixes such as "us + britain" are left out: there are
    thousands of them, and any one can still be drawn with make_map().

    Combinations that would draw no dots (e.g. agent india + weapon
    submarine: India's only credit is one shared surface-craft sinking) are
    skipped, not saved as blank maps. Their names are returned and printed
    so you can see what was left out and why.

    Returns (saved, skipped): the list of saved PNG paths and the list of
    skipped combinations as (agent, weapon, vessel) tuples.
    """
    # Check the fixed filters once before starting, so a wrong value stops
    # the run at once with the usual message instead of on the first map.
    check_filters(ship_type=ship_type, min_tonnage=min_tonnage, max_tonnage=max_tonnage)
    ships = load_ships()          # read the CSVs once for all maps
    helpers.set_readable_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # None means "all options" to check_filters(), so adding None to each
    # list adds the "all" choice for that filter.
    def choices(options):
        return list(options) + ([None] if include_all else [])

    saved, skipped = [], []
    for agent in choices(AGENT_OPTIONS):
        for weapon in choices(WEAPON_OPTIONS):
            for vessel in choices(VESSEL_OPTIONS):
                chosen = check_filters(agent=agent, weapon=weapon, vessel=vessel,
                                       ship_type=ship_type, min_tonnage=min_tonnage,
                                       max_tonnage=max_tonnage)
                selected = select_ships(ships, chosen)
                if drawable_count(selected) == 0:
                    skipped.append((agent or "all", weapon or "all", vessel or "all"))
                    continue
                out_path = OUT_DIR / file_name(chosen)
                draw_map(selected, chosen, out_path)
                saved.append(out_path)
                print(f"[{len(saved):3d}] {len(selected):5,} ships  {out_path.name}")

    print(f"\nSaved {len(saved)} maps to {OUT_DIR}")
    print(f"Skipped {len(skipped)} empty combinations (agent, weapon, vessel):")
    for combo in skipped:
        print("  " + ", ".join(combo))
    return saved, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map where Japanese ships were sunk, filtered by agent, weapon, "
                    "vessel kind, ship type and tonnage. Leave a filter out to include "
                    "all options.")
    add_filter_arguments(parser)   # --agent ... --max-tonnage, shared with the other scripts
    parser.add_argument("--options", action="store_true", help="list all valid values and stop")
    parser.add_argument("--all-combinations", action="store_true",
                        help="draw every agent x weapon x vessel combination, skipping "
                             "empty maps; --ship-type and the tonnage limits apply to all")
    args = parser.parse_args()

    if args.options:
        print(describe_options())
        sys.exit(0)
    try:
        if args.all_combinations:
            make_all_maps(ship_type=args.ship_type, min_tonnage=args.min_tonnage,
                          max_tonnage=args.max_tonnage)
        else:
            make_map(**filter_arguments(args))
    except OptionError as error:
        print(error, file=sys.stderr)
        sys.exit(2)
