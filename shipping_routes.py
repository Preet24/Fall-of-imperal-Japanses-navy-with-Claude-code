"""
The submarine campaign by shipping route: where Japan's merchant ships were
sunk, by what, and when each route was cut. (Analysis R)

Japan's war economy ran on sea lanes. Oil, rubber, bauxite and tin came north
from the Indies and Malaya through the South China Sea, past the Philippines
and Formosa; coal, iron ore and food came from China, Korea and Manchuria
across the Yellow Sea and the Sea of Japan; and the island garrisons of the
outer perimeter (New Guinea, the Solomons, the Carolines and Marianas) were
supplied outwards from Japan. The story to test: Allied submarines first
worked the outer perimeter, then cut the southern resource route in 1944,
and in 1945, with the south closed, the war moved onto the inner routes to
Korea and China and into Japan's own coastal waters, where aircraft and mines
took over from submarines.

Every merchant ship is put into one of 12 SEA AREAS by its printed position
(polygons below), and each area belongs to one of 4 ROUTE GROUPS:

    southern resource route   NEI & Malacca Strait, South China Sea,
                              Philippines, Luzon Strait, Formosa & Ryukyus
    inner zone (China-Korea)  East China & Yellow Sea, Sea of Japan & Korea Strait
    Japanese coastal waters   Japan's Pacific coast & Inland Sea
    outer perimeter           New Guinea & Solomons, Central Pacific,
                              North (Hokkaido, Kuriles, Aleutians), Indian Ocean

Analyses:
  R1. Losses (ships and GRT) by area and year, for all causes and for
      submarines alone, and the submarine share of each area's losses.
  R2. Did the submarines' hunting grounds shift over time? Area x year
      contingency table of submarine-sunk ships: chi-square with a
      permutation p-value, Cramer's V, adjusted residuals.
  R3. The same question as a smooth model: multinomial logistic regression
      of route group on time (a natural spline), likelihood-ratio test
      against "no change", with month-cluster bootstrap bands.
  R4. When was each route cut? The distribution of loss dates in each area
      (median with a month-cluster bootstrap interval, 10th-90th
      percentiles), Kruskal-Wallis across areas, and the difference in
      median loss date between the southern route and the inner zone.
  R5. The hand-over from submarines to aircraft and mines: logistic
      regression of "sunk by a submarine" on time x route group (standard
      errors clustered by month), with a Wald test of whether the decline
      differs between route groups.

Population: merchant ships only (2,346): these are the shipping lanes. All
sinking agents are included; 96% of submarine sinkings are by US submarines
(the rest British and Dutch), so "submarine" here is in practice the US
submarine campaign. A ship counts as "sunk by a submarine" when every one
of its credits is a submarine (shared kills between two submarines count);
ships sunk jointly by a submarine and another weapon (a few dozen) are left
out of the submarine-only tests R2-R3 and count as "not submarine" in R5. In
the weapon-share charts each credit gets an equal share of the ship (1/n,
as in us_loss_time_series.py).

Input:
    data/merchant/merchant_losses_45-115.csv   (via sinking_map.load_ships())
    data/basemap/ne_50m_land/                  (for the map)

Output:
    figures/routes/route_areas_map.png         the 12 areas and every merchant loss
    figures/routes/submarine_area_heatmap.png  submarine GRT by area and quarter
    figures/routes/submarine_area_residuals.png  R2: residuals of area x year
    figures/routes/submarine_route_shares.png  R3: modelled route-group shares
    figures/routes/route_timing.png            R4: when each area's losses happened
    figures/routes/weapon_handover.png         R5: weapon shares by route group
    data/routes/merchant_ship_areas.csv        one row per merchant ship: area, group, weapon
    data/routes/area_year_losses.csv           R1 table
    data/routes/submarine_area_year.csv        R2 counts, expected counts and residuals
    data/routes/route_share_model.csv          R3 modelled shares per month, with bands
    data/routes/area_timing.csv                R4 per-area date summary
    data/routes/route_tests.csv                every test, one row each

The method is explained in routes_tankers_explained.md.

Run:
    python shipping_routes.py
"""

import math
import time
import warnings

import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no window is needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from shapely.geometry import Point, Polygon, box

import sinking_map as smap
import table_plot_helpers as helpers
from sinking_animation import sinking_dates
from table_plot_helpers import (AQUA, BACKGROUND, BLUE, GREEN, GREY, GRID, INK, INK_SOFT,
                                ORANGE, PINK, VIOLET, YELLOW, tint)
from weapon_ship_analysis import RESIDUAL_CMAP, adjusted_residuals, chi_square, cramers_v

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
FIG_DIR = smap.PROJECT_DIR / "figures" / "routes"
DATA_DIR = smap.PROJECT_DIR / "data" / "routes"

MONTHS = pd.period_range("1941-12", "1945-08", freq="M")
WAR_START = pd.Timestamp("1941-12-07")
AVERAGE_MONTH_DAYS = 365.25 / 12
RANDOM_SEED = 12345
BOOTSTRAP_SAMPLES = 2000     # for medians (cheap)
MODEL_BOOTSTRAP_SAMPLES = 300  # for refitting the multinomial model (slower)
SPLINE_DF = 4                # flexibility of the time curve in R3 (as in us_loss_time_series.py)
ALPHA = 0.05

# ---------------------------------------------------------------------------
# Sea areas
# ---------------------------------------------------------------------------
# Each area is a polygon in (longitude, latitude), longitudes on the 0-360
# scale of sinking_map.load_ships(). Polygons may cover land: only ship
# positions are tested against them, and ships are at sea.
#
# The areas are tested IN THIS ORDER and a ship goes to the first area that
# contains it, so a later, broader box only catches what earlier ones did
# not. That keeps the definitions short (e.g. "the South China Sea is the box
# 0-23 N, 99-121 E, minus the Philippines and the Luzon Strait already
# taken"). The boundaries are judgment calls, drawn along the narrows that
# separated one leg of a route from the next; route_areas_map.png shows them.
# A ship within ~20 km of a boundary could reasonably sit on either side;
# with about 2,300 ships that affects a few dozen at most.
AREAS = [
    # The Shimonoseki (Kanmon) Strait joins the Inland Sea to the Sea of
    # Japan and was the most heavily mined spot of 1945. It is put with the
    # Inland Sea (Japan's coastal waters) before the Sea of Japan polygon
    # can claim its western mouth.
    ("japan_coast", "Japan: Pacific coast & Inland Sea",
     box(130.75, 33.75, 131.15, 34.15)),
    # Sea of Japan and Korea Strait: from the Korea Strait (south of
    # Tsushima) up the west coast of Honshu to the Tsugaru Strait, Hokkaido's
    # west coast and Sakhalin. The western edge runs down Korea (lon 127),
    # so Korea's west coast falls in the Yellow Sea. The eastern edge follows
    # Honshu's coast slightly inland.
    ("sea_of_japan", "Sea of Japan & Korea Strait",
     Polygon([(126.3, 33.3), (130.2, 33.3), (130.75, 33.75), (131.2, 34.2), (132.5, 34.7),
              (133.5, 35.3), (135.0, 35.4), (136.3, 35.9), (137.0, 36.5), (138.0, 37.0),
              (139.3, 37.7), (139.9, 38.7), (140.0, 40.0), (140.0, 41.2), (139.9, 42.3),
              (141.3, 43.0), (141.8, 44.5), (142.1, 45.6), (142.0, 55.0), (127.0, 55.0),
              (127.0, 34.6), (126.3, 34.4)])),
    # Formosa, the Formosa Strait and the Ryukyu chain up to Tanegashima, just
    # south of Kyushu. The south edge (lat 22) is Formosa's southern tip.
    ("formosa_ryukyu", "Formosa & Ryukyus",
     Polygon([(118.0, 22.0), (122.8, 22.0), (126.0, 23.5), (131.8, 28.5), (131.8, 30.3),
              (130.2, 30.6), (128.0, 28.0), (126.5, 26.8), (124.0, 25.8), (122.0, 25.8),
              (119.5, 26.0), (118.0, 24.5)])),
    # The Luzon Strait / Bashi Channel between Luzon and Formosa, the
    # submariners' "Convoy College": every southern convoy had to pass it.
    ("luzon_strait", "Luzon Strait",
     box(117.5, 18.6, 123.0, 22.0)),
    # The Philippines: Luzon to Mindanao and the Sulu archipelago, east into
    # the Philippine Sea to 127.5 E, plus Palawan in the west.
    ("philippines", "Philippines",
     box(119.3, 5.0, 127.5, 18.6).union(box(116.8, 7.5, 119.3, 12.5))),
    # Japan's Pacific coast, the Inland Sea and southern Hokkaido: from
    # southern Kyushu (30.3 N) to 42.5 N, out to 147 E (about 600 km
    # offshore). Kyushu's west coast (west of 130 E, south of 33.3 N) is left
    # to the East China Sea.
    ("japan_coast", "Japan: Pacific coast & Inland Sea",
     Polygon([(130.0, 30.3), (147.0, 30.3), (147.0, 42.5), (130.0, 42.5)])),
    # The East China Sea and Yellow Sea, with the Gulf of Bohai: the China
    # coast from Foochow north to Dairen, Korea's west coast, and the sea
    # west of Kyushu. The second box takes in the lower Yangtze up to Hankow
    # (5 river steamers sunk by aircraft near Kiukiang and Hankow): the
    # river was part of the same China supply system.
    ("yellow_ecs", "East China & Yellow Sea",
     box(116.0, 24.0, 130.0, 42.0).union(box(110.0, 28.0, 116.0, 33.0))),
    # Netherlands East Indies and the Malacca Strait: everything south of
    # 1 N from Sumatra to Halmahera, plus the Celebes Sea, North Borneo's
    # east coast and the Malacca Strait.
    ("nei_malaya", "NEI & Malacca Strait",
     box(95.0, -12.0, 131.0, 1.0).union(box(116.5, 1.0, 131.0, 5.0))
     .union(box(116.5, 5.0, 119.3, 7.5)).union(box(97.5, 1.0, 102.8, 7.0))),
    # South China Sea: Singapore to Hong Kong, Indochina, the Gulf of Siam,
    # Hainan, and Borneo's north-west coast.
    ("south_china_sea", "South China Sea",
     box(99.0, 1.0, 121.0, 24.0)),
    # New Guinea, the Bismarcks and the Solomons: south of 1 N, east of the
    # NEI, out to 165 E (east of the Solomons is the Gilberts, below).
    ("new_guinea_solomons", "New Guinea & Solomons",
     box(131.0, -15.0, 165.0, 1.0)),
    # The north: Hokkaido's Pacific and Okhotsk coasts, the Kuriles, Sakhalin's
    # east coast and the Aleutians.
    ("north", "North: Hokkaido, Kuriles, Aleutians",
     box(139.0, 42.5, 200.0, 60.0)),
    # Central Pacific: the Palaus, Carolines (Truk), Marianas, Marshalls,
    # Bonins and Wake, the open Philippine Sea, and the Gilberts and Nauru
    # just south of the equator.
    ("central_pacific", "Central Pacific islands",
     box(123.0, 1.0, 200.0, 42.5).union(box(165.0, -15.0, 200.0, 1.0))),
    # Indian Ocean: the Andaman Sea, Burma coast, Bay of Bengal and west of
    # Sumatra, south to 40 S (one ship, HOKOKU MARU, the commerce raider, was
    # sunk at 20 S, 93 E).
    ("indian_ocean", "Indian Ocean & Burma",
     box(40.0, -40.0, 99.0, 30.0)),
]


def effective_areas():
    """Each area's polygon minus every area tested before it (merged by key).

    Because a ship goes to the FIRST area that contains it, the region an
    area really covers is its own polygon minus the earlier ones. These
    non-overlapping shapes are what the map draws; assigning ships with
    them gives exactly the same result as the first-match rule.
    """
    taken = Polygon()
    shapes = {}
    for key, _, shape in AREAS:
        own = shape.difference(taken)
        shapes[key] = shapes[key].union(own) if key in shapes else own
        taken = taken.union(shape)
    return shapes
AREA_LABELS = dict((key, label) for key, label, _ in AREAS)
# Areas in the order used in tables and charts: along the routes, from the
# far south to Japan, then the outer perimeter.
AREA_ORDER = ["nei_malaya", "south_china_sea", "philippines", "luzon_strait", "formosa_ryukyu",
              "yellow_ecs", "sea_of_japan", "japan_coast",
              "new_guinea_solomons", "central_pacific", "north", "indian_ocean"]

# Route groups: the areas grouped by the job their sea lanes did.
GROUP_OF = {"nei_malaya": "southern", "south_china_sea": "southern", "philippines": "southern",
            "luzon_strait": "southern", "formosa_ryukyu": "southern",
            "yellow_ecs": "inner", "sea_of_japan": "inner",
            "japan_coast": "home",
            "new_guinea_solomons": "outer", "central_pacific": "outer", "north": "outer",
            "indian_ocean": "outer"}
GROUP_LABELS = {"southern": "Southern resource route", "inner": "Inner zone: China & Korea",
                "home": "Japanese coastal waters", "outer": "Outer perimeter"}
GROUP_ORDER = ["southern", "inner", "home", "outer"]
# Group colours. Four palette hues that pass the colour-blindness check next
# to each other (blue / orange / aqua / violet are palette neighbours). The
# fleet meaning of blue and orange in other charts does not apply here; the
# legend always says what a colour is.
GROUP_COLOURS = {"southern": ORANGE, "inner": BLUE, "home": VIOLET, "outer": AQUA}

# Text locations of the 31 merchant ships without a latitude/longitude,
# matched by keyword (after lower-casing and removing OCR noise). "off Japan"
# (4 ships) is too vague to place: Japan has coasts on three of the areas.
# NITTSU MARU's position is printed "38-05N, 124-93E" (an impossible minute);
# 124 E at 38 N is in the Yellow Sea whatever the minutes were.
TEXT_AREAS = [("east china sea", "yellow_ecs"), ("bast china sea", "yellow_ecs"),
              ("yellow sea", "yellow_ecs"), ("formosa", "formosa_ryukyu"),
              ("pormosa", "formosa_ryukyu"), ("okinawa", "formosa_ryukyu"),
              ("japan sea", "sea_of_japan"), ("okhotsk", "north"), ("okhotak", "north"),
              ("indian oc", "indian_ocean"), ("south china sea", "south_china_sea"),
              ("siam", "south_china_sea"), ("china", "yellow_ecs"),
              ("philippine", "philippines"), ("guinea", "new_guinea_solomons"),
              ("gainea", "new_guinea_solomons"), ("bonin", "central_pacific")]
TEXT_FIXES = {"NITTSU MARU": "yellow_ecs"}

# Weapons. Same grouping and colours as us_loss_time_series.py, so the same
# weapon has the same colour across the time-series and route charts.
WEAPON_ORDER = ["submarine", "aircraft", "mine", "surface", "other"]
WEAPON_LABELS = {"submarine": "Submarine", "aircraft": "Aircraft", "mine": "Mine",
                 "surface": "Surface warship", "other": "Other / unknown"}
WEAPON_COLOURS = {"submarine": VIOLET, "aircraft": AQUA, "surface": PINK, "mine": YELLOW,
                  "other": GREY}

# Sequential colour scale (one hue, light to dark) for the GRT heatmap: the
# cells show an amount, so a single hue is the honest encoding.
HEAT_CMAP = LinearSegmentedColormap.from_list("heat", ["#f4f3ee", tint(VIOLET, 0.35), VIOLET])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def area_of_position(lon, lat):
    """Return the first area (in AREAS order) whose polygon contains the position.

    Shapely's covers() is used rather than contains() so a ship exactly on a
    boundary line (positions are printed to the minute, so ties happen)
    still gets an area. Returns None for a position outside every polygon.
    """
    point = Point(lon, lat)
    for key, _, shape in AREAS:
        if shape.covers(point):
            return key
    return None


def area_of_text(name, location):
    """Place a ship without a position by keywords in its text location (see TEXT_AREAS)."""
    if name in TEXT_FIXES:
        return TEXT_FIXES[name]
    if not isinstance(location, str):
        return None
    text = location.lower().replace("  ", " ")
    for keyword, key in TEXT_AREAS:
        if keyword in text:
            return key
    return None


def weapon_shares(credits):
    """Split one ship between its weapons: {weapon: share}, shares adding up to 1.

    Each credit gets 1/n. Shore batteries, sabotage, accidents and unknown
    causes are pooled as "other". Japan-credited mines stay "mine": for a
    route, a ship lost on a mine is a ship lost on a mine.
    """
    shares = {}
    for _, weapon in credits:
        group = weapon if weapon in WEAPON_ORDER else "other"
        shares[group] = shares.get(group, 0) + 1 / len(credits)
    return shares


def merchant_ships():
    """Load the merchant ships and add date, time, area, route group and weapon columns.

    Columns added:
      sunk, month, t   sinking date, its month, and months since 7 Dec 1941
      area, group      sea area and route group (None if it cannot be placed)
      area_source      "position" or "text location"
      submarine_only   1 if every credit is a submarine (R2-R5)
      share_<weapon>   the ship's 1/n share for each weapon group
    """
    ships = smap.load_ships()
    ships = ships[ships["vessel"] == "merchant"].copy()
    # KENKOKU has no date; it cannot enter a time analysis.
    ships = ships[ships["year"].notna()].copy()
    ships["sunk"] = sinking_dates(ships)
    ships["month"] = ships["sunk"].dt.to_period("M")
    ships["t"] = (ships["sunk"] - WAR_START).dt.days / AVERAGE_MONTH_DAYS

    by_position = [area_of_position(lon, lat) if pd.notna(lat) else None
                   for lon, lat in zip(ships["lon"], ships["lat"])]
    by_text = [area_of_text(name, loc) for name, loc in zip(ships["vessel_name"],
                                                           ships["location"])]
    ships["area"] = [p if p is not None else t for p, t in zip(by_position, by_text)]
    ships["area_source"] = np.where(pd.notna(pd.Series(by_position, index=ships.index)),
                                    "position", np.where(ships["area"].notna(),
                                                         "text location", "unplaced"))
    ships["group"] = ships["area"].map(GROUP_OF)

    shares = ships["credits"].apply(weapon_shares)
    for weapon in WEAPON_ORDER:
        ships[f"share_{weapon}"] = shares.apply(lambda s: s.get(weapon, 0.0))
    ships["submarine_only"] = (ships["share_submarine"] == 1).astype(int)
    ships["us_submarine"] = ships["credits"].apply(
        lambda c: any(a == "us" and w == "submarine" for a, w in c))
    return ships


def year_of(month):
    """Year for the area x year tables, with December 1941 counted in 1942.

    Three weeks of war in 1941 give too few ships for a column of their own.
    """
    return max(month.year, 1942)


# ---------------------------------------------------------------------------
# R1: losses by area and year
# ---------------------------------------------------------------------------
def area_year_table(ships):
    """Ships and GRT lost per area and year: all causes, submarines, and the submarine share.

    Submarine numbers here use the 1/n shares, so they add up to the
    report's totals; the share is submarine GRT / all GRT.
    """
    placed = ships[ships["area"].notna()].assign(
        year=ships["month"].apply(year_of),
        sub_grt=ships["share_submarine"] * ships["tonnage"])
    table = (placed.groupby(["area", "year"])
             .agg(ships=("tonnage", "size"), grt=("tonnage", "sum"),
                  submarine_ships=("share_submarine", "sum"), submarine_grt=("sub_grt", "sum"))
             .reset_index())
    table["submarine_grt_share"] = table["submarine_grt"] / table["grt"]
    table["group"] = table["area"].map(GROUP_OF)
    table["area_label"] = table["area"].map(AREA_LABELS)
    table["area"] = pd.Categorical(table["area"], AREA_ORDER, ordered=True)
    return table.sort_values(["area", "year"])


# ---------------------------------------------------------------------------
# R2: area x year contingency for submarine sinkings
# ---------------------------------------------------------------------------
def submarine_area_year(ships, rng):
    """Chi-square test of area x year for ships sunk by submarines only.

    H0: the areas where submarines sank ships were in the same proportions
    every year. Areas with fewer than 20 submarine sinkings in the whole war
    (the Indian Ocean, the north) are pooled into "other areas", otherwise
    their tiny rows give expected counts near 0. The permutation p-value and
    the residual threshold are computed as in weapon_ship_analysis.py.
    """
    subs = ships[(ships["submarine_only"] == 1) & ships["area"].notna()].copy()
    subs["year"] = subs["month"].apply(year_of)
    counts = subs["area"].value_counts()
    small = [a for a in AREA_ORDER if counts.get(a, 0) < 20]
    subs["row"] = subs["area"].where(~subs["area"].isin(small), "other_areas")
    rows = [a for a in AREA_ORDER if a not in small] + (["other_areas"] if small else [])
    table = pd.crosstab(subs["row"], subs["year"]).reindex(index=rows, fill_value=0)
    result = chi_square(table, rng)
    v, v_corr = cramers_v(result["x2"], result["n"], *table.shape)
    residuals = adjusted_residuals(table, result["expected"])
    threshold = stats.norm.ppf(1 - ALPHA / (2 * table.size))
    return table, residuals, result, v, v_corr, threshold, small


# ---------------------------------------------------------------------------
# R3: multinomial model of route group on time
# ---------------------------------------------------------------------------
def fit_group_model(data):
    """Multinomial logit of route group on a natural cubic spline of time.

    log(P(group k) / P(southern)) = a_k + f_k(t), with f_k a natural spline
    with SPLINE_DF degrees of freedom (patsy's cr(), centred). Returns the
    fitted model and the intercept-only model, for the likelihood-ratio test.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = smf.mnlogit(f"code ~ cr(t, df={SPLINE_DF}, constraints='center')",
                           data).fit(disp=0, maxiter=500, method="bfgs")
        null = smf.mnlogit("code ~ 1", data).fit(disp=0, maxiter=500, method="bfgs")
    return full, null


def route_share_model(ships, rng):
    """R3: modelled share of each route group among submarine sinkings, month by month.

    Why a model as well as the chi-square of R2: the chi-square cuts time
    into years and says only "the mix changed"; the spline model gives a
    smooth curve of each group's share with an uncertainty band, and a
    single likelihood-ratio test of "no change over time":
        LR = 2 (logL_full - logL_null) ~ chi-square with (K-1) x SPLINE_DF df.
    The LR test treats ships as independent; sinkings in the same convoy
    attack are not, so the p-value is optimistic. The bands are therefore
    built with a cluster bootstrap: whole months are resampled with
    replacement, the model is refitted, and the 2.5th-97.5th percentiles of
    the predicted shares are taken.
    """
    subs = ships[(ships["submarine_only"] == 1) & ships["group"].notna()].copy()
    subs["code"] = subs["group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    full, null = fit_group_model(subs)
    lr = 2 * (full.llf - null.llf)
    dof = (len(GROUP_ORDER) - 1) * SPLINE_DF
    grid = pd.DataFrame({"t": (MONTHS.to_timestamp() + pd.Timedelta(days=14) - WAR_START).days
                         / AVERAGE_MONTH_DAYS})
    # Keep the grid inside the data's time range; a spline is not
    # extrapolated beyond its boundary knots.
    grid["t"] = grid["t"].clip(subs["t"].min(), subs["t"].max())
    predicted = np.asarray(full.predict(grid))

    by_month = {m: g for m, g in subs.groupby("month")}
    month_keys = list(by_month)
    boot = []
    for _ in range(MODEL_BOOTSTRAP_SAMPLES):
        pick = rng.integers(0, len(month_keys), len(month_keys))
        sample = pd.concat([by_month[month_keys[i]] for i in pick], ignore_index=True)
        if sample["code"].nunique() < len(GROUP_ORDER):
            continue   # a resample missing a whole group cannot be fitted; skip it
        try:
            fit, _ = fit_group_model(sample)
            boot.append(np.asarray(fit.predict(grid)))
        except Exception:   # a rare non-converging resample is skipped, not fatal
            continue
    boot = np.array(boot)
    rows = []
    for i, month in enumerate(MONTHS):
        for k, group in enumerate(GROUP_ORDER):
            rows.append({"month": str(month), "group": group, "share": predicted[i, k],
                         "ci_low": np.percentile(boot[:, i, k], 2.5),
                         "ci_high": np.percentile(boot[:, i, k], 97.5)})
    test = {"test": "R3 multinomial logit, route group of submarine sinkings ~ spline(time)",
            "statistic": lr, "df": dof, "p_value": stats.chi2.sf(lr, dof),
            "estimate": 1 - full.llf / null.llf, "unit": "McFadden pseudo-R2",
            "n": len(subs),
            "detail": f"{len(boot)} of {MODEL_BOOTSTRAP_SAMPLES} month-cluster bootstrap refits used"}
    return pd.DataFrame(rows), test, subs


# ---------------------------------------------------------------------------
# R4: when was each area's traffic hit?
# ---------------------------------------------------------------------------
def cluster_bootstrap_median(part, rng):
    """95% interval for the median sinking date, resampling whole months.

    Ships sunk in the same month (often the same convoy) are not
    independent; resampling months rather than ships keeps that
    dependence, so the interval is honest (and wider).
    """
    groups = [g["t"].to_numpy() for _, g in part.groupby("month")]
    meds = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups),
                                                                      len(groups))]))
            for _ in range(BOOTSTRAP_SAMPLES)]
    return np.percentile(meds, [2.5, 97.5])


def t_to_date(t):
    """Convert months-since-7-Dec-1941 back to a calendar date."""
    return WAR_START + pd.Timedelta(days=float(t) * AVERAGE_MONTH_DAYS)


def area_timing(ships, rng):
    """Per area: number of ships, GRT, and the 10th/25th/50th/75th/90th percentile loss dates.

    The median comes with a month-cluster bootstrap interval. Computed for
    all causes and, separately, for submarine-only sinkings.
    """
    rows = []
    for subset, part_all in [("all causes", ships),
                             ("submarine", ships[ships["submarine_only"] == 1])]:
        for area in AREA_ORDER:
            part = part_all[part_all["area"] == area]
            if len(part) < 10:
                continue
            low, high = cluster_bootstrap_median(part, rng)
            q = np.percentile(part["t"], [10, 25, 50, 75, 90])
            rows.append({"subset": subset, "area": area, "area_label": AREA_LABELS[area],
                         "group": GROUP_OF[area], "ships": len(part),
                         "grt": part["tonnage"].sum(),
                         **{f"p{p}": t_to_date(v).date() for p, v in zip([10, 25, 50, 75, 90], q)},
                         "median_ci_low": t_to_date(low).date(),
                         "median_ci_high": t_to_date(high).date()})
    return pd.DataFrame(rows)


def timing_tests(ships, rng):
    """Kruskal-Wallis across areas, and southern-route vs inner-zone median date difference.

    Kruskal-Wallis H0: the loss dates have the same distribution in every
    area. Effect size epsilon^2 = H / (n - 1) (Tomczak & Tomczak, 2014).
    It treats ships as independent, so its p-value is optimistic; the
    cluster-bootstrap interval for the difference in medians between the
    southern route and the inner zone does not.
    """
    placed = ships[ships["area"].notna()]
    groups = [placed.loc[placed["area"] == a, "t"] for a in AREA_ORDER
              if (placed["area"] == a).sum() >= 10]
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    tests = [{"test": "R4 Kruskal-Wallis, loss date across areas (all causes)",
              "statistic": h, "df": len(groups) - 1, "p_value": p,
              "estimate": h / (n - 1), "unit": "epsilon squared", "n": n}]

    south = placed[placed["group"] == "southern"]
    inner = placed[placed["group"].isin(["inner", "home"])]
    # The months are resampled ONCE for both groups (a paired cluster
    # bootstrap): a drawn month brings its southern and its inner/home ships
    # together. Resampling the two groups separately would ignore that they
    # share the same calendar, and gives a wider, less correct interval.
    both = pd.concat([south.assign(side="s"), inner.assign(side="i")])
    by_month = [(g.loc[g["side"] == "s", "t"].to_numpy(), g.loc[g["side"] == "i", "t"].to_numpy())
                for _, g in both.groupby("month")]
    # Both the median and the mean difference are bootstrapped. The inner +
    # home dates are bimodal (a 1942-43 hump of coastal submarine kills and
    # the big 1945 peak), so the median can jump from one hump to the other
    # in a resample and its interval is very wide; the mean moves smoothly
    # and gives the steadier comparison.
    diffs, mean_diffs = [], []
    for _ in range(BOOTSTRAP_SAMPLES):
        pick = rng.integers(0, len(by_month), len(by_month))
        s = np.concatenate([by_month[i][0] for i in pick])
        i_ = np.concatenate([by_month[i][1] for i in pick])
        diffs.append(np.median(i_) - np.median(s))
        mean_diffs.append(i_.mean() - s.mean())
    mean_low, mean_high = np.percentile(mean_diffs, [2.5, 97.5])
    tests.append({"test": "R4 Mean loss date, inner zone + home waters minus southern route",
                  "estimate": inner["t"].mean() - south["t"].mean(), "ci_low": mean_low,
                  "ci_high": mean_high, "unit": "months (CI: month-cluster bootstrap)",
                  "n": len(south) + len(inner)})
    estimate = inner["t"].median() - south["t"].median()
    low, high = np.percentile(diffs, [2.5, 97.5])
    u, p_u = stats.mannwhitneyu(inner["t"], south["t"], alternative="two-sided")
    tests.append({"test": "R4 Median loss date, inner zone + home waters minus southern route",
                  "statistic": u, "p_value": p_u, "estimate": estimate, "ci_low": low,
                  "ci_high": high, "unit": "months (CI: month-cluster bootstrap; "
                                           "p: Mann-Whitney U, ships independent)",
                  "n": len(south) + len(inner)})
    return tests


# ---------------------------------------------------------------------------
# R5: hand-over from submarines to other weapons
# ---------------------------------------------------------------------------
def handover_model(ships):
    """Logistic regression of 'sunk by a submarine' on time, by route group.

    Model: logit P(submarine_only) = a_g + b_g * t for route group g (the
    group-specific intercepts and slopes come from group * t). exp(12 b_g)
    is the odds ratio per year in group g. Standard errors are clustered by
    month. A Wald test of b_southern = b_inner = b_home = b_outer asks
    whether the swing away from submarines was the same everywhere.
    """
    data = ships[ships["group"].notna()].copy()
    data["group"] = pd.Categorical(data["group"], GROUP_ORDER)
    groups = data["month"].astype(str).factorize()[0]
    # "0 + C(group) + C(group):t" gives one intercept and one slope per group
    # directly, which is easier to read than a reference group plus
    # differences (the fit is identical either way).
    fit = smf.logit("submarine_only ~ 0 + C(group) + C(group):t", data).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    tests = []
    slope_names = [f"C(group)[{g}]:t" for g in GROUP_ORDER]
    for g, name in zip(GROUP_ORDER, slope_names):
        low, high = fit.conf_int().loc[name]
        tests.append({"test": f"R5 Logistic, P(sunk by submarine) per year, {GROUP_LABELS[g]}",
                      "estimate": math.exp(12 * fit.params[name]),
                      "ci_low": math.exp(12 * low), "ci_high": math.exp(12 * high),
                      "p_value": fit.pvalues[name], "unit": "odds ratio per year",
                      "n": int((data["group"] == g).sum())})
    constraint = ", ".join(f"{slope_names[0]} = {s}" for s in slope_names[1:])
    wald = fit.wald_test(constraint, scalar=True)
    tests.append({"test": "R5 Wald test, equal submarine-share slopes in all route groups",
                  "statistic": float(wald.statistic), "df": len(slope_names) - 1,
                  "p_value": float(wald.pvalue), "unit": "chi-square (cluster-robust)",
                  "n": len(data)})
    return tests


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def save_figure(fig, out_path):
    """Save a figure, retrying if Windows briefly refuses the file (Errno 22).

    Same reason as us_loss_time_series.save_figure: an image viewer or the
    search indexer can hold the old PNG open for a moment.
    """
    for attempt in range(3):
        try:
            fig.savefig(out_path, dpi=170, facecolor=BACKGROUND)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)
    plt.close(fig)


def draw_area_map(ships, out_path):
    """Map of the 12 areas (outlined and labelled) with every placed merchant loss.

    Dots are coloured by route group, so the reader sees both the area
    boundaries and which lane each group of sinkings belongs to. The map
    window and land come from sinking_map.style_map_axes, so it lines up
    with the project's other maps.
    """
    width = 16
    fig, ax = plt.subplots(figsize=(width, width / smap.map_aspect_ratio() + 1.6))
    smap.style_map_axes(ax)
    window = box(smap.LON_RANGE[0], smap.LAT_RANGE[0], smap.LON_RANGE[1], smap.LAT_RANGE[1])
    # Draw the effective (non-overlapping) areas, so each boundary is drawn
    # once and every point of sea visibly belongs to one area.
    for key, shape in effective_areas().items():
        clipped = shape.intersection(window)
        for part in getattr(clipped, "geoms", [clipped]):
            if part.is_empty or part.geom_type != "Polygon":
                continue
            x, y = part.exterior.xy
            ax.plot(x, y, color=tint(GROUP_COLOURS[GROUP_OF[key]], 0.15), linewidth=1.0,
                    linestyle="--", zorder=2)
    placed = ships[ships["area_source"] == "position"]
    for group in GROUP_ORDER:
        part = placed[placed["group"] == group]
        ax.scatter(part["lon"], part["lat"], s=12, color=GROUP_COLOURS[group], alpha=0.6,
                   linewidths=0, zorder=3)
    # Area labels at a hand-picked sea position inside each area, so the
    # label sits over water near the area's sinkings.
    label_at = {"nei_malaya": (112, -6.5), "south_china_sea": (111, 13),
                "philippines": (124.5, 12), "luzon_strait": (120.3, 20.3),
                "formosa_ryukyu": (126.5, 24.6), "yellow_ecs": (122.5, 36.5),
                "sea_of_japan": (134.5, 41.5), "japan_coast": (147, 31.5),
                "new_guinea_solomons": (153, -8), "central_pacific": (150, 15),
                "north": (165, 49), "indian_ocean": (88, 10)}
    for key, (x, y) in label_at.items():
        n = int((ships["area"] == key).sum())
        ax.text(x, y, f"{AREA_LABELS[key]}\n{n:,} ships", ha="center", va="center",
                fontsize=10, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=BACKGROUND, edgecolor="none",
                          alpha=0.85))
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=GROUP_COLOURS[g],
                      label=f"{GROUP_LABELS[g]}: {(ships['group'] == g).sum():,} ships")
               for g in GROUP_ORDER]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92,
              edgecolor=GRID, fontsize=11, title="Route group", title_fontsize=11)
    ax.set_title("Japanese merchant ships sunk, by sea area and shipping route\n",
                 loc="left", fontsize=18, fontweight="bold", color=INK)
    unplaced = int((ships["area_source"] == "unplaced").sum())
    by_text = int((ships["area_source"] == "text location").sum())
    fig.text(0.01, 0.01,
             f"Source: JANAC (1947), {len(ships):,} merchant ships. Dashed lines: area "
             f"boundaries (judgment calls along the narrows between route legs). {by_text} ships "
             f"placed from a text location are counted but not drawn; {unplaced} cannot be "
             "placed ('off Japan', no location).",
             fontsize=10, color=INK_SOFT)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.07)
    save_figure(fig, out_path)


def quarter_of(month):
    """Quarter label such as '1944 Q3', with Dec 1941 joined to 1942 Q1."""
    if month < pd.Period("1942-01", "M"):
        month = pd.Period("1942-01", "M")
    return f"{month.year} Q{(month.month - 1) // 3 + 1}"


def draw_heatmap(ships, out_path):
    """GRT sunk by submarines in each area, per quarter (a heatmap, one hue).

    Quarters rather than months: 12 areas x 45 months would be a wall of
    mostly empty cells. The colour scale is a square root of GRT, so the
    quieter areas are still visible next to the South China Sea's peak; the
    numbers written in the cells (thousands of GRT) give the exact values.
    """
    subs = ships[ships["area"].notna()].assign(
        quarter=ships["month"].apply(quarter_of),
        sub_grt=ships["share_submarine"] * ships["tonnage"])
    table = (subs.groupby(["area", "quarter"])["sub_grt"].sum().unstack(fill_value=0)
             .reindex(AREA_ORDER, fill_value=0))
    table = table[sorted(table.columns)]
    fig, ax = plt.subplots(figsize=(17, 8.2))
    ax.imshow(np.sqrt(table.to_numpy()), aspect="auto", cmap=HEAT_CMAP)
    top = np.sqrt(table.to_numpy().max())
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iat[i, j]
            if value >= 500:
                ax.text(j, i, f"{value / 1000:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if math.sqrt(value) > 0.55 * top else INK)
    ax.set_yticks(range(len(AREA_ORDER)),
                  [f"{AREA_LABELS[a]}" for a in AREA_ORDER])
    for tick, area in zip(ax.get_yticklabels(), AREA_ORDER):
        tick.set_color(GROUP_COLOURS[GROUP_OF[area]])
        tick.set_fontweight("bold")
    ax.set_xticks(range(table.shape[1]), [q.replace(" ", "\n") for q in table.columns],
                  fontsize=9.5)
    # Lines between route groups.
    for edge in [4.5, 6.5, 7.5]:
        ax.axhline(edge, color=INK_SOFT, linewidth=1)
    ax.set_title("Where submarines sank Japanese merchant ships, quarter by quarter\n",
                 loc="left", fontsize=17, fontweight="bold", color=INK)
    ax.text(0, 1.02, "Thousands of GRT sunk by submarines (1/n share for joint kills); "
            "darker = more. Area names coloured by route group: orange southern route, "
            "blue inner zone, violet coastal waters, aqua outer perimeter.",
            transform=ax.transAxes, fontsize=11, color=INK_SOFT)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.1)
    save_figure(fig, out_path)


def draw_residuals(table, residuals, threshold, result, v_corr, small, out_path):
    """R2: heatmap of adjusted residuals, area x year, for submarine sinkings.

    Diverging colours (blue = fewer than if the area mix never changed, red
    = more), capped at +-6 so one extreme cell does not wash out the rest.
    Cells beyond the Bonferroni threshold are outlined. The count is written
    in each cell, with the residual under it.
    """
    fig, ax = plt.subplots(figsize=(11, 8))
    values = residuals.to_numpy()
    ax.imshow(values, cmap=RESIDUAL_CMAP, vmin=-6, vmax=6, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{table.iat[i, j]}\n({values[i, j]:+.1f})", ha="center",
                    va="center", fontsize=10.5, color=INK)
            if abs(values[i, j]) > threshold:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=INK, linewidth=1.6))
    labels = [AREA_LABELS.get(a, "Other areas (" + ", ".join(AREA_LABELS[s].split(":")[0]
                                                            for s in small) + ")")
              for a in table.index]
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(table.shape[1]), [str(y) if y > 1942 else "1942\n(+Dec 1941)"
                                          for y in table.columns])
    # Title and subtitle are placed on the figure, from its left edge, so the
    # long text is not cut off by the wide area labels on the left.
    fig.suptitle("Submarine sinkings by area and year: more (red) or fewer (blue) than "
                 "an unchanging mix", x=0.02, ha="left", y=0.985, fontsize=14,
                 fontweight="bold", color=INK)
    fig.text(0.02, 0.905,
             f"Ships sunk by submarines only (n = {result['n']:,}). Cell: count (adjusted "
             f"residual). Outlined: |residual| > {threshold:.2f} (Bonferroni).\n"
             f"X² = {result['x2']:.0f}, df {result['dof']}, permutation p = "
             f"{result['p_permutation']:.4f}, bias-corrected Cramér's V = {v_corr:.2f}",
             fontsize=10, color=INK_SOFT)
    fig.subplots_adjust(left=0.3, right=0.98, top=0.88, bottom=0.08)
    save_figure(fig, out_path)


def draw_route_shares(model, subs, test, out_path):
    """R3: modelled share of each route group among submarine sinkings, with bands.

    One small panel per route group, rather than a stacked area: stacked
    shares hide the uncertainty bands and make every group but the bottom
    one hard to read. Dots are the observed quarterly shares (size = number
    of submarine sinkings that quarter), so the reader can judge the fit.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6), sharey=True)
    x = MONTHS.to_timestamp() + pd.Timedelta(days=14)
    subs = subs.assign(quarter=subs["month"].apply(lambda m: m.asfreq("Q")))
    quarterly = subs.groupby("quarter")["group"].value_counts(normalize=True).unstack(
        fill_value=0)
    sizes = subs.groupby("quarter").size()
    qx = [q.to_timestamp() + pd.Timedelta(days=45) for q in quarterly.index]
    for ax, group in zip(axes, GROUP_ORDER):
        part = model[model["group"] == group]
        colour = GROUP_COLOURS[group]
        ax.fill_between(x, part["ci_low"] * 100, part["ci_high"] * 100, color=tint(colour, 0.7),
                        linewidth=0, zorder=1)
        ax.plot(x, part["share"] * 100, color=colour, linewidth=2.4, zorder=3)
        ax.scatter(qx, quarterly.get(group, 0) * 100, s=sizes.to_numpy() / 2.5, color=colour,
                   alpha=0.45, edgecolors="white", linewidths=0.5, zorder=2)
        ax.set_title(GROUP_LABELS[group], loc="left", fontsize=12.5, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of submarine sinkings")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.suptitle("Where the submarines struck: each route group's share of their sinkings",
                 x=0.06, ha="left", y=0.98, fontsize=16, fontweight="bold", color=INK)
    fig.text(0.06, 0.885,
             f"Line: multinomial logit on a natural spline of time ({SPLINE_DF} df); band: "
             f"95% month-cluster bootstrap. Dots: observed quarterly share (size = ships). "
             f"LR test of no change: χ² = {test['statistic']:.0f}, df {test['df']}, "
             f"p = {test['p_value']:.1e}. n = {test['n']:,} ships sunk by submarines only.",
             fontsize=10.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.1, wspace=0.08)
    save_figure(fig, out_path)


def draw_timing(timing, out_path):
    """R4: when each area's losses happened (all causes), one row per area.

    For each area: a thin line from the 10th to the 90th percentile loss
    date, a thick bar for the middle half (25th-75th), and a white-ringed dot
    at the median with its month-cluster bootstrap interval as a small
    error bar. Rows are sorted by median date, so the order in which the
    routes were hit reads from top to bottom.
    """
    part = timing[timing["subset"] == "all causes"].sort_values("p50", ascending=False)
    fig, ax = plt.subplots(figsize=(14, 7.5))
    for y, (_, row) in enumerate(part.iterrows()):
        colour = GROUP_COLOURS[row["group"]]
        ax.plot([row["p10"], row["p90"]], [y, y], color=colour, linewidth=1.4, zorder=1)
        ax.plot([row["p25"], row["p75"]], [y, y], color=colour, linewidth=7,
                solid_capstyle="butt", alpha=0.55, zorder=2)
        ax.errorbar(pd.Timestamp(row["p50"]), y,
                    xerr=[[pd.Timestamp(row["p50"]) - pd.Timestamp(row["median_ci_low"])],
                          [pd.Timestamp(row["median_ci_high"]) - pd.Timestamp(row["p50"])]],
                    fmt="o", color=colour, markersize=9, markeredgecolor="white",
                    ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
        ax.text(pd.Timestamp("1945-09-10"), y, f"{row['ships']:,} ships, "
                f"{row['grt'] / 1000:,.0f}k GRT", va="center", fontsize=10, color=INK_SOFT)
    ax.set_yticks(range(len(part)), part["area_label"])
    for tick, group in zip(ax.get_yticklabels(), part["group"]):
        tick.set_color(GROUP_COLOURS[group])
        tick.set_fontweight("bold")
    ax.set_xlim(pd.Timestamp("1941-12-01"), pd.Timestamp("1946-02-15"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    handles = [Line2D([], [], color=INK_SOFT, linewidth=1.4, label="10th-90th percentile"),
               Line2D([], [], color=INK_SOFT, linewidth=7, alpha=0.55,
                      label="Middle half (25th-75th)"),
               Line2D([], [], marker="o", linestyle="", color=INK_SOFT, markersize=9,
                      label="Median, with 95% month-cluster bootstrap interval")]
    handles += [Patch(color=GROUP_COLOURS[g], label=GROUP_LABELS[g]) for g in GROUP_ORDER]
    # The legend goes under the plot: inside it, it would cover the latest
    # (bottom) rows, which carry the 1945 story.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.07), fontsize=10,
              frameon=False, ncol=4)
    ax.set_title("When each sea area's merchant losses happened (all causes)\n", loc="left",
                 fontsize=16, fontweight="bold", color=INK)
    ax.text(0, 1.02, "Sorted by median loss date, earliest at the top.",
            transform=ax.transAxes, fontsize=11, color=INK_SOFT)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.16)
    save_figure(fig, out_path)


def draw_handover(ships, tests, out_path):
    """R5: share of GRT sunk by each weapon, per quarter, in each route group.

    Stacked 100% bars per quarter show the hand-over from submarines
    (violet) to aircraft (aqua) and mines (yellow). Quarters with less than
    10,000 GRT lost in a group are drawn faded: their shares rest on very
    few ships. Each panel title carries the R5 odds ratio per year.
    """
    data = ships[ships["group"].notna()].assign(quarter=ships["month"].apply(quarter_of))
    quarters = sorted(data["quarter"].unique())
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
    odds = {t["test"].split(", ")[-1]: t for t in tests if t["test"].startswith("R5 Logistic")}
    for ax, group in zip(axes, GROUP_ORDER):
        part = data[data["group"] == group]
        grt = {w: (part[f"share_{w}"] * part["tonnage"]).groupby(part["quarter"]).sum()
               .reindex(quarters, fill_value=0) for w in WEAPON_ORDER}
        total = sum(grt.values())
        bottom = np.zeros(len(quarters))
        for weapon in WEAPON_ORDER:
            share = np.where(total > 0, grt[weapon] / total.replace(0, np.nan), 0) * 100
            share = np.nan_to_num(share)
            alphas = np.where(total >= 10000, 1.0, 0.35)
            for i in range(len(quarters)):
                ax.bar(i, share[i], bottom=bottom[i], color=WEAPON_COLOURS[weapon],
                       alpha=alphas[i], width=0.85, edgecolor=BACKGROUND, linewidth=0.8)
            bottom += share
        t = odds[GROUP_LABELS[group]]
        ax.set_title(f"{GROUP_LABELS[group]}: {part['tonnage'].sum() / 1e6:.2f} million GRT. "
                     f"Odds of a submarine kill changed ×{t['estimate']:.2f} per year "
                     f"(95% CI {t['ci_low']:.2f}–{t['ci_high']:.2f})",
                     loc="left", fontsize=12, color=INK)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_ylabel("% of GRT")
    axes[-1].set_xticks(range(len(quarters)), [q.replace(" ", "\n") for q in quarters],
                        fontsize=9.5)
    handles = [Patch(color=WEAPON_COLOURS[w], label=WEAPON_LABELS[w]) for w in WEAPON_ORDER]
    fig.legend(handles=handles, loc="upper right", ncol=5, frameon=False, fontsize=11,
               bbox_to_anchor=(0.99, 0.965))
    fig.suptitle("The hand-over: which weapon sank the GRT on each route, quarter by quarter",
                 x=0.06, ha="left", y=0.985, fontsize=16, fontweight="bold", color=INK)
    fig.text(0.06, 0.012, "Joint kills split equally between weapons. Faded bars: quarters "
             "with under 10,000 GRT lost in that route group. Odds ratios: logistic regression "
             "of 'sunk by submarine only' on time, standard errors clustered by month.",
             fontsize=10, color=INK_SOFT)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.07, hspace=0.35)
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def main():
    """Run R1-R5, save every table and chart, and print the main numbers."""
    helpers.set_readable_style()
    rng = np.random.default_rng(RANDOM_SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ships = merchant_ships()
    keep = ["vessel_name", "date", "sunk", "vessel_type", "ship_type", "tonnage", "location",
            "lat", "lon", "flag_of_agent", "agent_type", "area", "group", "area_source",
            "submarine_only"] + [f"share_{w}" for w in WEAPON_ORDER]
    ships[keep].to_csv(DATA_DIR / "merchant_ship_areas.csv", index=False, float_format="%.4g")
    print(ships["area_source"].value_counts().to_string())
    print(ships["area"].value_counts().to_string())
    print(f"US share of submarine-only sinkings: "
          f"{ships.loc[ships['submarine_only'] == 1, 'us_submarine'].mean():.3f}")

    area_year = area_year_table(ships)
    area_year.to_csv(DATA_DIR / "area_year_losses.csv", index=False, float_format="%.4g")

    table, residuals, result, v, v_corr, threshold, small = submarine_area_year(ships, rng)
    expected = pd.DataFrame(result["expected"], index=table.index, columns=table.columns)
    pd.concat({"observed": table, "expected": expected, "adjusted_residual": residuals},
              axis=1).to_csv(DATA_DIR / "submarine_area_year.csv", float_format="%.4g")
    tests = [{"test": "R2 Chi-square, area x year of submarine-only sinkings",
              "statistic": result["x2"], "df": result["dof"],
              "p_value": result["p_permutation"], "estimate": v_corr,
              "unit": "bias-corrected Cramer's V (p: permutation)", "n": result["n"],
              "detail": f"asymptotic p {result['p_asymptotic']:.3g}; plain V {v:.3f}; "
                        f"{result['cells_expected_below_5']} of {result['cells']} cells "
                        f"expected < 5; residual threshold {threshold:.2f}; pooled areas: "
                        + ", ".join(small)}]

    model, model_test, subs = route_share_model(ships, rng)
    model.to_csv(DATA_DIR / "route_share_model.csv", index=False, float_format="%.4g")
    tests.append(model_test)

    timing = area_timing(ships, rng)
    timing.to_csv(DATA_DIR / "area_timing.csv", index=False, float_format="%.4g")
    tests += timing_tests(ships, rng)
    tests += handover_model(ships)
    pd.DataFrame(tests).to_csv(DATA_DIR / "route_tests.csv", index=False, float_format="%.4g")

    draw_area_map(ships, FIG_DIR / "route_areas_map.png")
    draw_heatmap(ships, FIG_DIR / "submarine_area_heatmap.png")
    draw_residuals(table, residuals, threshold, result, v_corr, small,
                   FIG_DIR / "submarine_area_residuals.png")
    draw_route_shares(model, subs, model_test, FIG_DIR / "submarine_route_shares.png")
    draw_timing(timing, FIG_DIR / "route_timing.png")
    draw_handover(ships, tests, FIG_DIR / "weapon_handover.png")

    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.max_colwidth", 90):
        print(pd.DataFrame(tests).drop(columns="detail", errors="ignore").to_string(index=False))
        print(table.to_string())
        print(residuals.round(1).to_string())
        print(timing.to_string(index=False))
    print(f"Saved figures to {FIG_DIR} and tables to {DATA_DIR}")


if __name__ == "__main__":
    main()
