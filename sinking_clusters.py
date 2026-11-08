"""
Find the areas where many Japanese ships were sunk, and test them statistically.

Two methods, used together:

1. HDBSCAN (a density-based clustering method) finds groups of sinkings that
   lie close together. It needs no number of clusters in advance; you only
   set the smallest group that counts as a cluster (--min-cluster-size).
   Ships that belong to no group are labelled "noise" (isolated sinkings).

2. Getis-Ord Gi* (a "hot-spot" statistic) then checks the clusters. The ocean
   is cut into hexagons and the ships in each are counted. Gi* asks, for every
   hexagon: do this hexagon and its neighbours together hold more sinkings
   than chance would give? Its p-values are corrected for testing thousands of
   hexagons at once (Benjamini-Hochberg false discovery rate).

A cluster is "confirmed" when at least half of its ships lie in hexagons that
are significant hot spots after that correction. So HDBSCAN says WHERE the
groups are, and Gi* says whether each group is more than chance.

It uses the same filters as sinking_map.py (agent, weapon, vessel, ship_type,
min_tonnage, max_tonnage; leave one out to include all its options), so you
can cluster e.g. only US submarine sinkings, or only tankers. Wrong values
stop the program with a message naming the variable and listing its options.

Input (the same as sinking_map.py):
    data/naval/naval_losses_15-44.csv
    data/merchant/merchant_losses_45-115.csv
    data/basemap/ne_50m_land/ne_50m_land.shp

Output, named after the settings and filters, e.g. for the defaults:
    figures/clusters/clusters_mcs-15_agent-all_weapon-all_vessel-all_type-all_tons-all.png
        map of the clusters, coloured by whether Gi* confirms them
    figures/clusters/hotspots_hex-100km_mcs-15_agent-all_weapon-all_vessel-all_type-all_tons-all.png
        map of the Gi* hot-spot hexagons with the cluster outlines on top
    data/clusters/clusters_mcs-15_hex-100km_agent-all_weapon-all_vessel-all_type-all_tons-all.csv
        one row per cluster: place, ships, tonnage, dates, main agent and
        weapon, share of ships in hot spots, confirmed or not

Caveat: a hot spot means "many ships were sunk here", not "this sea was more
dangerous per ship". Ships were sunk where ships sailed (Japan's coast, the
Luzon Strait), and the report has no data on how many ships used each route.

Run:
    python sinking_clusters.py                                  # all ships, min size 15
    python sinking_clusters.py --min-cluster-size 30            # fewer, bigger clusters
    python sinking_clusters.py --agent us --weapon submarine --min-cluster-size 10
    python sinking_clusters.py --ship-type tanker --min-cluster-size 10
    python sinking_clusters.py --min-tonnage 5000               # only ships of 5,000+ tons
    python sinking_clusters.py --hex-km 150                     # coarser Gi* grid
    python sinking_clusters.py --options                        # list filter values
"""

import argparse
import math
import sys
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no window is needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from esda import G_Local
from libpysal.weights import W
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import Transformer
from shapely.geometry import MultiPoint
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import haversine_distances

import sinking_map as smap
import table_plot_helpers as helpers
from sinking_animation import sinking_dates
from sinking_map import OptionError, AGENT_OPTIONS, WEAPON_OPTIONS
from table_plot_helpers import BLUE, GREY, INK, INK_SOFT, ORANGE

FIG_DIR = smap.PROJECT_DIR / "figures" / "clusters"
DATA_DIR = smap.PROJECT_DIR / "data" / "clusters"

EARTH_RADIUS_KM = 6371.0

# ---------------------------------------------------------------------------
# Settings (all can be changed from the command line)
# ---------------------------------------------------------------------------
# HDBSCAN: the smallest number of ships that counts as a cluster. 15 picks
# out the main campaign areas for the whole war (about 3,000 ships) without
# splitting them into many tiny groups. For a filtered selection with fewer
# ships, use a smaller value (e.g. 5-10).
DEFAULT_MIN_CLUSTER_SIZE = 15

# Gi*: distance between the centres of neighbouring hexagons. Positions in
# the report are often rounded to whole degrees (about 110 km), so cells much
# smaller than ~100 km would pretend to a precision the data doesn't have.
# Much bigger cells would merge separate areas (e.g. Rabaul and Kavieng).
DEFAULT_HEX_KM = 100

# Gi*: number of random shuffles used to get each hexagon's p-value. The
# smallest possible p-value is 1 / (permutations + 1). With ~5,000 hexagons
# the false-discovery correction needs p-values far below 0.001, so 999
# shuffles (p >= 0.001) is too few; 9,999 (p >= 0.0001) is enough.
DEFAULT_PERMUTATIONS = 9999

# Significance level for the hot spots, after the false-discovery correction.
ALPHA = 0.05
# A cluster is confirmed when at least this share of its ships lie in
# significant hot-spot hexagons. Half is the natural "most of it" line.
CONFIRM_SHARE = 0.5

# The study area for Gi*: every hexagon within this distance of a hexagon
# with a sinking. Gi* compares each area with the study area's average, so
# the choice matters: the whole Pacific (mostly empty) would make every
# sinking look like a hot spot; only the hexagons with ships would hide the
# empty sea between campaigns. 300 km of sea around the sinkings is a fair
# "where ships could have been sunk" area.
STUDY_AREA_KM = 300

# Fixed seed so the permutation p-values are the same on every run.
RANDOM_SEED = 12345

# Equal-area projection centred on the western Pacific, used to build the
# hexagons. In plain latitude/longitude, a 1-degree cell at 45 N is 30% smaller
# than one at the equator, which would bias the counts; in an equal-area
# projection every hexagon covers the same sea area.
EQUAL_AREA = "+proj=laea +lat_0=15 +lon_0=150 +units=km"

# Colours. Cluster dots: orange = confirmed by Gi*, blue = not confirmed,
# grey = noise. Two status colours instead of one colour per cluster,
# because there are often 20-40 clusters, far more than the 8 colours that
# stay distinguishable (and clusters are identified by their number anyway).
# Orange next to blue passed the colour-blindness check in the other charts.
CONFIRMED_COLOUR = ORANGE
UNCONFIRMED_COLOUR = BLUE
NOISE_COLOUR = GREY
# Hot-spot hexagons: one hue, two strengths (dark = significant after the
# false-discovery correction, light = significant only before it).
HOT_STRONG = "#c4441a"            # a darker step of the palette's orange
HOT_WEAK = helpers.tint(ORANGE, 0.55)

# ---------------------------------------------------------------------------
# Place names for labelling clusters
# ---------------------------------------------------------------------------
# A hand-made list of well-known places in the war at sea (approximate
# latitude, longitude). Each cluster is named after the place nearest to its
# centre, if one is within NAME_MAX_KM; otherwise it is called "open sea".
# The names are a reading aid only; they don't affect the analysis.
PLACES = {
    "Shimonoseki Strait": (33.95, 130.95), "Inland Sea (Kure)": (34.2, 132.6),
    "Osaka / Kobe": (34.6, 135.2), "Nagoya": (35.0, 136.9), "Tokyo Bay": (35.4, 139.8),
    "Nagasaki": (32.75, 129.85), "Korea Strait": (34.5, 129.3),
    "Tsugaru Strait": (41.5, 140.5), "East Hokkaido": (43.0, 145.0),
    "Sanriku coast (NE Honshu)": (39.5, 142.0), "Amami Islands": (28.3, 129.5),
    "Kuril Islands": (50.5, 156.0), "Aleutians (Kiska)": (52.0, 177.5),
    "Korea (Wonsan)": (39.2, 127.5), "Yellow Sea": (36.0, 123.5),
    "Shanghai": (31.2, 121.8), "East China Sea": (29.0, 125.0),
    "Okinawa": (26.3, 127.8), "Formosa Strait": (24.0, 119.5),
    "North Formosa (Keelung)": (25.2, 121.7), "Luzon Strait": (20.5, 121.0),
    "Hong Kong": (22.3, 114.2), "Hainan": (19.5, 110.5),
    "Cam Ranh Bay": (11.9, 109.2), "Saigon": (10.4, 107.0),
    "Manila Bay": (14.5, 120.8), "Ormoc Bay (Leyte)": (10.9, 124.5),
    "Mindanao (Davao)": (7.0, 125.6), "Palau": (7.4, 134.5),
    "Marianas (Saipan)": (15.2, 145.7), "Bonin Islands": (27.1, 142.2),
    "Truk": (7.4, 151.8), "Marshall Islands": (9.0, 167.5),
    "Rabaul": (-4.2, 152.2), "Kavieng": (-2.6, 150.8),
    "Bougainville": (-6.5, 155.5), "Guadalcanal": (-9.4, 160.0),
    "Wewak (New Guinea)": (-3.5, 143.6), "Hollandia (New Guinea)": (-2.5, 140.7),
    "Halmahera": (0.5, 128.0), "Ambon": (-3.7, 128.2), "Kendari (Celebes)": (-4.0, 122.5),
    "Makassar Strait": (-2.0, 118.0), "Balikpapan": (-1.3, 116.8),
    "North Borneo": (5.5, 115.5), "Java Sea": (-5.5, 111.0),
    "Singapore": (1.3, 103.8), "Malacca Strait": (3.0, 100.5),
    "Gulf of Siam": (9.0, 101.0), "Andaman Islands": (11.7, 92.7),
}
NAME_MAX_KM = 400


# ---------------------------------------------------------------------------
# Checking the settings
# ---------------------------------------------------------------------------
def check_number(name, value, minimum):
    """Check that a numeric setting is a whole number >= minimum.

    Raises OptionError naming the setting and saying what is allowed, the
    same way the agent/weapon/vessel filters report a wrong value.
    """
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise OptionError(f"Wrong value for {name}: {value!r}. "
                          f"Options for {name}: a whole number of at least {minimum}.")
    return int(value)


# ---------------------------------------------------------------------------
# Step 1: HDBSCAN clusters
# ---------------------------------------------------------------------------
def find_clusters(ships, min_cluster_size, min_samples=None):
    """Run HDBSCAN on the ship positions and return a cluster number per ship.

    Distances are great-circle distances on the globe ("haversine"), so a
    degree of longitude near Japan and one near the equator are measured
    correctly, and the date line is no problem. HDBSCAN wants (latitude,
    longitude) in radians for this.

    min_samples sets how cautious HDBSCAN is (bigger = more ships called
    noise); left out, it equals min_cluster_size, scikit-learn's default.

    Clusters are renumbered by size, so cluster 1 is always the biggest.
    Noise is -1.
    """
    coordinates = np.radians(ships[["lat", "lon"]].to_numpy())
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                    metric="haversine", copy=True)
    raw = model.fit_predict(coordinates)
    sizes = Counter(label for label in raw if label >= 0)
    new_number = {old: rank + 1 for rank, (old, _) in enumerate(sizes.most_common())}
    return np.array([new_number.get(label, -1) for label in raw])


def medoid(rows):
    """The ship position closest (in total) to all others in the cluster.

    Used as the cluster's centre for naming and the label position. Unlike
    the average position, it is always a real sinking site, so a curved
    cluster (e.g. along a coast) doesn't get a centre on land.
    """
    points = np.radians(rows[["lat", "lon"]].to_numpy())
    return rows.iloc[haversine_distances(points).sum(axis=1).argmin()]


def place_name(lat, lon):
    """Name of the nearest place in PLACES, or 'Open sea' if none is close."""
    names = list(PLACES)
    places = np.radians(np.array(list(PLACES.values())))
    km = haversine_distances(np.radians([[lat, lon]]), places)[0] * EARTH_RADIUS_KM
    nearest = km.argmin()
    return names[nearest] if km[nearest] <= NAME_MAX_KM else "Open sea"


# ---------------------------------------------------------------------------
# Step 2: Gi* hot spots on a hexagon grid
# ---------------------------------------------------------------------------
# The hexagons use "axial" coordinates (q, r): every hexagon is a pair of
# whole numbers, a point is assigned to its hexagon by a formula (no search
# needed), and the six neighbours of (q, r) are fixed offsets. Hexagons are
# used instead of squares because all six neighbours of a hexagon are at the
# same distance; a square's diagonal neighbours are 41% further away.
NEIGHBOUR_STEPS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


def to_hexagon(x, y, size):
    """Axial (q, r) of the hexagon containing each point (x, y in km).

    size is the hexagon's centre-to-corner distance. The fractional
    coordinates are rounded in "cube" form (q + r + s = 0), the standard way
    to find the nearest hexagon centre.
    """
    q = (math.sqrt(3) / 3 * x - y / 3) / size
    r = (2 / 3 * y) / size
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs - s)
    # The coordinate that was rounded the most is recomputed from the others.
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    return rq.astype(int), rr.astype(int)


def hexagon_centre(q, r, size):
    """Centre (x, y in km) of axial hexagon (q, r); pointy-top layout."""
    return size * math.sqrt(3) * (q + r / 2), size * 1.5 * r


def hot_spots(ships, hex_km, permutations):
    """Count ships per hexagon and run Gi* on the counts.

    Returns (hexes, ship_hex): a table with one row per hexagon of the study
    area (counts, z-score, p-value, hot-spot class, and corners for drawing),
    and each ship's hexagon id.
    """
    size = hex_km / math.sqrt(3)  # centre-to-corner, so centres are hex_km apart
    to_km = Transformer.from_crs("EPSG:4326", EQUAL_AREA, always_xy=True)
    to_degrees = Transformer.from_crs(EQUAL_AREA, "EPSG:4326", always_xy=True)

    lon = np.where(ships["lon"] > 180, ships["lon"] - 360, ships["lon"])
    x, y = to_km.transform(lon, ships["lat"].to_numpy())
    q, r = to_hexagon(x, y, size)
    counts = Counter(zip(q, r))

    # Study area: every hexagon within STUDY_AREA_KM of a hexagon with ships.
    rings = math.ceil(STUDY_AREA_KM / hex_km)
    area = set()
    for cq, cr in counts:
        for dq in range(-rings, rings + 1):
            for dr in range(max(-rings, -dq - rings), min(rings, -dq + rings) + 1):
                area.add((cq + dq, cr + dr))
    hexes = pd.DataFrame(sorted(area), columns=["q", "r"])
    hexes["ships"] = [counts.get(key, 0) for key in zip(hexes["q"], hexes["r"])]

    # Drop empty hexagons whose centre is on land: no ship can sink there, so
    # counting them as "0 sinkings" would make the sea look busier than it
    # was. Hexagons with a sinking are kept even if their centre is on land
    # (a coastal hexagon is partly sea).
    cx, cy = hexagon_centre(hexes["q"].to_numpy(), hexes["r"].to_numpy(), size)
    clon, clat = to_degrees.transform(cx, cy)
    clon = np.where(clon < 0, clon + 360, clon)        # the map's 0-360 longitudes
    land = shapely.unary_union(smap.load_land().to_numpy())
    on_land = shapely.contains_xy(land, clon, clat)
    hexes = hexes[~on_land | (hexes["ships"] > 0)].reset_index(drop=True)

    # Neighbours: the six touching hexagons that are in the study area.
    index = {key: i for i, key in enumerate(zip(hexes["q"], hexes["r"]))}
    neighbours = {i: [index[(hq + a, hr + b)] for a, b in NEIGHBOUR_STEPS
                      if (hq + a, hr + b) in index]
                  for (hq, hr), i in index.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # see the note below
        weights = W(neighbours, silence_warnings=True)

    # Gi* ("star" = each hexagon counts itself as well as its neighbours).
    # Binary weights ("B"): each neighbour counts fully. The p-value comes
    # from shuffling the counts among hexagons `permutations` times.
    #
    # The study area falls apart into a few separate pieces (e.g. the
    # Aleutians are not connected to the rest by sea hexagons), and some
    # lone hexagons have no neighbours at all. libpysal warns about the
    # pieces and esda divides by zero for the lone hexagons. Both are
    # harmless here: a lone hexagon just gets no z-score (NaN), which is
    # never counted as a hot spot. So these warnings are silenced.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))
        gi = G_Local(hexes["ships"].to_numpy(dtype=float), weights, transform="B",
                     star=True, permutations=permutations, seed=RANDOM_SEED)
    hexes["z"] = gi.Zs
    hexes["p"] = gi.p_sim

    # Only hot spots (z > 0) matter here; "cold spots" would be empty sea,
    # which is expected and says nothing about where ships were sunk.
    threshold = fdr_threshold(hexes["p"].to_numpy(), ALPHA)
    hot = hexes["z"] > 0
    hexes["hot_fdr"] = hot & (hexes["p"] <= threshold)
    hexes["hot_uncorrected"] = hot & (hexes["p"] <= ALPHA)
    print(f"Gi*: {len(hexes):,} hexagons of {hex_km} km; "
          f"{hexes['hot_fdr'].sum()} hot spots after FDR correction "
          f"(p <= {threshold:.5f}), {hexes['hot_uncorrected'].sum()} before it")

    # Corners in map coordinates, for drawing. Corners are converted one by
    # one and moved to 0-360 longitude, so hexagons on the date line stay whole.
    angles = np.radians(30 + 60 * np.arange(6))
    hx, hy = hexagon_centre(hexes["q"].to_numpy(), hexes["r"].to_numpy(), size)
    corner_x = hx[:, None] + size * np.cos(angles)[None, :]
    corner_y = hy[:, None] + size * np.sin(angles)[None, :]
    corner_lon, corner_lat = to_degrees.transform(corner_x, corner_y)
    corner_lon = np.where(corner_lon < 0, corner_lon + 360, corner_lon)
    hexes["corners"] = [np.column_stack([a, b]) for a, b in zip(corner_lon, corner_lat)]

    ship_hex = np.array([index[key] for key in zip(q, r)])
    return hexes, ship_hex


def fdr_threshold(p_values, alpha):
    """Benjamini-Hochberg cut-off: p-values at or below it are significant.

    Testing thousands of hexagons at the 5% level would give about 5% false
    hot spots (hundreds) by chance alone. Benjamini-Hochberg instead keeps
    the expected share of false hot spots among those found at 5%: sort the
    m p-values, find the largest k with p(k) <= k/m * alpha, and accept the
    k smallest. Returns 0 if nothing passes.
    """
    ordered = np.sort(p_values)
    m = len(ordered)
    passing = ordered <= alpha * np.arange(1, m + 1) / m
    return ordered[np.nonzero(passing)[0].max()] if passing.any() else 0.0


# ---------------------------------------------------------------------------
# Step 3: summary per cluster
# ---------------------------------------------------------------------------
def most_common(credit_lists, position):
    """Most common agent (position 0) or weapon (1) in a cluster, with its share.

    A ship sunk jointly (e.g. US + Australian aircraft) contributes each of
    its credits, so shares are of credits, not of ships.
    """
    tally = Counter(credit[position] for credits in credit_lists for credit in credits)
    name, number = tally.most_common(1)[0]
    return f"{name} ({number / sum(tally.values()):.0%})"


def summarise(ships, hexes, ship_hex):
    """One row per cluster: where, how many, how big, when, by whom, confirmed?

    'attack_sites' counts distinct positions: ships sunk together in one
    convoy attack often share one printed position, so 40 ships may be only
    10 separate attacks. Both numbers are given so one big raid isn't
    mistaken for a long campaign.
    """
    ships = ships.assign(in_hot_spot=hexes["hot_fdr"].to_numpy()[ship_hex])
    rows = []
    for number, members in ships[ships["cluster"] > 0].groupby("cluster"):
        centre = medoid(members)
        naval = members["vessel"] == "naval"
        share_hot = members["in_hot_spot"].mean()
        rows.append({
            "cluster": number,
            "place": place_name(centre["lat"], centre["lon"]),
            "ships": len(members),
            "naval_ships": int(naval.sum()),
            "merchant_ships": int((~naval).sum()),
            "attack_sites": members[["lat", "lon"]].drop_duplicates().shape[0],
            "naval_tons": int(members.loc[naval, "tonnage"].sum()),
            "merchant_grt": int(members.loc[~naval, "tonnage"].sum()),
            "centre_lat": round(centre["lat"], 2),
            "centre_lon": round(centre["lon"] - 360 if centre["lon"] > 180 else centre["lon"], 2),
            "first_sinking": members["sunk"].min().date(),
            "last_sinking": members["sunk"].max().date(),
            "main_agent": most_common(members["credits"], 0),
            "main_weapon": most_common(members["credits"], 1),
            "share_in_hot_spots": round(share_hot, 3),
            "confirmed_by_gi_star": bool(share_hot >= CONFIRM_SHARE),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def map_figure(chosen, title):
    """A figure with the map on the left and a list panel on the right.

    The map is the same as in sinking_map.py. The right-hand panel lists the
    clusters, because 20-40 place names written on the map would cover the
    very areas they describe; the map only shows each cluster's number.
    """
    fig = plt.figure(figsize=(22, 11))
    map_ax = fig.add_axes([0.04, 0.10, 0.70, 0.78])
    list_ax = fig.add_axes([0.755, 0.10, 0.235, 0.78])
    list_ax.axis("off")
    smap.style_map_axes(map_ax)
    fig.text(0.04, 0.965, title, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.925, smap.selection_text(chosen), fontsize=13, color=INK_SOFT, va="top")
    return fig, map_ax, list_ax


def draw_outlines(ax, ships, summary, fill):
    """Draw each cluster's outline and its number on the map.

    The outline is the convex hull of the cluster's ships, padded by half a
    degree so a cluster whose ships share one position is still visible.
    fill=True colours the inside lightly (cluster map); fill=False draws a
    plain outline (hot-spot map, where the hexagons are the fill).
    """
    for _, row in summary.iterrows():
        members = ships[ships["cluster"] == row["cluster"]]
        hull = MultiPoint(members[["lon", "lat"]].to_numpy()).convex_hull.buffer(0.5)
        colour = CONFIRMED_COLOUR if row["confirmed_by_gi_star"] else UNCONFIRMED_COLOUR
        hx, hy = hull.exterior.xy
        if fill:
            ax.fill(hx, hy, color=colour, alpha=0.10, zorder=2)
        ax.plot(hx, hy, color=colour if fill else INK, linewidth=1.3, zorder=4)
        top = members.loc[members["lat"].idxmax()]
        ax.text(top["lon"], max(hy) + 0.3, str(row["cluster"]), ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8))


def draw_cluster_list(list_ax, summary):
    """Write the numbered cluster list in the right-hand panel.

    Each line: a coloured square (confirmed / not confirmed), the number,
    the place, the number of ships, and the share of its ships in hot spots.
    The text itself stays in ink colour; only the square carries colour.
    If there are too many clusters to fit, the rest are in the CSV.
    """
    list_ax.text(0, 1, "Clusters (largest first)", fontsize=14, fontweight="bold",
                 color=INK, va="top", transform=list_ax.transAxes)
    list_ax.text(0, 0.965, "ships  ·  share of its ships in Gi* hot spots",
                 fontsize=10.5, color=INK_SOFT, va="top", transform=list_ax.transAxes)
    # Key to the coloured squares, so the list explains itself on both maps.
    for x, colour, text in [(0, CONFIRMED_COLOUR, "confirmed by Gi*"),
                            (0.5, UNCONFIRMED_COLOUR, "not confirmed")]:
        list_ax.text(x, 0.935, "■", color=colour, fontsize=12, va="top",
                     transform=list_ax.transAxes)
        list_ax.text(x + 0.06, 0.935, text, color=INK_SOFT, fontsize=10.5, va="top",
                     transform=list_ax.transAxes)
    max_lines = 33
    line = 0.035 if len(summary) <= 24 else 0.87 / min(len(summary), max_lines)
    font = 12 if len(summary) <= 24 else 10.5
    for i, row in enumerate(summary.head(max_lines).itertuples()):
        y = 0.895 - i * line
        colour = CONFIRMED_COLOUR if row.confirmed_by_gi_star else UNCONFIRMED_COLOUR
        list_ax.text(0, y, "■", color=colour, fontsize=font + 2, va="top",
                     transform=list_ax.transAxes)
        list_ax.text(0.06, y, f"{row.cluster:>2}  {row.place}", fontsize=font, color=INK,
                     va="top", transform=list_ax.transAxes)
        list_ax.text(1, y, f"{row.ships}  ·  {row.share_in_hot_spots:.0%}", fontsize=font,
                     color=INK_SOFT, va="top", ha="right", transform=list_ax.transAxes)
    if len(summary) > max_lines:
        list_ax.text(0, 0.895 - max_lines * line, f"+ {len(summary) - max_lines} more "
                     "(see the CSV)", fontsize=font, color=INK_SOFT, va="top",
                     transform=list_ax.transAxes)


def map_legend(ax, handles):
    """Legend in the empty sea at the lower left of the map."""
    ax.legend(handles=handles, loc="lower left", fontsize=11.5, frameon=True,
              framealpha=0.92, edgecolor=helpers.GRID)


def status_handles():
    """Legend entries for cluster areas confirmed / not confirmed by Gi*."""
    return [
        Patch(facecolor=helpers.tint(CONFIRMED_COLOUR, 0.6), edgecolor=CONFIRMED_COLOUR,
              label=f"Cluster confirmed by Gi* (≥{CONFIRM_SHARE:.0%} of ships in hot spots)"),
        Patch(facecolor=helpers.tint(UNCONFIRMED_COLOUR, 0.6), edgecolor=UNCONFIRMED_COLOUR,
              label="Cluster not confirmed"),
    ]


def draw_cluster_map(ships, summary, chosen, settings, out_path):
    """Map 1: every ship, coloured by cluster status, with numbered outlines."""
    fig, ax, list_ax = map_figure(chosen, "Where Japanese ship losses cluster (HDBSCAN)")
    confirmed = set(summary.loc[summary["confirmed_by_gi_star"], "cluster"])
    colours = np.where(ships["cluster"] < 0, NOISE_COLOUR,
                       np.where(ships["cluster"].isin(confirmed), CONFIRMED_COLOUR,
                                UNCONFIRMED_COLOUR))
    noise = ships["cluster"] < 0
    # Noise first and faint, so the clusters stand out on top of it.
    ax.scatter(ships.loc[noise, "lon"], ships.loc[noise, "lat"], s=8, color=NOISE_COLOUR,
               alpha=0.45, linewidths=0, zorder=3)
    ax.scatter(ships.loc[~noise, "lon"], ships.loc[~noise, "lat"], s=14,
               c=colours[~noise.to_numpy()], alpha=0.8, edgecolors="white",
               linewidths=0.3, zorder=3)
    draw_outlines(ax, ships, summary, fill=True)
    map_legend(ax, [Line2D([], [], marker="o", linestyle="", color=NOISE_COLOUR,
                           markersize=7, label="Not in any cluster (noise)")]
               + status_handles())
    draw_cluster_list(list_ax, summary)
    footnote(fig, ships, summary, settings)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_hot_spot_map(hexes, ships, summary, chosen, settings, out_path):
    """Map 2: Gi* hot-spot hexagons, with the cluster outlines drawn on top.

    Where an outline sits on dark hexagons, the two methods agree. Only hot
    hexagons are filled; the rest of the study area is left blank so the
    map stays readable.
    """
    fig, ax, list_ax = map_figure(chosen, "Statistical hot spots of ship losses (Getis-Ord Gi*)")
    for column, colour in [("hot_uncorrected", HOT_WEAK), ("hot_fdr", HOT_STRONG)]:
        chosen_hexes = hexes[hexes[column]]
        ax.add_collection(PolyCollection(list(chosen_hexes["corners"]), facecolors=colour,
                                         edgecolors="white", linewidths=0.3, zorder=2))
    draw_outlines(ax, ships, summary, fill=False)
    # Outlines here are all black (the hexagons already use the orange hue),
    # so confirmed / not confirmed is shown only by the squares in the list.
    map_legend(ax, [
        Patch(color=HOT_STRONG, label=f"Hot spot, {1 - ALPHA:.0%} after FDR correction"),
        Patch(color=HOT_WEAK, label=f"Hot spot, {1 - ALPHA:.0%} before correction only"),
        Line2D([], [], color=INK, linewidth=1.3, label="Cluster outline (HDBSCAN), numbered"),
    ])
    draw_cluster_list(list_ax, summary)
    footnote(fig, ships, summary, settings)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def footnote(fig, ships, summary, settings):
    """Explain the settings, what was left out, and the main caveat."""
    noise = (ships["cluster"] < 0).sum()
    fig.text(0.04, 0.012,
             f"Source: JANAC (1947) ship lists. {len(ships):,} ships with a position on the map; "
             f"{len(summary)} clusters hold {len(ships) - noise:,} of them, {noise:,} are noise. "
             f"HDBSCAN: min cluster size {settings['min_cluster_size']}, min samples "
             f"{settings['min_samples'] or settings['min_cluster_size']}, great-circle distance. "
             f"Gi*: {settings['hex_km']} km hexagons, {settings['permutations']:,} permutations, "
             f"Benjamini-Hochberg FDR at {ALPHA}.\n"
             "A hot spot means many ships were sunk there, not that the area was more dangerous "
             "per ship: losses follow shipping routes, and the report has no traffic data.",
             fontsize=10.5, color=INK_SOFT, va="bottom")


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def make_clusters(agent=None, weapon=None, vessel=None, ship_type=None,
                  min_tonnage=None, max_tonnage=None,
                  min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE, min_samples=None,
                  hex_km=DEFAULT_HEX_KM, permutations=DEFAULT_PERMUTATIONS):
    """Run HDBSCAN and Gi* on the selected ships; save 2 maps and a CSV.

    Filters (agent, weapon, vessel, ship_type, min_tonnage, max_tonnage)
    work as in sinking_map.make_map(). Returns the summary table.
    Raises OptionError for a wrong filter or setting, naming it and listing
    the allowed values.
    """
    # Check every filter and setting first, then report all problems at once.
    problems = []
    try:
        chosen = smap.check_filters(agent=agent, weapon=weapon, vessel=vessel,
                                    ship_type=ship_type, min_tonnage=min_tonnage,
                                    max_tonnage=max_tonnage)
    except OptionError as error:
        problems += str(error).splitlines()[1:-1]
    for name, value, minimum in [("min_cluster_size", min_cluster_size, 2),
                                 ("min_samples", min_samples, 1),
                                 ("hex_km", hex_km, 10), ("permutations", permutations, 99)]:
        try:
            check_number(name, value, minimum)
        except (OptionError, TypeError, ValueError):
            problems.append(f"  {value!r} is not a valid {name}. Options for {name}: "
                            f"a whole number of at least {minimum}.")
    if problems:
        raise OptionError("Wrong value(s):\n" + "\n".join(problems))
    settings = dict(min_cluster_size=int(min_cluster_size),
                    min_samples=None if min_samples is None else int(min_samples),
                    hex_km=int(hex_km), permutations=int(permutations))

    helpers.set_readable_style()
    selected = smap.select_ships(smap.load_ships(), chosen)
    # Only ships with a position inside the map window can be clustered; the
    # ships with text-only locations and the two in the Atlantic are left out.
    ships = selected.dropna(subset=["lat", "lon"])
    ships = ships[ships["lon"].between(*smap.LON_RANGE)
                  & ships["lat"].between(*smap.LAT_RANGE)].copy()
    if len(ships) < settings["min_cluster_size"]:
        raise OptionError(f"Only {len(ships)} ships with a position match "
                          f"{smap.selection_text(chosen)}, fewer than min_cluster_size "
                          f"({settings['min_cluster_size']}). Use a smaller min_cluster_size.")
    ships["sunk"] = sinking_dates(ships)

    ships["cluster"] = find_clusters(ships, settings["min_cluster_size"], settings["min_samples"])
    hexes, ship_hex = hot_spots(ships, settings["hex_km"], settings["permutations"])
    summary = summarise(ships, hexes, ship_hex)
    if summary.empty:
        print("HDBSCAN found no clusters; try a smaller --min-cluster-size.")

    # File names carry the settings and the filters, like the other scripts.
    filters = smap.file_name(chosen).replace("sinkings_", "").replace(".png", "")
    mcs = f"mcs-{settings['min_cluster_size']}"
    if settings["min_samples"] is not None:
        mcs += f"_ms-{settings['min_samples']}"
    hex_part = f"hex-{settings['hex_km']}km"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cluster_png = FIG_DIR / f"clusters_{mcs}_{filters}.png"
    hot_png = FIG_DIR / f"hotspots_{hex_part}_{mcs}_{filters}.png"
    csv_path = DATA_DIR / f"clusters_{mcs}_{hex_part}_{filters}.csv"

    draw_cluster_map(ships, summary, chosen, settings, cluster_png)
    draw_hot_spot_map(hexes, ships, summary, chosen, settings, hot_png)
    summary.to_csv(csv_path, index=False)

    confirmed = int(summary["confirmed_by_gi_star"].sum()) if not summary.empty else 0
    print(f"{len(summary)} clusters, {confirmed} confirmed by Gi*")
    for path in (cluster_png, hot_png, csv_path):
        print(f"Saved {path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find clusters of Japanese ship losses (HDBSCAN) and confirm them with "
                    "Getis-Ord Gi* hot spots. Leave a filter out to include all its options.")
    smap.add_filter_arguments(parser)   # --agent ... --max-tonnage, as in sinking_map.py
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE,
                        help=f"smallest number of ships that counts as a cluster "
                             f"(default {DEFAULT_MIN_CLUSTER_SIZE})")
    parser.add_argument("--min-samples", type=int,
                        help="HDBSCAN caution: bigger = more noise (default: min cluster size)")
    parser.add_argument("--hex-km", type=int, default=DEFAULT_HEX_KM,
                        help=f"Gi* hexagon spacing in km (default {DEFAULT_HEX_KM})")
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS,
                        help=f"Gi* permutations for p-values (default {DEFAULT_PERMUTATIONS})")
    parser.add_argument("--options", action="store_true", help="list filter values and stop")
    args = parser.parse_args()

    if args.options:
        print(smap.describe_options())
        sys.exit(0)
    try:
        make_clusters(**smap.filter_arguments(args),
                      min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                      hex_km=args.hex_km, permutations=args.permutations)
    except OptionError as error:
        print(error, file=sys.stderr)
        sys.exit(2)
