"""
Maps of the submarine war: every ship sunk by a submarine, the hunting
grounds where the kills were concentrated, and the probable convoy routes
they lay across.

Two figures:
  1. The whole war on one map:
       - dots: every Japanese ship sunk by a submarine (circles = merchant
         ships, triangles = warships);
       - shaded "hunting grounds": the smallest areas of sea that hold 25%,
         50% and 75% of all submarine kills (highest-density regions of a
         kernel density estimate, see hunting_grounds());
       - dashed lines: the main Japanese convoy routes, drawn port to port
         (see ROUTES; they are approximate lanes, not tracked courses);
       - labels on the densest hot spots, named after the nearest place.
  2. Small multiples: the same map for each year 1942-1945, each with its
     own hunting grounds, so the reader sees them move from Japan's coast
     (1942) to the Central Pacific (1943), the Luzon Strait and South China
     Sea (1944) and the Yellow Sea / Sea of Japan (1945).

A ship counts as a submarine kill when at least one of its credits is a
submarine. Its weight in the density is its submarine share (1/n of its
credits, as in the other scripts), so a ship sunk jointly by a submarine and
an aircraft counts one half. By default every nation's submarines are
included (96% of the kills are American); --agent limits it, e.g. --agent us.

Input:
    data/naval/naval_losses_15-44.csv
    data/merchant/merchant_losses_45-115.csv
    data/basemap/ne_50m_land/             (all read through sinking_map.py)

Output (the agent filter is part of the file name):
    figures/submarine_war/submarine_warfare_map_agent-<agent>.png
    figures/submarine_war/submarine_hunting_grounds_by_year_agent-<agent>.png
    data/submarine_war/hot_spots_agent-<agent>.csv     the labelled density peaks

Run:
    python submarine_warfare_map.py                # all Allied submarines
    python submarine_warfare_map.py --agent us     # US submarines only
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # draw straight to PNG files; no window is needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.neighbors import KernelDensity

import sinking_map as smap
import table_plot_helpers as helpers
from shipping_routes import save_figure
from sinking_animation import sinking_dates
from sinking_clusters import place_name
from table_plot_helpers import BACKGROUND, GRID, INK, INK_SOFT, VIOLET, tint

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
FIG_DIR = smap.PROJECT_DIR / "figures" / "submarine_war"
DATA_DIR = smap.PROJECT_DIR / "data" / "submarine_war"

# Map window (longitude on sinking_map's 0-360 scale). Tighter than the
# project's standard map (70-232 E, 30 S-60 N): the submarine war happened
# between the Malacca Strait and the Marshalls, and zooming in gives the
# hot spots room. The few kills outside (Indian Ocean, Aleutians, Atlantic)
# are counted in the footnote.
LON_RANGE = (98, 172)
LAT_RANGE = (-11, 50)

# Kernel density: a Gaussian kernel on the sphere (haversine distance) with
# a 150 km bandwidth. 150 km is about one day's steaming of a slow convoy and
# the scale of a submarine patrol area; a much smaller bandwidth breaks one
# patrol area into many specks, a much larger one merges the Luzon Strait
# with the Formosa Strait. The density is evaluated on a GRID_STEP-degree grid.
BANDWIDTH_KM = 150
EARTH_RADIUS_KM = 6371.0
GRID_STEP = 0.35

# Highest-density regions: the smallest areas that hold this share of the
# (weighted) kills. Darker = more concentrated.
HDR_LEVELS = [0.75, 0.50, 0.25]
HDR_COLOURS = [tint(VIOLET, 0.82), tint(VIOLET, 0.6), tint(VIOLET, 0.35)]
YEAR_HDR_LEVELS = [0.75, 0.50]   # the yearly maps show two levels, to stay readable

# Hot-spot labels: the densest grid peaks, at least PEAK_SEPARATION_KM apart,
# each with the weighted number of kills within PEAK_RADIUS_KM.
MAX_PEAKS = 8
PEAK_SEPARATION_KM = 650
PEAK_RADIUS_KM = 250
# Label offsets (degrees lon, lat) from the peak. Most labels sit up and to
# the right; these two would cover the Manila and Moji / Pusan port names,
# so they are moved to clear water or empty land.
DEFAULT_LABEL_OFFSET = (2.2, 1.6)
LABEL_OFFSETS = {"Manila Bay": (4.8, -0.5), "Nagasaki": (-8.5, 3.2)}

# Dots: one per ship. Dark ink (not a colour) so they read on top of the
# violet hunting grounds; merchant circles and warship triangles.
DOT_COLOUR = INK
ROUTE_COLOUR = "#7a6f5a"   # warm grey-brown: reads as "lanes on a chart", not as data

# ---------------------------------------------------------------------------
# Probable convoy routes
# ---------------------------------------------------------------------------
# Each route is a list of (latitude, longitude) waypoints between the main
# ports, with extra points to keep the line at sea. These are APPROXIMATE
# lanes, drawn from the ports the convoys used (Moji, Takao, Manila,
# Singapore, Palembang, Balikpapan, Truk, Rabaul, ...) and from where the
# sinkings cluster; JANAC records where ships sank, not the courses they
# steered, so the lines are an interpretation, said so on the map.
PORTS = {
    "Moji": (33.95, 130.95), "Tokyo Bay": (35.1, 139.8), "Kobe": (34.6, 135.2),
    "Takao": (22.6, 120.25), "Manila": (14.45, 120.6), "Singapore": (1.25, 104.0),
    "Palembang": (-2.4, 104.9), "Balikpapan": (-1.3, 116.9), "Miri": (4.45, 113.9),
    "Shanghai": (31.0, 122.2), "Dairen": (38.9, 121.7), "Pusan": (35.05, 129.05),
    "Truk": (7.4, 151.8), "Saipan": (15.2, 145.7), "Palau": (7.35, 134.45),
    "Rabaul": (-4.2, 152.2), "Wewak": (-3.5, 143.6), "Cam Ranh": (11.9, 109.25),
    "Hainan": (18.2, 109.55), "Rangoon": (16.4, 96.3), "Paramushiro": (50.3, 156.1),
}
ROUTES = {
    # Japan - Formosa - (Hainan / Indochina coast) - Singapore: the oil route.
    "Southern oil route": [PORTS["Moji"], (33.7, 130.2), (32.4, 128.3), (27.5, 123.0),
                           (25.4, 121.6), (24.0, 119.9), PORTS["Takao"], (20.3, 116.0),
                           PORTS["Hainan"], (15.0, 110.0), PORTS["Cam Ranh"], (8.0, 106.5),
                           (3.0, 104.8), PORTS["Singapore"]],
    # Takao - Luzon Strait - Manila - west of Palawan - Miri (Borneo oil) - Singapore.
    "Philippines - Borneo route": [PORTS["Takao"], (19.5, 120.0), (16.0, 119.75),
                                   PORTS["Manila"], (12.0, 119.5), (10.0, 116.5),
                                   PORTS["Miri"], (2.5, 109.0), PORTS["Singapore"]],
    # Singapore - Palembang (Sumatran oil) and the Borneo oil port Balikpapan
    # north through the Makassar Strait and the Sulu Sea to Manila.
    "NEI oil feeders": [PORTS["Palembang"], (-1.8, 105.6), (0.0, 104.6), PORTS["Singapore"]],
    "Balikpapan - Manila": [PORTS["Balikpapan"], (1.0, 118.6), (5.0, 119.8), (7.5, 121.5),
                            (10.5, 121.3), (12.3, 120.1), (13.6, 120.3), PORTS["Manila"]],
    # Tokyo Bay - Bonins - Saipan - Truk - Rabaul: the Central Pacific supply line.
    "Central Pacific route": [PORTS["Tokyo Bay"], (33.0, 140.5), (27.1, 142.3),
                              PORTS["Saipan"], PORTS["Truk"], (0.0, 151.0), PORTS["Rabaul"]],
    # Palau, the hub between the Philippines and the New Guinea / Carolines fronts.
    # Manila to Palau through the Sibuyan Sea and the San Bernardino Strait.
    "Palau routes": [PORTS["Manila"], (13.9, 120.5), (12.7, 121.6), (12.3, 123.2),
                     (12.6, 124.2), (12.0, 126.5), PORTS["Palau"], PORTS["Saipan"]],
    "Palau - New Guinea": [PORTS["Palau"], (2.0, 138.0), PORTS["Wewak"]],
    # Japan's inner routes to China, Korea and Manchuria.
    "Korea route": [PORTS["Moji"], (34.4, 129.9), PORTS["Pusan"]],
    "Shanghai route": [(32.4, 128.3), PORTS["Shanghai"]],
    "Yellow Sea route": [(32.4, 128.3), (33.8, 125.3), (36.5, 123.6), PORTS["Dairen"]],
    # The Pacific coast of Honshu and the Inland Sea, Tokyo to Moji.
    "Coastal route": [PORTS["Tokyo Bay"], (34.5, 138.5), (33.4, 136.0), (33.8, 135.0),
                      PORTS["Kobe"], (34.3, 133.5), (33.95, 132.0), PORTS["Moji"]],
    # North to the Kuriles.
    "Kurile route": [PORTS["Tokyo Bay"], (36.5, 141.5), (39.5, 142.5), (42.0, 144.5),
                     (44.5, 148.5), PORTS["Paramushiro"]],
}
# Where each route's name is written (lat, lon) and its rotation in degrees.
# Only the main routes are named, to keep the map readable.
ROUTE_LABELS = {
    "Southern oil route": ((19.75, 113.6), 17),
    "Philippines - Borneo route": ((8.6, 115.2), 47),
    "Central Pacific route": ((22.0, 143.0), -76),
    "Yellow Sea route": ((35.3, 123.0), 55),
    "Kurile route": ((45.0, 150.5), 38),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def submarine_kills(agents):
    """Ships with at least one submarine credit from the chosen agents, with their weight.

    weight = the ship's submarine share from those agents (1/n per credit).
    Positions without lat/lon (text locations) are kept here, so the
    footnote can count them, and dropped where a position is needed.
    """
    ships = smap.load_ships()
    ships["weight"] = ships["credits"].apply(
        lambda credits: sum(1 for a, w in credits if w == "submarine" and a in agents)
        / len(credits))
    kills = ships[ships["weight"] > 0].copy()
    kills["sunk"] = sinking_dates(kills)
    kills["year"] = kills["sunk"].dt.year.clip(lower=1942)   # Dec 1941 joins 1942
    return kills


# ---------------------------------------------------------------------------
# Density and hunting grounds
# ---------------------------------------------------------------------------
def density_grid(kills):
    """Weighted Gaussian kernel density of kill positions on a lat/lon grid.

    Distances are great-circle (haversine), so a degree of longitude near
    the equator and near Japan is treated at its true length; the bandwidth
    is BANDWIDTH_KM expressed in radians. Returns (lon grid, lat grid,
    density on the grid, density at each kill, weights).
    """
    placed = kills.dropna(subset=["lat", "lon"])
    points = np.radians(placed[["lat", "lon"]].to_numpy())
    weights = placed["weight"].to_numpy()
    kde = KernelDensity(bandwidth=BANDWIDTH_KM / EARTH_RADIUS_KM, metric="haversine",
                        kernel="gaussian", algorithm="ball_tree")
    kde.fit(points, sample_weight=weights)
    lons = np.arange(LON_RANGE[0], LON_RANGE[1] + GRID_STEP, GRID_STEP)
    lats = np.arange(LAT_RANGE[0], LAT_RANGE[1] + GRID_STEP, GRID_STEP)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid = np.radians(np.column_stack([lat_grid.ravel(), lon_grid.ravel()]))
    density = np.exp(kde.score_samples(grid)).reshape(lon_grid.shape)
    at_points = np.exp(kde.score_samples(points))
    return lon_grid, lat_grid, density, at_points, weights


def hdr_thresholds(at_points, weights, levels):
    """Density thresholds of the highest-density regions holding each share of kills.

    The region {density >= c} holds share p of the kills when c is the
    weighted (1 - p) quantile of the density at the kills themselves
    (Hyndman, 1996). Using the kills, not the grid, makes "50%" mean "half
    of the actual sinkings", not half of an abstract probability mass.
    """
    order = np.argsort(at_points)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return [float(at_points[order][np.searchsorted(cumulative, 1 - p)]) for p in levels]


def hot_spots(kills, lon_grid, lat_grid, density):
    """The densest local peaks, at least PEAK_SEPARATION_KM apart, named and counted.

    Grid cells are visited from the densest down; a cell becomes a peak if
    no stronger peak is within PEAK_SEPARATION_KM (a simple non-maximum
    suppression). Each peak gets the nearest place name from
    sinking_clusters.PLACES and the weighted number of kills within
    PEAK_RADIUS_KM.
    """
    placed = kills.dropna(subset=["lat", "lon"])
    flat = np.argsort(density.ravel())[::-1]
    peaks = []
    for index in flat[:5000]:
        lat, lon = lat_grid.ravel()[index], lon_grid.ravel()[index]
        if all(haversine_km(lat, lon, p["lat"], p["lon"]) >= PEAK_SEPARATION_KM for p in peaks):
            near = haversine_km(lat, lon, placed["lat"].to_numpy(), placed["lon"].to_numpy())
            inside = near <= PEAK_RADIUS_KM
            peaks.append({"lat": lat, "lon": lon,
                          "place": place_name(lat, lon if lon <= 180 else lon - 360),
                          "kills_within_radius": placed.loc[inside, "weight"].sum(),
                          "first_year": int(placed.loc[inside, "year"].min()),
                          "peak_year": int(placed.loc[inside].groupby("year")["weight"].sum()
                                           .idxmax()),
                          "density": density.ravel()[index]})
        if len(peaks) == MAX_PEAKS:
            break
    return pd.DataFrame(peaks)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (works with numbers or numpy arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def base_map(ax):
    """Land, ticks and grid from sinking_map, zoomed to the submarine window."""
    smap.style_map_axes(ax)
    ax.set_xlim(*LON_RANGE)
    ax.set_ylim(*LAT_RANGE)


def draw_hunting_grounds(ax, lon_grid, lat_grid, density, thresholds, colours):
    """Filled contours of the highest-density regions, under the land.

    zorder 0.9 puts them below the land polygons (zorder 1), so a hunting
    ground near a coast is shown only over the sea, where the ships were.
    """
    levels = list(thresholds) + [density.max() * 1.01]
    ax.contourf(lon_grid, lat_grid, density, levels=levels, colors=colours, zorder=0.9)


def draw_routes(ax, label=True, alpha=1.0):
    """Draw the probable convoy routes as dashed lines, and name the main ones."""
    for name, points in ROUTES.items():
        lats, lons = zip(*points)
        ax.plot(lons, lats, color=ROUTE_COLOUR, linewidth=1.5, linestyle=(0, (5, 3)),
                alpha=alpha, zorder=2, solid_capstyle="round")
    for (lat, lon) in PORTS.values():
        if LON_RANGE[0] <= lon <= LON_RANGE[1] and LAT_RANGE[0] <= lat <= LAT_RANGE[1]:
            ax.plot(lon, lat, marker="s", markersize=3.5, color=ROUTE_COLOUR, alpha=alpha,
                    zorder=2)
    if label:
        for name, ((lat, lon), angle) in ROUTE_LABELS.items():
            ax.text(lon, lat, name, rotation=angle, fontsize=9.5, color=ROUTE_COLOUR,
                    fontstyle="italic", ha="center", va="center", zorder=4,
                    rotation_mode="anchor")
        for name, (lat, lon) in PORTS.items():
            if LON_RANGE[0] <= lon <= LON_RANGE[1] and LAT_RANGE[0] <= lat <= LAT_RANGE[1]:
                ax.text(lon + 0.5, lat - 0.2, name, fontsize=8, color=ROUTE_COLOUR, zorder=4,
                        va="top")


def draw_kills(ax, kills, size=10):
    """One dot per placed kill: circles merchant ships, triangles warships."""
    for vessel, marker in [("merchant", "o"), ("naval", "^")]:
        part = kills[kills["vessel"] == vessel]
        ax.scatter(part["lon"], part["lat"], s=size * (1.5 if vessel == "naval" else 1),
                   marker=marker, color=DOT_COLOUR, alpha=0.55, linewidths=0, zorder=3)


def selection_note(agents):
    """Words for the submarines included, e.g. 'All Allied submarines' or 'US submarines'."""
    if set(agents) == set(smap.AGENT_OPTIONS):
        return "All Allied submarines (96% of kills American)"
    return ", ".join(smap.AGENT_OPTIONS[a] for a in agents) + " submarines"


def footnote_counts(kills):
    """How many kills are drawn, and how many are left out (no position / outside window)."""
    placed = kills.dropna(subset=["lat", "lon"])
    inside = placed["lon"].between(*LON_RANGE) & placed["lat"].between(*LAT_RANGE)
    return len(kills), int(inside.sum()), len(kills) - len(placed), int((~inside).sum())


def draw_war_map(kills, grid, peaks, thresholds, agents, out_path):
    """Figure 1: the whole war, with hunting grounds, routes, kills and hot-spot labels."""
    lon_grid, lat_grid, density = grid
    fig, ax = plt.subplots(figsize=(15, 13.2))
    base_map(ax)
    draw_hunting_grounds(ax, lon_grid, lat_grid, density, thresholds, HDR_COLOURS)
    draw_routes(ax)
    draw_kills(ax, kills.dropna(subset=["lat", "lon"]))
    for rank, peak in enumerate(peaks.itertuples(), start=1):
        dx, dy = LABEL_OFFSETS.get(peak.place, DEFAULT_LABEL_OFFSET)
        ax.annotate(f"{rank}. {peak.place}\n{peak.kills_within_radius:.0f} kills, "
                    f"peak {peak.peak_year}",
                    xy=(peak.lon, peak.lat), xytext=(peak.lon + dx, peak.lat + dy),
                    fontsize=9.5, color=INK, fontweight="bold", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.8),
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=BACKGROUND,
                              edgecolor=GRID, alpha=0.92))
    total, drawn, no_position, outside = footnote_counts(kills)
    merchant = kills[kills["vessel"] == "merchant"]
    naval = kills[kills["vessel"] == "naval"]
    handles = [Patch(color=c, label=f"Hunting ground: densest area holding {p:.0%} of kills")
               for p, c in zip(HDR_LEVELS[::-1], HDR_COLOURS[::-1])]
    handles += [Line2D([], [], marker="o", linestyle="", color=DOT_COLOUR, alpha=0.6,
                       markersize=6, label=f"Merchant ship sunk ({len(merchant):,})"),
                Line2D([], [], marker="^", linestyle="", color=DOT_COLOUR, alpha=0.6,
                       markersize=7, label=f"Warship sunk ({len(naval):,})"),
                Line2D([], [], color=ROUTE_COLOUR, linewidth=1.5, linestyle=(0, (5, 3)),
                       marker="s", markersize=3.5, label="Probable convoy route and port")]
    # Legend in the open Pacific east of the Bonins (no kills, no route), so
    # it hides nothing; in the lower right it covered Rabaul.
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.66), frameon=True,
              framealpha=0.93, edgecolor=GRID, fontsize=10.5)
    ax.set_title("The submarine war: Japanese ships sunk by submarines, 1941–1945\n",
                 loc="left", fontsize=18, fontweight="bold", color=INK)
    ax.text(0, 1.012, f"{selection_note(agents)}. {total:,} ships: {len(merchant):,} merchant "
            f"({merchant['tonnage'].sum() / 1e6:.2f}M GRT), {len(naval):,} warships "
            f"({naval['tonnage'].sum() / 1e3:,.0f}k standard tons).",
            transform=ax.transAxes, fontsize=12, color=INK_SOFT)
    fig.text(0.02, 0.012,
             f"Source: JANAC (1947). Hunting grounds: weighted Gaussian kernel density "
             f"(great-circle distance, {BANDWIDTH_KM} km bandwidth), highest-density regions; "
             f"joint kills weighted by their submarine share. Labels: densest peaks at least "
             f"{PEAK_SEPARATION_KM} km apart, kills within {PEAK_RADIUS_KM} km.\nRoutes are "
             f"approximate lanes between the main ports, drawn for orientation; JANAC records "
             f"where ships sank, not their courses. {drawn:,} kills drawn; {no_position} have "
             f"only a text location and {outside} lie outside this window (Indian Ocean, "
             "Aleutians, Atlantic).", fontsize=9.5, color=INK_SOFT)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.07)
    save_figure(fig, out_path)


def draw_year_maps(kills, agents, out_path):
    """Figure 2: one map per year, each with that year's own hunting grounds.

    Each year's density is computed from that year's kills only, so the
    shaded areas show where that year's kills were concentrated (the
    shading is relative within the year; the kill count in each title gives
    the absolute size). Routes are drawn faintly and unlabelled for
    orientation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(17, 15.5))
    for ax, year in zip(axes.ravel(), [1942, 1943, 1944, 1945]):
        part = kills[kills["year"] == year]
        base_map(ax)
        lon_grid, lat_grid, density, at_points, weights = density_grid(part)
        thresholds = hdr_thresholds(at_points, weights, YEAR_HDR_LEVELS)
        draw_hunting_grounds(ax, lon_grid, lat_grid, density, thresholds,
                             [HDR_COLOURS[0], HDR_COLOURS[2]])
        draw_routes(ax, label=False, alpha=0.55)
        draw_kills(ax, part.dropna(subset=["lat", "lon"]), size=8)
        merchant = part[part["vessel"] == "merchant"]
        naval = part[part["vessel"] == "naval"]
        peaks = hot_spots(part, lon_grid, lat_grid, density).head(3)
        ax.set_title(f"{year if year > 1942 else '1942 (and Dec 1941)'}: "
                     f"{len(merchant):,} merchant ships ({merchant['tonnage'].sum() / 1e6:.2f}M "
                     f"GRT), {len(naval)} warships", loc="left", fontsize=13, color=INK)
        ax.text(0.01, 0.015, "Densest: " + "; ".join(peaks["place"]), transform=ax.transAxes,
                fontsize=10, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=BACKGROUND, edgecolor=GRID,
                          alpha=0.92))
        ax.tick_params(labelsize=9)
    handles = [Patch(color=HDR_COLOURS[2], label="Densest area holding 50% of the year's kills"),
               Patch(color=HDR_COLOURS[0], label="... holding 75%"),
               Line2D([], [], marker="o", linestyle="", color=DOT_COLOUR, alpha=0.6,
                      markersize=6, label="Merchant ship sunk"),
               Line2D([], [], marker="^", linestyle="", color=DOT_COLOUR, alpha=0.6,
                      markersize=7, label="Warship sunk"),
               Line2D([], [], color=ROUTE_COLOUR, linewidth=1.5, linestyle=(0, (5, 3)),
                      label="Probable convoy route")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"The hunting grounds move: where submarines sank Japanese ships, year by "
                 f"year ({selection_note(agents).split(' (')[0]})", x=0.04, ha="left", y=0.985,
                 fontsize=18, fontweight="bold", color=INK)
    fig.text(0.04, 0.008, f"Source: JANAC (1947). Each year's shading comes from that year's "
             f"kills only (weighted kernel density, {BANDWIDTH_KM} km bandwidth), so it shows "
             "where the year's kills were concentrated; the title gives the year's size.",
             fontsize=10, color=INK_SOFT)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.905, bottom=0.04, hspace=0.14, wspace=0.08)
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def main(agent=None):
    """Build both maps and the hot-spot table for the chosen submarine agents."""
    chosen = smap.check_filters(agent=agent)
    agents = chosen["agent"]
    helpers.set_readable_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = "all" if len(agents) == len(smap.AGENT_OPTIONS) else "+".join(agents)

    kills = submarine_kills(agents)
    if kills.dropna(subset=["lat", "lon"]).shape[0] < 30:
        raise smap.OptionError(f"Only {len(kills)} submarine kills for agent {tag}; "
                               "at least 30 are needed for a density map.")
    lon_grid, lat_grid, density, at_points, weights = density_grid(kills)
    thresholds = hdr_thresholds(at_points, weights, HDR_LEVELS)
    peaks = hot_spots(kills, lon_grid, lat_grid, density)
    peaks.to_csv(DATA_DIR / f"hot_spots_agent-{tag}.csv", index=False, float_format="%.4g")

    draw_war_map(kills, (lon_grid, lat_grid, density), peaks, thresholds, agents,
                 FIG_DIR / f"submarine_warfare_map_agent-{tag}.png")
    draw_year_maps(kills, agents, FIG_DIR / f"submarine_hunting_grounds_by_year_agent-{tag}.png")
    print(peaks.round(2).to_string(index=False))
    print(f"Saved maps to {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Maps of the submarine war: kills, hunting grounds and probable routes.")
    parser.add_argument("--agent", nargs="+",
                        help="whose submarines, e.g. us britain (default: all)")
    args = parser.parse_args()
    try:
        main(agent=args.agent)
    except smap.OptionError as error:
        print(error, file=sys.stderr)
        sys.exit(2)
