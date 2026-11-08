"""
Animated map of Japanese ship losses through time, December 1941 - August 1945.

The war plays out in one minute (by default): each ship appears on the map as
a large bright dot on the day it was sunk, then shrinks and fades into a small
dot that stays, so the map slowly fills up with the whole war's losses. Under
the map, a bar chart of ships sunk per month shows where in the war you are.

It uses the same filters, with the same options and defaults, as
sinking_map.py (whose code it reuses):

    agent      - the nation credited with the sinking (us, britain, australia, ...)
    weapon     - the kind of weapon (aircraft, submarine, mine, ...)
    vessel     - naval (warships) or merchant ships
    ship_type  - the Japanese ship's type (destroyer, carrier, tanker, cargo, ...)
    min_tonnage / max_tonnage - the range of the ship's tonnage

Leave a filter out and it includes all of its options. A wrong value stops
the program with a message that names the filter and lists its options. A
selection with no ship to draw (e.g. agent india + weapon submarine) makes no
video, and says so.

Input (the same as sinking_map.py):
    data/naval/naval_losses_15-44.csv
    data/merchant/merchant_losses_45-115.csv
    data/basemap/ne_50m_land/ne_50m_land.shp

Output: one video per run in figures/animations/, named after the filters, e.g.
    figures/animations/sinkings_agent-all_weapon-all_vessel-all_type-all_tons-all.mp4
    figures/animations/sinkings_agent-us_weapon-submarine_vessel-merchant_type-all_tons-all.mp4

Run (a 60-second MP4 of all ships takes about 3 minutes to render):
    python sinking_animation.py                                  # everything
    python sinking_animation.py --agent us --weapon submarine    # US submarine campaign
    python sinking_animation.py --vessel naval --duration 30     # 30-second video
    python sinking_animation.py --ship-type tanker               # the tanker war
    python sinking_animation.py --min-tonnage 10000              # only big ships
    python sinking_animation.py --agent britain --format gif     # GIF instead of MP4
    python sinking_animation.py --options                        # list every valid value

or from Python:
    from sinking_animation import make_animation
    make_animation(agent="us", weapon="aircraft")
    make_animation(ship_type=["battleship", "carrier"], duration=30)
"""

import argparse
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")  # render straight to the video file; no window is needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

import sinking_map as smap
import table_plot_helpers as helpers
from sinking_map import OptionError, VESSEL_COLOURS, VESSEL_OPTIONS
from table_plot_helpers import BACKGROUND, GRID, INK, INK_SOFT

OUT_DIR = smap.PROJECT_DIR / "figures" / "animations"

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
# The clock runs from the attack on Pearl Harbor to the end of August 1945.
# The first loss in the report is on 8 December 1941 and the last on
# 15 August 1945 (the day Japan announced its surrender).
WAR_START = pd.Timestamp("1941-12-07")
WAR_END = pd.Timestamp("1945-08-31")
DEFAULT_DURATION = 60    # seconds, as asked: the whole war in one minute
DEFAULT_FPS = 20         # frames per second. 20 looks smooth for growing and
                         # fading dots and keeps rendering time reasonable
                         # (1,200 frames); 30 would take 50% longer.
GIF_FPS = 10             # GIFs get big quickly; 10 fps keeps the file manageable
# The last HOLD_SECONDS of the video show the finished map without moving,
# so the viewer has time to take in the final picture.
HOLD_SECONDS = 3

# ---------------------------------------------------------------------------
# How a sinking looks
# ---------------------------------------------------------------------------
# A new sinking appears as a big solid dot and over FADE_DAYS (in war time)
# shrinks and fades to the same small dot as on the still maps. At the default
# speed one second of video is about 24 days of war, so 40 days is a glow of
# about 1.7 seconds: long enough to notice, short enough that only the recent
# fighting stands out.
FADE_DAYS = 40
NEW_DOT_SIZE = 130       # points^2, about 3x the size of a settled dot
NEW_DOT_ALPHA = 1.0

# Figure size. Video frames need an even number of pixels in both directions
# (an H.264 requirement), so the size is chosen in whole, even pixels.
DPI = 100
WIDTH_PX = 1600
# Vertical bands of the figure, from the bottom up, in inches. Fixed sizes
# (not fractions) keep text and the bar chart the same size whatever the
# map's height is.
FOOTNOTE_INCHES = 0.55   # two lines of footnote
TIMELINE_BOTTOM = 0.95   # footnote + room for the timeline's date labels
TIMELINE_HEIGHT = 1.1    # the monthly bar chart itself
MAP_BOTTOM = TIMELINE_BOTTOM + TIMELINE_HEIGHT + 0.45   # + the map's degree labels
TITLE_INCHES = 0.95      # title and selection line above the map


class EmptySelection(ValueError):
    """Raised when the filters match no ship that can be drawn."""


# ---------------------------------------------------------------------------
# Preparing the ships
# ---------------------------------------------------------------------------
def sinking_dates(selected):
    """Return each selected ship's sinking date.

    Six warships have only a month in the report (e.g. "September 1942").
    They are placed on the 15th, the middle of the month, so the error is at
    most about two weeks either way. Of the six, only I 30 has a position,
    so this changes the map for one dot.
    """
    dates = pd.to_datetime(selected["date"], errors="coerce")
    mid_month = pd.to_datetime(dict(year=selected["year"], month=selected["month"], day=15))
    return dates.fillna(mid_month)


def monthly_counts(selected, dates):
    """Count ships sunk per month and vessel kind, for the timeline bars.

    Every month of the war gets a row, even one with no losses, so the bars
    sit on a continuous time axis.
    """
    months = pd.period_range(WAR_START, WAR_END, freq="M")
    counts = (pd.crosstab(dates.dt.to_period("M"), selected["vessel"])
              .reindex(index=months, columns=list(VESSEL_OPTIONS), fill_value=0))
    return counts


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def build_figure(chosen):
    """Lay out the figure: title, the map, and the monthly timeline under it.

    The map keeps the same window, size and style as sinking_map.py, so a
    still map and the last frame of a video can be compared directly.
    """
    width_in = WIDTH_PX / DPI
    left, axes_width = 0.06, 0.92   # left margin leaves room for the y-labels
    map_height_in = width_in * axes_width / smap.map_aspect_ratio()
    height_in = MAP_BOTTOM + map_height_in + TITLE_INCHES
    height_in = round(height_in * DPI / 2) * 2 / DPI   # even pixel height

    fig = plt.figure(figsize=(width_in, height_in), dpi=DPI)
    # Positions are given in inches and turned into figure fractions.
    def frac(inches):
        return inches / height_in
    map_ax = fig.add_axes([left, frac(MAP_BOTTOM), axes_width, frac(map_height_in)])
    time_ax = fig.add_axes([left, frac(TIMELINE_BOTTOM), axes_width, frac(TIMELINE_HEIGHT)])
    smap.style_map_axes(map_ax)

    fig.text(left, 1 - frac(0.15), "Japanese ships sunk, 1941–1945", fontsize=20,
             fontweight="bold", color=INK, va="top")
    fig.text(left, 1 - frac(0.58), smap.selection_text(chosen), fontsize=13,
             color=INK_SOFT, va="top")
    return fig, map_ax, time_ax


def draw_timeline(time_ax, counts):
    """Draw the monthly bars once; only the shading over the future moves.

    The bars are stacked: warships at the bottom, merchant ships on top, in
    the same colours as the map. Instead of growing the bars frame by frame,
    the whole war is drawn at the start and the months still to come are
    covered by a pale band, with a line marking 'today'. The viewer sees at
    once how the current month compares with the rest of the war.
    """
    starts = mdates.date2num(counts.index.to_timestamp())
    width = 26  # days: a month minus a small gap between bars
    bottom = np.zeros(len(counts))
    for vessel in VESSEL_OPTIONS:
        if counts[vessel].sum() == 0:
            continue
        time_ax.bar(starts, counts[vessel], width=width, bottom=bottom, align="edge",
                    color=VESSEL_COLOURS[vessel], linewidth=0)
        bottom += counts[vessel].to_numpy()

    end = mdates.date2num(WAR_END)
    time_ax.set_xlim(mdates.date2num(WAR_START), end)
    time_ax.set_ylim(0, max(bottom.max(), 1) * 1.1)
    time_ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    time_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    time_ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
    time_ax.set_ylabel("Ships sunk\nper month", fontsize=11, color=INK_SOFT)
    time_ax.grid(axis="y", color=GRID, linewidth=0.6)
    time_ax.set_axisbelow(True)
    time_ax.tick_params(labelsize=11)

    # The band over the future and the 'today' line. Both are moved by
    # update() every frame.
    future = time_ax.axvspan(mdates.date2num(WAR_START), end, color=BACKGROUND,
                             alpha=0.82, linewidth=0, zorder=3)
    today = time_ax.axvline(mdates.date2num(WAR_START), color=INK, linewidth=1.5, zorder=4)
    return future, today


def make_animation(agent=None, weapon=None, vessel=None, ship_type=None,
                   min_tonnage=None, max_tonnage=None, duration=DEFAULT_DURATION,
                   fps=None, video_format="mp4"):
    """Check the filters, render the video, and return the path of the file.

    agent, weapon, vessel, ship_type, min_tonnage and max_tonnage work
    exactly as in sinking_map.make_map(): None means all options (or no
    tonnage limit); otherwise give one option or a list, and a number for
    each tonnage limit. duration is the video length in seconds; fps
    defaults to 20 for MP4 and 10 for GIF.

    Raises OptionError for a wrong filter value, and EmptySelection when no
    selected ship can be drawn (no blank videos are made).
    """
    chosen = smap.check_filters(agent=agent, weapon=weapon, vessel=vessel,
                                ship_type=ship_type, min_tonnage=min_tonnage,
                                max_tonnage=max_tonnage)
    if video_format not in ("mp4", "gif"):
        raise OptionError(f"'{video_format}' is not a valid format. Options for format: mp4, gif")
    fps = fps or (GIF_FPS if video_format == "gif" else DEFAULT_FPS)
    helpers.set_readable_style()

    selected = smap.select_ships(smap.load_ships(), chosen)
    drawable = smap.drawable_count(selected)
    if drawable == 0:
        raise EmptySelection(f"No ship can be drawn for {smap.selection_text(chosen)}; "
                             "no video made.")

    dates = sinking_dates(selected)
    counts = monthly_counts(selected, dates)
    fig, map_ax, time_ax = build_figure(chosen)
    future, today = draw_timeline(time_ax, counts)

    # Settled dots look like those on the still map for the same number of
    # ships (sinking_map.DOT_STYLES), so the last frame matches the PNG.
    old_size, old_alpha = next((size, alpha) for limit, size, alpha in smap.DOT_STYLES
                               if drawable <= limit)

    # One scatter per vessel kind holds all of its ships from the start.
    # Each frame only changes their sizes and colours: ships not yet sunk get
    # size 0 (invisible). This is much faster than adding new dots every
    # frame. Ships are sorted by date so newer dots are drawn on top of older
    # ones. Merchant ships are drawn first, warships on top (as on the maps).
    layers = []
    for kind in ["merchant", "naval"]:
        ships = selected[(selected["vessel"] == kind) & selected["lat"].notna()]
        if ships.empty:
            continue
        ship_dates = dates[ships.index]
        order = np.argsort(ship_dates.to_numpy())
        ships = ships.iloc[order]
        days = mdates.date2num(ship_dates.iloc[order])
        rgba = np.tile([*to_rgb(VESSEL_COLOURS[kind]), 0.0], (len(ships), 1))
        dots = map_ax.scatter(ships["lon"], ships["lat"], s=np.zeros(len(ships)),
                              c=rgba, edgecolors="white", linewidths=0.4, zorder=3)
        layers.append((dots, days, rgba))

    # The legend shows the kinds that can be in the selection (kinds_shown()
    # also drops e.g. merchant ships when only warship types were chosen).
    shown = smap.kinds_shown(chosen)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=10,
                      color=VESSEL_COLOURS[kind], label=VESSEL_OPTIONS[kind])
               for kind in ["naval", "merchant"] if kind in shown]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=14, color="#9a9890",
                          label=f"Big dot: sunk in the last {FADE_DAYS} days"))
    map_ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92,
                  edgecolor=GRID, fontsize=12)

    # Big date and running totals, in the empty ocean at the top right.
    date_text = map_ax.text(0.985, 0.97, "", transform=map_ax.transAxes, ha="right",
                            va="top", fontsize=30, fontweight="bold", color=INK, zorder=5)
    total_text = map_ax.text(0.985, 0.865, "", transform=map_ax.transAxes, ha="right",
                             va="top", fontsize=14, color=INK_SOFT, zorder=5,
                             linespacing=1.5)

    # Footnote: what can't be shown, and the tonnage units.
    no_position = len(selected) - drawable
    note = (f"Source: JANAC (1947) ship lists. {len(selected):,} ships match; {drawable:,} "
            "have a position on this map")
    note += (f" ({no_position} have only a text location or lie outside it and are counted "
             "but not drawn)." if no_position else ".")
    note += ("\nNaval tonnage is standard displacement (tons); merchant tonnage is gross "
             "register tons (GRT): different units, so they are never added together.")
    fig.text(0.06, 0.1 / fig.get_figheight(), note, fontsize=10, color=INK_SOFT, va="bottom")

    n_moving = max(int(round((duration - HOLD_SECONDS) * fps)), 2)
    n_frames = int(round(duration * fps))
    start, end = mdates.date2num(WAR_START), mdates.date2num(WAR_END)
    all_days = mdates.date2num(dates)
    tonnage = selected["tonnage"].to_numpy()
    is_naval = (selected["vessel"] == "naval").to_numpy()

    def update(frame):
        """Move the clock to this frame's date and restyle every dot.

        Each dot's age (days since it was sunk) decides its look: not yet
        sunk -> invisible; younger than FADE_DAYS -> between the big new
        dot and the settled dot; older -> the settled dot.
        """
        now = start + (end - start) * min(frame, n_moving - 1) / (n_moving - 1)
        for dots, days, rgba in layers:
            age = now - days
            fade = np.clip(age / FADE_DAYS, 0, 1)        # 0 = brand new, 1 = settled
            sizes = NEW_DOT_SIZE + (old_size - NEW_DOT_SIZE) * fade
            rgba[:, 3] = NEW_DOT_ALPHA + (old_alpha - NEW_DOT_ALPHA) * fade
            sunk = age >= 0
            dots.set_sizes(np.where(sunk, sizes, 0))
            rgba[~sunk, 3] = 0
            dots.set_facecolors(rgba)
            # The white rim would show on invisible dots, so hide it too.
            dots.set_edgecolors(np.where(sunk[:, None], [1, 1, 1, 0.8], [1, 1, 1, 0]))

        # Running totals of everything sunk so far, including ships that
        # can't be drawn, so the numbers end at the selection's full totals.
        done = all_days <= now
        lines = []
        for kind, mask in [("naval", is_naval), ("merchant", ~is_naval)]:
            if kind not in shown:
                continue
            unit = "tons" if kind == "naval" else "GRT"
            lines.append(f"{VESSEL_OPTIONS[kind].split(' (')[0]} sunk: "
                         f"{(done & mask).sum():,}  ({tonnage[done & mask].sum():,.0f} {unit})")
        total_text.set_text("\n".join(lines))
        date_text.set_text(mdates.num2date(now).strftime("%B %Y"))

        # Slide the 'future' band and the 'today' line on the timeline.
        future.set_x(now)
        future.set_width(end - now)
        today.set_xdata([now, now])

    # --- Rendering with blitting. ------------------------------------------
    # The usual FuncAnimation.save() redraws the whole figure for every
    # frame, coastlines and bar chart included: about 0.8 s per frame, or 16
    # minutes for a one-minute video. Almost all of that is the parts that
    # never change. So the figure is drawn once WITHOUT the moving parts
    # ("animated" artists are skipped by a normal draw), that picture is
    # kept, and each frame only pastes it back and draws the moving parts on
    # top. This is matplotlib's standard "blitting" technique, done by hand
    # because save() doesn't use it.
    moving = [dots for dots, _, _ in layers] + [date_text, total_text, future, today]
    for artist in moving:
        artist.set_animated(True)
    canvas = fig.canvas
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)
    width_px, height_px = canvas.get_width_height()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / smap.file_name(chosen).replace(".png", f".{video_format}")
    # The frames are piped straight into ffmpeg (the copy bundled with
    # imageio-ffmpeg) as raw pixels. MP4: H.264 at high quality keeps the
    # small dots and text sharp; yuv420p makes it play in every player.
    # GIF: ffmpeg's GIF encoder, which streams frames instead of holding
    # them all in memory like Pillow would. GIF compresses video badly (a
    # 4-second full-size test was 10 MB), so it is also scaled to half width
    # (800 px); a full minute is still tens of MB. Use MP4 where you can.
    if video_format == "mp4":
        output = dict(codec="libx264", quality=8, pix_fmt_out="yuv420p")
    else:
        output = dict(codec="gif", pix_fmt_out="rgb8",
                      output_params=["-vf", f"scale={WIDTH_PX // 2}:-2"])
    encoder = imageio_ffmpeg.write_frames(str(out_path), (width_px, height_px), fps=fps,
                                          pix_fmt_in="rgb24", macro_block_size=1,
                                          **output)
    encoder.send(None)  # start ffmpeg

    print(f"{len(selected):,} ships match ({smap.selection_text(chosen)}); "
          f"rendering {n_frames} frames...")
    step = max(n_frames // 10, 1)   # print progress every 10%
    try:
        for frame in range(n_frames):
            update(frame)
            canvas.restore_region(background)
            for artist in moving:
                fig.draw_artist(artist)
            # The canvas holds RGBA pixels; ffmpeg was told to expect RGB.
            pixels = np.asarray(canvas.buffer_rgba())[:, :, :3]
            encoder.send(np.ascontiguousarray(pixels).tobytes())
            if (frame + 1) % step == 0:
                print(f"  frame {frame + 1}/{n_frames}")
    finally:
        encoder.close()
        plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    # Same filter arguments as sinking_map.py, plus video settings.
    parser = argparse.ArgumentParser(
        description="Animate where Japanese ships were sunk through the war, filtered by "
                    "agent, weapon, vessel kind, ship type and tonnage. Leave a filter out "
                    "to include all options.")
    smap.add_filter_arguments(parser)   # --agent ... --max-tonnage
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help=f"video length in seconds (default {DEFAULT_DURATION})")
    parser.add_argument("--fps", type=int, help="frames per second (default 20 for mp4, 10 for gif)")
    parser.add_argument("--format", default="mp4", help="mp4 (default) or gif")
    parser.add_argument("--options", action="store_true", help="list all valid values and stop")
    args = parser.parse_args()

    if args.options:
        print(smap.describe_options())
        sys.exit(0)
    try:
        make_animation(**smap.filter_arguments(args), duration=args.duration, fps=args.fps, video_format=args.format)
    except (OptionError, EmptySelection) as error:
        print(error, file=sys.stderr)
        sys.exit(2)
