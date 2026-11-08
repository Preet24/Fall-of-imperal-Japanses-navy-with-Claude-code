"""
Interactive explorer for the JANAC (1947) list of Japanese ships sunk in WWII.

A Streamlit web app. The sidebar holds the filters; the main page shows the
selected ships as headline numbers, a map, monthly losses, breakdowns and a
table you can download.

Filters (all optional; leaving one empty means "all"):
    fleet         - warships (naval), merchant ships, or both
    ship type     - the Japanese ship's class (destroyer, tanker, cargo, ...);
                    the list only offers classes that belong to the chosen fleet
    weapon        - aircraft, submarine, surface warship, mine, ...
    nation        - the country credited with the sinking (US, Britain, ...)
    min / max tonnage - ship size limits, both ends included
    months        - a date range between Dec 1941 and Aug 1945
    assessment    - whether to include ships the report lists as "Probably sunk"

The filter definitions (options, aliases, the OCR fixes for the agent cells,
and the rule for matching shared sinkings) are NOT copied here: the app
imports them from sinking_map.py, so the app, the maps, the animation and the
cluster analysis always select exactly the same ships for the same filters.

Input (built by ship_lists/naval_pages.py and ship_lists/merchant_pages.py,
read through sinking_map.load_ships()):
    data/naval/naval_losses_15-44.csv
    data/merchant/merchant_losses_45-115.csv

Output: none on disk. The ship table can be downloaded from the app as a CSV.

Run it (it opens in your browser at http://localhost:8501):
    python -m streamlit run janac_explorer_app.py

Data caveats shown in the app as well:
  - Naval tonnage is standard displacement; merchant tonnage is gross register
    tons (GRT). They measure different things, so the app never adds them: every
    tonnage number and chart is given per fleet.
  - 38 ships have no printed latitude/longitude (e.g. "off Japan") and cannot be
    drawn on the map; they are still counted everywhere else. 6 warships have
    no day of the month; they still have a month, which is all the app uses.
  - A ship sunk by several agents together (e.g. US aircraft + a British
    submarine) is ONE ship. It matches a filter when any one of its credits
    matches. In the "by weapon" and "by nation" charts each credit gets an
    equal share (1/2 each for two agents), the same rule as
    us_loss_time_series.py, so the bars add up to the number of ships.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sinking_map as sm
from table_plot_helpers import AQUA, BLUE, GREEN, GREY, INK, ORANGE, PINK, VIOLET, YELLOW

# ---------------------------------------------------------------------------
# Labels and colours
# ---------------------------------------------------------------------------
# Short readable labels for the option keys of sinking_map.py. The long
# descriptions in sinking_map (e.g. "Aircraft (carrier, land-based, Army,
# unidentified)") are too long for a dropdown, so they appear as help text.
FLEET_LABELS = {"naval": "Warships (naval)", "merchant": "Merchant ships"}
WEAPON_LABELS = {"aircraft": "Aircraft", "submarine": "Submarine",
                 "surface": "Surface warship", "mine": "Mine",
                 "shore_battery": "Shore battery", "sabotage": "Sabotage",
                 "accident": "Accident", "unknown": "Unknown"}
AGENT_LABELS = {"us": "United States", "britain": "Great Britain",
                "netherlands": "Netherlands", "australia": "Australia",
                "new_zealand": "New Zealand", "india": "India", "china": "China",
                "russia": "Russia",
                # "japan" in sinking_map means losses the report credits to
                # Japan itself: its own minefields and accidents. Spelled out
                # so nobody reads it as "sunk by the Japanese navy in battle".
                "japan": "Japan (own mines, accidents)", "unknown": "Unknown"}
SHIP_TYPE_LABELS = {"battleship": "Battleship", "carrier": "Aircraft carrier",
                    "cruiser": "Cruiser", "destroyer": "Destroyer",
                    "submarine": "Submarine", "escort": "Escort",
                    "mine_warfare": "Minesweeper / minelayer",
                    "landing_ship": "Landing ship", "naval_auxiliary": "Naval auxiliary",
                    "cargo": "Cargo", "passenger_cargo": "Passenger-cargo",
                    "tanker": "Tanker", "transport": "Transport / ferry",
                    "converted_auxiliary": "Converted auxiliary",
                    "other_merchant": "Other merchant"}

# Fleet colours are the same as on the static maps (sinking_map.VESSEL_COLOURS):
# blue for warships, orange for merchant ships.
FLEET_COLOURS = sm.VESSEL_COLOURS
# Weapon colours follow table_2_plots.py for the four main weapons
# (submarine blue, surface orange, aircraft aqua, mines green), so a reader
# who has seen those charts finds the same colour for the same weapon. The
# four small groups take the remaining palette colours; "unknown" gets grey,
# which the palette keeps for leftover categories.
WEAPON_COLOURS = {"submarine": BLUE, "surface": ORANGE, "aircraft": AQUA, "mine": GREEN,
                  "shore_battery": YELLOW, "sabotage": PINK, "accident": VIOLET,
                  "unknown": GREY}
# Units of each fleet's tonnage, written on every tonnage number and axis.
TONNAGE_UNITS = {"naval": "tons (standard)", "merchant": "GRT"}
# Light grey for chart grid lines: visible, but quieter than the data.
GRID_COLOUR = "#eeeeee"

# The war months covered by the report: the first loss is in December 1941
# and the last in August 1945. Used for the month slider, and so the monthly
# charts show quiet months as zero instead of skipping them.
FIRST_MONTH = pd.Period("1941-12", "M")
LAST_MONTH = pd.Period("1945-08", "M")
ALL_MONTHS = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")

# Map view: the ship positions run from the Bay of Bengal (83 E) to Alaska
# (135 W), so the globe is turned to put 155 E in the middle. That keeps the
# whole theatre in one piece instead of splitting it at the date line (the
# same reason sinking_map.py draws longitudes 0-360).
MAP_CENTRE = {"lat": 15, "lon": 155}
# Zoom of the projection (1 = whole world). 2.6 fits roughly the same window
# as sinking_map.py's LON_RANGE / LAT_RANGE.
MAP_SCALE = 2.6

# Columns shown in the ship table and in the downloaded CSV, with their
# headings. The raw OCR columns are left out: they are for checking the
# extraction, not for exploring the losses.
TABLE_COLUMNS = {"date_text": "Date", "vessel_name": "Ship", "fleet": "Fleet",
                 "vessel_type": "Type (report)", "ship_type_label": "Type group",
                 "tonnage": "Tonnage", "tonnage_unit": "Unit",
                 "tonnage_estimated": "Tonnage estimated",
                 "location": "Location", "lat": "Latitude", "lon_original": "Longitude",
                 "flag_of_agent": "Flag of agent", "agent_type": "Agent",
                 "agent_probable": "Agent probable", "assessment": "Assessment",
                 "page": "JANAC page"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load both ship lists once and add the columns the app needs.

    st.cache_data keeps the result between reruns. Streamlit re-runs the
    whole script every time a filter changes; without the cache it would
    re-read the CSVs and re-parse ~3,000 agent credits each time.

    Added columns:
      month        - month of sinking as a Period, built from year/month
                     rather than 'date': 6 warships have no day and so no
                     full date, but they do have a month.
      fleet        - copy of sinking_map's 'vessel' column, named for the app.
      lon_original - longitude back in -180..180. load_ships() moves western
                     longitudes to 180..360 for its matplotlib map; plotly's
                     globe and the downloaded table use the usual -180..180.
      date_text    - a readable date, "1944-02-17", or "1944-02 (day not
                     printed)" for the 6 warships without a day.
    """
    ships = sm.load_ships()
    # KENKOKU (p54) is printed with only its name and tonnage, so it has no
    # month. It is kept, but only shown when the month range is the whole war.
    ships["month"] = [
        pd.Period(year=int(y), month=int(m), freq="M")
        if pd.notna(y) and pd.notna(m) else pd.NaT
        for y, m in zip(ships["year"], ships["month"])]
    ships["fleet"] = ships["vessel"]
    ships["lon_original"] = ships["lon"].where(ships["lon"] <= 180, ships["lon"] - 360)
    ships["date_text"] = ships["date"].fillna(
        ships["month"].astype(str) + " (day not printed)")
    ships["ship_type_label"] = ships["ship_type"].map(SHIP_TYPE_LABELS)
    ships["tonnage_unit"] = ships["fleet"].map({"naval": "tons", "merchant": "GRT"})
    return ships


def explode_credits(selected, chosen):
    """One row per (ship, agent, weapon) credit, with a 'share' weight.

    Each credit of a ship gets share = 1 / (number of credits), so a ship sunk
    by US aircraft and a US submarine counts 0.5 to aircraft and 0.5 to
    submarines. The shares of one ship add up to 1, so a chart by weapon
    adds up to the number of ships (same rule as us_loss_time_series.py).

    Credits that do not match the nation and weapon filters are then dropped:
    if you ask for "US" only, the British half of a shared sinking is not
    drawn as a "Great Britain" bar. So with shared sinkings selected the bars
    can add up to a little less than the ship count; the caption says so.
    """
    rows = selected[["vessel_name", "fleet", "tonnage", "month", "credits"]].copy()
    rows["share"] = 1 / rows["credits"].str.len()
    rows = rows.explode("credits")
    rows["agent"] = rows["credits"].str[0]
    rows["weapon"] = rows["credits"].str[1]
    rows = rows[rows["agent"].isin(chosen["agent"]) & rows["weapon"].isin(chosen["weapon"])]
    rows["share_tonnage"] = rows["share"] * rows["tonnage"]
    return rows


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
def sidebar_filters():
    """Draw the filter widgets and return the chosen values.

    Multiselects start empty, and empty means "all", the same rule as the
    command-line scripts. The alternative, starting with every option
    ticked, fills the sidebar with 15 chips for ship type alone and makes
    "show only tankers" a matter of deleting 14 of them.
    """
    st.sidebar.header("Filters")

    fleet_choice = st.sidebar.radio(
        "Fleet", ["Both", "Warships (naval)", "Merchant ships"],
        help="Warships are the Imperial Japanese Navy's own vessels. Merchant "
             "ships include those taken into navy service as auxiliaries.")
    fleets = {"Both": ["naval", "merchant"], "Warships (naval)": ["naval"],
              "Merchant ships": ["merchant"]}[fleet_choice]

    # Only offer ship types that belong to the chosen fleet: "Merchant ships"
    # plus "Destroyer" could never match a ship.
    type_options = [key for key in sm.SHIP_TYPE_OPTIONS
                    if any(key in sm.SHIP_TYPE_OF[f].values() for f in fleets)]
    ship_types = st.sidebar.multiselect(
        "Japanese ship type", type_options, format_func=SHIP_TYPE_LABELS.get,
        placeholder="All types",
        help="Groups of the report's ~50 vessel types:\n\n"
             + "\n\n".join(f"- **{SHIP_TYPE_LABELS[k]}**: {sm.SHIP_TYPE_OPTIONS[k]}"
                           for k in type_options))

    weapons = st.sidebar.multiselect(
        "Weapon", list(sm.WEAPON_OPTIONS), format_func=WEAPON_LABELS.get,
        placeholder="All weapons",
        help="Aircraft includes carrier-based, land-based, Army and unidentified "
             "aircraft. Mine includes Allied and Japanese mines.")
    agents = st.sidebar.multiselect(
        "Sunk by (nation)", list(sm.AGENT_OPTIONS), format_func=AGENT_LABELS.get,
        placeholder="All nations")

    # Tonnage limits. value=None shows an empty box, meaning "no limit". A
    # slider was rejected: tonnages run from about 10 to 72,000, and a linear
    # slider gives no fine control over the small ships that make up most
    # of the losses.
    st.sidebar.markdown("**Tonnage**")
    col_min, col_max = st.sidebar.columns(2)
    min_tonnage = col_min.number_input("Min", min_value=0, value=None, step=500,
                                       placeholder="no limit")
    max_tonnage = col_max.number_input("Max", min_value=0, value=None, step=500,
                                       placeholder="no limit")
    st.sidebar.caption("Warships: standard tons. Merchant ships: gross register "
                       "tons (GRT). Both ends are included.")

    months = st.sidebar.select_slider(
        "Months", options=list(ALL_MONTHS), value=(FIRST_MONTH, LAST_MONTH),
        format_func=lambda p: p.strftime("%b %Y"))

    include_probable = st.sidebar.checkbox(
        "Include \"Probably sunk\"", value=True,
        help="The report assesses 39 ships as only probably sunk "
             "(11 warships, 28 merchant ships).")

    return {"fleets": fleets, "ship_types": ship_types, "weapons": weapons,
            "agents": agents, "min_tonnage": min_tonnage, "max_tonnage": max_tonnage,
            "months": months, "include_probable": include_probable}


def apply_filters(ships, picks):
    """Select the ships matching the sidebar choices.

    Fleet, ship type, weapon, nation and tonnage go through
    sinking_map.check_filters() and select_ships(), so they behave exactly
    like the command-line scripts, including the min > max check, which
    raises OptionError. The month range and the assessment filter exist only
    in the app and are applied afterwards.
    """
    chosen = sm.check_filters(agent=picks["agents"], weapon=picks["weapons"],
                              vessel=picks["fleets"], ship_type=picks["ship_types"],
                              min_tonnage=picks["min_tonnage"],
                              max_tonnage=picks["max_tonnage"])
    selected = sm.select_ships(ships, chosen)

    first, last = picks["months"]
    if (first, last) != (FIRST_MONTH, LAST_MONTH):
        # Only filter when the range is narrowed, so KENKOKU (no month) stays
        # in the default whole-war view and the totals match Table I.
        in_range = selected["month"].apply(lambda m: pd.notna(m) and first <= m <= last)
        selected = selected[in_range]
    if not picks["include_probable"]:
        # Drop only "Probably sunk". KENKOKU has no assessment printed; it is
        # not "probably sunk", so it stays.
        selected = selected[selected["assessment"] != "Probably sunk"]
    return selected, chosen


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def headline_numbers(selected):
    """Show the selection's totals as four metric tiles.

    Tonnage is given per fleet and never summed: a standard ton of a warship
    and a gross register ton of a merchant ship are different units.
    """
    naval = selected[selected["fleet"] == "naval"]
    merchant = selected[selected["fleet"] == "merchant"]
    cols = st.columns(4)
    cols[0].metric("Ships selected", f"{len(selected):,}")
    cols[1].metric("Warships", f"{len(naval):,}",
                   f"{naval['tonnage'].sum():,.0f} standard tons", delta_color="off",
                   delta_arrow="off")
    cols[2].metric("Merchant ships", f"{len(merchant):,}",
                   f"{merchant['tonnage'].sum():,.0f} GRT", delta_color="off",
                   delta_arrow="off")
    cols[3].metric("Not on the map", f"{selected['lat'].isna().sum():,}",
                   "no lat/long printed", delta_color="off", delta_arrow="off")


def map_section(selected):
    """Draw every selected ship that has a position as a dot on a globe.

    plotly is used instead of the matplotlib map of sinking_map.py because it
    lets you zoom into crowded areas (the Philippines, the Japanese home
    waters) and hover a dot to read the ship's name, type, date and agent.
    Colour shows the fleet, as on the static maps. Dot size and opacity
    shrink as the number of ships grows, using sinking_map's DOT_STYLES, so
    3,000 overlapping dots show density instead of one solid blob.
    """
    on_map = selected.dropna(subset=["lat", "lon_original"])
    if on_map.empty:
        st.info("None of the selected ships has a printed position.")
        return
    size, opacity = next((s, a) for limit, s, a in sm.DOT_STYLES if len(on_map) <= limit)
    # DOT_STYLES sizes are matplotlib areas (points^2); plotly wants a
    # diameter in pixels, so take the square root (and enlarge a little,
    # because the web map is shown larger than the PNG).
    diameter = 1.4 * size ** 0.5

    fig = go.Figure()
    for fleet in ["naval", "merchant"]:
        part = on_map[on_map["fleet"] == fleet]
        if part.empty:
            continue
        hover = ("<b>" + part["vessel_name"] + "</b><br>"
                 + part["vessel_type"].fillna("type not printed") + ", "
                 + part["tonnage"].map("{:,.0f}".format) + " " + part["tonnage_unit"]
                 + "<br>" + part["date_text"].astype(str)
                 + "<br>" + part["flag_of_agent"].fillna("Unknown") + ": "
                 + part["agent_type"].fillna("Unknown")
                 + "<br>" + part["location"].fillna(""))
        fig.add_trace(go.Scattergeo(
            lat=part["lat"], lon=part["lon_original"], mode="markers",
            name=f"{FLEET_LABELS[fleet]} ({len(part):,})",
            marker={"size": diameter, "color": FLEET_COLOURS[fleet], "opacity": opacity,
                    # A thin white ring keeps overlapping dots apart.
                    "line": {"width": 0.5, "color": "white"}},
            text=hover, hoverinfo="text"))
    fig.update_geos(
        projection_type="natural earth", projection_rotation_lon=MAP_CENTRE["lon"],
        center=MAP_CENTRE, projection_scale=MAP_SCALE,
        showland=True, landcolor=sm.LAND_COLOUR, showocean=True, oceancolor="white",
        showcountries=False, coastlinecolor="#b8b6ae", showframe=False,
        lataxis_showgrid=True, lonaxis_showgrid=True,
        lataxis_gridcolor=GRID_COLOUR, lonaxis_gridcolor=GRID_COLOUR)
    fig.update_layout(height=620, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                      legend={"orientation": "h", "y": 1.02, "x": 0},
                      font={"color": INK})
    st.plotly_chart(fig, width="stretch")
    st.caption("Drag to pan, scroll to zoom, hover a dot for the ship. "
               f"{len(selected) - len(on_map):,} selected ships have no printed "
               "position and are not drawn.")


def monthly_table(rows, value_column, group_column):
    """Sum value_column per month and group, with every war month present.

    Months with no loss become 0 instead of disappearing, so a quiet month
    shows as a gap rather than the bars closing up and hiding it.
    """
    table = (rows.dropna(subset=["month"])
             .groupby(["month", group_column])[value_column].sum()
             .unstack(fill_value=0)
             .reindex(ALL_MONTHS, fill_value=0))
    table.index = table.index.to_timestamp()
    return table


def stacked_bars(table, colours, labels, y_title, height=330):
    """Draw a monthly stacked bar chart from a month x group table.

    Bars, not lines: the losses are totals per month, and stacked bars show
    both the month's total and each group's part of it. Groups that are all
    zero are left out, so the legend lists only what is drawn.
    """
    fig = go.Figure()
    for key in colours:
        if key in table and table[key].sum() > 0:
            fig.add_bar(x=table.index, y=table[key], name=labels[key],
                        marker_color=colours[key],
                        hovertemplate="%{x|%b %Y}: %{y:,.1f}<extra>" + labels[key]
                                      + "</extra>")
    fig.update_layout(barmode="stack", height=height, bargap=0.15,
                      margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      yaxis_title=y_title, legend={"orientation": "h", "y": 1.15},
                      font={"color": INK}, plot_bgcolor="white")
    fig.update_yaxes(gridcolor=GRID_COLOUR)
    return fig


def time_section(selected, chosen):
    """Monthly losses: number of ships, then tonnage for each fleet.

    The tonnage charts are separate for warships and merchant ships and are
    never stacked together, because their tons are different units. "Split
    by weapon" uses the 1/n credit shares (see explode_credits).
    """
    split = st.radio("Split bars by", ["Fleet", "Weapon"], horizontal=True)
    if split == "Fleet":
        rows = selected.assign(one=1.0, share_tonnage=selected["tonnage"])
        group, colours, labels, ship_value = "fleet", FLEET_COLOURS, FLEET_LABELS, "one"
    else:
        rows = explode_credits(selected, chosen)
        group, colours, labels, ship_value = "weapon", WEAPON_COLOURS, WEAPON_LABELS, "share"

    st.markdown("**Ships sunk per month**")
    st.plotly_chart(stacked_bars(monthly_table(rows, ship_value, group), colours, labels,
                                 "ships"), width="stretch")
    for fleet in ["naval", "merchant"]:
        part = rows[rows["fleet"] == fleet]
        if part.empty:
            continue
        st.markdown(f"**{FLEET_LABELS[fleet]}: tonnage sunk per month "
                    f"({TONNAGE_UNITS[fleet]})**")
        st.plotly_chart(stacked_bars(monthly_table(part, "share_tonnage", group), colours,
                                     labels, TONNAGE_UNITS[fleet], height=280),
                        width="stretch")
    if split == "Weapon":
        st.caption("A ship sunk by several weapons is split equally between them. "
                   "Credits to nations or weapons outside your filters are left out.")


def bar_breakdown(values, labels, colour, title, number_format):
    """One horizontal bar chart of a breakdown, largest bar at the top.

    Horizontal bars because the category names (e.g. "Minesweeper /
    minelayer") are long and stay readable on the y axis. One colour per
    chart: the categories are already named on the axis, so a colour per bar
    would add a legend's work without adding information.
    """
    values = values[values > 0].sort_values()
    fig = go.Figure(go.Bar(
        x=values.values, y=[labels.get(k, k) for k in values.index], orientation="h",
        marker_color=colour, text=[number_format.format(v) for v in values.values],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,.1f}<extra></extra>"))
    fig.update_layout(title={"text": title, "font": {"size": 14}},
                      height=80 + 28 * len(values),
                      margin={"l": 10, "r": 70, "t": 40, "b": 10},
                      font={"color": INK}, plot_bgcolor="white", showlegend=False)
    # Leave 30% of empty room right of the longest bar so its value label sits
    # inside the white plot area. Without it the label falls in the margin,
    # where Streamlit's dark theme paints the background dark and the dark
    # label text vanishes (the longest bar's number showed as "28..." or not at all).
    # 20% was tried first and still clipped "616.0" in the narrow nation chart.
    if len(values):
        fig.update_xaxes(range=[0, values.max() * 1.3])
    fig.update_xaxes(gridcolor=GRID_COLOUR)
    return fig


def breakdown_section(selected, chosen):
    """Ships or tonnage broken down by weapon, nation and ship type, per fleet.

    Counts by weapon and nation use the 1/n credit shares; ship types need no
    shares, because each ship has exactly one type. Tonnage is shown per
    fleet, in its own unit.
    """
    credits = explode_credits(selected, chosen)
    measure = st.radio("Measure", ["Ships", "Tonnage"], horizontal=True)
    for fleet in ["naval", "merchant"]:
        ships_f = selected[selected["fleet"] == fleet]
        if ships_f.empty:
            continue
        credits_f = credits[credits["fleet"] == fleet]
        st.subheader(FLEET_LABELS[fleet])
        if measure == "Ships":
            value, unit, fmt = "share", "ships", "{:,.1f}"
            by_type, type_fmt = ships_f["ship_type"].value_counts(), "{:,.0f}"
        else:
            value, unit, fmt = "share_tonnage", TONNAGE_UNITS[fleet], "{:,.0f}"
            by_type, type_fmt = ships_f.groupby("ship_type")["tonnage"].sum(), fmt
        colour = FLEET_COLOURS[fleet]
        left, middle, right = st.columns(3)
        left.plotly_chart(bar_breakdown(credits_f.groupby("weapon")[value].sum(),
                                        WEAPON_LABELS, colour, f"By weapon ({unit})", fmt),
                          width="stretch")
        middle.plotly_chart(bar_breakdown(credits_f.groupby("agent")[value].sum(),
                                          AGENT_LABELS, colour, f"By nation ({unit})", fmt),
                            width="stretch")
        right.plotly_chart(bar_breakdown(by_type, SHIP_TYPE_LABELS, colour,
                                         f"By ship type ({unit})", type_fmt),
                           width="stretch")
    st.caption("Shared sinkings are split equally between their credits, so a "
               "weapon or nation can have a fractional number of ships.")


def table_section(selected):
    """Show the selected ships as a sortable table with a CSV download.

    The download is the table exactly as filtered, so a selection made in
    the app can be taken into a notebook or a spreadsheet.
    """
    table = (selected[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)
             .sort_values(["Date", "Ship"]))
    st.dataframe(table, width="stretch", hide_index=True, height=520,
                 column_config={"Tonnage": st.column_config.NumberColumn(format="%d"),
                                "Latitude": st.column_config.NumberColumn(format="%.2f"),
                                "Longitude": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Download these ships (CSV)", table.to_csv(index=False),
                       file_name="janac_selected_ships.csv", mime="text/csv")


def about_section():
    """Explain the source and its caveats inside the app."""
    st.markdown(
        """
**Source.** Joint Army-Navy Assessment Committee, *Japanese Naval and Merchant
Shipping Losses During World War II by All Causes* (1947). The two ship lists
(pages 15-44 and 45-115) were read by OCR and checked against every monthly
subtotal and against Table I: 686 warships (1,965,646 standard tons) and
2,346 merchant ships (8,618,109 GRT).

**What is counted.** Only ships the committee assessed as *Sunk* or *Probably
sunk* (merchant ships of 500 GRT and over). Damaged ships are not in the report.

**Tonnage units.** Warships are measured in standard displacement tons,
merchant ships in gross register tons. They are different measures, so the
app never adds them together.

**Shared sinkings.** A ship credited to several agents matches a filter if any
of its credits matches. In the charts by weapon and by nation each credit gets
an equal share of the ship.

**Positions.** 38 ships have only a text location (e.g. "off Japan") and are
left off the map. Two submarines sunk in the Atlantic (RO 501, I 52) are on
the map but outside the starting view; zoom out to see them.

**Nation "Japan".** The report credits some losses to Japan itself: ships that
hit Japanese minefields, and accidents (marine casualties).
        """)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def main():
    """Lay out the page: title, filters, headline numbers and the five tabs."""
    st.set_page_config(page_title="JANAC ship losses", page_icon="⚓", layout="wide")
    st.title("Japanese ship losses in WWII")
    st.caption("Every Japanese warship and merchant ship the JANAC report (1947) "
               "lists as sunk. Choose ships with the filters on the left.")

    ships = load_data()
    picks = sidebar_filters()
    try:
        selected, chosen = apply_filters(ships, picks)
    except sm.OptionError as error:
        # The only error the widgets can produce is min > max tonnage; its
        # message is the second line of the OptionError text.
        st.error(str(error).splitlines()[1].strip())
        st.stop()

    headline_numbers(selected)
    if selected.empty:
        st.warning("No ship matches these filters. Try widening one of them.")
        st.stop()

    tabs = st.tabs(["Map", "Over time", "Breakdown", "Ships", "About the data"])
    with tabs[0]:
        map_section(selected)
    with tabs[1]:
        time_section(selected, chosen)
    with tabs[2]:
        breakdown_section(selected, chosen)
    with tabs[3]:
        table_section(selected)
    with tabs[4]:
        about_section()


main()
