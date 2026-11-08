# Project guide: code, statistics and analysis

This is the complete guide to the project. It explains how the data was
made, how the code is organised, which statistical methods are used and
why, and what each analysis found. It is meant to be read from top to
bottom by someone new to the project.

| If you want… | Read |
|---|---|
| The results only, with charts | [`analysis_findings.md`](analysis_findings.md) |
| Full statistical detail for one analysis | [`us_time_series_explained.md`](us_time_series_explained.md), [`analysis_A1_B_C_explained.md`](analysis_A1_B_C_explained.md), [`routes_tankers_explained.md`](routes_tankers_explained.md) |
| Commands and options | [`README.md`](README.md) |
| How it all fits together | this file |

**Contents**

1. [The question and the source](#1-the-question-and-the-source)
2. [Architecture: how the pieces fit](#2-architecture-how-the-pieces-fit)
3. [Stage 1: from scanned PDF to clean CSV](#3-stage-1-from-scanned-pdf-to-clean-csv)
4. [The datasets](#4-the-datasets)
5. [Shared building blocks](#5-shared-building-blocks)
6. [The scripts, one by one](#6-the-scripts-one-by-one)
7. [The statistical toolkit](#7-the-statistical-toolkit)
8. [Design principles used throughout](#8-design-principles-used-throughout)
9. [Results at a glance](#9-results-at-a-glance)
10. [Reproducing everything](#10-reproducing-everything)
11. [Limitations](#11-limitations)
12. [Glossary](#12-glossary)

---

## 1. The question and the source

**Question:** how, where, when and by what was the Imperial Japanese Navy and
merchant fleet destroyed in the Second World War, and what do statistical
tests say about the patterns we see?

**Source:** the JANAC report, *Japanese Naval and Merchant Shipping Losses
During World War II by All Causes* (Joint Army-Navy Assessment Committee,
1947), in `JANAC.pdf`. It is a 115-page scanned typescript:

| PDF pages | Content | Used for |
|---|---|---|
| 8–12 | Summary Tables I–V (losses by agent and weapon) | the summary charts and cross-checks |
| 15–44 | Japanese naval vessels sunk, in date order | `data/naval/` (686 ships) |
| 45–115 | Japanese merchant vessels sunk, in date order | `data/merchant/` (2,346 ships) |

Each ship row gives the date, name, type, tonnage, position, the nation that
sank it ("flag of agent"), the weapon ("type of agent"), and whether the
credit is firm or only probable.

**The units rule.** Warship tonnage is *standard displacement* (the ship's
weight in long tons). Merchant tonnage is *gross register tons* (GRT), a
measure of internal volume (1 GRT = 100 cubic feet). The two are not
comparable, so no code in this project ever adds them together. Every chart
gives each fleet its own axis or panel.

---

## 2. Architecture: how the pieces fit

```
JANAC.pdf
   │
   │  Stage 1: extraction (OCR + parsing + validation)
   ▼
extract_janac.py ──► ship_lists/naval_pages.py    ──► data/naval/naval_losses_15-44.csv
   │                 ship_lists/merchant_pages.py ──► data/merchant/merchant_losses_45-115.csv
   │                 summary_tables/page_08..12   ──► data/summary/table_I..V.csv
   │
   │  Stage 2: shared loading and filtering
   ▼
sinking_map.py   (load_ships, credits, weapon/ship-type mapping, filters, base map)
table_plot_helpers.py   (palette, chart style, summary-table helpers)
   │
   │  Stage 3: descriptive charts and maps
   ├─► table_1..5_plots.py        Tables I–V as bars, pies, donuts
   ├─► sinking_map.py             static maps (any filter)
   ├─► sinking_animation.py       animated maps (MP4/GIF)
   ├─► loss_line_graph.py         monthly line graphs
   ├─► janac_explorer_app.py      interactive Streamlit app
   │
   │  Stage 4: statistical analyses
   ├─► us_loss_time_series.py     trend, change points, seasonality (US)
   ├─► distance_from_tokyo.py     A1: how far from Tokyo, over time
   ├─► weapon_ship_analysis.py    B1/B2: which weapons sank which ships
   ├─► sinking_clusters.py        HDBSCAN clusters + Gi* hot spots
   ├─► sinking_events.py          C1/C2: events and concentration
   ├─► shipping_routes.py         R1–R5: the submarine war by route
   ├─► tanker_war.py              T1–T6: the tanker war
   └─► submarine_warfare_map.py   kernel-density hunting grounds + routes
```

The scripts reuse each other's functions instead of copying them:

| Script | Imports from |
|---|---|
| every analysis | `sinking_map` (ships, filters, map), `table_plot_helpers` (colours, style) |
| `sinking_animation` | — (gives `sinking_dates` to the others) |
| `sinking_events` | `sinking_clusters` (`PLACES`, `place_name`), `us_loss_time_series` |
| `weapon_ship_analysis` | `us_loss_time_series` |
| `shipping_routes` | `weapon_ship_analysis` (`chi_square`, `cramers_v`, `adjusted_residuals`) |
| `tanker_war` | `shipping_routes` (areas, route groups), `distance_from_tokyo` (`change_points`) |
| `submarine_warfare_map` | `shipping_routes`, `sinking_clusters`, `sinking_animation` |

**Folder convention:**
- every script writes its charts to `figures/<topic>/`;
- every script writes its tables to `data/<topic>/`;
- nothing ever writes to `JANAC.pdf`.

---

## 3. Stage 1: from scanned PDF to clean CSV

### 3.1 Why OCR

The PDF has a built-in text layer, but its digits are badly garbled. That
is fatal for tonnages and coordinates. `extract_janac.py` therefore re-reads
every page image at 300 dpi with **RapidOCR** (`--source ocr`, the default).
The OCR results are cached as JSON in `data/ocr_cache/`, so later runs are
fast. Other approaches were tried first, including Wikipedia ship lists and
an LLM-based reader. RapidOCR won because it reads the typescript reliably
and keeps word positions.

### 3.2 Rebuilding the table from word positions

The scan is a typewritten table with no ruled lines, so the parser has to
reconstruct rows and columns from where the words are.

1. **Phrases → words.** RapidOCR returns phrases. Typescript is monospaced, so a phrase is split into words by character index (`split_phrase`).
2. **Words → lines.** Each word is chained to its nearest left neighbour on the same typed line (`group_lines`). This copes with pages that are warped or skewed (`estimate_slope`).
3. **Words → columns.** Column boundaries come from the header row (`chrono_layout`). The left edge of the vessel-name column comes from where the ALL-CAPS names line up (`name_column_start`).
4. **Lines → rows.** Rows are assembled from the lines (`ChronoParser`), following the report's layout rules:
   - the month and year appear only on the first row of a month, so they are carried forward;
   - continuation lines always belong to the ship above;
   - a `-` in the day column means the report doesn't give the day.
5. **Cells → values.** Each cell is parsed into a value:
   - `parse_tonnage`: `8,000e` → (8000, estimated);
   - `parse_location`: `34-26N, 138-14E` → (34.433, 138.233);
   - the probable marker `*` → `agent_probable`.

### 3.3 Known OCR misreads, and how they are handled

| Misread | Handling |
|---|---|
| `E` read as `B` in longitudes | corrected in `parse_coord` |
| `*` read as `#` or junk characters | treated as the probable mark |
| `e` (estimate) read as `g` or `@` | treated as the estimate mark |
| `Vessels` read as `Veasela` or `4Vessel8` | subtotal lines matched approximately (`is_subtotal_label`) |
| spaces dropped (`December10`, `R0100`) | split by pattern |
| upside-down names (MIZUHO → OHOZIN) | avoided by switching off the text-direction classifier (`use_cls=False`) |

### 3.4 Validation: the report checks itself

The report prints a **subtotal after every month** (number of ships and
total tonnage). The parser compares each month's parsed rows with that
subtotal:

- **Repair:** if a month fails only because the last row's tonnage is garbled (it is underlined in the scan), that tonnage is recalculated from the subtotal.
- **Manual fixes:** cells OCR cannot read are typed in from the page image. They live in each list script's `MANUAL_FIXES` and are flagged `<column>_manual`.
- **Source errors:** subtotals that are wrong in the printed report itself are recorded in `SOURCE_ERRORS` rather than "corrected". For example, the August 1945 merchant subtotal says 48 ships, but the page lists 49, and Table I agrees with 49.

The summary tables (pages 8–12) are harder to read, because their numbers
are stacked and underlined. Each `summary_tables/page_*.py` script:

1. OCRs its page;
2. compares every cell with a hand-checked transcription in `tests/fixtures/table_<ROMAN>.csv`;
3. uses the transcription wherever the two disagree, recording each such cell in `corrected_cells`;
4. re-checks every implied sum (naval + merchant = total, children = parent, rows = TOTAL) with `sum_checks`.

**Final result:**

| Check | Naval | Merchant |
|---|---|---|
| Ships and tonnage vs Table I | 686 / 1,965,646 t, exact | 2,346 / 8,618,109 GRT, exact |
| Monthly subtotals passing | 44 of 44 | 44 of 45 (the 45th is the report's own error) |
| Probable marks vs Table V | 17 = 17 | 76 vs 79 |
| Rows still flagged | 2 unreadable positions | 3 rows (KENKOKU, NITTSU MARU, EIKA MARU) |

`test_extraction.py` gives each page a PASS, WARN or FAIL verdict against
the fixtures in `tests/fixtures/`.

---

## 4. The datasets

### 4.1 Ship lists: `data/naval/naval_losses_15-44.csv`, `data/merchant/merchant_losses_45-115.csv`

| Column | Meaning |
|---|---|
| `page`, `block` | PDF page, and the monthly subtotal the row is counted in |
| `date`, `year`, `month`, `day` | sinking date (`day` empty if the report gives none) |
| `vessel_name`, `vessel_type` | as printed (e.g. `Destroyer`, `Cargo`, `Tanker`) |
| `tonnage`, `tonnage_estimated` | standard tons (naval) or GRT (merchant); `True` if printed with `e` |
| `location`, `lat`, `lon` | printed text; decimal degrees (lon 0–360 in the loaded data) |
| `flag_of_agent`, `agent_type` | who and what; several agents are separated by `;` |
| `agent_probable` | the `*` mark |
| `assessment` | the report's assessment column |
| `issues` | any problem the parser found |
| `raw_*` | the raw OCR text of each cell, kept for auditing |

### 4.2 Summary tables: `data/summary/table_I.csv` … `table_V.csv`

One row per category in the report, with a `level` column for the row's
depth in the table. Charts use only `level == 0` rows, excluding `TOTAL`;
otherwise sub-rows would be counted twice.

### 4.3 Derived data

Each analysis saves its tables under `data/<topic>/`. Examples:

- `data/time_series/us_trend_segments.csv`
- `data/routes/route_tests.csv`
- `data/tankers/tanker_tests.csv`

Every number quoted in the documentation comes from one of these files.

---

## 5. Shared building blocks

### 5.1 `sinking_map.py`: loading, credits and filters

Nearly every analysis starts with `sinking_map.load_ships()`. It reads both
ship lists into one table and adds three columns:

- **`vessel`**: `naval` or `merchant`.
- **`ship_type`**: one of about 20 options (battleship, carrier, cruiser, destroyer, submarine, escort, tanker, cargo, …), mapped from the printed type by `SHIP_TYPE_OF`.
- **`credits`**: a list of `(agent, weapon)` pairs, built by `credits_of`.

**Why pairs?** Take a ship sunk by US aircraft and a British submarine. If
agents and weapons were stored as two separate lists, a filter for "British
aircraft" would wrongly match it. Storing pairs prevents this.

**Weapon mapping** (`weapon_of`) sorts the report's many agent types into
broad weapons, using keyword rules in a fixed order:

- "Marine Casualty" is checked first, because it means an accident, not the Marines.
- "Aircraft" is checked before "Unknown", so "Unknown Aircraft" counts as aircraft.
- "Mine" covers Army, Navy, Allied and Japanese mines.

An agent type that matches no rule stops the program. That way a new
spelling can never silently drop ships.

**Filters** are shared by the maps, animations, clusters, distance analysis
and app. `check_filters` validates them and `select_ships` applies them:

| Filter | Options |
|---|---|
| `--agent` | us, britain, netherlands, australia, new_zealand, china, russia, india, japan, unknown |
| `--weapon` | aircraft, submarine, surface, mine, shore_battery, sabotage, accident, … |
| `--vessel` | naval, merchant |
| `--ship-type` | battleship, carrier, cruiser, destroyer, … tanker, cargo, … |
| `--min-tonnage`, `--max-tonnage` | inclusive limits, compared in each ship's own unit |

A ship matches if its vessel kind, ship type and tonnage all match, **and**
at least one of its credits pairs a chosen agent with a chosen weapon.
Aliases are accepted (`usa` → `us`, `uk` → `britain`), and invalid values
raise `OptionError` with the list of valid values.

**The 1/n rule for shared credits.** When a ship has n credits, each credit
gets 1/n of the ship and 1/n of its tonnage. The analyses that split by
weapon or agent use this rule, so:

- weapon series add up exactly to the total;
- no ship is counted twice;
- monthly counts can be fractions, such as 12.5.

The maps work differently: they show a shared ship on the map of every
agent that took part.

**Map helpers:**

- `load_land` reads Natural Earth 50m land and shifts it to the 0–360° window, so the Pacific isn't split at the date line.
- `style_map_axes` and `map_aspect_ratio` set up the map window.
- `DOT_STYLES` makes dots smaller and more transparent as more ships are drawn.

### 5.2 `table_plot_helpers.py`: style and palette

- **`set_readable_style()`** sets large fonts, a light background and a recessive grid.
- **Palette:** `BLUE` (naval), `ORANGE` (merchant), `AQUA`, `YELLOW`, `PINK`, `GREEN`, `VIOLET`, `GREY`, plus ink and grid colours. The colours were chosen to stay distinguishable for colour-blind readers.
- **Fixed meanings:**
  - naval is always blue and merchant always orange;
  - for weapons, submarine is violet, aircraft aqua, surface warship pink and mine yellow.
- **`tint(colour, amount)`** mixes a colour with white; it is used for confidence bands and faded bars.
- **Summary-table helpers:**
  - `read_table`, `group_rows`, `detail_rows`;
  - `check_totals`, which stops the script if categories don't add up to the report's TOTAL;
  - `plot_bars`, `plot_pies`, `plot_detail_donuts`.

### 5.3 Dates and time

- **`sinking_animation.sinking_dates`** gives each ship a date. A ship whose day is missing is placed mid-month.
- **Time variable:** analyses measure time as *months since 7 December 1941*.
- **Partial months:** December 1941 has 24 war days and August 1945 has 15. Time-series models handle this as **exposure**: losses per average-length month (`EXPOSURE` in `us_loss_time_series.py`).

---

## 6. The scripts, one by one

Each entry gives the question, what goes in and out, the method, and the
key functions. The command to run each script is in Section 10.

### 6.1 Summary-table charts: `table_1_plots.py` … `table_5_plots.py`

- **Question:** what do the report's own summary tables say?
- **In:** `data/summary/table_*.csv`. **Out:** `figures/table_<ROMAN>/` (bars, pies and two-ring donuts, for ships and for tonnage).
- **Method:** each figure has three panels: naval, merchant, and the report's own combined total. Every panel has its own axis, because the units differ. Before drawing, each script checks that its categories add up to the TOTAL row.
- **Why horizontal bars:** the category labels are long, and bars let you compare lengths directly. Small slices are named only in the pie legend, so the labels don't collide.

### 6.2 Static maps: `sinking_map.py`

- **Question:** where were the ships sunk, by any combination of filters?
- **Out:** `figures/maps/sinkings_agent-…_weapon-…_vessel-….png`. `--all-combinations` draws every non-empty agent × weapon × vessel map.
- **Notes:** 2,992 of 3,032 ships have a position. 38 have only a text location (e.g. "off Japan"), and 2 lie outside the map window. The footnote states these counts on every map.

### 6.3 Animated maps: `sinking_animation.py`

- **Question:** how did the losses unfold over time?
- **Out:** `figures/animations/*.mp4` (default, 60 s at 20 fps) or `*.gif` (`--format gif`, 10 fps).
- **Method:**
  - each ship appears on its sinking date as a big dot and shrinks to a small one over `FADE_DAYS` (40 days);
  - a monthly bar strip under the map shows where you are in the war, with future months faded;
  - a counter shows the cumulative ships and tonnage for each fleet.
- **Video encoding:** ffmpeg comes from `imageio-ffmpeg`.

### 6.4 Monthly line graphs: `loss_line_graph.py`

- **Question:** how many ships and how much tonnage were lost each month?
- **Out:** `figures/line_graphs/losses_<measure>_….png`.
- **Method:** one panel per fleet; the peak month is marked; line names sit at the line ends (`place_end_labels`) instead of in a legend.

### 6.5 Interactive explorer: `janac_explorer_app.py`

- **What it is:** a Streamlit app with sidebar filters for fleet, ship type, weapon, nation, tonnage range, month range and "include probable".
- **Tabs:**
  - Map: a plotly globe;
  - Over time: stacked monthly bars;
  - Breakdown: by weapon, nation and ship type, using 1/n shares;
  - Ships: a table with CSV download;
  - About.
- **Code notes:**
  - `load_data` is cached with `@st.cache_data`;
  - `explode_credits` turns each ship into one row per credit, with weight 1/n;
  - `apply_filters` reuses `sinking_map.check_filters` and `select_ships`.
- **Check:** the default view reproduces Table I's totals.

### 6.6 US time series: `us_loss_time_series.py`

- **Question:** what was the trend of US sinkings, when did the loss rate change, how fast did it grow in each phase, and is there a season?
- **Why the US only:** the US has about 90% of the losses. Every other nation's monthly counts are mostly zero, too sparse for a time series.
- **In:** the ship lists, US credits only, weighted 1/n. The totals are checked against Table II (`check_against_report`).
- **Out:** in `data/time_series/`: `us_monthly_losses.csv`, `us_trend_segments.csv`, `us_series_tests.csv`, `us_weapon_comparison.csv`. In `figures/time_series/`: one figure per fleet and measure, plus the weapon-share chart.
- **Series:**
  - for each fleet: "all", plus every weapon with at least 100 losses (`MIN_SERIES_SHIPS`);
  - for each series, two measures: ships and tonnage.
- **Methods** (details in §7):
  1. STL trend with 13-month smoothing (a description, not a test);
  2. Mann-Kendall trend test with trend-free pre-whitening, plus Sen's slope;
  3. change points with `ruptures` exact dynamic programming, 0–3 breaks chosen by BIC, each phase at least 6 months;
  4. a segmented regression with a level and slope per phase: negative binomial with exposure for ship counts, OLS on log(1 + tonnage) for tonnage;
  5. seasonality: one sine/cosine pair added to the model and tested by likelihood ratio;
  6. a weapon comparison: one pooled model with a natural spline (4 df) for time, with LR tests for a shared trend shape and a shared season.
- **Key functions:**
  - `stabilise`: square root for counts and log(1 + x) for tonnage, which makes the noise even for change-point detection;
  - `change_points`, `design_matrix`, `fit_model`, `season_shape`, `compare_weapons`.
- **Full detail:** [`us_time_series_explained.md`](us_time_series_explained.md).

### 6.7 Distance from Tokyo (A1): `distance_from_tokyo.py`

- **Question:** did sinkings move closer to Japan over time ("the noose tightens")?
- **Out:** `data/distance/` (monthly distances, phases, tests) and `figures/distance/`.
- **Method:**
  - haversine distance to Tokyo for every positioned ship;
  - monthly medians, computed only for months with at least 5 ships;
  - Mann-Kendall test and change points on the medians;
  - a **median for each phase**, with a **cluster bootstrap** (2,000 resamples of whole months) for the confidence interval;
  - quantile regression at the 25th, 50th and 75th percentiles;
  - Spearman's ρ between date and distance;
  - a logistic regression of P(within 1,000 km of Tokyo) on time.
- **Reused elsewhere:** `change_points(values)` is used by `tanker_war.py`.

### 6.8 Weapon × ship type (B1, B2): `weapon_ship_analysis.py`

- **Question:** did particular weapons sink particular kinds of ship, and bigger or smaller ones?
- **Sample:** ships sunk by a single weapon (submarine, aircraft, surface or mine). That is 634 naval and 2,172 merchant ships. Every exclusion is recorded with its reason in `excluded_ships.csv`.
- **B1 method:**
  - a chi-square test of independence, with a permutation p-value (9,999 shuffles), because some expected counts are below 5;
  - bias-corrected Cramér's V (Bergsma, 2013);
  - adjusted residuals (Haberman, 1973), with a Bonferroni threshold for significant cells;
  - correspondence analysis (an SVD of the standardised residuals);
  - a log-linear model testing type × weapon *within* each year, to rule out timing as the explanation.
- **B2 method:**
  - a Kruskal-Wallis test of tonnage across weapons, followed by Dunn's pairwise tests with Cliff's δ as the effect size;
  - OLS of log tonnage on weapon, both raw and adjusted for ship type and year.
- **Reused elsewhere:** `chi_square`, `cramers_v`, `adjusted_residuals` and `RESIDUAL_CMAP` are used by the route and tanker analyses.

### 6.9 Clusters and hot spots: `sinking_clusters.py`

- **Question:** where are the significant concentrations of losses?
- **Two methods, used together:**
  1. **HDBSCAN** on (lat, lon) with haversine distance. It finds clusters of any shape and labels sparse points as noise (`--min-cluster-size`, default 15; the saved figures use 10). Clusters are numbered by size. `medoid` gives each cluster a representative position, and `place_name` names it after the nearest entry in `PLACES` (within 400 km).
  2. **Getis-Ord Gi\*** on ship counts in 100 km hexagons, built on an equal-area projection (Lambert azimuthal, centred at 15°N 150°E):
     - p-values come from 9,999 conditional permutations;
     - Benjamini-Hochberg FDR control is applied at 0.05.

     A cluster counts as *confirmed* if at least half its ships lie in significant hot-spot hexagons (`CONFIRM_SHARE` = 0.5).
- **Why both:** HDBSCAN finds the shapes but has no significance test, while Gi\* tests significance but only on a fixed grid. Agreement between the two is the evidence.

### 6.10 Events and concentration (C1, C2): `sinking_events.py`

- **C1, events:** two ships are linked if they sank within 1 day **and** 100 km of each other. Events are the connected components of the resulting graph (single linkage, `scipy.sparse.csgraph`). `sensitivity` re-runs the grouping at (0 d, 50 km), (1, 50), (2, 200) and (3, 300).
- **C2, concentration:**
  - Gini coefficient of losses per war day, and the share of losses on the worst 1%, 5% and 10% of days;
  - **bunching permutation test:** within each month, every ship gets a random day 999 times; the observed Gini is compared with these null Ginis;
  - **event-size models:** a geometric distribution against a discrete power law (zeta), compared with Vuong's test.

### 6.11 Shipping routes (R1–R5): `shipping_routes.py`

- **Question:** how did the submarine campaign move between Japan's shipping routes?
- **Population:** merchant ships only.
- **R1, placing ships:**
  - The theatre is divided into 12 sea areas (`AREAS`), as polygons tested first-match in order. The Shimonoseki Strait, for example, is tested before the Sea of Japan.
  - Ships with only a text location are placed by keywords (`TEXT_AREAS`).
  - Result: 2,315 placed by position, 26 by text, 5 unplaced.
  - Areas are grouped into four route groups (`GROUP_OF`): **southern** resource route, **inner** zone (China and Korea), **home** coastal waters, **outer** perimeter.
- **R2:** chi-square on area × year for the 1,153 ships sunk by submarines alone, with permutation p, V and adjusted residuals.
- **R3:** a multinomial logit of route group on a natural spline of time (4 df). An LR test compares it with "no change", and the confidence bands come from a month-cluster bootstrap (300 refits).
- **R4:** when each area's losses happened. A Kruskal-Wallis test across areas, plus a paired month-cluster bootstrap of the difference in median date between routes.
- **R5:** a logistic regression of "sunk by submarine" on time, with one slope per route group and standard errors clustered by month. A Wald test checks whether the slopes are equal.
- **Full detail:** [`routes_tankers_explained.md`](routes_tankers_explained.md).

### 6.12 Tanker war (T1–T6): `tanker_war.py`

- **Population:** 262 tankers (1.31M GRT), against all merchant losses.
- **T1:** change points in log(1 + monthly tanker GRT).
- **T2:** a segmented **beta-binomial** model of the tanker share of merchant ships sunk:
  - exhaustive break search (`best_breaks`), with BIC choosing the number of breaks;
  - a sup-LR test with a parametric bootstrap (299 simulations);
  - an LR test for overdispersion.
- **T3:** odds ratios by period for "tanker vs other merchant ship" × "submarine kill vs other". The periods are pooled with a Mantel-Haenszel estimate, and checked for homogeneity with Breslow-Day.
- **T4:** chi-square of route group × tanker.
- **T5:** median (quantile 0.5) regression of log GRT on time, and Spearman's ρ.
- **T6:** a logistic regression of aircraft vs submarine kills on time. The crossover date, where P = 0.5, has a bootstrap confidence interval.

### 6.13 Submarine warfare map: `submarine_warfare_map.py`

- **Question:** where were the submarines' hunting grounds, and how did they relate to the convoy routes?
- **Population:** every ship with a submarine credit, weighted by its submarine share: 1,382 ships. `--agent us` limits it to US submarines.
- **Method:**
  - **Kernel density:** a weighted Gaussian KDE on the sphere (`sklearn.neighbors.KernelDensity`, haversine metric), with a 150 km bandwidth on a 0.35° grid.
  - **Highest-density regions** (`hdr_thresholds`, Hyndman 1996): the smallest areas holding 25%, 50% and 75% of the kills, drawn as nested shades.
  - **Hot spots:** the 8 highest density peaks at least 650 km apart, each named, with the kills within 250 km counted.
  - **Routes** (`PORTS`, `ROUTES`): drawn by hand between historical convoy ports and labelled as *approximate*, because JANAC records where ships sank, not their tracks.
- **Out:** the whole-war map, a 2×2 map by year, and `data/submarine_war/hot_spots_agent-<tag>.csv`.

---

## 7. The statistical toolkit

The data has some awkward features, and they shaped every choice of method:

1. **Overdispersed counts.** Ships are sunk in bunches: one convoy attack can sink six. Counts therefore vary far more than a Poisson model allows.
2. **Autocorrelation.** A month tends to look like the month before it.
3. **Clustering by month.** Ships sunk in the same month share conditions, so they are not independent observations.
4. **Small cells.** Rare categories, such as battleships, leave expected counts below 5.
5. **Skewed tonnage.** One battleship weighs as much as twenty frigates.
6. **A short series.** 45 months gives each calendar month only 3–4 appearances.

Here is how each method answers one of these problems.

### 7.1 Describing trends

**STL decomposition** (Seasonal-Trend decomposition by Loess) splits a
series into trend + season + remainder. Only the trend is used, with a
13-month window, to draw a smooth line. The seasonal part is not used: with
under four years of data, STL cannot separate a season from a campaign.

### 7.2 Is there a monotonic trend? Mann-Kendall

- $H_0$: no monotonic trend. $H_1$: a monotonic increase or decrease.
- Statistic: $S = \sum_{i<j} \operatorname{sgn}(x_j - x_i)$. It is approximately normal under $H_0$.
- **Trend-free pre-whitening** (Yue et al., 2002): remove the Sen slope, remove the lag-1 autocorrelation, then add the slope back. Without this step, autocorrelation inflates the false-positive rate.
- **Sen's slope:** the median of all pairwise slopes $(x_j - x_i)/(j - i)$, a robust "typical change per month".
- **Used in:** `us_loss_time_series`, `distance_from_tokyo`.

### 7.3 When did the level change? Change points with BIC

- `ruptures` with exact dynamic programming (`Dynp`) finds the k break positions that minimise the total within-segment squared error. Each segment must be at least 6 months long.
- The number of breaks, k = 0…3, is chosen by **BIC** = n·ln(RSS/n) + p·ln(n). A break is kept only if it improves the fit by more than its complexity penalty.
- Series are first **variance-stabilised**: √ for counts, log(1 + x) for tonnage. Change-point detection with squared error assumes the noise is equally large everywhere.
- **Used in:** the US time series, the distance analysis, and tanker T1.

### 7.4 How fast in each phase? Segmented regression

- **Ship counts: negative binomial GLM** with a log link and log(exposure) as an offset:
  $$\log E[y_t] = \log(\text{exposure}_t) + \beta_{0,k} + \beta_{1,k}\, t \quad \text{for } t \text{ in phase } k$$
  NB2 variance $\mu + \alpha\mu^2$ allows for overdispersion. The likelihood also works with fractional counts, which the 1/n credits produce.
- **Tonnage: OLS on log(1 + rate).** Tonnage is not a count, and on the log scale it is close to normal.
- **Effect:** $100(e^{\beta_{1,k}} - 1)$ = % change per month, with a 95% CI.
- **Diagnostics:** Ljung-Box test on the residuals (lag 6).

### 7.5 Is there a season? Harmonic regression and LR test

- Add $\gamma_1 \sin(2\pi m/12) + \gamma_2 \cos(2\pi m/12)$ to the segmented model.
- $H_0: \gamma_1 = \gamma_2 = 0$. The test statistic is LR $= 2(\ell_1 - \ell_0) \sim \chi^2_2$.
- **Multiple testing:** Benjamini-Hochberg q-values are computed across all the series, for both the season tests and the Mann-Kendall tests (`mk_q`, `season_q` in `us_series_tests.csv`).
- **Result:** no series has a significant season. With 3–4 cycles of data, the test has little power.

### 7.6 Do weapons share a trend? Pooled spline model

- All of one fleet's weapon series are fitted in a single model, with a natural cubic spline of time (4 df).
- **Nested LR tests:**
  - one trend shape, shifted up or down per weapon, against a separate shape per weapon;
  - then a shared season against a separate season per weapon.

### 7.7 Association between two categorical variables

| Tool | Formula / idea | Why |
|---|---|---|
| Chi-square test of independence | $X^2 = \sum (O - E)^2 / E$ | the standard test that type and weapon are independent |
| **Permutation p-value** | shuffle one label 9,999 times, $p = (1 + \#\{X^2_\text{null} \ge X^2_\text{obs}\})/(1 + B)$ | stays valid when expected counts are below 5 |
| **Cramér's V, bias-corrected** | $\tilde V = \sqrt{\tilde\varphi^2 / \min(\tilde r - 1, \tilde c - 1)}$ (Bergsma 2013) | an effect size from 0 to 1 that does not grow with n |
| **Adjusted residuals** | $(O - E)/\sqrt{E(1 - r_i/n)(1 - c_j/n)}$ (Haberman 1973) | ≈ N(0,1) for each cell; shows *which* cells drive the result |
| Bonferroni cell threshold | $z_{1 - \alpha/(2\cdot\text{cells})}$, e.g. 3.20 | controls false alarms across many cells |
| **Correspondence analysis** | SVD of $D_r^{-1/2}(P - rc^\top)D_c^{-1/2}$ | a 2-D map placing types near the weapons that sank them |
| Log-linear conditional test | type × weapon given year: compare [TY][WY] against [TWY] | removes the confounding by timing |

### 7.8 Comparing sizes: rank tests

- **Kruskal-Wallis:** $H_0$ is that every weapon's tonnage comes from the same distribution. The effect size is ε² = H/(n − 1).
- **Dunn's test:** pairwise comparisons after Kruskal-Wallis, with Holm correction.
- **Cliff's δ:** P(a > b) − P(a < b), an effect size that needs no distributional assumption.
- **Why rank tests:** tonnage is heavily skewed. For effects adjusted for other variables, OLS on log tonnage is used instead.

### 7.9 Modelling shares and proportions

- **Logistic regression** for a binary outcome over time, e.g. "is this loss a submarine kill?" (R5) or "is it within 1,000 km of Tokyo?" (A1). The coefficient gives the odds ratio per year.
- **Multinomial logit** (R3): the shares of 4 route groups over time, with time entering as a natural spline. LR test against an intercept-only model.
- **Beta-binomial** (T2): the monthly tanker count $y_t$ out of $n_t$ merchant losses.
  - $y_t \sim \text{BetaBin}(n_t, p_k, \rho)$, with variance $n p(1 - p)[1 + (n - 1)\rho]$.
  - ρ is the intra-month correlation: ships sunk together in one convoy attack are not independent draws.
  - An LR test of ρ = 0 checks whether a plain binomial would do.
- **Sup-LR test with parametric bootstrap** (T2): the break month is chosen by maximising the likelihood, so the ordinary χ² distribution is wrong for the LR statistic (the Davies problem). Instead, data is simulated under no break (299 simulations), and the whole search is re-run on each simulated set.
- **Mantel-Haenszel and Breslow-Day** (T3): a common odds ratio across the period strata, with a test that the odds ratios are the same in every period.

### 7.10 Handling dependence: cluster-robust inference

Ships sunk in the same month are not independent. The project handles this
in two ways:

- **Cluster-robust (sandwich) standard errors** with month as the cluster (R5, T6).
- **Month-cluster bootstrap:** whole months are resampled with replacement, never single ships (A1 phase medians, R3 bands, R4, T2, T6). The bootstrap is **paired** when two quantities are compared on the same resample (R4).

This is why R4's Mann-Whitney p = 2e-34, which treats ships as
independent, is reported as misleading. The cluster bootstrap of the same
difference has a CI that includes 0.

### 7.11 Spatial statistics

- **Haversine distance:** $d = 2R \arcsin\sqrt{\sin^2(\Delta\phi/2) + \cos\phi_1\cos\phi_2\sin^2(\Delta\lambda/2)}$ with R = 6,371 km.
- **HDBSCAN:** density-based clustering that finds clusters of varying density and labels sparse points as noise. It needs no k.
- **Getis-Ord Gi\*:** $G_i^* = \dfrac{\sum_j w_{ij}x_j - \bar x \sum_j w_{ij}}{s\sqrt{[n\sum_j w_{ij}^2 - (\sum_j w_{ij})^2]/(n - 1)}}$ for hexagon counts. The weights cover each hexagon and its 6 neighbours, and p-values come from conditional permutation.
- **Benjamini-Hochberg FDR:** sort the p-values and reject all $p_{(i)} \le (i/m)\,\alpha$. Thousands of hexagons are tested at once, so FDR control is needed.
- **Kernel density with highest-density regions:** $\hat f(x) = \sum_i w_i K_h(d(x, x_i)) / \sum_i w_i$. An HDR at level q is $\{x : \hat f(x) \ge f_q\}$, where $f_q$ is chosen so the region holds a share q of the weight.

### 7.12 Concentration and heavy tails

- **Gini coefficient** of daily losses: 0 means every day is the same; values near 1 mean all the losses fell on one day.
- **Bunching permutation test:** the null hypothesis keeps every monthly total, so the long-term trend is preserved exactly, but spreads each ship uniformly over the war days of its month.
- **Power law (zeta) vs geometric:** both fitted to event sizes by maximum likelihood. **Vuong's test** compares non-nested models: a positive z favours the power law.

### 7.13 Change in a median: quantile regression

Quantile regression minimises $\sum \rho_\tau(y_i - x_i^\top\beta)$, where
$\rho_\tau$ is the "check" loss. τ = 0.5 gives the median, which is robust
to the few very large ships. It is used for distance at τ = 0.25, 0.5 and
0.75 (A1), and for tanker size at τ = 0.5 (T5).

---

## 8. Design principles used throughout

| Principle | Where it shows |
|---|---|
| **Never add standard tons and GRT** | separate panels, separate series, a footnote on every chart |
| **Validate before analysing** | monthly subtotals, Table I totals, `check_totals`, `check_against_report`; scripts stop on mismatches |
| **Shared credits = 1/n** | weapon series add up to the total; nothing is double-counted |
| **Permutation and bootstrap p-values, not only asymptotic** | small cells, clustered data, the Davies problem |
| **Cluster by month** | all inference about time-varying shares |
| **Report effect sizes, not just p** | Cramér's V, ε², Cliff's δ, odds ratios, % per month |
| **Correct for multiple tests** | Bonferroni for residual cells, BH-FDR for hexagons and seasons |
| **Fixed random seed** (`RANDOM_SEED = 12345`) | every permutation and bootstrap is reproducible |
| **Fail loudly** | unknown agent spellings, bad filters and failed sum checks all stop the program |
| **Keep the raw evidence** | every CSV row keeps its raw OCR text and an `issues` column |
| **Colour follows meaning** | naval is always blue, merchant always orange, and each weapon keeps its colour |
| **Heavy commenting** | every script has a module docstring (inputs, outputs, run command), a docstring on every function explaining *why*, and comments on every constant and data caveat |

---

## 9. Results at a glance

| Analysis | Main result | Key statistic |
|---|---|---|
| Table I | US sank ≈92% of both fleets' tonnage | 92.7% naval, 91.8% merchant |
| Table II | Submarines → merchant fleet; aircraft → warships | 60.4% of US merchant GRT; 39.8% of US naval tons |
| US time series | merchant losses stepped up in Nov 1943 | phase means 46k → 114k → 274k GRT/month |
| US time series | submarine tonnage fell from Dec 1944 | −12.5%/month (CI −21.6 to −2.3), p = 0.019 |
| US time series | no seasonality | e.g. merchant GRT p = 0.26 |
| A1 distance | the noose tightened in Mar 1945 | median 3,035 → 911 km; home share 10% → 57% |
| B1 naval | strong type × weapon link | V = 0.32; submarines sunk by surface warships, residual +12.7 |
| B1 merchant | weak type × weapon link | V = 0.11 |
| B2 | submarines sank bigger merchant ships | median 3,967 vs mines 1,500 GRT; KW p ≈ 7e-37 |
| Clusters | 72 clusters; chokepoints confirmed by Gi\* | Tsugaru Strait 138 ships, 86% in hot spots |
| C2 | losses bunched on single days | worst 5% of days = 76% of naval tons; bunching p = 0.001 |
| C1 | heavy-tailed event sizes | zeta exponent 2.72, Vuong p = 3e-9 |
| R2/R3 | submarine hunting grounds moved | X² = 424, V = 0.34; LR = 337, p ≈ 9e-65 |
| R5 | submarine share fell, fastest in home waters | OR/yr home 0.17, southern 0.48, outer 0.98 |
| T2 | tanker share jumped in Sep 1943 | 2.5% → 12.9%; sup-LR p = 0.003 |
| T3 | submarines did not prefer tankers | MH OR = 1.08 (0.81–1.43), p = 0.62 |
| T4 | tankers concentrated on the southern route | residual +11.4; V = 0.23 |
| T5 | tankers sunk got smaller | median −60%/year |
| T6 | aircraft overtook submarines vs tankers | crossover ≈ 5 Oct 1944 |
| Submarine map | densest grounds | Luzon Strait 75, Nagoya 73, Tsugaru 68 kills |

The story these results tell, with the charts, is in
[`analysis_findings.md`](analysis_findings.md).

---

## 10. Reproducing everything

All commands run from the project folder in PowerShell, with the packages
from `requirements.txt` installed. The cleaned CSVs are already in `data/`, so
Stage 1 is only needed to rebuild them from the scan. It needs the report saved
as `JANAC.pdf` in the project folder (it is not in the repository; see the
README for where to find it).

```powershell
# Stage 1 - extraction (slow on first run: ~10 s/page of OCR, then cached)
python ship_lists\naval_pages.py --pages 15-44
python ship_lists\merchant_pages.py
foreach ($s in Get-ChildItem summary_tables\page_*.py) { python $s.FullName }
python test_extraction.py --pages 8-18                 # optional page checks

# Stage 3 - descriptive charts
1..5 | ForEach-Object { python "table_$($_)_plots.py" }
python sinking_map.py --all-combinations
python sinking_animation.py                             # all ships, MP4
python sinking_animation.py --ship-type battleship carrier --format gif
python loss_line_graph.py

# Stage 4 - analyses
python us_loss_time_series.py
python distance_from_tokyo.py
python weapon_ship_analysis.py
python sinking_clusters.py --min-cluster-size 10
python sinking_events.py
python shipping_routes.py                               # ~30 s
python tanker_war.py                                    # ~60 s
python submarine_warfare_map.py
python submarine_warfare_map.py --agent us

# Interactive app
python -m streamlit run janac_explorer_app.py
```

Any script with filters accepts `--options` to list the valid values.
Results are deterministic, because every random step uses
`RANDOM_SEED = 12345`.

**Environment:**

- Python 3.12, with the exact package versions in `requirements.txt`;
- pandas stays on 2.x (the project was built with 2.2.3);
- the main libraries are pymupdf, rapidocr_onnxruntime, geopandas, shapely, pyproj, matplotlib, plotly, streamlit, scipy, statsmodels, scikit-learn, ruptures, pymannkendall, scikit-posthocs, esda, libpysal and imageio-ffmpeg.

---

## 11. Limitations

1. **No exposure data.** JANAC lists losses, not ships at sea. Every "rate" here is a rate of losses per month, not a risk per ship. Falling losses in 1945 reflect a fleet that was nearly gone.
2. **JANAC is conservative and incomplete for small craft.** It lists only confirmed sinkings, and almost no merchant ships under 500 GRT (one in this data).
3. **OCR positions:**
   - a few positions may be misread;
   - 38 ships have only a text location;
   - 2 naval positions are unreadable.
4. **Hand-drawn elements are judgement calls:**
   - the sea-area boundaries (R) and convoy routes (submarine map) are approximate;
   - the route results have not yet been tested for sensitivity to the boundaries.
5. **The 1/n rule is a convention.** Other ways of splitting shared credits would shift the weapon shares slightly.
6. **Short series.** 45 months limits the seasonality tests and the number of change points that can be detected.
7. **Many tests share the same data.** The headline effects have very small p-values and large effect sizes. Treat the borderline ones with care: seasonality, the 1942–43 tanker odds ratio, and the R4 timing difference.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **JANAC** | Joint Army-Navy Assessment Committee; its 1947 report is the source |
| **Standard displacement** | a warship's weight without fuel and reserve water, in long tons |
| **GRT** | gross register tons: a merchant ship's enclosed volume, 1 GRT = 100 ft³ |
| **Agent** | the nation (flag) and weapon (type) credited with a sinking |
| **Probable** | a credit marked `*`: the committee was not certain who sank the ship |
| **1/n credit** | a ship with n credits gives 1/n to each |
| **Exposure** | month length relative to an average month; corrects for partial months |
| **BIC** | Bayesian information criterion; penalises extra parameters by ln(n) each |
| **Overdispersion** | more variance than a Poisson or binomial model allows |
| **Adjusted residual** | a cell's (observed − expected) divided by its standard error; ≈ N(0,1) |
| **Cramér's V** | the strength of association in a contingency table, from 0 to 1 |
| **Cluster bootstrap** | resampling whole groups (here months) to respect dependence |
| **FDR** | false discovery rate: the expected share of false positives among the discoveries |
| **Gi\*** | Getis-Ord local statistic: is this place and its neighbours unusually high? |
| **HDBSCAN** | hierarchical density-based clustering with noise |
| **KDE / HDR** | kernel density estimate / highest-density region |
| **Sup-LR** | the largest likelihood ratio over all candidate break points |
| **Vuong's test** | a likelihood-ratio test for non-nested models |
| **Route groups** | southern resource route, inner zone (China and Korea), home waters, outer perimeter |
