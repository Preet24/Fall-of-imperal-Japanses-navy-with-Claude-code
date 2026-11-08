# The submarine campaign by route, and the tanker war: methods, tests and findings

This document explains two analyses of the JANAC merchant ship list:

| Script | Question | Outputs |
|---|---|---|
| `shipping_routes.py` (R1–R5) | Where along Japan's shipping lanes were merchant ships sunk, by which weapon, and how did that move over the war? | `figures/routes/`, `data/routes/` |
| `tanker_war.py` (T1–T6) | When, where and by what were Japan's oil tankers sunk? | `figures/tankers/`, `data/tankers/` |

```powershell
python shipping_routes.py     # ~30 s
python tanker_war.py          # ~60 s (parametric bootstrap)
```

Every number below is taken from the CSVs the scripts write (`route_tests.csv`, `tanker_tests.csv` and the other tables). Random seeds are fixed (12345), so reruns give the same results.

---

## 1. Summary in plain language

**Routes.**
- **The submarines' hunting grounds moved three times.**
  - **1942:** off Japan's own coast and in the South Pacific.
  - **1943:** the Central Pacific supply lines to Truk and the Marianas.
  - **1944:** the southern oil and resource route, above all the Luzon Strait ("Convoy College") and the Philippines.
  - **1945:** the inner zone, meaning the Yellow Sea and the Sea of Japan, the last sea lanes still open to Japan.

  The change is large and certain: bias-corrected Cramér's V = 0.34, permutation p = 0.0001.
- **Near Japan, submarines handed over to mines and aircraft.**
  - **Japanese coastal waters:** the odds that a loss was a submarine kill fell by a factor of about 6 per year (OR 0.17).
  - **Southern route and inner zone:** they fell by about half per year.
  - **Outer perimeter:** they did not change. The four rates differ significantly (Wald p = 5 × 10⁻⁵).
- **One test came out weaker than expected.** Once ships sunk in the same month are not treated as independent, the evidence that the inner and home routes were hit *later on average* than the southern route is weak. The shift shows clearly in *which* areas the submarines worked, not in a simple shift of average dates.

**Tankers.**
- **Losses jumped in September 1943.** Before that, Japan lost about 1,000–8,000 GRT of tankers a month. From September 1943 to the end of the war it lost about 51,000 GRT a month.
- **Tankers became a larger share of losses, in one step.** They were 2.5% of merchant ships sunk until August 1943 and 12.9% from September 1943 (bootstrap p = 0.003).
- **Submarines did *not* single out tankers more than other weapons did, once period is held fixed.** The pooled odds ratio is 1.08 (0.81–1.43). Only in 1942–43 are tankers over-represented among submarine kills (OR 2.98, 1.02–8.71).
- **Tankers died on the southern route.** It holds 72% of tanker losses against 39% of all merchant ships; adjusted residual +11.4.
- **Aircraft overtook submarines as the main tanker killer in about October 1944** (95% CI June 1944 – February 1945).
- **The tankers sunk got much smaller:** the median size fell about 60% per year. By 1945, 46% of the tankers sunk were under 1,000 GRT. These were the small wartime coastal tankers, sunk because the big ocean tankers were already gone.

---

## 2. Data and definitions

### 2.1 Population

- **Merchant ships only.** The question is about shipping lanes, and the merchant list (2,346 ships, 8.62 million GRT) is what used those lanes. Warships are left out.
- **One ship is dropped.** KENKOKU has no position and its date was inferred, so it is excluded from date-based work: it is counted in the totals but not in the area analysis.
- **Units.** All tonnage is gross register tons (GRT), so nothing here mixes it with naval standard tons.

### 2.2 Sea areas and route groups

Each ship's printed latitude/longitude is tested against 12 polygons (`AREAS` in `shipping_routes.py`):
- **First match wins.** The polygons are tested in a fixed order, so later boxes only catch what earlier ones did not.
- **The map shows the areas as they actually apply.** `effective_areas()` subtracts each earlier polygon, so the drawn areas don't overlap (`figures/routes/route_areas_map.png`).
- **Ships without coordinates.** 26 ships with only a text location ("East China Sea", "off Formosa") are placed by keyword. NITTSU MARU's impossible longitude "124-93E" is fixed to the Yellow Sea.
- **Unplaced ships.** 5 ships cannot be placed: 4 are "off Japan" and the fifth is KENKOKU.

| Route group | Areas | Ships | GRT |
|---|---|---|---|
| Southern resource route | NEI & Malacca Strait, South China Sea, Philippines, Luzon Strait, Formosa & Ryukyus | 921 | 3.79 M |
| Inner zone: China & Korea | East China & Yellow Sea (incl. lower Yangtze), Sea of Japan & Korea Strait | 444 | 1.35 M |
| Japanese coastal waters | Japan's Pacific coast & Inland Sea (incl. Shimonoseki Strait) | 384 | 1.05 M |
| Outer perimeter | New Guinea & Solomons, Central Pacific islands, North (Hokkaido/Kuriles/Aleutians), Indian Ocean & Burma | 592 | 2.42 M |

The boundaries are judgment calls along the narrows between legs of a route, such as the Luzon Strait, the Korea Strait and Tanegashima. A ship within roughly 20 km of a boundary could reasonably be on either side. With about 2,300 ships this moves a few dozen at most, which is too few to change any of the large effects below. The Japan/Sea of Japan boundary follows Honshu's coast, so a ship sunk in a port on that coast may fall on either side.

### 2.3 Weapon variables

- **Shares (charts).** Each credit gets 1/n of the ship, the same rule as `us_loss_time_series.py`. So 57% of merchant GRT was sunk by submarines overall: 61% on the southern route and in the inner zone, 48% in coastal waters and 54% on the outer perimeter.
- **`submarine_only` (tests).** This is 1 if *every* credit is a submarine. It covers 1,153 ships. Only 9 more ships have a partial submarine credit, so the choice hardly matters. US submarines account for 96.5% of submarine-only kills, so "submarine" here effectively means the US submarine campaign.
- **`aircraft_only`:** the same rule for aircraft (T6).

### 2.4 Time

- **`t`** is months since 7 December 1941 (average month = 30.44 days).
- **Missing days.** Ships without a printed day are placed on the 15th (`sinking_animation.sinking_dates`).
- **Months as clusters.** Ships sunk in the same month are often from the same convoy or raid, so they are **not independent**. Every standard error and bootstrap in both scripts is clustered by month for this reason.

---

## 3. Routes: methods, hypotheses and results

### R1. Losses by area and year (descriptive)

`data/routes/area_year_losses.csv` gives, for each area and year: ships, GRT, submarine ships and GRT (1/n shares), and the submarine share of GRT. `figures/routes/submarine_area_heatmap.png` shows submarine GRT by area and quarter. It uses a single-hue scale on √GRT, so quiet areas stay visible next to the peaks, and the exact thousands of GRT are written in the cells.

**Observation.** The darkest cells move from the Central Pacific (1943 Q4 – 1944 Q2: 212k, 185k, 173k GRT) to the Philippines and the Luzon Strait (1944 Q3–Q4: 165k, 118k; 119k, 144k) and the South China Sea (1944 Q4: 195k). The 1945 cells are mostly in the East China/Yellow Sea and the Sea of Japan.

### R2. Did the submarines' hunting grounds shift? Area × year contingency table

**Data.** The 1,153 submarine-only kills are cross-tabulated by area (rows) and year (columns; December 1941 is merged into 1942). The Indian Ocean (9 kills) has too few to keep as its own row. Its expected counts would be near zero, which makes the chi-square unreliable, so it is pooled into "other areas". The result is a 12 × 4 table.

**Hypotheses.**
- H₀: the area mix of submarine kills is the same every year (area ⟂ year).
- H₁: the mix differs between years.

**Test statistic.** Pearson's X² = Σ (O − E)² / E, with E = row total × column total / n.

**Null distribution.**
- **Asymptotic:** χ² with (r − 1)(c − 1) = 33 df.
- **Permutation (reported):** 4 of 48 cells have expected counts below 5, so the asymptotic approximation is not trusted alone. The year labels are shuffled 9,999 times, which keeps both margins fixed, and p = (1 + #{X²_perm ≥ X²_obs}) / 10,000.

**Effect size.** Cramér's V with the Bergsma (2013) bias correction.

**Cell diagnostics.** Haberman adjusted residuals are approximately N(0, 1) under H₀. A cell counts as notable at |r| > 3.28: the Bonferroni threshold z₁₋₀.₀₅/(2·48).

**Result.** X² = 423.7, df = 33, permutation p = 0.0001 (asymptotic p ≈ 2 × 10⁻⁶⁹). Plain V = 0.35; corrected V = **0.34**, a moderate-to-strong association for a table this size.

| Notable cells | Residual | Reading |
|---|---|---|
| Japan coast, 1942 | +5.3 | early-war patrols off Japan's own ports |
| New Guinea & Solomons, 1942 / 1943 | +3.4 / +4.4 | supply runs to Rabaul and the Solomons |
| Central Pacific, 1943 | **+7.6** | the Truk / Marianas supply lines |
| Luzon Strait, 1944 | **+7.7** | "Convoy College" |
| Philippines, 1944 | +6.3 | convoys reinforcing the Philippines before Leyte |
| Sea of Japan & Korea Strait, 1945 | **+11.7** | the last open route, entered from June 1945 |
| East China & Yellow Sea, 1945 | +3.6 | the China–Korea traffic |
| Central Pacific, 1945 | −6.6 | nothing left to sink there |
| Japan coast, 1944 | −5.7 | submarines had moved south |

Figure: `figures/routes/submarine_area_residuals.png`. Data: `data/routes/submarine_area_year.csv`.

**Caveat.** The permutation test treats ships as exchangeable. Kills in the same convoy attack are not independent, so the p-value is optimistic. With X² more than ten times its df, however, no plausible clustering correction would make the result non-significant.

### R3. A smooth model of the shift: multinomial logit on a spline of time

**Model.** For the four route groups k (reference: southern),

log[ P(group = k | t) / P(group = southern | t) ] = α_k + f_k(t),

where f_k is a natural cubic regression spline with 4 df (patsy `cr(t, df=4, constraints='center')`, the same flexibility as the pooled weapon model in `us_loss_time_series.py`). It is fitted by maximum likelihood (statsmodels `MNLogit`, BFGS).

**Hypotheses.**
- H₀: f_k ≡ 0 for all k (the group shares are constant over time).
- H₁: at least one share changes.

**Test.** Likelihood ratio LR = 2(ℓ_full − ℓ_null) ~ χ² with (K − 1) × 4 = 12 df.

**Uncertainty bands.** These come from a month-cluster bootstrap: the 45 months are resampled with replacement, the model is refitted 300 times, and the 2.5th–97.5th percentiles of the predicted shares are taken. All 300 refits converged.

**Result.** LR = 336.8, df = 12, p ≈ 9 × 10⁻⁶⁵. McFadden pseudo-R² = 0.11: time explains a real but modest part of *where* a given submarine kill happened, as expected when patrols were spread over many areas at once.

Modelled shares of submarine kills (95% bootstrap band):

| Month | Southern | Inner | Coastal | Outer |
|---|---|---|---|---|
| Jun 1942 | 39% (32–48) | 15% (3–22) | 25% (15–33) | 21% (16–30) |
| Jun 1943 | 20% (16–26) | 16% (8–27) | 18% (12–24) | **46% (34–55)** |
| Sep 1944 | **69% (61–76)** | 12% (6–18) | 5% (3–9) | 14% (8–22) |
| Jun 1945 | 18% (12–30) | **63% (42–72)** | 16% (9–31) | 3% (0–6) |

Figure: `figures/routes/submarine_route_shares.png`. It uses one panel per group rather than a stacked area, so each band stays readable. Dots show the observed quarterly shares, with size proportional to the number of kills.

**Why both R2 and R3?** R2 makes no assumption about the *shape* of change but chops time into years. R3 assumes smoothness but gives a continuous curve with uncertainty. They agree, which is the point of running both.

### R4. When was each area's traffic hit?

**Descriptive** (`data/routes/area_timing.csv`, `figures/routes/route_timing.png`). For each area the chart shows the 10th, 25th, 50th, 75th and 90th percentile loss dates (all causes). The median comes with a 95% month-cluster bootstrap interval (2,000 resamples). Areas are ordered by median date:

| Area | Median loss date | 95% CI |
|---|---|---|
| New Guinea & Solomons | Jun 1943 | Jan 1943 – Jan 1944 |
| Indian Ocean & Burma | Jan 1944 | Aug 1943 – Jun 1944 |
| Central Pacific | Feb 1944 | Nov 1943 – Apr 1944 |
| NEI & Malacca Strait | May 1944 | Dec 1943 – Aug 1944 |
| North | May 1944 | Jun 1943 – Sep 1944 |
| Luzon Strait | Sep 1944 | Jun – Oct 1944 |
| East China & Yellow Sea | Sep 1944 | Jan 1944 – Mar 1945 |
| Philippines | Sep 1944 | Jun – Nov 1944 |
| Formosa & Ryukyus | Oct 1944 | Feb 1944 – Jan 1945 |
| South China Sea | Oct 1944 | Mar 1944 – Jan 1945 |
| Japan: Pacific coast & Inland Sea | May 1945 | Oct 1943 – Jul 1945 |
| Sea of Japan & Korea Strait | Jun 1945 | Sep 1944 – Jul 1945 |

The order follows the story: the outer perimeter first, then the southern route, then Japan's own waters.

**Test 1: Kruskal–Wallis across areas.**
- H₀: the loss-date distribution is the same in all 12 areas.
- Statistic: H = 634.5, df = 11, p ≈ 6 × 10⁻¹²⁹.
- Effect size: ε² = H/(n − 1) = **0.27**, so area accounts for about a quarter of the rank variance in loss date.
- Caveat: this p-value treats ships as independent and is optimistic.

**Test 2: southern route vs inner zone + coastal waters.**
- **Estimates:** difference in median date 6.7 months, and in mean date 3.4 months (inner/home later).
- **Paired month-cluster bootstrap.** Months are resampled *once* for both groups, because they share the same calendar. The 95% intervals are **−6.8 to 11.6 months (median)** and **−3.5 to 7.6 months (mean)**. Both include zero.
- **Mann–Whitney U.** It gives p ≈ 2 × 10⁻³⁴, but it assumes independent ships.

**Interpretation, and why the two disagree.**
- **The coastal group has two peaks.** It holds a 1942–43 hump of submarine kills off Japan's ports *and* the big 1945 mine and air campaign. When whole months are resampled, the group's centre of mass moves a lot.
- **Each month weighs heavily.** Only 45 months carry all the information about timing, and a single heavy month (e.g. January 1945, July 1945) weighs a great deal. The Mann–Whitney p-value is not trustworthy here.
- **The honest conclusion:** there is strong evidence that the *mix* of areas changed (R2, R3), but only weak evidence for a simple "southern first, inner later on average" shift once dependence within months is respected. The clearest late area is the Sea of Japan alone: its median is June 1945, and its interval (September 1944 – July 1945) does not overlap the Central Pacific's or New Guinea's.

### R5. The hand-over from submarines to other weapons

**Model.** Logistic regression of `submarine_only` with one intercept and one slope per route group:

logit P(sub | g, t) = α_g + β_g t, with exp(12 β_g) = odds ratio per year in group g.

Standard errors are cluster-robust by month (sandwich estimator).

**Hypotheses.**
- Per group: H₀: β_g = 0.
- Joint: H₀: β_southern = β_inner = β_home = β_outer (the decline is the same everywhere), tested with a cluster-robust Wald χ² on 3 df.

**Results** (`data/routes/route_tests.csv`, `figures/routes/weapon_handover.png`):

| Route group | OR per year | 95% CI | p |
|---|---|---|---|
| Southern resource route | 0.48 | 0.28–0.80 | 0.006 |
| Inner zone | 0.44 | 0.29–0.69 | 0.0003 |
| Japanese coastal waters | **0.17** | 0.08–0.33 | 3 × 10⁻⁷ |
| Outer perimeter | 0.98 | 0.65–1.48 | 0.93 |

Wald test of equal slopes: χ² = 22.6, df = 3, p = 5 × 10⁻⁵.

**Reading.**
- **Coastal waters:** the hand-over was fastest here. Submarines dominated until mid-1944; in the second quarter of 1945 mines took about 80% of the GRT (Operation Starvation's aerial mining) and aircraft took most of the rest.
- **Southern route:** carrier aircraft took a large share from late 1944, including the 12 January 1945 carrier raid in the South China Sea.
- **Outer perimeter:** submarines and aircraft shared the work throughout, so there is no trend.

**Caveat.** A logistic model with a linear slope in t is a simplification: the coastal pattern is closer to a step in early 1945. The slope summarises the direction and strength of the change, not its exact shape. The stacked bars show the shape.

---

## 4. Tankers: methods, hypotheses and results

The population is the 262 merchant ships of report type "Tanker" (1.31 million GRT). Tankers are a ship *type*, so this is a sub-population of the routes data, and it uses the same area, group and weapon variables.

### T1. Monthly losses and change points in tanker GRT

**Method.**
- The series y_t = log(1 + tanker GRT in month t), over 45 months.
- The log stabilises the variance. The +1 keeps the 4 months with no tanker loss defined.
- Breaks come from an exact segmentation (`ruptures.Dynp`, L2 cost, minimum phase 6 months; reused from `distance_from_tokyo.change_points`). The number of breaks k ∈ {0,…,3} is chosen by BIC = T log(RSS/T) + (2k + 1) log T.

**Result.** BIC: k = 0: −476.9; k = 1: −511.2; **k = 2: −516.6**; k = 3: −512.7. Two breaks, in January 1943 and September 1943:

| Phase | Months | Tankers | GRT | Mean GRT / month |
|---|---|---|---|---|
| Dec 1941 – Dec 1942 | 13 | 3 | 16,038 | 1,234 |
| Jan 1943 – Aug 1943 | 8 | 9 | 67,533 | 8,442 |
| **Sep 1943 – Aug 1945** | 24 | **250** | **1,226,886** | **51,120** |

95% of the tanker GRT was lost in the last 24 months. The biggest single month is January 1945 (176,000 GRT, almost all by aircraft). On the 12th alone, 12 tankers (53,000 GRT) were sunk in the South China Sea, the day of the US carrier raid on the convoys off Indochina. Figure: `figures/tankers/tanker_monthly.png`. The three panels (ships, GRT, cumulative GRT) each have their own y-axis instead of a dual axis.

**Caveat.** The change points describe the *loss* series, which mixes Allied effort with how many tankers Japan had at sea. The report has no fleet-size data, so this is not a loss *rate*.

### T2. Did tankers become a larger share of merchant losses?

Counting tankers alone cannot separate "tankers were targeted" from "more of everything was sunk". The share of tankers *among* merchant losses does separate them.

**Model.** In month t, y_t of the n_t merchant ships sunk were tankers (n_t ≥ 9 every month).

y_t ~ BetaBinomial(n_t, p_j(t), ρ),

with a share p_j that is constant within phase j and a common intra-month correlation ρ. With a = p(1 − ρ)/ρ and b = (1 − p)(1 − ρ)/ρ:

E[y_t] = n_t p, Var[y_t] = n_t p(1 − p)[1 + (n_t − 1)ρ].

The beta-binomial is used rather than the binomial because tankers often sailed and died together (tanker convoys), which gives extra-binomial variance. Fitting is by maximum likelihood on the logit scale (L-BFGS-B).

**Break search.**
- Exhaustive over all break dates with phases ≥ 6 months, for k = 0, 1, 2 breaks.
- BIC = −2ℓ + q log T, where q = (k + 1) shares + 1 ρ + k break dates.
- BIC: k = 0: 218.1; **k = 1: 197.2**; k = 2: 197.6. One break, at **September 1943**: the same month as T1's main break, found by a different model on a different quantity.

**Test of "the share never changed": sup-LR.**
- H₀: p is constant over the war. H₁: there is one break at an unknown date.
- Statistic: LR = 2[max over break dates ℓ₁ − ℓ₀] = 28.5.
- **Null distribution:** the break date exists only under H₁ (Davies' problem), so LR is **not** χ². It is calibrated by a **parametric bootstrap**: 299 series are simulated from the fitted no-change beta-binomial, using the observed n_t, and the full break search is repeated on each. p = (1 + #{LR_sim ≥ 28.5}) / 300 = **0.003**. None of the simulations reached the observed value, so 0.003 is the smallest p-value 299 simulations can give.

**Overdispersion.**
- ρ̂ = 0.027.
- LR test of binomial (ρ = 0) against beta-binomial with the same phases: LR = 26.8. ρ = 0 is on the boundary, so p = ½ P(χ²₁ > LR) ≈ 1 × 10⁻⁷.
- The overdispersion is small but clear, and ignoring it would make the intervals too narrow.

**Phase shares** (with month-cluster bootstrap 95% CIs; `data/tankers/tanker_phases.csv`):

| Phase | Tankers / merchant ships sunk | Share (fitted) | 95% CI |
|---|---|---|---|
| Dec 1941 – Aug 1943 | 12 / 468 | **2.5%** | 0.9–4.3% |
| Sep 1943 – Aug 1945 | 250 / 1,878 | **12.9%** | 10.6–16.7% |

The share of merchant GRT (bottom panel of `figures/tankers/tanker_share.png`, descriptive only) is higher, peaking at about 40% in February–March 1945, because tankers were larger than the average merchant ship.

**Interpretation.**
- **What the jump shows.** From September 1943 a sunk merchant ship was about five times as likely to be a tanker.
- **What it could reflect.** Without fleet-composition data it cannot be split into (a) targeting and (b) tankers making up more of the ships at sea. Japan built many tankers in 1943–44, and many converted cargo ships became tankers. T3 addresses (a) indirectly.

### T3. Did submarines single out tankers?

**Design.** In each of four periods there is a 2 × 2 table:

|  | Sunk by submarine only | Any other cause |
|---|---|---|
| Tanker | a | b |
| Other merchant ship | c | d |

OR = ad / bc. OR > 1 means a tanker was more likely than another merchant ship to have been sunk by a submarine rather than by aircraft, mines, etc. The periods are 1942–43 (pooled, because 1942 had only 3 tankers), 1944 H1, 1944 H2 and 1945. Holding period fixed matters: both the tanker share and the weapon mix change over time, so a crude OR would be confounded by date.

**Tests.**
- **Cochran–Mantel–Haenszel:** H₀: the common OR = 1 given period.
- **Mantel–Haenszel pooled OR** with its 95% CI.
- **Breslow–Day:** H₀: the OR is the same in all periods.
- Per-period CIs use the Woolf (log-OR) normal approximation.

**Results** (`data/tankers/tanker_odds_by_period.csv`, `figures/tankers/tanker_weapons.png`, left):

| Period | Tankers (sub / other) | OR | 95% CI | p |
|---|---|---|---|---|
| 1942–43 | 23 / 4 | 2.98 | 1.02–8.71 | 0.037 |
| 1944 H1 | 27 / 18 | 1.03 | 0.55–1.94 | 0.92 |
| 1944 H2 | 45 / 42 | 0.93 | 0.59–1.47 | 0.75 |
| 1945 | 23 / 80 | 0.97 | 0.59–1.60 | 0.90 |
| **MH pooled** | | **1.08** | 0.81–1.43 | CMH p = 0.62 |

Breslow–Day χ² = 4.31, df = 3, p = 0.23.

**Interpretation.**
- **No relative focus overall.** Given period, a tanker was **not** more likely than another merchant ship to fall to a submarine as opposed to other weapons.
- **What this does and doesn't show.** It does *not* mean submarines ignored tankers: they sank 118.5 of them (1/n shares). It means the other weapons, especially carrier aircraft from 1944, hit tankers at least as hard *relative to* other ships.
- **The 1942–43 exception is fragile.** Tankers were over-represented among submarine kills only then, the CI's lower bound is 1.02, and it is one test of four. The Breslow–Day test cannot reject equal ORs across periods.
- **Limitation.** This design measures *relative* selectivity among ships that were sunk. Selectivity among ships *at sea* would need fleet data (see the open offer in the README).

### T4. Where tankers died compared with other merchant ships

**Test.** Chi-square on route group × {tanker, other merchant}, a 4 × 2 table, with a permutation p (9,999 shuffles), corrected V and adjusted residuals.

- H₀: tankers were lost in the same route groups, in the same proportions, as other merchant ships.

**Result.** X² = 130.6, df = 3, permutation p = 0.0001; corrected V = **0.23**.

| Route group | Tankers (observed / expected) | Adjusted residual |
|---|---|---|
| Southern resource route | **188 / 103** | **+11.4** |
| Inner zone | 27 / 50 | −3.8 |
| Coastal waters | 21 / 43 | −3.9 |
| Outer perimeter | 26 / 66 | −6.1 |

72% of the tankers lost (188 of 262; by tonnage 911,000 of 1.31 million GRT, 70%) went down on the southern route, which is exactly the oil route from Borneo, Sumatra and Malaya. Figure: `figures/tankers/tanker_where_size.png`, left.

### T5. Did the tankers sunk get smaller?

**Method.**
- Median (quantile 0.5) regression of log(GRT) on t, giving a percentage change per year of 100 [exp(12 β) − 1].
- **Why the median:** tanker sizes come in two clumps, ocean tankers of 5,000–10,000 GRT and wartime coastal tankers under 1,000 GRT. A mean would sit between the clumps.
- **Why the log:** it turns ratios into differences.
- A Spearman ρ is reported as an assumption-light check.

**Results.**
- Median size change **−60% per year** (95% CI −71% to −46%), p = 1.6 × 10⁻⁸.
- Spearman ρ = −0.41, p = 7 × 10⁻¹².
- The share of tankers sunk that were under 1,000 GRT: 8% in 1943, 21% in 1944, **46% in 1945**.

**Chart choice.** The right panel of `tanker_where_size.png` draws **half-year medians**, not the fitted line. A single straight line on the log scale cuts between the two clumps and, extended back to early 1942 (3 tankers), implies 30,000-GRT ships that never existed. The half-year medians show what actually happened: around 7,000 GRT until mid-1944, then a fall to about 1,000 GRT by 1945.

**Interpretation.** The drop is a *composition* effect. By 1945 the large ocean tankers were already sunk, and the losses were small wartime-built tankers used for coastal and short-sea work (31 of the 103 tankers sunk in 1945 are listed at exactly 834 GRT, which points to one standard wartime design). This is consistent with the route results: the 1945 tanker losses moved closer to Japan.

### T6. When did aircraft overtake submarines as the tanker killer?

**Model.** Among the 229 tankers sunk by aircraft only or submarines only:

logit P(aircraft | t) = β₀ + β₁ t, with cluster-robust SEs by month.

**Crossover.** The date when aircraft and submarines were equally likely, P = 0.5, is t* = −β₀/β₁. Its 95% CI comes from a month-cluster bootstrap with 2,000 refits, all usable. It is used instead of the delta method because a ratio of coefficients can have a skewed sampling distribution.

**Results.**
- Odds of an aircraft kill rising ×**5.2 per year** (95% CI 2.1–13.1, p = 0.0004).
- Crossover **5 October 1944** (95% CI 25 June 1944 – 23 February 1945).

The timing matches the arrival of the US fast carriers in the Philippines–South China Sea area (September 1944 – January 1945) and of land-based air from the Philippines. Figure: `figures/tankers/tanker_weapons.png`, right.

---

## 5. Limitations shared by both analyses

1. **Losses only.** JANAC lists sunk ships, not ships at sea. Every result is about the *composition and timing of losses*, not loss *rates*. Rates would need monthly fleet size by type and route (USSBS data), which is the "fleet-size exposure" step offered in the README.
2. **Clustering.** Months are used as clusters throughout. Convoys can span two months, and a raid can sink ships in two areas. Clustering by month is a practical approximation, not a perfect independence structure.
3. **Area boundaries.** These are hand-drawn and a few dozen ships could move between neighbouring areas. A sensitivity run, moving boundaries by ±1° and re-running R2, is a cheap robustness check that has not been done yet.
4. **Positions.** These are OCR-read and checked month by month against subtotals, but not individually against the scan. A misread degree would put a ship in the wrong area.
5. **Ship type.** "Tanker" is the report's type, so ships converted to tankers during the war may be typed by their original role. The tanker counts here are the report's.
6. **Multiple testing.** Each table reports its own p-values with no global correction. The main conclusions rest on p-values of 10⁻⁴ or smaller and on effect sizes, so a Bonferroni correction over the roughly 20 tests in the two scripts would not change any conclusion. The two borderline results (the T3 1942–43 OR, p = 0.037; and R4's date difference) are described as fragile.

---

## 6. Glossary

| Term | Meaning |
|---|---|
| Adjusted residual | (O − E) / √[E(1 − row share)(1 − col share)], ≈ N(0,1) under independence |
| Beta-binomial | Binomial whose p varies between groups (months) as a Beta distribution; allows overdispersion |
| BIC | −2 log-likelihood + (parameters) × log(n); lower is better; penalises extra phases |
| Breslow–Day | Test that a 2×2 odds ratio is the same in all strata |
| CMH test | Cochran–Mantel–Haenszel: test of association in 2×2 tables, holding the stratum fixed |
| Cluster bootstrap | Resample whole clusters (months) rather than individual ships, to respect dependence |
| Cramér's V | Effect size for a contingency table, from 0 (independent) to 1 |
| Davies' problem | A parameter (the break date) exists only under H₁, so the LR is not χ² |
| ε² (Kruskal–Wallis) | H/(n − 1): share of rank variance explained by the groups |
| McFadden R² | 1 − ℓ_model / ℓ_null for likelihood models |
| Natural spline | Piecewise cubic, linear beyond the boundary knots; a flexible smooth curve |
| Odds ratio | Ratio of odds between two groups; 1 = no difference |
| Parametric bootstrap | Simulate data from the fitted null model to get a test's null distribution |
| Quantile (median) regression | Regression for a chosen percentile instead of the mean; robust to skew |
| Sup-LR test | Likelihood ratio maximised over an unknown break date |

## 7. References

- Agresti, A. (2013). *Categorical Data Analysis*, 3rd ed. Wiley. (CMH, Breslow–Day, adjusted residuals, multinomial logit)
- Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica* 61, 821–856. (sup-LR)
- Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *J. Korean Stat. Soc.* 42, 323–328.
- Cameron, A. C., Gelbach, J. B. & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Rev. Econ. Stat.* 90, 414–427.
- Davies, R. B. (1987). Hypothesis testing when a nuisance parameter is present only under the alternative. *Biometrika* 74, 33–43.
- Haberman, S. J. (1973). The analysis of residuals in cross-classified tables. *Biometrics* 29, 205–220.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Killick, R., Fearnhead, P. & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *JASA* 107, 1590–1598. (background to `ruptures`)
- Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of ML estimators and LR tests under nonstandard conditions. *JASA* 82, 605–610. (½χ²₁ boundary test)
- Tomczak, M. & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends Sport Sci.* 21, 19–25. (ε²)
- Joint Army-Navy Assessment Committee (1947). *Japanese Naval and Merchant Shipping Losses During World War II by All Causes*.
