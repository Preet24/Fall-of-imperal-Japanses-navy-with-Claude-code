# Where, what and how suddenly: five analyses of Japan's ship losses explained

This document explains five analyses that build the middle chapters of the
project's story: the process, the statistics, the hypothesis tests, the
results and their limits. Each method has a plain-language summary followed
by the technical detail. Every number comes from the CSV files the scripts
write. Equations use LaTeX (`$…$`).

| Part | Question | Script | Outputs |
|---|---|---|---|
| **A1** | Did the war close in on Japan? | `distance_from_tokyo.py` | `figures/distance/`, `data/distance/` |
| **B1** | Which weapons sank which types of ship? | `weapon_ship_analysis.py` | `figures/weapon_ship/`, `data/weapon_ship/` |
| **B2** | Did some weapons sink bigger ships? | `weapon_ship_analysis.py` | same |
| **C1** | What were the biggest single sinking events? | `sinking_events.py` | `figures/events/`, `data/events/` |
| **C2** | How concentrated in time were the losses? | `sinking_events.py` | same |

```powershell
python distance_from_tokyo.py      # ~20 s; accepts the usual --agent/--weapon/... filters
python weapon_ship_analysis.py     # ~30 s
python sinking_events.py           # ~40 s
```

---

## Contents

1. [Data used by all five analyses](#1-data-used-by-all-five-analyses)
2. [A1: distance from Tokyo over time](#2-a1-distance-from-tokyo-over-time)
3. [B1: which weapons sank which ship types](#3-b1-which-weapons-sank-which-ship-types)
4. [B2: ship size by weapon](#4-b2-ship-size-by-weapon)
5. [C1: sinking events](#5-c1-sinking-events)
6. [C2: concentration of losses](#6-c2-concentration-of-losses)
7. [How the five fit the story](#7-how-the-five-fit-the-story)
8. [Summary of all hypothesis tests](#8-summary-of-all-hypothesis-tests)
9. [Limitations and next steps](#9-limitations-and-next-steps)
10. [Glossary](#10-glossary)
11. [References](#11-references)

---

## 1. Data used by all five analyses

- **Source:** the JANAC ship lists. There are 686 warships and 2,346 merchant ships, which is 3,032 ships in all.
- **Fleets are kept apart for tonnage.** Warship tonnage is *standard displacement* (weight) and merchant tonnage is *gross register tons* (GRT, volume). They are never added together.
- **Positions:** 2,994 ships have a printed latitude/longitude; 38 have only a text location ("off Japan").
- **Dates:** 3,026 ships have an exact date. Six warships have only a month.
- **Agents:** a ship can be credited to several agents (nation + weapon). Each analysis says how it handles this.

---

## 2. A1: distance from Tokyo over time

### 2.1 Question and hypotheses

**The story to test:** in 1942 Japan lost ships at the far edges of its new
empire; by 1945 it was losing them in its own harbours. If so, the typical
distance of a sinking from Tokyo should fall over the war, and the share of
sinkings in home waters should rise.

| Test | H₀ | H₁ |
|---|---|---|
| Quantile regression (ship level) | The 25th / 50th / 75th percentile of distance does not change with time | It changes linearly with time |
| Spearman (ship level) | No monotonic association between date and distance | Later sinkings tend to be nearer (or farther) |
| Mann-Kendall (monthly medians) | No monotonic trend in the monthly median | A monotonic trend |
| Logistic regression | P(sinking within 1,000 km of Tokyo) is constant over time | It changes with time |

Change points and phase medians are *descriptive*: model selection, not
hypothesis tests.

### 2.2 Process

1. **Distance.** For each of the 2,994 ships with a position, the great-circle
   distance to Tokyo (35.68° N, 139.77° E) is computed with the haversine formula:
   $$d = 2R\arcsin\sqrt{\sin^2\tfrac{\Delta\varphi}{2} + \cos\varphi_1\cos\varphi_2\sin^2\tfrac{\Delta\lambda}{2}}, \quad R = 6{,}371\text{ km}.$$
   Tokyo was chosen because readers can picture it. Distance to the nearest home-island coast is an alternative; it would count Shimonoseki (about 850 km from Tokyo) as 0.
2. **Time.** $t$ = months since 7 Dec 1941, using 30.44-day months. Ships with no day are placed on the 15th.
3. **Monthly summaries.** For all ships, warships and merchant ships: the median and the 25th–75th percentiles. Months with fewer than 5 ships in a series are left out of that series' tests, because a median of 2 ships is noise.
4. **Home waters:** within **1,000 km** of Tokyo. This covers the seas around Honshu, Shikoku, Kyushu and southern Hokkaido, the Inland Sea and the Korea Strait, but not Formosa (2,100 km).

### 2.3 Methods

**Quantile regression** (Koenker & Bassett, 1978). For a chosen quantile τ it
estimates $Q_\tau(d \mid t) = \beta_0 + \beta_1 t$ by minimising the
*check loss*:
$$\sum_i \rho_\tau(d_i - \beta_0 - \beta_1 t_i), \qquad \rho_\tau(u) = u\,(\tau - \mathbb{1}[u<0]).$$

Why it is used here:
- **It targets the median directly.** Distances are right-skewed (a few sinkings near Ceylon or Alaska are 5,000+ km away), and least squares would chase those few.
- **Three quantiles show the whole distribution.** Fitting τ = 0.25, 0.5 and 0.75 shows whether the whole distribution moved or only its middle.

The standard errors come from a kernel estimate of the error density
(Hall–Sheather bandwidth), which is robust to the spread changing over time.
They do treat ships as independent.

**Spearman's ρ:** Pearson's correlation computed on ranks. It is
assumption-light and measures any monotonic relation.

**Mann-Kendall with trend-free pre-whitening and Sen's slope** on the monthly
medians. This is the same test as in the time-series analysis (see
`us_time_series_explained.md`, §5.2). It allows for month-to-month
autocorrelation.

**Change points.** An exact search (`ruptures.Dynp`, L2 cost) on the monthly
median of all ships, with phases of at least 6 months and up to 3 breaks.
The number of breaks $k$ is chosen by
$\text{BIC}_k = n\log(\text{RSS}_k/n) + (2k+1)\log n$. Months with fewer than
5 ships are filled by linear interpolation, so the months stay evenly spaced.

**Phase medians with a cluster bootstrap.** The interval for each phase's
median resamples whole **months** (with replacement) and keeps all their ships,
2,000 times. The percentile interval is then taken. Resampling single ships
would ignore that ships sunk in the same raid share a place, and would give
an interval that is too narrow.

**Logistic regression with month-clustered standard errors.** The model is
$\operatorname{logit} P(\text{home}) = \beta_0 + \beta_1 t$, reported as the
odds ratio per year, $e^{12\beta_1}$. The standard errors use the cluster
sandwich estimator with the month as the cluster. That accounts for the strong
correlation between ships sunk in the same month (often in the same raid);
naive errors would be far too small.

### 2.4 Results

**Trend tests** (`data/distance/distance_tests_….csv`):

| Test | Estimate | 95% CI | p | n |
|---|---:|---|---:|---:|
| Quantile regression, 25th percentile | −49.9 km/month | −55.3 to −44.4 | <10⁻⁶⁰ | 2,994 ships |
| Quantile regression, median | **−85.3 km/month** | −93.3 to −77.3 | <10⁻⁹⁰ | 2,994 ships |
| Quantile regression, 75th percentile | −80.9 km/month | −85.4 to −76.4 | <10⁻²⁰⁰ | 2,994 ships |
| Spearman ρ, date vs distance | −0.38 | | <10⁻¹⁰⁰ | 2,994 ships |
| Mann-Kendall, monthly median, all ships | Sen −51.0 km/month | | 4.5 × 10⁻⁸ | 45 months |
| Mann-Kendall, monthly median, warships | Sen −99.7 km/month | | 2.8 × 10⁻⁷ | 37 months |
| Mann-Kendall, monthly median, merchant ships | Sen −32.5 km/month | | 9.1 × 10⁻⁵ | 45 months |
| Logistic, P(within 1,000 km), odds ratio per year | **2.32** | 1.14 to 4.73 | 0.020 | 2,994 ships, 45 clusters |

**Change points:** BIC chose $k$ = 3 (BIC 4.7, −22.0, −24.0, −27.8 for
$k$ = 0–3). That is very strong evidence against "no change" (ΔBIC = 32.5)
and positive evidence for 3 breaks over 2 (ΔBIC = 3.8).

| Phase | Months | Ships | Median distance (95% cluster-bootstrap CI) | Within 1,000 km |
|---|---|---:|---|---:|
| 1 | Dec 1941 – Oct 1942 | 241 | 3,483 km (3,278 – 3,625) | 20% |
| 2 | Nov 1942 – Apr 1943 | 204 | **4,370 km** (3,992 – 4,747) | 12% |
| 3 | May 1943 – Feb 1945 | 1,846 | 3,035 km (2,941 – 3,179) | 10% |
| 4 | **Mar 1945** – Aug 1945 | 703 | **911 km** (824 – 1,446) | **57%** |

### 2.5 Observations

1. **The noose did tighten, and the whole distribution moved.** All the trend
   tests reject H₀ by enormous margins. The 75th percentile fell about as fast
   as the median (−81 vs −85 km/month), so it was not only the typical sinking
   that came closer; the far-away ones came closer too.
2. **But it was not a steady tightening.** The change points show four phases:
   - **The distance first *increased*** (phase 2, Nov 1942 – Apr 1943, median 4,370 km), when the fighting was around Guadalcanal and New Guinea at the far southern edge of the empire.
   - **It then sat on a long plateau of about 3,000 km** for nearly two years (May 1943 – Feb 1945), while the war moved through Truk, Palau, the Philippines and Formosa.
   - **It collapsed in March 1945** to a median of 911 km. More than half of all sinkings (57%) were then in home waters, rising to about 80% in July 1945.
3. **Linear tests summarise a step function.** The −85 km/month slope is
   real, but no part of the war actually moved at that pace. Most of the 2,500
   km drop happened in one step in early 1945. The same lesson came out of the
   time-series analysis: a significant linear trend does not mean the change
   was gradual.
4. **The logistic odds ratio (2.3 per year) has a wide interval** (1.1–4.7).
   That is because the home-waters share is flat at 10–20% for three years
   and then jumps. A logit that is linear in time is misspecified for such a
   step, and the month-clustered errors correctly reflect that the evidence
   rests on only a few months.
5. **Warships and merchant ships tell slightly different stories.** The
   warship median falls about three times faster (Sen −100 vs −33 km/month).
   Warship losses moved from distant fleet battles (Solomons, Marianas, Leyte)
   to home harbours (the Kure raids of July 1945). Merchant losses had been
   spread along the sea lanes all war.

### 2.6 Possible explanations (hypotheses, not tested here)

| Break | Possible cause |
|---|---|
| Nov 1942 (distance up) | Fighting concentrated at the southern edge: the Guadalcanal naval battles and the Buna–Gona supply runs |
| May 1943 (distance down to the plateau) | Losses shift from the South Pacific periphery to the central sea lanes as US submarines and air power reach them |
| Mar 1945 (collapse) | The fall of the Philippines and Iwo Jima cut off the southern routes; carrier raids on Japan itself; Operation Starvation mining of Japan's straits and ports from late March 1945 |

### 2.7 Limitations

- **Ships without a position (38) are left out.** Most of them are "off Japan"-type entries, so the 1945 home-waters share is, if anything, slightly *understated*.
- **Quantile-regression intervals treat ships as independent,** so they are too narrow. The clustered logistic regression and the month-level Mann-Kendall test avoid this. Their p-values are much larger but still clearly significant.
- **The breaks are found and then described.** The phase intervals describe precision, not tests of the breaks.

---

## 3. B1: which weapons sank which ship types

### 3.1 Question and hypotheses

Did each weapon have its own typical victims? For example, did submarines
mainly hunt merchant ships on the open sea, aircraft strike harbours, and
mines close the coastal routes?

| Test | H₀ | H₁ |
|---|---|---|
| Chi-square test of independence (per fleet) | Ship type and weapon are independent | They are associated |
| Adjusted residual, per cell | This type × weapon count equals its expected value | It is higher or lower |
| Log-linear test given year | Type and weapon are independent *within each year* | They are associated even within years |

### 3.2 Which ships are analysed

A contingency table needs each ship in exactly one cell, so the analysis
keeps ships sunk by **one** of four weapons: submarine, aircraft, surface
warship or mine (all Allied nations). Left out:

| Reason | Ships |
|---|---:|
| Credited to Japan itself (accidents and Japan's own mines) | 131 |
| Sunk by two or more different weapons | 53 |
| Shore battery, sabotage or unknown cause | 42 |

That leaves **2,806 ships: 634 warships and 2,172 merchant ships.** Ship
types are the 15 grouped types used throughout the project. Warships and
merchant ships are separate tables, because their types don't overlap.

### 3.3 Methods

**Chi-square test of independence:**
$$X^2 = \sum_{ij}\frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \qquad E_{ij} = \frac{R_i C_j}{n}.$$

Asymptotically $X^2 \sim \chi^2_{(r-1)(c-1)}$, but the approximation is poor
when many expected counts are below 5. That is the case here: 11 of 36 cells
for warships and 7 of 24 for merchant ships. So the p-value reported is a
**permutation p-value**:
- weapon labels are shuffled among the ships 9,999 times, which keeps both margins fixed, exactly as the test conditions on them;
- $X^2$ is recomputed for each shuffle;
- $p = (1 + \#\{X^2_{\text{perm}} \ge X^2_{\text{obs}}\}) / 10{,}000$, so the smallest possible value is 0.0001.

**Effect size: Cramér's V, bias-corrected** (Bergsma, 2013).
$V = \sqrt{X^2 / (n(\min(r,c)-1))}$ runs from 0 (independent) to 1 (each
type sunk by only one weapon). Plain V is biased upwards in sparse tables. The
corrected version replaces $X^2/n$ by
$\max(0,\ X^2/n - (r-1)(c-1)/(n-1))$ and corrects $r$ and $c$ in the same way.

**Adjusted standardised residuals** (Haberman, 1973):
$$r_{ij} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}(1 - R_i/n)(1 - C_j/n)}} \;\overset{H_0}{\approx}\; N(0,1).$$

They show *which* cells drive the association. With 36 (or 24) cells
inspected, a cell is called significant only beyond the Bonferroni threshold
$z_{1-\alpha/(2m)}$, which is **3.20** for warships and **3.08** for merchant
ships.

**Correspondence analysis (CA)** maps rows and columns into one space:
- with $P$ = table/$n$ and masses $r$, $c$, take the SVD of $S = D_r^{-1/2}(P - rc^\top)D_c^{-1/2} = U\Sigma V^\top$;
- principal coordinates are $F = D_r^{-1/2}U\Sigma$ (types) and $G = D_c^{-1/2}V\Sigma$ (weapons);
- the total inertia equals $X^2/n$, and each dimension's share says how much of the association it shows.

In the symmetric map, a ship type lying near a weapon was sunk by it more
often than chance would predict.

**A log-linear test of type × weapon given year.** Weapons were active at
different times (mines almost only in 1945), and the mix of ship types afloat
changed too. A type–weapon association could therefore be nothing more than
both changing over time. The 3-way table (type × weapon × year) is fitted with
Poisson log-linear models:

- $H_0$: $\log\mu = [TY] + [WY]$ (type and weapon independent within each year);
- $H_1$: $\log\mu = [TY] + [WY] + [TW]$.

The drop in deviance is compared with $\chi^2_{(r-1)(c-1)}$. This is the
log-linear analogue of a Cochran–Mantel–Haenszel test.

### 3.4 Results

**Overall tests** (`data/weapon_ship/tests.csv`):

| Fleet | $X^2$ (df) | Permutation p | Cramér's V (corrected) | Type × weapon given year: LR (df), p | CA dimensions 1 / 2 |
|---|---|---:|---:|---|---|
| Warships | 218.7 (24) | **0.0001** (the minimum possible) | **0.32** (plain 0.34) | 194.3 (24), p < 10⁻²⁰ | 85% / 13% |
| Merchant ships | 94.0 (15) | **0.0002** | **0.11** (plain 0.12) | 61.0 (15), p < 10⁻⁶ | 76% / 20% |

**Cells beyond the Bonferroni threshold** (`contingency_*.csv`):

| Fleet | Ship type × weapon | Observed | Expected | Adjusted residual |
|---|---|---:|---:|---:|
| Warships | **Submarine × surface warships** | 70 | 21.7 | **+12.7** |
| | Submarine × aircraft | 13 | 47.2 | −7.2 |
| | Escort × surface warships | 7 | 36.1 | −6.4 |
| | Landing ship / fast transport × aircraft | 39 | 23.6 | +4.4 |
| | Escort × aircraft | 99 | 78.3 | +3.7 |
| | Mine warfare × submarines | 31 | 19.2 | +3.5 |
| Merchant | **Cargo × mines** | 213 | 161.5 | **+7.0** |
| | Cargo × submarines | 659 | 713.5 | −4.8 |
| | Passenger-cargo × submarines | 237 | 195.9 | +4.7 |
| | Tanker × aircraft | 111 | 84.5 | +3.8 |
| | Passenger-cargo × mines | 23 | 44.3 | −3.75 |
| | Converted auxiliary × mines | 2 | 15.4 | −3.75 |
| | Converted auxiliary × submarines | 85 | 68.0 | +3.1 |

### 3.5 Observations

1. **Each weapon had its own victims, and much more so among warships**
   (V = 0.32, a moderate association) than among merchant ships (V = 0.11,
   weak but clearly real).
2. **Japanese submarines were killed by surface warships.** 70 of the 112
   Japanese submarines in this table were sunk by surface ships, more than
   three times the 22 expected under independence. Aircraft sank only 13, a
   quarter of what would be expected. This is the largest signal in either
   table, and the hunt of submarines by Allied destroyers and destroyer escorts
   stands out in the data.
3. **Aircraft killed small escorts and transports.** Escorts (+3.7) and
   landing ships / fast transports (+4.4) were over-represented among aircraft
   victims. These were the ships that ran supply and reinforcement convoys into
   contested areas within range of land-based and carrier aircraft. Escorts
   were *under*-represented among surface-ship victims (−6.4).
4. **US submarines sank Japanese mine-warfare ships** (+3.5). Minesweepers
   and minelayers were widely used as convoy escorts, which put them in front
   of submarines.
5. **Among merchant ships, mines hit cargo ships** (+7.0) and not
   passenger-cargo ships or converted auxiliaries. **Submarines hit the bigger
   ocean-going types**: passenger-cargo ships (+4.7) and converted auxiliaries
   (+3.1). **Aircraft hit tankers** (+3.8). The CA map makes the pattern
   visible: dimension 1 (76%) runs from mines and cargo ships at one end to
   submarines, transports, passenger-cargo ships and converted auxiliaries at
   the other. Dimension 2 (20%) separates tankers and aircraft.
6. **The pattern is not just timing.** Holding the year fixed, the association
   remains overwhelming for warships (LR 194 vs $X^2$ 219). For merchant ships
   it drops by about a third (61 vs 94) but stays highly significant. So
   roughly a third of the merchant association is "this weapon was used in the
   year these ships were common", and the rest is genuine targeting or
   exposure.

### 3.6 Possible explanations (hypotheses)

- **Cargo × mines:** the 1945 mining campaign closed Japan's coastal and
  inland routes, used mostly by smaller cargo ships. Large passenger-cargo
  ships and converted auxiliaries were by then mostly already sunk, or no
  longer sailing those routes.
- **Tanker × aircraft:** tankers bringing oil from the south had to pass
  Formosa and Indochina, where the carrier raids of late 1944 and January 1945
  caught them in convoys and anchorages.
- **Submarine × surface:** submarines operating near Allied forces faced
  escort groups with sonar. Aircraft found it hard to catch a submerged boat.

### 3.7 Limitations

- **Single-weapon ships only.** The 53 multi-weapon ships are left out. That is about 2% of the ships, so the effect is small.
- **Types are grouped.** Within "escort" (frigates, subchasers, torpedo boats, gunboats) the pattern could differ.
- **The permutation p-value is bounded** at 1/(1 + 9,999). The merchant value 0.0002 is not a weakness of the result: sparse cells (e.g. "other merchant × surface", expected count 0.06) give the permutation distribution a long right tail, which the asymptotic χ² ignores.
- **Association, not cause:** weapon–type pairings reflect both *choice* (what attackers targeted) and *exposure* (where each type sailed). The data can't separate the two.

---

## 4. B2: ship size by weapon

### 4.1 Question and hypotheses

Did some weapons sink bigger ships? And if so, is that just because they sank
different *types* of ship, or ships from different *years*?

| Test | H₀ | H₁ |
|---|---|---|
| Kruskal–Wallis (per fleet) | Tonnage has the same distribution for every weapon | At least one weapon's victims tend to be larger or smaller |
| Dunn post-hoc (per pair, Holm-corrected) | The two weapons' victims have the same tonnage distribution | They differ |
| Regression coefficient (raw / adjusted) | Typical tonnage of this weapon's victims = that of submarine victims | It differs |

### 4.2 Methods

Same ships as B1 (2,806). Tonnage is as printed, with estimates included.
Warships and merchant ships are separate throughout, because of the different
units.

**Kruskal–Wallis test.** A rank-based one-way ANOVA:
$$H = \frac{12}{n(n+1)}\sum_g n_g \bar R_g^2 - 3(n+1) \;\overset{H_0}{\sim}\; \chi^2_{G-1}$$
with a tie correction. It needs no distributional assumption, which matters
because tonnage is very right-skewed. The effect size is
$\varepsilon^2 = H/(n-1)$, roughly the share of rank variance explained by
weapon.

**Dunn's test** compares each pair of weapons using the pooled ranks from
Kruskal–Wallis. p-values are **Holm**-corrected for the 6 pairs (step-down
Bonferroni: it controls the family-wise error rate and is uniformly more
powerful than Bonferroni).

**Cliff's δ** is the pairwise effect size:
$\delta = P(X_a > X_b) - P(X_a < X_b) = 2U/(n_a n_b) - 1$, from the
Mann–Whitney $U$. It runs from −1 to 1. As a guide: |δ| < 0.15 is negligible,
< 0.33 small, < 0.47 medium, and otherwise large (Romano et al., 2006).

**Regression on log tonnage**, with submarines as the reference:
$$\log(\text{tonnage}_i) = \beta_0 + \sum_w \beta_w\,\mathbb{1}[\text{weapon}_i = w]\;\big[+\ \text{ship type effects} + \text{year effects}\big] + \varepsilon_i.$$

- $100(e^{\beta_w}-1)$ is the % difference in *typical* (geometric-mean) tonnage against submarine victims.
- The **raw** model has weapon only. The **adjusted** model compares ships of the *same type sunk in the same year*.
- If a gap shrinks a lot after adjustment, it was mostly a different mix of types or periods.
- **HC3** heteroskedasticity-robust standard errors are used, because the spread of log tonnage differs between groups.

### 4.3 Results

**Overall tests:**

| Fleet | Kruskal–Wallis H (df) | p | ε² | Median tonnage: submarine / aircraft / surface / mine |
|---|---|---:|---:|---|
| Warships (tons) | 22.0 (3) | 0.0001 | 0.035 | 900 / 1,000 / **1,785** / 988 (n = 217 / 267 / 123 / 27) |
| Merchant (GRT) | 171.2 (3) | 7 × 10⁻³⁷ | 0.079 | **3,967** / 2,722 / 3,527 / **1,500** (n = 1,153 / 746 / 12 / 261) |

**Pairs** (Dunn, Holm-corrected; Cliff's δ is positive when the first weapon's
victims are bigger):

| Fleet | Pair | Holm p | Cliff's δ |
|---|---|---:|---:|
| Warships | Surface vs submarine | 0.002 | +0.24 (small) |
| | Surface vs aircraft | <0.001 | +0.27 (small) |
| | Surface vs mine | 0.017 | +0.35 (medium) |
| | Other 3 pairs | > 0.9 | |
| Merchant | Submarine vs aircraft | <0.001 | +0.24 (small) |
| | Submarine vs mine | <0.001 | **+0.48 (large)** |
| | Aircraft vs mine | <0.001 | +0.19 (small) |
| | Surface vs mine | 0.009 | +0.47 (medium–large; n = 12) |
| | Submarine vs surface; aircraft vs surface | 0.77; 0.17 | |

**Regression** (% difference in typical tonnage vs submarine victims, 95% CI,
HC3):

| Fleet | Weapon | Raw | Adjusted for type and year |
|---|---|---|---|
| Warships | Aircraft | −11% (−29, +11), p = 0.30 | **−16%** (−26, −4), p = 0.009 |
| | Surface | **+27%** (+2, +58), p = 0.03 | **−15%** (−26, −3), p = 0.018 |
| | Mine | −32% (−55, +3), p = 0.07 | −19% (−40, +10), p = 0.17 |
| | R² | 0.016 | 0.71 |
| Merchant | Aircraft | **−31%** (−36, −26), p < 0.001 | **−20%** (−26, −14), p < 0.001 |
| | Surface | 0% (−42, +73), p = 1.0 | −7% (−41, +45), p = 0.75 |
| | Mine | **−49%** (−54, −44), p < 0.001 | **−18%** (−27, −7), p = 0.001 |
| | R² | 0.086 | 0.22 |

### 4.4 Observations

1. **Merchant ships: submarines sank the biggest ones, mines the smallest.**
   The median submarine victim was about 4,000 GRT, against 2,700 for aircraft
   and 1,500 for mines. Submarine vs mine is a *large* effect (δ = 0.48).
2. **Most of the mine gap was timing and type, not the weapon.** The raw −49%
   shrinks to −18% once ship type and year are held fixed. On the log scale
   that removes about **70%** of the gap: from ln 0.51 = −0.68 to
   ln 0.82 = −0.20. Mines worked in 1945, when the large ships were already
   gone (consistent with the "ships got smaller" finding of the time-series
   analysis). For aircraft, about 40% of the gap is explained (−31% → −20%).
   A real gap remains: of two cargo ships sunk in the same year, the
   submarine's victim was typically about 20% larger.
3. **Warships: a reversal (a Simpson's-paradox pattern).** In the raw data,
   surface warships sank the *largest* victims (median 1,785 t, +27% vs
   submarines). Within the same ship type and year they sank *smaller* ones
   (−15%). The raw advantage comes from *what* surface ships fought: Japanese
   submarines and destroyers, which are mid-sized warships, rather than the
   many small escorts that aircraft and submarines sank (B1). Ship type
   explains most of the variation in warship size (R² rises from 0.02 to
   0.71), as you would expect.
4. **Effects are modest overall** (ε² = 0.035 and 0.079). Weapon explains
   little of the variation in ship size, much less than ship type does. The
   differences are real but secondary.

### 4.5 Limitations

- **Tonnage is the report's figure,** including estimates (printed with "e").
- **Few surface-ship sinkings of merchant ships** (n = 12), so that row's intervals are very wide.
- **The adjusted regression assumes additive effects** on the log scale (no weapon × type interaction).
- **The regression treats ships as independent.** Ships sunk in the same raid share conditions, so the intervals are a little optimistic.

---

## 5. C1: sinking events

### 5.1 Question

3,032 rows are hard to remember; *events* are not. Which raids, battles and
convoy massacres caused the most losses at once, and who caused them? C1 is
mainly descriptive, plus a sensitivity analysis. Its formal tests are in C2.

### 5.2 Method: space-time single-linkage grouping

Two ships are **linked** when both conditions hold:
$$|\text{date}_i - \text{date}_j| \le 1 \text{ day} \quad\text{and}\quad d_{ij} \le 100 \text{ km},$$
where $d_{ij}$ is the great-circle distance. An **event** is a connected
component of the link graph, i.e. single-linkage clustering. Chains are
allowed: a convoy attacked over several days by a moving submarine pack is
one event, as long as each sinking is close to the next.

- **Why 1 day and 100 km:** 1 day allows for attacks that run past midnight and for dates recorded a day apart; 100 km is about the reach of one carrier strike on a harbour, or one convoy battle.
- **Who can be linked:** the 2,993 ships with both a position and an exact date. The other 39 are one-ship events.
- **Naming:** each event's place is its **medoid**, the actual sinking site with the smallest total distance to the others, named after the nearest place in the project's gazetteer. The average position was tried first; on chained events it can fall between places, and it named the Kure raid after the Shimonoseki Strait.
- **Main agent and weapon:** the agent and weapon with the largest share of the event's credits (each ship's credits weighted 1/n).
- **Ranking:** events are ranked by number of ships. That unit is the same for both fleets; the two kinds of tonnage can't be added.

**Sensitivity:** the grouping is re-run with other thresholds, to see whether
the conclusions depend on the choice.

### 5.3 Results

**The 20 biggest events** (`data/events/events.csv`; tonnage per fleet):

| # | Date | Place | Ships (warships + merchant) | Main agent and weapon | Naval t / merchant GRT |
|---:|---|---|---|---|---|
| 1 | 17 Feb 1944 | Truk | 36 (8 + 28) | US aircraft | 18,440 / 178,643 |
| 2 | 14–15 Jul 1945 | Tsugaru Strait (Hokkaido) | 32 (5 + 27) | US aircraft | 4,030 / 57,228 |
| 3 | 30 Mar 1944 | Palau | 29 (7 + 22) | US aircraft | 11,210 / 99,538 |
| 4 | 21–22 Sep 1944 | Manila Bay | 27 (3 + 24) | US aircraft | 4,100 / 105,307 |
| 5 | 23–28 May 1945 | Shimonoseki Strait | 25 (1 + 24) | US mines | 800 / 50,027 |
| 6 | 13–14 Nov 1944 | Manila Bay | 24 (6 + 18) | US aircraft | 13,550 / 93,406 |
| 7 | 23–25 Jul 1945 | Inland Sea (Kure) | 20 (4 + 16) | US aircraft | **82,000** / 33,714 |
| 8 | 29 May – 7 Jun 1945 | Shimonoseki Strait | 19 (0 + 19) | US mines | 0 / 34,765 |
| 9 | 12 Oct 1944 | Formosa Strait | 17 (0 + 17) | US aircraft | 0 / 48,391 |
| 10 | 12 Jan 1945 | Cam Ranh Bay | 17 (3 + 14) | US aircraft | 7,600 / 56,709 |
| 11 | 12–18 Jul 1945 | Shimonoseki Strait | 16 (0 + 16) | US mines | 0 / 29,377 |
| 12 | 12 Jan 1945 | Saigon | 15 (4 + 11) | US aircraft | 3,500 / 51,554 |
| 13 | 12 Jun 1944 | Marianas (Saipan) | 14 (4 + 10) | US aircraft | 895 / 30,910 |
| 14 | 12 Sep 1944 | Ormoc Bay (Leyte) | 14 (2 + 12) | US aircraft | 932 / 22,759 |
| 15 | 22–25 Jun 1945 | Shimonoseki Strait | 13 (0 + 13) | US mines | 0 / 21,410 |
| 16 | 10–11 Nov 1944 | Ormoc Bay (Leyte) | 12 (6 + 6) | US aircraft | 8,630 / 30,080 |
| 17 | 11–17 May 1945 | Osaka / Kobe | 12 (0 + 12) | US mines | 0 / 32,315 |
| 18 | 3 Mar 1943 | Open sea (Bismarck Sea) | 11 (4 + 7) | US aircraft | 7,550 / 28,237 |
| 19 | 12 Jan 1945 | Cam Ranh Bay | 11 (6 + 5) | US aircraft | 3,950 / 20,096 |
| 20 | 5–7 May 1945 | Shimonoseki Strait | 10 (1 + 9) | US mines | 630 / 33,542 |

**Sensitivity** (`event_sensitivity.csv`):

| Days, km | Events | Multi-ship events | Ships in multi-ship events | Largest event | Median size of top 20 |
|---|---:|---:|---:|---|---:|
| 0, 50 | 2,215 | 379 | 39% | 33, Truk 17 Feb 1944 | 9 |
| 1, 50 | 2,103 | 392 | 44% | 33, Truk | 12 |
| **1, 100 (main)** | **1,954** | **407** | **49%** | **36, Truk** | **16.5** |
| 2, 200 | 1,707 | 438 | 58% | 102, Shimonoseki from 23 May 1945 | 17.5 |
| 3, 300 | 1,442 | 413 | 66% | 289, Shimonoseki from 30 Apr 1945 | 19.5 |

### 5.4 Observations

1. **Every one of the 20 biggest events was American: 14 carrier or
   land-based air strikes and 6 mining episodes.** None was a submarine
   attack, even though submarines sank more ships than any other weapon.
   Submarines sank ships one or two at a time, while aircraft destroyed ships
   by the dozen in harbours and anchorages: Truk, Palau, Manila, Kure,
   Hokkaido and the Indochina coast.
2. **The largest single event was the Truk raid of 17 February 1944.** It
   sank 36 ships, including 178,643 GRT of merchant shipping in one day, and
   it is the largest event at every tight threshold.
3. **Single events can contain a fleet's worth of tonnage.** The Kure raids of
   23–25 July 1945 sank 82,000 tons of warships, including battleships and
   carriers already immobilised in harbour, in 20 ship losses.
4. **The 12 January 1945 South China Sea raid splits into three events**
   (#10, #12, #19: 43 ships in one day) because its targets lay more than
   100 km apart along the Indochina coast. This is the threshold working as
   designed: these were separate strikes on separate harbours.
5. **Mining "events" are long.** Events #5, #8, #11 and #15 run over 4–10
   days in the same strait. A mine sinks a ship days after it is laid, so a
   minefield produces a steady trickle, chained together by the 1-day link.
6. **The only pre-1944 event in the top 20** is the Battle of the Bismarck
   Sea (3 March 1943, 11 ships, here labelled "open sea" as it is far from
   any gazetteer place). Mass destruction was a feature of 1944–45.
7. **Sensitivity: robust at tight thresholds, chaining at loose ones.** From
   0 days / 50 km to 1 day / 100 km, the same events lead, and about 40–50% of
   ships are sunk together with at least one other ship. At 2 days / 200 km,
   single linkage *chains* the whole spring-1945 mining of the Shimonoseki
   Strait into one 102-ship "event", and at 3 days / 300 km into 289 ships.
   That is a known weakness of single linkage, and the reason the main run
   uses tight thresholds.

---

## 6. C2: concentration of losses

### 6.1 Questions and hypotheses

Were losses spread evenly across the war's days, or packed into a few
catastrophic ones? And is that only because some *months* were worse than
others (the trend), or were losses bunched on particular *days* within months?

| Test | H₀ | H₁ |
|---|---|---|
| Within-month permutation test of the Gini (ships; naval tons; merchant GRT) | Within each month, each ship's day of sinking is uniform and independent of the others. Monthly totals, and so the trend, are fixed | Losses cluster on particular days more than this allows |
| Vuong test, event sizes | The geometric and zeta (power-law) models fit equally well | One fits better |

### 6.2 Methods

**Daily series.** Losses are summed for every day from 8 Dec 1941 to 15 Aug
1945 (**1,347 days**), with quiet days counting as 0. This is done for ships
(3,026 ships with an exact date), for naval tons and for merchant GRT.

**Concentration curve and Gini coefficient.** The days are sorted from worst
to best, and the cumulative share of losses is plotted against the
cumulative share of days. A perfectly even war is the diagonal. The Gini
coefficient (for $x$ sorted ascending, $i = 1, \dots, n$) is
$$G = \frac{2\sum_i i\,x_{(i)}}{n\sum_i x_i} - \frac{n+1}{n}.$$
It is 0 for even losses and approaches 1 when everything falls on one day.
The table also gives the share of losses on the worst 1%, 5% and 10% of days.

**Within-month permutation test.** A daily series has a high Gini partly
because some months were far worse than others. To isolate *day-level*
bunching:
- each ship is moved to a uniformly random day of its own month (within the war's dates);
- the Gini is recomputed;
- this is repeated 999 times.

This null keeps every monthly total, and so the whole trend, exactly. Only
the *arrangement within months* is random. The p-value is
$(1 + \#\{G_{\text{null}} \ge G_{\text{obs}}\})/1{,}000$, so the minimum is 0.001.

**Event-size distribution.** From C1, the sizes of the 1,915 events of
linkable ships. Two one-parameter models on {1, 2, 3, …} are fitted by
maximum likelihood:
- **Geometric:** $P(X = x) = p(1-p)^{x-1}$, with $\hat p = 1/\bar x$. This is what happens if, once an event starts, each extra ship is lost with a constant probability: big events are *exponentially* rare.
- **Zeta (discrete power law):** $P(X = x) = x^{-a}/\zeta(a)$, $a > 1$. Big events are only *polynomially* rare: a heavy tail.

**Vuong's test** compares the two non-nested models. With
$\ell_i^{Z} - \ell_i^{G} = d_i$ the per-observation log-likelihood difference,
$$V = \frac{\sqrt n\,\bar d}{s_d} \;\overset{H_0}{\approx}\; N(0,1).$$
$V > 0$ favours the zeta. Both models have one parameter, so no penalty term
is needed.

### 6.3 Results

**Concentration over days** (`data/events/concentration_tests.csv`):

| Measure | Days with no loss | Gini | Null Gini, mean (95% range) | Permutation p | Worst 1% of days | Worst 5% | Worst 10% |
|---|---:|---:|---|---:|---:|---:|---:|
| Ships | 29% | **0.61** | 0.51 (0.50–0.52) | **0.001** | 11% | **29%** | 41% |
| Naval tons | 71% | **0.92** | 0.89 (0.88–0.89) | **0.001** | **47%** | **76%** | 86% |
| Merchant GRT | 36% | **0.68** | 0.58 (0.57–0.59) | **0.001** | 13% | **32%** | 47% |

(1% of days is 13 days; 5% is 67 days.)

**Event sizes** (`event_size_models.csv`, 1,915 events):

| Statistic | Value |
|---|---|
| Events with a single ship | 78.8% |
| Ships in events of ≥ 2 / ≥ 5 / ≥ 10 ships | 49.6% / 24.1% / 12.8% |
| Geometric fit | $\hat p$ = 0.640, log-likelihood −1,956 |
| Zeta fit | $\hat a$ = 2.72, log-likelihood −1,616 |
| Vuong V | **+5.92**, p ≈ 3 × 10⁻⁹, favouring the **zeta (heavy tail)** |

### 6.4 Observations

1. **Losses were heavily concentrated.** On the worst 5% of days (67 days),
   Japan lost 29% of all its ships and 32% of its merchant tonnage. The
   observed Ginis (0.61 and 0.68) lie far above the whole null range,
   and the permutation p is at its minimum (0.001) for all three measures.
2. **This is day-level bunching, not just the trend.** The null keeps every
   month's total, and the observed Gini still exceeds every one of the 999
   reshuffles. Within any month, losses came in bursts (raids, convoy battles,
   minefield clusters) rather than being spread across the days.
3. **Warship tonnage was extraordinarily concentrated.** 47% of all warship
   tonnage lost in the war went down on just 13 days (1%), and 76% on 67 days.
   Those days are the great battles and raids: Midway, the Philippine Sea,
   Leyte Gulf, Kure, the YAMATO sortie. The null Gini is already high here
   (0.89), because warship losses are rare and tonnage is dominated by a few
   giants, but the observed value (0.92) is still outside its range.
4. **Event sizes have a heavy tail.** The geometric model badly
   under-predicts large events: it expects about 0.001 events of 15+ ships
   (1,915 × 0.36¹⁴), while there were 12. The power law with exponent 2.7 follows the observed
   tail across more than an order of magnitude (CCDF chart). Vuong's test
   strongly prefers it.
5. **But "heavy-tailed" is not "a power law".** The observed CCDF bends down
   beyond about 25 ships, because a harbour holds only so many ships. Only two
   models were compared; a lognormal or a truncated power law might fit as
   well or better. The safe conclusion is that event sizes are far more
   heavy-tailed than independent losses would produce, i.e. that losses came
   in catastrophes. Not that they follow a specific law.

---

## 7. How the five fit the story

| Chapter | Finding | Evidence |
|---|---|---|
| **Where** | The war first went *outwards* (Guadalcanal), sat on a long plateau about 3,000 km from Tokyo, then collapsed onto Japan's home waters in March 1945 | A1 change points; median 4,370 → 3,035 → 911 km; home share 10% → 57% |
| **Who killed what** | Each weapon had its victims: surface ships killed submarines, aircraft killed escorts, transports and tankers, mines killed coastal cargo ships, and submarines killed the big ocean-going ships | B1 residuals; association holds within years |
| **How big** | Submarines sank the largest merchant ships; the mines' "small ships" are mostly a 1945 effect | B2: −49% → −18% after adjustment |
| **How sudden** | Losses came in catastrophes: the 20 biggest events were all US air raids or minefields; 5% of days held about a third of losses; event sizes are heavy-tailed | C1 top 20; C2 Gini, permutation and Vuong tests |

Together with the time-series analysis (losses rose in *steps*, not
gradually), the picture is consistent at every scale:

- **Months:** step changes at a few points in the war.
- **Geography:** a sudden collapse onto home waters in early 1945.
- **Days:** bursts of loss concentrated on particular days.

Japan's maritime power was not worn down evenly. It was broken in a series
of discrete blows.

---

## 8. Summary of all hypothesis tests

| # | Test | H₀ | Result |
|---|---|---|---|
| A1 | Quantile regression, median distance on time | Slope = 0 | **Rejected:** −85 km/month, p < 10⁻⁹⁰ |
| A1 | Spearman, date vs distance | ρ = 0 | **Rejected:** ρ = −0.38 |
| A1 | Mann-Kendall (TFPW), monthly median | No trend | **Rejected** (all, warships, merchant: p ≤ 10⁻⁴) |
| A1 | Clustered logistic regression, home waters | Odds ratio = 1 | **Rejected:** OR 2.3/year, p = 0.02 |
| B1 | χ² independence (permutation), warships | Independent | **Rejected:** p = 0.0001, V = 0.32 |
| B1 | χ² independence (permutation), merchant | Independent | **Rejected:** p = 0.0002, V = 0.11 |
| B1 | Log-linear type × weapon given year | Conditional independence | **Rejected** for both fleets |
| B2 | Kruskal–Wallis, warship tonnage | Same distribution | **Rejected:** p = 0.0001, ε² = 0.035 |
| B2 | Kruskal–Wallis, merchant tonnage | Same distribution | **Rejected:** p = 7 × 10⁻³⁷, ε² = 0.079 |
| B2 | Adjusted regression, merchant aircraft / mine vs submarine | No difference | **Rejected:** −20%, −18% |
| C2 | Within-month permutation, Gini (3 measures) | No day-level bunching | **Rejected:** p = 0.001 (the minimum) for all |
| C2 | Vuong, geometric vs zeta | Equal fit | **Rejected** in favour of the heavy tail, V = 5.9 |

Multiple testing: the headline tests have p-values far below any reasonable
correction (e.g. Bonferroni over the 12 rows: α = 0.004). The exception is the
logistic home-waters test (p = 0.02). It is consistent with, but weaker than,
the A1 change-point and quantile results.

---

## 9. Limitations and next steps

**Common to all five**

- **No exposure.** The data records where and how ships died, not how many ships sailed where. Every "over-representation" mixes targeting with exposure. Fleet-size and route data, e.g. from the Wikipedia supplement, would allow rates.
- **Dependence.** Ships sunk in the same raid are not independent. A1 and C2 handle this by clustering or permutation by month; B2's regression does not.
- **OCR data.** The totals match the report, but individual positions or dates can contain misreads.

**Next steps**

1. **A1:** use distance to the nearest home-island coast, and an exposure-weighted version once route data is added.
2. **B1:** use finer ship types (e.g. frigate vs subchaser), and add credits split by nation.
3. **B2:** use mixed models with raid-level random effects; check robustness without estimated tonnages.
4. **C1:** try density-based space-time clustering (ST-DBSCAN), to avoid single-linkage chaining.
5. **C2:** fit lognormal and truncated power-law models to event sizes (Clauset et al., 2009); model burstiness with a self-exciting (Hawkes) point process.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Adjusted residual** | (observed − expected) divided by its standard error under independence; ≈ N(0, 1) |
| **Bonferroni / Holm** | Multiple-testing corrections controlling the chance of any false positive; Holm is the more powerful step-down version |
| **Check loss** | The asymmetric absolute loss minimised by quantile regression |
| **Cliff's δ** | P(a random value from A > one from B) − P(a random value from A < one from B) |
| **Cluster bootstrap** | Resampling whole groups (here months) instead of single observations, to respect within-group dependence |
| **Cluster-robust SE** | Standard errors allowing correlation within groups (here months) |
| **Correspondence analysis** | SVD-based map of a contingency table placing rows near the columns they co-occur with |
| **Cramér's V** | Effect size for a contingency table, from 0 (independent) to 1 |
| **Gini coefficient** | Inequality measure: 0 = equal, near 1 = concentrated in one unit |
| **HC3** | A heteroskedasticity-consistent standard error, good in small or unbalanced samples |
| **Kruskal–Wallis** | Rank-based test that several groups have the same distribution |
| **Log-linear model** | Poisson regression on cell counts of a multi-way table; tests association structures |
| **Medoid** | The member of a group with the smallest total distance to the others |
| **Permutation test** | Builds the null distribution by reshuffling labels that would be exchangeable under H₀ |
| **Quantile regression** | Regression for a chosen percentile instead of the mean |
| **Single linkage** | Clustering where a chain of close pairs joins a group; prone to "chaining" |
| **Simpson's paradox** | An association that reverses when a third variable (here ship type) is held fixed |
| **Vuong test** | Likelihood-ratio-based test comparing two non-nested models |
| **Zeta distribution** | Discrete power law, P(x) ∝ x⁻ᵃ |

---

## 11. References

- Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *Journal of the Korean Statistical Society*, 42, 323–328.
- Clauset, A., Shalizi, C. R. & Newman, M. E. J. (2009). Power-law distributions in empirical data. *SIAM Review*, 51, 661–703.
- Dunn, O. J. (1964). Multiple comparisons using rank sums. *Technometrics*, 6, 241–252.
- Greenacre, M. (2017). *Correspondence Analysis in Practice* (3rd ed.). CRC Press.
- Haberman, S. J. (1973). The analysis of residuals in cross-classified tables. *Biometrics*, 29, 205–220.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6, 65–70.
- Koenker, R. & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46, 33–50.
- Kruskal, W. H. & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *JASA*, 47, 583–621.
- Long, J. S. & Ervin, L. H. (2000). Using heteroscedasticity consistent standard errors in the linear regression model. *The American Statistician*, 54, 217–224.
- Romano, J., Kromrey, J. D., Coraggio, J. & Skowronek, J. (2006). Appropriate statistics for ordinal level data. Annual meeting of the Florida Association of Institutional Research.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica*, 57, 307–333.
- Yue, S., Pilon, P., Phinney, B. & Cavadias, G. (2002). The influence of autocorrelation on the ability to detect trend in hydrological series. *Hydrological Processes*, 16, 1807–1829.
