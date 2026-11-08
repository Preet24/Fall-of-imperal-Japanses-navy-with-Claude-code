# How the US destroyed Japan's fleets, month by month: a time-series analysis explained

This document explains the time-series analysis in `us_loss_time_series.py`:
the data, every method and statistical test (with model specifications,
estimators and null distributions), the diagnostics, the results, and how far
they can be trusted. Each method section starts with a plain-language summary
and then gives the technical detail. The [glossary](#13-glossary) defines
every term.

All numbers come from the files the script writes to `data/time_series/`. The
charts are in `figures/time_series/`. Equations use LaTeX (`$…$`), which
GitHub and VS Code's Markdown preview render.

---

## Contents

1. [The questions](#1-the-questions)
2. [The data](#2-the-data)
3. [The analysis in one picture](#3-the-analysis-in-one-picture)
4. [Preparing the series](#4-preparing-the-series)
5. [Methods and statistical tests](#5-methods-and-statistical-tests)
6. [Model diagnostics and assumptions](#6-model-diagnostics-and-assumptions)
7. [Results: long-term trend](#7-results-long-term-trend)
8. [Results: seasonality](#8-results-seasonality)
9. [Results: do the weapons differ?](#9-results-do-the-weapons-differ)
10. [Interpretation and hypotheses](#10-interpretation-and-hypotheses)
11. [Limitations, alternatives considered and robustness checks](#11-limitations-alternatives-considered-and-robustness-checks)
12. [Files and how to reproduce](#12-files-and-how-to-reproduce)
13. [Glossary](#13-glossary)
14. [References](#14-references)

---

## 1. The questions

The analysis asks four questions about Japanese ship losses to US forces,
December 1941 – August 1945:

| # | Question | Main method |
|---|---|---|
| Q1 | Did losses rise (or fall) over the war as a whole? | Mann-Kendall test with trend-free pre-whitening; Sen's slope |
| Q2 | Did the loss rate change *gradually* or in *sudden jumps*, and when? | Change-point detection (exact segmentation + BIC); segmented negative binomial / log-linear regression |
| Q3 | Is there a *seasonal* pattern, with some months of the year always worse than others? | Likelihood-ratio test of a first-order harmonic (sine + cosine) |
| Q4 | Did the weapons (submarines, aircraft, surface ships, mines) follow different patterns? | Pooled regression with weapon × spline and weapon × harmonic interactions; likelihood-ratio tests |

---

## 2. The data

### 2.1 Source

The ship lists of the JANAC report (1947), one row per Japanese ship sunk:

- `data/naval/naval_losses_15-44.csv`: 686 warships
- `data/merchant/merchant_losses_45-115.csv`: 2,346 merchant ships

Each row gives the date, ship type, tonnage, position, and who sank the ship
with what weapon (the "agent").

### 2.2 Why only the United States?

A time series needs enough events in each month to show a pattern. The US is
credited with about 90% of all Japanese losses. All other Allied nations
together sank about 70 ships in 45 months, so most of their months would be 0.
A trend or season cannot be estimated from such a sparse series.

### 2.3 The unit of analysis: one month

Each series is $y_t,\ t = 0,\dots,44$: one value per month from December 1941
(t = 0) to August 1945 (t = 44). Months with no losses count as 0. Leaving
them out would bias every rate upwards.

### 2.4 The series analysed

The two fleets are always kept apart. Each fleet gets one series for **all US
sinkings**, plus one per **weapon** with at least 100 (weighted) ships lost,
about 2 a month on average. Smaller weapon groups are too sparse to model but
still count in the "all" series.

| Fleet | Series | Ships (weighted) |
|---|---|---:|
| Warships | All US sinkings | 616.0 |
| | Submarines | 202.5 |
| | Aircraft | 268.1 |
| | Surface warships | 123.0 |
| Merchant ships | All US sinkings | 2,121.8 |
| | Submarines | 1,117.5 |
| | Aircraft | 735.8 |
| | Mines | 248.5 |

Left out as separate series: mines against warships (20 ships) and surface
ships against merchant ships (16 ships). Each of the 8 series is analysed
twice, as **ship count** and as **tonnage**, giving **16 series**.

### 2.5 Shared sinkings count as fractions

A ship with $n_i$ credited agents contributes weight $w_i = 1/n_i$ to each
agent, and $w_i \times \text{tonnage}_i$ of tonnage. The monthly value of a
series is therefore

$$y_t = \sum_{i \in \text{month } t} \sum_{c \in \text{credits}_i} \frac{1}{n_i}\,\mathbb{1}[c \text{ is US and matches the series}].$$

So $y_t$ is a non-negative real number (e.g. 12.5), not necessarily an
integer. This splitting guarantees that the weapon series add up to the "all"
series, and that no ship is counted twice.

### 2.6 Two kinds of tonnage

Warship tonnage is **standard displacement** (weight). Merchant tonnage is
**gross register tons, GRT** (internal volume, 100 ft³ = 1 GRT). These are
different units, so warship and merchant tonnage are never added together.

### 2.7 Checking the data against the report

Before any analysis, the script compares ships sunk by the US **alone** with
the report's Table II:

| | Our data | Report (Table II) |
|---|---|---|
| Warships | 611 ships, 1,822,210 t | 611 ships, 1,822,210 t ✔ exact |
| Merchant ships | 2,116 ships, 7,912,804 GRT | 2,117 ships, 7,913,858 GRT |

The merchant gap (1 ship, 1,054 GRT, 0.05% of ships) is explained in the
script:

- KENKOKU has no agent printed in the list, but the report counts it as a US sinking.
- There is also a 100 GRT misread that hasn't been found yet.

---

## 3. The analysis in one picture

```
 ship lists ──► US credits (weight 1/n) ──► 16 monthly series y_t
                                               │
                exposure e_t (war days / 30.44) ┘   rate r_t = y_t / e_t
                                               │
   ┌──────────────┬───────────────┬────────────┴───────┬──────────────────┐
   ▼              ▼               ▼                    ▼                  ▼
 STL trend   Mann-Kendall    change points       segmented GLM        harmonic LRT
 (describe)  TFPW (Q1)       Dynp + BIC (Q2)     NB2 / log-OLS (Q2)   (Q3)
                                                       │                  │
                              diagnostics: overdispersion LRT, Jarque-Bera,
                              Ljung-Box, AIC, BIC table ◄──────────────────┘
                                               │
                    Benjamini-Hochberg over the 16 series (MK and season)
                                               │
                pooled NB / OLS per fleet, weapon × spline / harmonic ──► Q4
```

---

## 4. Preparing the series

### 4.1 Exposure: adjusting for month length

Months differ in length, and the first and last months of the war are partial:

- **December 1941:** the first sinking is on 8 December (7 December in Hawaii is 8 December in Asia), so the month had **24 days** of war.
- **August 1945:** the last sinking is on 15 August, the day Japan announced its surrender, so the month had **15 days**.
- **Every other month** counts its real length (28–31 days).

The **exposure** of month *t* is

$$e_t = \frac{\text{war days in month } t}{30.44}, \qquad 30.44 = 365.25/12,$$

so an average-length month has $e_t \approx 1$ and August 1945 has $e_t = 0.49$.
The **rate** $r_t = y_t / e_t$ is "losses per average month". For example,
August 1945's 63 merchant ships in 15 days become about 128 per average month.

How exposure enters each method:

| Method | How exposure enters |
|---|---|
| STL, Mann-Kendall, change points | Run on the rate $r_t$ |
| Negative binomial (counts) | As an offset: $\log \mu_t = \log e_t + x_t^\top\beta$, so the model is for the rate and expects proportionally fewer losses in a short month |
| OLS (tonnage) | Response $\log(1 + r_t)$, using the tonnage rate |

Without exposure, August 1945 would look like a collapse at the end of the
war, and February's 28 days would create a fake seasonal dip.

### 4.2 Variance-stabilising transformations

Change-point detection with a squared-error cost assumes roughly constant
noise. Counts are roughly Poisson-like, so their variance grows with the mean.
Tonnage swings are roughly proportional to the level.

| Measure | Transformation | Justification |
|---|---|---|
| Ships | $z_t = \sqrt{r_t}$ | For a Poisson variable, $\operatorname{Var}(\sqrt{Y}) \approx 1/4$ whatever the mean (the delta method). It is close to the Anscombe transform $2\sqrt{Y + 3/8}$ |
| Tonnage | $z_t = \log(1 + r_t)$ | For multiplicative noise, the log makes the variance roughly constant. The "1 +" keeps months with zero tonnage usable. On the log scale, YAMATO (63,000 t) counts as only ~2.4× a 1,000-ton escort, instead of 63× |

The regressions model the raw counts directly (NB) or the log-tonnage (OLS),
so this table applies only to the change-point step.

---

## 5. Methods and statistical tests

> **Reading a hypothesis test.** Each test has a null hypothesis H₀ ("nothing
> is going on") and an alternative H₁. The p-value is $P(\text{statistic at
> least as extreme} \mid H_0)$. Here α = 0.05. A large p-value means "the data
> can't rule out H₀", **not** "H₀ is true".

### 5.1 STL decomposition: describing the trend

**In short.** STL splits the series into trend + season + remainder. Only the
**trend** is used, as a descriptive smoother (the black line in the charts).

**Technical detail.** STL (Cleveland et al., 1990) decomposes
$r_t = T_t + S_t + R_t$ by alternating LOESS smoothers:
1. **Inner loop:** detrend the series, smooth each calendar month's values (the "cycle-subseries") to get $S_t$, then smooth the deseasonalised series to get $T_t$.
2. **Outer loop (robust mode):** compute bisquare robustness weights from the remainder $R_t$ and repeat.

The weights downweight outlying months, such as a single carrier battle, so
they don't bend $T_t$.

Settings used (statsmodels `STL`):

| Parameter | Value | Why |
|---|---|---|
| `period` | 12 | Calendar year |
| `trend` (trend window) | **13** | The default for period 12 is ≈ 23 months, which on 45 months turned the 1945 mining campaign into a straight ramp from 1942. 13 is the smallest odd window longer than the period |
| `seasonal` | 7 (default) | Only matters for $S_t$, which is not used |
| `robust` | True | 15 outer iterations with bisquare weights |

**Why $S_t$ is not used.** With 3–4 cycles, each calendar month's seasonal
value is a LOESS smooth of only 3–4 points. It just copies whichever campaign
fell in that month. For mines it produced a "May–July season" that was really
the single 1945 mining campaign. The season is tested by a parametric model
instead (§5.5).

### 5.2 Mann-Kendall trend test with trend-free pre-whitening (Q1)

**In short.** The test counts, over all pairs of months, how often the later
month is higher. Mostly-higher means a rising trend. It is rank-based, so one
extreme month can't drive it.

**Hypotheses**

- **H₀:** the observations are independent and identically distributed. There is no monotonic trend.
- **H₁:** a monotonic trend (two-sided).

**Statistic.** For $n = 45$ observations:

$$S = \sum_{i<j} \operatorname{sgn}(x_j - x_i), \qquad
\operatorname{Var}(S) = \frac{n(n-1)(2n+5) - \sum_g t_g(t_g-1)(2t_g+5)}{18},$$

where $t_g$ is the size of the *g*-th group of tied values. Ties matter here,
because many months are exactly 0. The test then uses the continuity-corrected

$$Z = \frac{S - \operatorname{sgn}(S)}{\sqrt{\operatorname{Var}(S)}} \;\overset{H_0}{\sim}\; N(0,1).$$

**Sen's slope** (Theil–Sen estimator) is the median of all pairwise slopes:
$\hat\beta = \operatorname{median}_{i<j}\frac{x_j - x_i}{j - i}$. It has a
29% breakdown point, so up to 29% of the months could be arbitrary outliers
without moving it far.

**The autocorrelation problem.** H₀ assumes independence. Losses are
positively autocorrelated, because campaigns last several months. Positive
autocorrelation inflates $\operatorname{Var}(S)$ beyond the formula above, so
the plain test rejects H₀ too often.

**The fix: trend-free pre-whitening (TFPW; Yue et al., 2002).** It works in five steps:
1. Estimate Sen's slope $\hat\beta$ and detrend: $x'_t = x_t - \hat\beta t$.
2. Estimate the lag-1 autocorrelation $\hat\rho_1$ of $x'_t$.
3. Pre-whiten: $x''_t = x'_t - \hat\rho_1 x'_{t-1}$. This removes an AR(1) component without removing the trend.
4. Add the trend back: $x'''_t = x''_t + \hat\beta t$.
5. Run Mann-Kendall on $x'''_t$ (with $n - 1$ points).

Why TFPW and not the alternatives:
- **Plain pre-whitening** (without step 1) would also remove part of the trend.
- **The Hamed–Rao variance correction** was tried first. On these short, steep series its corrected variance turned negative for merchant aircraft tonnage, giving $\sqrt{\cdot}$ of a negative number and no p-value. Where both versions worked, they agreed.
- **Yue–Wang** gave clearly liberal results. For example, it called surface-warship tonnage significant (p < 0.0001) where TFPW, Hamed–Rao and even the plain test did not.

**Limits.** Mann-Kendall says nothing about the *shape* of the trend. A step
change and a steady ramp give similar $S$. That is why Q2 uses separate methods.

### 5.3 Change-point detection (Q2: when?)

**In short.** The method cuts the series into phases with different average
levels and finds the cut points that fit best. BIC decides how many cuts to
make.

**Model.** On the transformed series $z_t$ (§4.2), a piecewise-constant mean with $k$ breaks
$\tau_1 < \dots < \tau_k$:

$$z_t = m_s + \varepsilon_t \quad \text{for } t \in [\tau_{s-1}, \tau_s), \qquad \varepsilon_t \sim N(0, \sigma^2).$$

**Estimation.** For each $k \in \{0,1,2,3\}$, `ruptures.Dynp` solves the
optimal partition exactly, by dynamic programming, with the L2 cost:

$$\min_{\tau_1,\dots,\tau_k} \sum_{s=1}^{k+1} \sum_{t \in \text{seg}_s} (z_t - \bar z_s)^2 \;=\; \text{RSS}_k,$$

subject to a minimum segment length of **6 months**. Dynamic programming takes
$O(k n^2)$ time. That is trivial at $n = 45$, and unlike binary segmentation
(a greedy approximation) it is guaranteed to find the global optimum.

**Choosing k by BIC.** With the Gaussian likelihood profiled over $\sigma^2$:

$$\text{BIC}_k = n \log\!\left(\frac{\text{RSS}_k}{n}\right) + (2k+1)\log n.$$

The parameter count is $k+1$ segment means plus $k$ break locations. This
treats each break location as one parameter. Some authors penalise break
locations more heavily, e.g. the modified BIC of Zhang & Siegmund (2007).
The constant term for $\sigma^2$ is the same for every $k$, so it is dropped.
The $k$ with the lowest BIC wins. The full BIC table for every series is saved
in `us_series_tests.csv` (`bic_by_breaks`) and discussed in §6.4.

**Why a minimum of 6 months and at most 3 breaks.** Each phase needs enough
months to estimate both a level and a slope in §5.4. Without the minimum
length, the method would isolate single extreme months such as the Truk raid
(February 1944) or Leyte (October 1944) as "phases".

**Not a significance test.** BIC is model selection, not a hypothesis test,
and the breaks are estimated, not pre-specified. The evidence comes from two
places:
- **the size of the BIC gap** between the chosen $k$ and its neighbours (§6.4), and
- **whether the data-found dates match known events** (§10). That is stronger evidence than picking events first and testing for effects, which invites confirmation bias.

### 5.4 Segmented regression (Q2: how fast within each phase?)

**In short.** A separate straight line on the log scale for each phase found
in §5.3, allowed to jump at each break. Each slope converts to "% change per
month" with a 95% confidence interval.

**Design.** For phase $s$ starting at month $a_s$, two columns:

$$\text{level}_{s,t} = \mathbb{1}[t \in s], \qquad \text{slope}_{s,t} = \mathbb{1}[t \in s]\,(t - a_s).$$

There is no global intercept, because the level columns already span it. Each
phase therefore has its own intercept $\alpha_s$ and slope $\beta_s$, and the
model is discontinuous at the breaks. Phases whose total is exactly zero are
dropped from the fit, because their MLE level would be $-\infty$. In practice
no phase was all zeros, so all 45 months were modelled in every series.

#### Ship counts: negative binomial (NB2) regression

$$Y_t \sim \text{NB2}(\mu_t, \alpha), \qquad \log \mu_t = \log e_t + \alpha_{s(t)} + \beta_{s(t)}\,(t - a_{s(t)}),$$

$$\operatorname{E}[Y_t] = \mu_t, \qquad \operatorname{Var}[Y_t] = \mu_t + \alpha\mu_t^2.$$

The log-likelihood that is maximised is

$$\ell(\beta,\alpha) = \sum_t \Big[\log\Gamma(y_t + \alpha^{-1}) - \log\Gamma(\alpha^{-1}) - \log\Gamma(y_t + 1) + \alpha^{-1}\log\frac{1}{1+\alpha\mu_t} + y_t \log\frac{\alpha\mu_t}{1+\alpha\mu_t}\Big].$$

It is well defined for non-integer $y_t$, through the Gamma function, so the
fractional credits are no problem. Strictly, it is then a quasi-likelihood
rather than an exact likelihood.

Why these choices:
- **Why not Poisson?** Poisson forces $\operatorname{Var} = \mu$. Sinkings are clustered (one convoy attack can sink six ships), which gives overdispersion. Poisson standard errors would be far too small. The overdispersion test in §6.1 confirms this for 7 of the 8 count series.
- **Why NB2 rather than NB1** ($\operatorname{Var} = \mu(1+\alpha)$)? NB2 is the standard, and it fits a variance that grows faster than the mean. That suits campaign months, where losses are both higher and more variable.
- **Estimation.** Maximum likelihood with BFGS, started from the Poisson GLM solution (fitted by IRLS) with $\alpha_0 = 0.1$. Statsmodels' default start (zeros and Newton steps) overflowed $e^{x^\top\beta}$ to infinity on the merchant series, giving NaN. IRLS reaches the Poisson optimum stably, and the NB optimum is nearby.
- **Inference.** Wald tests and intervals from the inverse observed information (the Hessian): $\hat\beta_s \pm 1.96\,\widehat{\text{se}}$, reported as $100\,(e^{\hat\beta_s} - 1)$% per month.

#### Tonnage: OLS on the log rate

$$\log(1 + r_t) = \alpha_{s(t)} + \beta_{s(t)}\,(t - a_{s(t)}) + \varepsilon_t, \qquad \varepsilon_t \overset{iid}{\sim} N(0,\sigma^2).$$

- **Why not a GLM?** Tonnage is a continuous, heavily right-skewed amount, not a count. A Gamma or Tweedie GLM was an option, but Gamma can't take the months with zero tonnage, and log-OLS gives a directly interpretable slope.
- **Interpretation.** $100\,(e^{\beta}-1)$% is the change per month in $1 + r_t$. That is very close to the % change in tonnage whenever monthly tonnage is large (thousands of tons).
- **Back-transformation.** The fitted line drawn in the charts is $e^{\hat\mu} - 1$. That is a geometric-mean-type ("typical month") value, which sits below the arithmetic mean. No smearing correction is applied, so the dashed tonnage lines sit a little below the bars.
- **Inference.** Classical OLS t-intervals. These assume normal, homoskedastic, independent errors, which is checked in §6.2–6.3.

#### Hypotheses (for each phase's slope)

- **H₀:** $\beta_s = 0$. Within the phase, losses neither grew nor shrank.
- **H₁:** $\beta_s \neq 0$.

**Caution: post-selection inference.** The phases were chosen by the same data
(§5.3). Conditioning on data-chosen breaks makes the fitted model look better
than it would on new data. So the slope p-values are somewhat optimistic, and
the jumps between phases are not tested here at all, because a test of a jump
at a break chosen by the data is invalid.

### 5.5 Season test (Q3)

**In short.** The segmented model is fitted with and without a yearly wave.
A likelihood-ratio test asks whether the wave improves the fit by more than
chance.

**Model.** Add a first-order harmonic of the calendar month $m_t \in \{1,\dots,12\}$:

$$\log \mu_t = \log e_t + \alpha_{s(t)} + \beta_{s(t)}(t - a_{s(t)}) + \gamma_s \sin\frac{2\pi m_t}{12} + \gamma_c \cos\frac{2\pi m_t}{12}$$

(and the same terms on the right-hand side of the tonnage OLS). Any sinusoid
with a 12-month period and any phase can be written as
$A\cos(2\pi m/12 - \phi)$ with $A = \sqrt{\gamma_s^2+\gamma_c^2}$ and
$\phi = \operatorname{atan2}(\gamma_s, \gamma_c)$. So two parameters capture
any single smooth annual cycle.

**Why one harmonic and not 11 month dummies.** Dummies would use 11 degrees of
freedom out of 45 observations, many of them estimated from only 3–4 values.
That has low power and tends to fit noise. One harmonic is the most powerful
test for a smooth single-peaked cycle. It would miss a cycle with two peaks a
year, which has no known cause in naval warfare.

**Hypotheses**

- **H₀:** $\gamma_s = \gamma_c = 0$ (no annual cycle once the phase trend is allowed for).
- **H₁:** at least one of them is non-zero.

**Statistic.**

$$\Lambda = 2\big(\ell_{\text{trend+season}} - \ell_{\text{trend}}\big) \;\overset{H_0}{\sim}\; \chi^2_2 \quad\text{(asymptotically; Wilks' theorem).}$$

**Derived quantities**

- **Peak month:** $m^* = 12\phi/2\pi$ (mod 12), rounded.
- **Peak-to-trough ratio:** $e^{2A}$. On the log link the peak is $+A$ and the trough $-A$, so $e^{2A}$ is the ratio of expected losses in the peak month to the trough month, with the trend held fixed.

**Caveats**

- $\chi^2_2$ is an approximation with $n = 45$. Plain likelihood-ratio tests in small samples tend to be somewhat liberal.
- A significant harmonic does **not** prove a weather or seasonal *cause*. Any campaign concentrated in one part of one year can produce it (§8).

**The right-hand chart panels.** For each calendar month, the bars show the
mean of $\log\frac{r_t + 0.5}{\hat\mu^{\text{trend}}_t + 0.5}$ (counts), or of
$\log(1+r_t) - \hat\eta^{\text{trend}}_t$ (tonnage), converted to %. Here
$\hat\mu^{\text{trend}}$ is the **trend part of the seasonal model**, i.e. its
prediction with $\sin = \cos = 0$. Using the trend-only model instead would
measure the bars from a slightly different trend than the wave drawn over
them, and the two would disagree. The +0.5 avoids $\log 0$. The line is the
fitted harmonic, $100\,(e^{\hat\gamma_s\sin + \hat\gamma_c\cos} - 1)$.

### 5.6 Ljung-Box test: residual autocorrelation

**In short.** It checks whether the model's errors still follow each other in
time. If they do, the model missed time structure and its p-values are too
optimistic.

**Residuals**

- **NB:** Pearson residuals, $\hat\varepsilon_t = \dfrac{y_t - \hat\mu_t}{\sqrt{\hat\mu_t + \hat\alpha\hat\mu_t^2}}$, which scale each error by its own expected standard deviation.
- **OLS:** raw residuals.

Both come from the trend + season model.

**Hypotheses**

- **H₀:** $\rho_1 = \dots = \rho_h = 0$ (no autocorrelation up to lag $h = 6$).
- **H₁:** some $\rho_k \neq 0$.

**Statistic.**

$$Q = n(n+2)\sum_{k=1}^{h}\frac{\hat\rho_k^2}{n-k} \;\overset{H_0}{\sim}\; \chi^2_h.$$

No degrees of freedom are subtracted for the fitted parameters, because those
are regression coefficients, not ARMA terms. This makes the test approximate.
Here a **small** p-value is the *bad* outcome.

### 5.7 Multiple testing: Benjamini–Hochberg

**The problem.** 16 series × 2 families of tests (trend, season). At α = 0.05,
about 1 in 20 true nulls would be rejected by chance, so with 16 tests a false
alarm is likely.

**The procedure.** It controls the false discovery rate, FDR = E[false
rejections / all rejections], separately within each family (the 16
Mann-Kendall p-values; the 16 season p-values):

1. Sort the p-values: $p_{(1)} \le \dots \le p_{(m)}$, with $m = 16$.
2. Compute $q_{(i)} = \min_{j \ge i}\big(\min(1,\ m\,p_{(j)}/j)\big)$.
3. Call results with $q < 0.05$ discoveries. In expectation, at most 5% of them are false.

**Validity.** BH controls FDR under independence or *positive regression
dependence* (PRDS). The 16 series are positively dependent, because each "all"
series contains its weapon series, which is the PRDS-friendly case.
Benjamini–Yekutieli would be valid under any dependence but is much more
conservative. It isn't needed here.

**FDR rather than familywise error (Bonferroni).** This is exploratory
analysis. Missing real patterns is as costly as a few false leads, and
Bonferroni at $m = 16$ would demand $p < 0.003$.

**Rule used in this document:** p < 0.05 is "suggestive"; **q < 0.05 is a
firm finding**.

### 5.8 Pooled weapon comparison (Q4)

**In short.** For each fleet, all weapon series go into one model. Likelihood-ratio
tests compare "the weapons share a trend shape (or a season)" against "each
weapon has its own".

**Model** (counts; the tonnage version is OLS on $\log(1+r_{w,t})$ with the same right-hand side):

$$\log \mu_{w,t} = \log e_t + \alpha_w + f(t) + f_w(t) + \gamma_s\sin\tfrac{2\pi m_t}{12} + \gamma_c\cos\tfrac{2\pi m_t}{12} + \gamma_{s,w}\sin\tfrac{2\pi m_t}{12} + \gamma_{c,w}\cos\tfrac{2\pi m_t}{12}$$

- $w$ indexes the weapons, with the first weapon as the reference level.
- $f(t)$ is a **natural cubic regression spline** with 4 degrees of freedom: patsy `cr(t, df=4, constraints='center')`, with knots at quantiles of $t$. The centring constraint removes the intercept direction so the model is identifiable alongside $\alpha_w$.
- $f_w(t)$ is each weapon's deviation from the reference trend shape.
- **One shared dispersion $\alpha$** across weapons (counts), or one shared $\sigma^2$ (tonnage). This is an assumption.

Nested models and tests:

| Test | Restricted model (H₀) | Full model (H₁) | df |
|---|---|---|---|
| Trend shape | $f_w \equiv 0$ (parallel trends on the log scale) | $f_w$ free | $(W-1)\times 4 = 8$ |
| Season | $f_w$ free, $\gamma_{s,w} = \gamma_{c,w} = 0$ (shared harmonic) | harmonics free per weapon | $(W-1)\times 2 = 4$ |

Here $W = 3$ weapons per fleet. Each statistic is
$\Lambda = 2(\ell_1 - \ell_0) \sim \chi^2_{df}$ under H₀.

**Why a spline here, not the change points?** Each weapon has its own change
points, but a comparison needs one common, pre-specified time basis. A spline
is smooth, needs no break dates, and nests cleanly.

**Why pooled rather than comparing separate fits?** Separate fits can each
say "rising" while their shapes differ. The pooled interaction test tests the
*difference* directly, with one p-value, and it borrows strength for the
dispersion estimate.

**Convergence.** All 8 pooled NB fits converged (checked from
`mle_retvals`). Fitted $\hat\alpha$ fall from 0.38 (warships) and 0.77
(merchant ships) in the shared-trend models to about 0.20 in the fullest
models. This is expected: structure left out of the mean model shows up as
extra dispersion.

### 5.9 Summary of all tests

| Test | H₀ | H₁ | Statistic and null distribution | Purpose |
|---|---|---|---|---|
| Mann-Kendall (TFPW) | iid, no monotonic trend | monotonic trend | $Z \sim N(0,1)$ | Q1 |
| Wald, phase slope | $\beta_s = 0$ | $\beta_s \ne 0$ | $z$ (NB) or $t_{n-p}$ (OLS) | Q2 |
| LRT, harmonic | $\gamma_s = \gamma_c = 0$ | not both 0 | $\chi^2_2$ | Q3 |
| LRT, weapons' trend | parallel log-trends | weapon-specific spline | $\chi^2_8$ | Q4 |
| LRT, weapons' season | shared harmonic | weapon-specific harmonic | $\chi^2_4$ | Q4 |
| LRT, overdispersion | $\alpha = 0$ (Poisson) | $\alpha > 0$ | $\tfrac12\chi^2_0 + \tfrac12\chi^2_1$ | Model check |
| Jarque–Bera | normal residuals | non-normal | $\chi^2_2$ | Model check (tonnage) |
| Ljung–Box | no autocorrelation to lag 6 | some | $\chi^2_6$ | Model check |

α = 0.05 throughout. BH q-values are given for the Mann-Kendall and harmonic
families.

---

## 6. Model diagnostics and assumptions

These columns are saved in `data/time_series/us_series_tests.csv`.

### 6.1 Is the negative binomial needed? (overdispersion)

Test: H₀: α = 0 (Poisson), against H₁: α > 0. Because α = 0 lies on the
boundary of the parameter space, the likelihood-ratio statistic
$\Lambda = 2(\ell_{NB} - \ell_{Pois})$ has a null distribution that is a
**50:50 mixture** of a point mass at 0 and $\chi^2_1$ (Self & Liang, 1987). So
$p = \tfrac12 P(\chi^2_1 > \Lambda)$.

| Fleet | Series | $\hat\alpha$ | Overdispersion p |
|---|---|---:|---:|
| Warships | All | 0.126 | <0.0001 |
| | Submarines | 0.066 | 0.069 |
| | Aircraft | 0.322 | <0.0001 |
| | Surface | 0.283 | 0.0009 |
| Merchant | All | 0.102 | <0.0001 |
| | Submarines | 0.067 | <0.0001 |
| | Aircraft | 0.400 | <0.0001 |
| | Mines | 0.384 | <0.0001 |

**Reading.** 7 of 8 series are clearly overdispersed, so Poisson standard
errors would be too small. Warship submarines are borderline (p = 0.07,
$\hat\alpha$ = 0.07): their sinkings were more evenly spread. NB is used for
all series for consistency. When α is near 0, NB reduces to Poisson anyway,
so this choice costs nothing.

Interpretation of α:
- With α = 0.4 (aircraft, mines) and a mean of 20 ships, the standard deviation is $\sqrt{20 + 0.4 \times 400} \approx 13$, against $\sqrt{20} \approx 4.5$ under Poisson.
- Aircraft and mine sinkings come in raids and campaigns, so they are the most "bursty".

### 6.2 Tonnage models: fit and normality

| Fleet | Series | R² (trend model) | Jarque–Bera p (residuals) |
|---|---|---:|---:|
| Warships | All | 0.29 | **<0.0001** |
| | Submarines | 0.37 | 0.29 |
| | Aircraft | 0.28 | 0.86 |
| | Surface | 0.55 | 0.31 |
| Merchant | All | 0.74 | 0.64 |
| | Submarines | 0.78 | 0.50 |
| | Aircraft | 0.52 | **<0.0001** |
| | Mines | 0.62 | 0.09 |

**Reading**

- **Merchant tonnage is well described** by the phase model (R² 0.52–0.78).
- **Warship tonnage is not** (R² 0.28–0.37 apart from surface ships). It is dominated by a few battles: Midway, the Philippine Sea, Leyte, the YAMATO sortie.
- **Two series have clearly non-normal residuals** (warships all, merchant aircraft), with heavy tails from those single huge months. Their OLS t-intervals and their season LRT should be read as approximate. A bootstrap or a robust (Huber) regression would be the next check.

### 6.3 Residual autocorrelation (Ljung–Box, lag 6)

| Series flagged (p < 0.05) | p | Borderline (p < 0.06) | p |
|---|---:|---|---:|
| Merchant submarines, ships | 0.017 | Warships all, tonnage | 0.052 |
| Merchant submarines, tonnage | 0.003 | Warships surface, tonnage | 0.057 |
| Merchant aircraft, tonnage | 0.016 | | |

**Reading.** For these series, the phase model leaves some month-to-month
structure unexplained. The likely cause is that submarine sinkings rose and
fell with individual patrol cycles. Their standard errors and p-values are
too small. A remedy would be HAC (Newey–West) standard errors, or adding an
AR(1) term (e.g. an observation-driven count model). The headline results for
these series are large enough to survive this (see §7).

### 6.4 How sure is the number of change points? (BIC table)

$\Delta$BIC = BIC of a rival $k$ minus BIC of the chosen $k$. Following Kass &
Raftery (1995), a $\Delta$BIC of 0–2 is "barely worth mentioning", 2–6
"positive", 6–10 "strong" and above 10 "very strong" evidence.

| Fleet | Series | Measure | Chosen k | BIC (k = 0, 1, 2, 3) | Closest rival (ΔBIC) | Evidence |
|---|---|---|---:|---|---|---|
| Warships | All | ships | 1 | 47.1, **4.6**, 6.0, 9.0 | k = 2 (+1.4) | weak vs 2; very strong vs 0 |
| | Submarines | ships | 1 | 18.1, **−19.1**, −17.0, −12.4 | k = 2 (+2.1) | positive |
| | Aircraft | ships | 2 | 39.6, 15.1, **13.6**, 19.5 | k = 1 (+1.5) | weak vs 1 |
| | Surface | ships | 2 | 2.6, −4.6, **−7.2**, −5.7 | k = 3 (+1.5) | weak |
| | All | tonnage | 1 | 63.7, **59.7**, 61.3, 67.1 | k = 2 (+1.6) | weak; positive vs 0 (+4.0) |
| | Submarines | tonnage | 1 | 122.3, **110.2**, 113.1, 120.1 | k = 2 (+2.9) | positive |
| | Aircraft | tonnage | 1 | 125.0, **120.4**, 122.2, 126.6 | k = 2 (+1.8) | weak; positive vs 0 (+4.6) |
| | Surface | tonnage | 2 | 122.3, 110.4, **107.2**, 111.1 | k = 1 (+3.2) | positive |
| Merchant | All | ships | 2 | 89.8, 48.9, **46.8**, 48.7 | k = 3 (+1.9) | weak vs 1/3; very strong vs 0 |
| | Submarines | ships | 3 | 45.8, 30.7, 19.3, **12.8** | k = 2 (+6.5) | strong |
| | Aircraft | ships | 2 | 78.5, 47.9, **47.8**, 49.4 | k = 1 (+0.1) | **none: k = 1 and k = 2 tie** |
| | Mines | ships | 1 | 71.0, **15.2**, 16.9, 23.1 | k = 2 (+1.7) | weak vs 2; very strong vs 0 |
| | All | tonnage | 2 | −17.6, −47.8, **−61.5**, −60.2 | k = 3 (+1.3) | weak vs 3; very strong vs 1 |
| | Submarines | tonnage | 3 | −15.9, −30.5, −37.7, **−49.4** | k = 2 (+11.7) | very strong |
| | Aircraft | tonnage | 2 | 87.8, 70.6, **70.5**, 75.6 | k = 1 (+0.1) | **none: tie** |
| | Mines | tonnage | 2 | 142.7, 124.5, **120.2**, 123.3 | k = 1 (+4.3) | positive |

**Reading**

- **"There is at least one change point" is supported in every series**: ΔBIC against k = 0 is above 10 ("very strong") in 13 of 16 series, and 4.0–9.8 in the other three (warship surface ships, all-warship tonnage, warship aircraft tonnage). So the step-change picture is robust.
- **The exact *number* of breaks is often uncertain** (ΔBIC < 2 for the nearest rival in 10 of 16 series). Merchant aircraft, in particular, is a tie between one break (Oct 1943) and two (Oct 1943 + Sep 1944).
- **The best-supported breaks are those that persist whatever k is chosen.** Examples: Nov 1943 in the merchant series, Jan/Feb 1944 for warships, Mar 1945 for mines, Dec 1944 for merchant submarines (ΔBIC 6.5 and 11.7).

### 6.5 Information criteria for the season (AIC)

AIC = 2p − 2ℓ. A lower value means a better expected out-of-sample fit.
Adding the harmonic costs 2 parameters, i.e. 4 AIC points, so it must raise
the log-likelihood by more than 2 to lower AIC.

| Series where the harmonic lowers AIC | AIC trend → trend + season |
|---|---|
| Warships all, ships | 277.1 → **270.5** (−6.6) |
| Merchant aircraft, ships | 304.7 → **298.7** (−6.0) |
| Warships aircraft, tonnage | 242.4 → 240.6 (−1.8) |
| Merchant aircraft, tonnage | 190.4 → 189.1 (−1.3) |
| Warships surface, tonnage | 222.2 → 222.0 (−0.2) |

In the other 11 series the harmonic raises AIC. This agrees with the LRT: only
the first two series have a meaningful improvement, and neither survives the
FDR correction (§8).

### 6.6 Assumptions checklist

| Assumption | Where | How checked | Result |
|---|---|---|---|
| Counts overdispersed relative to Poisson | NB models | Boundary LRT for α (§6.1) | Yes in 7/8; NB justified |
| Correct mean structure (phases + harmonic) | All regressions | R², AIC, Ljung–Box | Good for merchant; warship tonnage dominated by single battles |
| Independent errors | All tests | Ljung–Box (§6.3); TFPW for MK | 3 series flagged; their p-values are optimistic |
| Normal errors | Tonnage OLS | Jarque–Bera (§6.2) | 2 series fail (heavy tails) |
| Constant noise level for change points | Dynp + L2 cost | √ / log transform (§4.2) | Approximately |
| Large-sample χ² for the LRTs | Season and weapon tests | Not checked (n = 45) | Treat p-values near 0.05 with caution |
| Positive dependence for BH | FDR correction | By construction (nested series) | Holds |

---

## 7. Results: long-term trend

### 7.1 Q1: losses rose over the war (Mann-Kendall, TFPW)

| Fleet | Series | Measure | Trend | p | q | Sen's slope (per month) |
|---|---|---|---|---:|---:|---:|
| Warships | All | ships | increasing | <0.0001 | <0.0001 | +0.60 ships |
| | Submarines | ships | increasing | <0.0001 | <0.0001 | +0.21 ships |
| | Aircraft | ships | increasing | <0.0001 | <0.0001 | +0.25 ships |
| | Surface | ships | **no trend** | 0.080 | 0.086 | +0.05 ships |
| | All | tonnage | increasing | 0.020 | 0.024 | +600 t |
| | Submarines | tonnage | increasing | 0.035 | 0.040 | +204 t |
| | Aircraft | tonnage | increasing | 0.0007 | 0.001 | +188 t |
| | Surface | tonnage | **no trend** | 0.103 | 0.103 | +53 t |
| Merchant | All | ships | increasing | <0.0001 | <0.0001 | +2.05 ships |
| | Submarines | ships | increasing | 0.0009 | 0.001 | +0.65 ships |
| | Aircraft | ships | increasing | <0.0001 | <0.0001 | +0.69 ships |
| | Mines | ships | increasing | <0.0001 | <0.0001 | +0.06 ships |
| | All | tonnage | increasing | <0.0001 | <0.0001 | +5,981 GRT |
| | Submarines | tonnage | increasing | 0.008 | 0.011 | +2,378 GRT |
| | Aircraft | tonnage | increasing | 0.0001 | 0.0002 | +1,713 GRT |
| | Mines | tonnage | increasing | <0.0001 | <0.0001 | +166 GRT |

**Observations**

- **14 of 16 series rose over the war**, all at q < 0.05.
- **Surface warships are the exception:** their losses rose in 1942–43, held steady, then collapsed in 1945, so there is no monotonic direction.
- **The mine Sen's slope is tiny** (+0.06 ships a month) even though mines sank about 40 ships a month in mid-1945. Sen's slope is a median over all 990 pairs of months, and 39 of the 45 months are near zero. This illustrates why a global monotonic-trend summary misleads for step-shaped series.

### 7.2 Q2: the rise came in jumps, not gradually

Average rate per phase (ships per average month) and within-phase growth
(NB2 slope, 95% Wald CI):

**Warships**

| Series | Phases (start of each) | Rate per phase | Within-phase growth |
|---|---|---|---|
| All | Dec 1941 → **Jan 1944** | 5.4 → **24.7** | +4.2%/mo (1.0, 7.5) \*; then +1.7%/mo (−1.7, 5.1) |
| Submarines | Dec 1941 → **Feb 1944** | 1.6 → **8.7** | +1.9%/mo (−2.3, 6.4); +1.0%/mo (−2.8, 4.8) |
| Aircraft | Dec 1941 → **Jul 1943** → **Sep 1944** | 1.3 → 5.3 → **14.9** | −3.3%/mo; +6.0%/mo; −5.1%/mo (none significant) |
| Surface | Dec 1941 → **Oct 1942** → **Mar 1945** | 0.5 → 3.9 → **1.1** | last phase −50.5%/mo (−78.4, +13.7) |

**Merchant ships**

| Series | Phases (start of each) | Rate per phase | Within-phase growth |
|---|---|---|---|
| All | Dec 1941 → **Oct 1942** → **Nov 1943** | 11.8 → 27.3 → **76.9** | +8.6%/mo (−1.6, 19.8); +0.7%/mo (−4.5, 6.1); +2.2%/mo (0.0, 4.5) \* (p = 0.049) |
| Submarines | Dec 1941 → **Aug 1942** → **Nov 1943** → **Dec 1944** | 8.4 → 20.0 → **44.6** → 20.4 | none significant |
| Aircraft | Dec 1941 → **Oct 1943** → **Sep 1944** | 3.7 → 19.1 → **38.7** | +7.9%/mo (1.0, 15.2) \*; −6.8%/mo; −2.4%/mo |
| Mines | Dec 1941 → **Mar 1945** | 0.7 → **40.2** | +11.4%/mo (5.4, 17.7) \*; +45.8%/mo (−2.9, 118.8) |

\* = significant at 0.05. Remember the post-selection caveat (§5.4).

**Tonnage adds two results to the ship counts**

- **Warship tonnage breaks in June 1942** (from about 9,900 to about 46,000 t a month), not in 1944 like the ship count. That is Midway: four fleet carriers and a heavy cruiser in one month.
- **Merchant tonnage sunk by submarines declined steadily from December 1944:** −12.5%/mo (95% CI −21.6 to −2.3). This is one of the few significant within-phase slopes. That series is flagged by Ljung–Box (§6.3), so the true interval is somewhat wider, but its upper end, −2.3%, still leaves room for that.

**Main observation: growth happened at the breaks, not within the phases.**
Of 22 phase slopes for ship counts, only 4 are significant at 0.05:
- three in early, low-level phases (all warships 1941–43, merchant aircraft 1941–43, mines before March 1945);
- the final merchant phase (+2.2%/mo), which only just passes (p = 0.049).

With post-selection optimism and 22 tests, about 1 of these is expected by
chance. Each break, by contrast, multiplies the rate by about 2–5. The rises at the breaks are only *described*, not *tested*, because
of post-selection (§5.4). But their size dwarfs any within-phase drift. Their
support for "a change happened" comes from the BIC gaps of §6.4.

### 7.3 Ships got smaller as the war went on (descriptive)

Average size of a ship sunk by the US (total tonnage ÷ ships, by period):

| Period | Merchant: GRT per ship | Warships: tons per ship |
|---|---:|---:|
| Dec 1941 – Oct 1943 | 4,393 | 3,627 |
| Nov 1943 – Dec 1944 | 4,209 | 3,113 |
| Jan 1945 – Aug 1945 | **2,574** | **2,309** |

This is a ratio of sums, not a tested result. A formal version would model
log-tonnage per ship at the ship level, e.g. a regression on period with
standard errors clustered by month. It is consistent with the merchant
"ships rising, tonnage flat" pattern in the last phase.

---

## 8. Results: seasonality

| Fleet | Series | Measure | LR ($\chi^2_2$) | p | q | Peak month | Peak/trough $e^{2A}$ |
|---|---|---|---:|---:|---:|---|---:|
| Warships | All | ships | 10.57 | **0.005** | 0.053 | Oct | 1.9 |
| | Submarines | ships | 1.74 | 0.42 | 0.67 | | |
| | Aircraft | ships | 1.07 | 0.59 | 0.77 | | |
| | Surface | ships | 2.53 | 0.28 | 0.57 | | |
| | All | tonnage | 2.29 | 0.32 | 0.57 | | |
| | Submarines | tonnage | 0.94 | 0.62 | 0.77 | | |
| | Aircraft | tonnage | 5.74 | 0.057 | 0.29 | (Oct) | (29, unstable) |
| | Surface | tonnage | 4.19 | 0.12 | 0.39 | | |
| Merchant | All | ships | 2.42 | 0.30 | 0.57 | | |
| | Submarines | ships | 0.16 | 0.92 | 0.96 | | |
| | Aircraft | ships | 10.04 | **0.007** | 0.053 | Dec | 3.6 |
| | Mines | ships | 0.98 | 0.61 | 0.77 | | |
| | All | tonnage | 2.71 | 0.26 | 0.57 | | |
| | Submarines | tonnage | 0.08 | 0.96 | 0.96 | | |
| | Aircraft | tonnage | 5.28 | 0.071 | 0.29 | (Dec) | (6.3) |
| | Mines | tonnage | 0.80 | 0.67 | 0.77 | | |

**Observations**

- **No series has a seasonal pattern at q < 0.05.**
- **Two come close:** all warships (peak October, p = 0.005, q = 0.053) and aircraft against merchant ships (peak December, p = 0.007, q = 0.053). Both are also the only series where the harmonic lowers AIC by more than 2 (§6.5).
- **Both near-misses match single campaigns rather than an annual cycle:**
  - **October:** Leyte Gulf (October 1944). With 46 US-credited warships and 333,377 t, it was the second-worst month of the war by number and the worst by tonnage.
  - **December–January:** US carrier strikes on shipping off the Philippines and China, December 1944 – January 1945.
- **Identifiability.** With 45 months, each calendar month appears 3–4 times. A harmonic that peaks in October can be fitted to a single October spike. Within this design the "season" is **confounded** with the timing of a few campaigns, and no statistical method can separate them using this series alone. Separating them would need either more years or covariates that cause a season (sea state, typhoon tracks, daylight).
- **Ignore the very large peak-to-trough ratios** for aircraft tonnage (29 and 6.3). They are driven by a handful of extreme months and by non-normal residuals (§6.2).

**Conclusion for Q3:** there is no convincing evidence of a seasonal cycle in
US sinkings. Under H₀, 16 tests would give on average 0.8 results with
p < 0.05 by chance, and we observe 2. That's slightly more than chance, and
both are explained by campaign timing.

---

## 9. Results: do the weapons differ?

| Fleet | Measure | Test | $\Lambda$ | df | p |
|---|---|---|---:|---:|---:|
| Warships | ships | trend shape differs by weapon | 30.3 | 8 | **0.0002** |
| | ships | season differs by weapon | 6.4 | 4 | 0.17 |
| | tonnage | trend shape differs by weapon | 21.0 | 8 | **0.007** |
| | tonnage | season differs by weapon | 7.8 | 4 | 0.10 |
| Merchant | ships | trend shape differs by weapon | 94.6 | 8 | **<0.0001** |
| | ships | season differs by weapon | 26.4 | 4 | **<0.0001** |
| | tonnage | trend shape differs by weapon | 49.8 | 8 | **<0.0001** |
| | tonnage | season differs by weapon | 5.4 | 4 | 0.25 |

(For the counts, the log-likelihoods are: warships −304.8 → −289.7
(trend), −287.5 → −284.4 (season); merchant ships −423.3 → −376.0 (trend),
−374.6 → −361.5 (season).)

**Observations**

- **The weapons followed different trend shapes**, in both fleets and both measures. The trend-shape LRTs are far beyond any multiple-testing threshold.
  - **Merchant ships:** submarines dominated until late 1944, aircraft took over in late 1944, and mines became major from spring 1945 (see `us_weapon_shares.png`).
  - **Warships:** surface ships and aircraft dominated in 1942–43 (the Solomons), and aircraft and submarines from 1944.
- **For merchant ship counts, "season" also differs by weapon** (p < 0.0001). By §8's argument, this is campaign timing: each weapon's peak campaign fell in a different part of the calendar (aircraft in Dec–Jan, mines in May–Jul 1945). It is not evidence of weapon-specific weather effects. The same test on merchant tonnage is not significant.
- **Caveat:** a 4-df spline cannot follow the mine series' near-vertical jump in March 1945. Part of the "season differs" signal may be the spline failing to fit that jump, picked up by the harmonics instead.

---

## 10. Interpretation and hypotheses

The statistics say **when** the loss rate changed. They do **not** say
**why**. The table matches each data-found change point with well-known
events of the war. These are **hypotheses**: plausible explanations to test
further, not conclusions proven by this data.

| Change point (found by the data) | Series | Possible explanation (historical hypothesis) |
|---|---|---|
| Jun 1942 | Warship tonnage | Battle of Midway: four Japanese fleet carriers sunk in one battle |
| Aug–Oct 1942 | Merchant (all, submarines); warships (surface) | Guadalcanal campaign: months of surface fighting around the Solomons; more US submarines on patrol |
| Jul 1943 | Warships (aircraft) | Air campaigns over the Solomons and New Guinea |
| Sep–Nov 1943 | Merchant (all, submarines, aircraft) | US submarine torpedo defects largely fixed by late 1943; more submarines; better use of decoded Japanese convoy messages |
| Jan–Feb 1944 | Warships (all, submarines) | Central Pacific offensive: the Marshall Islands and the carrier raid on Truk (Feb 1944) |
| Sep 1944 | Aircraft (both fleets) | US carrier strikes on the Philippines and Formosa, then Leyte Gulf (Oct 1944) |
| Dec 1944 | Merchant submarines: level and tonnage fall | Few targets left on the open sea; Japanese shipping pulled back to coastal and inland waters |
| Mar 1945 | Merchant mines; warship surface losses collapse | Operation Starvation: B-29 aerial mining of Japan's ports and straits from the end of March 1945; few Japanese warships left at sea |

### Hypotheses and how they could be tested

| Hypothesis | Status from this analysis | How to test it properly |
|---|---|---|
| **H-A, step changes:** the loss rate rose mainly through a few discrete changes, not steady build-up | Supported: at least one break preferred by BIC in every series (very strongly in 13 of 16); 18 of 22 count-phase slopes non-significant | Bayesian change-point model (e.g. Poisson/NB with an unknown number of breaks) for posterior break-date intervals |
| **H-B, weapon substitution:** as Japanese ships left the open ocean, submarine sinkings fell while aircraft and mine sinkings rose | Supported (§9 and the share chart) | Cross-correlation of pre-whitened weapon series; compositional (multinomial) regression of weapon shares on time |
| **H-C, running out of targets:** the 1945 fall in submarine sinkings and in ship size reflects a depleted fleet, not less effective submarines | Not testable with JANAC alone | Add monthly Japanese fleet tonnage afloat (exposure) and model the loss *rate*; ship-level regression of size on period |
| **H-D, no weather season:** US sinkings had no annual cycle | Weakly supported: no harmonic survives FDR; the 2 near-misses are explained by campaigns | Needs more years, or covariates (sea state, typhoons) in place of a harmonic |

---

## 11. Limitations, alternatives considered and robustness checks

### 11.1 Limitations

1. **Raw losses, not loss rates.** Losses depend on how many Japanese ships
   were afloat, which JANAC does not give. The proper exposure for a "risk"
   model would be ship-months at sea. This is the most important missing
   piece.
2. **A short series** (n = 45). Asymptotic χ² approximations, BIC and the
   harmonic all rest on large-sample theory. Seasonal effects are not
   identifiable separately from campaign timing.
3. **Post-selection inference.** Phases are chosen and then fitted on the
   same data, so phase-slope p-values are optimistic and jump sizes are not
   tested.
4. **Residual autocorrelation** in 3 series (and 2 borderline), so their
   standard errors are too small.
5. **Non-normal residuals** in 2 tonnage series, so their t-intervals and
   season LRTs are approximate.
6. **Measurement.** Credits are JANAC's 1947 assessments, some marked
   "probable". The data was read by OCR, and totals match the report almost
   exactly (§2.7).
7. **The fractional-credit choice.** 1/n splitting is one convention. Full
   credit to every agent would raise the totals slightly. Shared sinkings are
   rare, so the patterns would not change.

### 11.2 Alternatives considered (and why not used)

| Alternative | Why not used here |
|---|---|
| ARIMA / SARIMA | Assumes Gaussian, (difference-)stationary series. A seasonal AR or MA term needs many more than 4 cycles. Counts near zero (early mines) break the Gaussian assumption |
| Facebook Prophet | Gaussian likelihood; its automatic change points are tuned for long business series and overfit at n = 45; yearly Fourier terms have the same identifiability problem |
| Pettitt / CUSUM tests | Test for a single change point only. Dynp + BIC handles several |
| Binary segmentation / PELT | Approximate or penalty-tuned. The exact Dynp is cheap at n = 45 |
| Poisson regression | Overdispersion rejected in 7/8 series (§6.1) |
| Gamma / Tweedie GLM for tonnage | Gamma can't take zero months. Tweedie was possible but less interpretable; log-OLS was preferred, with its checks reported |
| INAR / observation-driven count models (e.g. `tscount`-style) | Would model the autocorrelation directly. A good next step for the series flagged by Ljung–Box |
| Bayesian structural time series | Gives full uncertainty for trend changes, but is heavy for an exploratory first pass |

### 11.3 Suggested robustness checks

1. **Minimum segment length** 4 and 9 months instead of 6: do the main break dates move?
2. **BIC penalty:** use the modified BIC (Zhang & Siegmund) or AIC: do the strongly supported breaks survive?
3. **Full credit for shared sinkings** instead of 1/n.
4. **Probable credits:** drop sinkings marked "probable" (`agent_probable`) and re-run.
5. **HAC (Newey–West) standard errors,** or an AR(1) term, for the 3 flagged series.
6. **Bootstrap** (moving-block bootstrap, block length about 3–6 months) for the phase slopes and the season LRT, avoiding the χ² approximation.
7. **Weekly series** (about 190 weeks; exact dates exist for nearly all ships) for sharper break dates. The number of yearly cycles is unchanged, so the season question can't be helped this way.

---

## 12. Files and how to reproduce

Run:

```powershell
python us_loss_time_series.py
```

| File | Contents |
|---|---|
| `data/time_series/us_monthly_losses.csv` | Monthly ships and tonnage for each fleet and series, with `active_days` (war days) for computing rates |
| `data/time_series/us_series_tests.csv` | One row per series: Mann-Kendall (trend, p, q, Sen's slope), change points, `bic_by_breaks`, season LR/p/q/peak/ratio, Ljung–Box p, `nb_alpha`, `overdispersion_p`, `r_squared`, `jarque_bera_p`, `aic_trend`, `aic_season`, `months_modelled`, `model_converged` |
| `data/time_series/us_trend_segments.csv` | One row per phase: dates, total, rate per average month, % change per month with 95% CI and p-value |
| `data/time_series/us_weapon_comparison.csv` | The pooled LR tests of §9 |
| `figures/time_series/us_{naval,merchant}_{ships,tonnage}.png` | Rate bars, STL trend, segmented fit, change points; seasonal profile with the fitted harmonic |
| `figures/time_series/us_weapon_shares.png` | Each weapon's share of US sinkings over time (centred 3-month window) |

**Software:**
- statsmodels: `STL`, `NegativeBinomial`, `GLM`, `OLS`, formula `negativebinomial` with patsy `cr()`, `acorr_ljungbox`, `multipletests`
- pymannkendall: `trend_free_pre_whitening_modification_test`
- ruptures: `Dynp`
- scipy: `chi2`, `jarque_bera`
- matplotlib: the charts

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **AIC** | Akaike information criterion, 2p − 2ℓ; estimates out-of-sample fit. Lower is better |
| **Autocorrelation** | Correlation of a series with its own past values |
| **BIC** | Bayesian information criterion, p log n − 2ℓ; penalises parameters more than AIC does. Lower is better |
| **Boundary LRT** | A likelihood-ratio test where the null value sits on the edge of the parameter space (e.g. α = 0); its null distribution is a mixture of chi-squares |
| **Change point** | A time at which the distribution (here the mean level) of a series shifts |
| **Exposure / offset** | The time at risk over which events are counted; enters a log-link model as log(e) with its coefficient fixed at 1 |
| **FDR** | False discovery rate: the expected share of false positives among rejected nulls |
| **GRT** | Gross register tons: a merchant ship's internal volume (100 ft³ = 1 GRT) |
| **Harmonic** | A sine/cosine pair at a given frequency; here one cycle per 12 months |
| **IRLS** | Iteratively reweighted least squares, the standard algorithm for fitting GLMs |
| **Jarque–Bera** | A test of normality based on sample skewness and kurtosis |
| **Ljung–Box** | A portmanteau test for autocorrelation up to lag h |
| **LRT** | Likelihood-ratio test: 2(ℓ_full − ℓ_restricted), compared with χ² |
| **NB2** | Negative binomial with Var = μ + αμ² |
| **Overdispersion** | Variance larger than the mean (more than Poisson allows) |
| **Pearson residual** | (y − μ̂) / √Var̂(y) |
| **Post-selection inference** | Inference after choosing the model from the same data; naive p-values are too small |
| **PRDS** | Positive regression dependence on a subset; a condition under which BH controls FDR |
| **q-value** | A BH-adjusted p-value |
| **Sen's slope** | The median of all pairwise slopes (Theil–Sen estimator) |
| **Spline (natural cubic regression)** | A piecewise cubic curve, linear beyond the outer knots; flexible trend with few parameters |
| **Standard displacement** | A warship's weight, fully equipped but without fuel and reserve feed water |
| **STL** | Seasonal-Trend decomposition using LOESS |
| **TFPW** | Trend-free pre-whitening: removes AR(1) autocorrelation before Mann-Kendall while keeping the trend |
| **Wald interval** | Estimate ± 1.96 × standard error |
| **Wilks' theorem** | Under regularity conditions, LRT statistics are asymptotically χ² with df = the number of restrictions |

---

## 14. References

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57, 289–300.
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29, 1165–1188.
- Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I. (1990). STL: a seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6, 3–73.
- Hilbe, J. M. (2011). *Negative Binomial Regression* (2nd ed.). Cambridge University Press.
- Kass, R. E. & Raftery, A. E. (1995). Bayes factors. *JASA*, 90, 773–795.
- Truong, C., Oudre, L. & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299 (the `ruptures` paper).
- Ljung, G. M. & Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika*, 65, 297–303.
- Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *JASA*, 82, 605–610.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *JASA*, 63, 1379–1389.
- Yue, S., Pilon, P., Phinney, B. & Cavadias, G. (2002). The influence of autocorrelation on the ability to detect trend in hydrological series. *Hydrological Processes*, 16, 1807–1829.
- Zhang, N. R. & Siegmund, D. O. (2007). A modified Bayes information criterion with applications to the analysis of comparative genomic hybridization data. *Biometrics*, 63, 22–32.
