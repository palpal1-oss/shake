Earthquake Energy Field Predictor (Depth-Aware Global Model)

This project contains two complementary seismic-analysis systems:

shake.py — surface-level stress & hotspot predictor (0–30 days ahead)

deep-shake.py — deep-structure energy inversion model for identifying long-term high-magnitude candidates and possible super-events.

Neither program performs deterministic “earthquake prediction”.
They calculate stress migration, energy accumulation, and likely activation zones using physics-based kernels and inversion.

⚠ Important Notes About Predictive Uncertainty

The model produces uncontrolled timing uncertainty of 0–72 hours for near-surface activation.

This means:

If the map shows a hotspot for “tomorrow”, the actual event may occur
any time between now and +72 hours.

This applies only to surface-level predictions in shake.py.
deep-shake.py does not give exact times — only long-term structural candidates.

What Each Program Does
✔ shake.py — Surface / Crustal Earthquake Stress Model

Produces a depth-aware global stress map with:

direct kernel energy

inversion reconstruction

b-value plasticity estimation

day-ahead time-decay evolution (0–30 days)

energy halos (radius ∝ log(E))

3D interactive globe visualization

toggle layers (USGS, direct, inversion)

hotspot emergence tracking

This script is ideal for:

short-term activation analysis

crustal stress estimation

brittle-zone identification

visualizing earthquake clustering

studying stress propagation over days to weeks

✔ deep-shake.py — Deep Earth Structure & Super-Event Candidate Model

This script implements:

long-window energy ingestion (60–180+ days)

depth-filtered stress field reconstruction

Tikhonov inversion tuned for deep structure

clustering of high-energy deep cells

candidate identification for possible M8+ / M9+ long-term events

each candidate marked with a time window, not an exact date

possible multiple exit-points of the same deep source (same color group)

This system is not for short-term prediction.
It identifies:

extremely slow energy accumulation

mantle-level stress anomalies

deep drivers that may eventually produce megaquakes

candidate regions decades ahead

It is intentionally conservative and may show 0 or very few candidates.

⚠ Important Requirements for Correct Operation
✔ Use 60+ days of historical data

Accuracy becomes meaningful only when using at least 60–120 days of past earthquakes.

Short windows (1–7 days) show only surface noise.

Long windows reveal:

deeper accumulation patterns

multi-week energy transfer

slow migration processes

hidden fault loading

✔ Do not analyze earthquakes deeper than 100 km

Deep events (>100 km):

do not effectively transfer stress to the crust

cause false hotspot formation

produce unrealistic inversion artifacts

“light up” mantle features that never reach the surface

Recommended depth filter:
depth ≤ 100 km

✔ Grid resolution impacts performance
Grid	Resolution	Performance	Use case
Low	40×20	very fast, coarse	rough testing
Medium	60×40	best balance	recommended
High	100×80+	extremely slow	research only

Going beyond 120×90 may freeze weaker machines.

Features (Both Systems)

reliable day-by-day USGS fetch (avoid timeouts)

depth-aware energy attenuation

direct kernel stress map

inversion-based field reconstruction

b-value plasticity (Aki-Utsu)

3D Plotly interactive globe

energy halo visualization

day-ahead slider (0–30 days)

layer toggle controls

optional hidden USGS layer

supports offline cached data
