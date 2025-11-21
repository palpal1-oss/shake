Earthquake Energy Field Predictor (Depth-Aware Global Model)

This project implements a depth-aware global stress-energy mapping system based on recent seismic activity.
It does not perform classical “earthquake prediction”.
Instead, it identifies:

energy concentration zones,

stress-accumulation clusters,

low-plasticity brittle areas,

possible future activation points,

day-ahead energy decay evolution (0–30 days).

The model visualizes results on a fully interactive 3D globe (Plotly) with multiple prediction layers (direct & inversion), energy halos, and time-decay slider.

⚠ Important Notes for Correct Operation
1. Use at least 60+ days of historical data

The model significantly improves accuracy when the input dataset covers 60 or more days before the date of analysis.
Shorter windows (1–7 days) show only superficial stress, while long windows reveal:

deeper structural patterns,

energy migration pathways,

long-term loading zones.

Recommended: 60–120 days.

2. Do not analyze depth below 100 km

Events deeper than ~100 km transfer very little energy to the crust.
Including them causes:

false hotspot formation,

appearance of deep-mantle signals that never reach the surface,

unrealistic inversion spikes.

Recommended:
Use only earthquakes with depth ≤ 100 km for constructive surface-level stress mapping.

3. Grid resolution strongly affects performance

Parameters:

NX – longitude resolution
NY – latitude resolution


Low grid (40×20):
● fast
● coarse, suitable for testing

Medium grid (60×40):
✔ recommended balance
✔ fast enough
✔ good detail level

High grid (100×80 or higher):
● extremely detailed
● very high CPU & RAM load
● kernel/inversion matrices become huge

Warning:
Increasing grid size beyond 120×90 may freeze weak computers.

🔧 Features

Robust multi-day USGS fetch (day-by-day, avoids timeouts)

Depth-aware attenuation model

Direct kernel energy mapping

Tikhonov inversion reconstruction ((AᵀA + λI)⁻¹Aᵀy)

Plasticity estimation via b-value (Aki-Utsu)

Interactive 3D globe:

direct prediction points

inversion prediction points

USGS real events

energy halos (radius ∝ log(E))

day-ahead slider (0–30 days)

toggle buttons for visibility
