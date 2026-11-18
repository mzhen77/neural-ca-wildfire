# Vegetation treatment planning: inputs, saved plans and scripts

This folder accompanies the manuscript

> I. Matei, M. Zhenirovskyy, H. Y. Wong. *Model-based vegetation treatment planning with
> cell-level wildfire protection constraints.* Submitted to Environmental Modelling & Software.

It holds what the study adds to this repository for the four fires of the paper (Pier 2017, Brattain 2020, Chimney 2016, Ferguson 2018): the management polygons, treatment programs and costs, the named assets, the saved plans, and three scripts. The scripts regenerate the fire-model inputs with the fine-tuned checkpoints of this branch and rebuild the protected cells and their limits. The planner code is not included. It will be shared upon request for review purposes or when the paper is accepted.

**Contact:** Ion Matei, imatei@fujitsu.com.

## Quick start

From the repository root, after `uv sync`:

```bash
python data/data_extractor.py                                            # unpack the event data
python vegetation_planning/scripts/prepare_fire_model.py --forecasts     # parameter maps and forecasts
pip install -r vegetation_planning/scripts/requirements-regions.txt
python vegetation_planning/scripts/collect_electrical_network.py --map-api   # download electrical layers
python vegetation_planning/scripts/build_protected_regions.py                # protected cells and limits
```

All outputs go to `vegetation_planning/generated/<fire>/`, which git ignores.

| Output | Content |
|---|---|
| `fire_model.npz` | daily CNN parameter maps, wind, terrain, burnable mask, initial state |
| `forecasts.npz` | daily unburned probability without treatment and under the reference |
| `band_150m/constraints.npz` | protected regions, asset cores, protected cells and their limits |
| `band_150m/assets.json` | identifier, name, kind and region index of every asset |
| `band_150m/comparison.json` | agreement of the rebuilt regions and limits with those of the paper |

**Hardware.** The study ran on an NVIDIA RTX 4090 GPU. On that hardware the script reproduces the study's parameter maps and both forecasts of all four fires exactly, in about four minutes. It runs each fire in its own process, because a process that has already handled another grid can evaluate the CNN with different float32 rounding. Other hardware can give slightly different maps, and the forecast amplifies the difference. The saved plans meet the cell limits under the study's forecasts. The region script reads the limits from your forecasts, so on other hardware they can differ from the study's limits. The protected regions do not depend on the hardware, because the exposure test reads the study's forecast from
`inputs/<fire>/exposure.npz`.

## Layout

```
inputs/<fire>/                 polygons, programs, costs, named assets, georeferencing
results/<fire>/two_phase/      reference, Phase 1 and Phase 2 plans, reference table of the protected cells, timings
results/<fire>/dpv_ordered_removal/   final plan of DPV-ordered removal
results/<fire>/dpv_guidance_comparison/   the same search guided by the Rothermel model
results/<fire>/benchmarks/     final plans of the three literature benchmarks
results/summary/               cost and time summaries behind the paper's tables, check values of every plan
sensitivity/                   fixed-plan sensitivity of three plans on the four fires
scripts/                       fire-model inputs, electrical download, protected cells and limits
configs/                       FBFM40 fuel loads
docs/                          reproducibility record and the algorithm-settings table
manifest.json                  original path and hash of every file
```

| Fire | Grid (rows × columns) | Forecast days | Polygons | Assets | Regions | Protected cells |
|---|---|---|---:|---:|---:|---:|
| Pier_2017 | 846 × 873 | 2017-08-29 to 2017-09-12 | 2,235 | 308 | 304 | 91,247 |
| Brattain_2020 | 1085 × 1008 | 2020-09-07 to 2020-09-21 | 3,289 | 22 | 22 | 67,000 |
| Chimney_2016 | 890 × 986 | 2016-08-13 to 2016-08-27 | 1,733 | 253 | 152 | 108,562 |
| Ferguson_2018 | 1316 × 1155 | 2018-07-13 to 2018-07-27 | 3,267 | 295 | 194 | 209,651 |

A region is an asset with its 150 m band, and assets with identical regions share one. A protected cell is a cell inside at least one region. Every protected cell has its own constraint.

## Conventions

- All rasters share the fire's full data grid. Cells are 30 m × 30 m, area 0.09 ha. Arrays
  are indexed `[row, column]`. `inputs/<fire>/georeferencing.json` gives the bounds in EPSG:3310.
- A forecast covers 15 days with 50 microsteps per day and mean burning duration `tau = 10`
  microsteps. Daily histories have 16 entries: the initial state and the end of each day.
- Forecast arrays store the **unburned** probability `u`. The affected probability of the
  paper is `c = 1 - u`.
- Polygon identifiers run from 1 to `I`. Identifier 0 marks cells outside every polygon.
  Per-polygon arrays have length `I + 1`, and entry 0 is unused.
- Programs: `0` no treatment, `1` prescribed burning, `2` mechanical treatment and burning.
- Cell operation codes (`labels`, `operation_maps`): `0` untreated, `1` forest burn history,
  `2` forest combined history, `3` grass burn, `4` shrub burn.
- Costs are gross 2025 US dollars.

## Fire model

`scripts/prepare_fire_model.py` applies the conventions of the study: the full event grid,
the elevation range of the training crop, a start at the first observed fire map, and wind
matched by calendar date. `fire_model.npz` holds

| Array | Shape | Meaning |
|---|---|---|
| `parameter_alpha_gamma` | (15, H, W) | receiving coefficient `a_n` |
| `parameter_p_base` | (15, H, W) | base transmission coefficient `p_n^0` |
| `parameter_fuel_factor` | (15, H, W) | CNN fuel coefficient `F_n`. Treatment multiplies this field by `q` |
| `parameter_alpha_w1`, `parameter_alpha_w2`, `parameter_alpha_s` | (15, H, W) | wind and slope coefficients |
| `speeds`, `directions` | (15, H, W) | daily scaled wind speed and direction (radians) |
| `slope`, `aspect` | (H, W) | scaled terrain slope and upslope direction (radians) |
| `mask` | (H, W) | burnable-fuel indicator `m` |
| `initial` | (3, H, W) | initial unburned, burning and burned probabilities |

The microstep is given in Section 2.1 of the paper: Equation (1), the ignition probability
that follows it, and the state update in Equation (2). The code also clips the receiving and
transmission factors to [0, 100] and the ignition hazard to at most 50. These clips come from
the original CA-CNN code, and the paper omits them.
`WildfireCA.compute_ignition_probability` in `src/model.py` evaluates the microstep. A
treatment field `q` multiplies `fuel_factor` before the call, which applies it at both the
receiving and the transmitting cell. The function `simulator` in the script is the complete
forecast.

## Inputs

| File | Array | Shape | Meaning |
|---|---|---|---|
| `operational_units.npz` | `ids` | (H, W) | polygon identifier of each cell |
| | `allowed` | (I+1, 3) | whether each program is eligible for the polygon |
| | `costs` | (I+1, 3) | total program cost over the polygon, `k_io` |
| | `initial` | (I+1,) | strongest eligible program, the reference `z_ref` |
| | `operation_maps` | (3, H, W) | cell operation code under each program |
| `candidate_operations.npz` | `multipliers` | (3, H, W) | residual fuel multiplier `q_io(x)` under each program |
| | `fuel_codes` | (H, W) | LANDFIRE FBFM40 fuel-model code |
| | `domains` | (2, H, W) | cells inside a polygon, and cells meeting the mechanical-access rules |
| | other masks | (H, W) | operational-review flags |
| `cost_fields.npz` | `cost_USD2025_ha` | (3, H, W) | per-hectare cost under each program |
| `exposure.npz` | `eligible`, `affected_day15` | (H, W) | cells that are burnable and initially unburned, and the study's untreated day-15 affected probability |
| `assets/protection_areas.json`, `assets/protection_area_cells.npz` | | | named settlements, roads and facilities with their cell indices |
| `planning_protocol.json` | | | forecast dates, list of selected named assets, model provenance |
| `operational_protocol.json`, `reference.json` | | | polygon construction rules and statistics, reference cost |

For an assignment `z` (program index per polygon), the treatment field and cost are

```python
q_z  = numpy.take_along_axis(multipliers, z[ids][None], axis=0)[0]
cost = costs[numpy.arange(len(z)), z].sum()
```

Both expressions were checked against every saved plan when this folder was assembled.

## Protected cells and limits

The paper protects named assets (settlements, roads, facilities) and electrical assets (power lines and electrical facilities). Each asset is expanded by a 150 m band. The plan must keep the day-15 affected probability of every protected cell at or below its limit. The limit of a cell is its day-15 affected probability under the reference plan, which applies the strongest eligible program to every polygon.

This folder ships the named assets. It does **not** ship the electrical layers or anything that encodes their geometry, because several source services restrict redistribution. Most regions derive from electrical features. The region masks, the positions of the protected cells and every array with one value per protected cell are therefore not shipped. Instead:

- `scripts/collect_electrical_network.py` queries the public services used in the paper.
  Check the terms of each service before you download.
- `scripts/build_protected_regions.py` rasterizes the layers, applies the exposure test, adds
  the protection band and merges identical regions. It then lists the protected cells and reads
  their limits from the reference forecast in `generated/<fire>/forecasts.npz`.
- `results/<fire>/two_phase/regions_reference_band_150m.json` lists the SHA-256 hash and cell
  count of every region and asset core of the paper, the number of protected cells, and one hash
  of all cell limits. It holds no geometry, names or voltages.

The script writes `constraints.npz` with

| Array | Shape | Meaning |
|---|---|---|
| `masks` | (R, H, W) | cells of every region |
| `cores` | (assets, H, W) | cells of every asset without its band |
| `protected_flat_indices` | (J,) | row-major index of every protected cell, `row * W + column` |
| `limits_probability` | (J,) | limit of every protected cell |

`comparison.json` states whether the rebuilt regions equal the paper's regions (`identical_to_paper`) and whether the limits equal the paper's limits (`limits_identical_to_paper`). When the regions are identical, the script orders them as in the paper.

**Snapshot.** The paper used downloads from 2026-11-03. The utility layers describe current infrastructure and change over time, so a later download can differ. Run with the authors' cached downloads and the study's forecasts, the script reproduced all regions, asset cores, protected cells and limits of the four fires exactly.

## Results

`results/<fire>/two_phase/` holds `reference.npz` (strongest-treatment reference),
`screened_plan.npz` (after Phase 1, grouped removal) and `optimized_plan.npz` (after Phase 2,
the final plan of the paper). Each plan file holds

| Array | Shape | Meaning |
|---|---|---|
| `assignment` | (I+1,) | program of every polygon |
| `labels` | (H, W) | cell operation code |
| `cost_USD2025` | scalar | plan cost |

The folder also holds the Phase 1 polygon order (`removal_order.npz`), run times, final status
and the run protocol.

To check a plan, forecast it with the treatment field of its assignment and compare the day-15
affected probability at the protected cells with `limits_probability`. For every saved plan,
`results/summary/plan_checks.json` gives the values the study obtained: the cost, the largest
excess over a cell limit, and the expected affected area summed over the protected cells.

`results/<fire>/dpv_ordered_removal/` holds the final plan and the completion record of
DPV-ordered removal (Section 4.2 of the paper). It applies the grouped removal test of Phase 1
on repeated passes, in an order derived from the downstream protection value. Routes traced
through the untreated CA-CNN forecast and local spread scores give this value. The paper's
benchmark tables and the files in `results/summary/` list the method under the key `dpv`.

`results/<fire>/dpv_guidance_comparison/rothermel/` holds the plan of the same grouped search
when the Rothermel model supplies the paths and the local spread rates. For Pier,
`ca_paths_rothermel_speeds/` holds the plan obtained with CA-CNN paths and Rothermel speeds.
The paper reports these costs in its text. `results/summary/dpv_guidance_four_fires.json` and
`dpv_guidance_pier.json` list them.

`results/<fire>/benchmarks/<method>/` holds the final plan and the completion record of the
three literature benchmarks: `annealing`, `nsga2` and `path_delay` (spread-path placement).
`results/summary/` also holds the time budgets (`benchmark_budgets.json`) and the values behind
the paper's cost, area, time, benchmark and sensitivity tables. Some of these files keep the
prefix `regional_` from an earlier draft.

## Sensitivity study (`sensitivity/`)

The study replays three fixed plans of every fire under changed treatment responses. The plans
are the strongest-treatment reference, the final two-phase plan and the DPV-ordered removal
plan. Their assignments, costs, protected cells and nominal cell limits stay fixed, and no
optimization is performed.

Scenarios: `grass_high_consumption` (95% and 100%), `grass_low_consumption` (80% and 90%),
`shrub_residual_017` and `shrub_residual_050`.

| File | Content |
|---|---|
| `protocol.json` | scenario definitions and their sources |
| `summary.json`, `comparison.csv` | the 60 comparisons behind the paper's sensitivity table |
| `<fire>/<scenario>/multipliers.npz` | array `multipliers` (3, H, W): residual fuel multiplier of every program under the scenario |
| `<fire>/assets_reference_two_phase.csv` | affected area of every asset under the reference and two-phase plans |
| `<fire>/assets_dpv_ordered_removal.csv` | the same for the DPV-ordered removal plan |

In these files the plan labels are `reference`, `two_phase` (the final two-phase plan) and
`dpv` (DPV-ordered removal). The column `affected_asset_core_ha` is the expected affected area
inside the assets. The column `summed_positive_cell_excess_ha` is the excess of the paper: the
increase of affected probability above the nominal limit, summed over the protected cells where
it is positive and multiplied by the cell area. To replay a scenario, use its `multipliers.npz`
in place of the array `multipliers` of `inputs/<fire>/candidate_operations.npz`, and keep the
plan assignment.

## Notes for independent implementations

- **Limits.** Every cell limit equals the reference outcome exactly. Feasibility uses an
  absolute tolerance of 1e-10 in affected probability per protected cell, which covers rounding
  only. A plan replayed with different arithmetic or hardware can exceed a limit by more than
  this tolerance. Compute the limits and replay the plan with the same forecasts.
- **Precision.** The CNN runs in float32. The cellular automaton runs in float64.
- **Paths.** File paths written inside JSON and CSV files refer to the authors' working
  repository. `manifest.json` maps every file here to its original path and states any change
  made on copy, such as arrays removed because they encode the protected cells.
- **Legacy fields.** Some JSON files carry fields from earlier studies on the same inputs, for
  example `budget_USD`. The method in the paper does not use them.

## Data sources

- Fuel models: LANDFIRE FBFM40. Roads: US Census TIGER and US Forest Service road layers.
- Water, protected and roadless areas: public federal and state GIS services listed in
  `operational_protocol.json` and `planning_protocol.json`.
- Named assets: US Census blocks and places, TIGER roads, HIFLD transmission lines,
  OpenStreetMap facilities (© OpenStreetMap contributors, Open Database License) and
  Microsoft US Building Footprints (Open Database License).
- Electrical lines and facilities, **not redistributed here**: PG&E GRIP, SCE DRPEP, the
  California Energy Commission transmission-line layer, the Oregon GIS Framework, HIFLD and
  OpenStreetMap. The download script queries these services directly.
- Treatment responses and costs: the publications cited in Section 2.2 of the paper.
