# Neural CA Wildfire

CNN-parameterized probabilistic Cellular Automaton for wildfire spread prediction. Code accompanying the paper *Neural-Parameterized Cellular Automata for Wildfire Spread* by Maksym Zhenirovskyy, Ion Matei, Rohit Vuppala, Takuya Kurihana and Hon Yung Wong.

A Multi-Scale CNN produces per-cell parameter maps for a 3-state probabilistic CA (`p_unburned`, `p_burning`, `p_burned`). The CA simulates fire spread over real LANDFIRE/ERA5 inputs for six historical fires: Bear 2020, Brattain 2020, Buck 2017, Chimney 2016, Ferguson 2018, Pier 2017.

---

## 1. Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
uv sync
```

This creates `.venv/` with all dependencies (JAX, Equinox, Optax, matplotlib, OpenCV, etc.).

**Data.** The raw data ships as a single HDF5 file, already included at `data/hdf5/dataset.hdf5` (sourced from the [Mendeley Data repository](https://data.mendeley.com/datasets/nx2wsksp9k/1)).

Run the extractor to unpack it into per-event numpy files:

```bash
.venv/bin/python data/data_extractor.py
```

This reads `data/hdf5/dataset.hdf5` and writes one directory per event under [data/data_real/](data/data_real/):

```text
data/data_real/
├── Bear_2020
├── Brattain_2020
├── Buck_2017
├── Chimney_2016
├── Ferguson_2018
└── Pier_2017
```

Each event directory holds LANDFIRE rasters and ERA5 wind components — `ELEV2020.npy`, `SLPD2020.npy`, `ASP2020.npy`, `230CBD.npy`, `230CC.npy`, `230CH.npy`, `230FBFM40.npy`, `u_component_of_wind_10m.npy`, `v_component_of_wind_10m.npy`, and `fire.npy` (daily burn perimeters) — plus a generated `<Event>.md` describing every variable.

---

## 2. Reproduce the paper

**The trained model used in the paper is already included at [result_saved/best_model/](result_saved/best_model/)** (`best_loss.eqx`, `best_iou.eqx`, `config.yaml`, `loss.csv`, `training.log`). The plotting notebooks are wired to this checkpoint, so you can regenerate every paper figure without retraining.

### 2.1 Generate the plots (no training required)

From the `src/` directory, open and run:

```bash
.venv/bin/jupyter notebook src/paper_plots_1d.ipynb
.venv/bin/jupyter notebook src/paper_plots_2d.ipynb
```

- **[src/paper_plots_1d.ipynb](src/paper_plots_1d.ipynb)** — 2×2 metric panels (IoU, Manhattan distance, Precision, Recall) per event over the full forecast horizon.
- **[src/paper_plots_2d.ipynb](src/paper_plots_2d.ipynb)** — per-day spatial comparisons (Prediction / Target / FP–FN diff).

Both notebooks load `result_saved/best_model/best_loss.eqx` for all six events and write PNGs to [result_saved/paper_plots/](result_saved/paper_plots/) (`<Event>_<chunk>.png`, `all_events_metrics.png`).

### 2.2 Train your own model (optional)

Train on all six events using the default config:

```bash
.venv/bin/python src/main.py
```

Or pass an explicit config:

```bash
.venv/bin/python src/main.py src/config.yaml
```

On a SLURM cluster:

```bash
sbatch src/submit.sh
```

Training writes to `result/run_MM_DD_HH_MM/`:

- `best_loss.eqx` — checkpoint with the lowest training loss
- `best_iou.eqx` — checkpoint with the best validation IoU
- `config.yaml` — copy of the config used
- `training.log` — full log
- `loss.csv` — per-epoch loss and IoU

To match paper-quality training, set `epochs: 5000` in [src/config.yaml](src/config.yaml) (it's set low by default for smoke tests) and run on GPU NVIDIA H200. Training takes ~10–15 h on a single GPU.

To plot your own checkpoint, either copy your `result/run_*/` over `result_saved/best_model/`, or edit the `events_model` dict at the top of each notebook to point at your run directory.

---

## 3. Configuration

All knobs live in [src/config.yaml](src/config.yaml). The most common adjustments:

| Section | Key | Meaning |
| --- | --- | --- |
| `data.train_events` | `<event>: {days, y1, y2, x1, x2}` | Day range and spatial crop per fire |
| `data.test_events` | same | Held-out events for validation |
| `model` | `kernel_size`, `branch_out_channels`, `burn_duration`, `dropout_rate` | CNN architecture and CA dynamics |
| `training` | `epochs`, `lr`, `steps_per_day`, loss weights | Optimization |
| `output` | `out_dir` | Result directory prefix (timestamp appended) |
| `pretrained_model_path` | path | Fine-tune from an existing `.eqx` checkpoint |

---

## 4. Project layout

```text
src/
├── main.py               # Entry point
├── config.yaml           # Training configuration
├── model.py              # FuelEmbedding + MSCNN + WildfireCA + WildfireModel
├── simulation.py         # Multi-day / single-day JAX scan loops
├── training.py           # Training loop, optimizer, validation
├── loss.py               # BCE, MSE, IoU, area, pooled MSE
├── data_loader.py        # FireEvent dataclass, per-event loader
├── preprocessing.py      # CNN/CA feature engineering, FBFM40, fuel mask
├── utils.py              # YAML loading, logging, config copy
├── submit.sh             # SLURM job script
├── paper_plots_1d.ipynb  # Metric figures
├── paper_plots_2d.ipynb  # Spatial figures
└── tests/                # pytest unit tests

data/data_real/           # Per-event LANDFIRE + ERA5 + fire data
result_saved/             # Saved checkpoints and paper figures
docs/paper/main.pdf       # The paper
```

---

## 5. Tests

```bash
.venv/bin/python -m pytest src/tests/ -v
```
