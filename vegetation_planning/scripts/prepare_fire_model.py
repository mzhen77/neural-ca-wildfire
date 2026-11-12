"""Generate the fire-model inputs of the vegetation-treatment study.

For each fire the script runs the fine-tuned CNN on the full data grid for the
15 forecast days and writes the daily parameter maps with the wind, terrain,
burnable mask and initial state. With --forecasts it also simulates the
untreated landscape and the strongest-treatment reference.

Conventions of the study:
  * The grid is the full event grid, not the training crop.
  * The elevation channel keeps the min-max range of the training crop, so the
    CNN inputs inside the crop equal those seen in training.
  * The forecast starts at the first observed fire map. Cells above Pth burn.
  * Wind stamped with date d drives the fire during day d.
  * The CNN runs in float32. The cellular automaton runs in float64 with
    50 microsteps per day.
  * A treatment multiplies the CNN fuel coefficient by q in [0, 1].
  * Each fire runs in its own process. On a GPU, a process that has already
    handled another grid can evaluate the CNN with different float32 rounding,
    and the forecast amplifies such differences. One fresh process per fire
    reproduced the study's maps and forecasts exactly on the study's hardware.

Usage, from the repository root after running data/data_extractor.py:
    python vegetation_planning/scripts/prepare_fire_model.py [--fire Pier_2017] [--forecasts]

Outputs in vegetation_planning/generated/<fire>/:
    fire_model.npz   parameter_* (15,H,W), speeds, directions, slope, aspect, mask, initial (3,H,W)
    forecasts.npz    untreated, strongest (16,H,W): daily unburned probability
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import equinox as eqx

REPO = Path(__file__).resolve().parents[2]
PLANNING = REPO/'vegetation_planning'
sys.path.insert(0, str(REPO/'src'))
from data_loader import load_fire_event  # noqa: E402
from model import WildfireCA, WildfireModel, compute_wind_speed_direction  # noqa: E402
from utils import load_config  # noqa: E402

FIRES = ['Pier_2017', 'Brattain_2020', 'Chimney_2016', 'Ferguson_2018']
DAYS = 15
STEPS_PER_DAY = 50


def fire_model(fire, data_dir):
    folder = REPO/'result_saved/stable_refit'/fire
    config = load_config(str(folder/'config.yaml'))
    config['data']['base_dir'] = str(data_dir)
    root = Path(data_dir)/fire
    crop = config['data']['train_events'][fire]
    y0, y1, x0, x1 = crop['y1'], crop['y2'], crop['x1'], crop['x2']

    files = config['data']['static_files']
    elevation = np.load(root/files['elevation']).astype(np.float32)
    height, width = elevation.shape
    dates = np.load(root/'fire_timestamps.npy').astype('datetime64[D]')
    with jax.experimental.enable_x64(False):
        event = load_fire_event(fire, config, dict(days=[0, len(dates)-1], y1=0, y2=height, x1=0, x2=width))
        model = WildfireModel(config, jax.random.PRNGKey(42))
        model = eqx.tree_inference(eqx.tree_deserialise_leaves(folder/'best_loss.eqx', model), True)

        # Elevation channel with the range of the training crop.
        missing = elevation <= config['preprocessing']['nodata_value']+1
        elevation[missing] = 0
        window = elevation[y0:y1, x0:x1][~missing[y0:y1, x0:x1]]
        lo, hi = window.min(), window.max()
        static = np.array(event.static_cnn_input)
        static[0] = np.where(missing, 0, np.clip((elevation-lo)/(hi-lo), 0, 1))

        # Wind by calendar date. Missing wind (ocean cells) is calm, as in training.
        dyn = config['data']['dynamic_files']
        stamps = np.load(root/dyn['wind_u'].replace('.npy', '_timestamps.npy')).astype('datetime64[D]')
        scale = config['preprocessing']['wind_speed_max']
        wind = [np.nan_to_num(np.load(root/dyn[k]).astype(np.float32), nan=0.0)/np.float32(scale)
                for k in ('wind_u', 'wind_v')]
        generate = jax.jit(lambda u, v: model.generate_params(jnp.asarray(static), u, v, jnp.asarray(event.fuel_type_map)))
        maps, speeds, directions = [], [], []
        for day in dates[0]+np.arange(DAYS).astype('timedelta64[D]'):
            index = int(np.flatnonzero(stamps == day).item())
            u, v = jnp.asarray(wind[0][index]), jnp.asarray(wind[1][index])
            maps.append({k: np.asarray(a) for k, a in generate(u, v).items()})
            speed, direction = compute_wind_speed_direction(u, v)
            speeds.append(np.asarray(speed)); directions.append(np.asarray(direction))
        pth = float(config['training']['Pth'])
        burning = (np.asarray(event.fire_seq[0]) > pth).astype(np.float64)

    arrays = {f'parameter_{k}': np.stack([m[k] for m in maps]).astype(np.float64) for k in maps[0]}
    arrays.update(speeds=np.stack(speeds).astype(np.float64), directions=np.stack(directions).astype(np.float64),
                  slope=np.asarray(event.slope_ca, np.float64), aspect=np.asarray(event.aspect_upslope, np.float64),
                  mask=np.asarray(event.fuel_mask, np.float64),
                  initial=np.stack([1-burning, burning, np.zeros_like(burning)]),
                  burn_duration=float(model.burn_duration), steps_per_day=STEPS_PER_DAY)
    return arrays


def simulator(arrays):
    """Daily unburned probability for a treatment field q. Equations (1)-(3) of the paper."""
    parameters = {k.removeprefix('parameter_'): jnp.asarray(v) for k, v in arrays.items() if k.startswith('parameter_')}
    speeds, directions, slope, aspect, mask = (jnp.asarray(arrays[k]) for k in ('speeds', 'directions', 'slope', 'aspect', 'mask'))
    initial = tuple(jnp.asarray(a) for a in arrays['initial'])
    tau = float(arrays['burn_duration'])
    clip = lambda a: jnp.clip(a, 0., 1.)

    def history(q):
        def day(state, weather):
            params, speed, direction = weather
            treated = {**params, 'fuel_factor': params['fuel_factor']*q}

            def micro(s, _):
                u, b, r = s
                p = clip(WildfireCA.compute_ignition_probability(b, treated, speed, direction, slope, aspect)*mask)
                new, spent = u*p, b/tau
                return (clip(u-new), clip(b+new-spent), clip(r+spent)), None
            final, _ = jax.lax.scan(micro, state, None, length=int(arrays['steps_per_day']))
            return final, final[0]
        _, daily = jax.lax.scan(day, initial, (parameters, speeds, directions))
        return jnp.concatenate([initial[0][None], daily])
    return jax.jit(history)


def reference_field(fire):
    """Fuel multipliers of the strongest eligible program on every management polygon."""
    with np.load(PLANNING/'inputs'/fire/'operational_units.npz') as a:
        ids, strongest = a['ids'], a['initial']
    with np.load(PLANNING/'inputs'/fire/'candidate_operations.npz') as a:
        multipliers = a['multipliers']
    return np.take_along_axis(multipliers, strongest[ids][None], axis=0)[0]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fire', action='append', choices=FIRES)
    p.add_argument('--data-dir', type=Path, default=REPO/'data/data_real')
    p.add_argument('--output', type=Path, default=PLANNING/'generated')
    p.add_argument('--forecasts', action='store_true', help='also simulate the untreated and reference landscapes')
    args = p.parse_args()
    fires = args.fire or FIRES
    if len(fires) > 1:
        for fire in fires:
            command = [sys.executable, __file__, '--fire', fire, '--data-dir', str(args.data_dir),
                       '--output', str(args.output)]+(['--forecasts'] if args.forecasts else [])
            subprocess.run(command, check=True)
        sys.exit(0)
    for fire in fires:
        dest = args.output/fire
        dest.mkdir(parents=True, exist_ok=True)
        arrays = fire_model(fire, args.data_dir)
        np.savez_compressed(dest/'fire_model.npz', **arrays)
        print(fire, 'fire model', arrays['initial'].shape[1:], flush=True)
        if args.forecasts:
            history = simulator(arrays)
            untreated = np.asarray(history(jnp.ones(arrays['mask'].shape)))
            strongest = np.asarray(history(jnp.asarray(reference_field(fire))))
            np.savez_compressed(dest/'forecasts.npz', untreated=untreated, strongest=strongest)
            area = .09*(1-untreated[-1]).sum(), .09*(1-strongest[-1]).sum()
            print(fire, f'affected area at day 15: untreated {area[0]:.1f} ha, reference {area[1]:.1f} ha', flush=True)
