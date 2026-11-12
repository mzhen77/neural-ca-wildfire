"""Rebuild the protected regions of the paper from the asset layers.

This folder ships the named assets (settlements, roads, facilities) but not the
electrical layers, whose terms restrict redistribution. This script rasterizes
the electrical layers downloaded by collect_electrical_network.py, applies the
exposure test, adds the protection band and merges identical regions. It then
compares the result with the reference table of the paper's regions.

Steps, as in the paper and its reproducibility record:
  1. A line covers the cells it touches. A facility covers its geometry plus 40 m.
  2. Candidate cells are burnable and unburned at the start of the forecast.
  3. An electrical feature is protected if the untreated day-15 affected
     probability reaches 0.5 in one of its cells.
  4. Each asset core is expanded by the band (150 m in the nominal study).
  5. Assets with identical expanded regions share one constraint.
  6. Each bound equals the reference (strongest-treatment) expected affected area.

Outputs in generated/<fire>/band_<width>m/:
  constraints.npz              masks (J,H,W), cores (assets,H,W), limits_ha (J,)
  assets.json                  id, name, kind and constraint index of every asset
  protection_masks.npz         union of regions and union of asset cores
  line_and_facility_threat.csv electrical features with their exposure
  comparison.json              agreement with the paper's regions

The exposure test reads inputs/<fire>/exposure.npz, the study's untreated day-15
forecast, so the regions do not depend on the hardware used to regenerate the
fire model. Bounds come from the reference table when the regions equal the
paper's regions. Otherwise they are recomputed from generated/<fire>/forecasts.npz,
written by prepare_fire_model.py --forecasts.

Usage, from the repository root:
    python vegetation_planning/scripts/build_protected_regions.py [--fire Pier_2017] [--band 150]
No fire model is evaluated and no GPU is needed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt

PLANNING = Path(__file__).resolve().parents[1]
FIRES = ['Pier_2017', 'Brattain_2020', 'Chimney_2016', 'Ferguson_2018']
CELL_HA = .09


class Grid:
    def __init__(self, georeferencing):
        m = json.loads(Path(georeferencing).read_text())
        self.xmin, self.ymin, self.xmax, self.ymax = m['bounds']
        self.h, self.w = m['height'], m['width']
        self.dx, self.dy = (self.xmax-self.xmin)/self.w, (self.ymax-self.ymin)/self.h
        self.x = self.xmin+(np.arange(self.w)+.5)*self.dx
        self.y = self.ymax-(np.arange(self.h)+.5)*self.dy

    def cells(self, geom, distance):
        """Flat indices of cells the geometry touches or whose centers lie within distance."""
        if geom.geom_type in ('LineString', 'MultiLineString'):
            parts = [self.window_cells(shapely.LineString([a, b]), distance)
                     for line in getattr(geom, 'geoms', [geom]) for a, b in zip(line.coords[:-1], line.coords[1:])]
            return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=int)
        return self.window_cells(geom, distance)

    def window_cells(self, geom, distance):
        x0, y0, x1, y1 = geom.bounds
        pad = distance+max(self.dx, self.dy)
        ix = np.flatnonzero((self.x >= x0-pad) & (self.x <= x1+pad))
        iy = np.flatnonzero((self.y >= y0-pad) & (self.y <= y1+pad))
        if not len(ix) or not len(iy):
            return np.zeros(0, dtype=int)
        xx, yy = np.meshgrid(self.x[ix], self.y[iy])
        hit = shapely.distance(shapely.points(xx.ravel(), yy.ravel()), geom) <= distance
        boxes = shapely.box(xx.ravel()-self.dx/2, yy.ravel()-self.dy/2, xx.ravel()+self.dx/2, yy.ravel()+self.dy/2)
        hit |= shapely.intersects(boxes, geom)
        rows, cols = np.meshgrid(iy, ix, indexing='ij')
        return (rows.ravel()*self.w+cols.ravel())[hit]


def osm_geometry(e, transform):
    if e['type'] == 'node':
        return shapely.Point(*transform.transform(e['lon'], e['lat']))
    coords = [transform.transform(p['lon'], p['lat']) for p in e.get('geometry', [])]
    if len(coords) < 2:
        return None
    if e['tags']['power'] in ('plant', 'substation', 'generator') and coords[0] == coords[-1] and len(coords) >= 4:
        return shapely.make_valid(shapely.Polygon(coords))
    return shapely.LineString(coords)


def inventory(sources):
    """Features of every downloaded layer, with the candidate rule of each source."""
    features = []
    transform = Transformer.from_crs(4326, 3310, always_xy=True)
    data = json.loads((sources/'osm_power.json').read_text())
    for e in data['elements']:
        tags = e.get('tags', {}); power = tags.get('power')
        if power not in ('line', 'minor_line', 'cable', 'minor_cable', 'plant', 'substation', 'generator', 'transformer'):
            continue
        geom = osm_geometry(e, transform)
        if geom is None:
            continue
        is_line = power in ('line', 'minor_line', 'cable', 'minor_cable')
        inactive = any(k in tags for k in ('construction:power', 'proposed:power', 'abandoned:power', 'disused:power'))
        overhead = power in ('line', 'minor_line') and tags.get('location') not in ('underground', 'underwater')
        if power in ('cable', 'minor_cable') and tags.get('location') in ('overhead', 'overground'):
            overhead = True
        features.append(dict(id=f"osm/{e['type']}/{e['id']}", source='OSM', kind='line' if is_line else 'facility',
            name=tags.get('name', power), power=power, voltage=tags.get('voltage', ''),
            overhead=overhead if is_line else None, candidate=(not inactive) and (overhead or not is_line),
            geometry=geom))
    for name in ('cec_lines', 'hifld_lines'):
        path = sources/f'{name}.json'
        if not path.exists():
            continue
        for e in json.loads(path.read_text())['features']:
            a = e['attributes']; paths = e.get('geometry', {}).get('paths', [])
            if not paths:
                continue
            cec = name == 'cec_lines'
            ident = a.get('OBJECTID' if cec else 'OBJECTID_1', a.get('OBJECTID'))
            overhead = str(a.get('Type' if cec else 'TYPE', '')).upper() in ('OH', 'OVERHEAD')
            status = str(a.get('Status' if cec else 'STATUS', '')).upper()
            features.append(dict(id=f'{name}/{ident}', source='CEC' if cec else 'HIFLD', kind='line',
                name=a.get('TLine_Name', '').strip() or a.get('Name', a.get('OWNER', 'line')),
                power='line', voltage=str(a.get('kV' if cec else 'VOLTAGE', '')), voltage_unit='kV',
                overhead=overhead, candidate=overhead and status in ('OPERATIONAL', 'IN SERVICE'),
                geometry=shapely.MultiLineString(paths)))
    for path in sorted(sources.glob('*.json')):
        name = path.stem
        if not (name.startswith(('pge_', 'sce_')) or name == 'oregon_lines'):
            continue
        data = json.loads(path.read_text())
        if 'features' not in data:
            continue
        for e in data['features']:
            a = e['attributes']; g = e.get('geometry', {})
            if 'paths' in g:
                geom = shapely.MultiLineString(g['paths']); kind = 'line'
            elif 'x' in g:
                geom = shapely.Point(g['x'], g['y']); kind = 'facility'
            else:
                continue
            features.append(dict(id=f"{name}/{a.get('OBJECTID', a.get('objectid', a.get('OBJECTID_1')))}",
                source='Oregon' if name == 'oregon_lines' else ('PG&E' if name.startswith('pge_') else 'SCE'),
                kind=kind, name=a.get('FeederName') or a.get('circuit_name') or a.get('CIRCT_NAM') or a.get('SUB_NAME')
                or a.get('SubstationName') or a.get('NAME') or name,
                power='line' if kind == 'line' else 'substation',
                voltage=str(a.get('voltage_kv') or a.get('circuit_voltage') or a.get('CIRCUIT_VOLTAGE') or a.get('Voltage_kV') or ''),
                voltage_unit='kV', overhead=None, candidate=True, geometry=geom))
    return features


def electrical_assets(grid, sources, eligible, final_affected):
    """Rows of the exposure table and the candidate cells of each feature."""
    bounds = shapely.box(grid.xmin, grid.ymin, grid.xmax, grid.ymax)
    features = [f for f in inventory(sources) if f['geometry'].intersects(bounds)]
    assert len({f['id'] for f in features}) == len(features), 'Duplicate source feature identifiers'
    rows, cells = [], []
    flat = final_affected.ravel()
    for f in features:
        geom = f['geometry'].intersection(bounds)
        index = grid.cells(geom, 0. if f['kind'] == 'line' else 40.)
        index = index[eligible.ravel()[index]]
        peak = float(flat[index].max()) if len(index) else 0.
        rows.append(dict(id=f['id'], source=f['source'], name=f['name'], kind=f['kind'], power=f['power'],
            voltage=f['voltage'], voltage_unit=f.get('voltage_unit', 'V'), overhead=f['overhead'],
            candidate=f['candidate'], threatened=bool(f['candidate'] and peak >= .5),
            length_in_grid_km=geom.length/1000 if f['kind'] == 'line' else None, cells=len(index),
            day15_max_affected_fraction=peak, day15_expected_affected_ha=float(CELL_HA*flat[index].sum())))
        cells.append(index)
    return rows, cells


def named_assets(inputs, shape):
    """Selected settlements, roads and facilities with their cells."""
    cfg = json.loads((inputs/'planning_protocol.json').read_text())
    assert cfg['window'] == [0, shape[0], 0, shape[1]], 'The four studies use the full data grid'
    names = {x['label']: x for x in cfg['regions']}
    areas = json.loads((inputs/'assets/protection_areas.json').read_text())['areas']
    items = []
    with np.load(inputs/'assets/protection_area_cells.npz') as cells:
        for k, a in enumerate(areas):
            if a['label'] not in names:
                continue
            core = np.zeros(shape, bool)
            core.ravel()[cells['index'][cells['offsets'][k]:cells['offsets'][k+1]]] = True
            if core.any():
                label = a['label']
                items.append(dict(id=label, name=names[label].get('name', label), kind=names[label].get('kind'), core=core))
    return items


def digest(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, bool).tobytes()).hexdigest()


def build(fire, band, sources, output):
    inputs = PLANNING/'inputs'/fire
    grid = Grid(inputs/'georeferencing.json')
    shape = (grid.h, grid.w)
    with np.load(inputs/'exposure.npz') as f:
        eligible, affected_day15 = f['eligible'], f['affected_day15']
    assert eligible.shape == shape
    rows, cells = electrical_assets(grid, sources, eligible, affected_day15)

    items = named_assets(inputs, shape)
    for row, index in zip(rows, cells):
        if not row['threatened'] or not len(index):
            continue
        core = np.zeros(shape, bool); core.ravel()[index] = True
        items.append(dict(id='E:'+row['id'], name=row['name'], kind=row['kind'], core=core))

    masks, cores, assets, seen = [], [], [], {}
    for item in items:
        core = item.pop('core')
        mask = distance_transform_edt(~core, sampling=30.) <= band
        key = digest(mask)
        if key not in seen:
            seen[key] = len(masks); masks.append(mask)
        item['constraint'] = seen[key]; assets.append(item); cores.append(core)
    masks, cores = np.asarray(masks), np.asarray(cores)
    limits = None

    # Compare with the regions of the paper. Identical sets take the paper's order and bounds.
    comparison = dict(fire=fire, band_m=band, regions=len(masks), assets=len(assets))
    reference_path = PLANNING/'results'/fire/'two_phase'/(f'regions_reference_band_{band:g}m.json')
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        mine = [digest(m) for m in masks]
        theirs = [r['sha256'] for r in reference['regions']]
        core_reference = {a['id']: a for a in reference['assets']}
        core_mine = {a['id']: digest(c) for a, c in zip(assets, cores)}
        comparison.update(paper_regions=len(theirs), paper_assets=len(core_reference),
            matching_regions=len(set(mine) & set(theirs)),
            regions_only_here=len(set(mine)-set(theirs)), regions_only_in_paper=len(set(theirs)-set(mine)),
            assets_only_here=sorted(set(core_mine)-set(core_reference)),
            assets_only_in_paper=sorted(set(core_reference)-set(core_mine)),
            assets_with_changed_cells=sorted(k for k in set(core_mine) & set(core_reference)
                                             if core_mine[k] != core_reference[k]['core_sha256']))
        identical = sorted(mine) == sorted(theirs) and not comparison['assets_only_here'] \
            and not comparison['assets_only_in_paper'] and not comparison['assets_with_changed_cells']
        comparison['identical_to_paper'] = identical
        if identical:
            order = [mine.index(h) for h in theirs]
            position = {old: new for new, old in enumerate(order)}
            masks = masks[order]
            for a in assets:
                a['constraint'] = position[a['constraint']]
            wanted = [a['id'] for a in reference['assets']]
            rank = {k: i for i, k in enumerate(wanted)}
            permutation = sorted(range(len(assets)), key=lambda i: rank[assets[i]['id']])
            assets = [assets[i] for i in permutation]; cores = cores[permutation]
            limits = np.array([r['limit_ha'] for r in reference['regions']])
            comparison['bounds'] = 'Paper bounds from the reference table'
    else:
        comparison['identical_to_paper'] = None
    if limits is None:
        forecasts = PLANNING/'generated'/fire/'forecasts.npz'
        if forecasts.exists():
            with np.load(forecasts) as f:
                reference_final = 1-f['strongest'][-1]
            limits = np.array([CELL_HA*reference_final[m].sum() for m in masks])
            comparison['bounds'] = 'Recomputed from generated/<fire>/forecasts.npz for the regions built here'
        else:
            limits = np.full(len(masks), np.nan)
            comparison['bounds'] = 'Not available. Run prepare_fire_model.py --forecasts, then rerun this script'

    dest = output/fire/f'band_{band:g}m'
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'constraints.npz', masks=masks, cores=cores, limits_ha=limits)
    (dest/'assets.json').write_text(json.dumps(assets, indent=2)+'\n')
    np.savez_compressed(dest/'protection_masks.npz', protected_regions=masks.any(axis=0), combined_core=cores.any(axis=0))
    with (dest/'line_and_facility_threat.csv').open('w') as f:
        writer = csv.DictWriter(f, list(rows[0]) if rows else ['id'])
        writer.writeheader(); writer.writerows(rows)
    (dest/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    return comparison


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fire', action='append', choices=FIRES)
    p.add_argument('--band', type=float, default=150., help='protection band in metres')
    p.add_argument('--sources', type=Path, help='folder with the downloaded electrical layers of one fire')
    p.add_argument('--output', type=Path, default=PLANNING/'generated')
    args = p.parse_args()
    if args.sources and len(args.fire or FIRES) != 1:
        p.error('--sources applies to one fire')
    for fire in args.fire or FIRES:
        sources = args.sources or PLANNING/'generated'/fire/'electrical_sources'
        result = build(fire, args.band, sources, args.output)
        print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}), flush=True)
