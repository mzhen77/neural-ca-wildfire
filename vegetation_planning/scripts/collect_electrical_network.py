"""Download the electrical line and facility layers for the four fire domains.

This repository does not redistribute these layers. This script queries the public
services used in the paper and stores each reply with its query and fetch time
in vegetation_planning/generated/<fire>/electrical_sources/. Check the terms of each service
before use. No fire model is evaluated.

The services describe current infrastructure and change over time. A later
download can differ from the snapshot used in the paper (2026-11-03).

Usage, from the repository root:
    python vegetation_planning/scripts/collect_electrical_network.py [--fire Pier_2017] [--map-api]
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from pyproj import Transformer

PLANNING = Path(__file__).resolve().parents[1]
FIRES = ['Pier_2017', 'Brattain_2020', 'Chimney_2016', 'Ferguson_2018']
CEC = 'https://services3.arcgis.com/bWPjFyq029ChCGur/arcgis/rest/services/Transmission_Line/FeatureServer/2'
HIFLD = 'https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0'
OREGON = 'https://services8.arcgis.com/8PAo5HGmvRMlF2eU/arcgis/rest/services/Electric_Transmission_Lines/FeatureServer/0'
PGE = 'https://services2.arcgis.com/mJaJSax0KPHoCNB6/arcgis/rest/services/DRPComplianceRelProd/FeatureServer'
SCE = 'https://services5.arcgis.com/z6hI6KRjKHvhNO0r/arcgis/rest/services/ICA_Layer/FeatureServer'
OVERPASS = ['https://overpass.private.coffee/api/interpreter',
            'https://overpass.kumi.systems/api/interpreter']
HEADERS = {'User-Agent': 'VegetationTreatmentPlanningData/1.0 (academic research)'}


def get(url, params):
    r = requests.get(url, params=params, headers=HEADERS, timeout=180)
    r.raise_for_status()
    data = r.json()
    if 'error' in data or 'remark' in data:
        raise RuntimeError(str(data.get('error', data.get('remark'))))
    return data


def arcgis(url, bounds):
    geometry = dict(zip(('xmin', 'ymin', 'xmax', 'ymax'), bounds))
    geometry['spatialReference'] = {'wkid': 3310}
    params = dict(f='json', geometry=json.dumps(geometry), geometryType='esriGeometryEnvelope',
                  inSR=3310, outSR=3310, spatialRel='esriSpatialRelIntersects', where='1=1')
    ids_reply = get(url + '/query', dict(params, returnIdsOnly='true'))
    if 'objectIds' not in ids_reply:
        raise RuntimeError(f'Missing objectIds: {ids_reply}')
    ids = sorted(ids_reply['objectIds'] or [])
    features = []
    for start in range(0, len(ids), 100):
        reply = get(url + '/query', dict(f='json', objectIds=','.join(map(str, ids[start:start+100])),
                                         outFields='*', outSR=3310, returnGeometry='true'))
        if reply.get('exceededTransferLimit'):
            raise RuntimeError('Truncated ArcGIS response')
        features.extend(reply['features'])
    assert len(features) == len(ids)
    return dict(source=url, query=params, object_ids=ids, spatialReference={'wkid': 3310},
                features=features, fetched_utc=datetime.now(timezone.utc).isoformat())


def osm(bounds):
    t = Transformer.from_crs(3310, 4326, always_xy=True)
    west, south, east, north = t.transform_bounds(*bounds)
    bbox = f'({south},{west},{north},{east})'
    # Full ways, including vertices outside the bbox, preserve crossing geometry.
    query = ('[out:json][timeout:120];('
             f'way[power~"^(line|minor_line|cable|minor_cable)$"]{bbox};'
             f'nwr[power~"^(plant|substation|generator|transformer)$"]{bbox};'
             ');out meta geom;')
    errors = []
    for url in OVERPASS:
        try:
            data = get(url, {'data': query})
            return dict(source=url, query=query, fetched_utc=datetime.now(timezone.utc).isoformat(), **data)
        except Exception as exc:
            errors.append(f'{url}: {exc}')
    raise RuntimeError('; '.join(errors))


def osm_map(bounds, cache):
    """Read small map tiles from the OSM API. The paper's snapshot used this route.

    Cache the electrical subset of each reply. All vertices of selected ways
    remain present. Split a tile if the API reports too many nodes.
    """
    import numpy as np
    t = Transformer.from_crs(3310, 4326, always_xy=True)
    west, south, east, north = t.transform_bounds(*bounds)
    url = 'https://api.openstreetmap.org/api/0.6/map.json'
    cache.mkdir(parents=True, exist_ok=True)
    elements, queries = {}, []

    def tile(box):
        params = {'bbox': ','.join(f'{v:.9f}' for v in box)}
        key = hashlib.sha256(params['bbox'].encode()).hexdigest()[:16]
        path = cache / f'{key}.json'
        if path.exists():
            result = json.loads(path.read_text())
        else:
            r = requests.get(url, params=params, headers=HEADERS, timeout=120)
            if r.status_code == 400 and ('too many' in r.text.lower() or 'maximum' in r.text.lower()):
                w, s, e, n = box
                if max(e-w, n-s) < .002:
                    raise RuntimeError(r.text)
                if e-w >= n-s:
                    tile((w, s, (w+e)/2, n)); tile(((w+e)/2, s, e, n))
                else:
                    tile((w, s, e, (s+n)/2)); tile((w, (s+n)/2, e, n))
                return
            r.raise_for_status()
            data = r.json()
            all_elements = {(e['type'], e['id']): e for e in data['elements']}
            selected = [e for e in data['elements'] if 'power' in e.get('tags', {})]
            for e in selected:
                if e['type'] == 'way':
                    e['geometry'] = [dict(lon=all_elements['node', n]['lon'],
                                          lat=all_elements['node', n]['lat']) for n in e['nodes']]
            result = dict(source=url, query=params, fetched_utc=datetime.now(timezone.utc).isoformat(),
                          response_sha256=hashlib.sha256(r.content).hexdigest(),
                          attribution=data.get('attribution'), license=data.get('license'), elements=selected)
            path.write_text(json.dumps(result)+'\n')
            time.sleep(.2)
        queries.append({k: v for k, v in result.items() if k != 'elements'})
        for e in result['elements']:
            elements[e['type'], e['id']] = e

    xx = np.linspace(west, east, int(np.ceil((east-west)/.08))+1)
    yy = np.linspace(south, north, int(np.ceil((north-south)/.08))+1)
    for w, e in zip(xx[:-1], xx[1:]):
        for s, n in zip(yy[:-1], yy[1:]):
            tile((w, s, e, n))
    return dict(source=url, queries=queries, elements=list(elements.values()),
                fetched_utc=datetime.now(timezone.utc).isoformat())


def run(fire, map_api=False):
    dest = PLANNING / 'generated' / fire / 'electrical_sources'
    dest.mkdir(parents=True, exist_ok=True)
    geo = json.loads((PLANNING / 'inputs' / fire / 'georeferencing.json').read_text())
    bounds = [v + (-1000 if i < 2 else 1000) for i, v in enumerate(geo['bounds'])]
    extra = [('oregon_lines', OREGON)] if fire == 'Brattain_2020' else [
        ('pge_distribution', PGE+'/3'), ('pge_distribution_unavailable', PGE+'/4'),
        ('pge_substations', PGE+'/0')]
    if fire == 'Pier_2017':
        extra += [(f'sce_lines_{i}', SCE+f'/{i}') for i in (1, 2, 3)] + [('sce_substations', SCE+'/0')]
    for name, fetch in [('hifld_lines', lambda: arcgis(HIFLD, bounds)),
                        *([] if fire == 'Brattain_2020' else [('cec_lines', lambda: arcgis(CEC, bounds))]),
                        *[(name, lambda url=url: arcgis(url, bounds)) for name, url in extra],
                        ('osm_power', lambda: osm_map(bounds, dest/'osm_tiles') if map_api else osm(bounds))]:
        path = dest / f'{name}.json'
        if path.exists():
            continue
        begin = time.monotonic()
        try:
            record = fetch()
            record['bounds_3310'] = bounds
            path.write_text(json.dumps(record) + '\n')
            (dest / f'{name}_error.json').unlink(missing_ok=True)
            print(fire, name, len(record.get('features', record.get('elements', []))),
                  f'{time.monotonic()-begin:.1f}s', flush=True)
        except Exception as exc:
            (dest / f'{name}_error.json').write_text(json.dumps(dict(error=str(exc), bounds=bounds,
                fetched_utc=datetime.now(timezone.utc).isoformat()), indent=2))
            print(fire, name, 'FAILED', str(exc), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fire', action='append', choices=FIRES)
    parser.add_argument('--map-api', action='store_true', help='use the OSM map API in tiles instead of Overpass')
    args = parser.parse_args()
    for fire in args.fire or FIRES:
        run(fire, args.map_api)
