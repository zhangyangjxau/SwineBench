#!/usr/bin/env python3
"""Assemble per-model mean-pooled embedding matrices (rows aligned to the NT-v2 cache) for the MLP."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EMBEDDINGS = ROOT / 'embeddings'
DATA = ROOT / 'data' / 'processed'
CANON = DATA / 'ntv2_meanpool'
BASE = DATA / 'features'
DIMS = {'ntv2': 1024, 'dnabert2': 768, 'genalm': 768, 'hyenadna': 256, 'evo2': 4096, 'generator': 3072}


def prepare(model):
    out = BASE / model
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        return
    start = time.time()
    dim = DIMS[model]
    rows = pd.read_csv(CANON / 'rows.csv.gz')
    keys = pd.MultiIndex.from_frame(rows[['gene_id', 'subject_id']])
    for name in ['rows.csv.gz', 'labels.npy']:
        p = out / name
        if not p.exists():
            p.symlink_to(CANON / name)

    if model == 'ntv2':
        p = out / 'embeddings.npy'
        if not p.exists():
            p.symlink_to(CANON / 'embeddings.npy')
        manifest = {'reused': str(CANON), 'shape': [len(rows), dim]}
    else:
        x = np.lib.format.open_memmap(out / 'embeddings.partial.npy', mode='w+', dtype='float32',
                                      shape=(len(rows), dim))
        filled = np.zeros(len(rows), dtype=bool)
        paths = sorted((EMBEDDINGS / model).glob('*_ind_*.csv.gz'),
                       key=lambda p: int(p.name.split('_ind_')[1].split('.')[0]))
        assert len(paths) == 36, (model, len(paths))
        cols = [f'embedding_{i}' for i in range(dim)]
        for k, p in enumerate(paths):
            d = pd.read_csv(p, dtype={c: np.float32 for c in cols})
            sourcekeys = pd.MultiIndex.from_frame(d[['gene_id', 'subject_id']])
            assert sourcekeys.is_unique
            ii = keys.get_indexer(sourcekeys)
            keep = ii >= 0
            dest = ii[keep]
            assert not filled[dest].any(), 'Duplicate source row'
            values = d.loc[keep, cols].to_numpy()
            assert np.isfinite(values).all()
            x[dest] = values
            filled[dest] = True
            print('LOAD', model, k + 1, len(paths), int(filled.sum()), 'seconds', round(time.time() - start, 1), flush=True)
        assert filled.all(), (model, int((~filled).sum()))
        x.flush()
        del x
        os.replace(out / 'embeddings.partial.npy', out / 'embeddings.npy')
        manifest = {'sources': [str(p) for p in paths], 'shape': [len(rows), dim]}

    manifest['seconds'] = time.time() - start
    (out / 'COMPLETE.json').write_text(json.dumps(manifest, indent=2))
    print('PREPARE_COMPLETE', model, round(time.time() - start, 1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('models', nargs='+', choices=list(DIMS))
    a = p.parse_args()
    for m in a.models:
        prepare(m)
