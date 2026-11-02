#!/usr/bin/env python3
"""Align the breed- and sex-corrected liver expression residuals (x100) to the embedding rows."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / 'data' / 'inputs'
DATA = ROOT / 'data' / 'processed'
SOURCE = INPUTS / 'expression' / 'allchr_liver_sexcorrected.txt'
RAW_LOGCPM = INPUTS / 'expression' / 'LogCPM_counts_GENE-SWitCH_liver.txt'
INDIVIDUALS = ROOT / 'data' / 'splits' / 'individuals.csv'
CACHE = DATA / 'ntv2_meanpool'
OUT = DATA / 'labels'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'COMPLETE.json').exists():
        print('ALREADY_PREPARED')
        return
    start = time.time()
    rows = pd.read_csv(CACHE / 'rows.csv.gz')
    oldkeys = rows[['gene_id', 'subject_id']].copy()
    corrected = pd.read_csv(SOURCE, sep=r'\s+', index_col=0)
    assert corrected.shape == (15710, 300) and corrected.index.is_unique and corrected.columns.is_unique
    gi = corrected.index.get_indexer(rows.gene_id)
    si = corrected.columns.get_indexer(rows.subject_id + '_L')
    assert (gi >= 0).all() and (si >= 0).all()
    y = corrected.to_numpy(dtype=np.float64)[gi, si]
    assert np.isfinite(y).all()

    # Check the corrected file against the per-breed/sex centring formula on the raw logCPM.
    raw = pd.read_csv(RAW_LOGCPM, sep=r'\s+', index_col=0)
    people = pd.read_csv(INDIVIDUALS).set_index('subject_id')
    meta = people.loc[raw.index.str.removesuffix('_L')]
    a = raw.to_numpy()
    expected = np.empty_like(a)
    for (breed, sex), index in meta.groupby(['breed', 'sex']).indices.items():
        expected[index] = (a[index] - a[index].mean(0)) * 100
    actual = corrected.loc[raw.columns, raw.index].to_numpy().T
    error = float(np.max(np.abs(expected - actual)))
    assert error < 1e-6, error

    rows = rows.drop(columns=['observed_logCPM'])
    rows['observed_author_corrected_logCPM_x100'] = y
    assert rows[['gene_id', 'subject_id']].equals(oldkeys) and len(rows) == 160200
    rows.to_csv(OUT / 'rows.csv.gz', index=False)
    np.save(OUT / 'labels.npy', y)
    info = {
        'status': 'complete', 'source': str(SOURCE), 'rows': len(rows), 'source_shape': [15710, 300],
        'source_formula_max_abs_error': error, 'units': 'residual logCPM multiplied by 100',
        'seconds': time.time() - start,
    }
    (OUT / 'COMPLETE.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
