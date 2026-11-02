#!/usr/bin/env python3
"""Gene-specific cis-SNP baselines (Top SNP, ridge, LASSO, ElasticNet) with training-only SNP selection; mixed-breed fold."""
import argparse
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import pysam
from sklearn.linear_model import enet_path, Ridge, LinearRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from mlp import ROOT, DATA, OUT, CANON, FOLDS, TARGET, corr, fold_indices, save_evaluations, js

GENO = DATA / 'snp_tss_6kb'
SPLITS = ROOT / 'data' / 'splits'
VCF = ROOT / 'data' / 'inputs' / 'variants' / 'ALL_300.phased.renamed.vcf.gz'
RIDGE_ALPHAS = np.logspace(-4, 4, 9)
FRACTIONS = np.array([1., .3, .1, .03, .01, .003, .001])
RATIOS = [1., .1, .5, .9]
CANDIDATES = [dict(model='SNP_TopSNP')] + [dict(model='SNP_Ridge', alpha=float(a)) for a in RIDGE_ALPHAS]
for ratio in RATIOS:
    CANDIDATES += [dict(model='SNP_LASSO' if ratio == 1 else 'SNP_ElasticNet', l1_ratio=ratio, alpha_fraction=float(a))
                   for a in FRACTIONS]


def prepare():
    """Extract biallelic SNP dosages in [tss-3000, tss+3000) for the 400 seen genes."""
    GENO.mkdir(parents=True, exist_ok=True)
    if (GENO / 'COMPLETE.json').exists():
        return
    start = time.time()
    people = pd.read_csv(SPLITS / 'individuals.csv').sort_values('subject_id').reset_index(drop=True)
    genes = pd.read_csv(SPLITS / 'genes.csv', dtype={'chr': str})
    genes = genes[genes.gene_split.eq('seen')].sort_values('gene_id').reset_index(drop=True)
    vcf = pysam.VariantFile(str(VCF))
    samples = [r.breed + '_' + r.subject_id for r in people.itertuples()]
    assert set(samples) == set(vcf.header.samples) and len(samples) == 300
    counts = []
    for k, g in enumerate(genes.itertuples()):
        start0 = max(0, int(g.tss) - 3000)
        end0 = int(g.tss) + 3000
        variants, dosage = [], []
        for rec in vcf.fetch(str(g.chr), start0, end0):
            if rec.pos - 1 < start0 or rec.pos - 1 >= end0:
                continue
            if rec.alts is None or len(rec.alts) != 1 or len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                continue
            if rec.ref not in 'ACGT' or rec.alts[0] not in 'ACGT':
                continue
            variants.append(f'{rec.contig}:{rec.pos}:{rec.ref}:{rec.alts[0]}')
            ds = []
            for s in samples:
                gt = rec.samples[s].get('GT')
                ds.append(sum(gt) if gt is not None and len(gt) == 2 and all(a in (0, 1) for a in gt) else -1)
            dosage.append(ds)
        assert len(variants) == len(set(variants)), g.gene_id
        x = np.array(dosage, dtype=np.int8).reshape(len(variants), 300).T
        np.savez_compressed(GENO / f'{g.gene_id}.npz', dosage=x, variants=np.asarray(variants, dtype=str))
        counts.append(dict(gene_id=g.gene_id, chr=g.chr, tss=int(g.tss), start0=start0, end0=end0, snps=len(variants)))
        if (k + 1) % 50 == 0:
            print('SNP_PREPARE', k + 1, 400, 'seconds', round(time.time() - start, 1), flush=True)
    people.to_csv(GENO / 'individuals.csv', index=False)
    genes.to_csv(GENO / 'genes.csv', index=False)
    pd.DataFrame(counts).to_csv(GENO / 'snp_counts.csv', index=False)
    js(GENO / 'COMPLETE.json', dict(
        status='complete', genes=400, individuals=300, seconds=time.time() - start, vcf=str(VCF),
        coordinate_rule='pysam.fetch(chrom, max(0, tss-3000), tss+3000), 0-based half-open',
        variant_type='biallelic A/C/G/T SNPs; GT ALT dosage 0/1/2; missing -1',
        filtering='train-fold missing<=5%, MAF>=1%, positive variance applied during fitting',
    ))
    print('SNP_PREPARE_COMPLETE', round(time.time() - start, 1), flush=True)


def fit_gene(gene, y, tr, va, verify=False):
    """Fit every candidate (Top SNP, ridge path, LASSO/ElasticNet paths) for one gene on the training individuals."""
    with threadpool_limits(limits=1):
        data = np.load(GENO / f'{gene}.npz')
        raw = data['dosage'].astype(np.float64)
        variants = data['variants']
        raw[raw < 0] = np.nan
        train = raw[tr]
        missing = np.isnan(train).mean(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            means = np.nanmean(train, axis=0)
        freq = means / 2
        mask = (missing <= .05) & (freq >= .01) & (freq <= .99) & np.isfinite(means)
        keep = np.flatnonzero(mask)
        raw = raw[:, keep]
        means = means[keep]
        x = np.where(np.isnan(raw), means, raw)
        scale = x[tr].std(0)
        good = scale > 0
        keep = keep[good]
        means = means[good]
        scale = scale[good]
        x = (x[:, good] - means) / scale
        ym = float(y[tr].mean())
        ys = float(y[tr].std())
        ys = ys if ys > 0 else 1.
        target = (y[tr] - ym) / ys
        xx = np.asfortranarray(x[tr])
        n, p, nc = len(tr), len(keep), len(CANDIDATES)
        coefs = np.zeros((p, nc))
        penalties = np.zeros(nc)
        dualgaps = np.zeros(nc)
        iters = np.zeros(nc, dtype=int)
        top = -1
        check_error = 0.
        top_regressor = None
        if p:
            cross = xx.T @ target / n
            top = int(np.argmax(np.abs(cross)))
            top_regressor = LinearRegression(fit_intercept=False).fit(xx[:, [top]], target)
            coefs[top, 0] = float(top_regressor.coef_[0])
            u, s, vt = np.linalg.svd(xx, full_matrices=False)
            uy = u.T @ target
            for j, a in enumerate(RIDGE_ALPHAS, 1):
                coefs[:, j] = vt.T @ ((s / (s * s + a)) * uy)
                penalties[j] = a
            if verify:
                ref = Ridge(alpha=float(RIDGE_ALPHAS[4]), fit_intercept=False, solver='svd').fit(xx, target)
                check_error = float(np.max(np.abs(ref.coef_ - coefs[:, 5])))
                assert check_error < 1e-8
            offset = 1 + len(RIDGE_ALPHAS)
            for ratio in RATIOS:
                alpha_max = float(np.max(np.abs(cross))) / ratio
                aa = np.maximum(alpha_max * FRACTIONS, 1e-12)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    _, cc, gaps, steps = enet_path(xx, target, l1_ratio=ratio, alphas=aa, max_iter=10000, tol=1e-5,
                                                   precompute=True, return_n_iter=True)
                    if np.max(gaps) > 1.1e-5:
                        _, cc, gaps, steps = enet_path(xx, target, l1_ratio=ratio, alphas=aa, max_iter=100000,
                                                       tol=1e-5, precompute=True, return_n_iter=True)
                coefs[:, offset:offset + len(aa)] = cc
                penalties[offset:offset + len(aa)] = aa
                dualgaps[offset:offset + len(aa)] = gaps
                iters[offset:offset + len(aa)] = steps
                offset += len(aa)
        pred = (x @ coefs) * ys + ym
        vals = np.repeat(y[va][None, :], nc, axis=0)
        pr = corr(vals, pred[va].T)
        mse = np.mean((pred[va] - y[va, None]) ** 2, axis=0)
        return dict(gene_id=gene, keep=keep, x_mean=means, x_scale=scale, y_mean=ym, y_scale=ys, coefs=coefs, pred=pred,
                    val_pearson=pr, val_mse=mse, alpha=penalties, dual_gaps=dualgaps, iterations=iters, snp_count=p,
                    top_variant=str(variants[keep[top]]) if top >= 0 else None, top_index=top,
                    top_snp_estimator=top_regressor, verified_ridge_error=check_error)


def run(fold, n_jobs):
    out = OUT / 'snp_6kb' / fold
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        return
    start = time.time()
    people = pd.read_csv(GENO / 'individuals.csv')
    genes = pd.read_csv(GENO / 'genes.csv').gene_id.tolist()
    rows = pd.read_csv(CANON / 'rows.csv.gz')
    y = np.load(CANON / 'labels.npy')
    trrows, varows, groups = fold_indices(rows, fold)
    groups = {k: v for k, v in groups.items() if k != 'unseen_genes_unseen_individuals'}
    allow = np.ones(len(people), bool) if fold == 'mixed' else people.breed.ne(fold.removeprefix('lobo_')).to_numpy()
    tr = np.flatnonzero(allow & people.split.eq('train').to_numpy())
    va = np.flatnonzero(allow & people.split.eq('val').to_numpy())
    assert len(tr) == (210 if fold == 'mixed' else 140) and len(va) == (30 if fold == 'mixed' else 20)
    seenrows = np.flatnonzero(rows.gene_split.eq('seen').to_numpy())
    assert np.array_equal(rows.iloc[seenrows].gene_id.to_numpy(), np.repeat(genes, 300))
    assert (rows.iloc[seenrows].subject_id.to_numpy().reshape(400, 300) == people.subject_id.to_numpy()[None, :]).all()
    yy = y[seenrows].reshape(400, 300)

    def jobs():
        for k, g in enumerate(genes):
            yield joblib.delayed(fit_gene)(g, yy[k], tr, va, k < 3)

    fits = []
    for f in joblib.Parallel(n_jobs=n_jobs, return_as='generator')(jobs()):
        fits.append(f)
        if len(fits) % 50 == 0:
            print('SNP_TRAIN', fold, len(fits), 400, 'seconds', round(time.time() - start, 1), flush=True)

    pr = np.array([f['val_pearson'] for f in fits])
    mse = np.array([f['val_mse'] for f in fits])
    gap = np.array([f['dual_gaps'] for f in fits])
    search = pd.DataFrame(CANDIDATES)
    search['candidate'] = np.arange(len(CANDIDATES))
    search['validation_mean_gene_pearson'] = pr.mean(0)
    search['validation_mse'] = mse.mean(0)
    search['max_dual_gap'] = gap.max(0)
    search['all_genes_converged'] = (gap <= 1.1e-5).all(0)
    selected = {}
    for model, z in search.groupby('model', sort=False):
        z = z[z.all_genes_converged]
        assert len(z), model
        selected[model] = int(z.sort_values(['validation_mean_gene_pearson', 'validation_mse'],
                                            ascending=[False, True]).iloc[0].candidate)
    search.to_csv(out / 'validation_search.csv', index=False)
    js(out / 'selection.json', {k: {**CANDIDATES[v], 'candidate': v, 'validation_mean_gene_pearson': float(pr[:, v].mean())}
                               for k, v in selected.items()})

    outputs, models, toprows = [], [], []
    maxerr = 0.
    for f in fits:
        d = np.load(GENO / f'{f["gene_id"]}.npz')
        raw = d['dosage'][:, f['keep']].astype(float)
        raw[raw < 0] = np.nan
        x = (np.where(np.isnan(raw), f['x_mean'], raw) - f['x_mean']) / f['x_scale']
        weights = {name: f['coefs'][:, idx] for name, idx in selected.items()}
        for name, idx in selected.items():
            p = x @ weights[name] * f['y_scale'] + f['y_mean']
            error = float(np.max(np.abs(p - f['pred'][:, idx])))
            assert error < 1e-9
            maxerr = max(maxerr, error)
        models.append({k: f[k] for k in ['gene_id', 'keep', 'x_mean', 'x_scale', 'y_mean', 'y_scale', 'snp_count']})
        models[-1].update(top_snp_estimator=f['top_snp_estimator'], top_index=f['top_index'], weights=weights,
                          variant_ids=d['variants'][f['keep']],
                          alpha={name: float(f['alpha'][idx]) for name, idx in selected.items()})
        j = f['top_index']
        beta = float(f['coefs'][j, 0] * f['y_scale'] / f['x_scale'][j]) if j >= 0 else 0.
        intercept = f['y_mean'] - beta * f['x_mean'][j] if j >= 0 else f['y_mean']
        toprows.append(dict(gene_id=f['gene_id'], variant_id=f['top_variant'], training_snps=f['snp_count'],
                            dosage_beta=beta, intercept=float(intercept), training_individuals=len(tr)))
    for model, idx in selected.items():
        p = np.full(len(rows), np.nan)
        p[seenrows] = np.concatenate([f['pred'][:, idx] for f in fits])
        outputs.append((model, -1, p))

    joblib.dump(dict(fold=fold, target=TARGET, training_subject_ids=people.iloc[tr].subject_id.tolist(),
                     validation_subject_ids=people.iloc[va].subject_id.tolist(), selection=selected,
                     candidates=CANDIDATES, genes=models), out / 'models.joblib', compress=3)
    loaded = joblib.load(out / 'models.joblib')
    assert len(loaded['genes']) == 400
    for before, after in zip(models, loaded['genes']):
        for name in selected:
            assert np.array_equal(before['weights'][name], after['weights'][name])
    pd.DataFrame(toprows).to_csv(out / 'top_snps.csv', index=False)
    save_evaluations(out, rows, y, groups, outputs)
    js(out / 'COMPLETE.json', dict(
        status='complete', seconds=time.time() - start, fold=fold, target=TARGET, genes=400,
        training_individuals=len(tr), validation_individuals=len(va),
        no_eligible_snp_genes=sum(f['snp_count'] == 0 for f in fits),
        max_prediction_reconstruction_error=maxerr,
        max_independent_ridge_error=max(f['verified_ridge_error'] for f in fits),
        selection='One validation-selected hyperparameter setting per method per fold; each gene fitted separately',
        top_snp_implementation='sklearn.linear_model.LinearRegression; training-centered inputs and response; raw intercept restored',
        selected_models=selected,
        selected_max_dual_gap={name: float(gap[:, idx].max()) for name, idx in selected.items()},
        unseen_genes='N/A: no unseen-gene expression labels used for fitting or variant selection',
    ))
    print('SNP_FOLD_COMPLETE', fold, round(time.time() - start, 1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--folds', nargs='*', choices=FOLDS)
    p.add_argument('--jobs', type=int, default=8)
    a = p.parse_args()
    if a.prepare:
        prepare()
    for f in a.folds or []:
        run(f, a.jobs)
