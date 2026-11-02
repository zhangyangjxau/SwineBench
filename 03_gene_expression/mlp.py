#!/usr/bin/env python3
"""Train a fixed MLP head (width 64, five seeds) on frozen mean-pooled embeddings; mixed and single-source-breed folds."""
import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'processed'
CANON = DATA / 'labels'
FEATURES = DATA / 'features'
OUT = ROOT / 'results'
DIMS = {'ntv2': 1024, 'dnabert2': 768, 'genalm': 768, 'hyenadna': 256, 'evo2': 4096, 'generator': 3072}
BREEDS = ['Duroc', 'Landrace', 'LargeWhite']
FOLDS = ['mixed'] + ['source_' + b for b in BREEDS]
CONFIG = dict(width=64, lr=.001, dropout=.1, weight_decay=.01)
SEEDS = [42, 13, 17, 123, 997]
TARGET = 'author_corrected'


def js(p, obj):
    p.write_text(json.dumps(obj, indent=2))


def corr(a, b):
    """Row-wise Pearson r; 0 where either row is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aa = a - a.mean(1, keepdims=True)
    bb = b - b.mean(1, keepdims=True)
    den = np.sqrt((aa * aa).sum(1) * (bb * bb).sum(1))
    valid = (np.ptp(a, axis=1) > 0) & (np.ptp(b, axis=1) > 0) & (den > 0)
    return np.divide((aa * bb).sum(1), den, out=np.zeros(len(a)), where=valid)


def fold_indices(rows, fold):
    assert fold in FOLDS
    seen = rows.gene_split.eq('seen').to_numpy()
    if fold == 'mixed':
        tr = np.flatnonzero(seen & rows.split.eq('train').to_numpy())
        va = np.flatnonzero(seen & rows.split.eq('val').to_numpy())
        groups = {
            'unseen_individuals': np.flatnonzero(seen & rows.split.eq('test').to_numpy()),
            'unseen_genes_unseen_individuals': np.flatnonzero(~seen & rows.split.eq('test').to_numpy()),
        }
        assert len(tr) == 84000 and len(va) == 12000
        assert len(groups['unseen_individuals']) == 24000 and len(groups['unseen_genes_unseen_individuals']) == 40200
        assert rows.iloc[tr].subject_id.nunique() == 210 and rows.iloc[va].subject_id.nunique() == 30
        assert set(rows.iloc[tr].gene_id).isdisjoint(set(rows.iloc[groups['unseen_genes_unseen_individuals']].gene_id))
        for ii in [va, *groups.values()]:
            assert set(rows.iloc[tr].subject_id).isdisjoint(set(rows.iloc[ii].subject_id))
        for ii in groups.values():
            assert set(rows.iloc[va].subject_id).isdisjoint(set(rows.iloc[ii].subject_id))
            assert rows.iloc[ii].subject_id.nunique() == 60
        return tr, va, groups

    source = fold.removeprefix('source_')
    allow = rows.breed.eq(source).to_numpy()
    tr = np.flatnonzero(seen & allow & rows.split.eq('train').to_numpy())
    va = np.flatnonzero(seen & allow & rows.split.eq('val').to_numpy())
    groups = {
        source + '_to_' + target: np.flatnonzero(seen & rows.breed.eq(target).to_numpy())
        for target in BREEDS if target != source
    }
    assert len(tr) == 28000 and len(va) == 4000
    assert rows.iloc[tr].subject_id.nunique() == 70 and rows.iloc[va].subject_id.nunique() == 10
    assert set(rows.iloc[tr].breed) == {source} and set(rows.iloc[va].breed) == {source}
    assert not set(rows.iloc[tr].subject_id) & set(rows.iloc[va].subject_id)
    for scenario, ii in groups.items():
        target = scenario.split('_to_')[1]
        assert len(ii) == 40000 and rows.iloc[ii].subject_id.nunique() == 100
        assert set(rows.iloc[ii].breed) == {target} and target != source
        assert not set(rows.iloc[tr].subject_id) & set(rows.iloc[ii].subject_id)
        assert not set(rows.iloc[va].subject_id) & set(rows.iloc[ii].subject_id)
        assert set(rows.iloc[ii].gene_id) == set(rows.iloc[tr].gene_id)
    return tr, va, groups


def layout(rows, ii):
    """Rows of a scenario are gene-major blocks of the same individuals; return (genes, n_individuals)."""
    d = rows.iloc[ii]
    counts = d.groupby('gene_id').size()
    assert counts.nunique() == 1
    n = int(counts.iloc[0])
    g = counts.index.to_numpy()
    assert np.array_equal(d.gene_id.to_numpy(), np.repeat(g, n))
    ss = d.subject_id.to_numpy().reshape(len(g), n)
    assert (ss == ss[:1]).all()
    return g, n


def score(y, p, g, n):
    return float(corr(y.reshape(len(g), n), p.reshape(len(g), n)).mean()), float(np.mean((y - p) ** 2))


def evaluate(rows, ii, y, p, scenario, model, seed):
    g, n = layout(rows, ii)
    yy = y[ii].reshape(len(g), n)
    pp = np.asarray(p).reshape(len(g), n)
    pr = corr(yy, pp)
    sr = corr(rankdata(yy, axis=1), rankdata(pp, axis=1))
    cp = np.ptp(pp, axis=1) == 0
    cy = np.ptp(yy, axis=1) == 0
    valid = ~(cp | cy)
    rng = np.random.default_rng(20260905)
    boot = pr[rng.integers(len(g), size=(1000, len(g)))].mean(1)
    mse = float(np.mean((yy - pp) ** 2))
    var = float(np.var(yy))
    sm = dict(
        scenario=scenario, model=model, seed=seed, genes=len(g), individuals=n,
        mean_gene_pearson=float(pr.mean()), median_gene_pearson=float(np.median(pr)),
        mean_gene_pearson_defined=float(pr[valid].mean()) if valid.any() else None,
        pearson_ci_low=float(np.quantile(boot, .025)), pearson_ci_high=float(np.quantile(boot, .975)),
        mean_gene_spearman=float(sr.mean()),
        constant_prediction_genes=int(cp.sum()), constant_target_genes=int(cy.sum()),
        mean_individual_cross_gene_pearson=float(corr(yy.T, pp.T).mean()),
        mean_individual_cross_gene_spearman=float(corr(rankdata(yy.T, axis=1), rankdata(pp.T, axis=1)).mean()),
        rmse=math.sqrt(mse), r2=1 - mse / var,
    )
    dt = pd.DataFrame(dict(
        scenario=scenario, model=model, seed=seed, gene_id=g, pearson=pr, spearman=sr,
        pearson_defined=np.where(valid, pr, np.nan), constant_prediction=cp, constant_target=cy,
        observed_sd=yy.std(1), predicted_sd=pp.std(1),
    ))
    return sm, dt


def save_evaluations(out, rows, y, groups, outputs):
    metrics, details, preds = [], [], []
    for model, seed, p in outputs:
        for scenario, ii in groups.items():
            assert np.isfinite(p[ii]).all()
            sm, dt = evaluate(rows, ii, y, p[ii], scenario, model, seed)
            metrics.append(sm)
            details.append(dt)
            d = rows.iloc[ii][['gene_id', 'subject_id', 'breed']].copy()
            d['observed'] = y[ii]
            d['predicted'] = p[ii]
            d['scenario'] = scenario
            d['model'] = model
            d['seed'] = seed
            preds.append(d)
            if scenario in ('unseen_individuals', 'unseen_genes_unseen_individuals'):
                for breed in BREEDS:
                    jj = ii[rows.iloc[ii].breed.eq(breed).to_numpy()]
                    sm, dt = evaluate(rows, jj, y, p[jj], scenario + '_within_' + breed, model, seed)
                    metrics.append(sm)
                    details.append(dt)
    metrics = pd.DataFrame(metrics)
    details = pd.concat(details)
    predframe = pd.concat(preds)
    metrics.to_csv(out / 'metrics.csv', index=False)
    details.to_csv(out / 'gene_metrics.csv.gz', index=False)
    predframe.to_csv(out / 'predictions.csv.gz', index=False)
    for scenario in groups:
        d = out / 'evaluations' / scenario
        d.mkdir(parents=True, exist_ok=True)
        metrics.query('scenario == @scenario').to_csv(d / 'metrics.csv', index=False)
        details.query('scenario == @scenario').to_csv(d / 'gene_metrics.csv.gz', index=False)
        predframe[predframe.scenario.eq(scenario)].to_csv(d / 'predictions.csv.gz', index=False)


def head(dim, c):
    return nn.Sequential(nn.Linear(dim, c['width']), nn.GELU(), nn.Dropout(c['dropout']), nn.Linear(c['width'], 1))


def run(model, fold, device):
    out = OUT / model / fold
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        return
    start = time.time()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False

    rows = pd.read_csv(CANON / 'rows.csv.gz')
    y = np.load(CANON / 'labels.npy')
    tr, va, groups = fold_indices(rows, fold)
    rawx = np.load(FEATURES / model / 'embeddings.npy', mmap_mode='r')
    assert rawx.shape == (len(rows), DIMS[model])

    scaler = StandardScaler().fit(rawx[tr])
    x = scaler.transform(rawx).astype(np.float32)
    ym = float(y[tr].mean())
    ys = float(y[tr].std())
    yn = ((y - ym) / ys).astype(np.float32)
    xt = torch.tensor(x, device=device)
    yt = torch.tensor(yn, device=device)
    tri = torch.tensor(tr, device=device)
    vai = torch.tensor(va, device=device)
    del x
    g, n = layout(rows, va)
    norm = dict(
        x_mean=torch.tensor(scaler.mean_), x_scale=torch.tensor(scaler.scale_), y_mean=ym, y_scale=ys,
        input_dim=DIMS[model],
        train_subject_ids=sorted(set(rows.iloc[tr].subject_id)),
        validation_subject_ids=sorted(set(rows.iloc[va].subject_id)),
        train_gene_ids=sorted(set(rows.iloc[tr].gene_id)),
    )
    hist = []

    def fit(c, seed, candidate):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        net = head(DIMS[model], c).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        best, state, stale, bestepoch = None, None, 0, 0
        t = time.time()
        for epoch in range(1, 101):
            net.train()
            perm = tri[torch.randperm(len(tri), device=device)]
            total = torch.zeros((), device=device)
            for ii in perm.split(512):
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.smooth_l1_loss(net(xt[ii]).squeeze(-1), yt[ii])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.)
                opt.step()
                total += loss.detach() * len(ii)
            trainloss = float(total) / len(tr)
            assert np.isfinite(trainloss)
            net.eval()
            with torch.no_grad():
                vp = net(xt[vai]).squeeze(-1).cpu().numpy() * ys + ym
            pr, mse = score(y[va], vp, g, n)
            hist.append(dict(candidate=candidate, seed=seed, epoch=epoch, train_smoothl1=trainloss,
                             val_gene_pearson=pr, val_mse=mse))
            if best is None or pr > best[0] + 1e-5 or (abs(pr - best[0]) <= 1e-5 and mse < -best[1] - 1e-6):
                best = (pr, -mse)
                state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                stale = 0
                bestepoch = epoch
            else:
                stale += 1
            if epoch % 25 == 0:
                print('TRAIN', model, fold, candidate, seed, epoch, 'val_Pearson', round(pr, 4),
                      'seconds', round(time.time() - t, 1), flush=True)
            if stale >= 15:
                break
        pd.DataFrame(hist).to_csv(out / 'training_history.csv', index=False)
        print('HEAD_DONE', model, fold, candidate, seed, bestepoch, best[0], flush=True)
        return dict(config=c, seed=seed, state=state, best_epoch=bestepoch,
                    validation_pearson=best[0], validation_mse=-best[1])

    js(out / 'fixed_config.json', dict(config=CONFIG, seeds=SEEDS,
                                       selection='No configuration search; checkpoint chosen on validation Pearson'))
    fitted = [fit(CONFIG, seed, 'fixed') for seed in SEEDS]
    outputs, checks, checkpoint_info = [], [], []
    for f in fitted:
        ck = {**norm, **{k: v for k, v in f.items() if k != 'state'}, 'head_state_dict': f['state'],
              'model': model, 'fold': fold, 'target': TARGET}
        path = out / f'mlp_seed{f["seed"]}.pt'
        torch.save(ck, path)
        loaded = torch.load(path, map_location=device, weights_only=True)
        net = head(DIMS[model], loaded['config']).to(device)
        net.load_state_dict(loaded['head_state_dict'])
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(b).squeeze(-1) for b in xt.split(8192)]).cpu().numpy() * ys + ym
        # Recompute a few held-out predictions from the raw input and the saved preprocessing.
        cpu = head(DIMS[model], f['config'])
        cpu.load_state_dict(f['state'])
        cpu.eval()
        for scenario, indices in groups.items():
            checkidx = indices[:20]
            rx = np.array(rawx[checkidx], copy=True)
            rx -= scaler.mean_
            rx /= scaler.scale_
            with torch.no_grad():
                p2 = cpu(torch.tensor(rx)).squeeze(-1).numpy() * ys + ym
            err = float(np.max(np.abs(p2 - pred[checkidx])))
            assert err < 1e-3
            checks.append(dict(seed=f['seed'], scenario=scenario, rows=20, max_absolute_error=err))
        outputs.append((model, f['seed'], pred))
        checkpoint_info.append({k: v for k, v in f.items() if k != 'state'})
    outputs.append((model + '_ensemble', -1, np.mean([p for _, _, p in outputs], axis=0)))
    save_evaluations(out, rows, y, groups, outputs)
    js(out / 'checkpoint_verification.json', checks)
    js(out / 'COMPLETE.json', dict(
        status='complete', seconds=time.time() - start, model=model, fold=fold, target=TARGET,
        selection='Fixed configuration; checkpoint selected by validation mean per-gene Pearson',
        config=CONFIG, seeds=SEEDS, checkpoints=checkpoint_info,
    ))
    print('FOLD_COMPLETE', model, fold, round(time.time() - start, 1), flush=True)
    del xt, yt, tri, vai
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=list(DIMS))
    p.add_argument('--folds', nargs='+', default=FOLDS, choices=FOLDS)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    for f in a.folds:
        run(a.model, f, a.device)
