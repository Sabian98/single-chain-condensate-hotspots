"""
explore_degree_descriptors.py: EXPLORATORY, outside the pre-registered plan (the An et al. data were already seen in Test B).
Question: does the full predicted degree profile carry more information on condensate dynamics at matched B2 than its mean?

Data: results/amendment4/test_b_per_sequence.csv (508 phase-separating sequences with D) and pred_degree_b.npz (per-residue
predicted single-chain degree, keyed by line index of seq_heteromeric.txt).
Model: ridge regression of log D on standardized features, the penalty chosen by closed-form leave-one-out error inside the
training fold. Folds = active-learning generations (leave one generation out), so related designs never sit on both sides.
Every feature set includes log(-B2) and N. Out-of-fold predictions are scored by
  (1) partial Spearman of the prediction with log D beyond log(-B2) and N (how much of the spread left by B2 is captured), and
  (2) matched-B2 pair accuracy (|B2| within 5%, D differing more than 1.5-fold), the degeneracy question.
  ~/.venvs/finches/bin/python explore_degree_descriptors.py   ->   ../results/amendment4/explore_descriptors.txt
"""
import csv
import os

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'results', 'amendment4')
ALPHAS = 10.0 ** np.arange(-3, 4.1, 0.5)


def profile_features(deg, seq):
    """Scalar descriptors of one predicted degree profile. deg: per-residue predicted degree (length N), seq: sequence string.
    Returns {name: value}. Keep the count modest (508 sequences) and prefer per-residue normalized quantities, since N runs 20-50."""
    f = {'deg_mean': deg.mean()}
    # TODO(human): add descriptors that capture what the mean throws away
    return f


def comp(seq, aas):
    return sum(c in aas for c in seq) / len(seq)


def load():
    rows = list(csv.DictReader(open(os.path.join(OUT, 'test_b_per_sequence.csv'))))
    z = np.load(os.path.join(OUT, 'pred_degree_b.npz'))
    recs = []
    for r in rows:
        seq, deg = r['seq'], z[r['index']]
        assert len(deg) == len(seq), r['index']
        hyd = {'lambda_mean': float(r['lambda_mean']), 'SHD': float(r['SHD']), 'SCD': float(r['SCD']),
               'FWY': comp(seq, 'FWY'), 'RK': comp(seq, 'RK'), 'DE': comp(seq, 'DE'), 'ILMV': comp(seq, 'ILMV'),
               'GSPNQ': comp(seq, 'GSPNQ')}
        recs.append({'logD': np.log(float(r['D'])), 'lb': float(r['log_neg_B2']), 'N': float(r['N']),
                     'gen': int(r['generation']), 'hyd': hyd, 'prof': profile_features(np.asarray(deg, float), seq)})
    return recs


def ridge_fit(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    ym = y.mean()
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    best = None
    for a in ALPHAS:                                            # closed-form leave-one-out error for each penalty
        d = s / (s ** 2 + a)
        w = Vt.T @ (d * (U.T @ (y - ym)))
        h = ((U * (s * d)) ** 2).sum(1) + 1 / len(y)
        loo = np.mean(((y - ym - Xs @ w) / (1 - h)) ** 2)
        if best is None or loo < best[0]:
            best = (loo, w)
    return lambda Xn: ym + ((Xn - mu) / sd) @ best[1]


def oof(X, y, gen):
    pred = np.empty_like(y)
    for g in np.unique(gen):
        te = gen == g
        pred[te] = ridge_fit(X[~te], y[~te])(X[te])
    return pred


def partial(x, y, Z):
    R = rankdata
    A = np.column_stack([np.ones(len(x))] + [R(c) for c in Z])
    res = lambda v: R(v) - A @ np.linalg.lstsq(A, R(v), rcond=None)[0]  # noqa: E731
    return np.corrcoef(res(x), res(y))[0, 1]


def main():
    recs = load()
    y = np.array([r['logD'] for r in recs])
    lb, N, gen = (np.array([r[k] for r in recs]) for k in ('lb', 'N', 'gen'))
    hyd_names = list(recs[0]['hyd'])
    prof_names = list(recs[0]['prof'])
    H = np.array([[r['hyd'][k] for k in hyd_names] for r in recs])
    P = np.array([[r['prof'][k] for k in prof_names] for r in recs])
    base = np.column_stack([lb, N])
    sets = {'B2 and N only': base,
            '+ hydropathy and composition': np.column_stack([base, H]),
            '+ mean degree': np.column_stack([base, P[:, [prof_names.index('deg_mean')]]]),
            '+ degree profile': np.column_stack([base, P]),
            '+ hydropathy, composition and degree profile': np.column_stack([base, H, P])}
    i, j = np.triu_indices(len(y), 1)
    m = (np.abs(lb[i] - lb[j]) < np.log(1.05)) & (np.abs(y[i] - y[j]) > np.log(1.5))
    i, j = i[m], j[m]
    L = ['Exploratory: does the degree profile carry more than its mean? (An et al. 2024 data, 508 sequences)',
         f'profile descriptors ({len(prof_names)}): {", ".join(prof_names)}',
         f'folds: leave one generation out ({len(np.unique(gen))}); matched-B2 pairs: {m.sum()}', '',
         f'{"feature set":48s} {"partial beyond B2, N":>21s} {"pair accuracy":>14s}']
    for name, X in sets.items():
        p = oof(X, y, gen)
        acc = np.mean((p[i] < p[j]) == (y[i] < y[j]))
        L.append(f'{name:48s} {partial(p, y, [lb, N]):+21.3f} {acc:14.2f}')
    L += ['', 'each profile descriptor alone, partial Spearman with log D beyond log(-B2) and N (in-sample):']
    for k, c in zip(prof_names, P.T):
        L.append(f'  {k:24s} {partial(c, y, [lb, N]):+.3f}   (Spearman with mean degree {spearmanr(c, P[:, 0]).correlation:+.2f})')
    txt = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'explore_descriptors.txt'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
