"""
explore_saturation.py: EXPLORATORY test of the saturation explanation for the KRTAP failures (outside the pre-registered plan).
Prediction written before running (2026-11-14): in the keratin-associated protein (KRTAP) slabs, the total dense-phase contacts per
residue, T_i = X_i + W_i (between-chain plus within-chain), vary little across residues (low coefficient of variation) and X_i and W_i
anti-correlate across residues; in other slabs T_i varies more and X_i and W_i rise together. Across slabs, H1 should fall as the
coefficient of variation of T falls.

Contacts, slab centre and dense half-width are computed as in slab_test.analyse. The script first checks on two slabs that its X equals
the X that slab_test.analyse passes to its H1 test (captured from that call), and stops if they differ.
  ~/.venvs/finches/bin/python explore_saturation.py   ->   ../results/saturation/{per_slab.csv, summary.txt}
"""
import csv
import os
import sys

import mdtraj as md
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results', 'saturation')
sys.path.insert(0, HERE)
import slab_test as S  # noqa: E402

END = 5   # residues this close to a chain end are left out of the spread of T (ends have fewer neighbours by construction)


def profiles(slab):
    seq, n, name = slab['seq'], slab['N'], slab['name']
    dcd = os.path.join(S.FR, slab['set'], name, f'{name}_sub.dcd')
    natoms = md.formats.DCDTrajectoryFile(dcd).read(n_frames=1)[0].shape[1]
    chains = natoms // n
    top = md.Topology()
    for _ in range(chains):
        ch = top.add_chain()
        for _ in seq:
            top.add_atom('CA', md.element.carbon, top.add_residue('ALA', ch))
    t = md.load_dcd(dcd, top=top)
    box = t.unitcell_lengths[0].astype(float)
    L = box[2]
    dzs = []
    for f in range(t.n_frames):
        cz = np.mod(t.xyz[f, :, 2], L).reshape(chains, n).mean(1)
        zc = (np.angle(np.exp(2j * np.pi * cz / L).mean()) % (2 * np.pi)) * L / (2 * np.pi)
        dzs.append(np.abs((cz - zc + L / 2) % L - L / 2))
    dzs = np.array(dzs)
    hist = np.histogram(dzs.ravel(), bins=np.arange(0, L / 2 + 1, 1.0))[0].astype(float)
    half = int(np.argmax(hist < hist.max() / 2)) if (hist < hist.max() / 2).any() else len(hist)
    half_width = float(max(half, 1))
    X, W, nd = np.zeros(n), np.zeros(n), 0
    for f in range(t.n_frames):
        x = np.mod(t.xyz[f].astype(float), box)
        pr = cKDTree(x, boxsize=box).query_pairs(S.CUT, output_type='ndarray')
        a, b = pr[:, 0], pr[:, 1]
        ca, cb, ia, ib = a // n, b // n, a % n, b % n
        same = ca == cb
        intra = same & (np.abs(ia - ib) >= 3)
        dense = dzs[f] < half_width
        cnt = lambda sel: np.bincount(ia[sel & dense[ca]], minlength=n) + np.bincount(ib[sel & dense[cb]], minlength=n)  # noqa: E731
        X += cnt(~same)
        W += cnt(intra)
        nd += int(dense.sum())
    return X / max(nd, 1), W / max(nd, 1)


def check(slab, pred, lam):
    captured = {}
    real = S.partial

    def rec(x, y, Z):
        captured.setdefault('X', np.array(y))
        return real(x, y, Z)

    S.partial = rec
    S.analyse(slab, pred, lam)
    S.partial = real
    X, _ = profiles(slab)
    diff = np.abs(X - captured['X']).max()
    print(f'check {slab["name"]}: max |X - X(slab_test)| = {diff:.2e}', flush=True)
    if diff > 1e-9:
        raise SystemExit('contact definitions differ from slab_test; stopping')


def main():
    os.makedirs(OUT, exist_ok=True)
    pred = np.load(os.path.join(S.OUT, 'pred_degree.npz'))
    lam = {r['one']: float(r['lambdas']) for r in csv.DictReader(open(S.CAL))}
    h1 = {r['name']: r for r in csv.DictReader(open(os.path.join(S.OUT, 'per_slab.csv')))}
    tab = {}
    for f in S.SETS.values():
        for r in csv.DictReader(open(os.path.join(S.TAB, f))):
            tab[r[list(r)[0]]] = r
    todo = [s for s in S.slabs() if s['tier'] in ('primary', 'replication') and s['name'] in h1
            and S.num(h1[s['name']]['ratio_ours']) > 10]
    for s in todo[:2]:
        check(s, pred[s['name']], lam)
    rows = []
    for s in todo:
        X, W = profiles(s)
        T = X + W
        mid = np.minimum(np.arange(s['N']), s['N'] - 1 - np.arange(s['N'])) >= END
        krtap = 'KRTAP' in (tab[s['name']].get('gene_name') or '')
        row = {'name': s['name'], 'tier': s['tier'], 'krtap': krtap, 'N': s['N'], 'h1': S.num(h1[s['name']]['h1_partial']),
               'dG': S.num(tab[s['name']].get('dG')), 'T_mean': T[mid].mean(), 'T_cv': T[mid].std() / T[mid].mean(),
               'X_share': X.sum() / T.sum(), 'rho_X_W': spearmanr(X[mid], W[mid]).correlation}
        rows.append(row)
        print(f"{s['name']:22s} KRTAP {str(krtap):5s} H1 {row['h1']:+.3f}  T mean {row['T_mean']:.2f} cv {row['T_cv']:.3f}  rho(X,W) {row['rho_X_W']:+.3f}", flush=True)
    with open(os.path.join(OUT, 'per_slab.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    k = np.array([r['krtap'] for r in rows])
    g = lambda key: np.array([r[key] for r in rows])  # noqa: E731
    L = [f'Exploratory saturation check, {len(rows)} phase-separated slabs ({k.sum()} KRTAP)', '']
    for key, label in (('T_cv', 'spread of total contacts per residue (coefficient of variation)'), ('T_mean', 'mean total contacts per residue'),
                       ('rho_X_W', 'Spearman of between-chain and within-chain contacts across residues'), ('X_share', 'share of contacts between chains')):
        a, b = g(key)[k], g(key)[~k]
        L.append(f'{label}: KRTAP median {np.median(a):+.3f}, others {np.median(b):+.3f} (Mann-Whitney P {mannwhitneyu(a, b).pvalue:.2g})')
    L += ['', 'across all slabs:',
          f'  Spearman(H1, spread of T) {spearmanr(g("h1"), g("T_cv")).correlation:+.3f}',
          f'  Spearman(H1, mean T) {spearmanr(g("h1"), g("T_mean")).correlation:+.3f}',
          f'  Spearman(H1, rho(X,W)) {spearmanr(g("h1"), g("rho_X_W")).correlation:+.3f}',
          f'  Spearman(mean T, dG) {spearmanr(g("T_mean"), g("dG"), nan_policy="omit").correlation:+.3f}',
          f'  slabs with negative rho(X,W): {(g("rho_X_W") < 0).sum()} ({(g("rho_X_W")[k] < 0).sum()} KRTAP)']
    txt = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'summary.txt'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
