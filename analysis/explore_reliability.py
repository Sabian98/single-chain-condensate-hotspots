"""
explore_reliability.py: EXPLORATORY noise ceiling for H1 (outside the pre-registered plan).
Question (2026-11-14): are the low H1 values of flat, saturated slabs (KRTAPs, Q86U02_1_117) a sampling-noise ceiling or real?
Per slab, the between-chain degree X is computed separately on two halves of the 200 frames, each half is reduced to its residual after
the H1 baseline (ranks, as in slab_test.partial), and the two residuals are correlated (split-half reliability r). Spearman-Brown gives
the reliability of the full set, R = 2r / (1 + r), and the highest H1 that noise allows is sqrt(R), since the prediction itself carries
no sampling noise. Two splits are reported: first against second half of the frames (primary, conservative) and even against odd.
If flat slabs have low R, their low H1 is mostly a noise ceiling; if R stays high, the failure is real.

Contacts and dense phase as in slab_test.analyse; the full-set X is checked against slab_test on two slabs before the run.
  ~/.venvs/finches/bin/python explore_reliability.py   ->   ../results/reliability/{per_slab.csv, summary.txt}
"""
import csv
import os
import sys

import mdtraj as md
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results', 'reliability')
sys.path.insert(0, HERE)
import slab_test as S  # noqa: E402


def per_frame_X(slab):
    """Between-chain degree summed per frame and dense-chain count per frame, so any split of frames can be averaged."""
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
    Xf, nd = np.zeros((t.n_frames, n)), np.zeros(t.n_frames)
    for f in range(t.n_frames):
        x = np.mod(t.xyz[f].astype(float), box)
        pr = cKDTree(x, boxsize=box).query_pairs(S.CUT, output_type='ndarray')
        a, b = pr[:, 0], pr[:, 1]
        ca, cb, ia, ib = a // n, b // n, a % n, b % n
        other = ca != cb
        dense = dzs[f] < half_width
        Xf[f] = np.bincount(ia[other & dense[ca]], minlength=n) + np.bincount(ib[other & dense[cb]], minlength=n)
        nd[f] = dense.sum()
    return Xf, nd


def resid(v, Z):
    A = np.column_stack([np.ones(len(v))] + [rankdata(z) for z in Z])
    r = rankdata(v)
    return r - A @ np.linalg.lstsq(A, r, rcond=None)[0]


def reliability(Xf, nd, sel_a, sel_b, Z):
    xa = Xf[sel_a].sum(0) / max(nd[sel_a].sum(), 1)
    xb = Xf[sel_b].sum(0) / max(nd[sel_b].sum(), 1)
    r = spearmanr(resid(xa, Z), resid(xb, Z)).correlation
    R = 2 * r / (1 + r) if r > -1 else float('nan')
    return r, R, (np.sqrt(R) if R > 0 else 0.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    pred = np.load(os.path.join(S.OUT, 'pred_degree.npz'))
    lam = {r['one']: float(r['lambdas']) for r in csv.DictReader(open(S.CAL))}
    h1 = {r['name']: r for r in csv.DictReader(open(os.path.join(S.OUT, 'per_slab.csv')))}
    sat = {r['name']: r for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'saturation', 'per_slab.csv')))}
    todo = [s for s in S.slabs() if s['name'] in sat]
    for s in todo[:2]:                                          # same X as slab_test
        captured = {}
        real = S.partial
        S.partial = lambda x, y, Z, c=captured, real=real: (c.setdefault('X', np.array(y)), real(x, y, Z))[1]
        S.analyse(s, pred[s['name']], lam)
        S.partial = real
        Xf, nd = per_frame_X(s)
        diff = np.abs(Xf.sum(0) / nd.sum() - captured['X']).max()
        print(f'check {s["name"]}: max |X - X(slab_test)| = {diff:.2e}', flush=True)
        if diff > 1e-9:
            raise SystemExit('contact definitions differ from slab_test; stopping')
    rows = []
    for s in todo:
        Xf, nd = per_frame_X(s)
        F = len(nd)
        Z = S.covariates(s['seq'], lam)
        idx = np.arange(F)
        r_blk, R_blk, c_blk = reliability(Xf, nd, idx < F // 2, idx >= F // 2, Z)
        r_eo, R_eo, c_eo = reliability(Xf, nd, idx % 2 == 0, idx % 2 == 1, Z)
        h = S.num(h1[s['name']]['h1_partial'])
        row = {'name': s['name'], 'tier': s['tier'], 'krtap': sat[s['name']]['krtap'], 'h1': h, 'T_cv': S.num(sat[s['name']]['T_cv']),
               'split_r_halves': r_blk, 'ceiling_halves': c_blk, 'split_r_evenodd': r_eo, 'ceiling_evenodd': c_eo,
               'h1_over_ceiling': h / c_blk if c_blk > 0 else float('nan')}
        rows.append(row)
        print(f"{s['name']:22s} KRTAP {row['krtap']:5s} H1 {h:+.3f}  ceiling halves {c_blk:.3f} even/odd {c_eo:.3f}", flush=True)
    with open(os.path.join(OUT, 'per_slab.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    g = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    k = np.array([r['krtap'] == 'True' for r in rows])
    h, c, ce, cv = g('h1'), g('ceiling_halves'), g('ceiling_evenodd'), g('T_cv')
    fail = h <= 0
    L = [f'Exploratory noise ceiling for H1, {len(rows)} slabs ({k.sum()} KRTAP, {fail.sum()} with H1 <= 0)', '',
         f'ceiling (first vs second half): all median {np.median(c):.3f}; KRTAP {np.median(c[k]):.3f}, others {np.median(c[~k]):.3f} '
         f'(Mann-Whitney P {mannwhitneyu(c[k], c[~k]).pvalue:.2g})',
         f'ceiling (even vs odd frames):   all median {np.median(ce):.3f}; KRTAP {np.median(ce[k]):.3f}, others {np.median(ce[~k]):.3f}',
         f'H1 as a fraction of its ceiling: KRTAP median {np.median(h[k] / c[k]):+.2f}, others {np.median(h[~k] / c[~k]):+.2f}',
         f'slabs with H1 <= 0: ceilings {", ".join(f"{x:.2f}" for x in c[fail])}',
         f'Spearman(ceiling, spread of total contacts) {spearmanr(c, cv).correlation:+.3f}',
         f'Spearman(H1, spread) {spearmanr(h, cv).correlation:+.3f}; partial beyond ceiling '
         f'{np.corrcoef(resid(h, [c]), resid(cv, [c]))[0, 1]:+.3f}']
    txt = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'summary.txt'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
