"""
single_chain_test.py: EXPLORATORY (outside the pre-registered plan). Does simulated single-chain degree predict contacts with other
chains in the condensate better than the degree model's prediction? Written 2026-11-14, before any single-chain trajectory was read.

Sources of single-chain degree:
  --idrome   IDRome CALVADOS-2 single chains (Tesei et al. 2024; 310 K, 150 mM), data/idrome_single/<name>/{traj.xtc, top.pdb},
             all 1,000 frames (the first 10 were already discarded by that study)
  --gaivi D  our own single-chain runs at 293 K from the CALVADOS IDP and OpenMPIPI pipelines, D/293K/<name>/<name>.dcd and top.pdb,
             frames evenly spaced over the production half
Single-chain degree follows the rule of the degree model's training labels (BENDER-DYN hub_frames.py): chain rebuilt from
minimum-image bonds, Calpha-Calpha < 0.8 nm, |i - j| >= 3, averaged over frames.
The slab side uses the between-chain degree X and the baseline of slab_test.analyse, cached once by --cache-x.

  ~/.venvs/finches/bin/python single_chain_test.py --cache-x
  ~/.venvs/finches/bin/python single_chain_test.py --idrome
  ~/.venvs/finches/bin/python single_chain_test.py --gaivi /path/to/outroot --label calvados_293K
Outputs: results/single_chain/{x_profiles.npz, <label>_per_slab.csv, <label>_summary.txt}
"""
import csv
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results', 'single_chain')
sys.path.insert(0, HERE)
import slab_test as S          # noqa: E402
import explore_saturation as ES  # noqa: E402

CUT = 0.8


def slab_list():
    ps = {r['name'] for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'amendment2', 'per_slab.csv')))
          if r['tier'] in ('primary', 'replication') and S.num(r['ratio_ours']) > 10}
    return [s for s in S.slabs() if s['name'] in ps or (s['tier'] == 'designed' and
            os.path.exists(os.path.join(S.FR, s['set'], s['name'], f"{s['name']}_sub.dcd")))]


def cache_x():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'x_profiles.npz')
    have = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for k, s in enumerate(slab_list()):
        if s['name'] in have:
            continue
        X, _ = ES.profiles(s)
        have[s['name']] = X
        np.savez_compressed(path, **have)
        print(f"{k + 1} {s['name']} cached", flush=True)
    print('x profiles cached:', len(have))


def degree_from_frames(frames):
    """frames: iterable of (n, 3) coordinate arrays in nm, already whole (no periodic images)."""
    tot, nf = None, 0
    for x in frames:
        d = squareform(pdist(x))
        n = len(x)
        c = (d < CUT) & (np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= 3)
        tot = c.sum(1).astype(float) if tot is None else tot + c.sum(1)
        nf += 1
    return tot / nf, nf


def whole(x, box):
    if box is None:
        return x
    d = np.diff(x, axis=0)
    d -= box * np.round(d / box)
    return np.vstack([x[:1], x[:1] + np.cumsum(d, axis=0)])


def load_single(path_traj, path_top, production_half):
    import mdtraj as md
    t = md.load(path_traj, top=path_top)
    ca = t.topology.select('name CA')
    if len(ca) == 0:
        ca = np.arange(t.n_atoms)
    idx = range(t.n_frames // 2, t.n_frames) if production_half else range(t.n_frames)
    boxes = t.unitcell_lengths
    return degree_from_frames(whole(t.xyz[f][ca].astype(float), None if boxes is None else boxes[f].astype(float)) for f in idx)


def compare(label, source):
    os.makedirs(OUT, exist_ok=True)
    X = np.load(os.path.join(OUT, 'x_profiles.npz'), allow_pickle=True)
    pred = np.load(os.path.join(ROOT, 'results', 'amendment2', 'pred_degree.npz'))
    lam = {r['one']: float(r['lambdas']) for r in csv.DictReader(open(S.CAL))}
    rows = []
    for s in slab_list():
        name = s['name']
        got = source(name)
        if got is None or name not in X.files:
            continue
        simdeg, nf = got
        if len(simdeg) != s['N']:
            print(f'{name}: length {len(simdeg)} != {s["N"]}, skipped', flush=True)
            continue
        Z = S.covariates(s['seq'], lam)
        x = X[name]
        p = pred[name] if name in pred.files else None
        row = {'name': name, 'tier': s['tier'], 'N': s['N'], 'frames': nf,
               'h1_simulated': S.partial(simdeg, x, Z),
               'h1_predicted': S.partial(p, x, Z) if p is not None else float('nan'),
               'raw_simulated': spearmanr(simdeg, x).correlation,
               'sim_vs_pred': spearmanr(simdeg, p).correlation if p is not None else float('nan')}
        rows.append(row)
        print(f"{s['tier']:11s} {name:22s} simulated {row['h1_simulated']:+.3f}  predicted {row['h1_predicted']:+.3f}", flush=True)
    with open(os.path.join(OUT, f'{label}_per_slab.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    g = lambda k, sub: np.array([r[k] for r in sub], float)  # noqa: E731
    L = [f'Exploratory: agreement beyond local stickiness with simulated against predicted single-chain degree ({label})', '']
    for tier in ('primary', 'replication', 'designed', 'all natural'):
        sub = [r for r in rows if (r['tier'] in ('primary', 'replication') if tier == 'all natural' else r['tier'] == tier)]
        if not sub:
            continue
        sim, prd = g('h1_simulated', sub), g('h1_predicted', sub)
        ok = ~np.isnan(prd)
        L.append(f'{tier} ({len(sub)} slabs): simulated median {np.median(sim):+.3f} ({(sim > 0).sum()}/{len(sim)} positive), '
                 f'predicted median {np.median(prd[ok]):+.3f}' + (f', paired difference median {np.median(sim[ok] - prd[ok]):+.3f}, '
                 f'two-sided Wilcoxon P {wilcoxon(sim[ok] - prd[ok]).pvalue:.2g}' if ok.sum() > 5 else ''))
        L.append(f'   simulated vs predicted single-chain degree per slab, median Spearman {np.nanmedian(g("sim_vs_pred", sub)):.2f}')
    txt = '\n'.join(L) + '\n'
    open(os.path.join(OUT, f'{label}_summary.txt'), 'w').write(txt)
    print(txt)


def main():
    if '--cache-x' in sys.argv:
        return cache_x()
    if '--idrome' in sys.argv:
        base = os.path.join(ROOT, 'data', 'idrome_single')

        def src(name):
            d = os.path.join(base, name)
            t, p = os.path.join(d, 'traj.xtc'), os.path.join(d, 'top.pdb')
            return load_single(t, p, production_half=False) if os.path.exists(t) and os.path.exists(p) else None
        return compare('idrome_310K', src)
    if '--gaivi' in sys.argv:
        base = sys.argv[sys.argv.index('--gaivi') + 1]
        label = sys.argv[sys.argv.index('--label') + 1] if '--label' in sys.argv else 'gaivi_293K'

        def src(name):
            d = os.path.join(base, '293K', name)
            t, p = os.path.join(d, f'{name}.dcd'), os.path.join(d, 'top.pdb')
            return load_single(t, p, production_half=True) if os.path.exists(t) and os.path.exists(p) else None
        return compare(label, src)
    sys.exit(__doc__)


if __name__ == '__main__':
    main()
