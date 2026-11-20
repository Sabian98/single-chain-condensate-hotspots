"""
recharged_slab_test.py: Amendment 3 of slab_followup_plan.md, H1 in Mpipi-Recharged slabs.

slab_test.py (Amendment 2) is used unchanged: each Recharged trajectory (1,000 frames over 1 microsecond) is reduced to 200 evenly
spaced frames of its second half, written as <tmp>/recharged/<name>/<name>_sub.dcd, and slab_test.analyse() is run on it with the
same consensus-model prediction D(i), CALVADOS-2 lambdas for the baseline, contact cutoff, phase definitions and criterion.
analyse() does not return the per-residue profiles, so the arrays it passes to its own partial() and spearmanr() calls are recorded:
X(i) (contacts with other chains, dense phase), the baseline Z, and W_dilute(i) (within-chain degree of Recharged's dilute chains).

Decision (fixed by Amendment 3): one-sided Wilcoxon of H1 over phase-separating slabs, replicated if P < 0.05, inconclusive below 6.
Secondary (fixed, no decision): H1 with Recharged's own dilute chains in place of D(i); per-sequence H1 Recharged vs CALVADOS-2; H2-H5.
Exploratory (not in the plan): CALVADOS-only and Recharged-only degree models on the Recharged slabs; CALVADOS-2 vs Recharged X(i).
  ~/.venvs/finches/bin/python recharged_slab_test.py    ->   ../results/recharged_slab/{per_slab.csv, summary.txt, profiles.npz}
"""
import csv
import os
import sys
import tempfile

import numpy as np
import scipy.stats
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import slab_test as S  # noqa: E402

RUNS = os.path.join(ROOT, 'data', 'slab_recharged')
OUT = os.path.join(ROOT, 'results', 'recharged_slab')
ORDER = ['O60477_1_58', 'Q9H5F2_1_150', 'Q9P2K5_306_526', 'Q14154_1_230', 'Q96P88_140_292', 'Q7Z5H4_162_357', 'P12035_1_202',
         'O00167_1_265', 'Q9Y6C7_1_94', 'Q6V1P9_1_111', 'Q5K131_1_121', 'Q8N3T1_1_161']   # Amendment 3


def subsample(name, tmp):
    """Second half of the 1,000 frames, 200 evenly spaced, in the layout slab_test.analyse() reads."""
    import mdtraj as md
    d = os.path.join(RUNS, name)
    t = md.load_dcd(os.path.join(d, f'{name}.dcd'), top=os.path.join(d, 'top.pdb'))
    idx = np.unique(np.round(np.linspace(t.n_frames // 2, t.n_frames - 1, 200)).astype(int))
    out = os.path.join(tmp, 'recharged', name)
    os.makedirs(out, exist_ok=True)
    t[idx].save_dcd(os.path.join(out, f'{name}_sub.dcd'))
    return t.n_frames, len(idx)


def run_one(slab, pred, lam):
    """slab_test.analyse() with its partial() and spearmanr() arguments recorded, to recover X, Z and W_dilute."""
    rec = {'partial': [], 'sp': []}
    orig_partial, orig_sp = S.partial, scipy.stats.spearmanr

    def partial_rec(x, y, Z):
        rec['partial'].append((np.asarray(x), np.asarray(y), Z))
        return orig_partial(x, y, Z)

    def sp_rec(a, b, *args, **kw):
        rec['sp'].append((np.asarray(a), np.asarray(b)))
        return orig_sp(a, b, *args, **kw)

    S.partial, scipy.stats.spearmanr = partial_rec, sp_rec
    try:
        row = S.analyse(slab, pred, lam)
    finally:
        S.partial, scipy.stats.spearmanr = orig_partial, orig_sp
    _, X, Z = rec['partial'][0]                                   # partial(pred, X, Z)
    last = rec['sp'][-1]
    Wl = last[1] if len(rec['sp']) == 5 and np.array_equal(last[0], np.asarray(pred)) else None   # sp(pred, Wl), dilute block ran
    return row, X, Z, Wl


def main():
    pred = np.load(os.path.join(S.OUT, 'pred_degree.npz'))
    ffp = np.load(os.path.join(ROOT, 'results', 'ff_specific', 'pred.npz'))
    lam = {r['one']: float(r['lambdas']) for r in csv.DictReader(open(S.CAL))}
    cal_rows = {r['name']: r for r in csv.DictReader(open(os.path.join(S.OUT, 'per_slab.csv')))}
    Xcal = np.load(os.path.join(ROOT, 'results', 'single_chain', 'x_profiles.npz'))
    info = {s['name']: s for s in S.slabs()}
    done = [n for n in ORDER if os.path.exists(os.path.join(RUNS, n, f'{n}.dcd'))]
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='recharged_')
    S.FR = tmp
    rows, prof = [], {}
    for name in done:
        frames, used = subsample(name, tmp)
        slab = dict(info[name], set='recharged')
        row, X, Z, Wl = run_one(slab, pred[name], lam)
        row.update({'frames_total': frames, 'frames_used': used, 'h1_cal_slab': float(S.num(cal_rows[name]['h1_partial'])),
                    'h1_calvados_only': S.partial(ffp[f'calvados_only|{name}'], X, Z),
                    'h1_recharged_only': S.partial(ffp[f'recharged_only|{name}'], X, Z),
                    'h1_own_dilute': S.partial(Wl, X, Z) if Wl is not None else float('nan'),
                    'x_cal_vs_rech': float(spearmanr(Xcal[name], X).correlation),
                    'gly': slab['seq'].count('G') / len(slab['seq'])})
        prof[name] = X
        if Wl is not None:
            prof[name + '|dilute'] = Wl
        rows.append(row)
        print(f"{name:18s} chains {row['chains']:4d} ratio {row['ratio_ours']:8.1f} H1 {row['h1_partial']:+.3f} "
              f"(CAL slab {row['h1_cal_slab']:+.3f}) own dilute {row['h1_own_dilute']:+.3f} X CAL vs Rech {row['x_cal_vs_rech']:+.2f}",
              flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('name', 'tier'), k))
    with open(os.path.join(OUT, 'per_slab.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(OUT, 'profiles.npz'), **prof)
    summary(rows, done)


def summary(rows, done):
    ps = [r for r in rows if r['ratio_ours'] > 10]
    out = [f'Amendment 3: H1 in Mpipi-Recharged slabs ({len(done)} of 12 finished: {", ".join(done)})',
           f'phase separated by the Amendment 2 criterion: {len(ps)}; not: {[r["name"] for r in rows if r["ratio_ours"] <= 10]}', '']
    h1 = np.array([r['h1_partial'] for r in ps])
    if len(ps) >= 6:
        p = wilcoxon(h1, alternative='greater').pvalue
        verdict = 'REPLICATED in a second force field' if p < 0.05 else 'not replicated'
    else:
        p, verdict = float('nan'), 'inconclusive (fewer than 6 phase-separating slabs)'
    out += [f'DECISION  H1 (consensus model D(i)): median {np.median(h1):+.3f}, {int((h1 > 0).sum())}/{len(h1)} positive, '
            f'one-sided Wilcoxon P {p:.2g} -> {verdict}', '']
    own = np.array([r['h1_own_dilute'] for r in ps if r['h1_own_dilute'] == r['h1_own_dilute']])
    out.append('secondary (no decision):')
    if len(own):
        po = wilcoxon(own, alternative='greater').pvalue if len(own) >= 6 else float('nan')
        out.append(f'  H1 with Recharged dilute chains in place of D(i): median {np.median(own):+.3f}, {int((own > 0).sum())}/{len(own)} '
                   f'positive, P {po:.2g}')
    cal = np.array([r['h1_cal_slab'] for r in ps])
    out.append(f'  H1 same sequences in CALVADOS-2 slabs: median {np.median(cal):+.3f}; Recharged minus CALVADOS-2 median '
               f'{np.median(h1 - cal):+.3f}; Spearman across sequences {spearmanr(h1, cal).correlation:+.2f}')
    for k, lab in (('h2_dense_vs_dilute', 'H2 dense vs dilute within-chain'), ('h4_rg_ratio', 'H4 Rg dense/dilute')):
        v = [r[k] for r in ps if k in r and r[k] == r[k]]
        if v:
            out.append(f'  {lab}: median {np.median(v):.3f} (n {len(v)})')
    v = [(r['h5_share_aro'], r['h5_share_gly']) for r in ps if 'h5_share_aro' in r]
    if v:
        out.append(f'  H5 aromatic share above Glycine share in {sum(a > g for a, g in v)}/{len(v)}')
    out += ['', 'exploratory (not in the plan):']
    xr = np.array([r['x_cal_vs_rech'] for r in ps])
    out.append(f'  contacts with other chains, CALVADOS-2 vs Recharged slab, Spearman per sequence: median {np.median(xr):+.2f}')
    for k in ('h1_calvados_only', 'h1_recharged_only'):
        out.append(f'  {k}: median {np.median([r[k] for r in ps]):+.3f}')
    out.append('  per sequence (Glycine fraction | H1 consensus Rech slab | CAL slab | CAL-only model | Rech-only model | own dilute):')
    for r in sorted(ps, key=lambda r: -r['gly']):
        out.append(f"    {r['name']:18s} {r['gly']:.2f} | {r['h1_partial']:+.3f} | {r['h1_cal_slab']:+.3f} | {r['h1_calvados_only']:+.3f} | "
                   f"{r['h1_recharged_only']:+.3f} | {r['h1_own_dilute']:+.3f}")
    open(os.path.join(OUT, 'summary.txt'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
