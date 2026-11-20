"""
recharged_ladder_test.py: Amendment 3b of slab_followup_plan.md, Mpipi-Recharged temperature ladder (280, 270, 260, 250 K) for the
8 sequences that did not phase separate at 293 K.

Each run is analysed exactly as in recharged_slab_test.py (slab_test.analyse unchanged, 200 frames of the second half). The rung used
for a sequence is the warmest one at which it phase separates by the Amendment 2 criterion (ratio_ours > 10), chosen by that criterion
alone. Runs are read from ../data/slab_recharged_lowT/T<K>/<name>/<name>.dcd.

  python recharged_ladder_test.py --check 280   # phase criterion for every finished run at one rung, and the next submit command
  python recharged_ladder_test.py               # decision over all phase-separating slabs (293 K ones from Amendment 3 + ladder)
Outputs: ../results/recharged_slab/ladder_per_run.csv, ladder_summary.txt, ladder_profiles.npz
"""
import csv
import os
import sys
import tempfile

import numpy as np
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recharged_slab_test as RT  # noqa: E402
import slab_test as S  # noqa: E402

LADDER = (280, 250)                                          # Amendment 3c: 270 and 260 K dropped
EIGHT = ['Q9P2K5_306_526', 'P12035_1_202', 'Q14154_1_230', 'Q96P88_140_292', 'O00167_1_265', 'Q9Y6C7_1_94', 'Q6V1P9_1_111',
         'Q5K131_1_121']
GLY_RICH = ('Q9P2K5_306_526', 'P12035_1_202')
LOWT = os.path.join(RT.ROOT, 'data', 'slab_recharged_lowT')
OUT = RT.OUT


def analyse_run(name, T, inputs, tmp):
    """One ladder run through the Amendment 3 pipeline; returns the row and the recorded profiles."""
    pred, ffp, lam, cal_rows, Xcal, info = inputs
    RT.RUNS = os.path.join(LOWT, f'T{T}')
    frames, used = RT.subsample(name, tmp)
    row, X, Z, Wl = RT.run_one(dict(info[name], set='recharged'), pred[name], lam)
    row.update({'temp': T, 'frames_total': frames, 'frames_used': used, 'h1_cal_slab': float(S.num(cal_rows[name]['h1_partial'])),
                'h1_calvados_only': S.partial(ffp[f'calvados_only|{name}'], X, Z),
                'h1_recharged_only': S.partial(ffp[f'recharged_only|{name}'], X, Z),
                'h1_own_dilute': S.partial(Wl, X, Z) if Wl is not None else float('nan'),
                'x_cal_vs_rech': float(spearmanr(Xcal[name], X).correlation),
                'gly': info[name]['seq'].count('G') / len(info[name]['seq'])})
    return row, X, Wl


def load_inputs():
    pred = np.load(os.path.join(S.OUT, 'pred_degree.npz'))
    ffp = np.load(os.path.join(RT.ROOT, 'results', 'ff_specific', 'pred.npz'))
    lam = {r['one']: float(r['lambdas']) for r in csv.DictReader(open(S.CAL))}
    cal_rows = {r['name']: r for r in csv.DictReader(open(os.path.join(S.OUT, 'per_slab.csv')))}
    Xcal = np.load(os.path.join(RT.ROOT, 'results', 'single_chain', 'x_profiles.npz'))
    info = {s['name']: s for s in S.slabs()}
    return pred, ffp, lam, cal_rows, Xcal, info


def finished(T):
    return [n for n in EIGHT if os.path.exists(os.path.join(LOWT, f'T{T}', n, f'{n}.dcd'))]


def check(T):
    inputs, tmp = load_inputs(), tempfile.mkdtemp(prefix='ladder_')
    S.FR = tmp
    done = finished(T)
    ps, left = [], []
    for name in done:
        row, _, _ = analyse_run(name, T, inputs, tmp)
        (ps if row['ratio_ours'] > 10 else left).append(name)
        print(f"{T} K  {name:18s} ratio {row['ratio_ours']:9.1f}  {'phase separated' if row['ratio_ours'] > 10 else 'not phase separated'}",
              flush=True)
    missing = [n for n in EIGHT if n not in done]
    print(f'\n{T} K: {len(ps)} phase separated, {len(left)} not, {len(missing)} not finished {missing}')
    nxt = [t for t in LADDER if t < T]
    if left and nxt:
        print(f'next rung: bash submit_recharged_slab_lowT.sh {nxt[0]} ' + ' '.join(left))
    elif left:
        print(f'not phase separated at 250 K, reported and left out: {left}')


def main():
    inputs, tmp = load_inputs(), tempfile.mkdtemp(prefix='ladder_')
    S.FR = tmp
    rows, prof, chosen = [], {}, {}
    for name in EIGHT:
        for T in LADDER:                                          # warmest first; stop at the first rung that phase separates
            if not os.path.exists(os.path.join(LOWT, f'T{T}', name, f'{name}.dcd')):
                continue
            row, X, Wl = analyse_run(name, T, inputs, tmp)
            rows.append(row)
            print(f"{T} K {name:18s} ratio {row['ratio_ours']:9.1f} H1 {row['h1_partial']:+.3f}", flush=True)
            if row['ratio_ours'] > 10:
                chosen[name] = row
                prof[f'{name}|T{T}'] = X
                break
    os.makedirs(OUT, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('name', 'temp'), k))
    with open(os.path.join(OUT, 'ladder_per_run.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(OUT, 'ladder_profiles.npz'), **prof)
    at293 = [dict(r, temp=293) for r in csv.DictReader(open(os.path.join(OUT, 'per_slab.csv'))) if float(r['ratio_ours']) > 10]
    for r in at293:
        for k in ('h1_partial', 'h1_cal_slab', 'h1_calvados_only', 'h1_recharged_only', 'h1_own_dilute', 'gly'):
            r[k] = S.num(r[k])
    allps = at293 + list(chosen.values())
    h1 = np.array([r['h1_partial'] for r in allps])
    if len(h1) >= 6:
        p = wilcoxon(h1, alternative='greater').pvalue
        verdict = 'REPLICATED in a second force field' if p < 0.05 else 'not replicated'
    else:
        p, verdict = float('nan'), 'inconclusive (fewer than 6 phase-separating slabs)'
    out = ['Amendment 3b: Mpipi-Recharged H1 over all phase-separating slabs (293 K from Amendment 3, ladder for the rest)',
           f"analysed rung per sequence: {', '.join(f'{n} {r['temp']} K' for n, r in chosen.items()) or 'none yet'}",
           f"not phase separated on the finished rungs: {[n for n in EIGHT if n not in chosen]}", '',
           f'DECISION  H1 (consensus D(i)), n {len(h1)}: median {np.median(h1):+.3f}, {int((h1 > 0).sum())}/{len(h1)} positive, '
           f'one-sided Wilcoxon P {p:.2g} -> {verdict}', '', 'Glycine-rich sequences (secondary, no decision):']
    for n in GLY_RICH:
        r = chosen.get(n)
        out.append(f"  {n}: " + (f"{r['temp']} K, Glycine {r['gly']:.2f}, CALVADOS-only {r['h1_calvados_only']:+.3f}, "
                                  f"Recharged-only {r['h1_recharged_only']:+.3f}, consensus {r['h1_partial']:+.3f}, "
                                  f"CALVADOS-2 slab {r['h1_cal_slab']:+.3f}" if r else 'not phase separated on the finished rungs'))
    own = np.array([r['h1_own_dilute'] for r in allps if r['h1_own_dilute'] == r['h1_own_dilute']])
    out += ['', 'secondary:', f'  H1 with Recharged dilute chains in place of D(i): ' +
            (f'median {np.median(own):+.3f}, {int((own > 0).sum())}/{len(own)} positive' if len(own) else 'no slab with enough dilute chains'),
            f"  H1 same sequences in CALVADOS-2 slabs: median {np.median([r['h1_cal_slab'] for r in allps]):+.3f}",
            '  per sequence (temperature | Glycine | consensus | CAL slab | CAL-only | Rech-only | own dilute):']
    for r in sorted(allps, key=lambda r: (r['temp'], -r['gly'])):
        out.append(f"    {r['name']:18s} {r['temp']} K | {r['gly']:.2f} | {r['h1_partial']:+.3f} | {r['h1_cal_slab']:+.3f} | "
                   f"{r['h1_calvados_only']:+.3f} | {r['h1_recharged_only']:+.3f} | {r['h1_own_dilute']:+.3f}")
    open(os.path.join(OUT, 'ladder_summary.txt'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    if '--check' in sys.argv:
        check(int(sys.argv[sys.argv.index('--check') + 1]))
    else:
        main()
