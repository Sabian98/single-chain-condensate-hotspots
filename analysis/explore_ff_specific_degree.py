"""
explore_ff_specific_degree.py: EXPLORATORY check of the force-field caveat on H1 (outside the pre-registered plan).
The slab tests use the consensus degree model, trained on the mean single-chain degree of CALVADOS-2 and Mpipi-Recharged (plus
Mpipi-GG for 907 proteins). The slabs are CALVADOS-2. If H1 came from CALVADOS-specific signal, a model trained on CALVADOS-2 labels
alone should beat one trained on Recharged labels alone on these slabs by a wide margin.

  --train  (bender env)   CALVADOS-only and Recharged-only degree models, same features, booster and proteins (those with labels in
                          both) as freeze_degree_consensus.py; predictions for every slab sequence -> results/ff_specific/pred.npz
  --score  (finches env)  for every fetched primary and replication slab: the between-chain degree X and covariates Z exactly as
                          slab_test.analyse computes them (captured from its H1 call), then the H1 partial for each model
                          -> results/ff_specific/{per_slab.csv, summary.txt}
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results', 'ff_specific')
EV = '/Volumes/User Homes/Nu Project/BENDER-DYN/redo_evidence'
sys.path.insert(0, HERE)
import slab_test as S  # noqa: E402

MODELS = ('consensus', 'calvados_only', 'recharged_only')


def train():
    import pickle
    from collections import defaultdict
    sys.path.insert(0, os.path.join(EV, 'hub_predictor'))
    import train_hub_model as T
    from train_hub_multiff import fit_w
    prots, y = T.load(os.path.join(EV, 'hub_predictor', 'data_v3'), 'deg_mean')
    seq_of = {}
    for (ff, uid), r in prots.items():
        seq_of.setdefault(uid, r['sequence'])
    by_uid = defaultdict(dict)
    for (ff, uid), lab in y.items():
        if len(lab) == len(seq_of[uid]) and prots[(ff, uid)]['sequence'] == seq_of[uid]:
            by_uid[uid][ff] = np.array([lab[i] for i in range(len(lab))])
    both = sorted(u for u, d in by_uid.items() if 'CALVADOS-2' in d and 'Mpipi-Recharged' in d)
    X = np.vstack([T.features(seq_of[u], 10) for u in both])
    models = {'consensus': pickle.load(open(os.path.join(EV, 'hub_predictor', 'degree_consensus_2026-10-13.pkl'), 'rb'))}
    for key, ff in (('calvados_only', 'CALVADOS-2'), ('recharged_only', 'Mpipi-Recharged')):
        Y = np.concatenate([by_uid[u][ff] for u in both])
        models[key] = fit_w('hgb', X, Y, np.ones(len(Y)))
        print(f'{key}: {len(both)} proteins, {len(Y)} residues', flush=True)
    os.makedirs(OUT, exist_ok=True)
    pred = {f'{k}|{s["name"]}': m.predict(T.features(s['seq'], 10)) for s in S.slabs() for k, m in models.items()}
    np.savez_compressed(os.path.join(OUT, 'pred.npz'), **pred)
    print(f'predicted {len(pred) // len(models)} sequences with {len(models)} models')


def score():
    from scipy.stats import wilcoxon, spearmanr
    pred = np.load(os.path.join(OUT, 'pred.npz'))
    lam = {r['one']: float(r['lambdas']) for r in csv.DictReader(open(S.CAL))}
    captured = {}
    real_partial = S.partial

    def recorder(x, y, Z):
        if 'X' not in captured:                                 # the first call in analyse is the H1 partial(pred, X, Z)
            captured.update(X=np.array(y), Z=[np.array(z) for z in Z])
        return real_partial(x, y, Z)

    S.partial = recorder
    rows = []
    for s in S.slabs():
        if s['tier'] not in ('primary', 'replication'):
            continue
        if not os.path.exists(os.path.join(S.FR, s['set'], s['name'], f'{s["name"]}_sub.dcd')):
            continue
        captured.clear()
        r = S.analyse(s, pred[f'consensus|{s["name"]}'], lam)
        if not r['ratio_ours'] > 10:
            continue
        row = {'name': s['name'], 'tier': s['tier'], 'N': s['N']}
        for k in MODELS:
            row[k] = real_partial(pred[f'{k}|{s["name"]}'], captured['X'], captured['Z'])
        row['cal_vs_rech_pred'] = spearmanr(pred[f'calvados_only|{s["name"]}'], pred[f'recharged_only|{s["name"]}']).correlation
        rows.append(row)
        print(f"{s['tier']:12s} {s['name']:22s} " + ' '.join(f'{k} {row[k]:+.3f}' for k in MODELS), flush=True)
    S.partial = real_partial
    with open(os.path.join(OUT, 'per_slab.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = ['Exploratory: H1 partial with force-field-specific degree models (CALVADOS-2 slabs)', '']
    for tier in ('primary', 'replication', 'all'):
        sub = [r for r in rows if tier == 'all' or r['tier'] == tier]
        L.append(f'{tier} ({len(sub)} slabs)')
        for k in MODELS:
            v = np.array([r[k] for r in sub])
            L.append(f'  {k:15s} median {np.median(v):+.3f}, {(v > 0).sum()}/{len(v)} positive')
        d = np.array([r['calvados_only'] - r['recharged_only'] for r in sub])
        L.append(f'  CALVADOS-only minus Recharged-only: median {np.median(d):+.3f}, CALVADOS-only higher in {(d > 0).sum()}/{len(d)}, '
                 f'two-sided Wilcoxon P {wilcoxon(d).pvalue:.2g}')
        L.append(f'  agreement of the two single-FF predictions per slab: median Spearman {np.median([r["cal_vs_rech_pred"] for r in sub]):.2f}')
    txt = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'summary.txt'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    train() if '--train' in sys.argv else score() if '--score' in sys.argv else sys.exit(__doc__)
