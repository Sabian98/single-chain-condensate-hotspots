"""
amendment4_test.py: analysis of Amendment 4 to slab_followup_plan.md. Written and hashed before the An, Webb and Jacobs (Sci Adv 2024)
data were downloaded and before any value of either test was computed.

  ~/.venvs/bender/bin/python amendment4_test.py --predict-b     degree-model predictions for the Test B sequences (needs scikit-learn)
  ~/.venvs/finches/bin/python amendment4_test.py --test B       Test B (dynamics at matched B2, published HPS data)
  ~/.venvs/finches/bin/python amendment4_test.py --test A       Test A (organization at matched c_sat); refuses until every
                                                                replication slab of Amendment 2 is fetched
Outputs: results/amendment4/{test_b.txt, test_b_per_sequence.csv, test_a.txt, test_a_per_slab.csv}

Partial Spearman: ranks of x and y regressed on the ranks of the covariates (least squares with intercept), Pearson correlation of the
residuals; one-sided P from the t distribution with n - 2 - k degrees of freedom. Gini of a profile v (clipped at 0):
sum_ij |v_i - v_j| / (2 n^2 mean v).
"""
import csv
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results', 'amendment4')
AN = os.path.join(ROOT, 'data', 'an2024')
EV = '/Volumes/User Homes/Nu Project/BENDER-DYN/redo_evidence'


def st():
    spec = importlib.util.spec_from_file_location('slab_test', os.path.join(HERE, 'slab_test.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gini(v):
    import numpy as np
    v = np.clip(np.asarray(v, float), 0, None)
    if v.mean() == 0:
        return float('nan')
    return float(np.abs(v[:, None] - v[None, :]).sum() / (2 * len(v) ** 2 * v.mean()))


def partial(x, y, Z, tail):
    """Partial Spearman and its one-sided P ('less' or 'greater')."""
    import numpy as np
    from scipy.stats import rankdata, t as tdist
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z])
    ex = rankdata(x) - A @ np.linalg.lstsq(A, rankdata(x), rcond=None)[0]
    ey = rankdata(y) - A @ np.linalg.lstsq(A, rankdata(y), rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    df = len(x) - 2 - len(Z)
    tt = r * np.sqrt(df / max(1e-12, 1 - r * r))
    p = float(tdist.cdf(tt, df) if tail == 'less' else tdist.sf(tt, df))
    return r, p, len(x)


# ---------------------------------------------------------------- Test B
def read_b():
    seqs = [l.strip() for l in open(os.path.join(AN, 'seq_heteromeric.txt')) if l.strip()]
    lab = list(csv.DictReader(open(os.path.join(AN, 'labels_heteromeric.csv'))))
    feat = list(csv.DictReader(open(os.path.join(AN, 'features_heteromeric.csv'))))
    assert len(seqs) == len(lab) == len(feat), (len(seqs), len(lab), len(feat))
    return seqs, lab, feat


def predict_b():
    import pickle
    import numpy as np
    sys.path.insert(0, os.path.join(EV, 'hub_predictor'))
    import train_hub_model as T
    model = pickle.load(open(os.path.join(EV, 'hub_predictor', 'degree_consensus_2026-10-13.pkl'), 'rb'))
    seqs = [l.strip() for l in open(os.path.join(AN, 'seq_heteromeric.txt')) if l.strip()]
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, 'pred_degree_b.npz'), **{str(i): model.predict(T.features(s, 10)) for i, s in enumerate(seqs)})
    print(f'predicted {len(seqs)} sequences')


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def col(row, *names):
    for n in names:
        for k in row:
            if k.strip().lower() == n.lower():
                return num(row[k])
    raise KeyError(names)


def test_b():
    import numpy as np
    seqs, lab, feat = read_b()
    pred = np.load(os.path.join(OUT, 'pred_degree_b.npz'))
    rows, dropped = [], {'not phase separating or no D': 0, 'B2 not negative': 0}
    for i, (s, l, f) in enumerate(zip(seqs, lab, feat)):
        d, b2, psp = num(l.get('diff')), num(l.get('B2')), num(l.get('psp'))
        if not (psp == 1 and np.isfinite(d)):
            dropped['not phase separating or no D'] += 1
            continue
        if not b2 < 0:
            dropped['B2 not negative'] += 1
            continue
        p = pred[str(i)]
        rows.append({'index': i, 'seq': s, 'N': len(s), 'generation': l.get('generation'), 'D': d, 'B2': b2,
                     'log_neg_B2': float(np.log(-b2)), 'M_pred': float(np.mean(p)), 'U_pred': gini(p),
                     'lambda_mean': col(f, 'sum lambda') / len(s), 'SHD': col(f, 'SHD'), 'SCD': col(f, 'SCD')})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'test_b_per_sequence.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    g = lambda k: [r[k] for r in rows]  # noqa: E731
    base = [g('log_neg_B2'), g('N')]
    r1, p1, n = partial(g('M_pred'), g('D'), base, 'less')
    L = ['Amendment 4, Test B: dynamics at matched B2 (An, Webb and Jacobs, Sci Adv 2024; HPS model, 300 K)', '',
         f'sequences: {len(seqs)} heteromeric; included {n}; left out {dropped}', '',
         f'B1 partial Spearman(M_pred, D | log(-B2), N) {r1:+.3f}, one-sided P (negative) {p1:.3g} -> '
         f'{"SUPPORTED" if (r1 < 0 and p1 < 0.05) else "NOT SUPPORTED"}', '', 'secondary:']
    r, p, _ = partial(g('U_pred'), g('D'), base, 'less')
    L.append(f'  U_pred in place of M_pred: {r:+.3f} (one-sided P, negative, {p:.3g})')
    r, p, _ = partial(g('M_pred'), g('D'), base + [g('lambda_mean'), g('SHD'), g('SCD')], 'less')
    L.append(f'  M_pred beyond log(-B2), N, mean lambda, SHD, SCD: {r:+.3f} (one-sided P {p:.3g})')
    from scipy.stats import spearmanr
    L.append(f'  context: Spearman(D, log(-B2)) {spearmanr(g("D"), g("log_neg_B2")).correlation:+.3f}; '
             f'Spearman(M_pred, log(-B2)) {spearmanr(g("M_pred"), g("log_neg_B2")).correlation:+.3f}')
    txt = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'test_b.txt'), 'w').write(txt)
    print(txt)


# ---------------------------------------------------------------- Test A
def test_a():
    import numpy as np
    from scipy.stats import spearmanr
    S = st()
    slabs = [s for s in S.slabs() if s['tier'] in ('primary', 'replication')]
    missing = [s['name'] for s in slabs if not os.path.exists(os.path.join(S.FR, s['set'], s['name'], f"{s['name']}_sub.dcd"))]
    if missing:
        sys.exit(f'Test A waits for the full set: {len(missing)} slabs not fetched yet')
    per = {r['name']: r for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'amendment2', 'per_slab.csv')))}
    cdil = {}
    for f in S.SETS.values():
        for r in csv.DictReader(open(os.path.join(S.TAB, f))):
            cdil[r[list(r)[0]]] = num(r['cdil'])
    pred = np.load(os.path.join(ROOT, 'results', 'amendment2', 'pred_degree.npz'))
    lam = {r['one']: float(r['lambdas']) for r in csv.DictReader(open(S.CAL))}
    rows = []
    for s in slabs:
        r = per.get(s['name'])
        if r is None or not float(r['ratio_ours']) > 10:
            continue
        X = profile_x(S, s)
        rows.append({'name': s['name'], 'tier': s['tier'], 'N': s['N'], 'log10_cdil': float(np.log10(cdil[s['name']])),
                     'lambda_mean': float(np.mean([lam[c] for c in s['seq']])), 'U_obs': gini(X), 'U_pred': gini(pred[s['name']]),
                     'rg_ratio': num(r.get('h4_rg_ratio')), 'share_between': num(r.get('share_between'))})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'test_a_per_slab.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    g = lambda k, rr=rows: [x[k] for x in rr]  # noqa: E731
    L = ['Amendment 4, Test A: organization at matched saturation concentration (natural slabs, CALVADOS-2)', '',
         f'slabs: {len(rows)} phase separated (primary and replication)', '', 'A1 (descriptive): Spearman with log10 cdil']
    rg = [x for x in rows if np.isfinite(x['rg_ratio'])]
    L.append(f'  dense/dilute Rg ratio {spearmanr(g("log10_cdil", rg), g("rg_ratio", rg)).correlation:+.3f} (n {len(rg)}); '
             f'share of contacts between chains {spearmanr(g("log10_cdil"), g("share_between")).correlation:+.3f} (n {len(rows)})')
    r, p, n = partial(g('U_pred'), g('U_obs'), [g('log10_cdil'), g('N'), g('lambda_mean')], 'greater')
    L += ['', f'A2 partial Spearman(U_pred, U_obs | log10 cdil, N, mean lambda) {r:+.3f}, one-sided P {p:.3g}, n {n} -> '
              f'{"SUPPORTED" if (r > 0 and p < 0.05) else "NOT SUPPORTED"}', '', 'A3 pairs (cdil within 10%, N within 20%):']
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if abs(a['log10_cdil'] - b['log10_cdil']) < np.log10(1.1) and abs(a['N'] - b['N']) <= 0.2 * max(a['N'], b['N']):
                L.append(f'  {a["name"]} / {b["name"]}: U_obs {a["U_obs"]:.3f} / {b["U_obs"]:.3f}, U_pred {a["U_pred"]:.3f} / {b["U_pred"]:.3f}')
    txt = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'test_a.txt'), 'w').write(txt)
    print(txt)


def profile_x(S, slab):
    """X(i), contacts with other chains per position in the dense phase, with slab_test.analyse's definitions."""
    import mdtraj as md
    import numpy as np
    from scipy.spatial import cKDTree
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
    hw = float(max(half, 1))
    X, nd = np.zeros(n), 0
    for f in range(t.n_frames):
        x = np.mod(t.xyz[f].astype(float), box)
        pr = cKDTree(x, boxsize=box).query_pairs(S.CUT, output_type='ndarray')
        a, b = pr[:, 0], pr[:, 1]
        ca, cb, ia, ib = a // n, b // n, a % n, b % n
        dense = dzs[f] < hw
        other = ca != cb
        X += np.bincount(ia[other & dense[ca]], minlength=n) + np.bincount(ib[other & dense[cb]], minlength=n)
        nd += int(dense.sum())
    return X / max(nd, 1)


if __name__ == '__main__':
    if '--predict-b' in sys.argv:
        predict_b()
    elif '--test' in sys.argv:
        {'A': test_a, 'B': test_b}[sys.argv[sys.argv.index('--test') + 1]]()
    else:
        print(__doc__)
