"""
slab_test.py: analysis of Amendment 2 to slab_followup_plan.md (degree in the condensate against single-chain degree, H1 to H5).
Written and hashed before any slab other than the loricrin pilot was fetched or read.

  ~/.venvs/bender/bin/python slab_test.py --predict        degree-model predictions for every listed sequence (needs scikit-learn)
  ~/.venvs/finches/bin/python slab_test.py                 every fetched slab, per-slab values and the tests
  ~/.venvs/finches/bin/python slab_test.py --smoke         loricrin only, into results/amendment2_smoke (checks the code, no test)
Outputs: results/amendment2/{per_slab.csv, summary.txt}.

Contacts: Calpha-Calpha < 0.8 nm, minimum image in all three dimensions, |i - j| >= 3 within a chain, every pair between chains.
Slab centre per frame: circular mean of chain centres along z. Dense phase: chains whose centre lies within the half-width at which the
time-averaged chain-centre density (1 nm bins of |dz|) first falls below half its maximum. Dilute phase: centre beyond 50 nm and no
contact with another chain in that frame. Phase separated (our frames): time-averaged bead density inside the dense half-width over that
beyond 50 nm, above 10. Baseline covariates: distance from the nearer chain end, and over residues i-10 to i+10 the mean CALVADOS-2
lambda, the fraction of G and S, the fraction of F, W, Y and the net charge (K, R +1; D, E -1).
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAB = os.path.join(ROOT, 'data', 'pspred_github')
FR = os.path.join(ROOT, 'data', 'erda_pspred')
OUT = os.path.join(ROOT, 'results', 'amendment2')
EV = '/Volumes/User Homes/Nu Project/BENDER-DYN/redo_evidence'
CAL = os.path.expanduser('~/Downloads/Scripts/CALVADOS IDP/input/residues_CALVADOS2.csv')
PILOT = 'P23490_1_312'
CUT, FAR, MIN_DILUTE, W = 0.8, 50.0, 100, 10
SETS = {'IDRome10': 'df_validation_IDRome10.csv', 'IDRome90': 'df_validation_IDRome90.csv', 'training': 'df_training.csv'}
BENDER_OVERLAP = {'P82970_1_282'}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def slabs():
    """Every slab with its tier, fixed by Amendment 2: phase separating in the source table (cden/cdil > 10)."""
    out = []
    for s, f in SETS.items():
        for r in csv.DictReader(open(os.path.join(TAB, f))):
            name = r[list(r)[0]]
            ratio = num(r['cden']) / num(r['cdil']) if num(r['cdil']) > 0 else float('nan')
            designed = re.match(r'^[A-Z0-9]{6,10}_\d+_\d+$', name) is None
            if name == PILOT:
                tier = 'pilot'
            elif not ratio > 10:
                tier = 'not phase separating (table)'
            elif s in ('IDRome10', 'IDRome90'):
                tier = 'primary'
            elif designed:
                tier = 'designed'
            elif name in BENDER_OVERLAP:
                tier = 'protein in BENDER'
            else:
                tier = 'replication'
            out.append({'set': s, 'name': name, 'tier': tier, 'seq': r['fasta'], 'N': len(r['fasta']), 'kappa': num(r['kappa']),
                        'scd': num(r['scd']), 'nu': num(r['nu']), 'table_ratio': ratio})
    return out


def predict():
    import pickle
    import numpy as np
    sys.path.insert(0, os.path.join(EV, 'hub_predictor'))
    import train_hub_model as T
    model = pickle.load(open(os.path.join(EV, 'hub_predictor', 'degree_consensus_2026-10-13.pkl'), 'rb'))
    pred = {s['name']: model.predict(T.features(s['seq'], 10)) for s in slabs()}
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, 'pred_degree.npz'), **pred)
    print(f'predicted {len(pred)} sequences')


def partial(x, y, Z):
    import numpy as np
    from scipy.stats import rankdata
    A = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z])
    rx, ry = rankdata(x), rankdata(y)
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return float('nan')
    return float(np.corrcoef(rankdata(ex), rankdata(ey))[0, 1])


def covariates(seq, lam):
    import numpy as np
    n = len(seq)
    l = np.array([lam[c] for c in seq])
    gs = np.array([c in 'GS' for c in seq], float)
    aro = np.array([c in 'FWY' for c in seq], float)
    q = np.array([1.0 if c in 'KR' else -1.0 if c in 'DE' else 0.0 for c in seq])
    win = lambda v: np.array([v[max(0, i - W):i + W + 1].mean() for i in range(n)])  # noqa: E731
    end = np.minimum(np.arange(n), n - 1 - np.arange(n)).astype(float)
    return [end, win(l), win(gs), win(aro), win(q)]


def analyse(slab, pred, lam):
    import mdtraj as md
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    seq, n, name = slab['seq'], slab['N'], slab['name']
    d = os.path.join(FR, slab['set'], name)
    dcd = os.path.join(d, f'{name}_sub.dcd')
    natoms = md.formats.DCDTrajectoryFile(dcd).read(n_frames=1)[0].shape[1]
    assert natoms % n == 0, (name, natoms, n)
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
    X, Wd, Wl = np.zeros(n), np.zeros(n), np.zeros(n)
    Wd_half, Wl_half = [np.zeros(n), np.zeros(n)], [np.zeros(n), np.zeros(n)]
    nd = nl = 0
    nd_half, nl_half = [0, 0], [0, 0]
    rg_d, rg_l = [], []
    share = {'aro': [0.0, 0.0], 'gly': [0.0, 0.0]}
    aro = np.array([c in 'FWY' for c in seq])
    gly = np.array([c == 'G' for c in seq])
    bead_dense = bead_far = 0.0
    for f in range(t.n_frames):
        x = np.mod(t.xyz[f].astype(float), box)
        dz = dzs[f]
        pr = cKDTree(x, boxsize=box).query_pairs(CUT, output_type='ndarray')
        a, b = pr[:, 0], pr[:, 1]
        ca, cb, ia, ib = a // n, b // n, a % n, b % n
        same = ca == cb
        intra = same & (np.abs(ia - ib) >= 3)
        partnered = np.zeros(chains, bool)
        partnered[ca[~same]] = True
        partnered[cb[~same]] = True
        dense = dz < half_width
        dilute = (dz > FAR) & ~partnered
        bz = np.abs((x[:, 2] - (np.angle(np.exp(2j * np.pi * x[:, 2] / L).mean()) % (2 * np.pi)) * L / (2 * np.pi) + L / 2) % L - L / 2)
        bead_dense += (bz < half_width).sum() / (2 * half_width)
        bead_far += (bz > FAR).sum() / max(L - 2 * FAR, 1e-9)
        cnt = lambda sel, mask: np.bincount(ia[sel & mask[ca]], minlength=n) + np.bincount(ib[sel & mask[cb]], minlength=n)  # noqa: E731
        xd = cnt(~same, dense)
        wd = cnt(intra, dense)
        wl = cnt(intra, dilute)
        X += xd
        Wd += wd
        Wl += wl
        Wd_half[f % 2] += wd
        Wl_half[f % 2] += wl
        nd += int(dense.sum())
        nl += int(dilute.sum())
        nd_half[f % 2] += int(dense.sum())
        nl_half[f % 2] += int(dilute.sum())
        for m, key in ((aro, 'aro'), (gly, 'gly')):
            share[key][0] += xd[m].sum()
            share[key][1] += xd[m].sum() + wd[m].sum()
        for c in np.flatnonzero(dense | dilute):
            p = x[c * n:(c + 1) * n].copy()
            dd = np.diff(p, axis=0)
            dd -= box * np.round(dd / box)
            p = np.vstack([p[:1], p[:1] + np.cumsum(dd, axis=0)])
            (rg_d if dense[c] else rg_l).append(float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())))
    X /= max(nd, 1)
    Wd /= max(nd, 1)
    Z = covariates(seq, lam)
    sp = lambda u, v: spearmanr(u, v).correlation  # noqa: E731
    row = dict(slab)
    row.pop('seq')
    row.update({'frames': t.n_frames, 'chains': chains, 'half_width_nm': half_width,
                'ratio_ours': bead_dense / max(bead_far, 1e-12) if bead_far > 0 else float('inf'),
                'dense_snapshots': nd, 'dilute_snapshots': nl,
                'h1_partial': partial(pred, X, Z), 'h1_position_only': partial(pred, X, Z[:1]), 'raw_pred_vs_inter': sp(pred, X),
                'share_between': float(X.sum() / max((X + Wd).sum(), 1e-12))})
    if nl >= MIN_DILUTE:
        Wl /= nl
        hd = [Wd_half[k] / max(nd_half[k], 1) for k in (0, 1)]
        hl = [Wl_half[k] / max(nl_half[k], 1) for k in (0, 1)]
        row.update({'h2_dense_vs_dilute': sp(Wd, Wl), 'h2_ceiling': float(np.sqrt(max(sp(*hd), 0) * max(sp(*hl), 0))),
                    'pred_vs_dilute': sp(pred, Wl), 'rg_dense': float(np.mean(rg_d)), 'rg_dilute': float(np.mean(rg_l)),
                    'h4_rg_ratio': float(np.mean(rg_d) / np.mean(rg_l))})
    if aro.sum() >= 3 and gly.sum() >= 3:
        row['h5_share_aro'] = share['aro'][0] / max(share['aro'][1], 1e-12)
        row['h5_share_gly'] = share['gly'][0] / max(share['gly'][1], 1e-12)
    return row


def tests(rows, out):
    import numpy as np
    from scipy.stats import spearmanr, wilcoxon
    ok = [r for r in rows if r.get('ratio_ours', 0) > 10]
    L = ['Amendment 2: degree in the condensate against single-chain degree (CALVADOS-2 slabs of von Buelow et al. 2025)', '']
    for tier in ('primary', 'replication', 'designed'):
        all_t = [r for r in rows if r['tier'] == tier]
        t = [r for r in ok if r['tier'] == tier]
        L.append(f'{tier}: {len(all_t)} slabs analysed, {len(t)} phase separated in our frames')
        if len(t) >= 5 and tier != 'designed':
            v = np.array([r['h1_partial'] for r in t])
            v = v[~np.isnan(v)]
            w = wilcoxon(v, alternative='greater').pvalue
            word = {'primary': 'CONFIRMED' if w < 0.05 else 'NOT CONFIRMED', 'replication': 'REPLICATED' if w < 0.05 else 'NOT REPLICATED'}[tier]
            L.append(f'  H1 partial (baseline and position): median {np.median(v):+.3f}, {int((v > 0).sum())}/{len(v)} positive, '
                     f'one-sided Wilcoxon P {w:.4g} -> {word}')
            p0 = np.array([r['h1_position_only'] for r in t])
            L.append(f'  H1 with position only: median {np.nanmedian(p0):+.3f}; raw Spearman median {np.nanmedian([r["raw_pred_vs_inter"] for r in t]):+.3f}')
        elif t:
            L.append(f'  descriptive: H1 partial median {np.nanmedian([r["h1_partial"] for r in t]):+.3f} over {len(t)}')
    both = [r for r in ok if r['tier'] in ('primary', 'replication') and not np.isnan(r['h1_partial'])]
    L += ['', 'secondary (primary and replication slabs together):']
    if len(both) >= 5:
        L.append(f'  H3 Spearman(H1 partial, kappa) {spearmanr([r["h1_partial"] for r in both], [r["kappa"] for r in both]).correlation:+.3f} '
                 f'(expected negative); with SCD {spearmanr([r["h1_partial"] for r in both], [r["scd"] for r in both]).correlation:+.3f}; n {len(both)}')
    h2 = [r for r in both if 'h2_dense_vs_dilute' in r]
    if h2:
        L.append(f'  H2 dense vs dilute within-chain profile: median {np.median([r["h2_dense_vs_dilute"] for r in h2]):+.3f} against ceiling '
                 f'{np.median([r["h2_ceiling"] for r in h2]):.3f}; model vs dilute chains median {np.median([r["pred_vs_dilute"] for r in h2]):+.3f}; n {len(h2)}')
    if len(h2) >= 5:
        L.append(f'  H4 Spearman(Rg dense/dilute, nu) {spearmanr([r["h4_rg_ratio"] for r in h2], [r["nu"] for r in h2]).correlation:+.3f} '
                 f'(expected negative); median ratio {np.median([r["h4_rg_ratio"] for r in h2]):.3f}; n {len(h2)}')
    h5 = [r for r in both if 'h5_share_aro' in r]
    if len(h5) >= 5:
        dlt = np.array([r['h5_share_aro'] - r['h5_share_gly'] for r in h5])
        L.append(f'  H5 share between chains, aromatic minus glycine: median {np.median(dlt):+.3f}, {int((dlt > 0).sum())}/{len(dlt)} positive, '
                 f'one-sided Wilcoxon P {wilcoxon(dlt, alternative="greater").pvalue:.4g}; n {len(dlt)}')
    L += ['', 'excluded by our phase-separation criterion: ' + ', '.join(f'{r["name"]} ({r["ratio_ours"]:.1f})' for r in rows if r.get('ratio_ours', 0) <= 10)]
    txt = '\n'.join(L) + '\n'
    open(os.path.join(out, 'summary.txt'), 'w').write(txt)
    print(txt)


def main():
    import numpy as np
    smoke = '--smoke' in sys.argv
    out = OUT + ('_smoke' if smoke else '')
    os.makedirs(out, exist_ok=True)
    pred = np.load(os.path.join(OUT, 'pred_degree.npz'))
    lam = {r['one']: float(r['lambdas']) for r in csv.DictReader(open(CAL))}
    rows = []
    for s in slabs():
        if smoke and s['name'] != PILOT:
            continue
        if not smoke and s['tier'] in ('pilot', 'not phase separating (table)', 'protein in BENDER'):
            continue
        if not os.path.exists(os.path.join(FR, s['set'], s['name'], f'{s["name"]}_sub.dcd')):
            continue
        r = analyse(s, pred[s['name']], lam)
        rows.append(r)
        print(f'{r["tier"]:12s} {r["name"]:22s} chains {r["chains"]:4d} ratio {r["ratio_ours"]:8.1f} H1 {r["h1_partial"]:+.3f}', flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('set', 'name', 'tier'), k))
    with open(os.path.join(out, 'per_slab.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if not smoke:
        tests(rows, out)


if __name__ == '__main__':
    predict() if '--predict' in sys.argv else main()
