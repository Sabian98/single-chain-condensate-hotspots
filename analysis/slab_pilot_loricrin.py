"""
slab_pilot_loricrin.py: exploratory pilot (plans/slab_pilot_loricrin.md). How does residue contact degree change between the single
chain and the condensate, for loricrin (P23490_1_312), from 200 frames of the von Buelow et al. (PNAS 2025) CALVADOS-2 slab?

Contacts as in the paper: Calpha-Calpha < 0.8 nm (minimum image), |i - j| >= 3 within a chain, every pair between chains.
Dense core: chains whose centre lies within 15 nm of the slab centre (circular mean along z); interface 15 to 25 nm.
Per sequence position, averaged over core chains and frames: intra-chain degree, inter-chain degree, total.
Single-chain reference: the degree model's prediction for the sequence (degree_consensus_2026-10-13.pkl, CALVADOS-2 single chains at
300 K; the slab ran at 293 K and pH 7.0), computed by pred_loricrin_degree.npy (written by the --predict step in the bender env).
Controls for the obvious: a 7-residue window of CALVADOS-2 stickiness (lambda) and the distance from the chain end.
  ~/.venvs/bender/bin/python slab_pilot_loricrin.py --predict      (once, writes ../results/pilot_loricrin/pred_degree.npy)
  ~/.venvs/finches/bin/python slab_pilot_loricrin.py               -> ../results/pilot_loricrin/{summary.txt, per_residue.csv, pilot.png}
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results', 'pilot_loricrin')
DCD = os.path.join(ROOT, 'data/erda_pspred/IDRome10/P23490_1_312/P23490_1_312_sub.dcd')
FASTA = os.path.join(ROOT, 'gaivi/pilot_loricrin_293K.fasta')
EV = '/Volumes/User Homes/Nu Project/BENDER-DYN/redo_evidence'
CAL = os.path.expanduser('~/Downloads/Scripts/CALVADOS IDP/input/residues_CALVADOS2.csv')
CHAINS, CUT, CORE, IFACE = 115, 0.8, 15.0, 25.0


def sequence():
    return ''.join(l.strip() for l in open(FASTA) if not l.startswith('>'))


def predict():
    import pickle
    import numpy as np
    sys.path.insert(0, os.path.join(EV, 'hub_predictor'))
    import train_hub_model as T
    model = pickle.load(open(os.path.join(EV, 'hub_predictor', 'degree_consensus_2026-10-13.pkl'), 'rb'))
    os.makedirs(OUT, exist_ok=True)
    p = model.predict(T.features(sequence(), 10))
    np.save(os.path.join(OUT, 'pred_degree.npy'), p)
    print('predicted degree profile', p.shape, float(p.mean()), float(p.std()))


def partial(x, y, Z):
    """Spearman of x and y after removing the ranks of the columns of Z (least squares on ranks, with intercept)."""
    import numpy as np
    from scipy.stats import rankdata
    A = np.column_stack([np.ones(len(x))] + [rankdata(z) for z in Z])
    rx, ry = rankdata(x), rankdata(y)
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(rankdata(ex), rankdata(ey))[0, 1])


def main():
    import mdtraj as md
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    seq = sequence()
    n = len(seq)
    top = md.Topology()
    for _ in range(CHAINS):
        ch = top.add_chain()
        for _ in seq:
            top.add_atom('CA', md.element.carbon, top.add_residue('ALA', ch))
    t = md.load_dcd(DCD, top=top)
    box = t.unitcell_lengths[0].astype(float)
    L = box[2]
    intra = {k: np.zeros(n) for k in ('core', 'iface')}
    inter = {k: np.zeros(n) for k in ('core', 'iface')}
    count = {k: 0 for k in ('core', 'iface')}
    pairtype = {}
    for f in range(t.n_frames):
        x = np.mod(t.xyz[f].astype(float), box)
        cz = x[:, 2].reshape(CHAINS, n).mean(1)
        zc = (np.angle(np.exp(2j * np.pi * cz / L).mean()) % (2 * np.pi)) * L / (2 * np.pi)
        dz = np.abs((cz - zc + L / 2) % L - L / 2)
        zone = np.where(dz < CORE, 'core', np.where(dz < IFACE, 'iface', 'dilute'))
        pr = cKDTree(x, boxsize=box).query_pairs(CUT, output_type='ndarray')
        a, b = pr[:, 0], pr[:, 1]
        ca, cb, ia, ib = a // n, b // n, a % n, b % n
        same = ca == cb
        keep_intra = same & (np.abs(ia - ib) >= 3)
        for k in ('core', 'iface'):
            m = zone == k
            count[k] += int(m.sum())
            for sel, acc in ((keep_intra, intra[k]), (~same, inter[k])):
                for cc, ii in ((ca, ia), (cb, ib)):
                    s = sel & m[cc]
                    acc += np.bincount(ii[s], minlength=n)
        # residue-type pairs of contacts between core chains
        s = (~same) & (zone[ca] == 'core') & (zone[cb] == 'core')
        for u, v in zip(np.array(list(seq))[ia[s]], np.array(list(seq))[ib[s]]):
            key = tuple(sorted((u, v)))
            pairtype[key] = pairtype.get(key, 0) + 1
    for k in ('core', 'iface'):
        intra[k] /= count[k]
        inter[k] /= count[k]
    pred = np.load(os.path.join(OUT, 'pred_degree.npy'))
    lam = {r['one']: float(r['lambdas']) for r in csv.DictReader(open(CAL))}
    lw = np.convolve(np.array([lam[c] for c in seq]), np.ones(7) / 7, mode='same')
    end = np.minimum(np.arange(n), n - 1 - np.arange(n)).astype(float)
    tot = intra['core'] + inter['core']
    rows = []
    for i in range(n):
        rows.append({'pos': i + 1, 'aa': seq[i], 'pred_single_chain': pred[i], 'intra_core': intra['core'][i], 'inter_core': inter['core'][i],
                     'total_core': tot[i], 'intra_interface': intra['iface'][i], 'inter_interface': inter['iface'][i], 'lambda_window7': lw[i]})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'per_residue.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sp = lambda u, v: spearmanr(u, v).correlation  # noqa: E731
    L_ = [f'Loricrin slab pilot (exploratory), {t.n_frames} frames, {count["core"] / t.n_frames:.1f} core chains per frame, '
          f'{count["iface"] / t.n_frames:.1f} at the interface', '',
          f'mean degree per residue in the core: within the chain {intra["core"].mean():.2f}, with other chains {inter["core"].mean():.2f} '
          f'(share between chains {inter["core"].sum() / tot.sum():.2f})',
          f'interface: within {intra["iface"].mean():.2f}, between {inter["iface"].mean():.2f}', '',
          'Spearman across the 312 positions (raw | given stickiness window and distance from the end):']
    for name, u, v in (('within-chain vs between-chain degree (core)', intra['core'], inter['core']),
                       ('predicted single-chain degree vs within-chain (core)', pred, intra['core']),
                       ('predicted single-chain degree vs between-chain (core)', pred, inter['core']),
                       ('predicted single-chain degree vs total (core)', pred, tot),
                       ('stickiness window vs between-chain (core)', lw, inter['core']),
                       ('core vs interface, between-chain', inter['core'], inter['iface'])):
        L_.append(f'  {name:55s} {sp(u, v):+.3f} | {partial(u, v, [lw, end]) if "stickiness window vs" not in name else partial(u, v, [end]):+.3f}')
    L_ += ['', 'mean degree by residue type in the core (within | between | share between):']
    for c in sorted(set(seq), key=lambda c: -seq.count(c)):
        m = np.array([s == c for s in seq])
        L_.append(f'  {c} (n {m.sum():3d})  {intra["core"][m].mean():5.2f} | {inter["core"][m].mean():5.2f} | {inter["core"][m].sum() / tot[m].sum():.2f}'
                  f'   predicted single-chain {pred[m].mean():+.2f}')
    comp = {c: seq.count(c) / n for c in set(seq)}
    totp = sum(pairtype.values())
    enr = sorted(((v / totp) / (comp[k[0]] * comp[k[1]] * (1 if k[0] == k[1] else 2)), k, v) for k, v in pairtype.items() if v > 200)
    L_ += ['', 'contacts between core chains by residue-type pair, enrichment over composition (most enriched):']
    L_ += [f'  {a}-{b}  {e:.2f}  (n {v})' for e, (a, b), v in enr[::-1][:10]]
    L_ += ['least enriched:'] + [f'  {a}-{b}  {e:.2f}  (n {v})' for e, (a, b), v in enr[:6]]
    top10 = np.argsort(-tot)[:round(0.1 * n)]
    L_ += ['', 'composition of the 10% highest-degree positions in the core: ' +
           ', '.join(f'{c} {sum(seq[i] == c for i in top10)}' for c in sorted(set(seq[i] for i in top10), key=lambda c: -sum(seq[i] == c for i in top10)))]
    txt = '\n'.join(L_) + '\n'
    open(os.path.join(OUT, 'summary.txt'), 'w').write(txt)
    print(txt)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    x = np.arange(1, n + 1)
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    ax[0].plot(x, z(pred), color='0.4', lw=1.2, label='Predicted single-chain degree')
    ax[0].plot(x, z(intra['core']), color='#4C9BD3', lw=1.2, label='Within the chain, condensate')
    ax[0].set_ylabel('Degree (standardized)')
    ax[0].legend(frameon=False, fontsize=8, loc='upper right')
    ax[1].plot(x, z(inter['core']), color='#E07B39', lw=1.2, label='With other chains, condensate')
    ax[1].plot(x, z(lw), color='0.6', lw=1, ls='--', label='Stickiness, 7-residue window')
    ax[1].set_ylabel('Standardized')
    ax[1].legend(frameon=False, fontsize=8, loc='upper right')
    cols = {'G': '#9ec9e2', 'S': '#cfe6c4', 'Y': '#E07B39', 'F': '#E07B39', 'K': '#5b7fbf', 'R': '#5b7fbf'}
    ax[2].bar(x, inter['core'] / tot, color=[cols.get(c, '0.8') for c in seq], width=1)
    ax[2].set_ylabel('Share of contacts\nwith other chains')
    ax[2].set_xlabel('Residue')
    ax[0].set_title('Loricrin in its condensate (CALVADOS-2 slab, exploratory pilot)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'pilot.png'), dpi=150)


if __name__ == '__main__':
    predict() if '--predict' in sys.argv else main()
