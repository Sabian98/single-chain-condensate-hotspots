"""
slab_pilot_dilute.py: exploratory pilot, second question (plans/slab_pilot_loricrin.md). Do loricrin chains differ between the dilute
and the dense phase of the same slab? Same frames, contact rule and slab centre as slab_pilot_loricrin.py.
Dilute: chain centre more than 40 nm from the slab centre and no contact with another chain in that frame. Dense: within 15 nm.
Per chain and frame: radius of gyration and within-chain degree per position; the dilute profile's noise is estimated by splitting its
chain-snapshots into two halves (even and odd frames), giving the correlation the dense profile could reach at best.
  ~/.venvs/finches/bin/python slab_pilot_dilute.py   ->   ../results/pilot_loricrin/{dilute_summary.txt, dilute_per_residue.csv, dilute.png}
"""
import csv
import os

import mdtraj as md
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results', 'pilot_loricrin')
DCD = os.path.join(ROOT, 'data/erda_pspred/IDRome10/P23490_1_312/P23490_1_312_sub.dcd')
FASTA = os.path.join(ROOT, 'gaivi/pilot_loricrin_293K.fasta')
CHAINS, CUT, CORE, FAR = 115, 0.8, 15.0, 40.0


def main():
    seq = ''.join(l.strip() for l in open(FASTA) if not l.startswith('>'))
    n = len(seq)
    top = md.Topology()
    for _ in range(CHAINS):
        ch = top.add_chain()
        for _ in seq:
            top.add_atom('CA', md.element.carbon, top.add_residue('ALA', ch))
    t = md.load_dcd(DCD, top=top)
    box = t.unitcell_lengths[0].astype(float)
    L = box[2]
    deg = {'dense': [], 'dilute': []}          # (frame, profile)
    rg = {'dense': [], 'dilute': []}
    for f in range(t.n_frames):
        x = np.mod(t.xyz[f].astype(float), box)
        cz = x[:, 2].reshape(CHAINS, n).mean(1)
        zc = (np.angle(np.exp(2j * np.pi * cz / L).mean()) % (2 * np.pi)) * L / (2 * np.pi)
        dz = np.abs((cz - zc + L / 2) % L - L / 2)
        pr = cKDTree(x, boxsize=box).query_pairs(CUT, output_type='ndarray')
        a, b = pr[:, 0], pr[:, 1]
        ca, cb, ia, ib = a // n, b // n, a % n, b % n
        partnered = np.zeros(CHAINS, bool)
        partnered[ca[ca != cb]] = True
        partnered[cb[ca != cb]] = True
        intra = (ca == cb) & (np.abs(ia - ib) >= 3)
        for c in range(CHAINS):
            phase = 'dense' if dz[c] < CORE else 'dilute' if (dz[c] > FAR and not partnered[c]) else None
            if phase is None:
                continue
            m = intra & (ca == c)
            prof = np.bincount(ia[m], minlength=n) + np.bincount(ib[m], minlength=n)
            deg[phase].append((f, prof))
            # radius of gyration of the unwrapped chain (consecutive beads are closer than half a box)
            xyz = x[c * n:(c + 1) * n].copy()
            d = np.diff(xyz, axis=0)
            d -= box * np.round(d / box)
            xyz = np.vstack([xyz[:1], xyz[:1] + np.cumsum(d, axis=0)])
            rg[phase].append(float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean())))
    pred = np.load(os.path.join(OUT, 'pred_degree.npy'))
    mean = {k: np.mean([p for _, p in v], 0) for k, v in deg.items()}
    half = {k: [np.mean([p for f, p in v if f % 2 == h], 0) for h in (0, 1)] for k, v in deg.items()}
    sp = lambda u, v: spearmanr(u, v).correlation  # noqa: E731
    sh = {k: sp(*half[k]) for k in deg}
    ceiling = np.sqrt(sh['dilute'] * sh['dense'])
    end = np.minimum(np.arange(n), n - 1 - np.arange(n))
    mid = end >= 20
    L_ = [f'Loricrin: dilute against dense phase of the same slab (exploratory), {t.n_frames} frames', '',
          f'chain-snapshots: dense {len(deg["dense"])}, dilute {len(deg["dilute"])} (centre beyond {FAR:.0f} nm and no contact with another chain)',
          f'radius of gyration (nm), mean and SD over snapshots: dense {np.mean(rg["dense"]):.2f} +- {np.std(rg["dense"]):.2f}, '
          f'dilute {np.mean(rg["dilute"]):.2f} +- {np.std(rg["dilute"]):.2f}',
          f'within-chain degree per residue, mean: dense {mean["dense"].mean():.2f}, dilute {mean["dilute"].mean():.2f}', '',
          'Spearman across positions:',
          f'  dense vs dilute within-chain profile        {sp(mean["dense"], mean["dilute"]):+.3f}   (all positions)   '
          f'{sp(mean["dense"][mid], mean["dilute"][mid]):+.3f}   (20 or more from the ends)',
          f'  predicted single-chain vs dilute             {sp(pred, mean["dilute"]):+.3f}',
          f'  predicted single-chain vs dense              {sp(pred, mean["dense"]):+.3f}',
          f'  split-half reliability: dilute {sh["dilute"]:+.3f}, dense {sh["dense"]:+.3f}; best reachable dense-dilute correlation about {ceiling:.3f}', '',
          'ratio dense/dilute within-chain degree by residue type:']
    for c in sorted(set(seq), key=lambda c: -seq.count(c))[:10]:
        m = np.array([s == c for s in seq])
        L_.append(f'  {c} (n {m.sum():3d})  dense {mean["dense"][m].mean():.2f}  dilute {mean["dilute"][m].mean():.2f}  ratio {mean["dense"][m].mean() / mean["dilute"][m].mean():.2f}')
    ratio = mean['dense'] / np.maximum(mean['dilute'], 1e-9)
    L_ += ['', f'ratio by distance from the chain end: ends (under 20) {ratio[~mid].mean():.2f}, middle {ratio[mid].mean():.2f}']
    txt = '\n'.join(L_) + '\n'
    open(os.path.join(OUT, 'dilute_summary.txt'), 'w').write(txt)
    print(txt)
    with open(os.path.join(OUT, 'dilute_per_residue.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['pos', 'aa', 'pred_single_chain', 'intra_dense', 'intra_dilute'])
        w.writerows([(i + 1, seq[i], pred[i], mean['dense'][i], mean['dilute'][i]) for i in range(n)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    x = np.arange(1, n + 1)
    sm = lambda v: np.convolve(v, np.ones(5) / 5, mode='same')  # noqa: E731
    ax[0].plot(x, sm(mean['dense']), color='#4C9BD3', lw=1.3, label=f'Dense phase ({len(deg["dense"])} snapshots)')
    ax[0].plot(x, sm(mean['dilute']), color='#E07B39', lw=1.3, label=f'Dilute phase ({len(deg["dilute"])} snapshots)')
    ax[0].set_ylabel('Contacts within the chain\n(5-residue mean)')
    ax[0].legend(frameon=False, fontsize=8, loc='upper right')
    ax[1].plot(x, sm(ratio), color='0.3', lw=1.2)
    ax[1].axhline(1, color='0.7', lw=0.8, ls='--')
    ax[1].set_ylabel('Dense / dilute')
    ax[1].set_xlabel('Residue')
    ax[0].set_title('Loricrin, dilute against dense phase of one slab (exploratory pilot)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dilute.png'), dpi=150)


if __name__ == '__main__':
    main()
