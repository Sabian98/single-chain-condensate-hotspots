"""
render_more.py: ChimeraX renders for Figure 2e (dense against dilute chain) and Figure 4e,f (two condensates at matched stability).
Same style as render_saturation.py. Phase definitions follow slab_test.analyse (dense half-width at half maximum of chain-centre density,
dilute beyond 50 nm without partners, contacts 0.8 nm, |i - j| >= 3 within a chain).
  ~/.venvs/finches/bin/python tools/render_more.py
"""
import os
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_chimerax as R    # noqa: E402
import render_saturation as RS  # noqa: E402

ROOT = RS.ROOT
OUT = RS.OUT
CACHE = RS.CACHE
EXAMPLE = 'Q96P88_140_292'
MATCHED = {'uneven': 'Q7Z4F1_487_713', 'even': 'P55316_274_489'}
PALETTE = '0.5,#D6E8F5:1.0,#0077BB:1.5,#00243A'


def phases(name):
    """Time-averaged within-chain profiles in the dense and dilute phases, and per-chain radii of gyration in the last frame."""
    path = os.path.join(CACHE, f'phases_{name}.npz')
    if os.path.exists(path):
        return np.load(path)
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import mdtraj as md
    import slab_test as S
    s = [x for x in S.slabs() if x['name'] == name][0]
    n = s['N']
    dcd = os.path.join(S.FR, s['set'], name, f'{name}_sub.dcd')
    natoms = md.formats.DCDTrajectoryFile(dcd).read(n_frames=1)[0].shape[1]
    nch = natoms // n
    top = md.Topology()
    for _ in range(nch):
        ch = top.add_chain()
        for _ in range(n):
            top.add_atom('CA', md.element.carbon, top.add_residue('ALA', ch))
    t = md.load_dcd(dcd, top=top)
    box = t.unitcell_lengths[0].astype(float)
    L = box[2]
    dzs = []
    for f in range(t.n_frames):
        cz = np.mod(t.xyz[f, :, 2], L).reshape(nch, n).mean(1)
        zc = (np.angle(np.exp(2j * np.pi * cz / L).mean()) % (2 * np.pi)) * L / (2 * np.pi)
        dzs.append(np.abs((cz - zc + L / 2) % L - L / 2))
    dzs = np.array(dzs)
    hist = np.histogram(dzs.ravel(), bins=np.arange(0, L / 2 + 1, 1.0))[0].astype(float)
    half_width = float(max(int(np.argmax(hist < hist.max() / 2)), 1))
    Wd, Wl, nd, nl = np.zeros(n), np.zeros(n), 0, 0
    for f in range(t.n_frames):
        x = np.mod(t.xyz[f].astype(float), box)
        pr = cKDTree(x, boxsize=box).query_pairs(S.CUT, output_type='ndarray')
        a, b = pr[:, 0], pr[:, 1]
        ca, cb, ia, ib = a // n, b // n, a % n, b % n
        same = ca == cb
        intra = same & (np.abs(ia - ib) >= 3)
        partnered = np.zeros(nch, bool)
        partnered[ca[~same]] = True
        partnered[cb[~same]] = True
        dense = dzs[f] < half_width
        dilute = (dzs[f] > S.FAR) & ~partnered
        cnt = lambda mask: np.bincount(ia[intra & mask[ca]], minlength=n) + np.bincount(ib[intra & mask[cb]], minlength=n)  # noqa: E731
        Wd += cnt(dense)
        Wl += cnt(dilute)
        nd += int(dense.sum())
        nl += int(dilute.sum())
    x = np.mod(t.xyz[-1].astype(float), box)
    rg = []
    for c in range(nch):
        p = x[c * n:(c + 1) * n].copy()
        d = np.diff(p, axis=0)
        d -= box * np.round(d / box)
        p = np.vstack([p[:1], p[:1] + np.cumsum(d, axis=0)])
        rg.append(float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())))
    pr = cKDTree(x, boxsize=box).query_pairs(S.CUT, output_type='ndarray')
    partnered = np.zeros(nch, bool)
    other = pr[:, 0] // n != pr[:, 1] // n
    partnered[pr[other, 0] // n] = True
    partnered[pr[other, 1] // n] = True
    np.savez_compressed(path, Wd=Wd / nd, Wl=Wl / nl, rg=np.array(rg), dz_last=dzs[-1], half_width=half_width,
                        partnered_last=partnered, n_dense=nd, n_dilute=nl)
    return np.load(path)


def whole_chain(xyz, box, n, c):
    p = xyz[c * n:(c + 1) * n].copy()
    d = np.diff(p, axis=0)
    d -= box * np.round(d / box)
    p = np.vstack([p[:1], p[:1] + np.cumsum(d, axis=0)])
    return (p - p.mean(0)) * 10.0


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ['set bgColor white']
    # Figure 2e: representative dense and dilute chains of the example slab
    ph = phases(EXAMPLE)
    d = RS.cached(EXAMPLE)
    xyz, box, n, seq = d['xyz'], d['box'], int(d['n']), ''.join(d['seq'])
    rg = ph['rg']
    dense_ch = np.flatnonzero(ph['dz_last'] < ph['half_width'])
    dil_ch = np.flatnonzero((ph['dz_last'] > 50) & ~ph['partnered_last'])
    # mean dense and dilute radii of gyration from the per-slab table, and the chain closest to each
    import csv
    row = [r for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'amendment2', 'per_slab.csv'))) if r['name'] == EXAMPLE][0]
    rgd, rgl = float(row['rg_dense']), float(row['rg_dilute'])
    cd = int(dense_ch[np.argmin(np.abs(rg[dense_ch] - rgd))])
    cl = int(dil_ch[np.argmin(np.abs(rg[dil_ch] - rgl))])
    print(f'Fig 2e: dense chain Rg {rg[cd]:.2f} (phase mean {rgd:.2f}), dilute chain Rg {rg[cl]:.2f} (phase mean {rgl:.2f}) nm; '
          f'dense vs dilute profile Spearman {np.corrcoef(np.argsort(np.argsort(ph["Wd"])), np.argsort(np.argsort(ph["Wl"])))[0, 1]:.2f}')
    for key, c, prof in (('dense', cd, ph['Wd']), ('dilute', cl, ph['Wl'])):
        P = whole_chain(xyz, box, n, c)[None]
        pdb = os.path.join(OUT, f'phase_{key}.pdb')
        R.write_pdb(pdb, [0], P, seq, bfac={0: np.clip(prof / prof.mean(), 0.5, 1.5)})
        png = os.path.join(OUT, f'phase_{key}.png')
        if os.path.exists(png):
            os.remove(png)
        lines += [f'open {pdb}', 'cartoon hide #1', 'show #1 atoms', 'style #1 sphere', 'size #1 atomRadius 3.4', f'color bfactor #1 palette {PALETTE}', 'lighting soft',
                  'graphics silhouettes true width 1.5', 'camera ortho', 'windowsize 900 900', 'view #1', 'zoom pixelSize 0.1667',
                  'clip off', f'save {png} width 1800 supersample 3', 'close']
    # Figure 4e,f: two condensates at matched dilute concentration and length
    for key, name in MATCHED.items():
        dd = RS.cached(name)
        P, sq, c0, partners = RS.chain_and_partners(dd)
        rel = np.asarray(dd['X']) / np.mean(dd['X'])
        bf = {c0: np.clip(rel, 0.5, 1.5)}
        for c in partners:
            bf[c] = np.full(P.shape[1], -1.0)
        pdb = os.path.join(OUT, f'matched_{key}.pdb')
        R.write_pdb(pdb, [c0] + partners, P, sq, bfac=bf)
        png = os.path.join(OUT, f'matched_{key}.png')
        if os.path.exists(png):
            os.remove(png)
        print(f'Fig 4: {key} {name}: relative contacts {rel.min():.2f} to {rel.max():.2f}, CV {rel.std():.3f}, {len(partners)} partners')
        lines += [f'open {pdb}', 'cartoon hide #1', 'show #1 atoms', 'style #1 sphere', 'size #1 atomRadius 1.3', 'size #1/A atomRadius 3.4', 'color #1 #B4B4B4',
                  f'color bfactor #1/A palette {PALETTE}', 'hide #1 & ~(/A :<12) atoms', 'lighting soft',
                  'graphics silhouettes true width 1.5', 'camera ortho', 'windowsize 900 900', 'view /A', 'zoom pixelSize 0.2611',
                  'clip off', f'save {png} width 1800 supersample 3', 'close']
    lines.append('exit')
    cxc = os.path.join(OUT, 'render_more.cxc')
    open(cxc, 'w').write('\n'.join(lines) + '\n')
    subprocess.run(['open', '-n', '-a', '/Applications/ChimeraX-1.12.app', cxc], check=True)
    t0 = time.time()
    while time.time() - t0 < 300 and not os.path.exists(os.path.join(OUT, 'matched_even.png')):
        time.sleep(3)
    time.sleep(3)
    print('renders:', sorted(f for f in os.listdir(OUT) if f.startswith(('phase_', 'matched_')) and f.endswith('.png')))


if __name__ == '__main__':
    main()
