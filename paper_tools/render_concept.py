"""
render_concept.py: ChimeraX render of one GNRHR2 single chain for the conceptual Figure 1, from the IDRome CALVADOS-2 trajectory
(Tesei et al. 2024). The frame closest to the ensemble-mean radius of gyration is drawn as beads coloured by time-averaged
single-chain degree (relative to its mean, grey to black), with the within-chain contacts of that frame (Calpha < 0.8 nm,
|i - j| >= 6 for legibility) as thin grey pseudobonds. Writes figs/render/{single.pdb, single.pb, single.png} and
results/figure_cache/single_degree_Q96P88_140_292.npz (degree profile and frame index).
  ~/.venvs/finches/bin/python tools/render_concept.py
"""
import os
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.distance import pdist, squareform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_chimerax as R  # noqa: E402

ROOT = '/Volumes/User Homes/Nu Project/DYNASEP/hub_condensate_followup'
NAME = 'Q96P88_140_292'
OUT = os.path.join(os.path.dirname(HERE), 'figs', 'render')
CACHE = os.path.join(ROOT, 'results', 'figure_cache', f'single_degree_{NAME}.npz')


def main():
    import mdtraj as md
    d = os.path.join(ROOT, 'data', 'idrome_single', NAME)
    t = md.load(os.path.join(d, 'traj.xtc'), top=os.path.join(d, 'top.pdb'))
    n = t.n_atoms
    idx = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    deg, rg = np.zeros(n), []
    frames = []
    for f in range(t.n_frames):
        x = t.xyz[f].astype(float)
        if t.unitcell_lengths is not None:
            b = t.unitcell_lengths[f].astype(float)
            dd = np.diff(x, axis=0)
            dd -= b * np.round(dd / b)
            x = np.vstack([x[:1], x[:1] + np.cumsum(dd, axis=0)])
        frames.append(x)
        D = squareform(pdist(x))
        deg += ((D < 0.8) & (idx >= 3)).sum(1)
        rg.append(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean()))
    deg /= t.n_frames
    rg = np.array(rg)
    f0 = int(np.argmin(np.abs(rg - rg.mean())))
    x = frames[f0]
    D = squareform(pdist(x))
    pairs = [(i, j) for i in range(n) for j in range(i + 6, n) if D[i, j] < 0.8]
    seq = ''.join(np.load(os.path.join(ROOT, 'results', 'figure_cache', 'fig1_example.npz'))['seq'])   # same fragment as the slab
    assert len(seq) == n, (len(seq), n)
    np.savez_compressed(CACHE, degree=deg, frame=f0, rg=rg[f0], rg_mean=rg.mean(), seq=np.array(list(seq)))
    os.makedirs(OUT, exist_ok=True)
    P = ((x - x.mean(0)) * 10.0)[None]
    rel = deg / deg.mean()
    R.write_pdb(os.path.join(OUT, 'single.pdb'), [0], P, seq, bfac={0: np.clip(rel, 0.5, 1.5)})
    with open(os.path.join(OUT, 'single.pb'), 'w') as fh:
        fh.write('; radius = 0.5\n; color = #333333\n; dashes = 8\n')
        for i, j in pairs:
            fh.write(f'/A:{i + 1}@CA /A:{j + 1}@CA\n')
    png = os.path.join(OUT, 'single.png')
    if os.path.exists(png):
        os.remove(png)
    cxc = os.path.join(OUT, 'render_concept.cxc')
    open(cxc, 'w').write('\n'.join([
        'set bgColor white', f"open {os.path.join(OUT, 'single.pdb')}", 'cartoon hide #1', 'show #1 atoms', 'style #1 sphere',
        'size #1 atomRadius 2.1', 'color bfactor #1 palette 0.5,#E8E8E8:1.0,#9A9A9A:1.5,#202020',
        f"open {os.path.join(OUT, 'single.pb')}", 'lighting soft', 'graphics silhouettes true width 1.5', 'camera ortho',
        'windowsize 900 900', 'view #1', 'clip off', f'save {png} width 1800 supersample 3', 'exit']) + '\n')
    subprocess.run(['open', '-n', '-a', '/Applications/ChimeraX-1.12.app', cxc], check=True)
    t0 = time.time()
    while time.time() - t0 < 300 and not os.path.exists(png):
        time.sleep(3)
    time.sleep(2)
    print(f'frame {f0}, Rg {rg[f0]:.2f} nm (mean {rg.mean():.2f}); {len(pairs)} contacts with |i-j| >= 6; render {os.path.exists(png)}')


if __name__ == '__main__':
    main()
