"""
render_chimerax.py: ChimeraX renders of the example slab for Figure 1a,b, in the style of the CALVADOS software paper
(von Buelow et al. 2025, arXiv 2504.10408, Fig. 6C): whole chains as overlapping beads, soft lighting, thin black box, white background.
Writes figs/render/{slab.pdb, box.pdb, zoom.pdb, render.cxc}, launches ChimeraX through macOS `open` (running the binary directly hangs),
waits for slab.png and zoom.png, and crops their white margins.
  ~/.venvs/finches/bin/python tools/render_chimerax.py
"""
import os
import subprocess
import time

import numpy as np
from scipy.spatial import cKDTree

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PAPER, 'figs', 'render')
CACHE = '/Volumes/User Homes/Nu Project/DYNASEP/hub_condensate_followup/results/figure_cache/fig1_example.npz'
THREE = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
         'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
CHAINS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


def frame():
    d = np.load(CACHE)
    xyz, box, n, seq = d['xyz'], d['box'], int(d['n']), ''.join(d['seq'])
    L = box[2]
    nch = len(xyz) // n
    cz = xyz[:, 2].reshape(nch, n).mean(1)
    zc = (np.angle(np.exp(2j * np.pi * cz / L).mean()) % (2 * np.pi)) * L / (2 * np.pi)
    zz = (xyz[:, 2] - zc + L / 2) % L
    # contacts of the chain nearest the slab centre with other chains, in this frame (0.8 nm, minimum image)
    c0 = int(np.argmin(np.abs(((cz - zc + L / 2) % L) - L / 2)))
    pr = cKDTree(xyz, boxsize=box).query_pairs(0.8, output_type='ndarray')
    ca, cb = pr[:, 0] // n, pr[:, 1] // n
    other = ca != cb
    cnt = np.bincount(pr[other & (ca == c0), 0] % n, minlength=n) + np.bincount(pr[other & (cb == c0), 1] % n, minlength=n)
    partners = sorted(set(np.concatenate([cb[other & (ca == c0)], ca[other & (cb == c0)]]).tolist()))
    # whole molecules in image axes: horizontal = z (slab long axis), vertical = x, depth = y
    P = np.column_stack([zz, xyz[:, 0], xyz[:, 1]]).reshape(nch, n, 3)
    per = np.array([L, box[0], box[1]])
    stp = np.diff(P, axis=1)
    stp -= per * np.round(stp / per)
    P = np.concatenate([P[:, :1], P[:, :1] + np.cumsum(stp, axis=1)], axis=1)
    P -= np.floor(P.mean(1, keepdims=True) / per) * per
    return P * 10.0, per * 10.0, seq, c0, cnt, partners          # angstrom


def write_pdb(path, chains, P, seq, bfac=None):
    with open(path, 'w') as fh:
        k = 0
        for j, c in enumerate(chains):
            cid = CHAINS[j % len(CHAINS)]
            for i in range(P.shape[1]):
                k += 1
                b = 0.0 if bfac is None else float(bfac[c][i])
                x, y, z = P[c, i]
                fh.write(f'ATOM  {k % 100000:5d}  CA  {THREE[seq[i]]} {cid}{(i + 1) % 10000:4d}    {x:8.2f}{y:8.2f}{z:8.2f}  1.00{b:6.2f}           C\n')
            fh.write('TER\n')
        fh.write('END\n')


def write_box(path, per):
    corners = [(x, y, z) for x in (0, per[0]) for y in (0, per[1]) for z in (0, per[2])]
    with open(path, 'w') as fh:
        for i, (x, y, z) in enumerate(corners):
            fh.write(f'HETATM{i + 1:5d}  C   BOX X   1    {x:8.2f}{y:8.2f}{z:8.2f}  1.00  0.00           C\n')
        for i, a in enumerate(corners):
            for j, b in enumerate(corners):
                if j > i and sum(abs(p - q) > 1e-6 for p, q in zip(a, b)) == 1:
                    fh.write(f'CONECT{i + 1:5d}{j + 1:5d}\n')
        fh.write('END\n')


def main():
    os.makedirs(OUT, exist_ok=True)
    P, per, seq, c0, cnt, partners = frame()
    nch = P.shape[0]
    write_pdb(os.path.join(OUT, 'slab.pdb'), list(range(nch)), P, seq)
    write_box(os.path.join(OUT, 'box.pdb'), per)
    bf = {c0: cnt}
    for c in partners:
        bf[c] = np.full(P.shape[1], -1.0)
    write_pdb(os.path.join(OUT, 'zoom.pdb'), [c0] + partners, P, seq, bfac=bf)
    for f in ('slab.png', 'zoom.png'):
        if os.path.exists(os.path.join(OUT, f)):
            os.remove(os.path.join(OUT, f))
    cxc = f"""set bgColor white
open {OUT}/slab.pdb
open {OUT}/box.pdb
style #1 sphere
size #1 atomRadius 3.2
color #1 #0077BB
style #2 stick
size #2 stickRadius 2.5
color #2 black
hide #2 atoms
show #2 bonds
lighting soft
lighting shadows false
graphics silhouettes true width 1.2
camera ortho
view
windowsize 1600 420
save {OUT}/slab.png width 3200 supersample 3
close
open {OUT}/zoom.pdb
style #1 sphere
size #1 atomRadius 2.4
size #1/A atomRadius 3.6
color #1 #C8C8C8
color bfactor #1/A palette 0,#8EC3E6:3,#0077BB:6,#00243A
transparency #1&~/A 55 target a
hide #1 & ~(/A :<15) atoms
lighting soft
graphics silhouettes true width 1.5
camera ortho
view /A pad 0.15
clip off
windowsize 900 900
save {OUT}/zoom.png width 1800 supersample 3
exit
"""
    open(os.path.join(OUT, 'render.cxc'), 'w').write(cxc)
    subprocess.run(['open', '-n', '-a', '/Applications/ChimeraX-1.12.app', os.path.join(OUT, 'render.cxc')], check=True)
    t0 = time.time()
    while time.time() - t0 < 300 and not os.path.exists(os.path.join(OUT, 'zoom.png')):
        time.sleep(3)
    time.sleep(3)
    print('slab.png', os.path.exists(os.path.join(OUT, 'slab.png')), '| zoom.png', os.path.exists(os.path.join(OUT, 'zoom.png')),
          f'| highlighted chain contacts max {int(cnt.max())}, partners {len(partners)}')


if __name__ == '__main__':
    main()
