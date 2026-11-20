"""
render_saturation.py: ChimeraX renders for Figure 3e,f, one chain from a typical condensate (GNRHR2) and one from a saturated
condensate (KRTAP6-2), each coloured by the time-averaged contacts of its residues with other chains relative to the chain mean, on
one shared scale, with the beads of nearby chains (within 1.2 nm) as small grey beads, at one common scale. Same render style as render_chimerax.py.
  ~/.venvs/finches/bin/python tools/render_saturation.py
"""
import os
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_chimerax as R  # noqa: E402

ROOT = '/Volumes/User Homes/Nu Project/DYNASEP/hub_condensate_followup'
CACHE = os.path.join(ROOT, 'results', 'figure_cache')
OUT = os.path.join(os.path.dirname(HERE), 'figs', 'render')
PAIR = {'typical': 'Q96P88_140_292', 'krtap': 'Q3LI66_1_62'}
LO, HI = 0.5, 1.5                                                # relative contacts with other chains, shared colour scale
FIELD = 130                                                      # field width in angstrom, the same for both renders


def cached(name):
    """Final frame and time-averaged between-chain profile of one slab, cached on the NAS."""
    path = os.path.join(CACHE, f'sat_{name}.npz')
    if not os.path.exists(path):
        sys.path.insert(0, os.path.join(ROOT, 'scripts'))
        import mdtraj as md
        import slab_test as S
        import explore_saturation as ES
        s = [x for x in S.slabs() if x['name'] == name][0]
        X, W = ES.profiles(s)
        n = s['N']
        dcd = os.path.join(S.FR, s['set'], name, f'{name}_sub.dcd')
        natoms = md.formats.DCDTrajectoryFile(dcd).read(n_frames=1)[0].shape[1]
        top = md.Topology()
        for _ in range(natoms // n):
            ch = top.add_chain()
            for _ in range(n):
                top.add_atom('CA', md.element.carbon, top.add_residue('ALA', ch))
        t = md.load_dcd(dcd, top=top)
        box = t.unitcell_lengths[-1]
        np.savez_compressed(path, X=X, W=W, xyz=np.mod(t.xyz[-1], box), box=box, n=n, seq=np.array(list(s['seq'])))
    return np.load(path)


def chain_and_partners(d):
    xyz, box, n, seq = d['xyz'], d['box'], int(d['n']), ''.join(d['seq'])
    L = box[2]
    nch = len(xyz) // n
    cz = xyz[:, 2].reshape(nch, n).mean(1)
    zc = (np.angle(np.exp(2j * np.pi * cz / L).mean()) % (2 * np.pi)) * L / (2 * np.pi)
    zz = (xyz[:, 2] - zc + L / 2) % L
    c0 = int(np.argmin(np.abs(((cz - zc + L / 2) % L) - L / 2)))
    pr = cKDTree(xyz, boxsize=box).query_pairs(0.8, output_type='ndarray')
    ca, cb = pr[:, 0] // n, pr[:, 1] // n
    other = ca != cb
    partners = sorted(set(np.concatenate([cb[other & (ca == c0)], ca[other & (cb == c0)]]).tolist()))
    P = np.column_stack([zz, xyz[:, 0], xyz[:, 1]]).reshape(nch, n, 3)
    per = np.array([L, box[0], box[1]])
    stp = np.diff(P, axis=1)
    stp -= per * np.round(stp / per)
    P = np.concatenate([P[:, :1], P[:, :1] + np.cumsum(stp, axis=1)], axis=1)
    # place every partner image nearest to the highlighted chain
    P -= np.round((P.mean(1, keepdims=True) - P[c0].mean(0)) / per) * per
    return P * 10.0, seq, c0, partners


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ['set bgColor white']
    for i, (key, name) in enumerate(PAIR.items()):
        d = cached(name)
        P, seq, c0, partners = chain_and_partners(d)
        rel = np.asarray(d['X']) / np.mean(d['X'])
        bf = {c0: np.clip(rel, LO, HI)}
        for c in partners:
            bf[c] = np.full(P.shape[1], -1.0)
        pdb = os.path.join(OUT, f'sat_{key}.pdb')
        R.write_pdb(pdb, [c0] + partners, P, seq, bfac=bf)
        png = os.path.join(OUT, f'sat_{key}.png')
        if os.path.exists(png):
            os.remove(png)
        print(f'{key}: {name}, {len(partners)} partner chains, relative contacts {rel.min():.2f} to {rel.max():.2f}, CV {rel.std():.3f}')
        # partners as small opaque beads, so they never tint the chain; one field width for both renders, so sizes compare
        lines += [f'open {pdb}', 'style #1 sphere', 'size #1 atomRadius 1.3', 'size #1/A atomRadius 3.4', 'color #1 #B4B4B4',
                  f'color bfactor #1/A palette {LO},#D6E8F5:1.0,#0077BB:{HI},#00243A',
                  'hide #1 & ~(/A :<12) atoms', 'lighting soft', 'graphics silhouettes true width 1.5', 'camera ortho',
                  'windowsize 900 900', 'view /A', f'zoom pixelSize {FIELD / 900:.4f}', 'clip off', f'save {png} width 1800 supersample 3', 'close']
    lines.append('exit')
    cxc = os.path.join(OUT, 'render_saturation.cxc')
    open(cxc, 'w').write('\n'.join(lines) + '\n')
    subprocess.run(['open', '-n', '-a', '/Applications/ChimeraX-1.12.app', cxc], check=True)
    t0 = time.time()
    while time.time() - t0 < 300 and not os.path.exists(os.path.join(OUT, 'sat_krtap.png')):
        time.sleep(3)
    time.sleep(3)
    print('renders:', [f for f in os.listdir(OUT) if f.startswith('sat_') and f.endswith('.png')])


if __name__ == '__main__':
    main()
