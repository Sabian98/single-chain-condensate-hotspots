"""
prep_fig2.py: data for Figure 2e,f from the example slab (GNRHR2 residues 140 to 292), cached on the NAS.
  e: time-averaged within-chain contact profiles of dense and dilute chains (from render_more.phases, same definitions as slab_test).
  f: radius of gyration of every chain in every frame against the distance of its centre from the slab centre, as in Farag et al.
     2022 (Fig. 4d) and Wang et al. 2024 (Fig. 4c), with the dense half-width of slab_test.
  ~/.venvs/finches/bin/python tools/prep_fig2.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_more as RM  # noqa: E402

OUT = os.path.join(RM.CACHE, 'fig2_rg_vs_z.npz')


def main():
    sys.path.insert(0, os.path.join(RM.ROOT, 'scripts'))
    import mdtraj as md
    import slab_test as S
    name = RM.EXAMPLE
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
    dz_all, rg_all = [], []
    for f in range(t.n_frames):
        x = np.mod(t.xyz[f].astype(float), box)
        cz = x[:, 2].reshape(nch, n).mean(1)
        zc = (np.angle(np.exp(2j * np.pi * cz / L).mean()) % (2 * np.pi)) * L / (2 * np.pi)
        P = x.reshape(nch, n, 3)
        d = np.diff(P, axis=1)
        d -= box * np.round(d / box)
        P = np.concatenate([P[:, :1], P[:, :1] + np.cumsum(d, axis=1)], axis=1)
        rg = np.sqrt(((P - P.mean(1, keepdims=True)) ** 2).sum(2).mean(1))
        cen = P.mean(1)[:, 2]                                      # centre of the whole chain, not of the wrapped beads
        dz = np.abs((cen - zc + L / 2) % L - L / 2)
        dz_all.append(dz)
        rg_all.append(rg)
    ph = RM.phases(name)
    np.savez_compressed(OUT, dz=np.concatenate(dz_all), rg=np.concatenate(rg_all), half_width=float(ph['half_width']),
                        Wd=ph['Wd'], Wl=ph['Wl'], L=L)
    dz, rg = np.concatenate(dz_all), np.concatenate(rg_all)
    hw = float(ph['half_width'])
    print(f'{len(rg)} chain-frames; dense half-width {hw:.0f} nm; mean Rg dense {rg[dz < hw].mean():.2f}, beyond 50 nm {rg[dz > 50].mean():.2f} nm')


if __name__ == '__main__':
    main()
