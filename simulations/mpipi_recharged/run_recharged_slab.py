#!/usr/bin/env python
"""
run_recharged_slab.py: one Mpipi-Recharged slab for Amendment 3 of slab_followup_plan.md (OpenMpipi in OpenMM).

Geometry follows the CALVADOS-2 slabs of von Buelow et al. (2025) that the analysis expects: box 20 x 20 x 200 nm with z the long axis.
  1. One chain is relaxed in isolation with OpenMpipi (get_compact_model, --relax-ns, default 10 ns), giving a compact start.
  2. Copies are placed on a grid filling x and y and stacked in z around the box centre.
  3. A weak harmonic force along z only pulls every bead toward z = 100 nm for --pull-ns, which forms the slab.
  4. The force is removed and the production run lasts --run-ns, saving a frame every --save-ns. The analysis uses the second half.
The production run is split into segments; state.xml and progress.json are written after each, so rerunning the same command after a
timeout resumes. Speed (ns/day) goes to <name>.log for the compute budget of Amendment 3.
Output: <outdir>/<name>/{<name>.dcd, top.pdb, <name>.log, run.json}

  python run_recharged_slab.py --name O60477_1_58 --fasta recharged_slab_293K.fasta --outdir /general/taseefr/slab_recharged
"""
import json
import os
from argparse import ArgumentParser

import mdtraj as md
import numpy as np
import openmm as mm
from openmm import app, unit
from Bio import SeqIO
from scipy.spatial import cKDTree

from OpenMpipi import IDP, get_mpipi_system

ap = ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--fasta', required=True)
ap.add_argument('--outdir', required=True)
ap.add_argument('--temp', type=float, default=293.0)
ap.add_argument('--ionic', type=float, default=0.15, help='M')
ap.add_argument('--chains', type=int, default=0, help='default max(64, round(15000 / N))')
ap.add_argument('--box', type=float, nargs=3, default=[20.0, 20.0, 200.0], help='nm, z is the long axis')
ap.add_argument('--relax-ns', type=float, default=10.0, help='isolated-chain relaxation before the slab is built')
ap.add_argument('--pull-ns', type=float, default=20.0)
ap.add_argument('--pull-k', type=float, default=0.005, help='kJ/mol/nm^2 per bead, along z only')
ap.add_argument('--run-ns', type=float, default=1000.0)
ap.add_argument('--save-ns', type=float, default=1.0)
ap.add_argument('--segments', type=int, default=10)
ap.add_argument('--seed', type=int, default=20261113)
ap.add_argument('--platform', default='CUDA', choices=['CUDA', 'CPU'])
a = ap.parse_args()

seq = next(str(r.seq) for r in SeqIO.parse(a.fasta, 'fasta') if r.id.split('|')[0] == a.name)
N = len(seq)
nch = a.chains or max(64, round(15000 / N))
T, csx, dt = a.temp, a.ionic * 1000, 0.01                      # K, mM, ps (10 fs)
Lx, Ly, Lz = a.box
path = os.path.join(a.outdir, a.name)
os.makedirs(path, exist_ok=True)
state_xml, prog_json = os.path.join(path, 'state.xml'), os.path.join(path, 'progress.json')
save_steps = int(round(a.save_ns * 1000 / dt))
n_frames = int(round(a.run_ns / a.save_ns))
assert n_frames % a.segments == 0, 'frames must split evenly into segments'
seg_steps = save_steps * n_frames // a.segments
platform = mm.Platform.getPlatformByName(a.platform)
props = {'Precision': 'mixed'} if a.platform == 'CUDA' else {}
rng = np.random.default_rng(a.seed)

chain = IDP(a.name, seq)
resuming = os.path.exists(state_xml) and os.path.exists(prog_json)
if resuming:
    pos = np.load(os.path.join(path, 'start_positions.npy'))
else:
    here = os.getcwd()
    os.chdir(path)                                              # get_compact_model writes a temporary file in the working directory
    chain.get_compact_model(simulation_time=a.relax_ns * unit.nanosecond, T=T * unit.kelvin, csx=csx)
    os.chdir(here)
    one = chain.min_rg_coords - chain.min_rg_coords.mean(0)
    off = 3.0 * float(chain.min_rg)                             # spacing as in OpenMpipi's build_model
    nx, ny = max(1, int(Lx // off)), max(1, int(Ly // off))
    layers = -(-nch // (nx * ny))
    zs = Lz / 2 + (np.arange(layers) - (layers - 1) / 2) * off
    assert zs.min() > 5 and zs.max() < Lz - 5, f'{nch} chains of this size do not fit along z ({layers} layers of {off:.1f} nm)'
    grid = [((i + 0.5) * Lx / nx, (j + 0.5) * Ly / ny, z) for z in zs for i in range(nx) for j in range(ny)][:nch]
    # Amendment 3a: an orientation that puts a bead within MIN_SEP of a bead already placed (periodic) is drawn again, since an
    # elongated compact chain reaches past half the 3 Rg spacing and overlapping copies blow up (Q96P88_140_292)
    MIN_SEP, TRIES = 0.5, 1000
    box3 = np.array([Lx, Ly, Lz])
    pos, redrawn = [], 0
    for g in grid:
        tree = cKDTree(np.mod(np.vstack(pos), box3), boxsize=box3) if pos else None
        for t in range(TRIES):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))        # random orientation of each copy
            x = one @ q.T + np.array(g)
            if tree is None or np.isinf(tree.query(np.mod(x, box3), distance_upper_bound=MIN_SEP)[0]).all():
                break
        else:
            raise RuntimeError(f'no orientation of copy {len(pos) + 1} clears {MIN_SEP} nm in {TRIES} tries')
        redrawn += t > 0
        pos.append(x)
    print(f'{a.name}: {redrawn} of {nch} copies reoriented to keep {MIN_SEP} nm between chains', flush=True)
    pos = np.vstack(pos)
    np.save(os.path.join(path, 'start_positions.npy'), pos)

model = app.Modeller(chain.topology, pos[:N] * unit.nanometer)
for c in range(1, nch):
    model.add(chain.topology, pos[c * N:(c + 1) * N] * unit.nanometer)
top = model.topology
box = [mm.Vec3(Lx, 0, 0), mm.Vec3(0, Ly, 0), mm.Vec3(0, 0, Lz)] * unit.nanometer
top.setPeriodicBoxVectors(box)
system = get_mpipi_system(pos, top, {a.name: []}, T, csx, CM_remover=False, periodic=True)
system.setDefaultPeriodicBoxVectors(*box)
pull = mm.CustomExternalForce('k*(z-z0)^2')
pull.addGlobalParameter('k', a.pull_k)
pull.addGlobalParameter('z0', Lz / 2)
for i in range(system.getNumParticles()):
    pull.addParticle(i, [])
pull_index = system.addForce(pull)

integrator = mm.LangevinMiddleIntegrator(T * unit.kelvin, 0.01 / unit.picosecond, dt * unit.picosecond)
integrator.setRandomNumberSeed(a.seed)
sim = app.Simulation(top, system, integrator, platform, props)
with open(os.path.join(path, 'top.pdb'), 'w') as fh:
    app.PDBFile.writeFile(top, pos * unit.nanometer, fh)


def save(progress):
    sim.saveState(state_xml + '.tmp')
    os.replace(state_xml + '.tmp', state_xml)
    json.dump(progress, open(prog_json + '.tmp', 'w'))
    os.replace(prog_json + '.tmp', prog_json)


if resuming:
    progress = json.load(open(prog_json))
    sim.loadState(state_xml)
    print(f"resuming {a.name}: pulled {progress['pulled']}, {progress['segments_done']}/{a.segments} segments", flush=True)
else:
    progress = {'pulled': False, 'segments_done': 0}
    sim.context.setPositions(pos * unit.nanometer)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(T * unit.kelvin, a.seed)
json.dump({**vars(a), 'N': N, 'chains': nch, 'beads': N * nch, 'frames': n_frames}, open(os.path.join(path, 'run.json'), 'w'))
log = open(os.path.join(path, f'{a.name}.log'), 'a')
sim.reporters.append(app.StateDataReporter(log, max(save_steps, seg_steps // 20), step=True, speed=True, elapsedTime=True,
                                           potentialEnergy=True, temperature=True, separator='\t'))
if not progress['pulled']:
    print(f'{a.name}: {nch} chains of {N}, {N * nch} beads; pulling along z for {a.pull_ns} ns', flush=True)
    sim.step(int(round(a.pull_ns * 1000 / dt)))
    sim.context.setParameter('k', 0.0)                          # the force stays in the system with zero strength
    progress['pulled'] = True
    save(progress)
else:
    sim.context.setParameter('k', 0.0)
for s in range(progress['segments_done'], a.segments):
    dcd = app.DCDReporter(os.path.join(path, f'seg_{s:02d}.dcd'), save_steps)
    sim.reporters.append(dcd)
    sim.step(seg_steps)
    sim.reporters.remove(dcd)
    del dcd
    progress['segments_done'] = s + 1
    save(progress)
    print(f'  segment {s + 1}/{a.segments} done', flush=True)
log.close()
segs = [os.path.join(path, f'seg_{s:02d}.dcd') for s in range(a.segments)]
if all(os.path.exists(f) for f in segs):
    traj = md.load(segs, top=os.path.join(path, 'top.pdb'))
    assert traj.n_frames == n_frames, (traj.n_frames, n_frames)
    traj.save_dcd(os.path.join(path, f'{a.name}.tmp.dcd'))
    os.replace(os.path.join(path, f'{a.name}.tmp.dcd'), os.path.join(path, f'{a.name}.dcd'))
    for f in segs:
        os.remove(f)
    print(f'{a.name}: done, {n_frames} frames', flush=True)
