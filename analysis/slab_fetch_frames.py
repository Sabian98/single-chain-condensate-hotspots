"""
slab_fetch_frames.py — fetch a subset of frames of the IDRome10 slab DCDs (von Bülow et al. 2025, ERDA) by HTTP byte
range, without downloading whole trajectories (slab_followup_plan.md, Amendment 1).

Layout (verified from one header and from every member size): 276-byte DCD header, then fixed-size frames with a unit
cell block, frame bytes = 56 + 3 * (8 + 4 * natoms). Selected frames: --frames evenly spaced over the second half.
Output per IDR: data/erda_pspred/IDRome10/<IDR>/<IDR>_sub.dcd (the original header with NSET patched to the number of selected frames,
followed by the selected frame blocks, i.e. a valid DCD) and frames.json (original frame indices).

  python3 slab_fetch_frames.py --plan     # sizes only, nothing downloaded
  python3 slab_fetch_frames.py --fetch --cap-bytes 3000000000
"""
import argparse
import csv
import glob
import json
import os
import re
import struct
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'data', 'erda_pspred')        # DYNASEP/hub_condensate_followup/data/erda_pspred
ARCHIVE = 'https://erda.ku.dk/archives/d29311f079323ca099fd9ef8dbd118ca/lindorff-group-BIO/sobuelow/PS_prediction/'
TARS = {'IDRome10': 'simulations_validation_IDRome10.tar', 'IDRome90': 'simulations_validation_IDRome90.tar',
        'training': 'simulations_training.tar'}
URL = ARCHIVE + TARS['IDRome10']   # set by --set


def header_len(off):
    """DCD header length: 92 bytes of the first block, then the title block (4 + 4 + 80 * ntitle + 4) and the natoms block (12).
    Most slabs have 2 title lines (276 bytes); some have 3 (356 bytes)."""
    blk = struct.unpack('<i', get(off + 92, 4))[0]
    assert (blk - 4) % 80 == 0, blk
    return 92 + 4 + blk + 4 + 12


def get(start, length):
    req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{start + length - 1}',
                                               'User-Agent': 'slab-frame-fetch/1.0'})
    with urllib.request.urlopen(req, timeout=300) as r:
        if r.status != 206:
            raise RuntimeError(f'range not honoured ({r.status})')
        data = r.read(length + 1)
    if len(data) != length:
        raise RuntimeError(f'expected {length} bytes, got {len(data)}')
    return data


def systems(n_sel, which, names):
    """Slabs of one archive (--set): chain count from the DCD header (natoms / chain length), so no config file is needed."""
    listing = {r['name'].split('/')[1]: r for r in csv.DictReader(open(os.path.join(BASE, f'{which}_tar_listing.csv')))
               if r['name'].endswith('.dcd')}
    sys.path.insert(0, HERE)
    import slab_test as st
    length = {s['name']: s['N'] for s in st.slabs()}             # from the published sequence, since YTH names carry no residue range
    out = []
    for name in sorted(names):
        if name not in listing:
            print(f'  {name}: not in the {which} archive listing')
            continue
        L = length[name]
        size, off = int(listing[name]['size_bytes']), int(listing[name]['data_offset'])
        hl = header_len(off)
        head = get(off, hl)
        assert struct.unpack('<i', head[:4])[0] == 84 and head[4:8] == b'CORD', name
        natoms = struct.unpack('<i', head[hl - 8:hl - 4])[0]
        nch, r = divmod(natoms, L)
        if r or nch == 0:
            raise RuntimeError(f'{name}: {natoms} atoms do not split into chains of {L}')
        fb = 56 + 3 * (8 + 4 * natoms)
        nf, rem = divmod(size - hl, fb)
        if rem:
            raise RuntimeError(f'{name}: size does not fit the frame layout')
        nset = struct.unpack('<i', head[8:12])[0]
        if nset != nf:
            raise RuntimeError(f'{name}: {nf} frames from the file size but NSET {nset} in the header')
        h = nf // 2
        idx = [h + (k * (nf - h)) // n_sel for k in range(n_sel)]
        out.append(dict(name=name, L=L, chains=nch, natoms=natoms, frame_bytes=fb, n_frames=nf, offset=off, idx=idx,
                        hl=hl, bytes=hl + n_sel * fb))
    return out


def tier_names(which, tier):
    """Slab names of one tier, exactly as fixed by Amendment 2 (slab_test.slabs())."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('slab_test', os.path.join(HERE, 'slab_test.py'))
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    return [s['name'] for s in st.slabs() if s['set'] == which and s['tier'] == tier]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--cap-bytes', type=int, default=0)
    ap.add_argument('--only', default=None)
    ap.add_argument('--set', default='IDRome10', choices=list(TARS))
    ap.add_argument('--tier', default='primary')
    a = ap.parse_args()
    global URL
    URL = ARCHIVE + TARS[a.set]
    names = [a.only] if a.only else tier_names(a.set, a.tier)
    S = systems(a.frames, a.set, names)
    total = sum(s['bytes'] for s in S)
    for s in S:
        print(f"  {s['name']:20s} L={s['L']:4d} chains={s['chains']:5d} frames={s['n_frames']:6d}  "
              f"fetch {a.frames} frames from {s['idx'][0]}..{s['idx'][-1]}  {s['bytes'] / 1e6:7.1f} MB")
    print(f'TOTAL {len(S)} IDRs, {total} bytes = {total / 1e9:.2f} GB ({a.frames} frames each)')
    if not a.fetch:
        return
    if not a.cap_bytes or total > a.cap_bytes:
        raise SystemExit('refusing: set --cap-bytes at least the total')
    for s in S:
        d = os.path.join(BASE, a.set, s['name'])
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{s['name']}_sub.dcd")
        if os.path.exists(path) and os.path.getsize(path) == s['bytes']:
            print(f"  {s['name']}: already fetched")
            continue
        hl = s['hl']
        head = bytearray(get(s['offset'], hl))
        assert struct.unpack('<i', head[:4])[0] == 84 and head[4:8] == b'CORD', s['name']
        natoms = struct.unpack('<i', head[hl - 8:hl - 4])[0]
        assert natoms == s['natoms'], (s['name'], natoms, s['natoms'])
        head[8:12] = struct.pack('<i', len(s['idx']))
        tmp = path + '.part'
        with open(tmp, 'wb') as fh:
            fh.write(head)
            for k in s['idx']:
                fh.write(get(s['offset'] + hl + k * s['frame_bytes'], s['frame_bytes']))
        os.replace(tmp, path)
        json.dump({'name': s['name'], 'source_frames': s['idx'], 'n_source_frames': s['n_frames'],
                   'frame_bytes': s['frame_bytes'], 'natoms': natoms, 'chains': s['chains'], 'L': s['L']},
                  open(os.path.join(d, 'frames.json'), 'w'))
        print(f"  {s['name']}: {os.path.getsize(path) / 1e6:.1f} MB", flush=True)


if __name__ == '__main__':
    main()
