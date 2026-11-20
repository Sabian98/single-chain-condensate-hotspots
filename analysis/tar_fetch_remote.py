"""
tar_fetch_remote.py — fetch selected members (or the first bytes of a member) of a remote tar archive by HTTP range
requests, using the listing written by tar_list_remote.py (columns name, size_bytes, type, data_offset).
  python3 tar_fetch_remote.py URL LISTING.csv OUTDIR --match config.yaml [--match top.pdb --only Q8N3T1_1_161]
      [--head-bytes N]   (fetch only the first N bytes of each matched member, saved as <name>.head)
      --cap-bytes C      (refuse to start if the total to fetch exceeds C)
"""
import argparse
import csv
import os
import sys
import urllib.request


def get(url, start, length):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{start + length - 1}',
                                               'User-Agent': 'tar-member-fetch/1.0'})
    with urllib.request.urlopen(req, timeout=300) as r:
        if r.status != 206:
            raise RuntimeError(f'range request not honoured (status {r.status})')
        data = r.read(length + 1)
    if len(data) != length:
        raise RuntimeError(f'expected {length} bytes, got {len(data)}')
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('listing')
    ap.add_argument('outdir')
    ap.add_argument('--match', action='append', required=True, help='substring the member name must end with')
    ap.add_argument('--only', action='append', default=None, help='restrict to members containing this substring')
    ap.add_argument('--head-bytes', type=int, default=0)
    ap.add_argument('--cap-bytes', type=int, required=True)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.listing))
            if any(r['name'].endswith(m) for m in a.match) and (not a.only or any(o in r['name'] for o in a.only))]
    sizes = [min(int(r['size_bytes']), a.head_bytes) if a.head_bytes else int(r['size_bytes']) for r in rows]
    total = sum(sizes)
    print(f'{len(rows)} members, {total} bytes to fetch (cap {a.cap_bytes})')
    if total > a.cap_bytes:
        sys.exit('refusing: total exceeds the cap')
    for r, n in zip(rows, sizes):
        rel = r['name'].split('/', 1)[1] if '/' in r['name'] else r['name']
        path = os.path.join(a.outdir, rel + ('.head' if a.head_bytes else ''))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'wb').write(get(a.url, int(r['data_offset']), n))
    print('done')


if __name__ == '__main__':
    main()
