"""
tar_list_remote.py — list the members of a remote tar archive without downloading it, by reading only the 512-byte
header in front of each member with HTTP range requests and skipping over the member's data.
Handles ustar names with prefix, GNU long names (typeflag L) and PAX extended headers (typeflag x, g).
Transfer is about 512 bytes per member (plus long-name / PAX blocks). Requests are sequential.
  python3 tar_list_remote.py URL OUT.csv [--max-members 50000]
"""
import argparse
import csv
import sys
import time
import urllib.request


def get(url, start, length, tries=4):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{start + length - 1}',
                                               'User-Agent': 'tar-header-lister/1.0'})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError(f'server did not honour the range request (status {r.status})')
                data = r.read(length + 1)
                if len(data) > length:
                    raise RuntimeError('server returned more than the requested range')
                return data
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def octal(b):
    b = b.rstrip(b'\0 ').strip()
    if not b:
        return 0
    if b[0] & 0x80:                      # GNU base-256 for large sizes
        return int.from_bytes(bytes([b[0] & 0x7f]) + b[1:], 'big')
    return int(b, 8)


def pax_path(data):
    i = 0
    out = {}
    while i < len(data):
        sp = data.index(b' ', i)
        n = int(data[i:sp])
        rec = data[sp + 1:i + n - 1].decode('utf-8', 'replace')
        k, _, v = rec.partition('=')
        out[k] = v
        i += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('out')
    ap.add_argument('--max-members', type=int, default=50000)
    a = ap.parse_args()
    off, n, transferred, longname, pax = 0, 0, 0, None, {}
    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['name', 'size_bytes', 'type', 'data_offset'])
        while n < a.max_members:
            h = get(a.url, off, 512)
            transferred += 512
            if h == b'\0' * 512:
                break
            size = octal(h[124:136])
            t = chr(h[156]) if h[156] else '0'
            data_off = off + 512
            skip = 512 + ((size + 511) // 512) * 512
            if t == 'L':
                longname = get(a.url, data_off, size).rstrip(b'\0').decode('utf-8', 'replace')
                transferred += size
            elif t in 'xg':
                pax = pax_path(get(a.url, data_off, size))
                transferred += size
            else:
                name = h[0:100].rstrip(b'\0').decode('utf-8', 'replace')
                if h[257:262] == b'ustar':
                    prefix = h[345:500].rstrip(b'\0').decode('utf-8', 'replace')
                    if prefix:
                        name = prefix + '/' + name
                name = pax.get('path') or longname or name
                if 'size' in pax:
                    size = int(pax['size'])
                    skip = 512 + ((size + 511) // 512) * 512
                w.writerow([name, size, t, data_off])
                n += 1
                longname, pax = None, {}
                if n % 200 == 0:
                    print(f'{n} members, offset {off / 1e9:.1f} GB, {transferred / 1e6:.2f} MB transferred', flush=True)
            off += skip
    print(f'done: {n} members listed, {transferred / 1e6:.2f} MB transferred, archive end at {off / 1e9:.2f} GB')


if __name__ == '__main__':
    sys.exit(main())
