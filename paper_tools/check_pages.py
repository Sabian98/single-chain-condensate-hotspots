"""Flag figure pages whose caption or content runs into the bottom margin of the built PDF (main.pdf)."""
import os, sys, pymupdf
d = pymupdf.open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'main.pdf'))
bad = 0
for i, p in enumerate(d):
    blocks = [b for b in p.get_text('blocks') if b[4].strip() and not b[4].strip().isdigit()]
    if not blocks: continue
    low = max(b[3] for b in blocks)
    if any(b[4].lstrip().startswith('Figure') for b in blocks) and low > p.rect.height - 60:
        bad += 1; print(f'page {i + 1}: content reaches {low:.0f} of {p.rect.height:.0f} pt')
print('overflowing figure pages:', bad)
