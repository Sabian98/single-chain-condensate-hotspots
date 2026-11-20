"""Sentence density of the section files against the STYLE_GUIDE targets (mean 24-30 words, <=20% over 40 words, <=20% with >=4 numbers).
  python3 tools/density.py [--long]
"""
import os, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
macros = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}', open(os.path.join(HERE, 'numbers.tex')).read()))
def clean(t):
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = re.sub(r'\\begin\{(figure|equation|table)\}.*?\\end\{\1\}', ' ', t, flags=re.S)
    for k, v in macros.items():
        t = t.replace('\\' + k + '{}', v).replace('\\' + k, v)
    t = re.sub(r'\$[^$]*\$', ' X ', t)
    t = re.sub(r'\\(cite|ref|eqref|label|pending)\{[^}]*\}', ' ', t)
    t = re.sub(r'\\(section|subsection|paragraph)\*?\{[^}]*\}', '.', t)
    t = re.sub(r'\\[a-zA-Z]+\*?', ' ', t)
    t = re.sub(r'[{}~]', ' ', t)
    return re.sub(r'\s+', ' ', t)
for f in ('abstract', 'intro', 'theory', 'methods', 'results', 'discussion'):
    s = [x for x in re.split(r'(?<=[.!?])\s+(?=[A-Z(])', clean(open(os.path.join(HERE, 'sections', f + '.tex')).read())) if len(x.split()) > 3]
    w = [len(x.split()) for x in s]
    n = [len(re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?', x)) for x in s]
    print(f'{f:11s} sentences {len(s):3d}  mean words {sum(w)/len(w):5.1f}  >40 words {100*sum(x>40 for x in w)/len(w):4.0f}%  >=4 numbers {100*sum(k>=4 for k in n)/len(n):4.0f}%')
    if '--long' in sys.argv:
        for x, k in zip(s, w):
            if k > 40:
                print('    LONG', k, x[:150])
