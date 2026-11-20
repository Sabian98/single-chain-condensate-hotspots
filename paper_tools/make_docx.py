"""
make_docx.py: Word version of the manuscript, for co-authors who edit in Word. The LaTeX source stays the master copy.
  1. tectonic build with intermediates kept, to read figure, section and equation numbers from main.aux
  2. flatten main.tex (\\input), resolve \\ref and \\eqref to those numbers, write equation numbers into the equations,
     turn \\pending{...} into visible [PENDING ...] text, point figures at PNG renders of the figure PDFs
  3. pandoc (conda env 'docs') with citeproc, refs.bib and tools/acs_numeric.csl, native figure numbering
  ~/.venvs/finches/bin/python tools/make_docx.py [out.docx]
"""
import os
import re
import subprocess
import sys
import tempfile

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANDOC = os.path.expanduser('~/miniforge3/envs/docs/bin/pandoc')
NAS = '/Volumes/User Homes/Nu Project/DYNASEP/hub_condensate_followup/manuscript'
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(NAS, 'JPCB_manuscript.docx')


def braced(s, i):
    """Index just past the brace group that opens at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        depth += {'{': 1, '}': -1}.get(s[j], 0)
        if depth == 0:
            return j + 1
    raise ValueError('unbalanced braces')


def flatten(path):
    s = open(path).read()
    s = re.sub(r'(?<!\\)%.*', '', s)                              # drop comments
    return re.sub(r'\\input\{([^}]+)\}', lambda m: flatten(os.path.join(PAPER, m.group(1) + ('' if m.group(1).endswith('.tex') else '.tex'))), s)


def labels(aux):
    out = {}
    for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}', open(aux).read()):
        out[m.group(1)] = m.group(2)
    return out


def main():
    tmp = tempfile.mkdtemp(prefix='docx_')
    subprocess.run(['tectonic', '--keep-intermediates', '--outdir', tmp, 'main.tex'], cwd=PAPER, check=True, capture_output=True)
    lab = labels(os.path.join(tmp, 'main.aux'))
    s = flatten(os.path.join(PAPER, 'main.tex'))
    # equation numbers written into the equation (Word math has no automatic numbering)
    def eq(m):
        body = m.group(1)
        k = re.search(r'\\label\{([^}]+)\}', body)
        body = re.sub(r'\\label\{[^}]+\}', '', body).strip()
        tag = ' \\text{' + '\u2003' * 3 + '}' + f'({lab[k.group(1)]})' if k else ''   # em spaces: Word math drops \qquad
        return f'\\[ {body}{tag} \\]'
    s = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', eq, s, flags=re.S)
    s = re.sub(r'\\eqref\{([^}]+)\}', lambda m: f'({lab[m.group(1)]})', s)
    s = re.sub(r'\\ref\{([^}]+)\}', lambda m: lab[m.group(1)], s)
    s = re.sub(r'\\label\{(sec:[^}]+)\}', '', s)
    s = re.sub(r'[ \t~]*(\\cite\{)', r'\1', s)                  # superscript citations sit on the word
    s = re.sub(r'\$\^\{?([^${}]+)\}?\$', r'\\textsuperscript{\1}', s)   # bare superscripts (nm$^3$) as text, not empty-base math
    # placeholders as visible text
    while '\\pending{' in s:
        i = s.index('\\pending{')
        j = braced(s, i + len('\\pending'))
        s = s[:i] + '[PENDING: ' + s[i + len('\\pending') + 1:j - 1] + ']' + s[j:]
    s = s.replace('\\newcommand{\\pending}[1]{\\textcolor{orange}{[#1]}}', '')
    # figures as PNG renders
    for pdf in sorted(set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+\.pdf)\}', s))):
        png = os.path.join(tmp, os.path.basename(pdf)[:-4] + '.png')
        subprocess.run(['osascript', '-l', 'JavaScript', os.path.join(PAPER, 'tools', 'pdf2png.js'), os.path.join(PAPER, pdf), png,
                        '300'], check=True, capture_output=True)
        s = s.replace(pdf, png)
    s = re.sub(r'\\includegraphics\[[^]]*\]', r'\\includegraphics[width=6.5in]', s)
    # achemso front matter and environments pandoc does not know
    s = re.sub(r'\\documentclass\[[^]]*\]\{achemso\}', r'\\documentclass{article}', s)
    for env, head in (('suppinfo', 'Supporting Information'), ('acknowledgement', 'Acknowledgements')):
        s = s.replace(f'\\begin{{{env}}}', f'\\section*{{{head}}}').replace(f'\\end{{{env}}}', '')
    s = re.sub(r'\\affiliation\{[^}]*\}|\\email\{[^}]*\}', '', s)
    s = s.replace('\\bibliography{refs}', '')
    src = os.path.join(tmp, 'flat.tex')
    open(src, 'w').write(s)
    subprocess.run([PANDOC, src, '-f', 'latex', '-t', 'docx+native_numbering', '-o', OUT, '--citeproc',
                    '--bibliography', os.path.join(PAPER, 'refs.bib'), '--csl', os.path.join(PAPER, 'tools', 'acs_numeric.csl'),
                    '--resource-path', tmp, '--reference-doc', os.path.join(PAPER, 'tools', 'reference.docx'), '-M', 'reference-section-title=References', '-M', 'link-citations=false'],
                   cwd=tmp, check=True)
    print(OUT)


if __name__ == '__main__':
    main()
