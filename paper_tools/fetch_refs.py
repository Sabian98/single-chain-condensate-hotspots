"""Build refs.bib from Crossref: every author in full. Keys and DOIs (or title queries) are listed here; matches by title are printed for checking."""
import json, sys, time, urllib.parse, urllib.request
UA = {'User-Agent': 'refs-fetch/1.0 (academic manuscript reference check)'}
REFS = [
 ('dignon2018', 'doi', '10.1073/pnas.1804177115'),
 ('lin2017', 'doi', '10.1016/j.bpj.2017.04.021'),
 ('farag2022', 'doi', '10.1038/s41467-022-35370-7'),
 ('choi2020', 'doi', '10.1146/annurev-biophys-121219-081629'),
 ('mittag2022', 'q', 'A conceptual framework for understanding phase separation and addressing open questions and challenges Mittag Pappu'),
 ('tesei2024', 'doi', '10.1038/s41586-023-07004-5'),
 ('vonbulow2025', 'qj', 'Prediction of phase separation propensities of disordered proteins from sequence von Bulow Tesei Lindorff-Larsen'),
 ('an2024', 'doi', '10.1126/sciadv.adj2448'),
 ('bremer2022', 'doi', '10.1038/s41557-021-00840-w'),
 ('martin2020', 'doi', '10.1126/science.aaw8653'),
 ('tesei2022', 'qj', 'Improved predictions of phase behaviour of intrinsically disordered proteins by tuning the interaction range Tesei Lindorff-Larsen'),
 ('joseph2021', 'doi', '10.1038/s43588-021-00155-3'),
 ('tejedor2025', 'doi', '10.1021/acscentsci.4c01617'),
 ('eastman2024', 'doi', '10.1021/acs.jpcb.3c06662'),
 ('banani2017', 'doi', '10.1038/nrm.2017.7'),
 ('brangwynne2009', 'doi', '10.1126/science.1172046'),
 ('ashbaugh2008', 'doi', '10.1021/ja802124e'),
 ('dignon2018plos', 'doi', '10.1371/journal.pcbi.1005941'),
 ('statt2020', 'qj', 'Model for disordered proteins with strongly sequence-dependent liquid phase behavior Statt Casademunt Brangwynne Panagiotopoulos'),
 ('das2020', 'qj', 'Comparative roles of charge pi and hydrophobic interactions in sequence-dependent phase separation of intrinsically disordered proteins Das Lin Chan'),
 ('zheng2020', 'qj', 'Hydropathy patterning complements charge patterning to describe conformational preferences of disordered proteins Zheng Dignon Mittal'),
 ('oliver2025', 'qj', 'When B2 is not enough evaluating simple metrics for predicting phase separation of intrinsically disordered proteins Oliver Jacobs Webb'),
 ('morton2025', 'doi', '10.64898/2025.12.12.694022'),
 ('kim2021', 'doi', '10.1073/pnas.2104897118'),
 ('murthy2019', 'doi', '10.1038/s41594-019-0250-x'),
 ('ryan2018', 'doi', '10.1016/j.molcel.2017.12.022'),
 ('fus2025', 'doi', '10.1038/s41589-024-01828-6'),
 ('nott2015', 'doi', '10.1016/j.molcel.2015.01.013'),
 ('borcherds2021', 'doi', '10.1016/j.sbi.2020.09.004'),
 ('burke2015', 'doi', '10.1016/j.molcel.2015.09.006'),
 ('zheng2020md', 'doi', '10.1021/acs.jpcb.0c10489'),
 ('lotthammer2024', 'doi', '10.1038/s41592-023-02159-5'),
 ('florestellez2023', 'doi', '10.1371/journal.pgen.1010980'),
 ('zeng2020', 'doi', '10.1016/j.bpj.2020.06.014'),
 ('ownrecomb', 'doi', '10.64898/2026.09.13.751321'),
 ('devarajan2024', 'doi', '10.1038/s41467-024-46223-w'),
 ('wang2024', 'doi', '10.1039/D4SC05004E'),
 ('pal2024', 'doi', '10.1021/acs.jpclett.4c01973'),
 ('ginell2025', 'doi', '10.1126/science.adq8381'),
]
# Crossref name errors, fixed after every fetch (particles braced so they stay in front; given/family split corrected)
NAME_FIXES = {'von Bülow, Sören': '{von Bülow}, Sören', 'R. Tejedor, Andrés': 'Tejedor, Andrés R.'}
def get(url):
    for k in range(4):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception as e:
            time.sleep(2 + 3 * k); err = e
    raise err
import re
def clean(t):
    t = t.replace('<i>π</i>', r'$\pi$').replace('<i>B</i><sub>2</sub>', '$B_2$')
    t = re.sub(r'<[^>]+>', '', t)
    t = __import__('html').unescape(t)
    t = re.sub(r'\s+', ' ', t).strip().replace('&', r'\&')
    t = t.replace('\u2212', '--').replace('\u2013', '--')   # Unicode minus and en dash are not in the LaTeX text font
    return t
def bib(key, m):
    typ = m.get('type', '')
    au = ' and '.join(f"{a.get('family','')}, {a.get('given','')}".strip(', ') for a in m.get('author', []))
    yr = (m.get('published-print') or m.get('published-online') or m.get('issued') or m.get('posted') or {}).get('date-parts', [[None]])[0][0]
    f = {'author': au, 'title': '{' + clean((m.get('title') or [''])[0]) + '}', 'year': yr, 'doi': m.get('DOI')}
    if typ == 'posted-content':
        f['journal'] = (m.get('institution') or [{}])[0].get('name', 'bioRxiv') if m.get('institution') else 'bioRxiv'
        kind = 'article'
    else:
        f.update({'journal': clean((m.get('container-title') or [''])[0]), 'volume': m.get('volume'), 'pages': m.get('page') or m.get('article-number')})
        kind = 'article'
    for bad, good in NAME_FIXES.items():
        f['author'] = f['author'].replace(bad, good)
    body = ',\n'.join(f'  {k} = {{{v}}}' for k, v in f.items() if v)
    return f'@{kind}{{{key},\n{body}\n}}\n'
out = []
for key, how, val in REFS:
  try:
      if how == 'doi':
          m = get('https://api.crossref.org/works/' + urllib.parse.quote(val))['message']
      elif how == 'q':
          m = get('https://api.crossref.org/works?rows=1&query.bibliographic=' + urllib.parse.quote(val))['message']['items'][0]
      else:
          m = get('https://api.crossref.org/works?rows=1&filter=type:journal-article&query.bibliographic=' + urllib.parse.quote(val))['message']['items'][0]
      print(f"{key:15s} {how:3s} -> {(m.get('title') or [''])[0][:90]} | {(m.get('container-title') or [''])[0][:30]} | {m.get('DOI')} | {len(m.get('author', []))} authors", flush=True)
      out.append(bib(key, m))
      time.sleep(0.3)
  except Exception as e:
    print(f'{key:15s} FAILED {e}', flush=True)
MANUAL = r'''
@article{jin2025,
  author = {Jin, Jessica and Oliver, Wesley and Webb, Michael A. and Jacobs, William M.},
  title = {{Predicting Heteropolymer Phase Separation Using Two-Chain Contact Maps}},
  journal = {arXiv},
  year = {2025},
  note = {arXiv:2503.04659}
}

@book{degennes1979,
  author = {de Gennes, Pierre-Gilles},
  title = {Scaling Concepts in Polymer Physics},
  publisher = {Cornell University Press},
  address = {Ithaca, NY},
  year = {1979}
}

'''
open('refs.bib', 'w').write('\n'.join(out) + MANUAL)
