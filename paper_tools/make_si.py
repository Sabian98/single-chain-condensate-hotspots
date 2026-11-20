"""
make_si.py: Supporting Information material built from the per-slab result files on the NAS.
  SI/Table_S1.xlsx      one row per slab (all 209 analyzed), with metadata and every per-slab result, plus a column sheet
  si_table.tex          condensed Table S1 for the SI PDF (longtable rows)
  figs/si_atlas_*.pdf   profile atlas, predicted single-chain degree and between-chain degree for every included slab,
                        24 panels per page, each set starting on a new page, ordered by agreement within a set
  ~/.venvs/finches/bin/python tools/make_si.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xlsx_writer import write_xlsx  # noqa: E402

ROOT = '/Volumes/User Homes/Nu Project/DYNASEP/hub_condensate_followup'
RES = os.path.join(ROOT, 'results')
TAB = os.path.join(ROOT, 'data', 'pspred_github')
PAPER = os.path.dirname(HERE)
SI = os.path.join(PAPER, 'SI')
FIGS = os.path.join(PAPER, 'figs')
PER_PAGE = 24
# three replication slabs have blank metadata in df_training.csv; gene and protein names from UniProt (looked up 2026-11-14)
FILL = {'P31942': ('HNRNPH3', 'Heterogeneous nuclear ribonucleoprotein H3'),
        'Q5VWX1': ('KHDRBS2', 'KH domain-containing, RNA-binding, signal transduction-associated protein 2'),
        'Q9UK96': ('FBXO10', 'F-box only protein 10')}
SET_LABEL = {'primary': 'Primary', 'replication': 'Replication', 'designed': 'Plant YTHDF'}
# plant slab name prefix -> species and group (Flores-Tellez et al. 2023 sequences used by von Bulow et al. 2025). Cp is assigned to
# the moss Ceratodon purpureus from its near-identical N terminus to the Marchantia protein (that study abbreviates the moss Cpu).
PLANT = {'Mpo': ('Marchantia polymorpha', 'liverwort'), 'Cp': ('Ceratodon purpureus', 'moss'), 'Cri': ('Ceratopteris richardii', 'fern'),
         'Gb': ('Ginkgo biloba', 'gymnosperm'), 'Atr': ('Amborella trichopoda', 'flowering plant'),
         'ECT': ('Arabidopsis thaliana', 'flowering plant')}


def rows(path):
    return {r['name']: r for r in csv.DictReader(open(os.path.join(RES, path)))}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def meta():
    tab = {}
    for f in ('df_training.csv', 'df_validation_IDRome10.csv', 'df_validation_IDRome90.csv'):
        for r in csv.DictReader(open(os.path.join(TAB, f))):
            tab[r['seq_name']] = r
    return tab


def species(name):
    for p in sorted(PLANT, key=len, reverse=True):
        if name.startswith(p):
            return PLANT[p]
    return ('Homo sapiens', 'human')


def build():
    a2 = rows('amendment2/per_slab.csv')
    ff, sat, rel = rows('ff_specific/per_slab.csv'), rows('saturation/per_slab.csv'), rows('reliability/per_slab.csv')
    ido, ta = rows('single_chain/idrome_310K_per_slab.csv'), rows('amendment4/test_a_per_slab.csv')
    tab = meta()
    out = []
    for name, r in a2.items():
        m = tab.get(name, {})
        seq = m.get('fasta', '')
        sp, grp = species(name)
        tier = r['tier']
        inc = num(r['ratio_ours']) > 10
        gene = m.get('gene_name') or ''
        first, last = (m.get('first') or ''), (m.get('last') or '')
        if tier != 'designed' and not m.get('UniProt_ID'):          # blank metadata: parse ACCESSION_FIRST_LAST from the name
            acc, first, last = name.split('_')
            m = dict(m, UniProt_ID=acc, protein_name=FILL[acc][1])
            gene = FILL[acc][0]
        if tier == 'designed':                                    # plant slabs: whole N-terminal IDR as named
            gene, first, last = name, '', ''
        out.append({
            'name': name, 'uniprot': m.get('UniProt_ID') or '', 'first': int(num(first)) if first else '',
            'last': int(num(last)) if last else '', 'protein': m.get('protein_name') or '', 'gene': gene,
            'species': sp, 'group': grp, 'set': SET_LABEL[tier], 'tier': tier, 'included': inc,
            'exclusion': '' if inc else 'Not phase separated in the analyzed frames (dense to dilute bead density ratio 10 or below)',
            'N': int(num(r['N'])), 'chains': int(num(r['chains'])), 'frames': int(num(r['frames'])),
            'ratio_table': num(r['table_ratio']), 'ratio_ours': num(r['ratio_ours']),
            'gly': seq.count('G') / len(seq) if seq else float('nan'),
            'aro': sum(seq.count(a) for a in 'FWY') / len(seq) if seq else float('nan'),
            'krtap': 'KRTAP' in gene,
            'h1': num(r['h1_partial']), 'raw': num(r['raw_pred_vs_inter']), 'pos_only': num(r['h1_position_only']),
            'cal': num(ff.get(name, {}).get('calvados_only')), 'rech': num(ff.get(name, {}).get('recharged_only')),
            'sim': num(ido.get(name, {}).get('h1_simulated')),
            'ceiling_x': num(rel.get(name, {}).get('ceiling_halves')),
            'h2': num(r['h2_dense_vs_dilute']), 'h2_ceiling': num(r['h2_ceiling']),
            'rg_dense': num(r['rg_dense']), 'rg_dilute': num(r['rg_dilute']), 'rg_ratio': num(r['h4_rg_ratio']), 'nu': num(r['nu']),
            'kappa': num(r['kappa']), 'scd': num(r['scd']),
            'share_aro': num(r['h5_share_aro']), 'share_gly': num(r['h5_share_gly']), 'share_between': num(r['share_between']),
            'T_mean': num(sat.get(name, {}).get('T_mean')), 'T_cv': num(sat.get(name, {}).get('T_cv')),
            'U_obs': num(ta.get(name, {}).get('U_obs')), 'U_pred': num(ta.get(name, {}).get('U_pred')),
            'log10_cdil': num(ta.get(name, {}).get('log10_cdil')),
        })
    order = {'primary': 0, 'replication': 1, 'designed': 2}
    out.sort(key=lambda d: (order[d['tier']], not d['included'], -d['h1'] if d['included'] else 0, d['name']))
    k = 0
    for d in out:
        if d['included']:
            k += 1
            d['panel'] = k
        else:
            d['panel'] = ''
    return out


COLUMNS = [
    ('Atlas panel', 'panel', 'Panel number in the profile atlas (Figure S1); empty for slabs left out'),
    ('Slab', 'name', 'Sequence name in von Bulow et al. 2025 (UniProt accession and residue range for human IDRs)'),
    ('UniProt', 'uniprot', 'UniProt accession'),
    ('First residue', 'first', 'First residue of the IDR in the full-length protein'),
    ('Last residue', 'last', 'Last residue of the IDR in the full-length protein'),
    ('Gene', 'gene', 'Gene name (plant slabs, the sequence name)'),
    ('Protein', 'protein', 'Protein name'),
    ('Species', 'species', 'Source species. Plant prefix Cp is assigned to the moss Ceratodon purpureus by sequence similarity'),
    ('Taxon group', 'group', 'Human, or the land-plant group of the plant YTHDF proteins'),
    ('Set', 'set', 'Primary (validation sets of von Bulow et al.), replication (their natural training sequences) or plant YTHDF'),
    ('Included', 'included', 'Whether the slab entered the analyses'),
    ('Reason left out', 'exclusion', 'Why a slab was left out'),
    ('Length', 'N', 'Residues per chain'),
    ('Chains', 'chains', 'Chains in the simulation box'),
    ('Frames analyzed', 'frames', 'Frames read from the second half of the trajectory'),
    ('Dense/dilute ratio, original study', 'ratio_table', 'Dense to dilute concentration ratio reported by von Bulow et al.'),
    ('Dense/dilute ratio, our frames', 'ratio_ours', 'Bead density inside the dense half-width over that beyond 50 nm in our frames'),
    ('Glycine fraction', 'gly', 'Fraction of Glycine residues'),
    ('Aromatic fraction', 'aro', 'Fraction of F, W and Y'),
    ('KRTAP', 'krtap', 'Keratin-associated protein'),
    ('Agreement beyond local stickiness', 'h1', 'Partial Spearman correlation of predicted single-chain degree (consensus model) with between-chain degree, beyond the local stickiness baseline and chain position'),
    ('Raw Spearman', 'raw', 'Spearman correlation of predicted single-chain degree with between-chain degree, no baseline'),
    ('Agreement beyond chain position only', 'pos_only', 'Partial Spearman correlation beyond distance to the chain end only'),
    ('Agreement, CALVADOS-2 model', 'cal', 'As agreement beyond local stickiness, with the model trained on CALVADOS-2 labels alone'),
    ('Agreement, Mpipi-Recharged model', 'rech', 'As agreement beyond local stickiness, with the model trained on Mpipi-Recharged labels alone'),
    ('Agreement, simulated single chain', 'sim', 'As agreement beyond local stickiness, with degree from the IDRome single-chain simulation (310 K)'),
    ('Noise ceiling, between-chain degree', 'ceiling_x', 'Highest agreement allowed by sampling noise, from first and second halves of the frames'),
    ('Dense vs dilute within-chain profile', 'h2', 'Spearman correlation of within-chain degree in dense and dilute chains'),
    ('Noise ceiling, within-chain profile', 'h2_ceiling', 'Noise ceiling of the dense vs dilute comparison, from even and odd frames'),
    ('Rg dense (nm)', 'rg_dense', 'Mean radius of gyration of dense-phase chains'),
    ('Rg dilute (nm)', 'rg_dilute', 'Mean radius of gyration of dilute-phase chains'),
    ('Rg ratio, dense/dilute', 'rg_ratio', 'Expansion in the dense phase'),
    ('Scaling exponent nu', 'nu', 'Single-chain scaling exponent from von Bulow et al.'),
    ('kappa', 'kappa', 'Charge patterning parameter from von Bulow et al.'),
    ('SCD', 'scd', 'Sequence charge decoration'),
    ('Aromatic share with other chains', 'share_aro', 'Share of dense-phase contacts of aromatic residues made with other chains'),
    ('Glycine share with other chains', 'share_gly', 'Share of dense-phase contacts of Glycine residues made with other chains'),
    ('Share of contacts with other chains', 'share_between', 'Share of all dense-phase contacts made with other chains'),
    ('Total contacts per residue', 'T_mean', 'Mean total dense-phase degree, within and between chains'),
    ('Variation of total contacts', 'T_cv', 'Coefficient of variation of total degree across positions at least five residues from the ends'),
    ('Unevenness, measured', 'U_obs', 'Gini coefficient of between-chain degree across residues'),
    ('Unevenness, predicted', 'U_pred', 'Gini coefficient of predicted single-chain degree across residues'),
    ('log10 dilute concentration', 'log10_cdil', 'Dilute-phase concentration from von Bulow et al.'),
]


def write_table(data):
    os.makedirs(SI, exist_ok=True)
    rows_ = [[d[k] for _, k, _ in COLUMNS] for d in data]
    write_xlsx(os.path.join(SI, 'Table_S1.xlsx'), [
        ('Table S1', [c for c, _, _ in COLUMNS], rows_),
        ('Columns', ['Column', 'Description'], [[c, desc] for c, _, desc in COLUMNS])])
    # condensed LaTeX rows
    esc = lambda s: str(s).replace('_', r'\_')  # noqa: E731
    L = []
    for d in data:
        res = f"{d['first']}--{d['last']}" if d['first'] != '' else ''
        f = lambda x: '' if x != x else f'{x:.2f}'.replace('-', r'$-$')  # noqa: E731
        L.append(' & '.join([str(d['panel']) if d['panel'] != '' else '--', esc(d['gene'] or d['uniprot']), esc(d['uniprot']) or '',
                             res, d['set'], str(d['N']), str(d['chains']), f(d['h1']), f(d['raw'])]) + r' \\')
    open(os.path.join(PAPER, 'si_table.tex'), 'w').write('\n'.join(L) + '\n')


def atlas(data):
    sys.path.insert(0, HERE)
    import make_figures as M
    plt = M.setup()
    X = np.load(os.path.join(RES, 'single_chain', 'x_profiles.npz'))
    P = np.load(os.path.join(RES, 'amendment2', 'pred_degree.npz'))
    inc = [d for d in data if d['included']]
    pages = []
    for tier in ('primary', 'replication', 'designed'):
        g = [d for d in inc if d['tier'] == tier]
        pages += [g[i:i + PER_PAGE] for i in range(0, len(g), PER_PAGE)]
    for f in os.listdir(FIGS):
        if f.startswith('si_atlas_'):
            os.remove(os.path.join(FIGS, f))
    for p, page in enumerate(pages):
        fig, axes = plt.subplots(6, 4, figsize=(7.0, 8.0))
        fig.subplots_adjust(left=0.04, right=0.99, top=0.935, bottom=0.04, hspace=0.75, wspace=0.12)
        for ax in axes.flat:
            ax.axis('off')
        for ax, d in zip(axes.flat, page):
            ax.axis('on')
            x, pr = X[d['name']].astype(float), P[d['name']].astype(float)
            pos = np.arange(1, len(x) + 1)
            ax.plot(pos, x / x.mean(), color=M.C['slab_line'], lw=0.6)
            ax.plot(pos, pr / pr.mean(), color='k', lw=0.6)
            ax.set_xlim(1, len(x))
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=5, length=1.5, pad=1)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_linewidth(0.4)
            label = d['gene'] if d['tier'] == 'designed' else f"{d['gene'] or d['uniprot']} {d['first']}–{d['last']}"
            col = '#EE3377' if d['krtap'] else 'k'
            ax.set_title(f"{d['panel']}  {label}", fontsize=5.6, loc='left', pad=6.5, color=col)
            ax.text(0, 1.02, f"Beyond stickiness {d['h1']:.2f}, raw {d['raw']:.2f}".replace('-', '−'), transform=ax.transAxes,
                    fontsize=4.9, color='0.35', va='bottom')
        fig.text(0.5, 0.008, 'Residue', ha='center', fontsize=6.5)
        fig.text(0.008, 0.5, 'Degree relative to the chain mean', rotation=90, va='center', fontsize=6.5)
        h = [plt.Line2D([], [], color='k', lw=0.8), plt.Line2D([], [], color=M.C['slab_line'], lw=0.8)]
        fig.legend(h, ['Predicted single-chain degree', 'Contacts with other chains in the condensate'], loc='upper center',
                   ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.998), handlelength=1.8)
        fig.savefig(os.path.join(FIGS, f'si_atlas_{p + 1:02d}.pdf'), dpi=300)
        plt.close(fig)
    meta_ = [(p + 1, page[0]['set'], page[0]['panel'], page[-1]['panel']) for p, page in enumerate(pages)]
    open(os.path.join(PAPER, 'si_atlas.tex'), 'w').write('\n'.join(
        rf"\begin{{figure}}[p]\centering\includegraphics[width=\textwidth]{{figs/si_atlas_{p:02d}.pdf}}"
        + (r"\caption{\textbf{Profile atlas of all included slabs.} Predicted single-chain degree (black, consensus model) and "
           r"time-averaged contacts with other chains in the dense phase (blue) along the sequence, each divided by its mean along the "
           r"chain. Panels are numbered as in Table S1 and ordered by agreement beyond local stickiness within each set, which is given "
           r"above each panel with the raw Spearman correlation. KRTAPs are labeled in magenta. Each set starts on a new page. "
           rf"This page shows {s.lower() if s != 'Plant YTHDF' else 'the plant YTHDF'} slabs, panels {a} to {b}.}}\label{{fig:atlas}}"
           if p == 1 else rf"\caption*{{\textbf{{Figure S1, continued.}} {s} slabs, panels {a} to {b}.}}")
        + r"\end{figure}" for p, s, a, b in meta_) + '\n')
    return len(pages)


def recharged_table():
    """Rows of SI Table S4: the 12 Mpipi-Recharged sequences, the temperature analyzed (warmest that phase separates), and agreement."""
    tab = meta()
    r293 = rows('recharged_slab/per_slab.csv')
    lad = list(csv.DictReader(open(os.path.join(RES, 'recharged_slab', 'ladder_per_run.csv'))))
    f = lambda x: '' if x != x else f'{x:.2f}'.replace('-', r'$-$')  # noqa: E731
    out = []
    for name, r in r293.items():
        m = tab.get(name, {})
        gene = m.get('gene_name') or FILL.get(name.split('_')[0], (name,))[0]
        res = f"{m.get('first')}--{m.get('last')}" if m.get('first') else '--'.join(name.split('_')[1:])
        gly = 100 * m.get('fasta', '').count('G') / max(len(m.get('fasta', '')), 1)
        cand = ([dict(r, temp='293')] if num(r['ratio_ours']) > 10 else
                sorted([x for x in lad if x['name'] == name and num(x['ratio_ours']) > 10], key=lambda x: -num(x['temp']))[:1])
        if cand:
            c = cand[0]
            cells = [c['temp'], 'yes', f(num(c['h1_partial'])), f(num(c['h1_cal_slab'])), f(num(c['h1_calvados_only'])),
                     f(num(c['h1_recharged_only']))]
        else:
            tried = sorted({x['temp'].split('.')[0] for x in lad if x['name'] == name}, key=float)
            if not tried:                                         # run below 293 K not completed (Q5K131_1_121), left out by the user
                continue
            note = f'mixed to {min(tried, key=float)}'
            cells = [note, 'no', '', f(num(r['h1_cal_slab'])), '', '']
        out.append((0 if cand else 1, -(num(cand[0]['h1_partial']) if cand else 0), ' & '.join([gene, res, f'{gly:.0f}'] + cells) + r' \\'))
    open(os.path.join(PAPER, 'si_recharged.tex'), 'w').write('\n'.join(x[2] for x in sorted(out)) + '\n\\bottomrule\n')   # rule inside the file: \bottomrule after \input breaks


def main():
    recharged_table()
    data = build()
    write_table(data)
    n = atlas(data)
    inc = [d for d in data if d['included']]
    print(f'{len(data)} slabs, {len(inc)} included, {n} atlas pages; left out: {[d["name"] for d in data if not d["included"]]}')
    print({s: sum(d['set'] == s for d in inc) for s in SET_LABEL.values()})


if __name__ == '__main__':
    main()
