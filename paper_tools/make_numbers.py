"""
make_numbers.py: every number quoted in the manuscript, recomputed from the per-slab result files on the NAS, written as LaTeX macros
to numbers.tex. Rerun after any analysis update; the text never contains a typed number from the results.
  ~/.venvs/finches/bin/python tools/make_numbers.py
"""
import csv
import os
import re

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

R = '/Volumes/User Homes/Nu Project/DYNASEP/hub_condensate_followup/results'
TAB = '/Volumes/User Homes/Nu Project/DYNASEP/hub_condensate_followup/data/pspred_github'
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(path):
    return list(csv.DictReader(open(os.path.join(R, path))))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def tex(x, digits=2):
    """A number as it should appear in running text of the manuscript, rounded to `digits` decimals.
    The result is pasted into LaTeX text mode (not math mode), for example 'the correlation was \\hFourRho{}'."""
    s = f'{x:.{digits}f}'
    if float(s) == 0:
        s = s.lstrip('-')                                       # never print a negative zero
    return '\\ensuremath{-}' + s[1:] if s.startswith('-') else s


def format_p(p):
    """P value as it should appear in the text (inside math mode)."""
    if p >= 0.01:
        return f'{p:.2f}'
    e = int(np.floor(np.log10(p)))
    return f'{p / 10 ** e:.0f}\\times10^{{{e}}}'


def main():
    tab = {}
    for f in ('df_training.csv', 'df_validation_IDRome10.csv', 'df_validation_IDRome90.csv'):
        for r in csv.DictReader(open(os.path.join(TAB, f))):
            tab[r['seq_name']] = r
    M = {}
    a2 = [r for r in rows('amendment2/per_slab.csv') if r['tier'] in ('primary', 'replication') and num(r['ratio_ours']) > 10]
    krtap = lambda r: 'KRTAP' in (tab[r['name']].get('gene_name') or '')  # noqa: E731
    for tier, key in (('primary', 'Prim'), ('replication', 'Repl')):
        h = np.array([num(r['h1_partial']) for r in a2 if r['tier'] == tier])
        M[f'n{key}'] = len(h)
        M[f'hOne{key}Med'] = tex(np.median(h), 2)
        M[f'hOne{key}Pos'] = int((h > 0).sum())
        M[f'hOne{key}P'] = format_p(wilcoxon(h, alternative='greater').pvalue)
        M[f'hOne{key}Raw'] = tex(np.median([num(r['raw_pred_vs_inter']) for r in a2 if r['tier'] == tier]), 2)
    h = np.array([num(r['h1_partial']) for r in a2])
    k = np.array([krtap(r) for r in a2])
    M['nAll'] = len(h)
    M['hOneAllMed'] = tex(np.median(h), 2)
    M['hOneAllPos'] = int((h > 0).sum())
    M['nKrtap'] = int(k.sum())
    M['nKrtapNeg'] = int((h[k] <= 0).sum())
    M['nOtherNeg'] = int((h[~k] <= 0).sum())
    M['nNoKrtap'] = int((~k).sum())
    M['hOneNoKrtapPos'] = int((h[~k] > 0).sum())
    M['hOneNoKrtapMed'] = tex(np.median(h[~k]), 2)
    dG = np.array([num(tab[r['name']]['dG']) for r in a2])
    ok = ~np.isnan(dG)
    M['hOneDG'] = tex(spearmanr(h[ok], dG[ok]).correlation, 2)
    # H2 to H5
    h2 = [r for r in a2 if r.get('h2_dense_vs_dilute')]
    M['hTwoN'] = len(h2)
    M['hTwoMed'] = tex(np.median([num(r['h2_dense_vs_dilute']) for r in h2]), 2)
    M['hTwoCeil'] = tex(np.median([num(r['h2_ceiling']) for r in h2]), 2)
    M['hTwoPredDil'] = tex(np.median([num(r['pred_vs_dilute']) for r in h2]), 2)
    rg = np.array([num(r['h4_rg_ratio']) for r in h2])
    nu = np.array([num(r['nu']) for r in h2])
    has_nu = ~np.isnan(nu)                                      # three slabs have no nu in the published table
    M['hFourN'] = int(has_nu.sum())
    M['hFourRho'] = tex(spearmanr(rg[has_nu], nu[has_nu]).correlation, 2)
    M['hFourRatio'] = tex(np.median(rg), 2)
    h5 = [r for r in a2 if r.get('h5_share_aro')]
    d5 = np.array([num(r['h5_share_aro']) - num(r['h5_share_gly']) for r in h5])
    M['hFiveN'] = len(h5)
    M['hFivePos'] = int((d5 > 0).sum())
    M['hFiveMed'] = tex(np.median(d5), 3)
    M['hFiveP'] = format_p(wilcoxon(d5, alternative='greater').pvalue)
    for tier, key in (('primary', 'Prim'), ('replication', 'Repl')):
        sub = [r for r in a2 if r['tier'] == tier and not np.isnan(num(r['kappa']))]
        M[f'hThree{key}'] = tex(spearmanr([num(r['h1_partial']) for r in sub], [num(r['kappa']) for r in sub]).correlation, 2)
    # force-field-specific degree models
    ff = rows('ff_specific/per_slab.csv')
    c = np.array([num(r['calvados_only']) for r in ff])
    m = np.array([num(r['recharged_only']) for r in ff])
    s = np.array([num(r['consensus']) for r in ff])
    M['ffN'] = len(ff)
    M['ffCons'], M['ffCal'], M['ffRech'] = (tex(np.median(v), 2) for v in (s, c, m))
    M['ffRechPos'] = int((m > 0).sum())
    M['ffGap'] = tex(np.median(c - m), 2)
    M['ffGapP'] = format_p(wilcoxon(c - m).pvalue)
    M['ffAgree'] = tex(np.median([num(r['cal_vs_rech_pred']) for r in ff]), 2)
    fG = np.array([tab[r['name']]['fasta'].count('G') / len(tab[r['name']]['fasta']) for r in ff])
    lo, hi = fG < 0.15, fG >= 0.15
    M['ffLowN'], M['ffHighN'] = int(lo.sum()), int(hi.sum())
    M['ffLowGap'], M['ffHighGap'] = tex(np.median(c[lo] - m[lo]), 2), tex(np.median(c[hi] - m[hi]), 2)
    M['ffHighGapP'] = format_p(wilcoxon(c[hi] - m[hi]).pvalue)
    M['ffHighCal'], M['ffHighRech'] = tex(np.median(c[hi]), 2), tex(np.median(m[hi]), 2)
    M['ffLowRechPos'] = int((m[lo] > 0).sum())
    # saturation and noise ceiling
    sat = rows('saturation/per_slab.csv')
    ks = np.array([r['krtap'] == 'True' for r in sat])
    T = np.array([num(r['T_mean']) for r in sat])
    cv = np.array([num(r['T_cv']) for r in sat])
    hs = np.array([num(r['h1']) for r in sat])
    M['satKrtapT'], M['satOtherT'] = tex(np.median(T[ks]), 1), tex(np.median(T[~ks]), 1)
    M['satKrtapCV'], M['satOtherCV'] = tex(100 * np.median(cv[ks]), 0), tex(100 * np.median(cv[~ks]), 0)
    M['satCVP'] = format_p(mannwhitneyu(cv[ks], cv[~ks]).pvalue)
    M['satRhoCV'] = tex(spearmanr(hs, cv).correlation, 2)
    M['satRhoXWKrtap'] = tex(np.median([num(r['rho_X_W']) for r in sat if r['krtap'] == 'True']), 2)
    rel = rows('reliability/per_slab.csv')
    ce = np.array([num(r['ceiling_halves']) for r in rel])
    M['ceilMed'], M['ceilMin'] = tex(np.median(ce), 3), tex(ce.min(), 2)
    # example slab of Fig. 1
    ex = [r for r in a2 if r['name'] == 'Q96P88_140_292'][0]
    M['exHOne'], M['exRaw'] = tex(num(ex['h1_partial']), 2), tex(num(ex['raw_pred_vs_inter']), 2)
    # overlap with the degree model's training proteins
    ov = open(os.path.join(R, 'overlap_robustness.txt')).read()
    M['nOverlap'] = len(re.findall(r"'[A-Z0-9]+_\d+_\d+'", ov.split('\n')[1]))
    M['overlapReplMed'] = tex(float(re.search(r'replication: .*?without overlap \d+ median ([+-][\d.]+)', ov).group(1)), 2)
    # Test A
    ta = open(os.path.join(R, 'amendment4', 'test_a.txt')).read()
    m2 = re.search(r'A2 partial .*? ([+-][\d.]+), one-sided P ([\d.]+), n (\d+)', ta)
    M['testAr'], M['testAP'], M['testAn'] = tex(float(m2.group(1)), 2), f'{float(m2.group(2)):.3f}', int(m2.group(3))
    pairs = re.findall(r'U_obs ([\d.]+) / ([\d.]+), U_pred ([\d.]+) / ([\d.]+)', ta)
    conc = [(float(a) > float(b)) == (float(p) > float(q)) for a, b, p, q in pairs if a != b and p != q]
    M['testAPairs'], M['testAConc'] = len(conc), int(sum(conc))
    tc = open(os.path.join(R, 'amendment4', 'test_a_composition_check.txt')).read()
    M['testArComp'] = tex(float(re.search(r'beyond 8 composition fractions as well: ([+-][\d.]+)', tc).group(1)), 2)
    # CAPRIN1 against digitized intermolecular NOEs
    cp = open(os.path.join(R, 'caprin1_digitized_comparison.txt')).read()
    M['capN'] = int(re.search(r'residues compared: (\d+)', cp).group(1))
    M['capRaw'] = tex(float(re.search(r'raw Spearman, predicted degree vs NOE: ([+-][\d.]+)', cp).group(1)), 2)
    M['capPartial'] = tex(float(re.search(r'partial beyond local stickiness and chain position \(the H1 baseline\): ([+-][\d.]+)', cp).group(1)), 2)
    M['capWin'] = tex(float(re.search(r'local stickiness window alone vs NOE: ([+-][\d.]+)', cp).group(1)), 2)
    M['capShift'] = int(re.search(r'observed exceeds (\d+)% of shifts', cp).group(1))
    dg = np.genfromtxt('/Volumes/User Homes/Nu Project/DYNASEP/hub_condensate_followup/data/experimental/caprin1_kim2021_si/'
                       'fig4b_sum_noe_digitized.csv', delimiter=',', skip_header=1)
    both = ~np.isnan(dg[:, 1]) & ~np.isnan(dg[:, 2])
    M['capRep'] = tex(spearmanr(dg[both, 1], dg[both, 2]).correlation, 2)   # agreement of the two digitized replicates
    # simulated single chains (IDRome, Tesei et al. 2024) against the degree models, on the same slabs
    sc = {x['name']: x for x in rows('single_chain/idrome_310K_per_slab.csv')}
    ffd = {x['name']: x for x in ff}
    common = [nm for nm in sc if nm in ffd]
    sim = np.array([num(sc[nm]['h1_simulated']) for nm in common])
    cons = np.array([num(ffd[nm]['consensus']) for nm in common])
    calo = np.array([num(ffd[nm]['calvados_only']) for nm in common])
    gly = np.array([tab[nm]['fasta'].count('G') / len(tab[nm]['fasta']) for nm in common])
    M['idN'], M['idPos'] = len(common), int((sim > 0).sum())
    M['idSim'], M['idCons'], M['idCal'] = tex(np.median(sim), 2), tex(np.median(cons), 2), tex(np.median(calo), 2)
    M['idDiffCal'], M['idDiffCalP'] = tex(np.median(sim - calo), 2), f'{wilcoxon(sim - calo).pvalue:.2f}'
    lo, hi = gly < 0.15, gly >= 0.15
    M['idLowN'], M['idHighN'] = int(lo.sum()), int(hi.sum())
    M['idLowSim'], M['idLowCons'] = tex(np.median(sim[lo]), 2), tex(np.median(cons[lo]), 2)
    M['idHighSim'], M['idHighCons'] = tex(np.median(sim[hi]), 2), tex(np.median(cons[hi]), 2)
    M['idHighP'] = format_p(wilcoxon(sim[hi] - cons[hi]).pvalue)
    M['idAgree'] = tex(np.nanmedian([num(sc[nm]['sim_vs_pred']) for nm in common]), 2)
    # YTH-domain IDRs from plants and other non-human species (the 'designed' tier of slab_test, set aside by the analysis plan)
    yth = [r for r in rows('amendment2/per_slab.csv') if r['tier'] == 'designed' and num(r['ratio_ours']) > 10]
    hy = np.array([num(r['h1_partial']) for r in yth])
    M['ythN'], M['ythPos'] = len(hy), int((hy > 0).sum())
    M['ythMed'], M['ythMin'] = tex(np.median(hy), 2), tex(hy.min(), 2)
    M['ythAll'] = len([r for r in rows('amendment2/per_slab.csv') if r['tier'] == 'designed'])
    # Mpipi-Recharged slabs: 12 sequences at 293 K, the 8 that dissolved rerun at 280 K and then 250 K; each sequence at the warmest
    # temperature at which it phase separates; Q5K131_1_121 left out because its 280 K run did not finish
    r293 = rows('recharged_slab/per_slab.csv')
    lad = rows('recharged_slab/ladder_per_run.csv')
    chosen = {r['name']: r for r in r293 if num(r['ratio_ours']) > 10}
    for r in sorted(lad, key=lambda r: -num(r['temp'])):
        if num(r['ratio_ours']) > 10 and r['name'] not in chosen:
            chosen[r['name']] = r
    ch = list(chosen.values())
    cold = [r for r in ch if r in lad]
    hr = np.array([num(r['h1_partial']) for r in ch])
    calo, recho = (np.array([num(r[k]) for r in ch]) for k in ('h1_calvados_only', 'h1_recharged_only'))
    own = np.array([num(r['h1_own_dilute']) for r in ch if r['h1_own_dilute'] not in ('', 'nan')])
    own = own[~np.isnan(own)]
    M['rechSeq'] = len(r293)
    M['rechWarm'] = len(ch) - len(cold)
    M['rechCold'] = len(cold)
    M['rechN'], M['rechPos'] = len(hr), int((hr > 0).sum())
    M['rechMed'], M['rechMin'], M['rechMax'] = tex(np.median(hr), 2), tex(hr.min(), 2), tex(hr.max(), 2)
    M['rechP'] = format_p(wilcoxon(hr, alternative='greater').pvalue)
    M['rechCalSlab'] = tex(np.median([num(r['h1_cal_slab']) for r in ch]), 2)
    M['rechMedWarm'] = tex(np.median([num(r['h1_partial']) for r in ch if r not in cold]), 2)
    M['rechMedCold'] = tex(np.median([num(r['h1_partial']) for r in cold]), 2)
    M['rechModelRech'], M['rechModelCal'] = tex(np.median(recho), 2), tex(np.median(calo), 2)
    M['rechModelWins'] = int((recho > calo).sum())
    M['rechOwnN'], M['rechOwnMin'], M['rechOwnMax'] = len(own), tex(own.min(), 2), tex(own.max(), 2)
    M['rechNotPS'] = len({r['name'] for r in lad} - set(chosen))
    with open(os.path.join(HERE, 'numbers.tex'), 'w') as fh:
        fh.write('% Generated by tools/make_numbers.py from the per-slab result files. Do not edit by hand.\n')
        for key, v in M.items():
            fh.write(f'\\newcommand{{\\{key}}}{{{v}}}\n')
    print(f'{len(M)} macros -> numbers.tex')


if __name__ == '__main__':
    main()
