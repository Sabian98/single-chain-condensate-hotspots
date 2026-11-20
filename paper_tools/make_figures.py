"""
make_figures.py: all manuscript figures from the result files on the NAS, as PDFs in figs/.
  ~/.venvs/finches/bin/python tools/make_figures.py --prep    one-time cache for Fig. 1 (example slab profiles and one frame)
  ~/.venvs/finches/bin/python tools/make_figures.py           all figures
Palette after the CALVADOS software paper (von Buelow et al. 2025, arXiv 2504.10408; Paul Tol 'vibrant'), colour-blind checked
(CIELAB gap >= 31 under protan, deutan and tritan simulation for every pair that shares a panel): slab chains and measured contacts
and the CALVADOS-2-only model blue, predicted degree (consensus model) black, Mpipi-Recharged-only model orange, experiment red,
KRTAPs magenta diamonds against blue circles. Full box frames, Arial throughout.
"""
import csv
import os
import sys

import numpy as np

ROOT = '/Volumes/User Homes/Nu Project/DYNASEP/hub_condensate_followup'
RES = os.path.join(ROOT, 'results')
TAB = os.path.join(ROOT, 'data', 'pspred_github')
PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(PAPER, 'figs')
CACHE = os.path.join(RES, 'figure_cache')
EXAMPLE = 'Q96P88_140_292'
C = {'slab': '#0077BB', 'slab_line': '#005A8C', 'pred': '#000000', 'cal': '#0077BB', 'cal_line': '#005A8C', 'rech': '#EE7733',
     'rech_line': '#B85315', 'exp': '#CC3311', 'exp_line': '#8F240C', 'krtap': '#EE3377', 'krtap_line': '#A8204F', 'other': '#0077BB',
     'other_line': '#005A8C', 'grid': '#E6E6E6'}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def rows(path):
    return list(csv.DictReader(open(os.path.join(RES, path))))


def table():
    tab = {}
    for f in ('df_training.csv', 'df_validation_IDRome10.csv', 'df_validation_IDRome90.csv'):
        for r in csv.DictReader(open(os.path.join(TAB, f))):
            tab[r['seq_name']] = r
    return tab


def prep():
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import mdtraj as md
    import slab_test as S
    import explore_saturation as ES
    s = [x for x in S.slabs() if x['name'] == EXAMPLE][0]
    X, W = ES.profiles(s)
    pred = np.load(os.path.join(RES, 'amendment2', 'pred_degree.npz'))[EXAMPLE]
    dcd = os.path.join(S.FR, s['set'], EXAMPLE, f'{EXAMPLE}_sub.dcd')
    n = s['N']
    natoms = md.formats.DCDTrajectoryFile(dcd).read(n_frames=1)[0].shape[1]
    top = md.Topology()
    for _ in range(natoms // n):
        ch = top.add_chain()
        for _ in range(n):
            top.add_atom('CA', md.element.carbon, top.add_residue('ALA', ch))
    t = md.load_dcd(dcd, top=top)
    f = t.n_frames - 1
    box = t.unitcell_lengths[f]
    xyz = np.mod(t.xyz[f], box)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, 'fig1_example.npz'), X=X, W=W, pred=pred, xyz=xyz, box=box, n=n, seq=np.array(list(s['seq'])))
    print('cached', EXAMPLE)


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'axes.spines.top': True, 'axes.spines.right': True, 'pdf.fonttype': 42,
                         'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial',
                         'mathtext.bf': 'Arial:bold', 'mathtext.default': 'regular'})
    return plt


def letter(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom', ha='left')


def half_violin(ax, data, x, color, line, width=0.38, points=True, rng=np.random.default_rng(0)):
    from scipy.stats import gaussian_kde
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    ys = np.linspace(data.min(), data.max(), 200)
    dens = gaussian_kde(data)(ys)
    dens = dens / dens.max() * width
    ax.fill_betweenx(ys, x - dens, x, color=color, alpha=0.55, lw=0)
    ax.plot(x - dens, ys, color=line, lw=0.6)
    if points:
        ax.scatter(x + 0.06 + rng.uniform(0, 0.22, len(data)), data, s=7, color=color, edgecolor=line, linewidth=0.3, zorder=3)
    med = np.median(data)
    ax.plot([x - width, x + 0.3], [med, med], color=line, lw=1.2, zorder=4)


# ---------- data ----------
def slab_rows():
    return [r for r in rows('amendment2/per_slab.csv') if r['tier'] in ('primary', 'replication') and num(r['ratio_ours']) > 10]


def is_krtap(tab, name):
    return 'KRTAP' in (tab[name].get('gene_name') or '')


def scat(ax, x, y, color=None, line=None, s=9, **kw):
    ax.scatter(x, y, s=s, color=color or C['slab'], edgecolor=line or C['slab_line'], linewidth=0.3, **kw)


# ---------- example slab panels ----------
def example_panels(fig, ax_snap, ax_zoom, ax_pred, ax_meas, tab):
    """Panels a and b are ChimeraX renders made by tools/render_chimerax.py (style of the CALVADOS software paper, Fig. 6C)."""
    import matplotlib.image as mpimg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    d = np.load(os.path.join(CACHE, 'fig1_example.npz'))
    X, pred, n = d['X'], d['pred'], int(d['n'])

    def crop(img, pad=8):
        rgb = img[..., :3]
        ink = np.where((rgb < 0.97).any(axis=2))
        y0, y1, x0, x1 = ink[0].min(), ink[0].max(), ink[1].min(), ink[1].max()
        return img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]

    render = os.path.join(FIGS, 'render')
    ax_snap.imshow(crop(mpimg.imread(os.path.join(render, 'slab.png'))), interpolation='lanczos')
    ax_snap.axis('off')
    ax_zoom.imshow(crop(mpimg.imread(os.path.join(render, 'zoom.png'))), interpolation='lanczos')
    ax_zoom.axis('off')
    cmap = LinearSegmentedColormap.from_list('contacts', ['#8EC3E6', '#0077BB', '#00243A'])   # the palette of render_chimerax.py
    pos = np.arange(1, n + 1)
    ax_pred.plot(pos, pred, color=C['pred'], lw=1.0)
    ax_pred.set_ylabel('Predicted single-\nchain degree')
    ax_pred.set_xlabel('Residue')
    ax_pred.set_title('Predicted from sequence', loc='left', fontsize=8)
    ax_meas.plot(pos, X, color=C['slab_line'], lw=1.0)
    ax_meas.set_ylabel('Contacts with\nother chains')
    ax_meas.set_xlabel('Residue')
    ax_meas.set_title('Measured in the condensate', loc='left', fontsize=8)
    ax_meas.set_xlim(1, n)


def h1_violins(ax, a2):
    yth = [r for r in rows('amendment2/per_slab.csv') if r['tier'] == 'designed' and num(r['ratio_ours']) > 10]
    groups = [[num(r['h1_partial']) for r in a2 if r['tier'] == 'primary'], [num(r['h1_partial']) for r in a2 if r['tier'] == 'replication'],
              [num(r['h1_partial']) for r in yth]]
    for i, v in enumerate(groups):
        half_violin(ax, v, i, C['slab'], C['slab_line'], width=0.34)
    ax.axhline(0, color='0.5', lw=0.6, ls='--')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f'Primary\n(n = {len(groups[0])})', f'Replication\n(n = {len(groups[1])})', f'Plant\nYTHDF\n(n = {len(groups[2])})'],
                       fontsize=6.5)
    ax.set_xlim(-0.45, 2.4)
    ax.set_ylabel('Agreement beyond\nlocal stickiness')


def h1_ranked(ax, a2, tab):
    from matplotlib.patches import Patch
    srt = sorted(a2, key=lambda r: num(r['h1_partial']))
    k = np.array([is_krtap(tab, r['name']) for r in srt])
    v = np.array([num(r['h1_partial']) for r in srt])
    x = np.arange(len(v))
    ax.bar(x[~k], v[~k], width=0.85, color=C['slab'], lw=0)
    ax.bar(x[k], v[k], width=0.85, color=C['krtap'], lw=0)
    ax.axhline(0, color='0.5', lw=0.6)
    ax.set_xlim(-1, len(v))
    ax.set_xticks([])
    ax.set_xlabel(f'Slabs, ranked ({len(v)})')
    ax.set_ylabel('Agreement beyond\nlocal stickiness')
    ax.legend(handles=[Patch(color=C['slab'], label='Other proteins'), Patch(color=C['krtap'], label='KRTAPs')], frameon=False, loc='upper left')


# ---------- physics panels ----------
def p_h2(ax, a2):
    h2 = [r for r in a2 if r.get('h2_dense_vs_dilute')]
    cx, cy = np.array([num(r['h2_ceiling']) for r in h2]), np.array([num(r['h2_dense_vs_dilute']) for r in h2])
    lo = np.floor(min(cx.min(), cy.min()) * 10) / 10
    ax.plot([lo, 1], [lo, 1], color='0.6', lw=0.6, ls='--')
    scat(ax, cx, cy)
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_xlabel('Noise ceiling')
    ax.set_ylabel('Dense against dilute\nwithin-chain profile')


def p_h4(ax, a2):
    ok = [r for r in a2 if r.get('h2_dense_vs_dilute') and not np.isnan(num(r['nu']))]
    scat(ax, [num(r['nu']) for r in ok], [num(r['h4_rg_ratio']) for r in ok])
    ax.axhline(1, color='0.6', lw=0.6, ls='--')
    ax.set_xlabel('Single-chain scaling exponent ν')
    ax.set_ylabel('Radius of gyration,\ndense over dilute')


def p_h5(ax, a2):
    h5 = [r for r in a2 if r.get('h5_share_aro')]
    g, a = np.array([num(r['h5_share_gly']) for r in h5]), np.array([num(r['h5_share_aro']) for r in h5])
    lo, hi = min(g.min(), a.min()) - 0.02, max(g.max(), a.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color='0.6', lw=0.6, ls='--')
    scat(ax, g, a)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Share of Glycine contacts\nwith other chains')
    ax.set_ylabel('Share of aromatic contacts\nwith other chains')


def p_h3(ax, a2):
    for tier, mk, lab in (('primary', 's', 'Primary'), ('replication', 'o', 'Replication')):
        sub = [r for r in a2 if r['tier'] == tier and not np.isnan(num(r['kappa']))]
        if tier == 'primary':
            ax.scatter([num(r['kappa']) for r in sub], [num(r['h1_partial']) for r in sub], s=14, marker=mk, facecolor='white',
                       edgecolor=C['slab_line'], linewidth=0.8, label=lab, zorder=3)
        else:
            scat(ax, [num(r['kappa']) for r in sub], [num(r['h1_partial']) for r in sub], alpha=0.7, label=lab)
    ax.axhline(0, color='0.5', lw=0.6, ls='--')
    ax.set_xlabel('Charge patterning κ')
    ax.set_ylabel('Agreement beyond\nlocal stickiness')
    ax.legend(frameon=False, loc='lower right')


# ---------- limits panels ----------
def p_ff(ax, ff):
    """Degree models trained on each force field's single-chain labels, and simulated CALVADOS-2 single chains (IDRome, Tesei et al.
    2024, 310 K) for the slabs that IDRome covers."""
    sim = [num(r['h1_simulated']) for r in rows('single_chain/idrome_310K_per_slab.csv')]
    groups = ((([num(r['consensus']) for r in ff]), C['pred'], C['pred']), (([num(r['calvados_only']) for r in ff]), C['cal'], C['cal_line']),
              (([num(r['recharged_only']) for r in ff]), C['rech'], C['rech_line']), (sim, '#9A9A9A', '#5A5A5A'))
    for i, (vals, col, line) in enumerate(groups):
        half_violin(ax, vals, i, col, line, width=0.34)
    ax.axhline(0, color='0.5', lw=0.6, ls='--')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Model\nboth force\nfields', 'Model\nCALVADOS-2', 'Model\nMpipi-\nRecharged', 'Simulated\nCALVADOS-2\nchains'],
                       fontsize=6.5)
    ax.set_xlim(-0.45, 3.4)
    ax.set_ylabel('Agreement beyond\nlocal stickiness')


def p_gap(ax, ff, tab):
    fG = np.array([tab[r['name']]['fasta'].count('G') / len(tab[r['name']]['fasta']) for r in ff])
    gap = np.array([num(r['calvados_only']) - num(r['recharged_only']) for r in ff])
    ax.axvline(15, color='0.6', lw=0.6, ls=':')
    ax.axhline(0, color='0.5', lw=0.6, ls='--')
    scat(ax, 100 * fG, gap)
    ax.set_xlabel('Glycine content (%)')
    ax.set_ylabel('CALVADOS-2 only minus\nMpipi-Recharged only')


def p_sat(ax, sat, key, xl, legend=False):
    k = np.array([r['krtap'] == 'True' for r in sat])
    xv = np.array([num(r[key]) for r in sat]) * (100 if key == 'T_cv' else 1)
    h = np.array([num(r['h1']) for r in sat])
    scat(ax, xv[~k], h[~k], alpha=0.75, label='Other proteins')
    ax.scatter(xv[k], h[k], s=18, marker='D', color=C['krtap'], edgecolor=C['krtap_line'], linewidth=0.3, label='KRTAPs', zorder=3)
    ax.axhline(0, color='0.5', lw=0.6, ls='--')
    ax.set_xlabel(xl)
    ax.set_ylabel('Agreement beyond\nlocal stickiness')
    if legend:
        ax.legend(frameon=False, loc='lower left')


# ---------- experiment and organization panels ----------
def p_caprin(axn, axp, axs):
    from scipy.stats import spearmanr
    noe = np.genfromtxt(os.path.join(ROOT, 'data/experimental/caprin1_kim2021_si/fig4b_sum_noe_digitized.csv'), delimiter=',', skip_header=1)
    pr = np.genfromtxt(os.path.join(RES, 'caprin1_pred_degree.csv'), delimiter=',', skip_header=1)
    pos, d = pr[:, 0], pr[:, 1]
    for ax in (axn, axp):
        for a, b in ((624, 626), (638, 640), (660, 666)):
            ax.axvspan(a - 0.5, b + 0.5, color='#E8E8E8', lw=0, zorder=0)
    scat(axn, noe[:, 0], noe[:, 3], C['exp'], C['exp_line'], s=8, zorder=3)
    axn.set_ylabel('Intermolecular\nNOEs (x10$^{-2}$)')
    axn.tick_params(labelbottom=False)
    axp.plot(pos, d, color=C['pred'], lw=1.0, zorder=3)
    axp.set_ylabel('Predicted single-\nchain degree')
    axp.set_xlabel('CAPRIN1 residue')
    axp.set_xlim(606, 710)
    m = {int(p): v for p, v in zip(pos, d)}
    xs = np.array([m[int(r)] for r in noe[:, 0]])
    scat(axs, xs, noe[:, 3], C['exp'], C['exp_line'])
    axs.set_xlabel('Predicted single-chain degree')
    axs.set_ylabel('Intermolecular NOEs (x10$^{-2}$)')
    axs.text(0.05, 0.95, f'Spearman {spearmanr(xs, noe[:, 3]).correlation:.2f}', transform=axs.transAxes, va='top', fontsize=7)


def p_testA(ax, tab):
    from scipy.stats import rankdata, spearmanr
    ta = rows('amendment4/test_a_per_slab.csv')
    Uo, Up = np.array([num(r['U_obs']) for r in ta]), np.array([num(r['U_pred']) for r in ta])
    Z = [np.array([num(r[c]) for r in ta]) for c in ('log10_cdil', 'N', 'lambda_mean')]
    A = np.column_stack([np.ones(len(Uo))] + [rankdata(z) for z in Z])
    res = lambda v: rankdata(v) - A @ np.linalg.lstsq(A, rankdata(v), rcond=None)[0]  # noqa: E731
    rx, ry = res(Up), res(Uo)
    k = np.array([is_krtap(tab, r['name']) for r in ta])
    scat(ax, rx[~k], ry[~k], alpha=0.75)
    ax.scatter(rx[k], ry[k], s=18, marker='D', color=C['krtap'], edgecolor=C['krtap_line'], linewidth=0.3, zorder=3)
    b = np.polyfit(rx, ry, 1)
    xx = np.array([rx.min(), rx.max()])
    ax.plot(xx, np.polyval(b, xx), color=C['pred'], lw=0.9)
    ax.set_xlabel('Predicted unevenness\n(rank residual)')
    ax.set_ylabel('Observed unevenness\n(rank residual)')
    ax.text(0.05, 0.95, f'Partial correlation {np.corrcoef(rx, ry)[0, 1]:.2f}', transform=ax.transAxes, va='top', fontsize=7)   # as in amendment4_test.partial


# ---------- composite figures ----------
def sequence_row(fig, ax, per_row=77):
    """Amino-acid letters of the highlighted chain of Figure 1b on cells coloured by its contacts with other chains in that frame
    (same values and palette as the render), with the colorbar at the right."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import render_chimerax as RC
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle
    fr = RC.frame()
    cnt, seq = fr[4], fr[2]
    cmap = LinearSegmentedColormap.from_list('contacts', ['#8EC3E6', '#0077BB', '#00243A'])   # palette of render_chimerax.py
    norm = Normalize(0, 6)
    rows_n = -(-len(seq) // per_row)
    for i, aa in enumerate(seq):
        r, c = divmod(i, per_row)
        col = cmap(norm(min(cnt[i], 6)))
        ax.add_patch(Rectangle((c, -r - 1), 0.92, 0.86, facecolor=col, edgecolor='none'))
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        ax.text(c + 0.46, -r - 0.57, aa, ha='center', va='center', fontsize=6.3, color='white' if lum < 0.45 else 'black')
    for r in range(rows_n):
        ax.text(-0.5, -r - 0.57, str(r * per_row + 1), ha='right', va='center', fontsize=6, color='0.35')
    ax.set_xlim(-2.5, per_row)
    ax.set_ylim(-rows_n - 0.1, 1.7)                              # headroom inside the panel for the title and colorbar
    ax.axis('off')
    ax.text(-2.5, 1.05, 'Sequence', fontsize=7.5, va='bottom', ha='left')
    x0 = 0.64 * per_row
    cax = ax.inset_axes([x0, 1.0, 0.35 * per_row, 0.3], transform=ax.transData)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cax.text(-0.03, 0.5, 'Contacts with other chains', ha='right', va='center', transform=cax.transAxes, fontsize=6.5)


def figure_concept(plt, tab):
    """Conceptual overview in four numbered cards: sequence (letters on tiles shaded by single-chain degree), single chain with its
    contact network (IDRome render, same shading), single-chain degree profile (simulated as a shaded area, predicted as a dashed
    line), condensate (render of a chain coloured by contacts with other chains). A shortcut arrow marks the degree model and a boxed
    note states the mean-field expectation."""
    import matplotlib.image as mpimg
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
    render = os.path.join(FIGS, 'render')
    single = np.load(os.path.join(CACHE, 'single_degree_Q96P88_140_292.npz'))
    pred = np.load(os.path.join(CACHE, 'fig1_example.npz'))['pred']
    seq = ''.join(single['seq'])
    sim = single['degree']
    rel = sim / sim.mean()
    cmap = LinearSegmentedColormap.from_list('deg', [(0.0, '#E8E8E8'), (0.5, '#9A9A9A'), (1.0, '#202020')])   # same as the render
    shade = lambda r: cmap(np.clip((r - 0.5) / 1.0, 0, 1))

    def crop(img, pad=10):
        ink = np.where((img[..., :3] < 0.97).any(axis=2))
        return img[max(0, ink[0].min() - pad):ink[0].max() + pad, max(0, ink[1].min() - pad):ink[1].max() + pad]

    def clear_bg(img):
        """RGBA copy with the white background (near-white pixels connected to the border) made transparent, so the render sits
        on the card colour. Highlights inside beads are not connected to the border and stay opaque."""
        from scipy import ndimage
        img = crop(img)
        lab, _ = ndimage.label((img[..., :3] > 0.975).all(axis=2))
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        bg = np.isin(lab, edge[edge > 0])
        bg = ndimage.binary_erosion(~bg, iterations=1) == 0
        return np.dstack([img[..., :3], np.where(bg, 0.0, 1.0)])

    fig = plt.figure(figsize=(7.0, 3.15))
    Y0, Y1 = 0.30, 0.97                                          # card bottom and top
    cards = [(0.005, 0.163), (0.183, 0.371), (0.391, 0.547), (0.567, 0.752), (0.802, 0.995)]
    fills = ['#F3F3F3', '#F3F3F3', '#F3F3F3', '#F3F3F3', '#E9F2F9']
    titles = ['Sequence', 'Single chain', 'Count contacts', 'Single-chain degree', 'Condensate']
    subs = ['Amino acids of an IDR', 'Contacts within the chain', 'In one conformation', 'Averaged over conformations',
            'Contacts with other chains']
    blue = C['slab_line']
    for k, ((x0, x1), fc) in enumerate(zip(cards, fills)):
        fig.patches.append(FancyBboxPatch((x0, Y0), x1 - x0, Y1 - Y0, boxstyle='round,pad=0,rounding_size=0.018',
                                          transform=fig.transFigure, fc=fc, ec='none', zorder=-10, mutation_aspect=7.0 / 3.15))
        col = blue if k == 4 else '0.15'
        fig.text(x0 + 0.015, Y1 - 0.066, str(k + 1), ha='center', va='center', color='white', fontsize=6.0, fontweight='bold',
                 bbox=dict(boxstyle='circle,pad=0.28', fc=col, ec='none'))
        fig.text(x0 + 0.031, Y1 - 0.066, titles[k], va='center', fontsize=7.4, fontweight='bold', color=col)
        fig.text(x0 + 0.031, Y1 - 0.135, subs[k], va='center', fontsize=5.7, color=blue if k == 4 else '0.4')
    # cost of the two simulations, at the foot of cards 2 and 5
    fig.text(0.277, Y0 + 0.045, 'One chain, cheap\ncoarse-grained simulation', ha='center', va='center', fontsize=5.6, color='0.3',
             linespacing=1.25)
    fig.text(0.8985, Y0 + 0.045, '235 chains, costly\nslab simulation', ha='center', va='center', fontsize=5.6, color=blue,
             linespacing=1.25)
    # 1 sequence on degree-shaded tiles
    ax1 = fig.add_axes([0.015, 0.355, 0.138, 0.44])
    ax1.set_xlim(0, 12); ax1.set_ylim(6.0, -0.2); ax1.axis('off')
    for i in range(72):
        r, c = divmod(i, 12)
        ax1.add_patch(FancyBboxPatch((c + 0.06, r + 0.08), 0.88, 0.84, boxstyle='round,pad=0,rounding_size=0.18', fc=shade(rel[i]),
                                     ec='none'))
        ax1.text(c + 0.5, r + 0.52, seq[i], ha='center', va='center', fontsize=4.9,
                 color='white' if rel[i] > 1.05 else '0.15', fontweight='bold')
    # 2 single chain
    ax2 = fig.add_axes([0.188, 0.375, 0.178, 0.44])
    ax2.imshow(clear_bg(mpimg.imread(os.path.join(render, 'single.png'))), interpolation='lanczos')
    ax2.axis('off')
    # 3 counting the contacts of one residue in one conformation: a hairpin, residue i dark, cutoff circle, contacts dashed,
    # chain neighbors |i - j| < 3 faded since they are not counted
    ax3 = fig.add_axes([0.398, 0.345, 0.142, 0.47])
    ax3.set_aspect('equal'); ax3.axis('off')
    sp_ = 0.38                                                    # bead spacing along the chain (nm)
    xa = np.arange(-3, 4) * sp_                                   # lower strand, i at the middle
    xb = np.arange(3, -4, -1) * sp_ + 0.12                        # upper strand, running back after a turn
    P = np.vstack([np.column_stack([xa, np.zeros_like(xa)]),
                   np.array([[xa[-1] + 0.3, 0.29]]),
                   np.column_stack([xb, np.full_like(xb, 0.62)])])
    ii = 3                                                        # residue i
    d = np.hypot(*(P - P[ii]).T)
    idx = np.arange(len(P))
    counted = (d < 0.8) & (np.abs(idx - ii) >= 3)
    near = (np.abs(idx - ii) < 3) & (idx != ii)
    ax3.plot(*P.T, color='0.55', lw=1.0, zorder=1)
    for j in np.flatnonzero(counted):
        ax3.plot(*np.array([P[ii], P[j]]).T, color='0.1', lw=0.8, ls=(0, (2.2, 1.2)), zorder=2)
    ax3.add_patch(plt.Circle(P[ii], 0.8, fc='none', ec='0.35', lw=0.7, ls=(0, (3, 2)), zorder=1))
    for j in idx:
        fc = '#202020' if j == ii else ('#8A8A8A' if counted[j] else ('#ECECEC' if near[j] else '#C8C8C8'))
        ax3.add_patch(plt.Circle(P[j], 0.13, fc=fc, ec='0.3', lw=0.4, zorder=3))
    ax3.text(*(P[ii] + [0, -0.3]), 'i', ha='center', va='top', fontsize=6, fontweight='bold', color='0.1')
    ax3.annotate('', xy=P[ii] + [0.8 * np.cos(-0.7), 0.8 * np.sin(-0.7)], xytext=P[ii], zorder=4,
                 arrowprops=dict(arrowstyle='-', color='0.35', lw=0.6))
    ax3.text(P[ii][0] + 0.62, P[ii][1] - 0.62, '0.8 nm', fontsize=5.0, color='0.3', ha='left', va='top')
    ax3.set_xlim(-1.45, 1.75); ax3.set_ylim(-1.15, 1.1)
    n_c = int(counted.sum())
    fig.text(0.469, Y0 + 0.045, f'{n_c} contacts here; average\nover conformations', ha='center', va='center', fontsize=5.6,
             color='0.3', linespacing=1.25)
    # 4 degree profile, simulated as an area shaded by degree, predicted dashed
    ax4 = fig.add_axes([0.585, 0.40, 0.155, 0.36])
    pos = np.arange(1, len(sim) + 1)
    ax4.set_facecolor('none')
    top = 1.08 * max(rel.max(), (pred / pred.mean()).max())
    area = ax4.fill_between(pos, 0, rel, color='none', lw=0)
    grad = ax4.imshow(shade(np.linspace(0, top, 256))[:, None, :3], extent=[1, len(sim), 0, top], origin='lower', aspect='auto',
                      interpolation='bilinear', zorder=1)     # shade by height, same scale as the beads and tiles
    grad.set_clip_path(area.get_paths()[0], transform=ax4.transData)
    ax4.plot(pos, pred / pred.mean(), color=C['pred'], lw=0.8, ls=(0, (3, 1.4)), zorder=3)
    ax4.set_xlim(1, len(sim)); ax4.set_ylim(0, top)
    ax4.set_xticks([]); ax4.set_yticks([])
    ax4.set_xlabel('Residue', fontsize=6.0, labelpad=2, color='0.3')
    for sp in ('top', 'right'):
        ax4.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax4.spines[sp].set_color('0.4')
    ax4.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc='#7A7A7A', ec='none'), plt.Line2D([], [], color='k', lw=0.8, ls=(0, (3, 1.4)))],
               labels=['Simulated', 'Predicted'], loc='upper right', bbox_to_anchor=(1.04, 1.14), fontsize=5.3, frameon=False,
               handlelength=1.5, handleheight=0.7, borderaxespad=0, labelspacing=0.3)
    # 5 condensate
    ax5 = fig.add_axes([0.808, 0.375, 0.181, 0.44])
    ax5.imshow(clear_bg(mpimg.imread(os.path.join(render, 'zoom.png'))), interpolation='lanczos')
    ax5.axis('off')

    def arrow(x0, y0, x1, y1, rad=0.0, ls='-', color='0.25', lw=1.3, ms=9):
        fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=ms,
                                           connectionstyle=f'arc3,rad={rad}', lw=lw, ls=ls, color=color, zorder=4,
                                           capstyle='round', joinstyle='round'))
    yA = 0.6
    arrow(0.162, yA, 0.186, yA)
    arrow(0.369, yA, 0.394, yA)
    arrow(0.545, yA, 0.570, yA)
    arrow(0.753, yA, 0.801, yA, color=C['slab'], lw=2.0, ms=11)
    fig.text(0.7765, yA + 0.04, 'Same\nresidues?', ha='center', va='bottom', fontsize=5.5, color=blue, fontweight='bold')
    # shortcut: degree model from sequence to profile
    arrow(0.085, 0.29, 0.66, 0.29, rad=0.2, ls=(0, (4, 2)), color='0.15', lw=1.2)
    fig.text(0.372, 0.145, 'Degree model, predicted from sequence alone', ha='center', va='center', fontsize=6.3, color='0.1', zorder=6,
             bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.6', fc='white', ec='0.15', lw=0.8))
    # mean-field expectation
    fig.text(0.8985, 0.13, 'Expected when pair stickiness\nis the average of two residues,\n'
             r'$\lambda_{ij} = (\lambda_i + \lambda_j)/2$, and partners' '\nare copies of the chain',
             ha='center', va='center', fontsize=5.9, color=C['slab_line'], linespacing=1.35,
             bbox=dict(boxstyle='round,pad=0.45,rounding_size=0.5', fc='#E9F2F9', ec='none'))
    fig.savefig(os.path.join(FIGS, 'figure_concept.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def figure_toc(plt, tab):
    """Table-of-contents graphic for ACS (3.25 x 1.75 in), drawn from scratch so that it shares nothing with the paper figures. Left:
    one chain folded into a hairpin whose two sticky stretches (amber) meet, with their contacts dashed. Right: a droplet of copies of
    the same chain, opened up as chains expand in the dense phase, in which the same amber stretches meet those of neighboring chains.
    Arial, upright, minimal text."""
    from matplotlib.patches import Circle, FancyArrowPatch
    from matplotlib.collections import LineCollection
    rng = np.random.default_rng(3)
    amber, halo, gray, edge, dark = '#F0A020', '#F7C66B', '#CBD1D9', '#7D8794', '#B8740A'
    sp_ = 0.135
    patchA, patchB = list(range(4, 8)), list(range(18, 22))           # the two sticky stretches
    hot = np.zeros(30, bool); hot[patchA + patchB] = True

    def hairpin():
        top = np.column_stack([np.linspace(-0.74, 0.74, 12), 0.17 + 0.025 * np.sin(np.linspace(0, 3 * np.pi, 12))])
        turn = np.column_stack([0.74 + 0.15 * np.sin([np.pi / 3, 2 * np.pi / 3]), 0.17 * np.cos([np.pi / 3, 2 * np.pi / 3])])
        bot = np.column_stack([np.linspace(0.74, -0.74, 12), -0.17 + 0.025 * np.sin(np.linspace(0, 3 * np.pi, 12) + 1)])
        tail = np.column_stack([-0.74 - 0.1 * np.sin(np.linspace(0.4, 2.0, 4)), -0.17 - np.linspace(0.13, 0.5, 4)])
        return np.vstack([top, turn, bot, tail])

    def opened():
        x = (np.arange(30) - 14.5) * sp_ * 0.92
        return np.column_stack([x, 0.05 * np.sin(np.arange(30) * 0.55)])

    def draw(ax, P, r, lw, contacts=(), clip=None):
        arts = ax.plot(*P.T, color=edge, lw=lw, zorder=2, solid_capstyle='round', solid_joinstyle='round')
        if len(contacts):
            arts.append(ax.add_collection(LineCollection(contacts, colors=dark, lw=lw * 0.85, zorder=3, linestyles=(0, (1.5, 1.1)))))
        for k, q in enumerate(P):
            if hot[k]:
                for f, al in ((2.6, 0.18), (1.9, 0.35)):           # soft glow in two layers
                    arts.append(ax.add_patch(Circle(q, r * f, fc=halo, ec='none', alpha=al, zorder=3)))
            body = amber if hot[k] else gray
            rim = dark if hot[k] else edge
            arts.append(ax.add_patch(Circle(q, r, fc=body, ec=rim, lw=0.35, zorder=4)))
            arts.append(ax.add_patch(Circle(q + np.array([0.18, 0.2]) * r * 0, r * 0.78, fc='none', ec=rim, lw=r * 9,
                                            alpha=0.18, zorder=4)))  # darker inner rim
            arts.append(ax.add_patch(Circle(q + np.array([-0.32, 0.34]) * r, r * 0.38, fc='white', ec='none',
                                            alpha=0.55 if hot[k] else 0.7, zorder=5)))   # specular highlight
        if clip is not None:
            for a in arts:
                a.set_clip_path(clip)

    fig = plt.figure(figsize=(3.25, 1.75))
    # left: one chain, the two sticky stretches meeting across the fold
    H = hairpin()
    axL = fig.add_axes([0.01, 0.06, 0.42, 0.88]); axL.set_aspect('equal'); axL.axis('off')
    pairs = [(a, patchB[int(np.argmin([np.hypot(*(H[a] - H[b])) for b in patchB]))]) for a in patchA]   # facing partner only
    for q in H:                                                  # soft shadow under the chain
        axL.add_patch(Circle(q + np.array([0.035, -0.06]), 0.066, fc='#5A6270', ec='none', alpha=0.10, zorder=1))
    draw(axL, H, 0.06, 1.0, [H[[a, b]] for a, b in pairs])
    axL.set_xlim(-1.0, 1.05); axL.set_ylim(-0.85, 0.75)
    # right: a droplet of opened copies stacked loosely so that sticky stretches of neighbors line up
    axR = fig.add_axes([0.565, 0.04, 0.43, 0.92]); axR.set_aspect('equal'); axR.axis('off')
    R = 1.0
    yy, xx = np.mgrid[-R:R:300j, -R:R:300j]
    shade_ = np.clip(0.35 + 0.65 * np.hypot(xx + 0.3, yy - 0.35) / 1.7, 0, 1)
    img = axR.imshow(shade_, extent=[-R, R, -R, R], origin='lower', zorder=0,
                     cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list('drop', ['#E4F0FA', '#A9CBE7']))
    img.set_clip_path(Circle((0, 0), R, transform=axR.transData))
    axR.add_patch(Circle((0, 0), R, fc='none', ec='#5E93C0', lw=1.0, zorder=1))
    for f, al in ((1.035, 0.25), (1.07, 0.12)):                  # faint outer halo
        axR.add_patch(Circle((0, 0), R * f, fc='none', ec='#9CC4E4', lw=1.6, alpha=al, zorder=0))
    from matplotlib.patches import Ellipse
    axR.add_patch(Ellipse((0.05, -1.075), 1.3, 0.07, fc='#4A5563', ec='none', alpha=0.13, zorder=-1))   # shadow below
    gloss = axR.add_patch(Ellipse((-0.42, 0.55), 0.62, 0.30, angle=35, fc='white', ec='none', alpha=0.28, zorder=6))
    gloss.set_clip_path(Circle((0, 0), R, transform=axR.transData))
    O = opened() * 0.44
    copies = []
    for y in (-0.58, -0.29, 0.0, 0.29, 0.58):
        th = rng.uniform(-0.22, 0.22)
        Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        copies.append(O @ Rm.T + np.array([rng.uniform(-0.05, 0.05), y + rng.uniform(-0.03, 0.03)]))
    links = []
    for a in range(len(copies) - 1):
        for k in patchA + patchB:
            for m in patchA + patchB:
                if k == m and np.hypot(*(copies[a][k] - copies[a + 1][m])) < 0.34:
                    links.append([copies[a][k], copies[a + 1][m]])
    disk = Circle((0, 0), R * 0.985, transform=axR.transData)
    for P in copies:
        draw(axR, P, 0.03, 0.6, clip=disk)
    axR.add_collection(LineCollection(links, colors=dark, lw=0.7, zorder=5, linestyles=(0, (1.3, 1.0)))).set_clip_path(disk)
    axR.set_xlim(-1.1, 1.1); axR.set_ylim(-1.1, 1.1)
    fig.patches.append(FancyArrowPatch((0.445, 0.50), (0.565, 0.50), transform=fig.transFigure,
                                       arrowstyle='simple,head_length=0.8,tail_width=0.22,head_width=0.6', mutation_scale=12,
                                       lw=0, color='#3A3F47', zorder=6))
    # no text: the graphic is meant to read without words (user, 2026-11-19)
    fig.savefig(os.path.join(FIGS, 'toc.pdf'), dpi=600)
    fig.savefig(os.path.join(FIGS, 'toc.tif'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    return len(pairs), len(links)


def figure1(plt, tab):
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    a2 = slab_rows()
    fig = plt.figure(figsize=(7.0, 7.5))
    gs = GridSpec(4, 1, height_ratios=[0.5, 1.0, 0.95, 1.15], hspace=0.6)
    a = fig.add_subplot(gs[0])
    row2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[1, 2.05], wspace=0.08)
    b = fig.add_subplot(row2[0])
    sqa = fig.add_subplot(row2[1])
    row3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.32)
    c = fig.add_subplot(row3[0])
    d = fig.add_subplot(row3[1], sharex=c)
    example_panels(fig, a, b, c, d, tab)
    sequence_row(fig, sqa, per_row=51)
    bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], width_ratios=[1, 2.1], wspace=0.32)
    e = fig.add_subplot(bot[0])
    f = fig.add_subplot(bot[1])
    h1_violins(e, a2)
    h1_ranked(f, a2, tab)
    for ax, l, x in ((a, 'a', -0.04), (b, 'b', -0.05), (c, 'c', -0.2), (d, 'd', -0.2), (e, 'e', -0.34), (f, 'f', -0.15)):
        ax.text(x, 1.04, l, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')
    fig.savefig(os.path.join(FIGS, 'figure1.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def p_phase_profiles(ax):
    """Within-chain contacts along the sequence for dense and dilute chains of the example slab (as 1D residue projections)."""
    from scipy.stats import spearmanr
    d = np.load(os.path.join(CACHE, 'fig2_rg_vs_z.npz'))
    Wd, Wl = d['Wd'], d['Wl']
    pos = np.arange(1, len(Wd) + 1)
    ax.plot(pos, Wd, color=C['slab_line'], lw=1.1, label='Dense phase')
    ax.plot(pos, Wl, color='0.35', lw=1.1, ls='--', label='Dilute phase')
    ax.set_xlim(1, len(Wd))
    lo, hi = min(Wd.min(), Wl.min()), max(Wd.max(), Wl.max())
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.42 * (hi - lo))      # headroom so the legend and label sit above the peaks
    ax.set_xlabel('GNRHR2 residue')
    ax.set_ylabel('Contacts within\nthe chain')
    ax.legend(frameon=False, loc='upper right', fontsize=7, borderaxespad=0.3, handlelength=2.2)
    ax.text(0.03, 0.95, f'Spearman {spearmanr(Wd, Wl).correlation:.2f}', transform=ax.transAxes, va='top', fontsize=7)


def p_rg_vs_z(ax, fig):
    """Radius of gyration against the distance of the chain centre from the slab centre (Farag 2022 Fig. 4d, Wang 2024 Fig. 4c),
    with the dense half-width shaded. Bins with fewer than 100 chain-frames are left out, and the region beyond 50 nm is one point."""
    d = np.load(os.path.join(CACHE, 'fig2_rg_vs_z.npz'))
    dz, rg, hw = d['dz'], d['rg'], float(d['half_width'])
    edges = np.arange(0, 34, 2.0)                               # centre and interface, densely sampled
    mid, mean, sem = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dz >= a) & (dz < b)
        if m.sum() >= 100:
            mid.append((a + b) / 2)
            mean.append(rg[m].mean())
            sem.append(rg[m].std() / np.sqrt(m.sum()))
    mid, mean, sem = map(np.array, (mid, mean, sem))
    ax.axvspan(0, hw, color='#E3EEF7', lw=0, zorder=0)
    ax.fill_between(mid, mean - 2 * sem, mean + 2 * sem, color=C['slab'], alpha=0.25, lw=0)
    ax.plot(mid, mean, color=C['slab_line'], lw=1.2, marker='o', ms=3)
    far = dz > 50                                               # the dilute region of slab_test, shown as one point, not joined
    xf = float(np.mean(dz[far]))
    ax.errorbar([xf], [rg[far].mean()], yerr=[2 * rg[far].std() / np.sqrt(far.sum())], fmt='s', color='0.35', ms=4, capsize=2, lw=1)
    ax.annotate('Beyond 50 nm', (xf, rg[far].mean()), xytext=(4, 8), textcoords='offset points', ha='right', fontsize=7, color='0.35')
    ax.text(hw / 2, 0.06, 'Dense\nphase', transform=ax.get_xaxis_transform(), ha='center', fontsize=7, color=C['slab_line'])
    ax.set_xlabel('Distance of chain center from slab center (nm)')
    ax.set_ylabel('Radius of gyration (nm)')


def figure2(plt, tab):
    a2 = slab_rows()
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 7.6), gridspec_kw={'wspace': 0.38, 'hspace': 0.55})
    for ax, fn, l in ((axs[0, 0], p_h2, 'a'), (axs[0, 1], p_h4, 'b'), (axs[1, 0], p_h5, 'c'), (axs[1, 1], p_h3, 'd')):
        fn(ax, a2)
        letter(ax, l)
    p_phase_profiles(axs[2, 0])
    p_rg_vs_z(axs[2, 1], fig)
    letter(axs[2, 0], 'e')
    letter(axs[2, 1], 'f')
    fig.savefig(os.path.join(FIGS, 'figure2.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def render_pair(fig, axes, files, titles, px_per_nm, cbar_label, cbar_rect, profiles=None):
    """Two ChimeraX renders saved at one pixel size, cropped with one common window so that they share a scale, each with a 2 nm
    bar, and one colorbar for the relative-contact palette of the render scripts."""
    import matplotlib.image as mpimg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    render = os.path.join(FIGS, 'render')
    imgs = [mpimg.imread(os.path.join(render, f)) for f in files]
    ink = [np.where((im[..., :3] < 0.97).any(axis=2)) for im in imgs]
    y0 = max(0, min(i[0].min() for i in ink) - 10)
    y1 = max(i[0].max() for i in ink) + 10
    x0 = max(0, min(i[1].min() for i in ink) - 10)
    x1 = max(i[1].max() for i in ink) + 10
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im[y0:y1, x0:x1], interpolation='lanczos')
        ax.axis('off')
        ax.set_title(title, loc='left', fontsize=8)
        h = y1 - y0
        ax.plot([20, 20 + 2 * px_per_nm], [h - 25, h - 25], color='0.3', lw=1.2)
        ax.text(20 + px_per_nm, h - 40, '2 nm', ha='center', va='bottom', fontsize=7, color='0.3')
    cmap = LinearSegmentedColormap.from_list('rel', ['#D6E8F5', '#0077BB', '#00243A'])   # palette of the render scripts
    if profiles is not None:                                      # the same values in sequence order, as a 1D residue strip
        for ax, prof in zip(axes, profiles):
            st = ax.inset_axes([0.05, -0.1, 0.9, 0.06])
            st.imshow(np.clip(np.asarray(prof)[None, :], 0.5, 1.5), cmap=cmap, vmin=0.5, vmax=1.5, aspect='auto',
                      extent=(1, len(prof), 0, 1))
            st.set_yticks([])
            st.tick_params(labelsize=6, length=2)
            st.set_xlabel('Residue', fontsize=7, labelpad=1)
    cax = fig.add_axes(cbar_rect)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0.5, 1.5), cmap=cmap), cax=cax, orientation='horizontal')
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


def sat_pair(fig, ax_t, ax_k, tab):
    render_pair(fig, (ax_t, ax_k), ('sat_typical.png', 'sat_krtap.png'),
                ('Typical condensate, GNRHR2', 'Saturated condensate, KRTAP6-2'), 20 / (130 / 900),
                'Contacts with other chains, relative to the chain mean', [0.33, 0.04, 0.36, 0.011],
                profiles=[rel_x('Q96P88_140_292'), rel_x('Q3LI66_1_62')])


def rel_x(name):
    d = np.load(os.path.join(CACHE, f'sat_{name}.npz'))
    return np.asarray(d['X']) / np.mean(d['X'])


def figure3(plt, tab):
    ff = rows('ff_specific/per_slab.csv')
    sat = rows('saturation/per_slab.csv')
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 7.6), gridspec_kw={'wspace': 0.38, 'hspace': 0.62, 'height_ratios': [1, 1, 1.1]})
    p_ff(axs[0, 0], ff)
    p_gap(axs[0, 1], ff, tab)
    p_sat(axs[1, 0], sat, 'T_mean', 'Total dense-phase\ncontacts per residue', legend=True)
    p_sat(axs[1, 1], sat, 'T_cv', 'Variation of total contacts\nacross residues (%)')
    sat_pair(fig, axs[2, 0], axs[2, 1], tab)
    for ax, l in zip(axs.flat, 'abcdef'):
        letter(ax, l)
    axs[2, 0].texts[-1].set_x(-0.07)
    axs[2, 1].texts[-1].set_x(-0.07)
    fig.savefig(os.path.join(FIGS, 'figure3.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def figure4(plt, tab):
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(7.0, 8.2))
    gs = GridSpec(3, 1, height_ratios=[1.25, 1, 1.35], hspace=0.55)
    top = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.12)
    a = fig.add_subplot(top[0])
    b = fig.add_subplot(top[1], sharex=a)
    mid = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.38)
    c = fig.add_subplot(mid[0])
    d = fig.add_subplot(mid[1])
    bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.1)
    e = fig.add_subplot(bot[0])
    f = fig.add_subplot(bot[1])
    p_caprin(a, b, c)
    p_testA(d, tab)
    render_pair(fig, (e, f), ('matched_uneven.png', 'matched_even.png'),
                ('Uneven condensate, Q7Z4F1', 'Even condensate, P55316'), 20 / 0.2611,
                'Contacts with other chains, relative to the chain mean', [0.33, 0.035, 0.36, 0.011],
                profiles=[rel_x('Q7Z4F1_487_713'), rel_x('P55316_274_489')])
    for ax, l, x in ((a, 'a', -0.12), (b, 'b', -0.12), (c, 'c', -0.2), (d, 'd', -0.2), (e, 'e', -0.1), (f, 'f', -0.1)):
        ax.text(x, 1.04, l, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')
    fig.savefig(os.path.join(FIGS, 'figure4.pdf'), bbox_inches='tight', dpi=300)
    plt.close(fig)


def main():
    if '--prep' in sys.argv:
        return prep()
    os.makedirs(FIGS, exist_ok=True)
    plt = setup()
    tab = table()
    for f in (figure_concept, figure1, figure2, figure3, figure4):
        f(plt, tab)
        print('done', f.__name__)


if __name__ == '__main__':
    main()
