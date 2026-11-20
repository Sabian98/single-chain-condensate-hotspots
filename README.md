# Single-chain contact networks identify the interaction hotspots of protein condensates

Code for the paper "Single-Chain Contact Networks Identify the Interaction Hotspots of Protein Condensates" (T. Rahman, submitted to
*The Journal of Physical Chemistry B*). The paper asks whether the contacts that a residue makes within one isolated chain tell where it
makes contacts with other chains in a condensate, and tests this residue by residue in slab simulations of 185 human and 21 plant
condensates, in condensates simulated with a second force field, and against intermolecular NMR contacts in a CAPRIN1 condensate.

## Layout

| Folder | Contents |
|---|---|
| `analysis/` | Analysis of the slab simulations, from reading frames to every per-slab result |
| `simulations/mpipi_recharged/` | Setup, run and submit scripts for the Mpipi-Recharged slabs, with the 12 sequences |
| `paper_tools/` | Scripts that turn the results into the numbers, figures, Supporting Information and Word version of the paper |

### analysis

| Script | What it does |
|---|---|
| `tar_list_remote.py`, `tar_fetch_remote.py`, `slab_fetch_frames.py` | Read 200 frames from the second half of each published slab trajectory by HTTP byte range, without downloading whole archives |
| `slab_pilot_loricrin.py`, `slab_pilot_dilute.py` | Pilot analysis of one slab, used to fix the definitions before the full set was analyzed |
| `slab_test.py` | The main analysis, which computes contacts, dense and dilute phases, contact degree, the local stickiness baseline and the agreement for every slab |
| `single_chain_test.py` | Agreement of single-chain degree taken from the IDRome single-chain simulations |
| `explore_ff_specific_degree.py` | Agreement of degree models trained on one force field at a time |
| `explore_saturation.py`, `explore_reliability.py` | Total contacts per residue and the split-half noise ceilings |
| `amendment4_test.py`, `explore_degree_descriptors.py` | Unevenness of predicted degree at matched stability, and the comparison with the data of An et al. |
| `recharged_slab_test.py`, `recharged_ladder_test.py` | The same analysis applied to the Mpipi-Recharged slabs at 293 K and at lower temperature |

## Data

The analysis reads data that are cited rather than redistributed here.

- CALVADOS-2 slab simulations of von Bülow et al., Proc. Natl. Acad. Sci. USA 122, e2417920122 (2025), from their public ERDA archive.
- Single-chain CALVADOS-2 simulations of the human IDRome, Tesei et al., Nature 626, 897 (2024).
- Intermolecular NOEs of the CAPRIN1 condensate, Kim et al., Proc. Natl. Acad. Sci. USA 118, e2104897118 (2021), digitized from the
  published figure.
- Condensate dynamics of designed sequences, An et al., Sci. Adv. 10, eadj2448 (2024).

Single-chain degree is predicted with the degree model of Rahman and Uversky, bioRxiv (2026), doi 10.64898/2026.09.13.751321, whose
code is in [beyond-stickers-spacers](https://github.com/Sabian98/beyond-stickers-spacers). The Mpipi-Recharged trajectories and the
per-slab results will be deposited with a DOI on publication. Per-slab results for all 209 slabs are also in Table S1 of the Supporting
Information.

## Requirements

Python 3.10 or later with numpy, scipy, mdtraj and matplotlib. The Mpipi-Recharged simulations need OpenMM and OpenMpipi on a GPU. The
paper tools also use tectonic for LaTeX, pandoc for the Word version and ChimeraX for the structure renders.

## Paths

The scripts were written for one project folder and set its location in a constant near the top of each file (`ROOT`, `RES` or
`PAPER`). Point these to your own copies of the data and results before running.

## License

MIT, see `LICENSE`.
