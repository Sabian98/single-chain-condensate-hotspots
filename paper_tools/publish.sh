#!/bin/bash
# Build the manuscript PDF, Word version and Supporting Information, and put the deliverables in the NAS manuscript folder
# (never in Downloads). Run from anywhere:  bash tools/publish.sh
set -euo pipefail
cd "$(dirname "$0")/.."
NAS="/Volumes/User Homes/Nu Project/DYNASEP/hub_condensate_followup/manuscript"
PY=~/.venvs/finches/bin/python
mkdir -p "$NAS"
$PY tools/make_si.py >/dev/null 2>&1
tectonic main.tex >/dev/null 2>&1
tectonic si.tex >/dev/null 2>&1
$PY tools/make_docx.py "$NAS/JPCB_manuscript.docx" >/dev/null
cp -X main.pdf "$NAS/JPCB_manuscript.pdf"
cp -X si.pdf "$NAS/JPCB_SI.pdf"
cp -X SI/Table_S1.xlsx "$NAS/JPCB_Table_S1.xlsx"
ls -la "$NAS"
# cover letter (Word, then PDF through LibreOffice) and TOC graphic
~/miniforge3/envs/docs/bin/pandoc cover_letter.md -o "$NAS/JPCB_cover_letter.docx" --reference-doc tools/reference.docx
TMPD=$(mktemp -d); cp "$NAS/JPCB_cover_letter.docx" "$TMPD/cl.docx"
(cd "$TMPD" && /Applications/LibreOffice.app/Contents/MacOS/soffice --headless --convert-to pdf cl.docx >/dev/null 2>&1)
cp -X "$TMPD/cl.pdf" "$NAS/JPCB_cover_letter.pdf"
$PY -c "import sys; sys.path.insert(0,'tools'); sys.argv=['x']; import make_figures as M; plt=M.setup(); M.figure_toc(plt, M.table())" 2>/dev/null
cp -X figs/toc.pdf "$NAS/JPCB_TOC.pdf"; cp -X figs/toc.tif "$NAS/JPCB_TOC.tif"
ls -la "$NAS" | grep -v ' \._'
