#!/bin/bash
# submit_recharged_slab_lowT.sh: Amendment 3b temperature ladder (280, 270, 260, 250 K) for the 8 sequences that did not phase separate
# in Mpipi-Recharged at 293 K. Same runner and settings as Amendment 3/3a, only --temp changes; each rung writes to its own folder.
# Activate the environment with OpenMpipi first; the job inherits it.
#   bash submit_recharged_slab_lowT.sh 280                    # all 8 at 280 K
#   bash submit_recharged_slab_lowT.sh 270 Q9P2K5_306_526 ... # only the sequences that did not phase separate at the rung above
#   bash submit_recharged_slab_lowT.sh all                    # every rung at once (the analysis still takes the warmest that separates)
# Rerunning resumes unfinished runs from their last segment. Output: $OUT_BASE/T<K>/<name>/<name>.dcd
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT_BASE="${OUT_BASE:-/general/taseefr/slab_recharged_lowT}"
TIME="${TIME:-2-00:00:00}"
EIGHT=(Q9P2K5_306_526 P12035_1_202 Q14154_1_230 Q96P88_140_292 O00167_1_265 Q9Y6C7_1_94 Q6V1P9_1_111 Q5K131_1_121)
case "${1:-}" in
  280|270|260|250) RUNGS=("$1"); shift ;;
  all) RUNGS=(280 270 260 250); shift ;;
  *) echo "usage: $0 280|270|260|250|all [names...]"; exit 1 ;;
esac
if [[ $# -gt 0 ]]; then NAMES=("$@"); else NAMES=("${EIGHT[@]}"); fi
for T in "${RUNGS[@]}"; do
  OUT="$OUT_BASE/T$T"
  mkdir -p "$OUT/logs"
  for NAME in "${NAMES[@]}"; do
    if [[ -f "$OUT/$NAME/$NAME.dcd" ]]; then echo "$NAME at $T K already done"; continue; fi
    sbatch --parsable --job-name="RSLAB_${T}_$NAME" --time="$TIME" \
      --output="$OUT/logs/${NAME}_%j.out" --error="$OUT/logs/${NAME}_%j.err" <<EOS
#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
echo "$NAME at $T K on \$(hostname)"
python "$HERE/run_recharged_slab.py" --name "$NAME" --fasta "$HERE/recharged_slab_293K.fasta" --outdir "$OUT" --temp $T
EOS
  done
done
