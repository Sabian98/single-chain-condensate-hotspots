#!/bin/bash
# submit_recharged_slab.sh: Mpipi-Recharged slabs of Amendment 3 (slab_followup_plan.md), one GPU job per sequence.
# Activate the environment with OpenMpipi first; the job inherits it.
#   bash submit_recharged_slab.sh pilot      # O60477_1_58 only: measures speed and slab stability first
#   bash submit_recharged_slab.sh all        # the 12 sequences, in the order fixed by Amendment 3
# Rerunning resumes unfinished runs from their last segment. Output: $OUT/<name>/<name>.dcd
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-/general/taseefr/slab_recharged}"
TIME="${TIME:-2-00:00:00}"
ORDER=(O60477_1_58 Q96P88_140_292 Q9Y6C7_1_94 Q9H5F2_1_150 Q7Z5H4_162_357 Q6V1P9_1_111 Q9P2K5_306_526 P12035_1_202 Q5K131_1_121 Q14154_1_230 O00167_1_265 Q8N3T1_1_161)
case "${1:-}" in
  pilot) NAMES=(O60477_1_58) ;;
  all)   NAMES=("${ORDER[@]}") ;;
  *) echo "usage: $0 pilot|all"; exit 1 ;;
esac
mkdir -p "$OUT/logs"
for NAME in "${NAMES[@]}"; do
  if [[ -f "$OUT/$NAME/$NAME.dcd" ]]; then echo "$NAME already done"; continue; fi
  sbatch --parsable --job-name="RSLAB_$NAME" --time="$TIME" \
    --output="$OUT/logs/${NAME}_%j.out" --error="$OUT/logs/${NAME}_%j.err" <<EOF
#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
echo "$NAME on \$(hostname)"
python "$HERE/run_recharged_slab.py" --name "$NAME" --fasta "$HERE/recharged_slab_293K.fasta" --outdir "$OUT"
EOF
done
