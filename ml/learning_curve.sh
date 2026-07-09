#!/usr/bin/env bash
# Learning-curve sweep for Astra-6 BC pretraining.
#
# Trains the policy at increasing data scales and records choose_action /
# respond_to_chain fidelity at each point, so we can extrapolate how many games
# are needed to close the generalization gap *before* committing to a 30K+ run.
#
# Each run uses the new pretrain pipeline: masked label smoothing, held-out
# best-policy checkpointing, and a frozen trunk during value/candidate fine-tune.
#
# Usage:  ./learning_curve.sh [game_sizes...]
#   ./learning_curve.sh                 # default sweep 1000 2000 4000 8000 16000
#   ./learning_curve.sh 2000 8000 32000
set -euo pipefail

cd "$(dirname "$0")"

SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=(1000 2000 4000 8000 16000)
fi

FID_GAMES="${FID_GAMES:-200}"     # games for the fidelity probe
VAL_GAMES="${VAL_GAMES:-500}"     # held-out games for checkpointing
OUTDIR="learning_curve_out"
mkdir -p "$OUTDIR"

BACKEND="${BACKEND:-gpu}"         # gpu (wgpu) or cpu (ndarray, no-default-features)
echo "Building release binary (backend=$BACKEND)..."
if [ "$BACKEND" = "cpu" ]; then
    cargo build --release --no-default-features
else
    cargo build --release --features gpu
fi

BIN=../target/release/novacana-ml
SUMMARY="$OUTDIR/summary.tsv"
printf "games\tchoose_agree\tchoose_top2\trespond_agree\teval_winrate\n" > "$SUMMARY"

for g in "${SIZES[@]}"; do
    model="$OUTDIR/pretrain_${g}.bin"
    trainlog="$OUTDIR/train_${g}.log"
    fidlog="$OUTDIR/fidelity_${g}.log"
    evallog="$OUTDIR/eval_${g}.log"

    echo "=== games=$g : training ==="
    "$BIN" pretrain --games "$g" --val-games "$VAL_GAMES" --out "$model" 2>&1 | tee "$trainlog"

    echo "=== games=$g : fidelity ($FID_GAMES games) ==="
    "$BIN" fidelity --model "$model" --games "$FID_GAMES" 2>&1 | tee "$fidlog"

    echo "=== games=$g : eval (hybrid win rate) ==="
    "$BIN" eval --model "$model" --games 300 2>&1 | tee "$evallog"

    # Pull headline metrics out of the logs (best-effort grep).
    # agree=/top2= are printed with a possible leading space ("{:5.1}" format).
    ca=$(grep -oP 'agree=\s*\K[0-9.]+%'   "$fidlog" | head -1 || echo "?")
    ct=$(grep -oP 'top2=\s*\K[0-9.]+%'    "$fidlog" | head -1 || echo "?")
    # respond_to_chain's agree is the first match *after* its section header,
    # not the 2nd match overall (that's the "turns 1-5" choose_action bucket).
    ra=$(sed -n '/respond_to_chain/,$p' "$fidlog" | grep -oP 'agree=\s*\K[0-9.]+%' | head -1 || echo "?")
    wr=$(grep -oP ':\s+\K[0-9.]+%'        "$evallog" | head -1 || echo "?")
    printf "%s\t%s\t%s\t%s\t%s\n" "$g" "$ca" "$ct" "$ra" "$wr" >> "$SUMMARY"
done

echo
echo "Sweep complete. Summary:"
column -t "$SUMMARY"
