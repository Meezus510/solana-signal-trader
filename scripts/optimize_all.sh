#!/usr/bin/env bash
# scripts/optimize_all.sh — Run all three optimizers sequentially.
#
# Logs output to reports/optimize_YYYY-MM-DD_HHMMSS.log while also printing
# to the terminal in real time.
#
# Each strategy runs its ML weights, TP/SL, and filters in order — so the
# full picture for one strategy is complete before moving to the next.
#
# Usage:
#   bash scripts/optimize_all.sh                        # all strategies, default rounds
#   bash scripts/optimize_all.sh --strategy quick_pop  # quick_pop only
#   bash scripts/optimize_all.sh --strategy moonbag    # moonbag only
#   bash scripts/optimize_all.sh --strategy trend_rider
#   bash scripts/optimize_all.sh --rounds 3                         # override all rounds
#   bash scripts/optimize_all.sh --strategy quick_pop --rounds 5
#   bash scripts/optimize_all.sh --strategy trend_rider --mode lenient
#   bash scripts/optimize_all.sh --strategy trend_rider --mode strict

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
STRATEGY="all"   # all | quick_pop | trend_rider | moonbag
MODE="balanced"  # balanced | strict — applies to trend_rider ML weights optimizer
ML_ROUNDS=4
TP_ROUNDS=6
FILTER_ROUNDS=5

# ---- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --strategy)      STRATEGY="$2";      shift 2 ;;
        --mode)          MODE="$2";          shift 2 ;;
        --rounds)        ML_ROUNDS="$2"; TP_ROUNDS="$2"; FILTER_ROUNDS="$2"; shift 2 ;;
        --ml-rounds)     ML_ROUNDS="$2";     shift 2 ;;
        --tp-rounds)     TP_ROUNDS="$2";     shift 2 ;;
        --filter-rounds) FILTER_ROUNDS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate mode
case "$MODE" in
    lenient|balanced|strict) ;;
    *) echo "Unknown mode: $MODE. Use: lenient | balanced | strict"; exit 1 ;;
esac

# Build mode flag for ML weights optimizer (only used by trend_rider)
TR_ML_FLAGS=""
if [[ "$MODE" == "strict" ]]; then
    TR_ML_FLAGS="--strict"
elif [[ "$MODE" == "lenient" ]]; then
    TR_ML_FLAGS="--lenient"
fi

# Validate strategy
case "$STRATEGY" in
    all|quick_pop|trend_rider|moonbag) ;;
    *) echo "Unknown strategy: $STRATEGY. Use: all | quick_pop | trend_rider | moonbag"; exit 1 ;;
esac

# ---- Log file ---------------------------------------------------------------
TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")
LOG_DIR="reports"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/optimize_${STRATEGY}_${TIMESTAMP}.log"

# Tee everything to log + terminal
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo "  optimize_all.sh  —  $(date)"
echo "  Strategy: $STRATEGY  |  Mode: $MODE"
echo "  ML rounds: $ML_ROUNDS  |  TP rounds: $TP_ROUNDS  |  Filter rounds: $FILTER_ROUNDS"
echo "  Log: $LOG_FILE"
echo "========================================================================"

PYTHON="venv/bin/python"

run_step() {
    local label="$1"; shift
    echo ""
    echo "------------------------------------------------------------------------"
    echo "  STEP: $label"
    echo "  CMD:  $*"
    echo "  $(date)"
    echo "------------------------------------------------------------------------"
    "$@"
    echo ""
    echo "  ✓ $label — done ($(date))"
}

# Helper: returns 0 (true) if the given strategy should run
should_run() {
    [[ "$STRATEGY" == "all" || "$STRATEGY" == "$1" ]]
}

# ---- quick_pop --------------------------------------------------------------
# TP/SL exists for quick_pop; trend_rider has no TP/SL optimizer.
if should_run "quick_pop"; then
    echo ""
    echo "######## quick_pop ########"
    run_step "ML weights — quick_pop" \
        $PYTHON scripts/optimize_ml_weights.py --strategy quick_pop --rounds "$ML_ROUNDS"
    run_step "TP/SL — quick_pop" \
        $PYTHON scripts/optimize_tp_sl.py --strategy quick_pop --rounds "$TP_ROUNDS"
    run_step "Filters — quick_pop" \
        $PYTHON scripts/optimize_filters.py --strategy quick_pop --rounds "$FILTER_ROUNDS"
fi

# ---- trend_rider ------------------------------------------------------------
# No TP/SL optimizer for trend_rider.
if should_run "trend_rider"; then
    echo ""
    echo "######## trend_rider  (mode: $MODE) ########"
    run_step "ML weights — trend_rider ($MODE)" \
        $PYTHON scripts/optimize_ml_weights.py --strategy trend_rider --rounds "$ML_ROUNDS" $TR_ML_FLAGS
    run_step "Filters — trend_rider ($MODE)" \
        $PYTHON scripts/optimize_filters.py --strategy trend_rider --rounds "$FILTER_ROUNDS" --mode "$MODE"
fi

# ---- moonbag ----------------------------------------------------------------
if should_run "moonbag"; then
    echo ""
    echo "######## moonbag ########"
    run_step "ML weights — moonbag" \
        $PYTHON scripts/optimize_ml_weights.py --strategy moonbag --rounds "$ML_ROUNDS"
    run_step "TP/SL — moonbag" \
        $PYTHON scripts/optimize_tp_sl.py --strategy moonbag --rounds "$TP_ROUNDS"
    run_step "Filters — moonbag" \
        $PYTHON scripts/optimize_filters.py --strategy moonbag --rounds "$FILTER_ROUNDS"
fi

# ---- Done -------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "  All optimizers complete — $(date)"
echo "  Strategy: $STRATEGY"
echo "  Full log saved to: $LOG_FILE"
echo "========================================================================"
