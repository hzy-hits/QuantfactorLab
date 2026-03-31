#!/bin/bash
# Autoresearch: AI-driven factor mining + strategy optimization.
#
# Three daily sessions:
#   02:00-04:00 CST — factor mining (new factors)
#   10:00-12:00 CST — strategy optimization (hyperparameters)
#   14:00-17:00 CST — factor mining (new factors)
#
# ALL results must pass shuffle test (p<0.05) to be considered valid.
# The backtest framework uses look-ahead-safe lookback windows.

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p logs reports

HOUR=$(date +%H)
DATE=$(date +%Y%m%d)
PYTHON=/home/ivena/miniconda3/bin/python3
export FACTOR_LAB_AGENT_BACKEND="${FACTOR_LAB_AGENT_BACKEND:-codex}"
export FACTOR_LAB_CODEX_MODEL="${FACTOR_LAB_CODEX_MODEL:-gpt-5.4}"
export FACTOR_LAB_CODEX_REASONING_EFFORT="${FACTOR_LAB_CODEX_REASONING_EFFORT:-xhigh}"

echo "=========================================="
echo "  Autoresearch Session — $DATE $(date '+%H:%M')"
echo "=========================================="
echo "  Agent backend: $FACTOR_LAB_AGENT_BACKEND"
echo "  Codex model:   $FACTOR_LAB_CODEX_MODEL"
echo "  Reasoning:     $FACTOR_LAB_CODEX_REASONING_EFFORT"

if [ "$HOUR" -ge 9 ] && [ "$HOUR" -lt 13 ]; then
    # Morning session: strategy optimization + hyperparameter search
    echo "  Mode: Strategy Optimization"
    echo ""

    # Grid search: lookback × hold × n_picks × ic_exit
    echo "=== CN Strategy Grid Search ==="
    for LOOKBACK in 20 40 60; do
        for HOLD in 5 10 20; do
            for NPICKS in 5 10 15; do
                echo "--- lb=$LOOKBACK hold=$HOLD n=$NPICKS ---"
                $PYTHON scripts/run_strategy.py --market cn --lookback $LOOKBACK --hold $HOLD --n-picks $NPICKS 2>&1 | grep -E "Ann Return|Sharpe|Max DD|Excess" || true
            done
        done
    done

    echo ""
    echo "=== US Strategy Grid Search ==="
    for LOOKBACK in 20 40 60; do
        for HOLD in 10 20; do
            echo "--- lb=$LOOKBACK hold=$HOLD ---"
            $PYTHON scripts/run_strategy.py --market us --lookback $LOOKBACK --hold $HOLD 2>&1 | grep -E "Ann Return|Sharpe|Max DD|Excess" || true
        done
    done

    echo ""
    echo "=== SigReg Diagnostics ==="
    $PYTHON scripts/sigreg_report.py || echo "SigReg failed"

else
    # Night/afternoon session: factor mining
    echo "  Mode: Factor Mining"
    BUDGET=30  # 30 experiments × ~30s = ~15min per market, fits in 1hr

    echo ""
    echo "=== [$(date '+%H:%M:%S')] A-Share Factor Mining ==="
    timeout 3600 $PYTHON -m src.agent.loop \
        --market cn \
        --budget $BUDGET \
        --output "reports/autoresearch_cn_${DATE}_${HOUR}.md" \
        || echo "CN ended (exit $?)"

    echo ""
    echo "=== [$(date '+%H:%M:%S')] US Factor Mining ==="
    timeout 3600 $PYTHON -m src.agent.loop \
        --market us \
        --budget $BUDGET \
        --output "reports/autoresearch_us_${DATE}_${HOUR}.md" \
        || echo "US ended (exit $?)"

    # Export
    echo ""
    echo "=== Exporting ==="
    $PYTHON -m src.mining.export_to_pipeline --market cn || echo "CN export failed"
    $PYTHON -m src.mining.export_to_pipeline --market us || echo "US export failed"
fi

echo ""
echo "=========================================="
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
