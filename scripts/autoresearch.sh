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

echo "=========================================="
echo "  Autoresearch Session — $DATE $(date '+%H:%M')"
echo "=========================================="

if [ "$HOUR" -ge 9 ] && [ "$HOUR" -lt 13 ]; then
    # Morning session: strategy optimization + hyperparameter search
    echo "  Mode: Strategy Optimization"
    echo ""

    # Run strategy backtest grid search with different params
    echo "=== Strategy Grid Search ==="
    for LOOKBACK in 20 40 60; do
        for HOLD in 5 10 20; do
            echo "--- lookback=$LOOKBACK hold=$HOLD ---"
            $PYTHON scripts/run_strategy.py --market cn --lookback $LOOKBACK --hold $HOLD 2>&1 | grep -E "Ann Return|Sharpe|Max DD|Excess" || true
        done
    done

    echo ""
    echo "=== SigReg Diagnostics ==="
    $PYTHON scripts/sigreg_report.py || echo "SigReg failed"

else
    # Night/afternoon session: factor mining
    echo "  Mode: Factor Mining"
    BUDGET=50

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
