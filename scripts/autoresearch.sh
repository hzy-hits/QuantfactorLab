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
PYTHON="${PYTHON_BIN:-python3}"
export FACTOR_LAB_AGENT_BACKEND="${FACTOR_LAB_AGENT_BACKEND:-codex}"
export FACTOR_LAB_CODEX_MODEL="${FACTOR_LAB_CODEX_MODEL:-gpt-5.4}"
export FACTOR_LAB_CODEX_REASONING_EFFORT="${FACTOR_LAB_CODEX_REASONING_EFFORT:-xhigh}"
US_EXPECTED_DATE="${FACTOR_LAB_US_EXPECTED_DATE:-$(TZ=America/New_York date +%Y-%m-%d)}"
MARKET="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --market)
            MARKET="${2:-}"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option '$1'"
            echo "Usage: bash scripts/autoresearch.sh [--market cn|us|all]"
            exit 1
            ;;
    esac
done

if [[ "$MARKET" != "all" && "$MARKET" != "cn" && "$MARKET" != "us" ]]; then
    echo "ERROR: --market must be one of cn, us, all"
    exit 1
fi

RUN_CN=false
RUN_US=false
[[ "$MARKET" == "all" || "$MARKET" == "cn" ]] && RUN_CN=true
[[ "$MARKET" == "all" || "$MARKET" == "us" ]] && RUN_US=true

check_us_ready() {
    $PYTHON -m src.data_readiness --market us --expected-date "$US_EXPECTED_DATE"
}

echo "=========================================="
echo "  Autoresearch Session — $DATE $(date '+%H:%M')"
echo "=========================================="
echo "  Agent backend: $FACTOR_LAB_AGENT_BACKEND"
echo "  Codex model:   $FACTOR_LAB_CODEX_MODEL"
echo "  Reasoning:     $FACTOR_LAB_CODEX_REASONING_EFFORT"
echo "  Market scope:  $MARKET"

US_READY=false
if [[ "$RUN_US" == true ]]; then
    echo ""
    echo "=== US Data Readiness ==="
    if check_us_ready; then
        US_READY=true
    else
        echo "Skipping US jobs until quant-research-v1 has refreshed $US_EXPECTED_DATE"
        if [[ "$RUN_CN" != true ]]; then
            echo ""
            echo "=========================================="
            echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "=========================================="
            exit 0
        fi
    fi
fi

if [ "$HOUR" -ge 9 ] && [ "$HOUR" -lt 13 ]; then
    # Morning session: strategy optimization + hyperparameter search
    echo "  Mode: Strategy Optimization"
    echo ""

    # Grid search: lookback × hold × n_picks × ic_exit
    if [[ "$RUN_CN" == true ]]; then
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
    fi

    if [[ "$RUN_US" == true && "$US_READY" == true ]]; then
        echo "=== US Strategy Grid Search ==="
        for LOOKBACK in 20 40 60; do
            for HOLD in 10 20; do
                echo "--- lb=$LOOKBACK hold=$HOLD ---"
                $PYTHON scripts/run_strategy.py --market us --lookback $LOOKBACK --hold $HOLD 2>&1 | grep -E "Ann Return|Sharpe|Max DD|Excess" || true
            done
        done
        echo ""
    fi

    if [[ "$RUN_CN" == true ]]; then
        echo "=== CN SigReg Diagnostics ==="
        $PYTHON scripts/sigreg_report.py --market cn || echo "CN SigReg failed"
        echo ""
    fi

    if [[ "$RUN_US" == true && "$US_READY" == true ]]; then
        echo "=== US SigReg Diagnostics ==="
        $PYTHON scripts/sigreg_report.py --market us || echo "US SigReg failed"
    fi

else
    # Night/afternoon session: factor mining
    echo "  Mode: Factor Mining"
    BUDGET=30  # hard cap, but time budget is now the primary stop condition
    TIME_BUDGET_MINUTES=55

    if [[ "$RUN_CN" == true ]]; then
        echo ""
        echo "=== [$(date '+%H:%M:%S')] A-Share Factor Mining ==="
        timeout 3600 $PYTHON scripts/run_autoresearch_session.py \
            --market cn \
            --budget $BUDGET \
            --time-budget-minutes $TIME_BUDGET_MINUTES \
            --output "reports/autoresearch_cn_${DATE}_${HOUR}.md" \
            || echo "CN ended (exit $?)"
    fi

    if [[ "$RUN_US" == true && "$US_READY" == true ]]; then
        echo ""
        echo "=== [$(date '+%H:%M:%S')] US Factor Mining ==="
        timeout 3600 $PYTHON scripts/run_autoresearch_session.py \
            --market us \
            --budget $BUDGET \
            --time-budget-minutes $TIME_BUDGET_MINUTES \
            --output "reports/autoresearch_us_${DATE}_${HOUR}.md" \
            || echo "US ended (exit $?)"
    fi

    # Export
    echo ""
    echo "=== Exporting ==="
    if [[ "$RUN_CN" == true ]]; then
        $PYTHON -m src.mining.export_to_pipeline --market cn || echo "CN export failed"
    fi
    if [[ "$RUN_US" == true && "$US_READY" == true ]]; then
        $PYTHON -m src.mining.export_to_pipeline --market us || echo "US export failed"
    fi
fi

echo ""
echo "=========================================="
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
