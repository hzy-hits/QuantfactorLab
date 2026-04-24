#!/bin/bash
# Daily factor lifecycle: mine → backtest → promote → health check
# Run before morning pipeline (06:00 CST)
#
# Cron:
#   0 6 * * 1-5 cd $FACTOR_LAB_ROOT && bash scripts/daily_factors.sh >> logs/daily.log 2>&1

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p logs data
PYTHON="${PYTHON_BIN:-python3}"
export FACTOR_LAB_AGENT_BACKEND="${FACTOR_LAB_AGENT_BACKEND:-codex}"
export FACTOR_LAB_CODEX_MODEL="${FACTOR_LAB_CODEX_MODEL:-gpt-5.4}"
export FACTOR_LAB_CODEX_REASONING_EFFORT="${FACTOR_LAB_CODEX_REASONING_EFFORT:-xhigh}"
CN_EXPECTED_DATE="${FACTOR_LAB_CN_EXPECTED_DATE:-$(TZ=Asia/Shanghai date +%Y-%m-%d)}"
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
            echo "Usage: bash scripts/daily_factors.sh [--market cn|us|all]"
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
echo "  Daily Factor Pipeline"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo "  Agent backend: $FACTOR_LAB_AGENT_BACKEND"
echo "  Codex model:   $FACTOR_LAB_CODEX_MODEL"
echo "  Reasoning:     $FACTOR_LAB_CODEX_REASONING_EFFORT"
echo "  Market scope:  $MARKET"
echo "  CN as-of:      $CN_EXPECTED_DATE"
echo "  US as-of:      $US_EXPECTED_DATE"

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

# CN factors
if [[ "$RUN_CN" == true ]]; then
    echo ""
    echo "=== A-Share Factors ==="
    $PYTHON -m src.mining.daily_pipeline --market cn --max-factors 500 --date "$CN_EXPECTED_DATE"
fi

# US factors
if [[ "$RUN_US" == true && "$US_READY" == true ]]; then
    echo ""
    echo "=== US Factors ==="
    $PYTHON -m src.mining.daily_pipeline --market us --max-factors 500 --date "$US_EXPECTED_DATE"
fi

# SigReg diagnostics
if [[ "$RUN_CN" == true ]]; then
    echo ""
    echo "=== CN SigReg Diagnostics ==="
    $PYTHON scripts/sigreg_report.py --market cn || echo "CN SigReg report failed (non-fatal)"
fi

if [[ "$RUN_US" == true && "$US_READY" == true ]]; then
    echo ""
    echo "=== US SigReg Diagnostics ==="
    $PYTHON scripts/sigreg_report.py --market us || echo "US SigReg report failed (non-fatal)"
fi

# ═══════════════════════════════════════════════════════
# Research candidates (subordinate to pipeline reports/execution gates)
# ═══════════════════════════════════════════════════════
if [[ "$RUN_CN" == true ]]; then
    echo ""
    echo "=== A-Share Factor Lab Research Candidates ==="
    $PYTHON scripts/run_strategy.py --market cn --today --date "$CN_EXPECTED_DATE" || echo "CN failed"
fi

if [[ "$RUN_US" == true && "$US_READY" == true ]]; then
    echo ""
    echo "=== US Factor Lab Research Candidates ==="
    $PYTHON scripts/run_strategy.py --market us --today --date "$US_EXPECTED_DATE" || echo "US failed"
fi

echo ""
echo "=========================================="
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
