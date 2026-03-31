#!/bin/bash
# Daily factor lifecycle: mine → backtest → promote → health check
# Run before morning pipeline (06:00 CST)
#
# Cron:
#   0 6 * * 1-5 cd /home/ivena/coding/python/factor-lab && bash scripts/daily_factors.sh >> logs/daily.log 2>&1

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p logs data
export FACTOR_LAB_AGENT_BACKEND="${FACTOR_LAB_AGENT_BACKEND:-codex}"
export FACTOR_LAB_CODEX_MODEL="${FACTOR_LAB_CODEX_MODEL:-gpt-5.4}"
export FACTOR_LAB_CODEX_REASONING_EFFORT="${FACTOR_LAB_CODEX_REASONING_EFFORT:-xhigh}"

echo "=========================================="
echo "  Daily Factor Pipeline"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo "  Agent backend: $FACTOR_LAB_AGENT_BACKEND"
echo "  Codex model:   $FACTOR_LAB_CODEX_MODEL"
echo "  Reasoning:     $FACTOR_LAB_CODEX_REASONING_EFFORT"

# CN factors
echo ""
echo "=== A-Share Factors ==="
/home/ivena/miniconda3/bin/python3 -m src.mining.daily_pipeline --market cn --max-factors 500

# US factors
echo ""
echo "=== US Factors ==="
/home/ivena/miniconda3/bin/python3 -m src.mining.daily_pipeline --market us --max-factors 500

# SigReg diagnostics
echo ""
echo "=== SigReg Factor Diagnostics ==="
/home/ivena/miniconda3/bin/python3 scripts/sigreg_report.py || echo "SigReg report failed (non-fatal)"

# ═══════════════════════════════════════════════════════
# Trading Signals (independent from pipeline reports)
# ═══════════════════════════════════════════════════════
echo ""
echo "=== A-Share Trading Signal ==="
/home/ivena/miniconda3/bin/python3 scripts/run_strategy.py --market cn --today || echo "CN failed"
echo ""
echo "=== US Trading Signal ==="
/home/ivena/miniconda3/bin/python3 scripts/run_strategy.py --market us --today || echo "US failed"

echo ""
echo "=========================================="
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
