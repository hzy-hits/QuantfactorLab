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

echo "=========================================="
echo "  Daily Factor Pipeline"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

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

# Today's rolling best-factor picks
echo ""
echo "=== Today's Rolling Best-Factor Picks ==="
/home/ivena/miniconda3/bin/python3 scripts/run_strategy.py --market cn --today || echo "CN picks failed (non-fatal)"
echo ""
/home/ivena/miniconda3/bin/python3 scripts/run_strategy.py --market us --today || echo "US picks failed (non-fatal)"

echo ""
echo "=========================================="
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
