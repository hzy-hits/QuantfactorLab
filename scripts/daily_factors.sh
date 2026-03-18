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

echo ""
echo "=========================================="
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
