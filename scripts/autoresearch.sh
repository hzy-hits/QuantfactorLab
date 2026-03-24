#!/bin/bash
# Autoresearch: AI-driven factor mining via Claude agent loop.
# Runs src.agent.loop which uses Claude to generate hypotheses,
# write DSL formulas, evaluate them, and promote winners.
#
# Unlike batch_mine (fixed templates), this discovers genuinely new factors.
#
# Cron: Tue/Thu 22:00 CST (after market hours, before daily pipeline at 04:00)
#   0 22 * * 2,4 cd /home/ivena/coding/python/factor-lab && bash scripts/autoresearch.sh >> logs/autoresearch.log 2>&1

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p logs reports

echo "=========================================="
echo "  Autoresearch Session"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# CN market: 50 experiments
echo ""
echo "=== A-Share Autoresearch ==="
/home/ivena/miniconda3/bin/python3 -m src.agent.loop \
    --market cn \
    --budget 50 \
    --output "reports/autoresearch_cn_$(date +%Y%m%d).md" \
    || echo "CN autoresearch failed (exit $?)"

# US market: 50 experiments
echo ""
echo "=== US Autoresearch ==="
/home/ivena/miniconda3/bin/python3 -m src.agent.loop \
    --market us \
    --budget 50 \
    --output "reports/autoresearch_us_$(date +%Y%m%d).md" \
    || echo "US autoresearch failed (exit $?)"

# Export new factors to pipeline DBs
echo ""
echo "=== Exporting to pipelines ==="
/home/ivena/miniconda3/bin/python3 -m src.mining.export_to_pipeline --market cn || echo "CN export failed"
/home/ivena/miniconda3/bin/python3 -m src.mining.export_to_pipeline --market us || echo "US export failed"

echo ""
echo "=========================================="
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
