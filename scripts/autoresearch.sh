#!/bin/bash
# Autoresearch: AI-driven factor mining via Claude agent loop.
# Runs nightly Mon-Fri 22:00-02:00 CST (4 hours).
# CN 2 hours (~150 experiments), US 2 hours (~150 experiments).
#
# Cron: Mon-Fri 22:00 CST
#   0 22 * * 1-5 cd /home/ivena/coding/python/factor-lab && bash scripts/autoresearch.sh >> logs/autoresearch.log 2>&1

set -uo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"
mkdir -p logs reports

BUDGET=150  # ~2 hours per market at ~1 min/experiment
DATE=$(date +%Y%m%d)
PYTHON=/home/ivena/miniconda3/bin/python3

echo "=========================================="
echo "  Autoresearch Session — $DATE"
echo "  Start: $(date '+%H:%M:%S')"
echo "  Budget: $BUDGET per market (CN+US)"
echo "=========================================="

# CN market: ~2 hours
echo ""
echo "=== [$(date '+%H:%M:%S')] A-Share Autoresearch ==="
timeout 7200 $PYTHON -m src.agent.loop \
    --market cn \
    --budget $BUDGET \
    --output "reports/autoresearch_cn_${DATE}.md" \
    || echo "CN autoresearch ended (exit $?)"

echo "=== [$(date '+%H:%M:%S')] CN done ==="

# US market: ~2 hours
echo ""
echo "=== [$(date '+%H:%M:%S')] US Autoresearch ==="
timeout 7200 $PYTHON -m src.agent.loop \
    --market us \
    --budget $BUDGET \
    --output "reports/autoresearch_us_${DATE}.md" \
    || echo "US autoresearch ended (exit $?)"

echo "=== [$(date '+%H:%M:%S')] US done ==="

# Export new factors to pipeline DBs
echo ""
echo "=== [$(date '+%H:%M:%S')] Exporting to pipelines ==="
$PYTHON -m src.mining.export_to_pipeline --market cn || echo "CN export failed"
$PYTHON -m src.mining.export_to_pipeline --market us || echo "US export failed"

echo ""
echo "=========================================="
echo "  Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
