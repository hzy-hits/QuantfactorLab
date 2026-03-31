#!/bin/bash
# Run a factor mining agent session.
#
# Usage:
#   ./scripts/run_session.sh          # defaults: cn market, 50 budget
#   ./scripts/run_session.sh us 30    # US market, 30 experiments

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

MARKET="${1:-cn}"
BUDGET="${2:-50}"
OUTPUT="reports/session_${MARKET}_$(date +%Y%m%d_%H%M%S).md"

echo "Factor Lab Session"
echo "  Market: ${MARKET}"
echo "  Budget: ${BUDGET}"
echo "  Output: ${OUTPUT}"
echo ""

python3 -m src.agent.loop \
    --market "${MARKET}" \
    --budget "${BUDGET}" \
    --output "${OUTPUT}"
