"""
Agent-powered factor quality review + regime-aware selection.

Called by daily_pipeline.py after programmatic mining.
Uses the shared Factor Lab backend order:
  1. Claude SDK / CLI when available
  2. Codex fallback when Claude is unavailable or broken
"""
import json
import os
from datetime import date
from pathlib import Path

import duckdb

from src.agent.backends import call_agent


REPO_ROOT = Path(__file__).resolve().parents[2]


def _call_agent(prompt: str, max_tokens: int = 2000) -> str:
    try:
        return call_agent(
            prompt,
            model=os.environ.get("FACTOR_LAB_AGENT_MODEL", "claude-sonnet-4-6"),
            max_tokens=max_tokens,
            repo_root=REPO_ROOT,
        )
    except Exception as exc:
        print(f"  Agent call failed ({exc})")
        return ""


def agent_quality_review(candidates: list[dict], market: str) -> list[dict]:
    """
    Have agent review top 30 candidates for false signals.
    Returns filtered list with agent's assessment.
    """
    if not candidates:
        return candidates

    # Build review prompt
    factor_table = "\n".join(
        f"  {i+1}. IC={c['ic']:.4f} IR={c['ic_ir']:.3f} Q5-Q1={c['q5_q1']:.3f}% Mono={c['mono']:.2f} | {c['formula']}"
        for i, c in enumerate(candidates[:30])
    )

    prompt = f"""You are a quantitative researcher reviewing factor candidates for {market.upper()} market.

These {len(candidates[:30])} factors passed initial IC/IR gates. Review each for FALSE SIGNALS:

{factor_table}

Known false signal patterns:
- Amihud illiquidity: abs(ret)/volume creates alpha from zero-volume stocks + volatility clustering
- Size effect: rank(volume) or rank(market_cap) is size factor, not alpha
- Look-ahead: factors using today's return to predict tomorrow (autocorrelation, not prediction)
- Survivorship: factors that only work on stocks that survive (low-price penny stock bias)

For EACH factor, respond with ONE line:
KEEP [number] — [brief reason]
or
REJECT [number] — [false signal type]: [explanation]

Then at the end:
TOP3: [numbers of the 3 best factors for today, considering current regime]
REGIME_NOTE: [1 sentence on current market regime and which factor types to favor]"""

    response = _call_agent(prompt)
    if not response:
        print("  Agent review failed, using programmatic ranking")
        return candidates

    # Parse rejections
    rejected_indices = set()
    top3_indices = []

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("REJECT"):
            try:
                num = int(line.split()[1]) - 1
                rejected_indices.add(num)
                print(f"  Agent rejected #{num+1}: {line[line.index('—')+1:].strip()}")
            except (ValueError, IndexError):
                pass
        elif line.startswith("TOP3:"):
            try:
                nums = [int(x.strip().rstrip(',')) - 1 for x in line[5:].split() if x.strip().rstrip(',').isdigit()]
                top3_indices = [n for n in nums if n not in rejected_indices]
            except Exception:
                pass
        elif line.startswith("REGIME_NOTE:"):
            print(f"  Agent regime: {line[12:].strip()}")

    # Filter out rejected
    filtered = [c for i, c in enumerate(candidates[:30]) if i not in rejected_indices]
    # Add remaining candidates beyond 30
    filtered.extend(candidates[30:])

    # If agent identified top 3, move them to front
    if top3_indices:
        top3 = [candidates[i] for i in top3_indices if i < len(candidates)]
        rest = [c for c in filtered if c not in top3]
        filtered = top3 + rest
        print(f"  Agent selected top 3: {[candidates[i]['name'] for i in top3_indices[:3]]}")

    print(f"  After review: {len(rejected_indices)} rejected, {len(filtered)} remaining")
    return filtered


def agent_regime_selection(promoted_factors: list[dict], market: str,
                           regime_dist: dict | None = None) -> dict[str, float]:
    """
    Have agent decide factor weights based on current regime.
    Returns {factor_id: weight} for today's lab_composite.
    """
    if not promoted_factors:
        return {}

    # Build context
    factor_list = "\n".join(
        f"  {f['factor_id']}: {f['formula'][:60]} | IC_7d={f.get('ic_7d',0):.4f} | status={f['status']}"
        for f in promoted_factors
    )

    regime_info = ""
    if regime_dist:
        regime_info = f"""
Current regime distribution:
  Trending: {regime_dist.get('trending', 0)}%
  Mean-reverting: {regime_dist.get('mean_reverting', 0)}%
  Noisy: {regime_dist.get('noisy', 0)}%"""

    prompt = f"""You are a portfolio manager allocating weights across alpha factors for {market.upper()} market.

Active promoted factors:
{factor_list}
{regime_info}

Assign percentage weights to each factor for TODAY's composite signal.
Rules:
- Weights must sum to 100%
- Factors with declining 7D IC should get lower weight
- In mean-reverting regime, favor reversal/value factors
- In trending regime, favor momentum/breakout factors
- In noisy regime, favor volume-driven factors

Output format (one factor per line):
[factor_id]: [weight]%
REASONING: [1-2 sentences why these weights]"""

    response = _call_agent(prompt, max_tokens=500)
    if not response:
        # Equal weight fallback
        w = 1.0 / len(promoted_factors)
        return {f["factor_id"]: w for f in promoted_factors}

    # Parse weights
    weights = {}
    for line in response.split("\n"):
        line = line.strip()
        if ":" in line and "%" in line and not line.startswith("REASONING"):
            try:
                parts = line.split(":")
                fid = parts[0].strip()
                pct = float(parts[1].strip().replace("%", ""))
                weights[fid] = pct / 100.0
            except (ValueError, IndexError):
                pass
        elif line.startswith("REASONING:"):
            print(f"  Agent weighting: {line[10:].strip()}")

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        w = 1.0 / len(promoted_factors)
        weights = {f["factor_id"]: w for f in promoted_factors}

    return weights


def generate_factor_commentary(top_factors: list[dict], market: str) -> str:
    """Generate brief commentary for research report."""
    if not top_factors:
        return ""

    factor_info = "\n".join(
        f"  - {f['name']}: {f['formula'][:50]} (IC={f['ic']:.4f}, {f['hypothesis']})"
        for f in top_factors[:3]
    )

    prompt = f"""Write 2-3 sentences in Chinese summarizing today's active alpha factors for the {market.upper()} research report.

Active factors:
{factor_info}

Requirements:
- 中文输出
- 简洁专业
- 说明每个因子捕捉什么信号
- 不给投资建议"""

    return _call_agent(prompt, max_tokens=300)
