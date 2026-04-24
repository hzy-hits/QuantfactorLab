"""
Prompt templates for the factor mining agent.

The agent proposes factor hypotheses with economic logic, formulas in the
constrained DSL, and a direction.  The system evaluates them — the agent
never touches backtest parameters or OOS data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# DSL reference (compact, for embedding in prompts)
# ---------------------------------------------------------------------------

DSL_REFERENCE = """\
## DSL Operators

### Time-series (per-stock, along time axis)
ts_mean(x, n), ts_std(x, n), ts_max(x, n), ts_min(x, n), ts_sum(x, n)
ts_rank(x, n), ts_argmax(x, n), ts_argmin(x, n)
ts_corr(x, y, n), ts_cov(x, y, n), ts_skew(x, n), ts_kurt(x, n)
ts_count(cond, n), ts_product(x, n)
delta(x, n), pct_change(x, n), shift(x, n)
decay_linear(x, n), decay_exp(x, n)

### Cross-sectional (all stocks on same day)
rank(x), zscore(x), demean(x), quantile(x, n), neutralize(x, group)

### Universal
abs(x), sign(x), log(x), sqrt(x), power(x, p), clamp(x, lo, hi)
max(x, y), min(x, y), if_then(cond, a, b)
Infix: + - * /    Unary: -

### Allowed lookback windows
1, 2, 3, 5, 10, 14, 20, 30, 40, 60, 120, 250

### Max expression depth: 4
### Max expression length: 200 characters

## Available features (ONLY use these — anything else causes immediate FAIL)
# Price: close, open, high, low
# Returns: ret_1d, ret_5d, ret_20d
# Volume: volume, volume_ratio
# CN-only (from daily_basic): amount, turnover_rate, pe_ttm, pb, ps_ttm, market_cap, circ_market_cap
# US: price + volume only — NO fundamentals, NO amount, NO turnover_rate
#
# WARNING: If your formula references a feature not listed above, it WILL fail.
# Do NOT guess features. Only use what is listed here.

## IMPORTANT: What we know works and what doesn't

### Verified facts:
- Single-factor strategies outperform multi-factor composites
- amount_stability_20 is the most promising factor (OOS Sharpe 2.14, p=0.12)
- range_vol_20 looked great in-sample but FAILED out-of-sample (Sharpe -1.53)
- Voting/IC-weighted composites destroy signal
- 10-20 day holding periods match factor prediction horizon
- Factors predicting 1-day returns have NO alpha in A-shares

### Critical anti-overfit rules:
- ALL backtests use look-ahead-safe lookback (shifted back by hold period)
- Results must pass shuffle test (p<0.05) to be considered real
- Train/test split: first half trains, second half tests. Never mix.

### What to search for (beyond factor formulas):
1. FACTOR HEALTH THRESHOLDS: When should a factor be retired?
   Current: IC < 0.005 for 5 days → watchlist → 5 more days → retire
   Search: What IC threshold? What lookback window? Rolling vs fixed?

2. FACTOR ROTATION RULES: When to switch from one factor to another?
   Current: every 20 days, pick factor with best lookback Sharpe
   Search: What triggers a switch? IC decay? Regime change? Calendar?

3. HOLDING PERIOD: How long to hold positions?
   Current: fixed 20 days
   Search: Should it adapt to volatility regime? Factor IC persistence?

4. POSITION SIZING: How to weight picks?
   Current: rank-based (top pick 18%, bottom 2%)
   Search: Equal weight? Inverse vol? Kelly criterion?
"""

# ---------------------------------------------------------------------------
# Regime labels
# ---------------------------------------------------------------------------

REGIME_DESCRIPTIONS = {
    "trending": "Positive lag-1 autocorrelation (>0.15). Momentum strategies tend to work.",
    "mean_reverting": "Negative lag-1 autocorrelation (<-0.10). Reversal strategies tend to work.",
    "noisy": "Low autocorrelation. Neither momentum nor reversal has clear edge. Try microstructure or volume factors.",
}


def build_system_prompt(
    market: str,
    regime_dist: dict[str, float] | None = None,
    existing_factors: list[dict] | None = None,
    session_context: str | None = None,
) -> str:
    """Build the system prompt with DSL reference, market context, and constraints.

    Parameters
    ----------
    market : str
        'cn' or 'us'.
    regime_dist : dict, optional
        Distribution of regimes in recent data, e.g. {"trending": 0.3, "mean_reverting": 0.2, "noisy": 0.5}.
    existing_factors : list of dict, optional
        Previously discovered factors (name, formula, is_ic).

    Returns
    -------
    str : system prompt.
    """
    market_label = "A-Share (China)" if market == "cn" else "US Equity"
    cost_note = (
        "Transaction cost: ~0.3% round-trip (stamp duty + commission). Turnover matters."
        if market == "cn"
        else "Transaction cost: ~0.1% round-trip. Turnover still matters but less punishing."
    )

    if market == "cn":
        feature_reminder = (
            "\n## YOUR AVAILABLE FEATURES (CN)\n"
            "close, open, high, low, volume, volume_ratio, ret_1d, ret_5d, ret_20d, "
            "amount, turnover_rate, pe_ttm, pb, ps_ttm, market_cap, circ_market_cap\n"
            "DO NOT use any feature not in this list.\n"
        )
    else:
        feature_reminder = (
            "\n## YOUR AVAILABLE FEATURES (US)\n"
            "close, open, high, low, volume, volume_ratio, ret_1d, ret_5d, ret_20d\n"
            "DO NOT use amount, turnover_rate, pe_ttm, or any fundamental — they do NOT exist for US.\n"
        )

    regime_section = ""
    if regime_dist:
        regime_section = "\n## Current Market Regime Distribution\n"
        for regime, pct in regime_dist.items():
            desc = REGIME_DESCRIPTIONS.get(regime, "")
            regime_section += f"- {regime}: {pct:.0%} of recent days. {desc}\n"

    existing_section = ""
    if existing_factors:
        promoted = [f for f in existing_factors if f.get("status") == "promoted"]
        retired = [f for f in existing_factors if f.get("status") in ("retired", "watchlist")]

        if promoted:
            existing_section += "\n## Currently Promoted Factors (avoid correlation > 0.6)\n"
            for f in promoted:
                note = f.get("note", "")
                existing_section += f"- {f['name']}: `{f['formula']}` (IC={f['is_ic']:.3f} {note})\n"

        if retired:
            existing_section += "\n## Recently Retired/Watchlist Factors (learn from their failure)\n"
            for f in retired:
                note = f.get("note", "")
                existing_section += f"- {f['name']}: `{f['formula']}` — {note}\n"

    session_section = ""
    if session_context:
        session_section = "\n## Resumable Session Context\n"
        session_section += session_context.strip() + "\n"

    return f"""\
You are a quantitative researcher mining alpha factors for {market_label}.

Your goal: propose factor hypotheses with ECONOMIC LOGIC, express them as DSL formulas,
and iterate based on IS backtest feedback. You have a limited experiment budget — use it wisely.

## Rules
1. Every factor MUST have an economic hypothesis (1-2 sentences explaining WHY it should predict returns).
2. Formulas use the DSL below. No arbitrary code.
3. You see IS (in-sample) metrics only. OOS results are shown as PASS/FAIL only at session end.
4. Avoid proposing factors highly correlated with existing ones (corr > 0.7 = auto-reject).
5. Direction: "long" means high factor value → expect positive return. "short" means high → negative.
6. Think about turnover: noisy daily signals cost more in transaction costs.
7. Combine operators creatively. rank(x) * rank(y) is a useful interaction pattern.
8. Consider regime context: momentum works in trending markets, reversal in mean-reverting.

{DSL_REFERENCE}

{cost_note}
{feature_reminder}
{regime_section}
{existing_section}
{session_section}

## Response Format (STRICT — always use this exact format)
HYPOTHESIS: <1-2 sentences explaining the economic logic>
FORMULA: <DSL expression>
DIRECTION: <long or short>
NAME: <snake_case short name>
"""


def build_feedback_prompt(experiments: list[dict], budget_remaining: int) -> str:
    """Build prompt showing previous experiment results for agent to iterate.

    Parameters
    ----------
    experiments : list of dict
        Each dict has keys: name, formula, hypothesis, direction, is_ic, is_ic_ir,
        is_sharpe, is_turnover, is_monotonicity, gates_passed, gate_details.
    budget_remaining : int
        How many experiments the agent can still run.

    Returns
    -------
    str : feedback prompt.
    """
    lines = [f"## Session Progress ({len(experiments)} experiments done, {budget_remaining} remaining)\n"]

    if not experiments:
        lines.append("No experiments yet. Propose your first factor hypothesis.")
        return "\n".join(lines)

    lines.append("| # | Name | IC | IC_IR | Sharpe | Turnover | Mono | Gates |")
    lines.append("|---|------|-----|-------|--------|----------|------|-------|")

    for i, exp in enumerate(experiments, 1):
        gates_str = "PASS" if exp.get("gates_passed") else "FAIL"
        lines.append(
            f"| {i} | {exp.get('name', '?')} | "
            f"{exp.get('is_ic', 0):.4f} | "
            f"{exp.get('is_ic_ir', 0):.3f} | "
            f"{exp.get('is_sharpe', 0):.3f} | "
            f"{exp.get('is_turnover', 0):.3f} | "
            f"{exp.get('is_monotonicity', 0):.3f} | "
            f"{gates_str} |"
        )

    # Show gate details for the most recent experiment
    last = experiments[-1]
    if "gate_details" in last and last["gate_details"]:
        lines.append(f"\nLast experiment gate details:")
        for gate_name, info in last["gate_details"].items():
            status = "PASS" if info["passed"] else "FAIL"
            val = info.get("abs_value", info.get("value", "?"))
            lines.append(f"  {gate_name}: {status} (value={val}, threshold={info.get('threshold', '?')})")

    # Guidance
    lines.append(f"\nBudget remaining: {budget_remaining} experiments.")
    if budget_remaining <= 10:
        lines.append("LOW BUDGET: Focus on refining your best-performing factors.")
    lines.append("\nPropose your next factor hypothesis. Use the STRICT response format.")

    return "\n".join(lines)


@dataclass
class ParsedResponse:
    hypothesis: str
    formula: str
    direction: str  # "long" or "short"
    name: str
    raw: str  # original response text


def parse_agent_response(response_text: str) -> ParsedResponse | None:
    """Parse agent's response to extract hypothesis, formula, direction, name.

    Returns None if the response cannot be parsed.

    Parameters
    ----------
    response_text : str
        Raw text from the agent.

    Returns
    -------
    ParsedResponse or None.
    """
    text = response_text.strip()

    # Extract fields using prefix matching (case insensitive)
    hypothesis = _extract_field(text, "HYPOTHESIS")
    formula = _extract_field(text, "FORMULA")
    direction = _extract_field(text, "DIRECTION")
    name = _extract_field(text, "NAME")

    if not formula:
        return None

    # Clean formula: remove backticks if the agent wrapped it
    formula = formula.strip("`").strip()

    # Normalise direction
    direction = (direction or "long").strip().lower()
    if direction not in ("long", "short"):
        direction = "long"

    # Default name if not provided
    if not name:
        name = "unnamed_factor"
    name = re.sub(r"[^a-z0-9_]", "_", name.strip().lower())

    return ParsedResponse(
        hypothesis=hypothesis or "No hypothesis provided.",
        formula=formula,
        direction=direction,
        name=name,
        raw=text,
    )


def _extract_field(text: str, field_name: str) -> str | None:
    """Extract a field value from 'FIELD_NAME: value' format."""
    pattern = rf"(?i){field_name}\s*:\s*(.+?)(?:\n[A-Z_]+\s*:|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
