I ran the exact script on the CN DuckDB. It loaded `198,838` price rows across `5,569` symbols. The IC in this codebase is daily cross-sectional Spearman IC against `fwd_5d`.

Two of the 10 formulas did not run as written because the raw price frame does not contain those derived fields:
- `bb_mean_revert`: missing `ma20_dist`
- `rsi_contrarian`: missing `rsi_14`

**Results**
| Factor | IC | IC_IR | Q5-Q1 % | Mono |
|---|---:|---:|---:|---:|
| `vol_shrink_reversal` | `0.0241` | `0.149` | `-0.005` | `-0.20` |
| `vol_price_corr` | `0.0228` | `0.133` | `-0.099` | `0.00` |
| `shrink_then_surge` | `0.0191` | `0.176` | `0.102` | `0.80` |
| `reversal_5d` | `0.0149` | `0.078` | `-0.020` | `-0.60` |
| `momentum_5d` | `-0.0149` | `-0.078` | `0.019` | `0.60` |
| `vol_breakout` | `-0.0035` | `-0.021` | `0.047` | `0.10` |
| `kbar_strength` | `-0.0033` | `-0.018` | `0.060` | `0.30` |
| `high_position` | `-0.0020` | `-0.011` | `0.312` | `0.90` |

The best raw ICs were:
- `vol_shrink_reversal`: `0.0241`
- `vol_price_corr`: `0.0228`
- `shrink_then_surge`: `0.0191`

The clearest overall candidate is `shrink_then_surge`, not because it has the highest IC, but because its IC, best IC_IR (`0.176`), positive long-short spread (`0.102%`), and good monotonicity (`0.80`) all line up.

**Patterns**
- Short-horizon reversal looks better than momentum in this A-share sample. `momentum_5d` has negative IC, while `reversal_5d` is the mirror image with positive IC.
- Volume-conditioned contrarian/exhaustion ideas are stronger than pure breakout or candlestick signals. The top three are all volume-state driven.
- `vol_shrink_reversal` and `vol_price_corr` have decent ICs but poor quintile shape, so the signal is weak/non-monotonic in the tails.
- `high_position` is interesting: near-zero IC, but the strongest quintile spread and very high monotonicity. That usually means the effect is more bucketed/nonlinear than smoothly rank-linear.
- `vol_breakout` and `kbar_strength` look close to noise on this setup.

If you want, I can rewrite the two failed factors into pure DSL terms and rerun a full 10/10 session.