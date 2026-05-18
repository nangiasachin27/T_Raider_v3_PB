# T_Raider Quant v2
### A Hybrid Quantitative Screener for Nifty 50 — Python · NSE · Paper-Trading Ready

---

## What It Does
T_Raider scans the Nifty 50 universe daily, assigns each stock its statistically most stable trading strategy (validated via Monte Carlo simulation), and surfaces actionable BUY/SELL signals with position sizing, confidence tiers, and portfolio-aware concentration limits. It is designed for **human-reviewed, semi-automated trading** — not blind execution.

---

## Architecture

```
auto_optimizer.py          ← Monthly: finds best strategy per stock via 5,000 Monte Carlo runs
        ↓
config/optimal_params.json ← "The Brain": strategy + params + stability score per ticker
        ↓
daily_screener.py          ← Daily: runs signals, applies regime gate, sector filter, volume check
        ↓
autopilot/bot.py           ← 3:15 PM: executes approved trades using ATR-based sizing
        ↓
autopilot/tracker.py       ← Tracks live net worth, realised P&L, and STCG tax estimate
```

| Module | File | Role |
|---|---|---|
| Optimizer | `auto_optimizer.py` | Monte Carlo strategy selection |
| Screener | `daily_screener.py` | Daily signal engine + regime awareness |
| Strategies | `strategies/` | Trend, RSI, MACD, Bollinger, Breakout, Stretch |
| Execution | `autopilot/bot.py` | ATR position sizing + order logging |
| Tracker | `autopilot/tracker.py` | Portfolio P&L + equity curve |
| Memory | `config/` | `portfolio.json`, `optimal_params.json`, `stocks.json` |

---

## Six Strategies

| Strategy | Type | Entry Condition |
|---|---|---|
| **Trend Follower** | Trend | 50-day MA crosses above 200-day MA (Golden Cross) |
| **Mean Reversion** | RSI | RSI(14) crosses below 30 (oversold) |
| **MACD** | Momentum | MACD Line crosses above Signal Line |
| **Bollinger** | Volatility | Price crosses below Lower Band (2σ) |
| **Breakout** | Breakout | Price closes above 20-day rolling high |
| **Stretch** | Pullback | Price dips >5% below 20-day MA then recovers |

---

## Three Risk Layers

**Layer 1 — Monte Carlo Filter**
Each strategy must achieve ≥60% Probability of Profit across 5,000 reshuffled trade sequences before it is assigned to a stock. Unstable strategies are rejected.

**Layer 2 — ATR Position Sizing**
```
Qty = (1% of Portfolio Value) / (ATR × 2)
```
Volatile stocks get smaller positions automatically. No flat lot sizes.

**Layer 3 — Dual Stop Loss**
Every trade carries a 10% fixed stop (from entry) and a 15% trailing stop (from peak). The higher of the two is always active, locking in gains as the trade moves in your favour.

---

## Daily Workflow

```bash
# 1. Run once a month — updates strategy assignments
python auto_optimizer.py

# 2. Run daily at 3:15 PM IST — generates today's signals
python daily_screener.py --mode CONSERVATIVE

# 3. Review HIGH confidence signals, execute manually or via bot
python autopilot/bot.py

# 4. Check portfolio status anytime
python autopilot/tracker.py
```

---

## Screener Modes

| Mode | Downtrend Behaviour | When to Use |
|---|---|---|
| `CONSERVATIVE` | All BUYs suppressed | Default — safest, recommended for live use |
| `BALANCED` | Mean-reversion BUYs if Nifty ≥5% below 50d high | After 3+ months of validated paper trading |
| `AGGRESSIVE` | Mean-reversion BUYs if Nifty ≥3% below 50d high | Not recommended until thresholds are backtested |

---

## Signal Confidence Tiers

| Tier | Meaning | Action |
|---|---|---|
| 🟢 **HIGH** | Stability ≥70%, within concentration limits, volume confirmed | Pre-approved for execution |
| 🟡 **MEDIUM** | Stability 60–70% or marginal sizing | Review before acting |
| ❌ **SKIP** | Failed regime, volume, cash, or concentration gate | Logged only, no action |

---

## Portfolio Guardrails

- Maximum **20% of portfolio** in any single stock
- Minimum **60% stability score** for any BUY signal
- Volume confirmation: today's volume must be ≥80% of 20-day average
- Market regime gate: Nifty 50 close vs EMA-50 checked before every session
- Sector momentum: signals boosted (1.2×) or penalised (0.8×) by sector RS score
- Transaction friction: 0.2% applied per trade side (covers STT, brokerage, slippage)

---

## Tech Stack

| | |
|---|---|
| **Language** | Python 3.11+ |
| **Data** | yfinance, jugaad-data, curl_cffi (anti-block) |
| **Analysis** | pandas, numpy, matplotlib |
| **Automation** | GitHub Actions (daily 3:15 PM IST trigger) |
| **Key deps** | `requirements.txt` — 60+ packages |

---

## Setup

```bash
git clone https://github.com/nangiasachin27/T_Raider_Quant_v2
cd T_Raider_Quant_v2
pip install -r requirements.txt

# First run — build strategy assignments (takes ~10–20 min)
python auto_optimizer.py

# Then run the screener
python daily_screener.py --mode CONSERVATIVE
```

---

## Status

> **Paper trading mode** — CONSERVATIVE, running daily since May 2026.
> Signals are reviewed manually before any execution.
> Target: 60 validated out-of-sample signals before considering live capital.

---

*Built for the Indian equity market. Not financial advice.*
