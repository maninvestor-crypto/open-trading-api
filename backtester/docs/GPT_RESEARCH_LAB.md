# GPT Research Lab (KIS + LEAN + MCP)

This extension turns the existing backtester into a research-only MCP surface for
GPT/agent workflows. It does **not** add broker order execution.

## Why this fits the existing repository

The repository already has the hard parts:

- KIS OAuth/data provider
- daily KRX data -> LEAN CSV conversion/cache
- QuantConnect LEAN execution in Docker
- YAML/preset strategies
- asynchronous backtest jobs
- batch backtests and grid/random parameter optimization
- Streamable HTTP MCP server

The research extension adds the missing robustness layer instead of replacing the
existing engine.

## macOS / Apple Silicon

The Python/KIS REST layer is OS-neutral. The project requires Python 3.11+, `uv`,
Node only for the optional UI, and Docker for LEAN.

Recommended setup:

```bash
brew install uv
# Install Docker Desktop for Mac separately and start it.
cd backtester
uv sync
bash scripts/setup_lean_data.sh
docker pull quantconnect/lean:latest
```

Sanity checks:

```bash
docker info
docker image inspect quantconnect/lean:latest --format '{{.Architecture}}'
uv run python -c 'import platform; print(platform.machine())'
```

Start the research MCP server:

```bash
cd backtester
uv run python -m kis_mcp.research_server
```

Default endpoint: `http://127.0.0.1:3846/mcp`.

> ChatGPT's hosted custom MCP connection needs a reachable HTTPS MCP endpoint;
> `127.0.0.1` is intentionally local-only. Keep the local bind for safety and put
> an authenticated HTTPS tunnel/reverse proxy in front of it only when a hosted
> client must connect. Never expose KIS secrets or a live-trading endpoint.

## KIS credentials

The existing provider reads `~/KIS/config/kis_devlp.yaml`. Keep it outside the
repository. The research tools need market-data authentication only.

## KIS long-history audit

Do not assume that requesting 2000-01-01 means KIS actually returned that range.
Use the new MCP tools:

- `probe_kis_history_tool(symbol, requested_start, ...)`
- `audit_kis_history_tool(symbols, requested_start, ...)`

They report the actual earliest/latest returned daily bars and whether the full
requested range was covered. This is important because a partial dataset can
otherwise look like a valid 20-year backtest.

Example research request:

```text
Audit 005930 and 000660 from 2000-01-01. If KIS does not cover the requested
start, stop and report the missing range rather than silently backtesting it.
```

## Robust strategy workflow

Recommended agent loop:

1. Audit data coverage.
2. Propose a strategy hypothesis with an economic/market rationale.
3. Use `run_optimize` only inside training data.
4. Use `run_walk_forward_tool` to freeze the chosen parameters and test the next
   unseen window.
5. Run `run_cost_stress_tool` on survivors.
6. Prefer parameter plateaus and consistent OOS performance over a single peak.
7. Keep a final untouched holdout period for the final candidate set.

Example walk-forward call conceptually:

```json
{
  "strategy_id": "sma_crossover",
  "symbols": ["005930"],
  "parameters": [
    {"name": "fast_period", "min": 5, "max": 30, "step": 5},
    {"name": "slow_period", "min": 40, "max": 200, "step": 20}
  ],
  "start_date": "2005-01-01",
  "end_date": "2025-12-31",
  "train_months": 60,
  "test_months": 12,
  "step_months": 12,
  "search_type": "random",
  "max_samples": 30,
  "target": "sharpe_ratio",
  "slippage": 0.001
}
```

The returned `robustness_score` is a transparent heuristic convenience score,
not proof of alpha. Raw OOS windows remain the primary evidence.

## Data-source abstraction: next step for 20+ years

KIS remains the default live/recent data source. For a genuine long-history
research database, add a provider adapter rather than changing the backtester.
Normalize every source into the existing `Bar` / LEAN daily CSV shape.

Recommended adapter contract:

```python
class HistoricalDailyProvider(Protocol):
    def get_history(self, symbol, start, end) -> list[Bar]: ...
```

Then route:

```text
KIS recent/official broker feed  ---> normalize ---> LEAN CSV cache
Long-history vendor/KRX dataset ---> normalize ---> LEAN CSV cache
                                           |
                                           v
                                      same backtester
```

For institutional-quality research, the long-history layer should also preserve
point-in-time universe membership, delisted securities, corporate actions,
adjustment factors, and symbol changes. Current-survivor-only OHLCV is not enough
for trustworthy cross-sectional 20-year strategy research.

## Research guardrails

- Backtest/research tools only; no order execution.
- Never optimize on the final holdout.
- Record commission, tax and slippage assumptions.
- Treat a strategy with too few trades as inconclusive.
- Avoid selecting only by CAGR/total return.
- Stress multiple market regimes and symbols.
- Keep KIS credentials outside Git and never return tokens through MCP.
