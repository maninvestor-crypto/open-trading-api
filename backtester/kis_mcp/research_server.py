"""Research-only extension of the existing KIS Backtest MCP server.

Run on macOS/Linux:
    cd backtester
    uv run python -m kis_mcp.research_server

This imports the existing MCP app and adds robust research tools.  It exposes no
broker order/trading action; the surface is intentionally backtest/data only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from kis_mcp.server import mcp
from kis_mcp.tools.research import (
    audit_kis_history,
    probe_kis_history,
    run_cost_stress,
    run_walk_forward,
)

logger = logging.getLogger(__name__)


@mcp.tool()
def probe_kis_history_tool(
    symbol: str,
    requested_start: str = "2000-01-01",
    end_date: Optional[str] = None,
    mode: Literal["live", "paper"] = "live",
    tolerance_days: int = 10,
) -> dict:
    """Measure the actual KIS daily-history coverage for one Korean stock."""
    return probe_kis_history(
        symbol=symbol,
        requested_start=requested_start,
        end_date=end_date,
        mode=mode,
        tolerance_days=tolerance_days,
    )


@mcp.tool()
async def audit_kis_history_tool(
    symbols: List[str],
    requested_start: str = "2000-01-01",
    end_date: Optional[str] = None,
    mode: Literal["live", "paper"] = "live",
    tolerance_days: int = 10,
) -> dict:
    """Audit actual KIS daily-history coverage across multiple symbols."""
    return await audit_kis_history(
        symbols=symbols,
        requested_start=requested_start,
        end_date=end_date,
        mode=mode,
        tolerance_days=tolerance_days,
    )


@mcp.tool()
async def run_walk_forward_tool(
    strategy_id: str,
    symbols: List[str],
    parameters: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    train_months: int = 60,
    test_months: int = 12,
    step_months: int = 12,
    max_windows: int = 12,
    search_type: Literal["grid", "random"] = "random",
    max_samples: int = 20,
    target: str = "sharpe_ratio",
    initial_capital: float = 10_000_000,
    commission_rate: float = 0.00015,
    tax_rate: float = 0.002,
    slippage: float = 0.001,
    seed: Optional[int] = 42,
    timeout_per_stage: float = 900.0,
) -> dict:
    """Rolling train->OOS research: tune only in train, freeze params in OOS."""
    return await run_walk_forward(
        strategy_id=strategy_id,
        symbols=symbols,
        parameters=parameters,
        start_date=start_date,
        end_date=end_date,
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        max_windows=max_windows,
        search_type=search_type,
        max_samples=max_samples,
        target=target,
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        tax_rate=tax_rate,
        slippage=slippage,
        seed=seed,
        timeout_per_stage=timeout_per_stage,
    )


@mcp.tool()
async def run_cost_stress_tool(
    strategy_id: str,
    symbols: List[str],
    start_date: str,
    end_date: str,
    param_overrides: Optional[Dict[str, Any]] = None,
    slippage_scenarios: Optional[List[float]] = None,
    commission_rate: float = 0.00015,
    tax_rate: float = 0.002,
    initial_capital: float = 10_000_000,
    timeout: float = 900.0,
) -> dict:
    """Re-run a frozen strategy under progressively harsher slippage assumptions."""
    return await run_cost_stress(
        strategy_id=strategy_id,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        param_overrides=param_overrides,
        slippage_scenarios=slippage_scenarios,
        commission_rate=commission_rate,
        tax_rate=tax_rate,
        initial_capital=initial_capital,
        timeout=timeout,
    )


if __name__ == "__main__":
    logger.info("KIS GPT Research MCP server starting (backtest/data tools only)")
    mcp.run(transport="streamable-http")
