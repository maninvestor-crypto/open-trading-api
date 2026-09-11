"""GPT research tools for robust strategy discovery.

This module intentionally keeps research/backtesting separate from live trading.
It builds on the existing KIS + LEAN backtester and adds:

- strict KIS history coverage probing
- rolling walk-forward optimization / out-of-sample validation
- transaction-cost/slippage stress tests
- a compact robustness summary for GPT/agent ranking

The key design rule is simple: optimize only on the training window and evaluate
those frozen parameters on the following test window.
"""

from __future__ import annotations

import asyncio
import calendar
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Literal, Optional

from kis_backtest.models import Resolution
from kis_backtest.providers.kis.auth import KISAuth
from kis_backtest.providers.kis.data import KISDataProvider
from kis_mcp.schemas import error_response, success_response
from kis_mcp.tools.backtest import (
    get_backtest_result_wait,
    run_optimize,
    run_preset_backtest,
)


# ---------------------------------------------------------------------------
# Generic response helpers
# ---------------------------------------------------------------------------


def _data(response: Dict[str, Any]) -> Dict[str, Any]:
    """Return the payload of the project's {success, data} response envelope."""
    if not isinstance(response, dict):
        return {}
    payload = response.get("data")
    return payload if isinstance(payload, dict) else {}


def _result_from_wait_response(response: Dict[str, Any]) -> Dict[str, Any]:
    payload = _data(response)
    result = payload.get("result")
    return result if isinstance(result, dict) else {}


def _metric(result: Dict[str, Any], name: str) -> float:
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    basic = metrics.get("basic", {}) if isinstance(metrics, dict) else {}
    risk = metrics.get("risk", {}) if isinstance(metrics, dict) else {}
    trading = metrics.get("trading", {}) if isinstance(metrics, dict) else {}
    lookup = {
        "total_return": basic.get("total_return", 0.0),
        "annual_return": basic.get("annual_return", 0.0),
        "max_drawdown": basic.get("max_drawdown", 0.0),
        "sharpe_ratio": risk.get("sharpe_ratio", 0.0),
        "sortino_ratio": risk.get("sortino_ratio", 0.0),
        "win_rate": trading.get("win_rate", 0.0),
        "profit_loss_ratio": trading.get("profit_loss_ratio", 0.0),
    }
    try:
        return float(lookup.get(name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Date/window helpers
# ---------------------------------------------------------------------------


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _add_months(value: date, months: int) -> date:
    if months < 0:
        raise ValueError("months must be >= 0")
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    def as_dict(self) -> Dict[str, str]:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def build_walk_forward_windows(
    start_date: str,
    end_date: str,
    train_months: int = 60,
    test_months: int = 12,
    step_months: int = 12,
    max_windows: int = 12,
) -> List[WalkForwardWindow]:
    """Build fixed-length rolling train -> test windows without overlap leakage."""
    if train_months < 1 or test_months < 1 or step_months < 1:
        raise ValueError("train_months, test_months and step_months must be >= 1")
    if max_windows < 1:
        raise ValueError("max_windows must be >= 1")

    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start >= end:
        raise ValueError("start_date must be earlier than end_date")

    windows: List[WalkForwardWindow] = []
    train_start = start

    while len(windows) < max_windows:
        train_end = _add_months(train_start, train_months) - timedelta(days=1)
        test_start = train_end + timedelta(days=1)
        test_end = _add_months(test_start, test_months) - timedelta(days=1)

        if test_start > end:
            break
        test_end = min(test_end, end)
        if test_end < test_start:
            break

        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_start = _add_months(train_start, step_months)

    return windows


# ---------------------------------------------------------------------------
# KIS history coverage
# ---------------------------------------------------------------------------


def probe_kis_history(
    symbol: str,
    requested_start: str = "2000-01-01",
    end_date: Optional[str] = None,
    mode: Literal["live", "paper"] = "live",
    tolerance_days: int = 10,
) -> Dict[str, Any]:
    """Ask KIS for a long daily range and report the actual returned coverage.

    We deliberately do not hard-code a KIS 'oldest date'.  The server's actual
    retention can differ from the requested range, so the safest approach is to
    measure what the authenticated account/API returns.
    """
    req_start = _parse_date(requested_start)
    req_end = _parse_date(end_date) if end_date else date.today()
    if req_start >= req_end:
        return error_response("requested_start must be earlier than end_date")

    try:
        auth = KISAuth.from_env(mode=mode)
        provider = KISDataProvider(auth)
        bars = provider.get_history(
            symbol=symbol,
            start=req_start,
            end=req_end,
            resolution=Resolution.DAILY,
        )
    except Exception as exc:
        return error_response(f"KIS history probe failed: {exc}")

    if not bars:
        return error_response(
            f"KIS returned no daily bars for {symbol}",
            details={
                "symbol": symbol,
                "requested_start": req_start.isoformat(),
                "requested_end": req_end.isoformat(),
                "mode": mode,
            },
        )

    dates = sorted(bar.time.date() for bar in bars)
    unique_dates = sorted(set(dates))
    actual_start = unique_dates[0]
    actual_end = unique_dates[-1]
    start_gap_days = (actual_start - req_start).days
    end_gap_days = (req_end - actual_end).days
    start_covered = actual_start <= req_start + timedelta(days=tolerance_days)
    end_covered = actual_end >= req_end - timedelta(days=tolerance_days)

    return success_response(
        {
            "symbol": symbol,
            "mode": mode,
            "requested_start": req_start.isoformat(),
            "requested_end": req_end.isoformat(),
            "actual_start": actual_start.isoformat(),
            "actual_end": actual_end.isoformat(),
            "bars": len(unique_dates),
            "duplicates_removed_for_audit": len(dates) - len(unique_dates),
            "start_gap_days": start_gap_days,
            "end_gap_days": end_gap_days,
            "start_covered": start_covered,
            "end_covered": end_covered,
            "full_range_covered": start_covered and end_covered,
            "warning": None
            if start_covered
            else (
                "KIS returned data later than the requested start. "
                "Do not treat this as a full-period backtest."
            ),
        },
        message="KIS daily-history coverage measured from the actual API response.",
    )


async def audit_kis_history(
    symbols: List[str],
    requested_start: str,
    end_date: Optional[str] = None,
    mode: Literal["live", "paper"] = "live",
    tolerance_days: int = 10,
) -> Dict[str, Any]:
    """Probe multiple symbols sequentially to stay friendly to KIS rate limits."""
    if not symbols:
        return error_response("symbols is empty")

    rows: List[Dict[str, Any]] = []
    for symbol in symbols:
        response = await asyncio.to_thread(
            probe_kis_history,
            symbol,
            requested_start,
            end_date,
            mode,
            tolerance_days,
        )
        if response.get("success"):
            rows.append({"success": True, **_data(response)})
        else:
            rows.append({"success": False, "symbol": symbol, "error": response.get("error")})

    covered = [r for r in rows if r.get("success") and r.get("full_range_covered")]
    earliest_dates = [r.get("actual_start") for r in rows if r.get("success") and r.get("actual_start")]

    return success_response(
        {
            "requested_start": requested_start,
            "requested_end": end_date or date.today().isoformat(),
            "mode": mode,
            "all_symbols_fully_covered": len(covered) == len(symbols),
            "fully_covered_count": len(covered),
            "symbol_count": len(symbols),
            "earliest_returned_date": min(earliest_dates) if earliest_dates else None,
            "results": rows,
        },
        message="History audit complete. Long backtests should proceed only when coverage is explicit.",
    )


# ---------------------------------------------------------------------------
# Walk-forward research
# ---------------------------------------------------------------------------


def summarize_walk_forward(windows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [w for w in windows if w.get("status") == "completed"]
    if not completed:
        return {
            "completed_windows": 0,
            "positive_window_rate": 0.0,
            "median_oos_return": 0.0,
            "median_oos_sharpe": 0.0,
            "worst_oos_drawdown": 0.0,
            "robustness_score": 0.0,
            "score_note": "Heuristic score; no completed OOS windows.",
        }

    returns = [float(w.get("oos_metrics", {}).get("total_return", 0.0) or 0.0) for w in completed]
    sharpes = [float(w.get("oos_metrics", {}).get("sharpe_ratio", 0.0) or 0.0) for w in completed]
    drawdowns = [abs(float(w.get("oos_metrics", {}).get("max_drawdown", 0.0) or 0.0)) for w in completed]

    positive_rate = sum(1 for value in returns if value > 0) / len(returns)
    median_return = statistics.median(returns)
    median_sharpe = statistics.median(sharpes)
    worst_drawdown = max(drawdowns) if drawdowns else 0.0

    # LEAN statistics in this project are percentage points (e.g. 12.5 == 12.5%).
    # This score is intentionally transparent and secondary to the raw OOS metrics.
    consistency_component = 35.0 * positive_rate
    sharpe_component = 25.0 * max(0.0, min(median_sharpe / 2.0, 1.0))
    drawdown_component = 20.0 * max(0.0, 1.0 - min(worst_drawdown / 50.0, 1.0))

    if len(returns) >= 2:
        dispersion = statistics.pstdev(returns)
        stability_ratio = abs(median_return) / (dispersion + abs(median_return) + 1e-9)
    else:
        stability_ratio = 0.5
    stability_component = 20.0 * max(0.0, min(stability_ratio, 1.0))

    return {
        "completed_windows": len(completed),
        "positive_window_rate": round(positive_rate, 4),
        "median_oos_return": round(median_return, 6),
        "median_oos_sharpe": round(median_sharpe, 6),
        "worst_oos_drawdown": round(worst_drawdown, 6),
        "robustness_score": round(
            consistency_component + sharpe_component + drawdown_component + stability_component,
            2,
        ),
        "score_note": (
            "Heuristic 0-100 research score: 35% positive-window consistency, "
            "25% median OOS Sharpe, 20% worst OOS drawdown resilience, "
            "20% OOS-return stability. Always inspect raw windows too."
        ),
    }


async def run_walk_forward(
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
) -> Dict[str, Any]:
    """Optimize on each training window and evaluate frozen params on the next window."""
    try:
        windows = build_walk_forward_windows(
            start_date=start_date,
            end_date=end_date,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            max_windows=max_windows,
        )
    except Exception as exc:
        return error_response(f"walk-forward window build failed: {exc}")

    if not windows:
        return error_response("No valid walk-forward windows for the requested range.")

    results: List[Dict[str, Any]] = []

    for index, window in enumerate(windows, start=1):
        row: Dict[str, Any] = {"window": index, **window.as_dict()}

        optimize_response = run_optimize(
            strategy_id=strategy_id,
            symbols=symbols,
            start_date=window.train_start.isoformat(),
            end_date=window.train_end.isoformat(),
            parameters=parameters,
            search_type=search_type,
            max_samples=max_samples,
            target=target,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            tax_rate=tax_rate,
            seed=None if seed is None else seed + index,
            slippage=slippage,
        )
        optimize_payload = _data(optimize_response)
        optimize_job_id = optimize_payload.get("job_id")
        if not optimize_response.get("success") or not optimize_job_id:
            row.update(status="failed", stage="train_optimize", error=optimize_response.get("error"))
            results.append(row)
            continue

        optimize_wait = await get_backtest_result_wait(optimize_job_id, timeout=timeout_per_stage)
        optimize_result = _result_from_wait_response(optimize_wait)
        best_params = optimize_result.get("best_params") if optimize_result else None
        if not optimize_wait.get("success") or not isinstance(best_params, dict):
            row.update(
                status="failed",
                stage="train_optimize",
                optimize_job_id=optimize_job_id,
                error=optimize_wait.get("error") or "optimizer returned no best_params",
            )
            results.append(row)
            continue

        row["optimize_job_id"] = optimize_job_id
        row["best_params"] = best_params
        row["train_best_metrics"] = optimize_result.get("best_metrics")

        oos_submit = run_preset_backtest(
            strategy_id=strategy_id,
            symbols=symbols,
            start_date=window.test_start.isoformat(),
            end_date=window.test_end.isoformat(),
            initial_capital=initial_capital,
            param_overrides=best_params,
            commission_rate=commission_rate,
            tax_rate=tax_rate,
            slippage=slippage,
        )
        oos_job_id = _data(oos_submit).get("job_id")
        if not oos_submit.get("success") or not oos_job_id:
            row.update(status="failed", stage="oos_submit", error=oos_submit.get("error"))
            results.append(row)
            continue

        oos_wait = await get_backtest_result_wait(oos_job_id, timeout=timeout_per_stage)
        oos_result = _result_from_wait_response(oos_wait)
        if not oos_wait.get("success") or not oos_result:
            row.update(
                status="failed",
                stage="oos_test",
                oos_job_id=oos_job_id,
                error=oos_wait.get("error") or "OOS result unavailable",
            )
            results.append(row)
            continue

        row.update(
            status="completed",
            stage="done",
            oos_job_id=oos_job_id,
            oos_metrics={
                "total_return": _metric(oos_result, "total_return"),
                "annual_return": _metric(oos_result, "annual_return"),
                "max_drawdown": _metric(oos_result, "max_drawdown"),
                "sharpe_ratio": _metric(oos_result, "sharpe_ratio"),
                "sortino_ratio": _metric(oos_result, "sortino_ratio"),
                "win_rate": _metric(oos_result, "win_rate"),
                "profit_loss_ratio": _metric(oos_result, "profit_loss_ratio"),
            },
        )
        results.append(row)

    summary = summarize_walk_forward(results)
    return success_response(
        {
            "strategy_id": strategy_id,
            "symbols": symbols,
            "configuration": {
                "start_date": start_date,
                "end_date": end_date,
                "train_months": train_months,
                "test_months": test_months,
                "step_months": step_months,
                "search_type": search_type,
                "max_samples": max_samples,
                "target": target,
                "commission_rate": commission_rate,
                "tax_rate": tax_rate,
                "slippage": slippage,
            },
            "summary": summary,
            "windows": results,
        },
        message=(
            "Walk-forward complete. Parameters were selected only on each training window "
            "and then frozen for the following OOS window."
        ),
    )


# ---------------------------------------------------------------------------
# Cost / slippage stress
# ---------------------------------------------------------------------------


async def run_cost_stress(
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
) -> Dict[str, Any]:
    scenarios = slippage_scenarios or [0.0, 0.0005, 0.001, 0.002]
    scenarios = sorted(set(float(v) for v in scenarios))
    if any(v < 0 for v in scenarios):
        return error_response("slippage_scenarios must be >= 0")

    async def run_one(slippage: float) -> Dict[str, Any]:
        submitted = run_preset_backtest(
            strategy_id=strategy_id,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            param_overrides=param_overrides or {},
            commission_rate=commission_rate,
            tax_rate=tax_rate,
            slippage=slippage,
        )
        job_id = _data(submitted).get("job_id")
        if not submitted.get("success") or not job_id:
            return {"slippage": slippage, "status": "failed", "error": submitted.get("error")}

        waited = await get_backtest_result_wait(job_id, timeout=timeout)
        result = _result_from_wait_response(waited)
        if not waited.get("success") or not result:
            return {
                "slippage": slippage,
                "job_id": job_id,
                "status": "failed",
                "error": waited.get("error") or "result unavailable",
            }
        return {
            "slippage": slippage,
            "job_id": job_id,
            "status": "completed",
            "total_return": _metric(result, "total_return"),
            "annual_return": _metric(result, "annual_return"),
            "max_drawdown": _metric(result, "max_drawdown"),
            "sharpe_ratio": _metric(result, "sharpe_ratio"),
        }

    rows = await asyncio.gather(*(run_one(s) for s in scenarios))
    completed = [r for r in rows if r.get("status") == "completed"]

    degradation = None
    if len(completed) >= 2:
        lowest_cost = min(completed, key=lambda r: r["slippage"])
        highest_cost = max(completed, key=lambda r: r["slippage"])
        degradation = {
            "from_slippage": lowest_cost["slippage"],
            "to_slippage": highest_cost["slippage"],
            "return_change": highest_cost["total_return"] - lowest_cost["total_return"],
            "sharpe_change": highest_cost["sharpe_ratio"] - lowest_cost["sharpe_ratio"],
        }

    return success_response(
        {
            "strategy_id": strategy_id,
            "symbols": symbols,
            "param_overrides": param_overrides or {},
            "scenarios": rows,
            "degradation": degradation,
        },
        message="Transaction-cost/slippage stress test complete.",
    )
