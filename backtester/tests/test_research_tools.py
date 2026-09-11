from kis_mcp.tools.research import build_walk_forward_windows, summarize_walk_forward


def test_walk_forward_windows_do_not_leak():
    windows = build_walk_forward_windows(
        start_date="2005-01-01",
        end_date="2012-12-31",
        train_months=36,
        test_months=12,
        step_months=12,
        max_windows=10,
    )

    assert windows
    for window in windows:
        assert window.train_start <= window.train_end
        assert window.train_end < window.test_start
        assert window.test_start <= window.test_end

    assert windows[0].train_start.isoformat() == "2005-01-01"
    assert windows[0].test_start.isoformat() == "2008-01-01"


def test_walk_forward_respects_max_windows():
    windows = build_walk_forward_windows(
        start_date="2000-01-01",
        end_date="2025-12-31",
        train_months=60,
        test_months=12,
        step_months=12,
        max_windows=3,
    )
    assert len(windows) == 3


def test_robustness_summary_prefers_consistent_oos():
    strong = summarize_walk_forward(
        [
            {"status": "completed", "oos_metrics": {"total_return": 12, "sharpe_ratio": 1.4, "max_drawdown": 10}},
            {"status": "completed", "oos_metrics": {"total_return": 10, "sharpe_ratio": 1.2, "max_drawdown": 12}},
            {"status": "completed", "oos_metrics": {"total_return": 8, "sharpe_ratio": 1.0, "max_drawdown": 15}},
        ]
    )
    weak = summarize_walk_forward(
        [
            {"status": "completed", "oos_metrics": {"total_return": 30, "sharpe_ratio": 1.6, "max_drawdown": 40}},
            {"status": "completed", "oos_metrics": {"total_return": -25, "sharpe_ratio": -0.4, "max_drawdown": 45}},
            {"status": "completed", "oos_metrics": {"total_return": -10, "sharpe_ratio": 0.0, "max_drawdown": 50}},
        ]
    )

    assert strong["positive_window_rate"] == 1.0
    assert strong["robustness_score"] > weak["robustness_score"]
