"""
Тесты watchdog — проверяют критерии приёмки из README, а не «код не падает».

Каждый тест соответствует одному утверждению архитектурного решения:
управление возвращается через timeout_s, а не через столько, сколько
висит инференс; процесс не падает ни при таймауте, ни при исключении
модели; зависшие потоки считаются отдельно; когда все потоки заняты —
немедленный таймаут, не очередь; защёлка снимается только оператором.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from safety_watchdog import (
    SafetyWatchdog,
    WatchdogConfig,
    WatchdogConfigError,
    WatchdogOutcome,
)


def test_config_rejects_nonpositive_timeout() -> None:
    with pytest.raises(WatchdogConfigError):
        WatchdogConfig(timeout_s=0)


def test_config_from_dict_ignores_unknown_keys() -> None:
    cfg = WatchdogConfig.from_config({"timeout_s": 0.5, "unrelated": 1})
    assert cfg.timeout_s == 0.5
    assert cfg.failures_to_fault == 3


def test_ok_call_returns_value_and_resets_consecutive_failures() -> None:
    with SafetyWatchdog(WatchdogConfig(timeout_s=0.2)) as wd:
        result = wd.run(lambda: "GOOD")
        assert result.ok
        assert result.value == "GOOD"
        assert not result.safe_stop_required
        assert wd.stats.calls == 1
        assert wd.stats.consecutive_failures == 0


def test_timeout_returns_control_within_timeout_not_within_inference_duration() -> None:
    """Главный критерий: при зависшем инференсе run() возвращается через
    timeout_s, а не через время самого инференса."""
    hang_s = 1.0
    timeout_s = 0.1
    finished = threading.Event()

    def hung_inference() -> str:
        time.sleep(hang_s)
        finished.set()
        return "LATE"

    with SafetyWatchdog(WatchdogConfig(timeout_s=timeout_s, max_workers=1)) as wd:
        started = time.monotonic()
        result = wd.run(hung_inference)
        elapsed = time.monotonic() - started

    assert result.timed_out
    assert result.safe_stop_required
    assert result.value is None
    # Запас: вернулись заметно раньше, чем закончился сам инференс.
    assert elapsed < hang_s / 2, f"вернулись через {elapsed:.3f} с — ждали зависший поток"
    assert wd.stats.abandoned == 1
    # Поток действительно продолжал жить после таймаута — это ограничение
    # платформы, которое модуль честно учитывает, а не прячет.
    assert finished.wait(hang_s * 2)


def test_model_exception_is_reported_not_raised() -> None:
    def broken_model() -> None:
        raise RuntimeError("CUDA context lost")

    with SafetyWatchdog(WatchdogConfig(timeout_s=0.2)) as wd:
        result = wd.run(broken_model)

    assert result.outcome is WatchdogOutcome.ERROR
    assert "CUDA context lost" in result.error
    assert result.safe_stop_required
    assert wd.stats.errors == 1


def test_all_workers_busy_gives_immediate_timeout_instead_of_queueing() -> None:
    release = threading.Event()

    def hang_until_released() -> None:
        release.wait(5.0)

    try:
        with SafetyWatchdog(WatchdogConfig(timeout_s=0.05, max_workers=1, latch=False)) as wd:
            first = wd.run(hang_until_released)
            assert first.timed_out
            assert wd.busy_workers == 1

            started = time.monotonic()
            second = wd.run(lambda: "never runs")
            elapsed = time.monotonic() - started

            assert second.timed_out
            assert "заняты" in second.error
            # Не ждали ни освобождения потока, ни даже полного таймаута.
            assert elapsed < 0.05
            assert wd.stats.abandoned == 1  # второй вызов потока не занимал
    finally:
        release.set()


def test_latch_engages_after_consecutive_failures_and_only_operator_resets() -> None:
    def broken() -> None:
        raise ValueError("boom")

    with SafetyWatchdog(WatchdogConfig(timeout_s=0.2, failures_to_fault=2)) as wd:
        wd.run(broken)
        assert not wd.latched
        wd.run(broken)
        assert wd.latched
        assert wd.degraded

        # Успешный вызов НЕ снимает защёлку сам по себе.
        ok = wd.run(lambda: "GOOD")
        assert ok.ok
        assert wd.latched

        wd.reset()
        assert not wd.latched
        assert wd.stats.consecutive_failures == 0
        # История замеров сохранена для бенчмарка.
        assert wd.stats.calls == 3


def test_latch_can_be_disabled_by_config() -> None:
    def broken() -> None:
        raise ValueError("boom")

    with SafetyWatchdog(WatchdogConfig(timeout_s=0.2, failures_to_fault=1, latch=False)) as wd:
        wd.run(broken)
        assert wd.degraded
        assert not wd.latched


def test_run_refuses_to_be_called_inside_event_loop() -> None:
    async def inside() -> None:
        with SafetyWatchdog() as wd:
            with pytest.raises(RuntimeError, match="guard_sync"):
                wd.run(lambda: None)

    asyncio.run(inside())


def test_guard_async_timeout_and_ok() -> None:
    async def scenario() -> None:
        with SafetyWatchdog(WatchdogConfig(timeout_s=0.05)) as wd:

            async def slow() -> str:
                await asyncio.sleep(1.0)
                return "LATE"

            async def fast() -> str:
                return "FAST"

            slow_result = await wd.guard(slow)
            assert slow_result.timed_out

            fast_result = await wd.guard(fast)
            assert fast_result.ok and fast_result.value == "FAST"

    asyncio.run(scenario())


def test_stats_percentile_reports_tail_not_mean() -> None:
    with SafetyWatchdog(WatchdogConfig(timeout_s=1.0)) as wd:
        for _ in range(9):
            wd.run(lambda: None)
        wd.run(lambda: time.sleep(0.05))

    stats = wd.stats
    assert stats.calls == 10
    assert stats.percentile_s(0.95) >= 0.04
    assert stats.mean_duration_s < stats.percentile_s(0.95)
    assert set(stats.as_dict()) >= {"calls", "timeouts", "abandoned", "p95_duration_s"}
