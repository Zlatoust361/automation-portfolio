"""
Тесты PID — воспроизводят два утверждения README числами, а не словами:

1. Антивиндап: после 1000 тактов в насыщении интеграл остаётся малым,
   а без защиты уходит на порядки выше.
2. Производная по измерению: скачок уставки не даёт derivative kick.

Плюс контринтуитивный факт про D на апериодическом объекте первого
порядка — он тоже проверяется прогоном, а не принимается на слово.
"""

from __future__ import annotations

import math

import pytest

from pid_controller import PIDConfig, PIDController, PIDError


def _unprotected_integral(kp: float, ki: float, error: float, dt: float, steps: int) -> float:
    """Интегратор без антивиндапа — для сравнения."""
    integral = 0.0
    for _ in range(steps):
        integral += error * dt
    return integral


def test_config_rejects_negative_gains_and_bad_limits() -> None:
    with pytest.raises(PIDError):
        PIDConfig(kp=-1.0)
    with pytest.raises(PIDError):
        PIDConfig(kp=1.0, min_output=10.0, max_output=10.0)
    with pytest.raises(PIDError):
        PIDConfig(kp=math.nan)


def test_config_from_dict_requires_kp_and_ignores_unknown_keys() -> None:
    cfg = PIDConfig.from_config({"kp": 2.0, "ki": 0.5, "ramp_rate": 10})
    assert (cfg.kp, cfg.ki, cfg.kd) == (2.0, 0.5, 0.0)
    with pytest.raises(PIDError):
        PIDConfig.from_config({"ki": 0.5})


def test_step_rejects_nan_and_nonpositive_dt() -> None:
    pid = PIDController(PIDConfig(kp=1.0))
    with pytest.raises(PIDError):
        pid.step(10.0, math.nan, 0.1)
    with pytest.raises(PIDError):
        pid.step(10.0, 0.0, 0.0)


def test_output_is_clamped_to_actuator_limits() -> None:
    pid = PIDController(PIDConfig(kp=100.0, min_output=0.0, max_output=50.0))
    assert pid.step(setpoint=10.0, value=0.0, dt=0.1) == 50.0
    assert pid.saturated
    assert pid.step(setpoint=-10.0, value=0.0, dt=0.1) == 0.0


def test_antiwindup_keeps_integral_bounded_under_saturation() -> None:
    """Критерий из README: 1000 тактов в насыщении — интеграл < 100,
    без защиты > 50000. Проверяем оба числа."""
    kp, ki, dt, steps = 1.0, 1.0, 0.1, 1000
    error = 600.0  # измерение далеко от уставки, выход упёрся в предел

    pid = PIDController(PIDConfig(kp=kp, ki=ki, min_output=0.0, max_output=100.0))
    for _ in range(steps):
        pid.step(setpoint=error, value=0.0, dt=dt)

    assert pid.saturated
    assert pid.integral < 100.0
    assert _unprotected_integral(kp, ki, error, dt, steps) > 50000.0


def test_integral_resumes_when_error_changes_sign_in_saturation() -> None:
    """Замораживаем интегратор только когда ошибка толкает дальше в
    насыщение; при смене знака он должен двигаться обратно."""
    pid = PIDController(PIDConfig(kp=1.0, ki=1.0, min_output=0.0, max_output=10.0))
    for _ in range(200):
        pid.step(setpoint=5.0, value=0.0, dt=0.1)  # kp*5 + ki*I упирается в 10 при I=5
    frozen_integral = pid.integral
    assert pid.saturated
    assert frozen_integral == pytest.approx(5.0, abs=0.2)

    pid.step(setpoint=5.0, value=6.0, dt=0.1)  # ошибка стала отрицательной, выход внутри пределов
    assert pid.integral < frozen_integral


def test_no_derivative_kick_on_setpoint_change() -> None:
    """Производная берётся по измерению: скачок уставки при неподвижном
    процессе меняет выход только на kp * Δsetpoint, без выброса от D."""
    kp, kd = 1.0, 50.0
    pid = PIDController(PIDConfig(kp=kp, ki=0.0, kd=kd, min_output=-1e6, max_output=1e6))
    pid.step(setpoint=0.0, value=0.0, dt=0.1)
    before = pid.step(setpoint=0.0, value=0.0, dt=0.1)
    after = pid.step(setpoint=10.0, value=0.0, dt=0.1)
    assert after - before == pytest.approx(kp * 10.0)


def test_derivative_reacts_to_measurement_change() -> None:
    pid = PIDController(PIDConfig(kp=0.0, ki=0.0, kd=1.0, min_output=-1e6, max_output=1e6))
    pid.step(setpoint=0.0, value=0.0, dt=0.1)
    out = pid.step(setpoint=0.0, value=1.0, dt=0.1)
    assert out == pytest.approx(-10.0)  # -kd * dvalue/dt


def test_reset_clears_integral_and_history() -> None:
    pid = PIDController(PIDConfig(kp=1.0, ki=1.0))
    for _ in range(10):
        pid.step(setpoint=5.0, value=0.0, dt=0.1)
    assert pid.integral > 0
    pid.reset()
    assert pid.integral == 0.0
    assert pid.state.steps == 0
    assert pid.state.last_value is None


def _simulate_first_order(kd: float, *, steps: int = 400) -> float:
    """Апериодический объект первого порядка: tau * dy/dt = u - y.
    Возвращает максимальное перерегулирование относительно уставки."""
    tau, dt, setpoint = 1.0, 0.01, 1.0
    pid = PIDController(PIDConfig(kp=2.0, ki=4.0, kd=kd, min_output=0.0, max_output=3.0))
    y = 0.0
    peak = 0.0
    for _ in range(steps):
        u = pid.step(setpoint, y, dt)
        y += dt * (u - y) / tau
        peak = max(peak, y)
    return peak - setpoint


def test_counterintuitive_fact_d_does_not_reduce_overshoot_on_first_order_plant() -> None:
    """Утверждение README: на объекте первого порядка D замедляет подъём,
    интеграл копится дольше — перерегулирование не меньше, а больше."""
    overshoot_without_d = _simulate_first_order(kd=0.0)
    overshoot_with_d = _simulate_first_order(kd=0.5)
    assert overshoot_without_d > 0  # вообще есть перерегулирование от I
    assert overshoot_with_d > overshoot_without_d
