"""
PID-регулятор тока/напряжения сварки.

Чистый Python (только stdlib): регулятор не знает ни про Modbus, ни про БД, ни
про безопасность. В неделе 7 тот же класс будет управлять реальным источником
питания без единой правки.

Две вещи, которые отличают рабочий PID от учебного и здесь сделаны намеренно:

* **Антивиндап.** Когда выход упёрся в предел, интегральная составляющая
  продолжала бы расти и накопила бы «долг», который потом придётся отрабатывать
  перерегулированием. Интегратор замораживается, пока выход в насыщении и
  ошибка толкает его дальше в ту же сторону.

* **Производная по измерению, а не по ошибке.** При скачке уставки производная
  ошибки даёт мгновенный выброс (derivative kick) — для сварочного источника это
  бросок тока. Производная берётся от измеренного значения, поэтому смена
  уставки на неё не влияет.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class PIDError(ValueError):
    """Некорректные параметры или аргументы регулятора."""


@dataclass(frozen=True)
class PIDConfig:
    """
    Настройки регулятора.

    `min_output` / `max_output` — физические пределы исполнительного механизма
    (для сварочного источника это диапазон задания тока). Регулятор обязан их
    знать, иначе антивиндап невозможен в принципе.
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    min_output: float = 0.0
    max_output: float = 100.0

    def __post_init__(self) -> None:
        for name in ("kp", "ki", "kd"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise PIDError(f"{name} должен быть конечным числом, получено {value}")
            if value < 0:
                raise PIDError(
                    f"{name}={value}: отрицательные коэффициенты меняют знак обратной "
                    f"связи и превращают регулятор в генератор"
                )
        if not math.isfinite(self.min_output) or not math.isfinite(self.max_output):
            raise PIDError("Пределы выхода должны быть конечными")
        if self.min_output >= self.max_output:
            raise PIDError(
                f"min_output={self.min_output} >= max_output={self.max_output}"
            )

    @classmethod
    def from_config(cls, config: dict) -> PIDConfig:
        """Собрать из секции конфига, игнорируя посторонние ключи."""
        known = {f: config[f] for f in cls.__dataclass_fields__ if f in config}
        if "kp" not in known:
            raise PIDError("В конфиге PID нет обязательного kp")
        return cls(**known)

    @property
    def output_span(self) -> float:
        return self.max_output - self.min_output


@dataclass
class PIDState:
    """Внутреннее состояние — вынесено отдельно, чтобы его можно было осмотреть."""

    integral: float = 0.0
    last_value: float | None = None
    last_error: float = 0.0
    last_output: float = 0.0
    saturated: bool = False
    steps: int = field(default=0)


class PIDController:
    """Классический PID с ограничением выхода и антивиндапом."""

    def __init__(self, config: PIDConfig) -> None:
        self._config = config
        self._state = PIDState()

    @property
    def config(self) -> PIDConfig:
        return self._config

    @property
    def state(self) -> PIDState:
        return self._state

    @property
    def integral(self) -> float:
        return self._state.integral

    @property
    def saturated(self) -> bool:
        """Выход упёрся в предел на прошлом такте."""
        return self._state.saturated

    def reset(self) -> None:
        """
        Сбросить историю.

        Вызывать при запуске цикла и после аварийной остановки: иначе накопленный
        интеграл прошлого шва даст бросок тока на первом же такте следующего.
        """
        self._state = PIDState()

    def step(self, setpoint: float, value: float, dt: float) -> float:
        """
        Один такт регулирования: уставка и измерение -> команда на выход.

        `dt` — время с прошлого такта в секундах, передаётся снаружи. Регулятор
        не читает часы сам: так он детерминирован в тестах и одинаково работает
        в симуляции с ускоренным временем и на реальной линии.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise PIDError(f"dt={dt}: шаг должен быть положительным конечным числом")
        if not math.isfinite(setpoint) or not math.isfinite(value):
            # NaN с датчика не должен молча проходить в регулятор: сравнения с
            # NaN всегда ложны, и ошибка стала бы NaN, отравив интегратор навсегда.
            raise PIDError(f"Нечисловые данные: setpoint={setpoint}, value={value}")

        config = self._config
        state = self._state

        error = setpoint - value

        proportional = config.kp * error

        # Производная по измерению со знаком минус: d(error)/dt = -d(value)/dt
        # при постоянной уставке, а при её смене выброса не возникает.
        if state.last_value is None:
            derivative = 0.0
        else:
            derivative = -config.kd * (value - state.last_value) / dt

        # Антивиндап: интегрируем, только если это не загоняет выход глубже
        # в насыщение. Проверяем на пробном значении, не трогая состояние.
        candidate_integral = state.integral + error * dt
        unclamped = proportional + config.ki * candidate_integral + derivative
        clamped = self._clamp(unclamped)

        pushing_further = (
            (unclamped > config.max_output and error > 0)
            or (unclamped < config.min_output and error < 0)
        )
        if not pushing_further:
            state.integral = candidate_integral
            output = clamped
        else:
            # Интеграл замораживаем и пересчитываем выход на старом значении.
            frozen = proportional + config.ki * state.integral + derivative
            output = self._clamp(frozen)

        state.saturated = output != unclamped
        state.last_value = value
        state.last_error = error
        state.last_output = output
        state.steps += 1
        return output

    def _clamp(self, value: float) -> float:
        return min(self._config.max_output, max(self._config.min_output, value))
