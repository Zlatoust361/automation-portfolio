"""
Watchdog вокруг вызова CV-инференса.

ЗАЧЕМ

На симуляторе инференс всегда возвращается. На железе — нет. Модель на Jetson
может уйти в своп при нехватке unified memory, TensorRT-контекст может
подвиснуть на GPU, USB-камера — отвалиться посреди чтения. Во всех этих случаях
`await detect(frame)` просто не завершается.

Без watchdog это выглядит так: деталь стоит в зоне контроля, конвейер ждёт
вердикта, вердикта нет, оператор смотрит на неподвижную линию. Худший вариант —
конвейер продолжает движение, а необработанные детали уезжают дальше как годные.

С watchdog: через `timeout_s` вызов снимается, деталь получает REJECT, конвейер
уходит в безопасный останов, в журнал ложится предупреждение. Процесс при этом
НЕ падает — упавший управляющий процесс на линии хуже остановленной линии,
потому что после него никто не выключит привод.

ПОЧЕМУ REJECT, А НЕ MANUAL_REVIEW

Соблазнительно отправить деталь человеку: мы же не знаем, годная она или нет.
Но MANUAL_REVIEW — это утверждение «модель посмотрела и засомневалась», а здесь
модель не посмотрела вовсе. Деталь, проехавшая без контроля, должна быть
отбракована: разбраковать годное дешевле, чем выпустить непроверенное.

ЧЕГО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ

`asyncio.wait_for` отменяет задачу, но отменяемость — свойство самой задачи.
Синхронный инференс, запущенный через `asyncio.to_thread`, при таймауте
продолжает крутиться в своём потоке: у CPython нет способа убить поток извне.
Управление мы возвращаем вовремя и линию останавливаем правильно, но поток
живёт до конца вызова. Поэтому:

* `guard_sync` считает такие «повисшие» вызовы (`abandoned`) — по этому счётчику
  видно, что сервис не просто медленный, а копит зависшие потоки;
* пул потоков ограничен (`max_workers`), и когда все они заняты зависшими
  вызовами, следующий вызов не ставится в очередь, а сразу отвечает таймаутом.
  Иначе он ждал бы освобождения потока, отчитывался бы таймаутом по чужой
  причине и маскировал бы настоящую — что инференс-сервис мёртв;
* защёлка (`latch`) не даёт линии продолжать работу, пока их не разберут.

Настоящее решение — вынести инференс в отдельный процесс и убивать его. Это
неделя 11, где сервисы уже разделены. Здесь честный компромисс с явными
границами, а не иллюзия управления.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    # Self появился в typing только в 3.11, а JetPack 6 несёт 3.10.
    # Аннотации отложены (from __future__ import annotations), поэтому в
    # рантайме этот импорт не выполняется и на плате ничего не ломает.
    from typing import Self

logger = logging.getLogger(__name__)


class WatchdogOutcome(Enum):
    """Чем закончился охраняемый вызов."""

    OK = "OK"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class WatchdogConfigError(ValueError):
    """Параметры watchdog заданы неверно."""


@dataclass(frozen=True)
class WatchdogConfig:
    """Настройки. Заполняется из секции `watchdog` конфига недели."""

    timeout_s: float = 0.35
    failures_to_fault: int = 3
    latch: bool = True
    # Сколько синхронных инференсов может выполняться одновременно. Больше двух
    # не нужно: вызовы идут по одному, второй поток — это запас на один
    # брошенный. Когда заняты все, сервис уже точно нездоров.
    max_workers: int = 2

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise WatchdogConfigError(f"timeout_s={self.timeout_s} должен быть больше нуля")
        if self.failures_to_fault < 1:
            raise WatchdogConfigError(
                f"failures_to_fault={self.failures_to_fault} должен быть не меньше 1"
            )
        if self.max_workers < 1:
            raise WatchdogConfigError(f"max_workers={self.max_workers} должен быть не меньше 1")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> WatchdogConfig:
        known = {f: config[f] for f in cls.__dataclass_fields__ if f in config}
        return cls(**known)


@dataclass(frozen=True)
class WatchdogResult:
    """
    Результат охраняемого вызова.

    `value` есть только при `OK`. При таймауте и ошибке решение принимает
    вызывающая сторона, а не watchdog: он сообщает, что случилось, и требует
    безопасного останова, но вердикт по детали формирует `line_controller` —
    там же, где живут пороги приёмки недели 4.
    """

    outcome: WatchdogOutcome
    duration_s: float
    value: Any = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome is WatchdogOutcome.OK

    @property
    def timed_out(self) -> bool:
        return self.outcome is WatchdogOutcome.TIMEOUT

    @property
    def safe_stop_required(self) -> bool:
        """Любой неуспех означает безопасный останов, а не только таймаут."""
        return self.outcome is not WatchdogOutcome.OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "duration_s": round(self.duration_s, 4),
            "error": self.error,
        }


@dataclass
class WatchdogStats:
    """Счётчики за время работы — их читает бенчмарк и health-check."""

    calls: int = 0
    timeouts: int = 0
    errors: int = 0
    consecutive_failures: int = 0
    abandoned: int = 0
    total_duration_s: float = 0.0
    max_duration_s: float = 0.0
    durations: list[float] = field(default_factory=list)

    @property
    def failures(self) -> int:
        return self.timeouts + self.errors

    @property
    def mean_duration_s(self) -> float:
        return (self.total_duration_s / self.calls) if self.calls else 0.0

    def percentile_s(self, share: float) -> float:
        """
        Перцентиль длительности.

        Для watchdog важен именно хвост: среднее в 40 мс при p99 в 900 мс — это
        сервис, который регулярно не укладывается в таймаут, а по среднему
        выглядит здоровым.
        """
        if not self.durations:
            return 0.0
        ordered = sorted(self.durations)
        index = min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))
        return ordered[index]

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "timeouts": self.timeouts,
            "errors": self.errors,
            "abandoned": self.abandoned,
            "consecutive_failures": self.consecutive_failures,
            "mean_duration_s": round(self.mean_duration_s, 4),
            "p95_duration_s": round(self.percentile_s(0.95), 4),
            "max_duration_s": round(self.max_duration_s, 4),
        }


class SafetyWatchdog:
    """
    Обёртка с таймаутом вокруг инференса.

    Не бросает исключений наружу ни при каких обстоятельствах: и таймаут, и
    падение модели, и отмена возвращаются как `WatchdogResult`. Вызывающий код —
    цикл управления линией — обязан продолжать работать при любом поведении
    AI-сервиса.
    """

    def __init__(self, config: WatchdogConfig | None = None) -> None:
        self._config = config or WatchdogConfig()
        self._stats = WatchdogStats()
        self._latched = False

        # Свой пул, а не общий по умолчанию: нужно точно знать, сколько потоков
        # заняты зависшими вызовами, и не делить их с чужим кодом.
        self._executor = ThreadPoolExecutor(
            max_workers=self._config.max_workers, thread_name_prefix="watchdog-infer"
        )
        self._busy = 0
        self._busy_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- состояние --------------------------------------------------------

    @property
    def config(self) -> WatchdogConfig:
        return self._config

    @property
    def stats(self) -> WatchdogStats:
        return self._stats

    @property
    def latched(self) -> bool:
        """
        Защёлка сработала: линия не должна продолжать работу.

        Снимается только `reset()` — то есть осознанным действием оператора или
        перезапуском сервиса. Автоматически снимать нельзя: единственный признак
        того, что AI-сервис починился, — это что кто-то его починил.
        """
        return self._latched

    @property
    def busy_workers(self) -> int:
        """Сколько потоков сейчас занято — включая зависшие с прошлых вызовов."""
        with self._busy_lock:
            return self._busy

    @property
    def degraded(self) -> bool:
        """Отказов подряд накопилось столько, что сервис считается нерабочим."""
        return self._stats.consecutive_failures >= self._config.failures_to_fault

    def reset(self) -> None:
        """Сбросить защёлку и счётчик отказов подряд. Историю замеров не трогает."""
        if self._latched:
            logger.info("Watchdog: защёлка снята оператором")
        self._latched = False
        self._stats.consecutive_failures = 0

    # --- охраняемый вызов -------------------------------------------------

    async def guard(
        self,
        call: Callable[[], Awaitable[Any]] | Awaitable[Any],
        description: str = "инференс",
    ) -> WatchdogResult:
        """
        Выполнить корутину под таймаутом.

        Принимает и корутину, и фабрику корутин. Фабрика предпочтительнее: при
        повторном вызове корутину нельзя переиспользовать, а фабрику можно.
        """
        awaitable = call() if callable(call) else call
        started = time.monotonic()

        try:
            value = await asyncio.wait_for(awaitable, timeout=self._config.timeout_s)
        except asyncio.TimeoutError:
            return self._record_timeout(time.monotonic() - started, description)
        except asyncio.CancelledError:
            # Отмену снаружи (остановка сервиса) не проглатываем: это не отказ
            # инференса, а штатное завершение, и оно должно дойти до цикла.
            raise
        except Exception as exc:  # noqa: BLE001 — модель может упасть как угодно
            return self._record_error(time.monotonic() - started, description, exc)

        return self._record_ok(time.monotonic() - started, value)

    async def guard_sync(
        self,
        call: Callable[[], Any],
        description: str = "инференс",
    ) -> WatchdogResult:
        """
        То же для обычной синхронной функции — а инференс почти всегда такой.

        Вызов уезжает в собственный пул потоков. Ограничение из докстринга
        модуля действует здесь в полную силу: при таймауте поток продолжает
        работу, и такие вызовы считаются в `stats.abandoned`.
        """
        started = time.monotonic()

        if not self._reserve_worker():
            # Все потоки заняты зависшими вызовами. Ставить задачу в очередь
            # нельзя: она дождалась бы освобождения потока и отчиталась
            # таймаутом по чужой причине, спрятав настоящую.
            return self._record_timeout(
                time.monotonic() - started,
                description,
                note=f"все {self._config.max_workers} рабочих потоков заняты зависшими вызовами",
            )

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, call)
        future.add_done_callback(self._release_worker)

        try:
            # shield — чтобы wait_for не пытался отменить задачу, которую всё
            # равно не отменить: поток из CPython снаружи не убить.
            value = await asyncio.wait_for(asyncio.shield(future), timeout=self._config.timeout_s)
        except asyncio.TimeoutError:
            self._stats.abandoned += 1
            future.add_done_callback(self._note_abandoned_finished)
            return self._record_timeout(time.monotonic() - started, description)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:  # noqa: BLE001
            return self._record_error(time.monotonic() - started, description, exc)

        return self._record_ok(time.monotonic() - started, value)

    def run(self, call: Callable[[], Any], description: str = "инференс") -> WatchdogResult:
        """
        Синхронная точка входа для цикла управления.

        Цикл линии написан обычным синхронным кодом — так его проще читать
        наладчику, и так он ближе к тому, как устроен сканер ПЛК.

        Здесь НЕ используется `asyncio.run`, и это принципиально: при выходе он
        вызывает `shutdown_default_executor()`, который ЖДЁТ завершения потоков.
        То есть на брошенном по таймауту инференсе `run()` возвращал бы
        управление не через `timeout_s`, а через столько, сколько модель
        провисит, — и весь watchdog не работал бы ровно в том случае, ради
        которого написан. Цикл поднимается один раз и живёт вместе с watchdog.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "run() вызван изнутри событийного цикла — используйте await guard_sync()"
            )

        return self._own_loop().run_until_complete(self.guard_sync(call, description))

    def close(self) -> None:
        """
        Отпустить цикл и пул.

        `wait=False` намеренно: если поток завис на инференсе, ждать его при
        остановке сервиса значит не остановиться вовсе.
        """
        if self._loop is not None:
            self._loop.close()
            self._loop = None
        self._executor.shutdown(wait=False)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- пул и цикл -------------------------------------------------------

    def _own_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _reserve_worker(self) -> bool:
        with self._busy_lock:
            if self._busy >= self._config.max_workers:
                return False
            self._busy += 1
            return True

    def _release_worker(self, _future: Any) -> None:
        with self._busy_lock:
            self._busy = max(0, self._busy - 1)

    # --- учёт -------------------------------------------------------------

    def _record_ok(self, duration: float, value: Any) -> WatchdogResult:
        self._note(duration)
        self._stats.consecutive_failures = 0
        return WatchdogResult(WatchdogOutcome.OK, duration, value=value)

    def _record_timeout(
        self, duration: float, description: str, note: str = ""
    ) -> WatchdogResult:
        self._note(duration)
        self._stats.timeouts += 1
        self._stats.consecutive_failures += 1
        logger.warning(
            "Watchdog: %s не уложился в %.3f с%s (отказов подряд: %d) — "
            "деталь в брак, конвейер в безопасный останов",
            description, self._config.timeout_s,
            f" [{note}]" if note else "",
            self._stats.consecutive_failures,
        )
        self._maybe_latch()
        error = f"таймаут {self._config.timeout_s:.3f} с"
        return WatchdogResult(
            WatchdogOutcome.TIMEOUT, duration, error=f"{error}: {note}" if note else error
        )

    def _record_error(self, duration: float, description: str, exc: BaseException) -> WatchdogResult:
        self._note(duration)
        self._stats.errors += 1
        self._stats.consecutive_failures += 1
        logger.warning(
            "Watchdog: %s упал (%s: %s), отказов подряд: %d — "
            "деталь в брак, конвейер в безопасный останов",
            description, type(exc).__name__, exc, self._stats.consecutive_failures,
        )
        self._maybe_latch()
        return WatchdogResult(
            WatchdogOutcome.ERROR, duration, error=f"{type(exc).__name__}: {exc}"
        )

    def _note(self, duration: float) -> None:
        self._stats.calls += 1
        self._stats.total_duration_s += duration
        self._stats.max_duration_s = max(self._stats.max_duration_s, duration)
        self._stats.durations.append(duration)

    def _maybe_latch(self) -> None:
        if self._config.latch and self.degraded and not self._latched:
            self._latched = True
            logger.error(
                "Watchdog: %d отказов подряд — линия защёлкнута до сброса оператором",
                self._stats.consecutive_failures,
            )

    @staticmethod
    def _note_abandoned_finished(task: asyncio.Future) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("Брошенный по таймауту вызов завершился ошибкой: %s", exc)
        else:
            logger.info("Брошенный по таймауту вызов всё-таки завершился")
