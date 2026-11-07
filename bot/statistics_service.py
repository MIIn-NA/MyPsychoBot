"""Агрегированные отчёты для психолога без исходных сообщений студентов.

Проверку роли психолога выполняет Telegram-обработчик до вызова get_summary()
или get_report(), то есть до запроса к Storage. Сам сервис не авторизует
пользователя. Отправку отчёта также выполняет вызывающий компонент.
"""

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO
from threading import Lock

if __package__:
    from .storage import Storage, _parse_iso_date
else:
    from storage import Storage, _parse_iso_date


CLASS_NAMES = ("Норма", "Стресс", "Тревога")
_CHART_LOCK = Lock()


class StatisticsDataError(ValueError):
    """Сохранённая запись содержит некорректные данные для статистики."""


@dataclass(frozen=True)
class DailyAggregate:
    """Количество анализов по классам 0, 1, 2 за день с наблюдениями."""

    analysis_date: str
    counts: tuple[int, int, int]

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def percentages(self) -> tuple[float, float, float]:
        return tuple(count / self.total * 100 for count in self.counts)


@dataclass(frozen=True)
class GroupAggregates:
    counts: tuple[int, int, int]
    daily: tuple[DailyAggregate, ...]

    @property
    def total(self) -> int:
        return sum(self.counts)


@dataclass(frozen=True)
class GroupReport:
    summary: str
    chart_png: bytes | None
    daily_aggregates: tuple[DailyAggregate, ...]


def aggregate_results(results: Iterable[dict]) -> GroupAggregates:
    """Чистый расчёт: каждая запись — один анализ, включая повторные за день.

    Не использует совет, уверенность и идентификаторы студентов. Дни без
    записей не получают искусственные оценки или проценты.
    """
    counts = [0, 0, 0]
    days = {}
    for record in results:
        class_id = record.get("predicted_class")
        if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id not in (0, 1, 2):
            raise StatisticsDataError("Некорректный класс в сохранённых результатах: ожидается 0, 1 или 2.")
        day = record.get("analysis_date")
        try:
            _parse_iso_date(day, "Результат: analysis_date")
        except ValueError:
            raise StatisticsDataError(
                "Некорректная analysis_date в сохранённых результатах: ожидается реальная дата YYYY-MM-DD."
            ) from None
        daily_counts = days.setdefault(day, [0, 0, 0])
        daily_counts[class_id] += 1
        counts[class_id] += 1
    return GroupAggregates(
        counts=tuple(counts),
        daily=tuple(DailyAggregate(day, tuple(days[day])) for day in sorted(days)),
    )


def _format_summary(group_name: int, start_date: str, end_date: str, data: GroupAggregates) -> str:
    if not data.total:
        return f"За выбранный период данных нет. Группа {group_name}: {start_date} — {end_date}."
    lines = [
        f"Статистика группы {group_name} за период {start_date} — {end_date}",
        f"Число анализов: {data.total}",
        "По классам: число анализов (доля анализов)",
    ]
    for name, count in zip(CLASS_NAMES, data.counts):
        lines.append(f"{name}: {count} ({count / data.total * 100:.1f}%)")
    if len(data.daily) == 1:
        lines.append("Наблюдения есть только за один день; сравнивать динамику пока не с чем.")
    return "\n".join(lines)


def _chart_series(daily: tuple[DailyAggregate, ...]) -> tuple[list[date], list[list[float]]]:
    """Подготовить дневные доли; NaN разрывает линии между днями с пропусками."""
    dates = []
    series = [[], [], []]
    previous = None
    for item in daily:
        day = date.fromisoformat(item.analysis_date)
        if previous is not None and (day - previous).days > 1:
            dates.append(previous + timedelta(days=1))
            for values in series:
                values.append(float("nan"))
        dates.append(day)
        for values, percentage in zip(series, item.percentages):
            values.append(percentage)
        previous = day
    return dates, series


def _render_chart(group_name: int, start_date: str, end_date: str,
                  daily: tuple[DailyAggregate, ...]) -> bytes:
    # Matplotlib загружается только для непустого отчёта с изображением.
    # Agg и объектный API работают без GUI и не создают глобальные pyplot-фигуры.
    with _CHART_LOCK:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.dates import AutoDateLocator, DateFormatter, date2num
        from matplotlib.figure import Figure

        dates, series = _chart_series(daily)

        figure = Figure(figsize=(9, 5), dpi=120, layout="constrained")
        canvas = FigureCanvasAgg(figure)
        try:
            axes = figure.subplots()
            for name, values, color, marker in zip(
                CLASS_NAMES, series, ("#198754", "#ca8a04", "#b83b5e"), ("o", "s", "^")
            ):
                axes.plot(dates, values, marker=marker, label=name, color=color, clip_on=False)
            axes.set_title(f"Группа {group_name}: динамика состояний\n{start_date} — {end_date}")
            axes.set_xlabel("Дата анализа; разрывы — дни без данных")
            axes.set_ylabel("Доля анализов за день, %")
            axes.set_ylim(0, 100)
            axes.set_yticks(range(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
            start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
            if start == end:
                axes.set_xlim(date2num(start) - .5, date2num(end) + .5)
            else:
                # Небольшой отступ сохраняет маркеры на обеих границах целиком.
                axes.set_xlim(date2num(start) - .2, date2num(end) + .2)
            if start == end:
                axes.set_xticks([date2num(start)])
            else:
                axes.xaxis.set_major_locator(AutoDateLocator(minticks=3, maxticks=7))
            axes.xaxis.set_major_formatter(DateFormatter("%d.%m.%Y"))
            axes.tick_params(axis="x", labelrotation=25)
            axes.grid(alpha=.25)
            figure.legend(loc="outside lower center", ncols=3)
            with BytesIO() as output:
                canvas.print_png(output)
                return output.getvalue()
        finally:
            figure.clear()


class StatisticsService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    async def _get_aggregates(self, group_name: int, start_date: str, end_date: str) -> GroupAggregates:
        if isinstance(group_name, bool) or not isinstance(group_name, int) or group_name <= 0:
            raise ValueError("group_name должен быть положительным целым числом.")
        start = _parse_iso_date(start_date, "start_date")
        end = _parse_iso_date(end_date, "end_date")
        if start > end:
            raise ValueError("start_date должна быть не позже end_date.")
        # Ошибки чтения проходят вызывающему коду, не превращаясь в пустую выборку.
        results = await self.storage.get_results(group_name, start_date, end_date)
        return aggregate_results(results)

    async def get_summary(self, group_name: int, start_date: str, end_date: str) -> str:
        """Составить текстовую сводку, не загружая библиотеку отрисовки."""
        data = await self._get_aggregates(group_name, start_date, end_date)
        return _format_summary(group_name, start_date, end_date, data)

    async def get_report(self, group_name: int, start_date: str, end_date: str) -> GroupReport:
        """Текст и PNG по одной выборке; при отсутствии данных PNG равен None.

        Telegram-обработчик обязан проверить роль психолога до вызова метода.
        Доступность этого API не означает, что групповой сценарий подключён.
        Ошибки запроса, чтения и отрисовки передаются вызывающему коду.
        """
        data = await self._get_aggregates(group_name, start_date, end_date)
        summary = _format_summary(group_name, start_date, end_date, data)
        chart = None
        if data.total:
            chart = await asyncio.to_thread(_render_chart, group_name, start_date, end_date, data.daily)
        return GroupReport(summary=summary, chart_png=chart, daily_aggregates=data.daily)
