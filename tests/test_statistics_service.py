"""Проверки расчётов отдельно от PNG и интеграция с настоящим Storage."""
import sqlite3
import math
from datetime import date
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from bot.statistics_service import (
    DailyAggregate, GroupReport, StatisticsDataError, StatisticsService,
    aggregate_results, _chart_series,
)
from bot.storage import Storage


class AggregateTests(unittest.TestCase):
    def test_first_and_third_day_have_correct_shares_with_gap_on_second_day(self):
        data = aggregate_results([
            dict(analysis_date='2026-11-03', predicted_class=2),
            dict(analysis_date='2026-11-01', predicted_class=0),
            dict(analysis_date='2026-11-03', predicted_class=1),
            dict(analysis_date='2026-11-01', predicted_class=1),
        ])
        self.assertEqual(data.total, 4)
        self.assertEqual(data.counts, (1, 2, 1))
        self.assertEqual([day.percentages for day in data.daily], [(50, 50, 0), (0, 50, 50)])
        dates, series = _chart_series(data.daily)
        self.assertEqual(dates, [date(2026, 11, day) for day in (1, 2, 3)])
        self.assertEqual([values[0] for values in series], [50, 50, 0])
        self.assertTrue(all(math.isnan(values[1]) for values in series))
        self.assertEqual([values[2] for values in series], [0, 50, 50])

    def test_counts_daily_percentages_and_order(self):
        rows = [
            dict(analysis_date=day, predicted_class=label, advice='PRIVATE_ADVICE')
            for day, label in [('2026-11-07', 2), ('2026-11-01', 0),
                               ('2026-11-01', 1), ('2026-11-07', 2)]
        ]
        data = aggregate_results(rows)
        self.assertEqual(data.counts, (1, 1, 2))
        self.assertEqual(data.total, 4)
        self.assertEqual(data.daily, (
            DailyAggregate('2026-11-01', (1, 1, 0)),
            DailyAggregate('2026-11-07', (0, 0, 2)),
        ))
        self.assertEqual(data.daily[0].percentages, (50, 50, 0))
        self.assertEqual(data.daily[1].percentages, (0, 0, 100))
        self.assertNotIn('PRIVATE_ADVICE', repr(data))

    def test_empty_and_invalid_saved_data(self):
        self.assertEqual(aggregate_results([]).total, 0)
        self.assertEqual(aggregate_results([]).daily, ())
        for label in (True, False, 3, -1, '1', 1.0):
            with self.subTest(label=label), self.assertRaises(ValueError):
                aggregate_results([dict(analysis_date='2026-11-07', predicted_class=label)])
        with self.assertRaises(ValueError):
            aggregate_results([dict(analysis_date='2026-02-29', predicted_class=0)])

    def test_period_counts_weight_analyses_instead_of_averaging_daily_shares(self):
        rows = [dict(analysis_date='2026-11-01', predicted_class=0)]
        rows += [dict(analysis_date='2026-11-07', predicted_class=2) for _ in range(3)]
        data = aggregate_results(rows)
        self.assertEqual(data.counts, (1, 0, 3))
        self.assertEqual(data.total, 4)
        self.assertEqual(data.daily[0].percentages, (100, 0, 0))
        self.assertEqual(data.daily[1].percentages, (0, 0, 100))

    def test_daily_series_preserves_precision_zero_classes_and_gaps(self):
        daily = (DailyAggregate('2026-11-01', (1, 2, 0)),
                 DailyAggregate('2026-11-04', (0, 0, 1)))
        dates, series = _chart_series(daily)
        self.assertEqual(dates, [date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 4)])
        self.assertEqual(series[0][0], 1 / 3 * 100)
        self.assertEqual(series[1][0], 2 / 3 * 100)
        self.assertEqual(series[2][0], 0)
        self.assertEqual([values[-1] for values in series], [0, 0, 100])
        for values in series:
            self.assertTrue(math.isnan(values[1]))

    def test_aggregation_consumes_records_once(self):
        class SinglePass:
            def __init__(self):
                self.used = False

            def __iter__(self):
                if self.used:
                    raise AssertionError('Повторный проход по записям')
                self.used = True
                yield dict(analysis_date='2026-11-07', predicted_class=2)
                yield dict(analysis_date='2026-11-07', predicted_class=2)

        data = aggregate_results(SinglePass())
        self.assertEqual(data.total, 2)
        self.assertEqual(data.daily[0].counts, (0, 0, 2))


class StatisticsServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rows = [dict(analysis_date='2026-11-07', predicted_class=x) for x in (0, 1, 2, 2)]
        self.storage = SimpleNamespace(get_results=AsyncMock(return_value=self.rows))
        self.service = StatisticsService(self.storage)

    async def test_acceptance_four_analyses_produce_summary_and_real_png_from_one_read(self):
        from io import BytesIO
        from PIL import Image
        self.storage.get_results.return_value = [
            dict(analysis_date='2026-11-07', predicted_class=label)
            for label in (0, 1, 1, 2)
        ]
        report = await self.service.get_report(7, '2026-11-01', '2026-11-07')
        self.storage.get_results.assert_awaited_once_with(7, '2026-11-01', '2026-11-07')
        self.assertIsInstance(report.summary, str)
        self.assertIn('Число анализов: 4', report.summary)
        self.assertIn('Норма: 1 (25.0%)', report.summary)
        self.assertIn('Стресс: 2 (50.0%)', report.summary)
        self.assertIn('Тревога: 1 (25.0%)', report.summary)
        self.assertEqual(report.daily_aggregates[0].percentages, (25, 50, 25))
        with Image.open(BytesIO(report.chart_png)) as image:
            self.assertEqual(image.format, 'PNG')
            image.verify()

    async def test_report_selects_once_and_uses_same_aggregates_for_text_and_chart(self):
        with patch('bot.statistics_service._render_chart', return_value=b'PNG') as render:
            report = await self.service.get_report(7, '2026-11-01', '2026-11-07')
        self.assertIsInstance(report, GroupReport)
        self.storage.get_results.assert_awaited_once_with(7, '2026-11-01', '2026-11-07')
        self.assertIn('Число анализов: 4', report.summary)
        self.assertIn('Норма: 1 (25.0%)', report.summary)
        self.assertIn('Стресс: 1 (25.0%)', report.summary)
        self.assertIn('Тревога: 2 (50.0%)', report.summary)
        self.assertEqual(report.chart_png, b'PNG')
        render.assert_called_once_with(7, '2026-11-01', '2026-11-07', report.daily_aggregates)
        self.assertEqual(report.daily_aggregates[0].counts, (1, 1, 2))

    async def test_summary_matches_report_without_rendering(self):
        with patch('bot.statistics_service._render_chart', return_value=b'PNG') as render:
            summary = await self.service.get_summary(7, '2026-11-01', '2026-11-07')
            render.assert_not_called()
            report = await self.service.get_report(7, '2026-11-01', '2026-11-07')
        self.assertEqual(summary, report.summary)

    async def test_single_observed_day_has_distribution_and_no_dynamics_notice(self):
        for start, end in (('2026-11-07', '2026-11-07'), ('2026-01-01', '2026-12-31')):
            with self.subTest(start=start, end=end):
                summary = await self.service.get_summary(7, start, end)
                self.assertIn(f'группы 7 за период {start} — {end}', summary)
                self.assertIn('Число анализов: 4', summary)
                self.assertIn('сравнивать динамику пока не с чем', summary)
                positions = [summary.index(name + ':') for name in ('Норма', 'Стресс', 'Тревога')]
                self.assertEqual(positions, sorted(positions))
        self.storage.get_results.return_value = self.rows + [
            dict(analysis_date='2026-11-01', predicted_class=0)
        ]
        summary = await self.service.get_summary(7, '2026-11-01', '2026-11-07')
        self.assertNotIn('сравнивать динамику пока не с чем', summary)

    async def test_report_and_chart_receive_only_aggregates(self):
        private = dict(user_id=987654321, text='PRIVATE_TEXT', advice='PRIVATE_ADVICE',
                       model_confidence=.987654321)
        self.storage.get_results.return_value = [dict(row, **private) for row in self.rows]
        with patch('bot.statistics_service._render_chart', return_value=b'PNG') as render:
            report = await self.service.get_report(7, '2026-11-01', '2026-11-07')
        for value in ('987654321', 'PRIVATE_TEXT', 'PRIVATE_ADVICE', 'model_confidence', 'user_id'):
            self.assertNotIn(value, repr(report))
            self.assertNotIn(value, repr(render.call_args))
        self.assertEqual(report.daily_aggregates, (DailyAggregate('2026-11-07', (1, 1, 2)),))

    async def test_png_long_period_and_independent_figures(self):
        from io import BytesIO
        from PIL import Image
        from matplotlib.figure import Figure
        figures = []

        def make_figure(*args, **kwargs):
            figure = Figure(*args, **kwargs)
            figures.append(figure)
            return figure

        self.storage.get_results.return_value = [
            dict(analysis_date=day, predicted_class=label)
            for day, label in (('2020-01-01', 0), ('2023-03-01', 1), ('2026-12-31', 2))
        ]
        with patch('matplotlib.figure.Figure', side_effect=make_figure):
            for _ in range(2):
                report = await self.service.get_report(7, '2020-01-01', '2026-12-31')
                with Image.open(BytesIO(report.chart_png)) as image:
                    self.assertEqual(image.format, 'PNG')
                    self.assertEqual(image.size, (1080, 600))
                    image.verify()
        self.assertEqual(len(figures), 2)
        self.assertIsNot(figures[0], figures[1])
        self.assertTrue(all(not figure.axes for figure in figures))

    async def test_period_shares_use_period_total_and_format_only_at_output(self):
        self.storage.get_results.return_value = [dict(analysis_date='2026-11-01', predicted_class=0)]
        self.storage.get_results.return_value += [
            dict(analysis_date='2026-11-07', predicted_class=2) for _ in range(2)
        ]
        summary = await self.service.get_summary(7, '2026-11-01', '2026-11-07')
        self.assertIn('Число анализов: 3', summary)
        self.assertIn('доля анализов', summary)
        self.assertIn('Норма: 1 (33.3%)', summary)
        self.assertIn('Стресс: 0 (0.0%)', summary)
        self.assertIn('Тревога: 2 (66.7%)', summary)

    async def test_corrupt_record_is_data_error_for_both_methods(self):
        for record in (
            dict(analysis_date='2026-11-07', predicted_class=3),
            dict(analysis_date='2026-02-29', predicted_class=0),
            dict(analysis_date='20261107', predicted_class=0),
            dict(analysis_date=None, predicted_class=0),
            dict(predicted_class=0),
            dict(analysis_date='2026-11-07'),
        ):
            self.storage.get_results.return_value = [record]
            with patch('bot.statistics_service._render_chart') as render:
                for method in (self.service.get_summary, self.service.get_report):
                    with self.subTest(record=record, method=method.__name__):
                        with self.assertRaises(StatisticsDataError):
                            await method(7, '2026-11-01', '2026-11-07')
                render.assert_not_called()

    async def test_empty_results_do_not_render_or_create_percentages(self):
        self.storage.get_results.return_value = []
        with patch('bot.statistics_service._render_chart') as render:
            report = await self.service.get_report(7, '2026-11-01', '2026-11-07')
        render.assert_not_called()
        self.storage.get_results.assert_awaited_once()
        self.assertEqual(report.summary,
                         'За выбранный период данных нет. Группа 7: 2026-11-01 — 2026-11-07.')
        self.assertNotIn('%', report.summary)
        self.assertIsNone(report.chart_png)
        self.assertEqual(report.daily_aggregates, ())

    async def test_invalid_request_does_not_read_storage(self):
        arguments = [(x, '2026-11-01', '2026-11-07') for x in (True, False, 0, -1, 1.5, '7', None)]
        for field in (1, 2):
            for value in (None, '', '2026-02-29', '20261107', '2026-W45-6', '2026-11-7'):
                args = [7, '2026-11-01', '2026-11-07']
                args[field] = value
                arguments.append(tuple(args))
        arguments.append((7, '2026-11-08', '2026-11-07'))
        for method in (self.service.get_summary, self.service.get_report):
            for args in arguments:
                with self.subTest(method=method.__name__, args=args), self.assertRaises(ValueError):
                    await method(*args)
        self.storage.get_results.assert_not_awaited()

    async def test_read_failure_propagates_and_does_not_render(self):
        failure = sqlite3.OperationalError('read failed')
        self.storage.get_results.side_effect = failure
        with patch('bot.statistics_service._render_chart') as render:
            for method in (self.service.get_summary, self.service.get_report):
                with self.assertRaises(sqlite3.OperationalError) as caught:
                    await method(7, '2026-11-01', '2026-11-07')
                self.assertIs(caught.exception, failure)
        render.assert_not_called()

    async def test_render_failure_is_not_empty_report_or_second_read(self):
        with patch('bot.statistics_service._render_chart', side_effect=RuntimeError('render failed')):
            with self.assertRaisesRegex(RuntimeError, 'render failed'):
                await self.service.get_report(7, '2026-11-01', '2026-11-07')
        self.storage.get_results.assert_awaited_once()

    async def test_png_is_readable_for_one_day_and_gaps(self):
        from PIL import Image
        from io import BytesIO
        for rows in (self.rows, self.rows + [dict(analysis_date='2026-11-01', predicted_class=0)]):
            self.storage.get_results.return_value = rows
            report = await self.service.get_report(7, '2026-11-01', '2026-11-07')
            self.assertTrue(report.chart_png.startswith(b'\x89PNG\r\n\x1a\n'))
            with Image.open(BytesIO(report.chart_png)) as image:
                image.verify()
            self.assertNotIn('advice', report.summary)
        # Один день в запросе также должен формировать график.
        self.storage.get_results.return_value = self.rows
        report = await self.service.get_report(7, '2026-11-07', '2026-11-07')
        self.assertTrue(report.chart_png.startswith(b'\x89PNG'))

    async def test_real_storage_filters_groups_and_inclusive_period(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(Path(directory) / 'results.db')
            await storage.init_table()
            for group, day, label in ((7, '2026-10-31', 2), (7, '2026-11-01', 0),
                                      (7, '2026-11-07', 1), (7, '2026-11-08', 2),
                                      (8, '2026-11-07', 2)):
                await storage.save_result(10, group, day, label, .9, 'PRIVATE_ADVICE')
            with patch.object(storage, 'get_results', wraps=storage.get_results) as read:
                with patch('bot.statistics_service._render_chart', return_value=b'PNG'):
                    report = await StatisticsService(storage).get_report(7, '2026-11-01', '2026-11-07')
                read.assert_awaited_once_with(7, '2026-11-01', '2026-11-07')
            self.assertIn('Число анализов: 2', report.summary)
            self.assertEqual([item.counts for item in report.daily_aggregates], [(1, 0, 0), (0, 1, 0)])
            self.assertNotIn('PRIVATE_ADVICE', repr(report))

    async def test_sqlite_reports_for_two_groups_include_both_boundaries_and_real_png(self):
        from io import BytesIO
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(Path(directory) / 'nested' / 'results.db')
            await storage.init_table()
            for group in (7, 8):
                await storage.save_result(10, group, '2026-10-31', 0, .91, 'OUTSIDE_BEFORE')
                await storage.save_result(20, group, '2026-11-08', 1, .92, 'OUTSIDE_AFTER')
            for group, day, label in (
                (7, '2026-11-07', 2), (7, '2026-11-01', 0),
                (7, '2026-11-07', 1), (7, '2026-11-01', 1),
                (8, '2026-11-07', 2), (8, '2026-11-01', 2),
            ):
                await storage.save_result(987654321, group, day, label, .987654321, 'PRIVATE_ADVICE')

            service = StatisticsService(storage)
            cases = (
                (7, 4, [(1, 1, 0), (0, 1, 1)],
                 ('Норма: 1 (25.0%)', 'Стресс: 2 (50.0%)', 'Тревога: 1 (25.0%)')),
                (8, 2, [(0, 0, 1), (0, 0, 1)],
                 ('Норма: 0 (0.0%)', 'Стресс: 0 (0.0%)', 'Тревога: 2 (100.0%)')),
            )
            for group, total, daily_counts, class_lines in cases:
                with self.subTest(group=group):
                    with patch.object(storage, 'get_results', wraps=storage.get_results) as read:
                        report = await service.get_report(group, '2026-11-01', '2026-11-07')
                        read.assert_awaited_once_with(group, '2026-11-01', '2026-11-07')
                    self.assertIn(f'Число анализов: {total}', report.summary)
                    for line in class_lines:
                        self.assertIn(line, report.summary)
                    self.assertEqual([day.analysis_date for day in report.daily_aggregates],
                                     ['2026-11-01', '2026-11-07'])
                    self.assertEqual([day.counts for day in report.daily_aggregates], daily_counts)
                    with Image.open(BytesIO(report.chart_png)) as image:
                        self.assertEqual(image.format, 'PNG')
                        image.verify()
                    for secret in ('OUTSIDE_BEFORE', 'OUTSIDE_AFTER', 'PRIVATE_ADVICE', '987654321'):
                        self.assertNotIn(secret, repr(report))

    async def test_summary_does_not_import_matplotlib(self):
        code = '''
import asyncio, sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
from bot.statistics_service import StatisticsService
service = StatisticsService(SimpleNamespace(get_results=AsyncMock(return_value=[
    dict(analysis_date='2026-11-07', predicted_class=0)])))
asyncio.run(service.get_summary(1, '2026-11-07', '2026-11-07'))
assert 'matplotlib' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
