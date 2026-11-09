"""Проверки настоящей SQLite во временных каталогах, без Telegram и модели."""
import ast
import datetime
import hashlib
import importlib.util
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from bot import storage as storage_module
from bot.storage import DEFAULT_DB_PATH, Storage, StorageError, StorageInitializationError


class StorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'nested' / 'results.db'
        self.storage = Storage(self.path)

    async def save(self, user_id=10, group=1, day='2026-11-07', label=2):
        await self.storage.save_result(user_id, group, day, label, .85, 'Совет')

    async def test_default_path_does_not_depend_on_working_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.directory.name)
            storage = Storage()
        finally:
            os.chdir(previous)
        expected = Path(storage_module.__file__).resolve().parent.parent / 'db' / 'results.db'
        self.assertEqual(storage.storage_file, expected)
        self.assertEqual(DEFAULT_DB_PATH, expected)

    async def test_invalid_values_rejected_before_opening_database(self):
        valid = dict(user_id=10, group=1, analysis_date='2024-02-29',
                     predicted_class=2, model_confidence=.85, advice='Совет')
        invalid = {
            'user_id': (True, False, 0, -1, 1.5, '10', None),
            'group': (True, False, 0, -1, 1.5, '1', None),
            'analysis_date': (None, 20261107, '', '2026-02-29', '2024-02-30',
                              '2026-13-01', '20261107', '2026-W45-6', '2026-1-01',
                              '2026-11-07T00:00:00', ' 2026-11-07'),
            'predicted_class': (True, False, -1, 3, 1.0, '1', None),
            'model_confidence': (True, False, -.01, 1.01, float('nan'),
                                 float('inf'), float('-inf'), '0.8', None),
            'advice': (None, 123, '', ' \t\n '),
        }
        with patch.object(storage_module.aiosqlite, 'connect') as connect:
            for field, values in invalid.items():
                for value in values:
                    with self.subTest(field=field, value=value):
                        with self.assertRaisesRegex(ValueError, f'save_result: {field}'):
                            await self.storage.save_result(**(valid | {field: value}))
            connect.assert_not_called()

    async def test_multiple_daily_analyses_preserve_values_and_precision(self):
        await self.storage.init_table()
        advice = "Совет с кавычкой '); DROP TABLE results; --"
        for label, confidence in ((0, 0), (1, .6000000000000001), (2, 1)):
            self.assertIsNone(await self.storage.save_result(
                10, 7, '2024-02-29', label, confidence, advice
            ))
        # Независимое соединение видит только подтверждённые записи.
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                'SELECT user_id, group_name, analysis_date, predicted_class, '
                'model_confidence, advice FROM results ORDER BY id'
            ).fetchall()
        self.assertEqual(rows, [
            (10, 7, '2024-02-29', label, confidence, advice)
            for label, confidence in ((0, 0), (1, .6000000000000001), (2, 1))
        ])

    async def test_new_storage_instance_reads_both_daily_analyses(self):
        await self.storage.init_table()
        await self.storage.save_result(10, 7, '2026-11-07', 1, .8123456789012345, 'Первый совет')
        await self.storage.save_result(10, 7, '2026-11-07', 2, .95, 'Второй совет')
        reopened = Storage(self.path)
        await reopened.init_table()
        rows = await reopened.get_results(7, '2026-11-07', '2026-11-07')
        self.assertEqual(rows, [
            dict(id=1, group_name=7, analysis_date='2026-11-07', predicted_class=1,
                 model_confidence=.8123456789012345, advice='Первый совет'),
            dict(id=2, group_name=7, analysis_date='2026-11-07', predicted_class=2,
                 model_confidence=.95, advice='Второй совет'),
        ])
        self.assertEqual(await reopened.get_user_class_counts(10), {1: 1, 2: 1})

    async def test_invalid_operations_leave_existing_database_unchanged(self):
        await self.storage.init_table()
        await self.save()
        before = await self.storage.get_results(1, '2026-11-01', '2026-11-30')
        valid = dict(user_id=10, group=1, analysis_date='2026-11-07',
                     predicted_class=2, model_confidence=.85, advice='Совет')
        for invalid in ({'analysis_date': '2026-02-29'}, {'predicted_class': 3},
                        {'model_confidence': float('nan')}, {'model_confidence': 1.01}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                await self.storage.save_result(**(valid | invalid))
        with self.assertRaises(ValueError):
            await self.storage.get_results(1, '2026-11-08', '2026-11-07')
        self.assertEqual(await self.storage.get_results(1, '2026-11-01', '2026-11-30'), before)

    async def test_insert_failure_hides_sqlite_message_and_private_data(self):
        await self.storage.init_table()
        secret = 'PRIVATE_STUDENT_ADVICE_987654321'
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TRIGGER reject_result BEFORE INSERT ON results BEGIN "
                f"SELECT RAISE(ABORT, '{secret}'); END"
            )
        with self.assertRaisesRegex(StorageError, r'save_result \(INSERT\).*ограничение') as caught:
            await self.storage.save_result(987654321, 1, '2026-11-07', 2, .85, secret)
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn('987654321', str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertEqual(await self.storage.get_user_class_counts(987654321), {})

    async def test_commit_failure_is_not_reported_as_success_and_rolls_back(self):
        await self.storage.init_table()
        connection = storage_module.aiosqlite.connect(self.path)
        with patch.object(storage_module.aiosqlite, 'connect', return_value=connection), \
                patch.object(connection, 'commit', new=AsyncMock(
                    side_effect=sqlite3.OperationalError('PRIVATE_COMMIT_DETAILS')
                )) as commit:
            with self.assertRaisesRegex(StorageError, r'save_result \(commit\)') as caught:
                await self.save()
            commit.assert_awaited_once()
        self.assertNotIn('PRIVATE_COMMIT_DETAILS', str(caught.exception))
        self.assertEqual(await self.storage.get_user_class_counts(10), {})

    async def test_open_failure_reports_operation_without_record_data(self):
        with self.assertRaisesRegex(StorageError, r'save_result \(connect\).*открыть'):
            await self.save()

    async def test_init_creates_parent_and_repeat_init_preserves_results(self):
        self.assertFalse(self.path.parent.exists())
        await self.storage.init_table()
        await self.save()
        await self.storage.init_table()
        self.assertEqual(await self.storage.get_user_class_counts(10), {2: 1})
        rows = await self.storage.get_results(1, '2026-11-07', '2026-11-07')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['advice'], 'Совет')
        self.assertEqual(rows[0]['model_confidence'], .85)

    async def test_same_storage_filters_user_group_and_inclusive_period(self):
        await self.storage.init_table()
        await self.save(label=0, day='2026-11-01')
        await self.save(label=1, day='2026-11-07')
        await self.save(user_id=20, label=2, day='2026-11-08')
        await self.save(user_id=20, group=2, label=2, day='2026-11-07')
        await self.save(label=2, day='2026-10-31')
        await self.save(user_id=30, day='2026-11-09')
        self.assertEqual(await self.storage.get_user_class_counts(10), {0: 1, 1: 1, 2: 1})
        self.assertEqual(await self.storage.get_user_class_counts(20), {2: 2})
        self.assertEqual(await self.storage.get_user_class_counts(999), {})
        rows = await self.storage.get_results(1, '2026-11-01', '2026-11-08')
        self.assertEqual([row['predicted_class'] for row in rows], [0, 1, 2])
        self.assertEqual(await self.storage.get_results(99, '2026-11-01', '2026-11-08'), [])
        # Существующий сервис использует тот же экземпляр, без новой базы.
        spec = importlib.util.spec_from_file_location('statistics_service_test', 'bot/statistics_service.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'storage': storage_module}):
            spec.loader.exec_module(module)
        service = module.StatisticsService(self.storage)
        self.assertIs(service.storage, self.storage)
        summary = await service.get_summary(1, '2026-11-01', '2026-11-08')
        self.assertIn('Число анализов: 3', summary)
        self.assertIn('данных нет', await service.get_summary(99, '2026-11-01', '2026-11-08'))

    async def test_group_dates_validated_before_opening_database(self):
        with patch.object(storage_module.aiosqlite, 'connect') as connect:
            for field in ('start_date', 'end_date'):
                for invalid in (None, '', '2026-02-29', '20261107',
                                '2026-W45-6', '2026-11-7', '2026-11-07 '):
                    arguments = dict(group=1, start_date='2026-11-07', end_date='2026-11-07')
                    arguments[field] = invalid
                    with self.subTest(field=field, value=invalid):
                        with self.assertRaisesRegex(ValueError, f'get_results: {field}'):
                            await self.storage.get_results(**arguments)
            with self.assertRaisesRegex(ValueError, 'start_date.*end_date'):
                await self.storage.get_results(1, '2026-11-08', '2026-11-07')
            connect.assert_not_called()

    async def test_group_rows_keep_keys_and_sort_by_date_then_id(self):
        await self.storage.init_table()
        await self.save(day='2024-03-01', label=2)
        await self.save(day='2024-02-29', label=1)
        await self.save(day='2024-02-29', label=0)
        rows = await self.storage.get_results(1, '2024-02-29', '2024-03-01')
        self.assertEqual([row['id'] for row in rows], [2, 3, 1])
        self.assertEqual(set(rows[0]), {
            'id', 'group_name', 'analysis_date', 'predicted_class', 'model_confidence', 'advice'
        })
        self.assertEqual(len(await self.storage.get_results(1, '2024-02-29', '2024-02-29')), 2)

    async def test_read_errors_are_not_returned_as_empty_statistics(self):
        await self.storage.init_table()
        with sqlite3.connect(self.path) as db:
            db.execute('DROP TABLE results')
        with self.assertRaises(sqlite3.OperationalError):
            await self.storage.get_results(1, '2026-11-01', '2026-11-07')
        with self.assertRaises(sqlite3.OperationalError):
            await self.storage.get_user_class_counts(10)

    async def test_previous_personal_statistics_method_remains_compatible(self):
        await self.storage.init_table()
        await self.save()
        self.assertEqual(await self.storage.get_user_stats(10), {2: 1})

    async def test_index_added_to_existing_compatible_database(self):
        await self.storage.init_table()
        await self.save()
        with sqlite3.connect(self.path) as db:
            db.execute('DROP INDEX idx_results_group_date')
        await self.storage.init_table()
        with sqlite3.connect(self.path) as db:
            columns = [row[2] for row in db.execute('PRAGMA index_info(idx_results_group_date)')]
            self.assertEqual(columns, ['group_name', 'analysis_date'])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM results').fetchone()[0], 1)

    async def test_legacy_schema_rejected_without_migration_or_data_changes(self):
        self.path.parent.mkdir()
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE results (id INTEGER PRIMARY KEY, user_id INTEGER, confidence REAL)')
            db.execute('INSERT INTO results VALUES (1, 10, .85)')
        before = hashlib.sha256(self.path.read_bytes()).digest()
        with self.assertRaisesRegex(StorageInitializationError, 'Несовместимая схема.*group_name'):
            await self.storage.init_table()
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).digest(), before)

    async def test_invalid_file_and_parent_have_clear_initialization_errors(self):
        self.path.parent.mkdir()
        self.path.write_text('not a database')
        with self.assertRaisesRegex(StorageInitializationError, 'Не удалось инициализировать базу'):
            await self.storage.init_table()
        with self.assertRaises(StorageInitializationError):
            await Storage(self.path / 'results.db').init_table()

    async def test_incompatible_primary_key_rejected_before_insert(self):
        self.path.parent.mkdir()
        for definition in ('id TEXT PRIMARY KEY', 'id INTEGER',
                           'id INTEGER, PRIMARY KEY (id, user_id)'):
            with self.subTest(definition=definition):
                with sqlite3.connect(self.path) as db:
                    db.execute('DROP TABLE IF EXISTS results')
                    db.execute(
                        'CREATE TABLE results (user_id INTEGER, group_name INTEGER, '
                        'analysis_date TEXT, predicted_class INTEGER, '
                        'model_confidence REAL, advice TEXT, ' + definition + ')'
                    )
                before = self.path.read_bytes()
                with self.assertRaisesRegex(StorageInitializationError, 'INTEGER PRIMARY KEY'):
                    await self.storage.init_table()
                self.assertEqual(self.path.read_bytes(), before)

    async def test_extra_required_column_rejected_without_changing_data(self):
        await self.storage.init_table()
        with sqlite3.connect(self.path) as db:
            db.execute('ALTER TABLE results ADD COLUMN source TEXT NOT NULL')
        before = self.path.read_bytes()
        with self.assertRaisesRegex(StorageInitializationError, 'обязательные колонки.*source'):
            await self.storage.init_table()
        self.assertEqual(self.path.read_bytes(), before)

    async def test_compatible_extra_columns_preserved(self):
        await self.storage.init_table()
        with sqlite3.connect(self.path) as db:
            db.execute('ALTER TABLE results ADD COLUMN note TEXT')
            db.execute("ALTER TABLE results ADD COLUMN source TEXT NOT NULL DEFAULT 'bot'")
        await self.storage.init_table()
        await self.save()
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute('SELECT note, source FROM results').fetchone(), (None, 'bot'))


class StorageIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def handlers(self):
        tree = ast.parse(Path('bot/main_bot.py').read_text())
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef) and node.name in ('cmd_my_stats', 'show_statistics', 'main'):
                node.decorator_list = []
                nodes.append(node)
        namespace = dict(types=SimpleNamespace(Message=object, BufferedInputFile=Mock()), logging=Mock(),
                         DEFAULT_GROUP_ID=1, GROUP_REPORT_DAYS=30, datetime=datetime,
                         StorageInitializationError=StorageInitializationError,
                         get_main_keyboard=lambda: 'main',
                         dp=SimpleNamespace(start_polling=AsyncMock()), bot=object())
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<storage_handlers>', 'exec'), namespace)
        return namespace

    async def test_solver_result_is_saved_by_handler_to_same_storage(self):
        from test_dialogue import handlers
        from bot.knowledge_base import KnowledgeBase
        from bot.solver import Solver
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(Path(directory) / 'results.db')
            await storage.init_table()
            namespace = handlers()
            namespace['storage'] = storage
            model = Mock()
            model.predict.return_value = (2, .85)
            namespace['solver'] = Solver(model, KnowledgeBase())
            message = SimpleNamespace(from_user=SimpleNamespace(id=10),
                                      text='Я боюсь провалить экзамен', answer=AsyncMock())
            with patch.dict(os.environ, {'BOT_DEMO_MODE': '0'}):
                await namespace['handle_message'](message)
            rows = await storage.get_results(1, '2000-01-01', '2100-01-01')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['predicted_class'], 2)
            self.assertEqual(rows[0]['model_confidence'], .85)
            self.assertIn(rows[0]['advice'], message.answer.call_args.args[0])
            self.assertEqual(await storage.get_user_class_counts(10), {2: 1})

    async def test_personal_statistics_reads_real_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(Path(directory) / 'results.db')
            await storage.init_table()
            await storage.save_result(10, 1, '2026-11-07', 2, .85, 'Совет')
            await storage.save_result(20, 1, '2026-11-07', 0, .9, 'Другой совет')
            namespace = self.handlers()
            namespace['storage'] = storage
            for user_id, expected in ((10, 'Тревога: 1 (100%)'), (999, 'пока нет сохраненных результатов')):
                message = SimpleNamespace(from_user=SimpleNamespace(id=user_id), answer=AsyncMock())
                await namespace['cmd_my_stats'](message)
                self.assertIn(expected, message.answer.call_args.args[0])

    async def test_statistics_button_sends_personal_totals_and_group_chart(self):
        from io import BytesIO
        from PIL import Image
        from bot.statistics_service import StatisticsService

        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(Path(directory) / 'results.db')
            await storage.init_table()
            for user_id, group, day, label in (
                (10, 1, '2026-10-10', 0),  # Граница периода включена.
                (10, 1, '2026-10-09', 2),  # Только в личной статистике за всё время.
                (20, 1, '2026-11-08', 1),  # Только в групповой статистике.
                (30, 2, '2026-11-08', 2),  # Другая группа.
            ):
                await storage.save_result(user_id, group, day, label, .85, 'Личный совет')
            namespace = self.handlers()
            namespace['storage'] = storage
            namespace['stats_service'] = StatisticsService(storage)
            namespace['datetime'] = SimpleNamespace(
                date=SimpleNamespace(today=lambda: datetime.date(2026, 11, 8)),
                timedelta=datetime.timedelta,
            )
            message = SimpleNamespace(chat=SimpleNamespace(type='private'),
                                      from_user=SimpleNamespace(id=10),
                                      answer=AsyncMock(), answer_photo=AsyncMock())
            await namespace['show_statistics'](message)

            message.answer.assert_awaited_once()
            personal = message.answer.call_args.args[0]
            self.assertIn('Норма: 1 (50%)', personal)
            self.assertIn('Тревога: 1 (50%)', personal)
            self.assertIn('Стресс: 0 (0%)', personal)
            message.answer_photo.assert_awaited_once()
            summary = message.answer_photo.call_args.kwargs['caption']
            self.assertIn('2026-10-10 — 2026-11-08', summary)
            self.assertIn('Число анализов: 2', summary)
            self.assertIn('Стресс: 1 (50.0%)', summary)
            self.assertIn('Тревога: 0 (0.0%)', summary)
            self.assertNotIn('Личный совет', summary)
            photo = namespace['types'].BufferedInputFile
            self.assertIs(message.answer_photo.call_args.kwargs['photo'], photo.return_value)
            with Image.open(BytesIO(photo.call_args.args[0])) as chart:
                self.assertEqual(chart.format, 'PNG')
                chart.verify()

    async def test_statistics_button_keeps_empty_group_and_report_failure_distinct(self):
        from bot.statistics_service import StatisticsService

        namespace = self.handlers()
        storage = SimpleNamespace(
            get_user_class_counts=AsyncMock(return_value={0: 1}),
            get_results=AsyncMock(return_value=[]),
        )
        namespace.update(storage=storage, stats_service=StatisticsService(storage))
        message = SimpleNamespace(chat=SimpleNamespace(type='private'),
                                  from_user=SimpleNamespace(id=10),
                                  answer=AsyncMock(), answer_photo=AsyncMock())
        await namespace['show_statistics'](message)
        self.assertIn('Сохранено анализов: 1', message.answer.call_args_list[0].args[0])
        self.assertIn('За выбранный период данных нет', message.answer.call_args.args[0])
        message.answer_photo.assert_not_awaited()

        message.answer.reset_mock()
        storage.get_results.side_effect = sqlite3.OperationalError('PRIVATE_READ_DETAILS')
        await namespace['show_statistics'](message)
        self.assertIn('Сохранено анализов: 1', message.answer.call_args_list[0].args[0])
        self.assertIn('Не удалось сформировать общую статистику', message.answer.call_args.args[0])
        self.assertNotIn('PRIVATE_READ_DETAILS', str(namespace['logging'].error.call_args))
        message.answer_photo.assert_not_awaited()

    async def test_statistics_button_keeps_reports_in_private_chat(self):
        namespace = self.handlers()
        namespace['storage'] = SimpleNamespace(get_user_class_counts=AsyncMock())
        namespace['stats_service'] = SimpleNamespace(get_report=AsyncMock())
        message = SimpleNamespace(chat=SimpleNamespace(type='group'), answer=AsyncMock())
        await namespace['show_statistics'](message)
        self.assertIn('личный чат', message.answer.call_args.args[0])
        namespace['storage'].get_user_class_counts.assert_not_awaited()
        namespace['stats_service'].get_report.assert_not_awaited()

    async def test_real_sqlite_write_failure_still_delivers_solver_result(self):
        from test_dialogue import handlers
        from bot.knowledge_base import KnowledgeBase
        from bot.solver import Solver
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage(Path(directory) / 'results.db')
            await storage.init_table()
            with sqlite3.connect(storage.storage_file) as db:
                db.execute(
                    "CREATE TRIGGER reject_result BEFORE INSERT ON results BEGIN "
                    "SELECT RAISE(ABORT, 'PRIVATE_TRIGGER_DETAILS'); END"
                )
            namespace = handlers()
            namespace['storage'] = storage
            namespace['solver'] = Solver(Mock(predict=Mock(return_value=(2, .85))), KnowledgeBase())
            message = SimpleNamespace(from_user=SimpleNamespace(id=10),
                                      text='Боюсь экзамена', answer=AsyncMock())
            with patch.dict(os.environ, {'BOT_DEMO_MODE': '0'}):
                await namespace['handle_message'](message)
            reply = message.answer.call_args.args[0]
            self.assertIn('Тревога', reply)
            self.assertIn('85%', reply)
            self.assertIn('Совет:', reply)
            self.assertNotIn('сохран', reply.lower())
            self.assertEqual(await storage.get_user_class_counts(10), {})
            namespace['logging'].error.assert_called_once()
            self.assertNotIn('PRIVATE_TRIGGER_DETAILS', str(namespace['logging'].error.call_args))

    async def test_personal_statistics_read_failure_has_distinct_response(self):
        namespace = self.handlers()
        namespace['storage'] = SimpleNamespace(get_user_class_counts=AsyncMock(
            side_effect=sqlite3.OperationalError('PRIVATE_READ_DETAILS')
        ))
        message = SimpleNamespace(from_user=SimpleNamespace(id=10), answer=AsyncMock())
        await namespace['cmd_my_stats'](message)
        namespace['storage'].get_user_class_counts.assert_awaited_once_with(10)
        reply = message.answer.call_args.args[0]
        self.assertEqual(reply, 'Ошибка при чтении базы данных.')
        self.assertNotIn('PRIVATE_READ_DETAILS', str(namespace['logging'].error.call_args))

    async def test_incompatible_schema_prevents_polling(self):
        namespace = self.handlers()
        namespace['storage'] = SimpleNamespace(
            init_table=AsyncMock(side_effect=StorageInitializationError('Несовместимая схема'))
        )
        await namespace['main']()
        namespace['dp'].start_polling.assert_not_awaited()
        namespace['logging'].error.assert_called_once()

    async def test_initialization_precedes_polling(self):
        namespace = self.handlers()
        order = []
        async def initialize():
            order.append('init')
        async def poll(bot):
            order.append('poll')
        namespace['storage'] = SimpleNamespace(init_table=initialize)
        namespace['dp'].start_polling.side_effect = poll
        await namespace['main']()
        self.assertEqual(order, ['init', 'poll'])
