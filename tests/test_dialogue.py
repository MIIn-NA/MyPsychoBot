"""Проверяем настоящие обработчики без запуска Telegram и SQLite."""
import ast
import datetime
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from bot.errors import TextTooLongError
from bot.knowledge_base import KnowledgeBase
from bot.mock_model import MockModel
from bot.solver import Solver, SolverError
from bot.storage import StorageError


def handlers():
    tree = ast.parse(Path('bot/main_bot.py').read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in (
            'build_model', 'handle_message', 'handle_context_button', 'process_solver'
        ):
            node.decorator_list = []
            selected.append(node)
    namespace = dict(types=SimpleNamespace(Message=object), os=os,
                     datetime=datetime, logging=Mock(), user_states={},
                     DEFAULT_GROUP_ID=1,
                     SolverError=SolverError, TextTooLongError=TextTooLongError, StorageError=StorageError,
                     MockModel=MockModel, storage=SimpleNamespace(save_result=AsyncMock()),
                     get_main_keyboard=lambda: 'main', get_context_keyboard=lambda: 'context')
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<handlers>', 'exec'), namespace)
    return namespace


class DialogueTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'BOT_DEMO_MODE': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.ns = handlers()
        self.model = Mock()
        self.ns['solver'] = Solver(self.model, KnowledgeBase())

    async def send(self, text, button=False):
        message = SimpleNamespace(text=text, from_user=SimpleNamespace(id=1), answer=AsyncMock())
        await self.ns['handle_context_button' if button else 'handle_message'](message)
        return message.answer.call_args

    async def test_anxiety_example_matches_display_and_saved_result(self):
        self.model.predict.return_value = (2, .85)
        reply = await self.send('Я боюсь провалить экзамен')
        expected = KnowledgeBase().get_recommendation('Я боюсь провалить экзамен', 2)
        self.assertIn('Тревога', reply.args[0])
        self.assertIn('85%', reply.args[0])
        self.assertIn(expected, reply.args[0])
        saved = self.ns['storage'].save_result.call_args.kwargs
        self.assertEqual((saved['predicted_class'], saved['model_confidence'], saved['advice']),
                         (2, .85, expected))
        self.model.predict.assert_called_once_with('Я боюсь провалить экзамен')
        self.ns['storage'].save_result.assert_awaited_once()

    async def test_empty_initial_inputs_call_neither_model_nor_knowledge_base(self):
        kb = Mock()
        self.ns['solver'] = Solver(self.model, kb)
        for text in ('', '   ', None):
            with self.subTest(text=text):
                reply = await self.send(text)
                self.assertIn('Сообщение пустое', reply.args[0])
        self.model.predict.assert_not_called()
        kb.get_recommendation_with_metadata.assert_not_called()
        self.ns['storage'].save_result.assert_not_awaited()

    async def test_invalid_class_and_nan_never_reach_storage(self):
        kb = Mock()
        self.ns['solver'] = Solver(self.model, kb)
        for prediction in ((9, .85), (2, float('nan'))):
            with self.subTest(prediction=prediction):
                self.model.predict.return_value = prediction
                reply = await self.send('Описание состояния')
                self.assertIn('Анализ сейчас недоступен', reply.args[0])
                self.assertNotIn('Совет:', reply.args[0])
        kb.get_recommendation_with_metadata.assert_not_called()
        self.ns['storage'].save_result.assert_not_awaited()

    async def test_threshold_boundary_controls_clarification_and_storage(self):
        for confidence, preliminary in ((.59999, True), (.6, False)):
            with self.subTest(confidence=confidence):
                self.ns['user_states'].clear()
                self.ns['storage'].save_result.reset_mock()
                self.model.predict.return_value = (1, confidence)
                reply = await self.send('У меня завтра экзамен')
                self.assertEqual('Предварительная оценка' in reply.args[0], preliminary)
                self.assertEqual(self.ns['user_states'][1]['awaiting_clarification'], preliminary)
                if preliminary:
                    self.ns['storage'].save_result.assert_not_awaited()
                else:
                    self.ns['storage'].save_result.assert_awaited_once()

    async def test_first_preliminary_then_text_final_saved_once(self):
        self.model.predict.side_effect = [(2, .59), (1, .8)]
        first = await self.send('Я НЕ сплю!')
        self.assertIn('Предварительная оценка', first.args[0])
        self.assertEqual(first.kwargs['reply_markup'], 'context')
        self.ns['storage'].save_result.assert_not_awaited()
        self.assertEqual(self.ns['user_states'][1]['previous_text'], 'Я НЕ сплю!')
        final = await self.send('Завтра экзамен')
        self.model.predict.assert_called_with('Я НЕ сплю! Завтра экзамен')
        saved = self.ns['storage'].save_result.call_args.kwargs
        self.assertEqual(saved['predicted_class'], 1)
        self.assertEqual(saved['model_confidence'], .8)
        self.assertIn(saved['advice'], final.args[0])
        self.assertIn('Стресс', final.args[0])
        self.ns['storage'].save_result.assert_awaited_once()
        self.assertEqual(self.ns['user_states'][1]['previous_text'], '')

    async def test_button_is_topic_and_second_preliminary_allows_new_analysis(self):
        self.model.predict.side_effect = [(0, .3), (2, .4), (0, .9)]
        await self.send('Исходное описание')
        second = await self.send('Из-за отношений', button=True)
        self.model.predict.assert_called_with('Исходное описание Из-за отношений')
        self.assertIn('Предварительная оценка', second.args[0])
        self.assertIn('Тревога', second.args[0])
        self.assertIn('начать новый анализ', second.args[0])
        self.assertNotIn('Уточните описание', second.args[0])
        self.assertEqual(second.kwargs['reply_markup'], 'main')
        self.ns['storage'].save_result.assert_not_awaited()
        await self.send('Новый анализ')
        self.model.predict.assert_called_with('Новый анализ')
        self.ns['storage'].save_result.assert_awaited_once()

    async def test_cancel_discards_original_and_stale_button_does_not_analyze(self):
        self.model.predict.return_value = (1, .4)
        await self.send('Исходный текст')
        await self.send('Отмена', button=True)
        state = self.ns['user_states'][1]
        self.assertFalse(state['awaiting_clarification'])
        self.assertEqual(state['previous_text'], '')
        await self.send('Из-за отношений', button=True)
        self.model.predict.assert_called_once()
        self.model.predict.return_value = (0, .8)
        await self.send('Новое описание')
        self.model.predict.assert_called_with('Новое описание')

    async def test_empty_clarification_keeps_original_and_does_not_predict(self):
        self.model.predict.return_value = (1, .4)
        await self.send('Исходный текст')
        reply = await self.send('   ')
        self.assertIn('Сообщение пустое', reply.args[0])
        self.assertTrue(self.ns['user_states'][1]['awaiting_clarification'])
        self.assertEqual(self.ns['user_states'][1]['previous_text'], 'Исходный текст')
        self.model.predict.assert_called_once()

    async def test_vague_distress_asks_for_context_before_predicting_or_saving(self):
        self.model.predict.return_value = (1, .90)
        first = await self.send('Всё плохо!')
        self.assertIn('нелегко', first.args[0])
        self.assertEqual(first.kwargs['reply_markup'], 'context')
        self.model.predict.assert_not_called()
        self.ns['storage'].save_result.assert_not_awaited()

        final = await self.send('Из-за учебы (сессия)', button=True)
        self.model.predict.assert_called_once_with('Всё плохо! Из-за учебы (сессия)')
        self.assertIn('Стресс', final.args[0])
        self.assertEqual(final.kwargs['reply_markup'], 'main')
        self.ns['storage'].save_result.assert_awaited_once()

    async def test_vague_distress_and_all_context_buttons_in_demo_mode(self):
        self.ns['solver'] = Solver(MockModel(), KnowledgeBase())
        for topic, expected in (
            ('Из-за учебы (сессия)', 'Стресс'),
            ('Из-за отношений', 'Тревога'),
            ('Проблемы со сном / недосып', 'Тревога'),
            ('Будущее (работа, планы)', 'Тревога'),
            ('Просто выгорание', 'Стресс'),
        ):
            with self.subTest(topic=topic):
                first = await self.send('все плохо')
                self.assertEqual(first.kwargs['reply_markup'], 'context')
                final = await self.send(topic, button=True)
                self.assertIn(expected, final.args[0])

    async def test_predict_failure_and_unavailable_model(self):
        self.model.predict.side_effect = RuntimeError('internal secret')
        for solver in (self.ns['solver'], None):
            self.ns['solver'] = solver
            reply = await self.send('Описание')
            self.assertIn('Анализ сейчас недоступен', reply.args[0])
            self.assertNotIn('internal secret', reply.args[0])
        self.ns['storage'].save_result.assert_not_awaited()

    async def test_database_failure_still_shows_final_result(self):
        self.model.predict.return_value = (2, .6)
        self.ns['storage'].save_result.side_effect = RuntimeError('db failure')
        reply = await self.send('Боюсь экзамена')
        self.assertIn('Тревога', reply.args[0])
        self.assertIn('60%', reply.args[0])
        self.assertNotIn('сохран', reply.args[0].lower())
        self.ns['logging'].error.assert_called_once()
        self.assertEqual(self.ns['storage'].save_result.call_args.kwargs['model_confidence'], .6)

    async def test_storage_failure_is_logged_without_private_data(self):
        self.model.predict.return_value = (1, .85)
        failure = StorageError('save_result (commit): база занята.')
        self.ns['storage'].save_result.side_effect = failure
        reply = await self.send('PRIVATE_STUDENT_TEXT экзамен')
        saved = self.ns['storage'].save_result.call_args.kwargs
        self.assertIn('Стресс', reply.args[0])
        self.assertIn(saved['advice'], reply.args[0])
        self.assertNotIn('сохран', reply.args[0].lower())
        self.ns['logging'].error.assert_called_once_with(
            'Ошибка записи результата в SQLite: %s', failure
        )
        log_call = self.ns['logging'].error.call_args
        log_message = log_call.args[0] % log_call.args[1:]
        self.assertIn('commit', log_message)
        self.assertIn('база занята', log_message)
        self.assertNotIn('PRIVATE_STUDENT_TEXT', log_message)
        self.assertNotIn(saved['advice'], log_message)


class ModelModeTests(unittest.TestCase):
    def test_mock_requires_explicit_demo_mode(self):
        ns = handlers()
        classifier = SimpleNamespace(RuBertClassifier=Mock(side_effect=RuntimeError),
                                     ModelLoadError=RuntimeError)
        with patch.dict('sys.modules', {'classifier': classifier}):
            with patch.dict(os.environ, {'BOT_DEMO_MODE': '0'}):
                self.assertIsNone(ns['build_model']())
            with patch.dict(os.environ, {'BOT_DEMO_MODE': '1'}):
                self.assertIsInstance(ns['build_model'](), MockModel)
        classifier.RuBertClassifier.assert_called_once()

    def test_missing_dependencies_leave_analysis_unavailable(self):
        with patch.dict(os.environ, {'BOT_DEMO_MODE': '0'}), patch.dict('sys.modules', {'classifier': None}):
            self.assertIsNone(handlers()['build_model']())
