"""Проверка реакции Telegram на ошибку длины от модели."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from bot.errors import TextTooLongError
from bot.solver import Solver, SolverError


class TelegramLengthTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_and_combined_text_use_length_check_and_allow_retry(self):
        # Изолируем настоящие функции обработчика от старта бота, SQLite и RuBERT.
        tree = ast.parse(Path('bot/main_bot.py').read_text())
        functions = []
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef) and node.name in ('handle_message', 'process_solver'):
                node.decorator_list = []
                functions.append(node)
        model = Mock()
        model.predict.side_effect = TextTooLongError(6, 5)
        kb = Mock()
        namespace = dict(
            types=SimpleNamespace(Message=object), user_states={},
            solver=Solver(model, kb), SolverError=SolverError,
            TextTooLongError=TextTooLongError, logging=Mock(),
            get_main_keyboard=lambda: None,
        )
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<handlers>', 'exec'), namespace)
        for previous in ('', 'Прежний текст'):
            model.reset_mock()
            state = dict(previous_text=previous, awaiting_clarification=bool(previous),
                         awaiting_feedback=False, clarification_count=0)
            namespace['user_states'][1] = state
            message = SimpleNamespace(from_user=SimpleNamespace(id=1), text='Новый текст', answer=AsyncMock())
            await namespace['handle_message'](message)
            expected = f'{previous} Новый текст'.strip()
            model.predict.assert_called_once_with(expected)
            message.answer.assert_awaited_once()
            self.assertIn('сократите сообщение', message.answer.call_args.args[0])
            self.assertFalse(state['awaiting_clarification'])
            self.assertFalse(state['awaiting_feedback'])
            self.assertEqual(state['previous_text'], '')
        kb.get_recommendation_with_metadata.assert_not_called()
