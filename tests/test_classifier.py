"""Проверка лимита без установки torch, transformers и загрузки весов."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from bot.errors import TextTooLongError
from bot.solver import Solver, SolverError


def load_classifier():
    spec = importlib.util.spec_from_file_location('bot._classifier_test', 'bot/classifier.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'torch': Mock(), 'transformers': Mock()}):
        spec.loader.exec_module(module)
    return module


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.module = load_classifier()
        self.classifier = self.module.RuBertClassifier.__new__(self.module.RuBertClassifier)
        self.classifier.max_length = 5
        self.classifier.device = 'cpu'
        self.classifier.model = Mock()
        self.inputs = Mock()
        self.inputs.__getitem__ = Mock(return_value=SimpleNamespace(shape=(1, 5)))
        self.inputs.to.return_value = {"input_ids": Mock()}
        self.classifier.tokenizer = Mock(return_value=self.inputs)
        self.module.torch.no_grad.return_value.__enter__ = Mock()
        self.module.torch.no_grad.return_value.__exit__ = Mock(return_value=False)
        self.module.torch.softmax.return_value = [[.1, .8, .1]]
        self.module.torch.argmax.return_value = 1

    def test_exact_limit_is_accepted_without_truncation(self):
        self.assertEqual(self.classifier.predict('Текст'), (1, .8))
        self.classifier.tokenizer.assert_called_once_with(
            'Текст', add_special_tokens=True, truncation=False, return_tensors='pt'
        )
        self.classifier.model.assert_called_once()

    def test_over_limit_rejected_before_network_and_knowledge_base(self):
        self.inputs.__getitem__.return_value.shape = (1, 6)
        kb = Mock()
        with self.assertRaises(TextTooLongError) as caught:
            Solver(self.classifier, kb).solve('Длинный текст')
        self.assertEqual(caught.exception.token_count, 6)
        self.assertEqual(caught.exception.max_length, 5)
        self.classifier.model.assert_not_called()
        self.inputs.to.assert_not_called()
        kb.get_recommendation_with_metadata.assert_not_called()

    def test_model_position_limit_caps_configured_limit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            (root / 'inference_config.json').write_text(
                '{"max_length": 1024, "confidence_threshold": 0.6}'
            )
            model = self.module.AutoModelForSequenceClassification.from_pretrained.return_value
            model.config.max_position_embeddings = 512
            classifier = self.module.RuBertClassifier(root)
            self.assertEqual(classifier.max_length, 512)


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
