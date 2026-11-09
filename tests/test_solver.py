"""Запуск без Telegram, SQLite, torch и transformers: unittest discover -s tests."""
import dataclasses
import subprocess
import sys
import unittest
from unittest.mock import Mock

from bot.knowledge_base import KnowledgeBase, Recommendation
from bot.mock_model import MockModel
from bot.solver import Solver, SolverError


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        self.model.predict.return_value = (1, 0.8)
        self.kb = Mock()
        self.kb.get_recommendation_with_metadata.return_value = Recommendation(
            "План подготовки", "context:экзамен:1", "Упомянут экзамен."
        )
        self.solver = Solver(self.model, self.kb)

    def test_invalid_input_does_not_call_dependencies(self):
        for text in (None, 123, [], {}, b'text', '', ' \t\n '):
            with self.subTest(text=text), self.assertRaises((TypeError, ValueError)):
                self.solver.solve(text)
        self.model.predict.assert_not_called()
        self.kb.get_recommendation_with_metadata.assert_not_called()

    def test_preserves_case_punctuation_and_negation(self):
        text = '  Я НЕ   боюсь\nЭКЗАМЕНА!  '
        result = self.solver.solve(text)
        self.model.predict.assert_called_once_with('Я НЕ боюсь ЭКЗАМЕНА!')
        self.kb.get_recommendation_with_metadata.assert_called_once_with('Я НЕ боюсь ЭКЗАМЕНА!', 1)
        self.assertEqual(result.rule_id, 'context:экзамен:1')
        self.assertEqual(result.explanation, 'Упомянут экзамен.')
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.confidence = 0.1

    def test_invalid_predictions_do_not_call_knowledge_base(self):
        for prediction in (
            None, '10', {}, (1,), (1, .8, 'extra'), (3, .8), (-1, .8),
            ('1', .8), (1.0, .8), (True, .8), (1, float('nan')),
            (1, float('inf')), (1, float('-inf')), (1, -.01), (1, 1.01),
            (1, '0.8'), (1, None), (1, True),
        ):
            with self.subTest(prediction=prediction):
                self.model.reset_mock()
                self.model.predict.return_value = prediction
                with self.assertRaises(SolverError):
                    self.solver.solve('Текст')
                self.model.predict.assert_called_once_with('Текст')
                self.kb.get_recommendation_with_metadata.assert_not_called()

    def test_model_failure_is_wrapped(self):
        failure = RuntimeError('failure')
        self.model.predict.side_effect = failure
        with self.assertRaises(SolverError) as caught:
            self.solver.solve('Текст')
        self.assertIs(caught.exception.__cause__, failure)
        self.model.predict.assert_called_once()
        self.kb.get_recommendation_with_metadata.assert_not_called()

    def test_threshold_uses_unrounded_confidence(self):
        for confidence, preliminary in ((0, True), (.59999, True), (.6, False), (1, False)):
            with self.subTest(confidence=confidence):
                self.model.predict.return_value = (1, confidence)
                result = self.solver.solve('Текст')
                self.assertEqual(result.is_preliminary, preliminary)
                self.assertEqual(result.confidence, confidence)
                message = result.to_message()
                self.assertIn('По сообщению похоже, что', message)
                self.assertIn('Уверенность модели:', message)
                self.assertIn(result.advice, message)
                self.assertIn(result.explanation, message)
                self.assertEqual('Предварительная оценка' in message, preliminary)
                self.assertEqual('Уточните описание' in message, preliminary)
        self.model.predict.return_value = (1, .7)
        self.assertTrue(Solver(self.model, self.kb, threshold=.75).solve('Текст').is_preliminary)

    def test_invalid_threshold(self):
        for threshold in (None, True, '0.6', -.1, 1.1, float('nan'), float('inf')):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                Solver(self.model, self.kb, threshold=threshold)

    def test_acceptance_examples_return_expected_rules_and_explanations(self):
        for text, label, name, rule_id in (
            ('Сегодня всё хорошо', 0, 'Норма', 'base:0'),
            ('У меня завтра экзамен', 1, 'Стресс', 'context:экзамен:1'),
        ):
            with self.subTest(text=text):
                self.model.predict.return_value = (label, .85)
                result = Solver(self.model, KnowledgeBase()).solve(text)
                self.assertEqual(result.state_name, name)
                self.assertEqual(result.rule_id, rule_id)
                self.assertTrue(result.explanation)
                if label == 1:
                    self.assertIn('план подготовки', result.advice)
                    self.assertIn('экзамен', result.explanation)
                else:
                    self.assertIn('базовый совет', result.explanation)

    def test_real_knowledge_base_keeps_class_and_returns_one_rule(self):
        for label, name in ((0, 'Норма'), (1, 'Стресс'), (2, 'Тревога')):
            with self.subTest(label=label):
                self.model.predict.return_value = (label, .85)
                result = Solver(self.model, KnowledgeBase()).solve('Боюсь ЭКЗАМЕНА!')
                self.assertEqual(result.predicted_class, label)
                self.assertEqual(result.state_name, name)
                self.assertEqual(result.rule_id, 'base:0' if label == 0 else f'context:экзамен:{label}')
                self.assertTrue(result.advice)
                self.assertTrue(result.explanation)
        result = Solver(MockModel(), KnowledgeBase()).solve('Сегодня всё хорошо')
        self.assertEqual(result.rule_id, 'base:0')
        self.assertIn('базовый совет', result.explanation)

    def test_standalone_import_has_no_heavy_dependencies(self):
        code = '''
import sys
from bot.solver import Solver
from bot.mock_model import MockModel
from bot.knowledge_base import KnowledgeBase
Solver(MockModel(), KnowledgeBase()).solve('Я НЕ боюсь экзамена!')
assert not {'aiogram', 'sqlite3', 'aiosqlite', 'torch', 'transformers'} & sys.modules.keys()
'''
        subprocess.run([sys.executable, '-S', '-c', code], check=True)


if __name__ == '__main__':
    unittest.main()
