import unittest
from bot.knowledge_base import KnowledgeBase


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase()

    def test_legacy_string_uses_same_algorithm(self):
        for label in (0, 1, 2):
            text = 'Учеба и ЭКЗАМЕН'
            result = self.kb.get_recommendation_with_metadata(text, label)
            self.assertEqual(self.kb.get_recommendation(text, label), result.advice)
            self.assertIsInstance(self.kb.get_recommendation(text, label), str)

    def test_priority_beats_declaration_order(self):
        result = self.kb.get_recommendation_with_metadata('учеба экзамен', 1)
        self.assertEqual(result.rule_id, 'context:экзамен:1')

    def test_equal_priority_uses_declaration_order_and_logs_only_ids(self):
        text = 'ЛИЧНЫЕ_ДАННЫЕ сессия экзамен'
        with self.assertLogs('bot.knowledge_base', level='WARNING') as logs:
            result = self.kb.get_recommendation_with_metadata(text, 2)
        self.assertEqual(result.rule_id, 'context:экзамен:2')
        log = ' '.join(logs.output)
        self.assertIn('context:экзамен:2', log)
        self.assertIn('context:сесси:2', log)
        self.assertNotIn('ЛИЧНЫЕ_ДАННЫЕ', log)

    def test_only_rules_for_model_class_apply(self):
        result = self.kb.get_recommendation_with_metadata('дедлайн', 2)
        self.assertEqual(result.rule_id, 'base:2')
        self.kb.base_recommendations.pop(1)
        result = self.kb.get_recommendation_with_metadata('экзамен', 1)
        self.assertEqual(result.rule_id, 'context:экзамен:1')

    def test_missing_rules_return_neutral_and_log_without_text(self):
        self.kb.base_recommendations.pop(0)
        with self.assertLogs('bot.knowledge_base', level='WARNING') as logs:
            result = self.kb.get_recommendation_with_metadata('ЛИЧНЫЕ_ДАННЫЕ', 0)
        self.assertEqual(result.rule_id, 'neutral')
        self.assertTrue(result.advice)
        self.assertNotIn('ЛИЧНЫЕ_ДАННЫЕ', ' '.join(logs.output))

    def test_unknown_class_rejected(self):
        with self.assertRaises(ValueError):
            self.kb.get_recommendation('экзамен', 3)
