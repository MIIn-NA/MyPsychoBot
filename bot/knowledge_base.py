from dataclasses import dataclass
import re
import logging
from numbers import Integral

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Recommendation:
    advice: str
    rule_id: str
    explanation: str


class KnowledgeBase:
    def __init__(self):
        # Базовые рекомендации для каждого класса
        self.base_recommendations = {
            0: "Отлично! Вы в ресурсном состоянии. Поддерживайте режим сна и физическую активность.",
            1: "Похоже, вы испытываете стресс. Попробуйте дыхательное упражнение 4-7-8: вдох на 4 секунды, задержка на 7, выдох на 8.",
            2: "Я замечаю признаки тревоги. Попробуйте технику заземления 5-4-3-2-1: назовите 5 предметов вокруг, 4 звука, 3 ощущения, 2 запаха и 1 вкус."
        }

        # Контекстные правила (используем корни слов для поиска)
        self.context_rules = {
            "учеб": {
                1: "Учеба вызывает стресс. Составьте четкий план, разбейте задачи на маленькие шаги и не забывайте отдыхать.",
                2: "Тревога из-за учебы. Попробуйте выписать все свои страхи на бумагу, а затем составьте план действий на завтра."
            },
            "экзамен": {
                1: "Упоминание экзамена. Составьте четкий план подготовки, разбейте материал на небольшие блоки.",
                2: "Тревога из-за экзамена. Напомните себе, что вы уже сдавали экзамены раньше, и у вас получалось."
            },
            "сесси": {
                1: "Стресс из-за сессии. Работайте по технике Pomodoro: 25 минут учебы, 5 минут отдыха.",
                2: "Тревога перед сессией. Важно высыпаться. Не учите всю ночь перед экзаменом."
            },
            "дедлайн": {
                1: "Дедлайн вызывает стресс. Сфокусируйтесь на одной задаче прямо сейчас, отложив все остальные."
            },
            "сон": {
                2: "Проблемы со сном. Попробуйте за час до сна убрать телефон и выпить травяной чай."
            },
            "устал": {
                1: "Вы чувствуете усталость. Сделайте перерыв на 15 минут, отвлекитесь от учебы и выпейте воды.",
                2: "Вы истощены. В таком состоянии сложно учиться. Разрешите себе сегодня лечь спать пораньше."
            },
            "груст": {
                1: "Вам грустно. Это нормально. Попробуйте вспомнить о том, что приносило вам радость раньше, и сделать это сегодня.",
                2: "Вы чувствуете грусть и тревогу. Позвольте себе прожить эту эмоцию, но не оставайтесь в ней слишком долго."
            },
            "отношен": {
                2: "Переживания из-за отношений отнимают много сил. Попробуйте переключить внимание на простые физические действия (прогулка).",
                1: "Конфликты выматывают. Постарайтесь сегодня ограничить общение с теми, кто вас расстраивает, и побыть в тишине."
            },
            "будущ": {
                2: "Тревога о будущем. Сфокусируйтесь на том, что вы можете контролировать сегодня."
            },
            "одинок": {
                2: "Чувство одиночества. Попробуйте написать или позвонить старому другу или родственнику."
            }
        }

        # Большее число означает более высокий приоритет. Порядок словаря
        # context_rules разрешает ничью и остаётся порядком объявления.
        self.context_priorities = {
            "учеб": 10, "экзамен": 20, "сесси": 20, "дедлайн": 20,
            "сон": 20, "устал": 10, "груст": 10, "отношен": 20,
            "будущ": 20, "одинок": 20,
        }

    def get_recommendation(self, text, predicted_class):
        """Совместимый строковый интерфейс с тем же выбором правила."""
        return self.get_recommendation_with_metadata(text, predicted_class).advice

    def get_recommendation_with_metadata(self, text, predicted_class):
        """Выбрать правило только для заданного класса, не меняя прогноз."""
        if (isinstance(predicted_class, bool)
                or not isinstance(predicted_class, Integral)
                or predicted_class not in (0, 1, 2)):
            raise ValueError("Ожидается класс модели 0, 1 или 2.")
        text_lower = text.lower()
        applicable = []
        for keyword, rules in self.context_rules.items():
            if predicted_class not in rules:
                continue
            match = re.search(r'\b' + re.escape(keyword) + r'\w*', text_lower)
            if match:
                recommendation = Recommendation(
                    advice=rules[predicted_class],
                    rule_id=f"context:{keyword}:{predicted_class}",
                    explanation=(
                        "Выбран контекстный совет, потому что в сообщении "
                        f"найдено слово «{match.group()}» "
                        "и правило соответствует классу модели."
                    ),
                )
                applicable.append((self.context_priorities.get(keyword, 0), recommendation))

        if applicable:
            priority = max(item[0] for item in applicable)
            winners = [rule for rank, rule in applicable if rank == priority]
            if len(winners) > 1:
                logger.warning(
                    "Конфликт правил с приоритетом %s: %s; выбрано %s",
                    priority, ", ".join(rule.rule_id for rule in winners), winners[0].rule_id,
                )
            return winners[0]

        if predicted_class in self.base_recommendations:
            return Recommendation(
                advice=self.base_recommendations[predicted_class],
                rule_id=f"base:{predicted_class}",
                explanation=(
                    "Выбран базовый совет для класса модели: "
                    "подходящих контекстных правил в сообщении не найдено."
                ),
            )

        logger.warning("Нет применимых правил для класса %s; выбран neutral", predicted_class)
        return Recommendation(
            advice="Сделайте небольшой перерыв и обратите внимание на своё самочувствие.",
            rule_id="neutral",
            explanation="Выбран нейтральный совет: для класса модели нет подходящих правил.",
        )
