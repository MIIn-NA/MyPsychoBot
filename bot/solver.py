"""Классификация сообщения и подбор рекомендации без внешних сервисов."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

if __package__:
    from .errors import TextTooLongError
else:
    from errors import TextTooLongError


class SolverError(RuntimeError):
    """Модель не смогла вернуть корректный результат анализа."""


@dataclass(frozen=True)
class SolverResult:
    predicted_class: int
    state_name: str
    confidence: float
    advice: str
    is_preliminary: bool
    explanation: str
    rule_id: str

    def to_message(self, *, offer_clarification=True) -> str:
        """Единое представление результата для Telegram."""
        heading = f"По сообщению похоже, что ваше состояние — **{self.state_name}**."
        if self.is_preliminary:
            heading = f"**Предварительная оценка**\n{heading}"
        message = (
            f"{heading}\nУверенность модели: {self.confidence:.0%}\n\n"
            f"💡 Совет: {self.advice}\n"
            f"Основание: {self.explanation}"
        )
        if self.is_preliminary and offer_clarification:
            message += "\n\nУточните описание: что происходит и как вы себя чувствуете?"
        return message


class Solver:
    CLASS_NAMES = {0: "Норма", 1: "Стресс", 2: "Тревога"}

    def __init__(self, model, knowledge_base, threshold=0.60):
        """threshold задаётся снаружи, для RuBERT — из конфигурации модели."""
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("Порог уверенности должен быть конечным числом от 0 до 1.")
        self.model = model
        self.knowledge_base = knowledge_base
        self.threshold = float(threshold)

    def solve(self, text: str) -> SolverResult:
        """Проверить текст и прогноз, затем выбрать правило базы знаний.

        Ошибки входа: TypeError/ValueError. Ошибки модели: SolverError.
        База знаний возвращает объект с advice, explanation и rule_id.
        """
        if not isinstance(text, str):
            raise TypeError("Сообщение должно быть строкой.")
        text = " ".join(text.split())
        if not text:
            raise ValueError("Сообщение не должно быть пустым.")

        try:
            prediction = self.model.predict(text)
        except TextTooLongError:
            raise
        except Exception as error:
            raise SolverError("Не удалось проанализировать сообщение: ошибка модели.") from error

        if not isinstance(prediction, (tuple, list)) or len(prediction) != 2:
            raise SolverError("Модель должна вернуть пару (класс, уверенность).")
        predicted_class, confidence = prediction
        if (
            isinstance(predicted_class, bool)
            or not isinstance(predicted_class, Integral)
            or predicted_class not in self.CLASS_NAMES
        ):
            raise SolverError("Модель вернула неизвестный класс: ожидается 0, 1 или 2.")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, Real)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise SolverError("Уверенность модели должна быть конечным числом от 0 до 1.")

        recommendation = self.knowledge_base.get_recommendation_with_metadata(text, predicted_class)
        return SolverResult(
            predicted_class=predicted_class,
            state_name=self.CLASS_NAMES[predicted_class],
            confidence=float(confidence),
            advice=recommendation.advice,
            is_preliminary=confidence < self.threshold,
            explanation=recommendation.explanation,
            rule_id=recommendation.rule_id,
        )
