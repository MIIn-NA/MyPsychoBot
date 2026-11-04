"""ТАНЯ Связывает модель и базу знаний, собирает результат"""
"""Решатель: классификация сообщения и подбор рекомендации без Telegram."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverResult:
    predicted_class: int
    state_name: str
    confidence: float
    advice: str

    def to_message(self) -> str:
        """Сформировать текст ответа с существующим Markdown-оформлением."""
        return (
            f"Я определил твое состояние как: **{self.state_name}** "
            f"(уверенность: {int(self.confidence * 100)}%)\n\n"
            f"💡 Совет: {self.advice}"
        )


class Solver:
    CLASS_NAMES = {0: "Норма", 1: "Стресс", 2: "Тревога"}

    def __init__(self, model, knowledge_base):
        # Модель передаётся снаружи: MockModel можно заменить на RuBERT.
        self.model = model
        self.knowledge_base = knowledge_base

    def solve(self, text: str) -> SolverResult:
        """Определить состояние и выбрать совет с учётом исходного текста."""
        predicted_class, confidence = self.model.predict(text)
        advice = self.knowledge_base.get_recommendation(text, predicted_class)

        return SolverResult(
            predicted_class=predicted_class,
            state_name=self.CLASS_NAMES[predicted_class],
            confidence=confidence,
            advice=advice,
        )
