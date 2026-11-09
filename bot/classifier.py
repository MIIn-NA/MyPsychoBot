
"""
Ответственная: Даша.
Классификатор состояния на основе дообученного RuBERT.
"""

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "rubert_classifier"
)

CLASS_NAMES = {
    0: "норма",
    1: "стресс",
    2: "тревога",
}


class ModelLoadError(RuntimeError):
    """Не удалось загрузить модель."""


class ClassificationError(RuntimeError):
    """Не удалось классифицировать сообщение."""


def clean_text(text: str) -> str:
    """Удаляет лишние пробелы из текста."""
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


class RuBertClassifier:

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = Path(model_dir)

        config_path = self.model_dir / "inference_config.json"
        model_config_path = self.model_dir / "config.json"

        if not model_config_path.exists():
            raise ModelLoadError(
                f"Модель не найдена: {self.model_dir}. "
                "Сначала обучите её: python bot/train_model.py"
            )

        if not config_path.exists():
            raise ModelLoadError(
                f"Не найден файл настроек: {config_path}"
            )

        try:
            with open(config_path, encoding="utf-8") as file:
                config = json.load(file)

            self.max_length = int(config["max_length"])
            self.low_confidence_threshold = float(
                config["confidence_threshold"]
            )

            if self.max_length <= 0:
                raise ValueError("max_length должен быть положительным.")

            if not 0 <= self.low_confidence_threshold <= 1:
                raise ValueError(
                    "confidence_threshold должен быть от 0 до 1."
                )

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            )

            if self.model.config.num_labels != len(CLASS_NAMES):
                raise ValueError(
                    "Количество классов модели не совпадает "
                    "с количеством классов классификатора."
                )

            self.model.to(self.device)
            self.model.eval()

        except Exception as error:
            raise ModelLoadError(
                f"Ошибка загрузки модели: {error}"
            ) from error

    def predict(self, text: str) -> tuple[int, float]:
        """
        Возвращает номер класса и уверенность модели.
        """

        text = clean_text(text) if isinstance(text, str) else ""

        if not text:
            raise ClassificationError(
                "Пустое сообщение: нечего классифицировать."
            )

        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                logits = self.model(**inputs).logits

            probabilities = torch.softmax(logits, dim=1)[0]

            label = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[label].item())

            return label, confidence

        except Exception as error:
            raise ClassificationError(
                f"Ошибка классификации: {error}"
            ) from error


if __name__ == "__main__":
    classifier = RuBertClassifier()

    testText = "Мне хорошо, всё отлично"

    for i in range(5):
        label, confidence = classifier.predict(testText)

        print(
            f"Попытка {i + 1}: "
            f"{CLASS_NAMES[label]}, "
            f"уверенность: {confidence:.4f}"
        )
