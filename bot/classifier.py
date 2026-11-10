
"""
Ответственная: Даша.
Классификатор состояния на основе дообученного RuBERT.
"""

import json
import re
import threading
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __package__:
    from .errors import ClassificationError, TextTooLongError
else:
    from errors import ClassificationError, TextTooLongError

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


def clean_text(text: str) -> str:
    """Удаляет лишние пробелы из текста (так же, как при обучении)."""
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

            max_positions = getattr(
                self.model.config, "max_position_embeddings", None
            )
            if max_positions is not None and self.max_length > max_positions:
                raise ValueError(
                    f"max_length ({self.max_length}) больше, чем "
                    f"поддерживает модель ({max_positions})."
                )

            self.model.to(self.device)
            self.model.eval()

            # Один запрос к модели за раз: безопасно при вызове из потоков.
            self._lock = threading.Lock()

        except Exception as error:
            raise ModelLoadError(
                f"Ошибка загрузки модели: {error}"
            ) from error

    def predict(self, text: str) -> tuple[int, float]:
        """
        Возвращает номер класса и уверенность модели.

        Ошибки:
        - ClassificationError — пустой текст или сбой модели;
        - TextTooLongError (наследник ClassificationError) — текст не
          помещается в max_length токенов, включая специальные.
        """

        text = clean_text(text) if isinstance(text, str) else ""

        if not text:
            raise ClassificationError(
                "Пустое сообщение: нечего классифицировать."
            )

        try:
            # Без обрезки: сначала нужно узнать реальную длину.
            inputs = self.tokenizer(
                text,
                truncation=False,
                return_tensors="pt",
            )
        except Exception as error:
            raise ClassificationError(
                f"Ошибка токенизации: {error}"
            ) from error

        # Проверка вне try, чтобы TextTooLongError не превратилась
        # в общий ClassificationError.
        token_count = int(inputs["input_ids"].shape[1])

        if token_count > self.max_length:
            raise TextTooLongError(token_count, self.max_length)

        try:
            inputs = inputs.to(self.device)

            with self._lock, torch.inference_mode():
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

    test_texts = [
        "Мне хорошо, всё отлично",
        "Завтра экзамен, я не успеваю и очень устал",
        "Мне страшно, постоянно думаю о будущем и не могу уснуть",
        "Мне плохо",
    ]

    for test_text in test_texts:
        try:
            label, confidence = classifier.predict(test_text)
        except ClassificationError as error:
            print(f"{test_text!r}: ошибка — {error}")
            continue

        marker = (
            " (ниже порога)"
            if confidence < classifier.low_confidence_threshold
            else ""
        )

        print(
            f"{test_text!r}: {CLASS_NAMES[label]}, "
            f"уверенность: {confidence:.4f}{marker}"
        )
