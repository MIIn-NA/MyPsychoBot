"""Ответственная: Даша.
Классификатор состояния на основе дообученного RuBERT.
"""

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __package__:
    from .errors import ClassificationError, TextTooLongError
else:
    from errors import ClassificationError, TextTooLongError

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "rubert_classifier"

CLASS_NAMES = {0: "норма", 1: "стресс", 2: "тревога"}


class ModelLoadError(RuntimeError):
    """Не удалось загрузить модель."""


def clean_text(text):
    # Те же правила, что в train_model.py
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


class RuBertClassifier:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)

        if not (model_dir / "config.json").exists():
            raise ModelLoadError(
                f"Модель не найдена: {model_dir}. "
                "Сначала обучите её: python bot/train_model.py"
            )

        try:
            with open(model_dir / "inference_config.json", encoding="utf-8") as file:
                config = json.load(file)

            self.max_length = config["max_length"]
            self.low_confidence_threshold = config["confidence_threshold"]
            self.model_dir = model_dir
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            # Учитываем и настройку инференса, и физический лимит RuBERT.
            self.max_length = min(self.max_length, self.model.config.max_position_embeddings)
            self.model.to(self.device)
            self.model.eval()
        except Exception as error:
            raise ModelLoadError(f"Ошибка загрузки модели: {error}") from error

    def predict(self, text):
        """Возвращает (класс, уверенность)."""
        text = clean_text(text) if isinstance(text, str) else ""

        if not text:
            raise ClassificationError("Пустое сообщение: нечего классифицировать.")

        try:
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt",
            )
            token_count = inputs["input_ids"].shape[-1]
            if token_count > self.max_length:
                raise TextTooLongError(token_count, self.max_length)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probabilities = torch.softmax(logits, dim=1)[0]
            label = int(torch.argmax(probabilities))

            return label, float(probabilities[label])
        except TextTooLongError:
            raise
        except Exception as error:
            raise ClassificationError(f"Ошибка классификации: {error}") from error


if __name__ == "__main__":
    classifier = RuBertClassifier()
    print(f"Порог низкой уверенности: {classifier.low_confidence_threshold:.2f}")

    for sample in [
        "Сегодня отличная погода, всё хорошо",
        "У меня завтра дедлайн, я не успеваю и очень устал",
        "Мне страшно, что я провалю экзамен и не могу уснуть",
    ]:
        label, confidence = classifier.predict(sample)
        print(f"{sample} -> {CLASS_NAMES[label]} ({confidence:.2f})")