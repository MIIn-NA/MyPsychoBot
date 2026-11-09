
"""
Ответственная: Даша.
Назначение: подготовка данных, обучение и оценка модели RuBERT.
"""

import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Пути проекта
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "data.csv"
MODEL_DIR = PROJECT_DIR / "models" / "rubert_classifier"
RESULTS_DIR = PROJECT_DIR / "results"


# Параметры модели
MODEL_NAME = "DeepPavlov/rubert-base-cased"

CLASS_NAMES = {
    0: "норма",
    1: "стресс",
    2: "тревога",
}

ID_TO_LABEL = {
    0: "normal",
    1: "stress",
    2: "anxiety",
}

LABEL_TO_ID = {
    "normal": 0,
    "stress": 1,
    "anxiety": 2,
}

RANDOM_SEED = 42
MAX_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

TARGET_ACCURACY = 0.80
MIN_THRESHOLD = 0.6  # Минимальный порог уверенности

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Установка случайных значений для воспроизводимости
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Нормализация текста
def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Загрузка и проверка датасета
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Файл датасета не найден: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    required_columns = {"text", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "В CSV должны присутствовать столбцы text и label."
        )

    data = data[["text", "label"]].copy()

    data["text"] = data["text"].apply(clean_text)
    data["label"] = pd.to_numeric(
        data["label"], errors="coerce"
    )

    data = data.dropna(subset=["text", "label"])

    if not data["label"].apply(
        lambda value: float(value).is_integer()
    ).all():
        raise ValueError("Метки классов должны быть целыми числами.")

    data["label"] = data["label"].astype(int)

    data = data[data["text"] != ""]

    if not data["label"].isin(CLASS_NAMES.keys()).all():
        raise ValueError(
            "Допустимые метки классов: 0, 1, 2."
        )

    # Проверяем, нет ли одинаковых текстов с разными метками
    normalized = data["text"].str.casefold()

    conflicting = (
        data.assign(normalized=normalized)
        .groupby("normalized")["label"]
        .nunique()
    )

    if (conflicting > 1).any():
        raise ValueError(
            "Обнаружены одинаковые тексты с разными метками. "
            "Исправьте разметку датасета."
        )

    # Удаляем точные дубликаты без учета регистра
    data["normalized"] = normalized
    data = data.drop_duplicates(subset=["normalized"])
    data = data.drop(columns=["normalized"])

    counts = data["label"].value_counts()

    for class_id in CLASS_NAMES:
        if counts.get(class_id, 0) < 3:
            raise ValueError(
                f"Для класса {class_id} недостаточно примеров."
            )

    print("\nРаспределение классов:")

    for class_id, name in CLASS_NAMES.items():
        print(f"{class_id} — {name}: {counts.get(class_id, 0)}")

    print(f"\nВсего примеров: {len(data)}")

    return data


# Разделение данных на три выборки
def split_data(data):
    train_data, temporary_data = train_test_split(
        data,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=data["label"],
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_data["label"],
    )

    print("\nРазмеры выборок:")
    print(f"Обучающая: {len(train_data)}")
    print(f"Валидационная: {len(validation_data)}")
    print(f"Тестовая: {len(test_data)}")

    return train_data, validation_data, test_data


# Подготовка данных для RuBERT
class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }

        item["labels"] = self.labels[index]

        return item


# Создание DataLoader
def create_loader(data, tokenizer, shuffle=False):
    dataset = TextDataset(
        data["text"].tolist(),
        data["label"].tolist(),
        tokenizer,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


# Оценка модели на выборке
def evaluate_model(model, loader):
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            labels = batch.pop("labels")

            outputs = model(**batch)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


# Подбор порога уверенности по валидационной выборке
def find_confidence_threshold(labels, predictions, probabilities):
    confidences = probabilities.max(axis=1)

    best_threshold = None
    best_coverage = -1.0
    best_accuracy = 0.0

    # Рассматриваем пороги не ниже MIN_THRESHOLD
    candidates = np.unique(
        np.append(
            confidences[confidences >= MIN_THRESHOLD],
            MIN_THRESHOLD,
        )
    )

    for threshold in candidates:
        accepted = confidences >= threshold

        coverage = accepted.mean()

        # Требуем, чтобы модель принимала не менее 50% сообщений
        if coverage < 0.50:
            continue

        accuracy = accuracy_score(
            labels[accepted],
            predictions[accepted],
        )

        # Выбираем порог с максимальным покрытием,
        # при котором достигнута целевая Accuracy
        if accuracy >= TARGET_ACCURACY:
            if coverage > best_coverage:
                best_threshold = float(threshold)
                best_coverage = float(coverage)
                best_accuracy = float(accuracy)

    # Если целевая Accuracy не достигнута
    if best_threshold is None:
        best_threshold = MIN_THRESHOLD

        accepted = confidences >= best_threshold
        coverage = float(accepted.mean())

        if accepted.any():
            best_accuracy = float(
                accuracy_score(
                    labels[accepted],
                    predictions[accepted],
                )
            )
        else:
            best_accuracy = None

        print(
            "\nПредупреждение: не удалось достичь "
            f"Accuracy {TARGET_ACCURACY:.0%} "
            "при покрытии не менее 50%. "
            f"Используется минимальный порог {MIN_THRESHOLD:.2f}."
        )

        return {
            "threshold": best_threshold,
            "target_accuracy": TARGET_ACCURACY,
            "achieved_accuracy": best_accuracy,
            "coverage": coverage,
            "target_reached": False,
        }

    return {
        "threshold": best_threshold,
        "target_accuracy": TARGET_ACCURACY,
        "achieved_accuracy": best_accuracy,
        "coverage": best_coverage,
        "target_reached": True,
    }


# Основной процесс обучения
def train_model():
    set_seed(RANDOM_SEED)

    print(f"Устройство: {DEVICE}")

    if torch.cuda.is_available():
        print(f"Видеокарта: {torch.cuda.get_device_name(0)}")

    data = load_data()

    train_data, validation_data, test_data = split_data(data)

    print("\nЗагрузка токенизатора...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Загрузка RuBERT...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    model.to(DEVICE)

    train_loader = create_loader(
        train_data, tokenizer, shuffle=True
    )

    validation_loader = create_loader(
        validation_data, tokenizer
    )

    test_loader = create_loader(
        test_data, tokenizer
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.10)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        return max(
            0.0,
            (total_steps - current_step)
            / max(1, total_steps - warmup_steps),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )

    best_f1 = -1.0
    best_state = None

    history = []

    # Цикл обучения
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        print(f"\nЭпоха {epoch + 1}/{EPOCHS}")

        for batch in train_loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)

            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        labels, predictions, probabilities = evaluate_model(
            model,
            validation_loader,
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        )

        print(f"Training loss: {average_loss:.4f}")
        print(
            f"Validation accuracy: "
            f"{accuracy_score(labels, predictions):.4f}"
        )
        print(f"Validation F1: {f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "training_loss": average_loss,
            "validation_precision": float(precision),
            "validation_recall": float(recall),
            "validation_f1": float(f1),
        })

        if f1 > best_f1:
            best_f1 = f1

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Не удалось сохранить лучшую модель.")

    model.load_state_dict(best_state)
    model.to(DEVICE)

    # Подбор порога только на валидационной выборке
    val_labels, val_predictions, val_probabilities = evaluate_model(
        model,
        validation_loader,
    )

    threshold_info = find_confidence_threshold(
        val_labels,
        val_predictions,
        val_probabilities,
    )

    # Итоговая оценка на тестовой выборке
    print("\nИтоговая оценка на тестовой выборке")

    test_labels, test_predictions, test_probabilities = (
        evaluate_model(model, test_loader)
    )

    accuracy = accuracy_score(
        test_labels,
        test_predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            test_labels,
            test_predictions,
            average="macro",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        test_labels,
        test_predictions,
        labels=[0, 1, 2],
    )

    report = classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1, 2],
        target_names=[
            CLASS_NAMES[0],
            CLASS_NAMES[1],
            CLASS_NAMES[2],
        ],
        output_dict=True,
        zero_division=0,
    )

    # Измерение времени обработки одного сообщения
    model.eval()

    timing_texts = test_data["text"].tolist()

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for text in timing_texts:
            inputs = tokenizer(
                clean_text(text),
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(DEVICE)

            model(**inputs)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time

    average_time = total_time / len(timing_texts)

    # Сохранение модели и токенизатора
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Сохранение соответствия классов
    class_mapping = {
        str(class_id): name
        for class_id, name in CLASS_NAMES.items()
    }

    with open(
        MODEL_DIR / "class_mapping.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            class_mapping,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # Настройки для classifier.py
    inference_config = {
        "max_length": MAX_LENGTH,
        "confidence_threshold": threshold_info["threshold"],
    }

    with open(
        MODEL_DIR / "inference_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(inference_config, file, indent=4)

    # Сохранение матрицы ошибок
    matrix_frame = pd.DataFrame(
        matrix,
        index=list(CLASS_NAMES.values()),
        columns=list(CLASS_NAMES.values()),
    )

    matrix_frame.to_csv(
        RESULTS_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    # Сохранение результатов
    results = {
        "model_name": MODEL_NAME,
        "device": str(DEVICE),
        "dataset_size": len(data),
        "train_size": len(train_data),
        "validation_size": len(validation_data),
        "test_size": len(test_data),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "average_inference_time_seconds": average_time,
        "confidence_threshold": threshold_info,
        "classification_report": report,
        "training_history": history,
    }

    with open(
        RESULTS_DIR / "training_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\nРезультаты:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 (macro): {f1:.4f}")

    print("\nМатрица ошибок:")
    print(matrix)

    print(
        f"\nСреднее время обработки сообщения: "
        f"{average_time * 1000:.2f} мс"
    )

    print(
        f"\nПорог уверенности: "
        f"{threshold_info['threshold']:.4f}"
    )

    print(
        f"Целевая Accuracy достигнута: "
        f"{threshold_info['target_reached']}"
    )

    print(
        f"Accuracy при выбранном пороге: "
        f"{threshold_info['achieved_accuracy']}"
    )

    print(
        f"Покрытие: "
        f"{threshold_info['coverage']:.2%}"
    )

    print(f"\nМодель сохранена: {MODEL_DIR}")
    print(f"Результаты сохранены: {RESULTS_DIR}")


if __name__ == "__main__":
    train_model()
