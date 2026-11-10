import argparse
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
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

MIN_EXAMPLES_PER_CLASS = 10  # иначе стратифицированное разбиение ненадёжно

TARGET_ACCURACY = 0.80
MIN_THRESHOLD = 0.60  # минимальный порог уверенности
MAX_THRESHOLD = 0.95
MIN_COVERAGE = 0.50  # модель должна принимать не менее 50% сообщений
MIN_ACCEPTED = 10  # и не менее 10 примеров, иначе оценка случайна

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


# Нормализация текста (должна совпадать с classifier.py)
def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Загрузка и проверка датасета
def load_data(data_path=DATA_PATH):
    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл датасета не найден: {data_path}"
        )

    data = pd.read_csv(data_path, encoding="utf-8-sig")

    required_columns = {"text", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "В CSV должны присутствовать столбцы text и label."
        )

    data = data[["text", "label"]].copy()

    # Сначала убираем пустые значения, и только потом чистим текст,
    # иначе NaN превращается в строку "nan".
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["text", "label"])

    data["text"] = data["text"].apply(clean_text)
    data = data[data["text"] != ""]

    if not data["label"].apply(
            lambda value: float(value).is_integer()
    ).all():
        raise ValueError("Метки классов должны быть целыми числами.")

    data["label"] = data["label"].astype(int)

    if not data["label"].isin(CLASS_NAMES.keys()).all():
        raise ValueError(
            "Допустимые метки классов: 0, 1, 2."
        )

    # Проверяем, нет ли одинаковых текстов с разными метками
    data["normalized"] = data["text"].str.casefold()

    conflicting = data.groupby("normalized")["label"].nunique()

    if (conflicting > 1).any():
        examples = conflicting[conflicting > 1].index[:5].tolist()
        raise ValueError(
            "Обнаружены одинаковые тексты с разными метками. "
            f"Исправьте разметку датасета. Примеры: {examples}"
        )

    # Удаляем точные дубликаты без учета регистра
    before = len(data)
    data = data.drop_duplicates(subset=["normalized"])
    data = data.drop(columns=["normalized"])
    removed = before - len(data)

    if removed:
        print(f"Удалено дубликатов: {removed}")

    counts = data["label"].value_counts()

    for class_id in CLASS_NAMES:
        if counts.get(class_id, 0) < MIN_EXAMPLES_PER_CLASS:
            raise ValueError(
                f"Для класса {class_id} недостаточно примеров "
                f"(нужно минимум {MIN_EXAMPLES_PER_CLASS})."
            )

    print("\nРаспределение классов:")

    for class_id, name in CLASS_NAMES.items():
        print(f"{class_id} — {name}: {counts.get(class_id, 0)}")

    print(f"\nВсего примеров: {len(data)}")

    return data.reset_index(drop=True)


# Проверка, не обрезаются ли тексты при заданном MAX_LENGTH
def report_token_lengths(data, tokenizer, max_length):
    encoded = tokenizer(data["text"].tolist(), truncation=False)
    lengths = np.array([len(ids) for ids in encoded["input_ids"]])

    too_long = float((lengths > max_length).mean())

    print("\nДлина текстов в токенах (с учётом специальных):")
    print(f"медиана: {int(np.median(lengths))}")
    print(f"95-й процентиль: {int(np.percentile(lengths, 95))}")
    print(f"максимум: {int(lengths.max())}")
    print(
        f"Длиннее MAX_LENGTH={max_length}: {too_long:.1%} "
        "(такие примеры будут обрезаны при обучении)"
    )

    if too_long > 0.05:
        print(
            "Предупреждение: больше 5% примеров обрезается. "
            "Увеличьте --max-length."
        )


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

    return (
        train_data.reset_index(drop=True),
        validation_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )


# Подготовка данных для RuBERT (padding выполняется по батчу)
class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: values[index]
            for key, values in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


def make_collate_fn(tokenizer):
    def collate(batch):
        labels = torch.tensor(
            [item.pop("labels") for item in batch],
            dtype=torch.long,
        )
        padded = dict(
            tokenizer.pad(batch, padding=True, return_tensors="pt")
        )
        padded["labels"] = labels
        return padded

    return collate


# Создание DataLoader
def create_loader(data, tokenizer, max_length, shuffle=False):
    dataset = TextDataset(
        data["text"].tolist(),
        data["label"].tolist(),
        tokenizer,
        max_length,
    )

    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(RANDOM_SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=make_collate_fn(tokenizer),
        generator=generator,
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

            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


# Точность и покрытие при заданном пороге
def accuracy_at_threshold(labels, predictions, probabilities, threshold):
    confidences = probabilities.max(axis=1)
    accepted = confidences >= threshold

    coverage = float(accepted.mean())

    if not accepted.any():
        return None, coverage

    accuracy = float(
        accuracy_score(labels[accepted], predictions[accepted])
    )

    return accuracy, coverage


# Подбор порога уверенности по валидационной выборке
def find_confidence_threshold(labels, predictions, probabilities):
    confidences = probabilities.max(axis=1)

    candidates = np.round(
        np.arange(MIN_THRESHOLD, MAX_THRESHOLD + 0.0001, 0.01), 2
    )

    feasible = []

    for threshold in candidates:
        accepted = confidences >= threshold

        coverage = float(accepted.mean())

        if coverage < MIN_COVERAGE or accepted.sum() < MIN_ACCEPTED:
            continue

        accuracy = float(
            accuracy_score(labels[accepted], predictions[accepted])
        )

        feasible.append((float(threshold), accuracy, coverage))

    # Среди порогов с достигнутой целевой Accuracy берём тот,
    # у которого покрытие максимально.
    reached = [item for item in feasible if item[1] >= TARGET_ACCURACY]

    if reached:
        threshold, accuracy, coverage = max(
            reached, key=lambda item: (item[2], -item[0])
        )

        return {
            "threshold": threshold,
            "target_accuracy": TARGET_ACCURACY,
            "achieved_accuracy": accuracy,
            "coverage": coverage,
            "target_reached": True,
        }

    print(
        "\nПредупреждение: не удалось достичь "
        f"Accuracy {TARGET_ACCURACY:.0%} "
        f"при покрытии не менее {MIN_COVERAGE:.0%}."
    )

    if feasible:
        # Берём порог с лучшей Accuracy (при равенстве — меньший порог).
        threshold, accuracy, coverage = max(
            feasible, key=lambda item: (item[1], -item[0])
        )
        print(f"Выбран порог {threshold:.2f} с лучшей Accuracy.")
    else:
        threshold = MIN_THRESHOLD
        accuracy, coverage = accuracy_at_threshold(
            labels, predictions, probabilities, threshold
        )
        print(f"Используется минимальный порог {MIN_THRESHOLD:.2f}.")

    return {
        "threshold": threshold,
        "target_accuracy": TARGET_ACCURACY,
        "achieved_accuracy": accuracy,
        "coverage": coverage,
        "target_reached": False,
    }


# Веса классов: редкие классы получают больший вес
def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)

    return torch.tensor(weights, dtype=torch.float)


# Основной процесс обучения
def train_model(epochs=EPOCHS, max_length=MAX_LENGTH, data_path=DATA_PATH):
    set_seed(RANDOM_SEED)

    print(f"Устройство: {DEVICE}")

    if torch.cuda.is_available():
        print(f"Видеокарта: {torch.cuda.get_device_name(0)}")

    data = load_data(data_path)

    train_data, validation_data, test_data = split_data(data)

    print("\nЗагрузка токенизатора...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    report_token_lengths(data, tokenizer, max_length)

    print("\nЗагрузка RuBERT...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    model.to(DEVICE)

    train_loader = create_loader(
        train_data, tokenizer, max_length, shuffle=True
    )

    validation_loader = create_loader(
        validation_data, tokenizer, max_length
    )

    test_loader = create_loader(
        test_data, tokenizer, max_length
    )

    class_weights = compute_class_weights(train_data["label"].to_numpy())
    print(f"\nВеса классов: {class_weights.tolist()}")

    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * epochs
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
    best_epoch = 0
    best_state = None

    history = []

    # Цикл обучения
    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        print(f"\nЭпоха {epoch + 1}/{epochs}")

        for batch in train_loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            labels = batch.pop("labels")

            optimizer.zero_grad()

            outputs = model(**batch)

            loss = loss_function(outputs.logits, labels)

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

        validation_accuracy = accuracy_score(labels, predictions)

        print(f"Training loss: {average_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy:.4f}")
        print(f"Validation F1: {f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "training_loss": float(average_loss),
            "validation_accuracy": float(validation_accuracy),
            "validation_precision": float(precision),
            "validation_recall": float(recall),
            "validation_f1": float(f1),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Не удалось сохранить лучшую модель.")

    print(f"\nЛучшая эпоха: {best_epoch} (F1 на валидации {best_f1:.4f})")

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

    accuracy = accuracy_score(test_labels, test_predictions)

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

    # Как поведёт себя выбранный порог на данных, которых модель не видела
    test_threshold_accuracy, test_threshold_coverage = accuracy_at_threshold(
        test_labels,
        test_predictions,
        test_probabilities,
        threshold_info["threshold"],
    )

    # Измерение времени обработки одного сообщения
    # (так же, как в classifier.py: по одному, без padding)
    model.eval()

    timing_texts = test_data["text"].tolist()[:100]

    if timing_texts:
        with torch.no_grad():
            # Прогрев, чтобы первый вызов не искажал среднее
            warmup = tokenizer(
                timing_texts[0],
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(DEVICE)
            model(**warmup)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            for text in timing_texts:
                inputs = tokenizer(
                    clean_text(text),
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(DEVICE)

                model(**inputs)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        average_time = (
                               time.perf_counter() - start_time
                       ) / len(timing_texts)
    else:
        average_time = 0.0

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
        "max_length": max_length,
        "confidence_threshold": threshold_info["threshold"],
        "target_reached": threshold_info["target_reached"],
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

    # Предсказания на тесте: помогают разобрать, где модель ошибается
    predictions_frame = pd.DataFrame({
        "text": test_data["text"].tolist(),
        "true_label": [CLASS_NAMES[int(v)] for v in test_labels],
        "predicted_label": [CLASS_NAMES[int(v)] for v in test_predictions],
        "confidence": test_probabilities.max(axis=1).round(4),
    })

    predictions_frame.to_csv(
        RESULTS_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions_frame[
        predictions_frame["true_label"]
        != predictions_frame["predicted_label"]
        ].sort_values("confidence", ascending=False).to_csv(
        RESULTS_DIR / "misclassified_test.csv",
        index=False,
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
        "epochs": epochs,
        "best_epoch": best_epoch,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": max_length,
        "class_weights": class_weights.tolist(),
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "average_inference_time_seconds": float(average_time),
        "confidence_threshold": threshold_info,
        "test_accuracy_at_threshold": test_threshold_accuracy,
        "test_coverage_at_threshold": test_threshold_coverage,
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

    print("\nМатрица ошибок (строки — истина, столбцы — прогноз):")
    print(matrix_frame)

    print(
        f"\nСреднее время обработки сообщения: "
        f"{average_time * 1000:.2f} мс"
    )

    print(
        f"\nПорог уверенности (подобран на валидации): "
        f"{threshold_info['threshold']:.2f}"
    )

    print(
        f"Целевая Accuracy достигнута на валидации: "
        f"{threshold_info['target_reached']}"
    )

    if test_threshold_accuracy is None:
        print("На тесте при этом пороге нет ни одного принятого сообщения.")
    else:
        print(
            f"На тесте при пороге: Accuracy {test_threshold_accuracy:.4f}, "
            f"покрытие {test_threshold_coverage:.2%}"
        )

    print(f"\nМодель сохранена: {MODEL_DIR}")
    print(f"Результаты сохранены: {RESULTS_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение классификатора RuBERT."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    if arguments.epochs <= 0 or arguments.max_length <= 0:
        raise SystemExit("--epochs и --max-length должны быть положительными.")

    train_model(
        epochs=arguments.epochs,
        max_length=arguments.max_length,
        data_path=arguments.data,
    )
