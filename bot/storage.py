"""Хранение результатов анализа и выборки для личной и групповой статистики."""

import logging
import math
from datetime import date
from numbers import Real
import sqlite3
from pathlib import Path

import aiosqlite


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "db" / "results.db"
logger = logging.getLogger(__name__)


def _parse_iso_date(value: str, field: str) -> date:
    """Проверить реальную дату строго в формате YYYY-MM-DD."""
    if not isinstance(value, str):
        raise ValueError(f"{field} должна быть датой YYYY-MM-DD.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field} должна быть реальной датой YYYY-MM-DD.") from None
    if parsed.isoformat() != value:
        raise ValueError(f"{field} должна быть в формате YYYY-MM-DD.")
    return parsed


class StorageError(RuntimeError):
    """Ошибка операции хранилища без содержимого записываемых данных."""


class StorageInitializationError(StorageError):
    """Не удалось подготовить SQLite-базу с совместимой схемой."""


class Storage:
    REQUIRED_COLUMNS = {
        "id", "user_id", "group_name", "analysis_date",
        "predicted_class", "model_confidence", "advice",
    }

    def __init__(self, storage_file: str | Path = DEFAULT_DB_PATH):
        self.storage_file = Path(storage_file).expanduser().resolve()

    async def init_table(self) -> None:
        """Подготовить каталог, проверить схему и добавить индекс без удаления данных."""
        try:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.storage_file) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS results (
                        id INTEGER PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        group_name INTEGER NOT NULL,
                        analysis_date TEXT NOT NULL,
                        predicted_class INTEGER NOT NULL,
                        model_confidence REAL NOT NULL,
                        advice TEXT
                    )
                """)
                async with db.execute("PRAGMA table_info(results)") as cursor:
                    columns = {row[1]: row for row in await cursor.fetchall()}
                missing = self.REQUIRED_COLUMNS - columns.keys()
                if missing:
                    raise StorageInitializationError(
                        f"Несовместимая схема results в {self.storage_file}: "
                        f"отсутствуют обязательные колонки: {', '.join(sorted(missing))}. "
                        "Автоматическое изменение старой схемы не выполняется."
                    )
                # INSERT не передаёт id: SQLite должен генерировать его сам.
                primary_key = [row[1] for row in columns.values() if row[5]]
                if columns["id"][2].upper() != "INTEGER" or primary_key != ["id"]:
                    raise StorageInitializationError(
                        f"Несовместимая схема results в {self.storage_file}: "
                        "id должен быть единственным INTEGER PRIMARY KEY."
                    )
                unsupported = [
                    name for name, row in columns.items()
                    if name not in self.REQUIRED_COLUMNS and row[3] and row[4] is None
                ]
                if unsupported:
                    raise StorageInitializationError(
                        f"Несовместимая схема results в {self.storage_file}: "
                        "дополнительные обязательные колонки без значения по умолчанию: "
                        f"{', '.join(sorted(unsupported))}."
                    )
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_results_group_date
                    ON results (group_name, analysis_date)
                """)
                await db.commit()
        except (OSError, aiosqlite.Error) as error:
            raise StorageInitializationError(
                f"Не удалось инициализировать базу {self.storage_file}: {error}"
            ) from error
        logger.info("Хранилище результатов готово: %s", self.storage_file)

    async def save_result(
        self,
        user_id: int,
        group: int,
        analysis_date: str,
        predicted_class: int,
        model_confidence: float,
        advice: str,
    ) -> None:
        """Сохранить итог анализа без округления уверенности.

        Классы: 0 — Норма, 1 — Стресс, 2 — Тревога. Несколько анализов
        одного студента за день — отдельные записи. ValueError означает
        ошибку входных данных, StorageError — сбой SQLite. group хранится
        в group_name, дата — строго YYYY-MM-DD. Возврат None означает,
        что commit завершился успешно; порог уверенности задаёт решатель.
        """
        for name, value in (("user_id", user_id), ("group", group)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"save_result: {name} должен быть положительным целым числом.")
        _parse_iso_date(analysis_date, "save_result: analysis_date")
        if (isinstance(predicted_class, bool) or not isinstance(predicted_class, int)
                or predicted_class not in (0, 1, 2)):
            raise ValueError("save_result: predicted_class должен быть 0, 1 или 2.")
        if (isinstance(model_confidence, bool) or not isinstance(model_confidence, Real)
                or not 0 <= model_confidence <= 1 or not math.isfinite(model_confidence)):
            raise ValueError("save_result: model_confidence должна быть конечным числом от 0 до 1.")
        if not isinstance(advice, str) or not advice.strip():
            raise ValueError("save_result: advice должен быть непустой строкой.")

        operation = "connect"
        try:
            async with aiosqlite.connect(self.storage_file) as db:
                operation = "INSERT"
                await db.execute(
                    "INSERT INTO results (user_id, group_name, analysis_date, "
                    "predicted_class, model_confidence, advice) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, group, analysis_date, predicted_class, float(model_confidence), advice),
                )
                operation = "commit"
                await db.commit()
        except aiosqlite.Error as error:
            # Сообщения SQLite (например, из триггера) могут содержать данные.
            # Диагностика использует только безопасную категорию ошибки.
            reasons = {
                sqlite3.SQLITE_BUSY: "база занята",
                sqlite3.SQLITE_LOCKED: "база заблокирована",
                sqlite3.SQLITE_READONLY: "база доступна только для чтения",
                sqlite3.SQLITE_CANTOPEN: "не удалось открыть файл базы",
                sqlite3.SQLITE_FULL: "недостаточно места для записи",
                sqlite3.SQLITE_IOERR: "ошибка ввода-вывода",
                sqlite3.SQLITE_CONSTRAINT: "нарушено ограничение SQLite",
                sqlite3.SQLITE_CORRUPT: "повреждена база SQLite",
                sqlite3.SQLITE_NOTADB: "файл не является базой SQLite",
                sqlite3.SQLITE_ERROR: "ошибка SQL или схемы базы",
            }
            code = getattr(error, "sqlite_errorcode", sqlite3.SQLITE_ERROR) & 255
            reason = reasons.get(code, "ошибка SQLite")
            raise StorageError(f"save_result ({operation}): {reason}.") from None
        except OverflowError:
            raise StorageError(
                f"save_result ({operation}): целое число вне диапазона SQLite INTEGER."
            ) from None

    async def get_user_class_counts(self, user_id: int) -> dict[int, int]:
        """Количество итоговых оценок каждого класса только для этого пользователя."""
        async with aiosqlite.connect(self.storage_file) as db:
            async with db.execute(
                "SELECT predicted_class, COUNT(*) FROM results "
                "WHERE user_id = ? GROUP BY predicted_class",
                (user_id,),
            ) as cursor:
                return dict(await cursor.fetchall())

    async def get_user_stats(self, user_id: int) -> dict[int, int]:
        """Совместимость с прежними вызовами личной статистики."""
        return await self.get_user_class_counts(user_id)

    async def get_results(self, group: int, start_date: str, end_date: str) -> list[dict]:
        """Записи группы за включительный период, по дате и id.

        [] означает успешную выборку без записей; ошибки чтения передаются
        вызывающему коду. Агрегация выполняется вне Storage.
        """
        start = _parse_iso_date(start_date, "get_results: start_date")
        end = _parse_iso_date(end_date, "get_results: end_date")
        if start > end:
            raise ValueError("get_results: start_date должна быть не позже end_date.")
        async with aiosqlite.connect(self.storage_file) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT id, group_name, analysis_date, predicted_class, "
                "model_confidence, advice FROM results "
                "WHERE group_name = ? AND analysis_date BETWEEN ? AND ? "
                "ORDER BY analysis_date, id",
                (group, start_date, end_date),
            ) as cursor:
                return [dict(row) for row in await cursor.fetchall()]
