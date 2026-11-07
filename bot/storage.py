# Ответственная: Таня.
# Назначение: хранение результатов мониторинга.
#
# Что нужно реализовать:
# - Создание хранилища и структуры данных.
# - Сохранение идентификатора пользователя, учебной группы, даты анализа,
#   класса состояния, уверенности модели и выбранной рекомендации.
# - Получение результатов по учебной группе и выбранному периоду.
# - Обработку ошибок сохранения и чтения данных.
#
# Решатель передаёт сюда результаты анализа, а модуль статистики
# получает отсюда записи для расчётов.
import aiosqlite
from datetime import datetime

class Storage:

    def __init__(self, storage_file: str):
        self.storage_file = storage_file

    async def init_table(self) -> None:
        """
        Инициализация таблицы
        """
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

            await db.commit()
            print('Таблица успешно инициализирована')

    async def save_result(self,
                     user_id: int,
                     group: int,
                     analysis_date: str,
                     predicted_class: int, 
                     model_confidence: float,
                     advice: str
                    ) -> None:
        """
        Сохранение результатов анализа в хранилище.

        analysis_date - дата в формате ISO 8601: ГГГГ-ММ-ДД(например, "2026-11-05"). \n
        из datetime.date перевод через isoformat() \n
        для обратного преобразования - date.fromisoformat().
        """
        async with aiosqlite.connect(self.storage_file) as db:
            await db.execute(
                "INSERT INTO results (user_id, group_name, analysis_date, predicted_class, model_confidence, advice) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, group, analysis_date, predicted_class, model_confidence, advice)
            )
            await db.commit()
            print(f'Результат успешно сохранён в БД: (user_id: {user_id}, analysis_date: {analysis_date})')

    async def get_results(self,
                         group: int,
                         start_date: str, 
                         end_date: str):
        """
        Получение результатов по учебной группе и выбранному периоду.
        """
        async with aiosqlite.connect(self.storage_file) as db:
            db.row_factory = aiosqlite.Row
            query = """
                SELECT id, group_name, analysis_date, predicted_class,
                       model_confidence, advice
                FROM results as r
                WHERE r.group_name=? AND r.analysis_date BETWEEN ? AND ?
            """
            async with db.execute(query, (group, start_date, end_date)) as cursor:
                results = [dict(row) for row in await cursor.fetchall()]
        
            return results
