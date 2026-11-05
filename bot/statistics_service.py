# Ответственная: Таня.
# Назначение: агрегированная статистика и графики для психолога.
#
# Что нужно реализовать:
# - Получение сохранённых результатов по учебной группе и периоду.
# - Подсчёт количества и доли состояний: норма, стресс, тревога.
# - Расчёт изменения распределения состояний во времени.
# - Формирование графика и краткого текстового отчёта для Telegram.
# - Сообщение об отсутствии данных, если записей за период нет.
#
# Статистика формируется без раскрытия личных сообщений студентов.
# Настя подключает этот модуль к командам и кнопкам бота и ограничивает
# доступ к ним ролью психолога. Код пока не реализован.

from storage import Storage

class StatisticsService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    async def get_summary(self,
                        group_name: int,
                        start_date: str,
                        end_date: str) -> str:
        """Составить текстовый отчёт по числу анализов за выбранный период."""
        results = await self.storage.get_results(group_name, start_date, end_date)

        if not results:
            return f"Группа {group_name}: за период {start_date} — {end_date} данных нет."

        class_names = {0: "Норма", 1: "Стресс", 2: "Тревога"}
        counts = {class_id: 0 for class_id in class_names}
        for record in results:
            class_id = record["predicted_class"]
            if class_id not in counts:
                raise ValueError(f"Неизвестный класс состояния: {class_id}")
            counts[class_id] += 1

        total = len(results)
        lines = [
            f"Статистика группы {group_name} за период {start_date} — {end_date}",
            f"Всего анализов: {total}",
        ]
        for class_id, name in class_names.items():
            percentage = counts[class_id] / total * 100
            lines.append(f"{name}: {counts[class_id]} ({percentage:.1f}%)")

        return "\n".join(lines)
