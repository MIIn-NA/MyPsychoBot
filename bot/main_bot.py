import asyncio
import logging
import os
import random
import datetime
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command

# Импортируем модули команды
from solver import Solver, SolverError, TextTooLongError
from mock_model import MockModel
from knowledge_base import KnowledgeBase
from storage import Storage, StorageError, StorageInitializationError
from statistics_service import StatisticsService

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher()

# --- ИНИЦИАЛИЗАЦИЯ ---
def build_model():
    if os.getenv("BOT_DEMO_MODE", "").strip() == "1":
        logging.warning("Включён демонстрационный режим MockModel.")
        return MockModel()
    # Импорт отложен: бот может ответить о недоступности анализа даже без
    # установленных зависимостей RuBERT.
    try:
        from classifier import RuBertClassifier, ModelLoadError
    except ImportError:
        logging.error("Недоступны зависимости RuBERT.")
        return None
    try:
        return RuBertClassifier()
    except ModelLoadError:
        logging.error("RuBERT не удалось загрузить.")
        return None

model = build_model()
kb = KnowledgeBase()
solver = (
    Solver(model=model, knowledge_base=kb,
           threshold=getattr(model, "low_confidence_threshold", 0.60))
    if model is not None else None
)

# Запись, личная статистика и StatisticsService используют одну базу.
storage = Storage()
stats_service = StatisticsService(storage)
DEFAULT_GROUP_ID = 1
GROUP_REPORT_DAYS = 30
STATISTICS_BUTTON = "Статистика 📊"
STATISTICS_BUTTON_TEXTS = {STATISTICS_BUTTON, "Моя статистика 📊"}

user_states = {}

# --- КЛАВИАТУРЫ ---

def get_main_keyboard():
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text="Упражнения 🧘‍♀️"), types.KeyboardButton(text="Интересный факт 💡")],
            [types.KeyboardButton(text="Помощь ❓"), types.KeyboardButton(text=STATISTICS_BUTTON)]
        ],
        resize_keyboard=True
    )
    return keyboard

def get_context_keyboard():
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text="Из-за учебы (сессия)")],
            [types.KeyboardButton(text="Из-за отношений")],
            [types.KeyboardButton(text="Проблемы со сном / недосып")],
            [types.KeyboardButton(text="Будущее (работа, планы)")],
            [types.KeyboardButton(text="Просто выгорание")],
            [types.KeyboardButton(text="Отмена")]
        ],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    return keyboard

# --- КОМАНДЫ ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_states[message.from_user.id] = {
        "previous_text": "",
        "clarification_count": 0,
        "awaiting_clarification": False,
        "awaiting_feedback": False
    }
    await message.answer(
        "Привет! Я бот для мониторинга психоэмоционального состояния.\n"
        "Напиши, как ты себя чувствуешь, или выбери действие на клавиатуре ниже 👇",
        reply_markup=get_main_keyboard()
    )

@dp.message(Command("help"))
@dp.message(F.text == "Помощь ❓")
async def cmd_help(message: types.Message):
    text = (
        "📝 **Как я работаю:**\n"
        "Напиши мне любое сообщение о том, как ты себя чувствуешь.\n"
        "Например: «Я устал от учебы» или «Боюсь завалить экзамен».\n\n"
        "Я проанализирую текст с помощью нейросети и подберу рекомендацию. "
        "Если я не совсем пойму твое состояние, я задам уточняющий вопрос."
    )
    await message.answer(text, parse_mode="Markdown", reply_markup=get_main_keyboard())

@dp.message(Command("exercise"))
@dp.message(F.text == "Упражнения 🧘‍♀️")
async def cmd_exercise(message: types.Message):
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=[
        [types.InlineKeyboardButton(text="🫁 Дыхание 4-7-8", callback_data="ex_breath")],
        [types.InlineKeyboardButton(text="🖐 Техника заземления", callback_data="ex_ground")],
        [types.InlineKeyboardButton(text="💪 Релаксация мышц", callback_data="ex_muscle")]
    ])
    await message.answer("Выбери упражнение, которое поможет снять напряжение:", reply_markup=keyboard)

@dp.callback_query(F.data.startswith('ex_'))
async def process_exercise(callback: types.CallbackQuery):
    exercise_type = callback.data.split('_')[1]
    if exercise_type == "breath":
        text = "🫁 **Дыхание 4-7-8**\n\n1. Вдох носом на 4 сек.\n2. Задержка на 7 сек.\n3. Выдох ртом на 8 сек.\n\nПовтори 4 раза."
    elif exercise_type == "ground":
        text = "🖐 **Заземление 5-4-3-2-1**\n\nНайди: 5 вещей (видишь), 4 (трогаешь), 3 (слышишь), 2 (чувствуешь запах), 1 (вкус)."
    else:
        text = "💪 **Релаксация**\n\nНапряги кулаки на 5 сек, расслабь на 10. Затем плечи, лицо, живот, ноги."

    await callback.message.edit_text(text, parse_mode="Markdown")
    await callback.answer()

@dp.message(Command("fact"))
@dp.message(F.text == "Интересный факт 💡")
async def cmd_fact(message: types.Message):
    facts = [
        "🧠 10 минут смеха сжигают столько же калорий, сколько 15 минут ходьбы.",
        "☕ Кофеин не дает энергии, он просто блокирует рецепторы усталости.",
        "💤 Во время сна мозг очищается от токсинов.",
        "🚶 5 минут прогулки на свежем воздухе снижают тревожность."
    ]
    await message.answer(random.choice(facts), reply_markup=get_main_keyboard())

@dp.message(Command("stats"))
async def cmd_my_stats(message: types.Message):
    user_id = message.from_user.id
    try:
        counts = await storage.get_user_class_counts(user_id)
        total = sum(counts.values())
    except Exception as error:
        logging.error("Ошибка чтения личной статистики (%s).", type(error).__name__)
        await message.answer("Ошибка при чтении базы данных.", reply_markup=get_main_keyboard())
        return

    if total == 0:
        await message.answer("У вас пока нет сохраненных результатов. Напишите мне, как вы себя чувствуете!", reply_markup=get_main_keyboard())
        return

    norm = counts.get(0, 0)
    stress = counts.get(1, 0)
    anxiety = counts.get(2, 0)
    text = (
        "📊 Твоя личная статистика\n\n"
        f"Сохранено анализов: {total}\n\n"
        f"🟢 Норма: {norm} ({int(norm/total*100)}%)\n"
        f"🟡 Стресс: {stress} ({int(stress/total*100)}%)\n"
        f"🔴 Тревога: {anxiety} ({int(anxiety/total*100)}%)"
    )
    await message.answer(text, reply_markup=get_main_keyboard())


@dp.message(F.text.in_(STATISTICS_BUTTON_TEXTS))
async def show_statistics(message: types.Message):
    if message.chat.type != "private":
        await message.answer("Откройте личный чат с ботом, чтобы посмотреть статистику.")
        return

    await cmd_my_stats(message)
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=GROUP_REPORT_DAYS - 1)
    try:
        report = await stats_service.get_report(
            DEFAULT_GROUP_ID, start_date.isoformat(), today.isoformat(),
        )
    except Exception as error:
        logging.error("Ошибка формирования общей статистики (%s).", type(error).__name__)
        await message.answer("Не удалось сформировать общую статистику. Попробуйте позже.")
        return

    if report.chart_png is None:
        await message.answer(report.summary)
        return

    await message.answer_photo(
        photo=types.BufferedInputFile(report.chart_png, filename="group_statistics.png"),
        caption=report.summary,
    )


# --- ОБРАБОТКА КНОПОК КОНТЕКСТА ---

@dp.message(F.text.in_({"Из-за учебы (сессия)", "Из-за отношений", "Проблемы со сном / недосып", "Будущее (работа, планы)", "Просто выгорание", "Отмена"}))
async def handle_context_button(message: types.Message):
    user_id = message.from_user.id
    text = message.text
    state = user_states.get(user_id, {"previous_text": "", "clarification_count": 1, "awaiting_clarification": False, "awaiting_feedback": False})

    user_states[user_id] = state
    if text == "Отмена":
        state["awaiting_clarification"] = False
        state["awaiting_feedback"] = False
        state["previous_text"] = ""
        state["clarification_count"] = 0
        await message.answer("Хорошо, давай вернемся к началу.", reply_markup=get_main_keyboard())
        return

    if not state.get("awaiting_clarification"):
        await message.answer("Напишите новое описание своего состояния.", reply_markup=get_main_keyboard())
        return

    # Контекст уточняет текст, а класс по-прежнему определяет модель.
    combined_text = f"{state['previous_text']} {text}".strip()
    state["clarification_count"] += 1
    await process_solver(message, combined_text, state)


# --- ОСНОВНАЯ ЛОГИКА ДИАЛОГА ---

@dp.message(~Command(commands=["start", "help", "exercise", "fact", "stats"]), F.text, ~F.text.in_({"Упражнения 🧘‍♀️", "Интересный факт 💡", "Помощь ❓"} | STATISTICS_BUTTON_TEXTS))
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    text = message.text

    if user_id not in user_states:
        user_states[user_id] = {"previous_text": "", "clarification_count": 0, "awaiting_clarification": False, "awaiting_feedback": False}

    state = user_states[user_id]
    if not isinstance(text, str) or not text.strip():
        await message.answer(
            "Сообщение пустое. Напишите, что происходит и как вы себя чувствуете.",
            reply_markup=get_context_keyboard() if state.get("awaiting_clarification") else get_main_keyboard(),
        )
        return

    # Для общего описания сначала просим контекст, чтобы не делать вывод по одной фразе.
    normalized = " ".join(text.casefold().replace("ё", "е").split()).strip(" .,!?:…")
    # Только самостоятельный короткий ответ считаем отзывом. Описание состояния
    # (в том числе содержащее «да» или «лучше») должно попадать в решатель.
    if state.get("awaiting_feedback"):
        state["awaiting_feedback"] = False
        if normalized in {
            "да", "да, спасибо", "спасибо", "спасибо большое", "помогло",
            "легче", "лучше", "стало легче", "стало лучше",
            "да, стало легче", "да, стало лучше", "мне стало легче",
            "мне стало лучше",
        }:
            await message.answer(
                "Рад, что тебе стало легче. Если захочешь обсудить что-то ещё, напиши.",
                reply_markup=get_main_keyboard(),
            )
            return
        if normalized in {
            "нет", "не помогло", "не помогает", "не легче", "не лучше",
            "не стало легче", "не стало лучше", "мне не стало легче",
            "мне не стало лучше", "нет, не помогло", "нет, не стало легче",
            "нет, не стало лучше", "стало хуже", "хуже", "хочу рассказать подробнее",
        }:
            state.update(previous_text="", clarification_count=0,
                         awaiting_clarification=True)
            await message.answer(
                "Расскажи, что сейчас беспокоит сильнее всего, или выбери тему ниже. "
                "Я проанализирую это описание и подберу совет.",
                reply_markup=get_context_keyboard(),
            )
            return

    if not state.get("awaiting_clarification") and normalized in {
        "плохо", "все плохо", "мне плохо", "очень плохо", "мне очень плохо",
        "все очень плохо", "тяжело", "мне тяжело",
    }:
        state.update(previous_text=text, clarification_count=0,
                     awaiting_clarification=True, awaiting_feedback=False)
        await message.answer(
            "Похоже, тебе сейчас нелегко. Чтобы я мог дать более подходящий совет, "
            "выбери, что сейчас больше всего давит, или расскажи чуть подробнее.",
            reply_markup=get_context_keyboard(),
        )
        return

    # Если ждем уточнения (пользователь ввел текст вместо кнопки)
    if state.get("awaiting_clarification"):
        combined_text = f"{state['previous_text']} {text}".strip()
        state["clarification_count"] += 1
        state["awaiting_clarification"] = False
        await process_solver(message, combined_text, state)
        return

    state["clarification_count"] = 0
    state["previous_text"] = text
    await process_solver(message, text, state)


async def process_solver(message: types.Message, text: str, state: dict):
    if solver is None:
        state.update(previous_text="", awaiting_clarification=False, awaiting_feedback=False,
                     clarification_count=0)
        await message.answer("Анализ сейчас недоступен. Попробуйте позже.", reply_markup=get_main_keyboard())
        return
    try:
        result = solver.solve(text)
    except TextTooLongError as error:
        # Повторный ввод заменяет всё слишком длинное объединённое описание.
        state.update(previous_text="", awaiting_clarification=False, awaiting_feedback=False,
                     clarification_count=0)
        await message.answer(str(error), reply_markup=get_main_keyboard())
        return
    except (TypeError, ValueError):
        await message.answer(
            "Сообщение пустое или некорректное. Напишите, что происходит и как вы себя чувствуете.",
            reply_markup=get_main_keyboard(),
        )
        return
    except SolverError:
        logging.error("Ошибка предсказания модели; анализ недоступен.")
        state.update(previous_text="", awaiting_clarification=False, awaiting_feedback=False,
                     clarification_count=0)
        await message.answer("Анализ сейчас недоступен. Попробуйте позже.", reply_markup=get_main_keyboard())
        return

    first_preliminary = result.is_preliminary and state.get("clarification_count", 0) == 0
    response = result.to_message(offer_clarification=first_preliminary)
    if os.getenv("BOT_DEMO_MODE", "").strip() == "1":
        response = "Демонстрационный режим (MockModel).\n\n" + response
    if result.is_preliminary:
        state["awaiting_feedback"] = False
        state["awaiting_clarification"] = first_preliminary
        if first_preliminary:
            state["previous_text"] = text
            response += "\nМожно выбрать тему уточнения кнопкой ниже."
        else:
            state["previous_text"] = ""
            state["clarification_count"] = 0
            response += "\n\nОценка остаётся предварительной. Напишите новое сообщение, чтобы начать новый анализ."
        await message.answer(
            response, parse_mode="Markdown",
            reply_markup=get_context_keyboard() if first_preliminary else get_main_keyboard(),
        )
        return

    state.update(previous_text="", awaiting_clarification=False, awaiting_feedback=True,
                 clarification_count=0)
    response += (
        "\n\nПомог ли совет? Можно ответить «стало легче» или «не помогло». "
        "Если хочешь рассказать подробнее, напиши, что происходит — я проанализирую новое описание."
    )
    try:
        today = datetime.date.today().isoformat()
        # Ограничение: до подключения учебных групп все записи относятся к группе 1.
        await storage.save_result(
            user_id=message.from_user.id, group=DEFAULT_GROUP_ID, analysis_date=today,
            predicted_class=result.predicted_class, model_confidence=result.confidence, advice=result.advice
        )
    except StorageError as error:
        logging.error("Ошибка записи результата в SQLite: %s", error)
    except Exception as error:
        logging.error("Ошибка записи результата в SQLite (%s).", type(error).__name__)

    await message.answer(response, parse_mode="Markdown", reply_markup=get_main_keyboard())


# --- ЗАПУСК ---
async def main():
    logging.basicConfig(level=logging.INFO)
    try:
        await storage.init_table()
    except StorageInitializationError as error:
        logging.error("Запуск бота отменён: %s", error)
        return
    print("Бот запущен! Жду сообщений...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")
