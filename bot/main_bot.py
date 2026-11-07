import asyncio
import logging
import os
import random
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command

# Импортируем то, что сделали девочки
from solver import Solver
from classifier import RuBertClassifier, ModelLoadError
from mock_model import MockModel
from knowledge_base import KnowledgeBase
from storage import Storage
from statistics_service import StatisticsService

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher()

# --- ИНИЦИАЛИЗАЦИЯ МОДЕЛИ И СЕРВИСОВ ---
def build_model():
    try:
        return RuBertClassifier()
    except ModelLoadError as error:
        logging.warning("RuBERT недоступен (%s). Используется MockModel.", error)
        return MockModel()

# Инициализируем всё, что нужно для работы
model = build_model()
kb = KnowledgeBase()
solver = Solver(model=model, knowledge_base=kb)

# Инициализируем хранилище для статистики
storage = Storage("db/psycho_bot.db")
stats_service = StatisticsService(storage)

# Память для диалога
user_states = {}

# --- КЛАВИАТУРА ---
def get_main_keyboard():
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text="Упражнения 🧘‍♀️"), types.KeyboardButton(text="Интересный факт 💡")],
            [types.KeyboardButton(text="Помощь ❓")]
        ],
        resize_keyboard=True
    )
    return keyboard

# --- КОМАНДЫ ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_states[message.from_user.id] = {
        "previous_text": "",
        "clarification_count": 0,
        "awaiting_clarification": False,
        "last_question_topic": None
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
        "Я анализирую текст с помощью нейросети RuBERT и подбираю рекомендацию."
    )
    await message.answer(text, parse_mode="Markdown")

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
    await message.answer(random.choice(facts))

# Команда статистики (используем код Тани)
@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    # Для теста берем группу 1 и текущую дату (можно потом доработать под ввод пользователя)
    # Таня написала асинхронный метод get_summary
    summary = await stats_service.get_summary(group_name=1, start_date="2026-11-01", end_date="2026-11-30")
    await message.answer(summary)


# --- ЛОГИКА ДИАЛОГА ---

@dp.message(~Command(commands=["start", "help", "exercise", "fact", "stats"]), F.text, ~F.text.in_({"Упражнения 🧘‍♀️", "Интересный факт 💡", "Помощь ❓"}))
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    text = message.text
    
    if user_id not in user_states:
        user_states[user_id] = {
            "previous_text": "",
            "clarification_count": 0,
            "awaiting_clarification": False,
            "last_question_topic": None
        }
        
    state = user_states[user_id]
    
    # Если бот ждет уточнения, склеиваем текст
    if state.get("awaiting_clarification"):
        combined_text = f"{state['previous_text']} {text}"
        state["clarification_count"] += 1
        state["awaiting_clarification"] = False
        await process_solver(message, combined_text, state)
        return

    # Обычная обработка
    state["clarification_count"] = 0
    state["previous_text"] = text
    await process_solver(message, text, state)

async def process_solver(message: types.Message, text: str, state: dict):
    # Вызываем Solver Тани!
    result = solver.solve(text)
    
    # Проверяем, уверена ли модель
    if result.confidence < 0.60 and state["clarification_count"] < 2:
        state["awaiting_clarification"] = True
        
        if state["clarification_count"] == 0:
            if result.predicted_class == 1:
                question = "Я замечаю признаки стресса. Это связано с учебой или с чем-то личным?"
                state["last_question_topic"] = "stress"
            elif result.predicted_class == 2:
                question = "Похоже на тревогу. Ты переживаешь из-за экзаменов или есть другая причина?"
                state["last_question_topic"] = "anxiety"
            else:
                question = "Ты кажешься спокойным. Что-то происходит в последнее время?"
                state["last_question_topic"] = "norm"
        else:
            question = "Я все еще не совсем понимаю. Попробуй описать чувства подробнее ('я устал', 'боюсь')."
            state["last_question_topic"] = None
            
        await message.answer(f"🤔 {question}")
        return
    
    # Если уверенность высокая, выдаем результат (используем result.to_message() от Тани)
    state["awaiting_clarification"] = False
    state["last_question_topic"] = None
    
    # Сохраняем в БД (функция из кода Тани)
    try:
        # В коде Тани Storage.save_result асинхронный
        await storage.save_result(
            user_id=message.from_user.id,
            group=1, # Заглушка, нужна группа студента
            analysis_date="2026-11-07", # Заглушка, нужна реальная дата
            predicted_class=result.predicted_class,
            model_confidence=result.confidence,
            advice=result.advice
        )
    except Exception as e:
        logging.error(f"Ошибка сохранения в БД: {e}")

    # Отправляем ответ студенту
    await message.answer(result.to_message(), parse_mode="Markdown")

# --- ЗАПУСК ---
async def main():
    # Инициализируем таблицу БД при старте
    await storage.init_table()
    
    logging.basicConfig(level=logging.INFO)
    print("Бот запущен! Жду сообщений в Telegram...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")