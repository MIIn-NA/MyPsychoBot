import asyncio
import logging
import os
import random
import datetime
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command

# Импортируем модули команды
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

# --- ИНИЦИАЛИЗАЦИЯ ---
def build_model():
    try:
        return RuBertClassifier()
    except ModelLoadError as error:
        logging.warning("RuBERT недоступен (%s). Используется MockModel.", error)
        return MockModel()

model = build_model()
kb = KnowledgeBase()
solver = Solver(model=model, knowledge_base=kb)

storage = Storage("psycho_bot.db")
stats_service = StatisticsService(storage)

user_states = {}

# --- КЛАВИАТУРЫ ---

def get_main_keyboard():
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text="Упражнения 🧘‍♀️"), types.KeyboardButton(text="Интересный факт 💡")],
            [types.KeyboardButton(text="Помощь ❓"), types.KeyboardButton(text="Моя статистика 📊")]
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
@dp.message(F.text == "Моя статистика 📊")
async def cmd_my_stats(message: types.Message):
    user_id = message.from_user.id
    import aiosqlite
    try:
        async with aiosqlite.connect("psycho_bot.db") as db:
            query = "SELECT predicted_class, COUNT(*) FROM results WHERE user_id = ? GROUP BY predicted_class"
            async with db.execute(query, (user_id,)) as cursor:
                stats = dict(await cursor.fetchall())
            async with db.execute("SELECT COUNT(*) FROM results WHERE user_id = ?", (user_id,)) as cursor:
                total = (await cursor.fetchone())[0]
    except Exception:
        await message.answer("Ошибка при чтении базы данных.", reply_markup=get_main_keyboard())
        return

    if total == 0:
        await message.answer("У вас пока нет сохраненных результатов. Напишите мне, как вы себя чувствуете!", reply_markup=get_main_keyboard())
        return

    norm = stats.get(0, 0)
    stress = stats.get(1, 0)
    anxiety = stats.get(2, 0)
    
    text = (
        "📊 **Твоя личная статистика**\n\n"
        f"Всего проанализировано сообщений: **{total}**\n\n"
        f"🟢 Норма: {norm} ({int(norm/total*100)}%)\n"
        f"🟡 Стресс: {stress} ({int(stress/total*100)}%)\n"
        f"🔴 Тревога: {anxiety} ({int(anxiety/total*100)}%)\n\n"
        "💡 *Ты можешь показать эту статистику психологу.*"
    )
    await message.answer(text, parse_mode="Markdown", reply_markup=get_main_keyboard())


# --- ОБРАБОТКА КНОПОК КОНТЕКСТА ---

@dp.message(F.text.in_({"Из-за учебы (сессия)", "Из-за отношений", "Проблемы со сном / недосып", "Будущее (работа, планы)", "Просто выгорание", "Отмена"}))
async def handle_context_button(message: types.Message):
    user_id = message.from_user.id
    text = message.text
    state = user_states.get(user_id, {"previous_text": "", "clarification_count": 1, "awaiting_clarification": False, "awaiting_feedback": False})

    if text == "Отмена":
        await message.answer("Хорошо, давай вернемся к началу.", reply_markup=get_main_keyboard())
        return

    # Принудительно задаем класс и текст для базы знаний
    if text == "Из-за учебы (сессия)":
        predicted_class = 1
        kb_text = "экзамен сессия дедлайн учеба"
    elif text == "Из-за отношений":
        predicted_class = 2
        kb_text = "отношения конфликт"
    elif text == "Проблемы со сном / недосып":
        predicted_class = 2
        kb_text = "сон бессонница"
    elif text == "Будущее (работа, планы)":
        predicted_class = 2
        kb_text = "будущее работа"
    else: # Просто выгорание
        predicted_class = 1
        kb_text = "устал выгорание"

    # Получаем совет из базы знаний напрямую
    advice = kb.get_recommendation(kb_text, predicted_class)
    
    # Собираем ответ вручную (не трогаем объект SolverResult)
    class_names = {0: "Норма", 1: "Стресс", 2: "Тревога"}
    state_name = class_names[predicted_class]
    
    response = f"Я определил твое состояние как: **{state_name}**\n\n"
    response += f"💡 Совет: {advice}\n\n"
    response += "Стало ли тебе немного легче после этого совета? Или хочешь рассказать подробнее?"
    
    await message.answer(response, parse_mode="Markdown", reply_markup=get_main_keyboard())
    
    # Сохраняем в БД
    try:
        today = datetime.date.today().isoformat()
        await storage.save_result(
            user_id=message.from_user.id, group=1, analysis_date=today,
            predicted_class=predicted_class, model_confidence=1.0, advice=advice
        )
    except Exception as e:
        logging.error(f"Ошибка БД: {e}")
        
    # Ставим флаг, что ждем фидбек на совет
    state["awaiting_feedback"] = True


# --- ОСНОВНАЯ ЛОГИКА ДИАЛОГА ---

@dp.message(~Command(commands=["start", "help", "exercise", "fact", "stats"]), F.text, ~F.text.in_({"Упражнения 🧘‍♀️", "Интересный факт 💡", "Помощь ❓", "Моя статистика 📊"}))
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    text = message.text.lower()
    
    if user_id not in user_states:
        user_states[user_id] = {"previous_text": "", "clarification_count": 0, "awaiting_clarification": False, "awaiting_feedback": False}
        
    state = user_states[user_id]
    
    # Если ждем реакции на совет
    if state.get("awaiting_feedback"):
        state["awaiting_feedback"] = False 
        if any(word in text for word in ["да", "легче", "лучше", "спасибо"]):
            await message.answer("Я очень рад, что смог помочь! Если что-то еще случится, я всегда на связи. 😊", reply_markup=get_main_keyboard())
        else:
            await message.answer("Мне жаль, что всё еще тяжело. Иногда нужно время. Попробуй выполнить упражнение из меню или написать мне еще раз, чтобы я подобрал другой совет.", reply_markup=get_main_keyboard())
        return
    
    # Если ждем уточнения (пользователь ввел текст вместо кнопки)
    if state.get("awaiting_clarification"):
        combined_text = f"{state['previous_text']} {text}"
        state["clarification_count"] += 1
        state["awaiting_clarification"] = False
        await process_solver(message, combined_text, state)
        return

    state["clarification_count"] = 0
    state["previous_text"] = text
    await process_solver(message, text, state)


async def process_solver(message: types.Message, text: str, state: dict):
    # Вызываем Солвер Тани
    result = solver.solve(text)
    
    predicted_class = result.predicted_class
    confidence = result.confidence
    
    # Защита от ошибок модели
    text_lower = text.lower()
    negative_words = ["плохо", "сложно", "тяжело", "ужасно", "не могу", "достало", "бесит", "устал", "не хочу", "грустно", "тревожно", "бессонница", "учеб", "сесси", "экзамен", "дедлайн"]
    
    if any(word in text_lower for word in negative_words):
        if "сон" in text_lower or "бессонница" in text_lower:
            predicted_class = 2
        elif "учеб" in text_lower or "сесси" in text_lower or "экзамен" in text_lower or "дедлайн" in text_lower:
            predicted_class = 1
        else:
            predicted_class = 1
        confidence = 0.50
    
    # Если уверенность низкая, спрашиваем кнопками
    if confidence < 0.60 and state.get("clarification_count", 0) == 0:
        state["awaiting_clarification"] = True
        state["previous_text"] = text
        
        if predicted_class in [1, 2]:
            question = ("Я замечаю, что тебе сейчас нелегко. Чтобы я мог дать более подходящий совет, "
                        "выбери, что сейчас больше всего давит:")
            await message.answer(question, reply_markup=get_context_keyboard())
        else:
            question = ("Ты кажешься спокойным, но я хочу убедиться. Что-то происходит в последнее время?")
            await message.answer(question, reply_markup=get_main_keyboard())
        return
    
    # Выдаем результат
    state["awaiting_clarification"] = False
    state["awaiting_feedback"] = True
    
    try:
        today = datetime.date.today().isoformat()
        await storage.save_result(
            user_id=message.from_user.id, group=1, analysis_date=today,
            predicted_class=predicted_class, model_confidence=confidence, advice=result.advice
        )
    except Exception as e:
        logging.error(f"Ошибка БД: {e}")

    class_names = {0: "Норма", 1: "Стресс", 2: "Тревога"}
    state_name = class_names.get(predicted_class, "Неизвестно")
    
    response = f"Я определил твое состояние как: **{state_name}** (уверенность: {int(confidence*100)}%)\n\n"
    response += f"💡 Совет: {result.advice}\n\n"
    response += "Стало ли тебе немного легче после этого совета? Или хочешь рассказать подробнее?"
    
    await message.answer(response, parse_mode="Markdown", reply_markup=get_main_keyboard())


# --- ЗАПУСК ---
async def main():
    await storage.init_table()
    logging.basicConfig(level=logging.INFO)
    print("Бот запущен! Жду сообщений...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")