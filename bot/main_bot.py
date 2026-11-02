import asyncio
import logging
import os
import random
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandObject

# Импортируем наши модули
from real_model import RealModel
from knowledge_base import KnowledgeBase

# Загружаем токен
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

# Инициализация бота
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Инициализация модели и базы
model = RealModel()
kb = KnowledgeBase()

# --- БАЗА ДАННЫХ ---
def init_db():
    conn = sqlite3.connect('psycho_bot.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            text TEXT,
            predicted_class INTEGER,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_result(user_id, username, text, predicted_class, confidence):
    conn = sqlite3.connect('psycho_bot.db')
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO results (user_id, username, text, predicted_class, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, username, text, predicted_class, confidence, timestamp))
    conn.commit()
    conn.close()

init_db()
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
        "Я анализирую текст с помощью нейросети RuBERT и подбираю рекомендацию.\n\n"
        "Ты также можешь использовать кнопки внизу экрана."
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

@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    conn = sqlite3.connect('psycho_bot.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM results")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT predicted_class, COUNT(*) FROM results GROUP BY predicted_class")
    stats = dict(cursor.fetchall())
    conn.close()
    
    norm = stats.get(0, 0)
    stress = stats.get(1, 0)
    anxiety = stats.get(2, 0)
    
    text = (
        "📊 **Статистика мониторинга**\n\n"
        f"Всего сообщений: **{total}**\n\n"
        f"🟢 Норма: {norm} ({int(norm/total*100) if total else 0}%)\n"
        f"🟡 Стресс: {stress} ({int(stress/total*100) if total else 0}%)\n"
        f"🔴 Тревога: {anxiety} ({int(anxiety/total*100) if total else 0}%)"
    )
    await message.answer(text, parse_mode="Markdown")


# --- ЛОГИКА ДИАЛОГА (ТОЛЬКО ДЛЯ ОБЫЧНОГО ТЕКСТА) ---
# Самая главная строчка: мы явно говорим НЕ обрабатывать команды и кнопки
@dp.message(~Command(commands=["start", "help", "exercise", "fact", "stats"]), F.text, ~F.text.in_({"Упражнения 🧘‍♀️", "Интересный факт 💡", "Помощь ❓"}))
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    text = message.text.lower()
    
    if user_id not in user_states:
        user_states[user_id] = {
            "previous_text": "",
            "clarification_count": 0,
            "awaiting_clarification": False,
            "last_question_topic": None
        }
        
    state = user_states[user_id]
    
    if state.get("awaiting_clarification"):
        combined_text = f"{state['previous_text']} {text}"
        state["clarification_count"] += 1
        state["awaiting_clarification"] = False
        await analyze_and_respond(message, combined_text, state)
        return

    state["clarification_count"] = 0
    state["previous_text"] = text
    await analyze_and_respond(message, text, state)

async def analyze_and_respond(message: types.Message, text: str, state: dict):
    predicted_class, confidence = model.predict(text)
    
    # Логика ответов на уточнения
    if state.get("last_question_topic"):
        topic = state["last_question_topic"]
        if any(word in text for word in ["да", "учеба", "учеб", "лаба", "экзамен", "сессия"]):
            if topic == "stress":
                predicted_class = 1
                confidence = 0.95
            elif topic == "anxiety":
                predicted_class = 2
                confidence = 0.95
                
            state["last_question_topic"] = None
            advice = kb.get_recommendation(text, predicted_class)
            class_names = {0: "Норма", 1: "Стресс", 2: "Тревога"}
            state_name = class_names[predicted_class]
            
            save_result(message.from_user.id, message.from_user.username, text, predicted_class, confidence)
            
            response = f"Понял тебя. Твое состояние: **{state_name}** (уверенность: {int(confidence*100)}%)\n\n"
            response += f"💡 Совет: {advice}"
            await message.answer(response, parse_mode="Markdown")
            return

    # Если модель не уверена
    if confidence < 0.60 and state["clarification_count"] < 2:
        state["awaiting_clarification"] = True
        
        if state["clarification_count"] == 0:
            if predicted_class == 1:
                question = "Я замечаю признаки стресса. Это связано с учебой или с чем-то личным?"
                state["last_question_topic"] = "stress"
            elif predicted_class == 2:
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
    
    # Выдача результата
    state["awaiting_clarification"] = False
    state["last_question_topic"] = None
    
    save_result(message.from_user.id, message.from_user.username, text, predicted_class, confidence)
    
    advice = kb.get_recommendation(text, predicted_class)
    class_names = {0: "Норма", 1: "Стресс", 2: "Тревога"}
    state_name = class_names[predicted_class]
    
    response = f"Я определил твое состояние как: **{state_name}** (уверенность: {int(confidence*100)}%)\n\n"
    response += f"💡 Совет: {advice}"
    
    await message.answer(response, parse_mode="Markdown")

# --- ЗАПУСК ---
async def main():
    logging.basicConfig(level=logging.INFO)
    print("Бот запущен! Жду сообщений в Telegram...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")