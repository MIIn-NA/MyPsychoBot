import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

from real_model import RealModel
from knowledge_base import KnowledgeBase
import os
from dotenv import load_dotenv

load_dotenv() 

TOKEN = os.getenv("BOT_TOKEN")

# Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher()

model = RealModel()
kb = KnowledgeBase()

# Обработчик команды /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я бот для мониторинга психоэмоционального состояния. Напиши, как ты себя чувствуешь.")

# Обработчик всех текстовых сообщений
@dp.message()
async def handle_message(message: types.Message):
    text = message.text
    
    # 1. Получаем предсказание от модели
    predicted_class, confidence = model.predict(text)
    
    # 2. Получаем рекомендацию из базы знаний
    advice = kb.get_recommendation(text, predicted_class)
    
    # 3. Формируем ответ
    class_names = {0: "Норма", 1: "Стресс", 2: "Тревога"}
    state_name = class_names[predicted_class]
    
    response = f"Я определил твое состояние как: **{state_name}** (уверенность: {int(confidence*100)}%)\n\n"
    response += f"💡 Совет: {advice}"
    
    # Отправляем ответ студенту
    await message.answer(response, parse_mode="Markdown")

# Запуск бота
async def main():
    logging.basicConfig(level=logging.INFO)
    print("Бот запущен! Жду сообщений в Telegram...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

# Импортируем наши модули
from mock_model import MockModel
from knowledge_base import KnowledgeBase

# ВСТАВЬ СЮДА СВОЙ ТОКЕН ОТ BOTFATHER
TOKEN = "ТВОЙ_ТОКЕН_ЗДЕСЬ"

# Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Инициализация нашей модели и базы знаний
model = MockModel()
kb = KnowledgeBase()

# Обработчик команды /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я бот для мониторинга психоэмоционального состояния. Напиши, как ты себя чувствуешь.")

# Обработчик всех текстовых сообщений
@dp.message()
async def handle_message(message: types.Message):
    text = message.text
    
    # 1. Получаем предсказание от модели
    predicted_class, confidence = model.predict(text)
    
    # 2. Получаем рекомендацию из базы знаний
    advice = kb.get_recommendation(text, predicted_class)
    
    # 3. Формируем ответ
    class_names = {0: "Норма", 1: "Стресс", 2: "Тревога"}
    state_name = class_names[predicted_class]
    
    response = f"Я определил твое состояние как: **{state_name}** (уверенность: {int(confidence*100)}%)\n\n"
    response += f"💡 Совет: {advice}"
    
    # Отправляем ответ студенту
    await message.answer(response, parse_mode="Markdown")

# Запуск бота
async def main():
    logging.basicConfig(level=logging.INFO)
    print("Бот запущен! Жду сообщений в Telegram...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")