"""НАСТЯ Telegram-бот: команды, кнопки, получение сообщений и отправка ответов"""

import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

# Импортируем наши модули
from mock_model import MockModel
from knowledge_base import KnowledgeBase
from solver import Solver
import os
from dotenv import load_dotenv

load_dotenv() 

TOKEN = os.getenv("BOT_TOKEN")

# Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Инициализация решателя с моделью-заглушкой и базой знаний
solver = Solver(model=MockModel(), knowledge_base=KnowledgeBase())

# Обработчик команды /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я бот для мониторинга психоэмоционального состояния. Напиши, как ты себя чувствуешь.")

# Обработчик всех текстовых сообщений
@dp.message()
async def handle_message(message: types.Message):
    result = solver.solve(message.text)
    await message.answer(result.to_message(), parse_mode="Markdown")

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
