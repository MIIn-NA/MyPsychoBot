"""Озвучка по запросу, без сохранения текстов и аудио на диск."""

import asyncio
import logging

from aiogram import F, Router, types

router = Router()
_pending_users = set()


async def synthesize(text: str) -> bytes:
    # Отложенный импорт позволяет пользоваться текстовым ботом без edge-tts.
    from edge_tts import Communicate

    audio = bytearray()
    async with asyncio.timeout(60):
        voice = Communicate(text, "ru-RU-SvetlanaNeural", connect_timeout=5, receive_timeout=15)
        async for chunk in voice.stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
    result = bytes(audio)
    if not result:
        raise ValueError("Empty audio")
    return result


async def offer_speech(message: types.Message, user_id: int):
    # Отдельная кнопка сохраняет основную клавиатуру и кнопки уточнения.
    # Текст берём из reply_to_message: кнопки работают и после перезапуска.
    await message.reply(
        "Можно прослушать ответ. При нажатии текст ответа передаётся Microsoft для озвучки.",
        reply_markup=types.InlineKeyboardMarkup(inline_keyboard=[[
            types.InlineKeyboardButton(text="🔊 Озвучить", callback_data=f"tts:{user_id}")
        ]]),
    )


@router.callback_query(F.data.startswith("tts:"))
async def process_speech(callback: types.CallbackQuery):
    if callback.data != f"tts:{callback.from_user.id}":
        await callback.answer("Эта кнопка предназначена автору запроса.", show_alert=True)
        return
    panel = callback.message
    if not isinstance(panel, types.Message):
        await callback.answer("Ответ больше недоступен.", show_alert=True)
        return
    source = panel.reply_to_message
    if source is None or not source.text:
        await callback.answer("Ответ недоступен. Получи новый ответ бота.", show_alert=True)
        return
    user_id = callback.from_user.id
    if user_id in _pending_users:
        await callback.answer("Озвучка уже готовится. Подожди немного.")
        return

    _pending_users.add(user_id)
    try:
        await callback.answer("Готовлю озвучку…")
        audio = await synthesize(source.text)
        await source.reply_audio(
            types.BufferedInputFile(audio, filename="answer.mp3"),
            title="Озвучка ответа", performer="MyPsychoBot",
        )
    except Exception as error:
        # Не записываем личные тексты или сетевые адреса с текстом в логи.
        logging.warning("Ошибка озвучки (%s).", type(error).__name__)
        await panel.answer(
            "Не удалось озвучить ответ. Проверь подключение к интернету и попробуй позже. "
            "Текст ответа доступен выше."
        )
    finally:
        _pending_users.discard(user_id)
