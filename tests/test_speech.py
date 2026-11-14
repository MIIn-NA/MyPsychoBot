"""Проверки озвучки без запросов в Telegram и Microsoft."""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from aiogram import types
from bot import speech


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        speech._pending_users.clear()
        self.source = types.Message(
            message_id=10, date=0, chat=types.Chat(id=1, type="private"),
            text="Тревога. Попробуй дыхательное упражнение.",
        )
        self.panel = types.Message(
            message_id=11, date=0, chat=self.source.chat,
            text="Можно прослушать ответ.", reply_to_message=self.source,
        )
        self.callback = SimpleNamespace(
            data="tts:1", from_user=SimpleNamespace(id=1),
            message=self.panel, answer=AsyncMock(),
        )

    async def test_button_replies_to_exact_response_and_has_owner(self):
        with patch.object(types.Message, "reply", new_callable=AsyncMock) as reply:
            await speech.offer_speech(self.source, 1)
        self.assertIn("Microsoft", reply.call_args.args[0])
        button = reply.call_args.kwargs["reply_markup"].inline_keyboard[0][0]
        self.assertEqual(button.callback_data, "tts:1")

    async def test_sends_audio_for_original_response_without_disk(self):
        with patch.object(speech, "synthesize", new_callable=AsyncMock, return_value=b"mp3") as generate, \
                patch.object(types.Message, "reply_audio", new_callable=AsyncMock) as send:
            await speech.process_speech(self.callback)
        generate.assert_awaited_once_with(self.source.text)
        self.assertEqual(send.call_args.args[0].data, b"mp3")
        self.assertFalse(speech._pending_users)
        self.callback.answer.assert_awaited_once()

    async def test_other_user_and_duplicate_click_do_not_generate(self):
        with patch.object(speech, "synthesize") as generate:
            self.callback.data = "tts:2"
            await speech.process_speech(self.callback)
            self.callback.data = "tts:1"
            speech._pending_users.add(1)
            await speech.process_speech(self.callback)
        generate.assert_not_called()

    async def test_deleted_source_does_not_generate(self):
        self.callback.message = self.panel.model_copy(update={"reply_to_message": None})
        with patch.object(speech, "synthesize") as generate:
            await speech.process_speech(self.callback)
        generate.assert_not_called()
        self.callback.answer.assert_awaited_once()

    async def test_network_failure_is_retryable_and_does_not_log_text(self):
        with patch.object(speech, "synthesize", new_callable=AsyncMock, side_effect=RuntimeError("PRIVATE")), \
                patch.object(types.Message, "answer", new_callable=AsyncMock) as answer, \
                patch.object(speech.logging, "warning") as log:
            await speech.process_speech(self.callback)
        self.assertIn("Не удалось озвучить", answer.call_args.args[0])
        self.assertNotIn("PRIVATE", str(log.call_args))
        self.assertFalse(speech._pending_users)

    async def test_synthesis_uses_russian_and_returns_mp3_bytes(self):
        async def stream():
            yield {"type": "SentenceBoundary", "text": "Привет"}
            yield {"type": "audio", "data": b"mp"}
            yield {"type": "audio", "data": b"3"}

        engine = Mock()
        engine.stream.side_effect = stream
        with patch("edge_tts.Communicate", return_value=engine) as factory:
            self.assertEqual(await speech.synthesize("Привет"), b"mp3")
        self.assertEqual(factory.call_args.args, ("Привет", "ru-RU-SvetlanaNeural"))
        self.assertEqual(factory.call_args.kwargs["connect_timeout"], 5)
