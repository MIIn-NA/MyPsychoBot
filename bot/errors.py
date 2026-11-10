"""Общие ошибки анализа без зависимостей от модели и Telegram."""


class ClassificationError(RuntimeError):
    """Не удалось классифицировать сообщение."""


class TextTooLongError(ClassificationError):
    """Текст после токенизации превышает доступный лимит."""

    def __init__(self, token_count, max_length):
        self.token_count = token_count
        self.max_length = max_length
        super().__init__(
            f"Сообщение слишком длинное: {token_count} токенов при лимите "
            f"{max_length} (включая специальные токены). "
            "Пожалуйста, сократите сообщение и отправьте его заново."
        )
