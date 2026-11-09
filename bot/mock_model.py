""" ДАША заменит её настоящей моделью классификации, сейчас это заглушка(Пока используется MockModel; позже её нужно заменить на модель Даши.)"""
class MockModel:
    """
    Заглушка для NLP-модели. Работает по ключевым словам.
    """
    def __init__(self):
        self.stress_words = [
            'экзамен', 'дедлайн', 'устал', 'не успеваю', 'сессия', 'проблема',
            'достало', 'просто выгорание',
        ]
        self.anxiety_words = [
            'боюсь', 'тревога', 'страшно', 'паника', 'не могу уснуть', 'переживаю',
            'из-за отношений', 'проблемы со сном', 'будущее (работа, планы)',
        ]
        
    def predict(self, text):
        """
        Возвращает кортеж: (класс, уверенность).
        Классы: 0 - норма, 1 - стресс, 2 - тревога.
        """
        text_lower = text.lower()
        
        for word in self.anxiety_words:
            if word in text_lower:
                return 2, 0.85
                
        for word in self.stress_words:
            if word in text_lower:
                return 1, 0.90
                
        return 0, 0.75

# --- Проверка ---
if __name__ == "__main__":
    model = MockModel()
    test_texts = ["У меня завтра экзамен, я не успеваю", "Я так боюсь провалить сессию", "Сегодня отличная погода"]
    
    for text in test_texts:
        pred_class, confidence = model.predict(text)
        print(f"Текст: '{text}' -> Класс: {pred_class}, Уверенность: {confidence}")
