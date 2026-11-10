from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class RealModel:
    def __init__(self):
        # Прямой абсолютный путь к папке с моделью
        model_path = "/home/ubuntu/psycho_bot/model"
        
        print(f"Загрузка нейросети RuBERT из {model_path}...")
        
        # Загружаем токенизатор и саму модель
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        print("Модель успешно загружена!")

    def predict(self, text):
        # Подготавливаем текст (токенизируем)
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # Отключаем вычисление градиентов (для ускорения работы)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Преобразуем выходные данные в вероятности (от 0 до 1)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Находим класс с наибольшей вероятностью
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence