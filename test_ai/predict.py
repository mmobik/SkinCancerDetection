# Загрузка модели

import torch
from dataset import HMNIST_28x28_PredictionDS
from model import ImprovedHMNIST_28x28
from config import device, CLASS_NAMES, CLASS_NAMES_FULL

def load_model(model_path):
    model = ImprovedHMNIST_28x28()
    model.load_state_dict(torch.load(model_path, map_location=device))  # Загружаем веса
    model.to(device)
    model.eval()  # Переводим модель в режим оценки
    return model

def predict_image(image_path, model):
    dataset = HMNIST_28x28_PredictionDS([image_path])
    image, _ = dataset[0]  # Загружаем изображение
    image = image.unsqueeze(0).to(device)  # Добавляем размерность для batch
    with torch.no_grad():  # Отключаем градиенты для экономии памяти
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Вычисляем вероятности
        confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

    # Сортируем вероятности и возвращаем топ-3
    sorted_confidences = dict(sorted(confidences.items(), key=lambda item: item[1], reverse=True))
    top3_predictions = {CLASS_NAMES_FULL[k]: sorted_confidences[k] for k in list(sorted_confidences)[:3]}  # Заменяем сокращения на полные названия
    return top3_predictions
