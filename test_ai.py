import os

import gradio as gr  # Import Gradio
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# === Не менять! (Параметры обучения из оригинального скрипта) ===
BATCH_SIZE = 32  # Обработка картинок
LEARNING_RATE = 0.001
EPOCHS = 50  # Сколько раз пройтись по сету

# Замените сокращения на полные названия (на русском)
CLASS_NAMES_FULL = {
    'akiec': 'Актинический кератоз',  # Actinic Keratosis
    'bcc': 'Базальноклеточный рак',  # Basal Cell Carcinoma
    'bkl': 'Доброкачественный кератоз',  # Benign Keratosis-like Lesion
    'df': 'Дерматофиброма',  # Dermatofibroma
    'mel': 'Меланома',  # Melanoma
    'nv': 'Невус (родинка)',  # Nevus (Mole)
    'vasc': 'Сосудистое образование'  # Vascular Lesion
}

CLASS_NAMES = list(CLASS_NAMES_FULL.keys())  # Оставляем список сокращений для совместимости с моделью
# ===============================================================


class HMNIST_28x28_PredictionDS(Dataset):  # Упрощенный Dataset для предсказаний
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((28, 28)),  # Важно для совместимости с моделью!
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # <- Важно!
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Убедитесь, что изображение в RGB
        image = self.transform(image)
        return image, image_path


class ImprovedHMNIST_28x28(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

        # Инициализация весов (если нужно)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model(model_path, metadata_path=None):  # Функция загрузки модели
    model = ImprovedHMNIST_28x28()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Важно перевести модель в evaluation mode!
    return model


def predict_image(image_path, model):
    dataset = HMNIST_28x28_PredictionDS([image_path])
    image, _ = dataset[0]
    image = image.unsqueeze(0).to(device)  # Добавляем batch dimension
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Вероятности
        confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

    # Отсортировать вероятности и вернуть топ-3
    sorted_confidences = dict(sorted(confidences.items(), key=lambda item: item[1], reverse=True))
    top3_predictions = {CLASS_NAMES_FULL[k]: sorted_confidences[k] for k in list(sorted_confidences)[:3]}  # Заменяем сокращения

    return top3_predictions


def create_gradio_interface(model):
    def process_image(img):  # Функция для Gradio
        if img is None:
            return "No image uploaded"
        temp_image_path = "temp_image.jpg"
        img.save(temp_image_path)  # PIL умеет проще сохранять из numpy array
        predictions = predict_image(temp_image_path, model)
        os.remove(temp_image_path)
        return predictions

    image_input = gr.Image(type="pil", label="Upload Mole Image")
    label = gr.Label(num_top_classes=3, label="Top 3 Predictions")
    interface = gr.Interface(fn=process_image, inputs=image_input, outputs=label,
                             title="Skin Cancer AI Detector",
                             description="Upload an image of a mole to get a risk assessment.",
                             live=False)  # <- Сделаем live=False

    return interface


if __name__ == '__main__':
    # *** ВАЖНО: Укажите правильные пути к вашей обученной модели! ***
    MODEL_PATH = 'saved_model/improved_hmnist.pth'  # <---- !!!
    # Лучше не использовать metadata_path, а просто указать CLASS_NAMES вручную
    # metadata_path = 'saved_model/improved_hmnist_metadata.json'
    # ================================================================
    model = load_model(MODEL_PATH)
    gradio_interface = create_gradio_interface(model)
    gradio_interface.launch(share=False)
