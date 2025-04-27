# Класс HMNIST_28x28_PredictionDS
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class HMNIST_28x28_PredictionDS(Dataset):  # Упрощенный Dataset
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        # Преобразования для подготовки изображений к модели
        self.transform = transform or transforms.Compose([
            transforms.Resize((28, 28)),  # Приводим изображение к размеру 28x28
            transforms.ToTensor(),  # Преобразуем в тензор
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Открытие изображения в RGB
        image = self.transform(image)  # Применяем преобразования
        return image, image_path
