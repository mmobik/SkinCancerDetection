import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HMNIST_28x28_DS(Dataset):
    # Конструктор класса для загрузки данных
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)  # Загружаем данные из CSV
        self.transform = transform or transforms.Compose([  # Преобразования для изображений
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
        # Словарь для преобразования меток в числовые значения
        self.dx_to_label = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
        self.label_to_dx = {v: k for k, v in self.dx_to_label.items()}  # Обратное преобразование

    # Возвращаем количество данных в датасете
    def __len__(self):
        return len(self.data)

    # Метод для получения изображения и его метки
    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # Получаем строку данных по индексу
        pixels = row[:-1].values.astype(np.uint8)  # Преобразуем пиксели в массив
        label = row.iloc[-1]  # Метка

        if isinstance(label, str):
            label = self.dx_to_label[label]  # Преобразуем строковую метку в число
        else:
            label = int(label)

        # Преобразуем пиксели в 3-канальное изображение
        r = pixels[:784].reshape(28, 28)
        g = pixels[784:1568].reshape(28, 28)
        b = pixels[1568:].reshape(28, 28)
        image = np.stack([r, g, b], axis=-1)

        image = torch.from_numpy(image).float() / 255  # Нормализуем изображение
        image = image.permute(2, 0, 1)  # Меняем размерность для PyTorch

        if self.transform:
            image = self.transform(image)  # Применяем преобразования

        return image, label  # Возвращаем изображение и метку
