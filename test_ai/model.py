# Класс ImprovedHMNIST_28x28

import torch.nn as nn

class ImprovedHMNIST_28x28(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Свёртки и пулы для выделения признаков
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),  # Нормализация
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Пуллинг для уменьшения размерности

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))  # Адаптивное усреднение для фиксированного размера
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout для регуляризации
            nn.Linear(512, 7)  # Выходной слой с 7 классами
        )

        self._initialize_weights()  # Инициализация весов

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Инициализация для свёрток
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # Для нормализации
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)  # Проход через свёрточные слои
        x = torch.flatten(x, 1)  # Выпрямляем тензор
        x = self.classifier(x)  # Проход через классификатор
        return x
