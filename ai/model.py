import torch
from torch import nn

class ImprovedHMNIST_28x28(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(  # Составляем блоки сверток и подвыборки
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
            nn.AdaptiveAvgPool2d((7, 7))  # Адаптивная подвыборка
        )

        self.classifier = nn.Sequential(  # Полносвязный слой для классификации
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )

        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)  # Пропускаем через блоки свертки
        x = torch.flatten(x, 1)  # Преобразуем в одномерный вектор
        x = self.classifier(x)  # Пропускаем через классификатор
        return x
