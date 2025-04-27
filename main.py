from dataset import HMNIST_28x28_DS
from model import ImprovedHMNIST_28x28
from train import train_model
from utils import visualize_predictions, evaluate_model, save_model
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

# Инициализация данных и модели
dataset = HMNIST_28x28_DS('data/hmnist_28_28_RGB.csv')
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = ImprovedHMNIST_28x28()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Обучение модели
train_losses, test_losses, accuracies = train_model(
    model, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs=EPOCHS
)

# Визуализация результатов
visualize_predictions(model, test_dataset, class_names, num_samples=10)
evaluate_model(model, test_loader, class_names)

# Сохранение модели
save_model(model)
