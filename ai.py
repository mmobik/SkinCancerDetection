import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import random
from tqdm import tqdm
from torchvision import transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

BATCH_SIZE = 32 # Обработка картинок 
LEARNING_RATE = 0.001  
EPOCHS = 50  # Сколько раз пройтись по сету

class HMNIST_28x28_DS(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
        self.dx_to_label = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
        self.label_to_dx = {v: k for k, v in self.dx_to_label.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = row[:-1].values.astype(np.uint8)
        label = row.iloc[-1]

        if isinstance(label, str):
            label = self.dx_to_label[label]
        else:
            label = int(label)

        r = pixels[:784].reshape(28, 28)
        g = pixels[784:1568].reshape(28, 28)
        b = pixels[1568:].reshape(28, 28)
        image = np.stack([r, g, b], axis=-1)

        image = torch.from_numpy(image).float() / 255
        image = image.permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, label

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
        
        # Инициализация весов
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

def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=10):
    model.to(device)
    train_losses, test_losses, accuracies = [], [], []
    best_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0
    early_stopping_patience = 15
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_iter.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = 100 * correct / total

        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {accuracy:.2f}% | "
              f"LR: {current_lr:.2e}")

        if test_loss < best_loss:
            best_loss = test_loss
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nBest Test Loss: {best_loss:.4f} | Best Test Accuracy: {best_accuracy:.2f}%")
    return train_losses, test_losses, accuracies


def visualize_predictions(model, dataset, class_names, num_samples=10):
    model.eval()
    plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, label = dataset[idx]
        input_image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_image)
            _, predicted = torch.max(output, 1)

        true_label = class_names[int(label)]
        predicted_label = class_names[int(predicted.item())]
        image = image.permute(1, 2, 0).cpu().numpy()

        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10,
                  color='green' if true_label == predicted_label else 'red')

    plt.tight_layout()
    plt.show()

    
def evaluate_model(model, dataloader, class_names):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

import json
import os

def save_model(model, save_dir='saved_model', model_name='improved_hmnist'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Сохраняем веса модели
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save(model.state_dict(), model_path)
    
    # Сохраняем метаданные
    metadata = {
        "model_name": model_name,
        "train_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "best_accuracy": max(accuracies)
    }
    
    metadata_path = os.path.join(save_dir, f'{model_name}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✅ Model and metadata saved to '{save_dir}'")


if __name__ == '__main__':
    dataset = HMNIST_28x28_DS('data/hmnist_28_28_RGB.csv')
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ImprovedHMNIST_28x28()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20,eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, test_losses, accuracies = train_model(
        model, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs=EPOCHS
    )

    # Визуализация результатов
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    visualize_predictions(model, test_dataset, class_names, num_samples=10)
    evaluate_model(model, test_loader, class_names)

    save_model(model)
