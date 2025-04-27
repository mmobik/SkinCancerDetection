import torch
from tqdm import tqdm

def train_model(model, train_loader, test_loader, loss_fn, optimizer, scheduler=None, epochs=10):
    model.to(device)  # Отправляем модель на устройство
    train_losses, test_losses, accuracies = [], [], []
    best_loss = float('inf')  # Начальная лучшая ошибка
    best_accuracy = 0.0
    patience_counter = 0
    early_stopping_patience = 15  # Количество эпох до ранней остановки
    scaler = torch.cuda.amp.GradScaler()  # Для работы с половинной точностью

    for epoch in range(epochs):
        model.train()  # Переводим модель в режим тренировки
        running_loss, correct, total = 0.0, 0, 0

        train_iter = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)

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

        # Оценка на тестовой выборке
        model.eval()  # Переводим модель в режим оценки
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
        print(f"Epoch {epoch + 1}/{epochs} | "
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
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"\nBest Test Loss: {best_loss:.4f} | Best Test Accuracy: {best_accuracy:.2f}%")
    return train_losses, test_losses, accuracies
