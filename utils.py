import torch
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

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
