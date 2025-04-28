import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Оголошення класів
classes = ['agaric', 'agaricus', 'boletus', 'earthstar', 'gyromitra', 'russula', 'stinkhorn']

# Трансформації для тестових зображень
mean = [0.4143, 0.3704, 0.2914]
std = [0.2253, 0.2066, 0.1967]

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def plot_confusion_matrix(model, data_loader, classes):
    """
    Створює кастомну confusion matrix з кількістю зображень та відсотками.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    # Збираємо всі передбачення і реальні значення
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Обчислення матриці
    cm = confusion_matrix(all_labels, all_predictions)
    cm_sum = np.sum(cm)
    cm_percentage = cm.astype('float') / cm_sum * 100  # Відсотки

    # Відображення
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.Blues

    im = ax.imshow(cm_percentage, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # Підписи
    tick_marks = np.arange(len(classes))  # Локації для підписів класів
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    # Заповнюємо матрицю значеннями
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(
                j, i,
                f"{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)",
                ha="center", va="center", color=text_color, fontsize=10
            )

    # Підпис підсумкових значень
    total_correct = np.trace(cm)  # Сума елементів на діагоналі
    total_accuracy = total_correct / cm_sum * 100
    plt.title(f"Confusion Matrix (Accuracy: {total_accuracy:.1f}%)", fontsize=16)
    plt.xlabel("Predicted Class", fontsize=14)
    plt.ylabel("True Class", fontsize=14)
    plt.tight_layout()
    plt.show()


# Завантаження тестових даних
mushrooms_dataset_path = 'D:\\programs\\dyplomna\\other_set\\data\\test'
mushrooms_dataset = torchvision.datasets.ImageFolder(root=mushrooms_dataset_path, transform=test_transforms)
mushrooms_loader = torch.utils.data.DataLoader(mushrooms_dataset, batch_size=32, shuffle=False)

# Завантаження попередньо навченої моделі
model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device("cpu")))

# Побудова Confusion Matrix
plot_confusion_matrix(model, mushrooms_loader, classes)
