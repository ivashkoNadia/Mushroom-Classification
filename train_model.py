import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# Підготовка даних
training_dataset_path = 'D:\\programs\\dyplomna\\other_set\\data\\train'
test_dataset_path = 'D:\\programs\\dyplomna\\other_set\\data\\validation'

mean = [0.4143, 0.3704, 0.2914]
std = [0.2253, 0.2066, 0.1967]

train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

train_dataset = torchvision.datasets.ImageFolder(root=training_dataset_path, transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Визначення пристрою
def set_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Функція збереження моделі
def save_checkpoints(model, epoch, optimizer, best_acc):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best accuracy': best_acc,
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, 'model_best_checkpoint.pth.tar')

# Функція навчання моделі
def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()
    model.to(device)
    best_acc = 0

    for epoch in range(n_epochs):
        print(f"Epoch number {epoch + 1}")
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * running_correct / total
        print(f"    Training dataset: {running_correct}/{total} ({epoch_acc:.2f}%), Loss: {epoch_loss:.4f}")

        # Тестування моделі
        test_acc = evaluate_model_on_test_set(model, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoints(model, epoch, optimizer, best_acc)

    print("Training finished")
    return model

# Функція оцінки моделі
def evaluate_model_on_test_set(model, test_loader):
    device = set_device()
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"    Testing dataset: {correct}/{total} ({acc:.2f}%)")
    return acc

# Створення моделі та навчання
resnet18_model = models.resnet18(weights=None)
resnet18_model.fc = nn.Linear(resnet18_model.fc.in_features, 7)
device = set_device()
resnet18_model = resnet18_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

train_nn(resnet18_model, train_loader, test_loader, criterion, optimizer, 25)
torch.save(resnet18_model.state_dict(), 'best_model.pth')
