# Cassava Leaf Disease Detection Project

# ===================== PART 1: TRAINING =====================
"""
Requirements:
pip install torch torchvision scikit-learn matplotlib pillow numpy
Dataset structure:
dataset/
  class_1/
  class_2/
  ...
"""

import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler

# Configuration
DATA_DIR = 'dataset'
MODEL_SAVE_PATH = 'best_resnet50_cassava.pth'
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

full_ds = datasets.ImageFolder(DATA_DIR)
idx = np.arange(len(full_ds))
train_idx, temp_idx = train_test_split(idx, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.33, random_state=42)

train_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=train_tf), train_idx)
val_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=val_tf), val_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

classes = full_ds.classes
num_classes = len(classes)

model = models.resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = GradScaler()

best_val = 0.0
for epoch in range(EPOCHS):
    model.train()
    correct = total = 0
    loss_sum = 0
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            out = model(x)
            loss = criterion(out,y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item()*x.size(0)
        pred = out.argmax(1)
        correct += (pred==y).sum().item(); total += y.size(0)
    train_acc = correct/total

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total += y.size(0)
    val_acc = correct/total
    print(f'Epoch {epoch+1}/{EPOCHS} | Train {train_acc:.4f} | Val {val_acc:.4f}')
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print('Training complete. Best model saved.')
