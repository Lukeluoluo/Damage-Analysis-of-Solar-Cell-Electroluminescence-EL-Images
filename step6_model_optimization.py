# step4_train_weighted.py
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# ====================配置====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = r"D:\项目\太阳能电池的破损分析\split_dataset"
train_dir = os.path.join(data_root, "train")
val_dir = os.path.join(data_root, "val")
batch_size = 16
epochs = 20
lr = 1e-4

# 预处理，和之前保持不变
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print("类别：", train_dataset.classes)

# ==========关键：设置类别权重==========
# defect权重调高，这里给2.0，你后续可以试2.0 /2.5 /3.0
# 索引0:defect, 索引1:normal
class_weights = torch.tensor([2.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 模型
model = torchvision.models.resnet18(pretrained=False)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_acc = 0.0
train_loss_list = []
val_acc_list = []

print(f"训练使用设备: {device}")
print("====开始训练====")
for epoch in range(1, epochs+1):
    # train
    model.train()
    total_train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    # val
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += torch.sum(preds == labels)
    val_acc = correct.item() / total

    train_loss_list.append(avg_train_loss)
    val_acc_list.append(val_acc)
    print(f"Epoch {epoch:2d} | train_loss:{avg_train_loss:.4f} | val_acc:{val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_solar_weighted.pth")
        print(f"  >> 更新最优模型，best_val_acc={best_val_acc:.4f}")

# 绘制训练曲线，修复中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="train_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("训练损失曲线")

plt.subplot(1,2,2)
plt.plot(val_acc_list, label="val_acc", color="orange")
plt.xlabel("Epoch")
plt.ylabel("准确率")
plt.legend()
plt.title("验证集准确率曲线")
plt.tight_layout()
plt.savefig("train_curve_weighted.png", dpi=200)
plt.show()