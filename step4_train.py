# step4_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ----------------路径设置----------------
data_root = r"D:\项目\太阳能电池的破损分析\split_dataset"
save_model_path = r"best_solar.pth"
# ---------------------------------------

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练使用设备: {device}")

# 数据预处理，和后续评估必须保持一致
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),  # 灰度图转为3通道适配ResNet
    transforms.ToTensor(),
])

# 加载数据集
train_set = datasets.ImageFolder(data_root + "/train", transform=transform)
val_set = datasets.ImageFolder(data_root + "/val", transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

# 构建模型 resnet18，二分类输出
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model.to(device)

# --------策略：先冻结主干网络，只训练最后全连接层--------
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# 损失函数、优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# 训练超参
epoch_num = 15
best_val_acc = 0.0  # 记录最优验证精度

train_loss_list = []
val_acc_list = []

print("\n====开始训练====\n")
for epoch in range(epoch_num):
    # 训练模式
    model.train()
    total_train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)

    # 验证模式
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1)
            correct += torch.sum(pred == labels).item()
    val_acc = correct / len(val_set)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1:2d} | train_loss:{avg_train_loss:.4f} | val_acc:{val_acc:.4f}")

    # 如果当前验证精度是历史最高，保存模型权重
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_model_path)
        print(f"  >> 更新最优模型，best_val_acc={best_val_acc:.4f}")

# 绘制训练曲线
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list)
plt.title("训练损失 loss")

plt.subplot(1, 2, 2)
plt.plot(val_acc_list)
plt.title("验证集准确率 acc")

plt.tight_layout()
plt.savefig("step4_train_curve.png", dpi=200)
plt.show()

print(f"\n训练结束！最优验证集精度 {best_val_acc:.4f}，权重保存在 {save_model_path}")