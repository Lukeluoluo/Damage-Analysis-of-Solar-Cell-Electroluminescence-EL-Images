# step5_test.py
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ---------------------- 配置 路径改对！ ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 注意！测试集是split_dataset下面的test
data_dir = r"D:\项目\太阳能电池的破损分析\split_dataset\test"
model_path = "best_solar.pth"
batch_size = 16

# !!!!! 预处理必须和训练完全一模一样！！！
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# 加载测试集
test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
class_names = test_dataset.classes
print("类别：", class_names)

# 加载模型
model = torchvision.models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []
all_probs = []

# 推理
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:,1].cpu().numpy())

# 指标计算
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names)
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

print("===== 测试集评估结果 =====")
print("混淆矩阵：")
print(cm)
print("\n分类报告：")
print(report)
print(f"AUC = {roc_auc:.4f}")

# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
# 混淆矩阵
ax1.matshow(cm, cmap=plt.cm.Blues)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax1.text(j, i, str(cm[i,j]), va="center", ha="center", fontsize=14)
ax1.set_xticks([0,1])
ax1.set_yticks([0,1])
ax1.set_xticklabels(class_names)
ax1.set_yticklabels(class_names)
ax1.set_xlabel("预测标签")
ax1.set_ylabel("真实标签")
ax1.set_title("混淆矩阵")

# ROC曲线
ax2.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}')
ax2.plot([0,1],[0,1],'k--')
ax2.set_xlabel('假阳性率FPR')
ax2.set_ylabel('真阳性率TPR')
ax2.set_title('ROC曲线')
ax2.legend()
plt.tight_layout()
plt.savefig("step5_test_result.png", dpi=200)
plt.show()