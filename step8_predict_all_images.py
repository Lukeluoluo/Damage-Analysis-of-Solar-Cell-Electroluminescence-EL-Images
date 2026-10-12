# step8_predict_all_images.py
import os
import json
import csv
import torch
import torchvision
from torchvision import transforms
from PIL import Image

# =====================路径配置=====================
json_path = r"D:\项目\太阳能电池的破损分析\InfraredSolarModules-master\2020-02-14_InfraredSolarModules\InfraredSolarModules\module_metadata.json"
img_root = r"D:\项目\太阳能电池的破损分析\InfraredSolarModules-master\2020-02-14_InfraredSolarModules\InfraredSolarModules\images"
model_weight = r"best_solar_weighted.pth"
out_csv_name = "all_images_predict_result.csv"
# =================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"推理设备：{device}")

# 预处理必须和训练完全一致
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# 构建模型，加载权重
model = torchvision.models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 2)
model.load_state_dict(torch.load(model_weight, map_location=device))
model.to(device)
model.eval()

# 读取原始元数据
with open(json_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

# 打开csv写入
csv_file = open(out_csv_name, "w", newline="", encoding="utf-8-sig")
writer = csv.writer(csv_file)
# 表头说明
writer.writerow([
    "image_id",
    "true_anomaly_class",
    "true_bin_label",      # 0=defect缺陷  1=normal正常
    "pred_bin_label",       # 模型预测 0=defect 1=normal
    "prob_defect",          # 是缺陷的置信概率0~1
    "prob_normal"           # 是正常的置信概率0~1
])

print("开始对全部图片推理，总数量：", len(meta))
count = 0

with torch.no_grad():
    for img_id, info in meta.items():
        true_cls = info["anomaly_class"]
        # 真实二分类标签
        if true_cls == "No-Anomaly":
            true_bin = 1
        else:
            true_bin = 0

        img_path = os.path.join(img_root, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            print(f"警告，图片不存在 {img_id}")
            continue

        # 加载单张图片推理
        pil_img = Image.open(img_path).convert("L")
        tensor_img = transform(pil_img).unsqueeze(0).to(device)
        output = model(tensor_img)
        prob = torch.softmax(output, dim=1)[0]
        prob_defect = float(prob[0].cpu())
        prob_normal = float(prob[1].cpu())
        pred_bin = int(torch.argmax(prob, dim=0).cpu())

        writer.writerow([
            img_id,
            true_cls,
            true_bin,
            pred_bin,
            round(prob_defect, 4),
            round(prob_normal, 4)
        ])

        count += 1
        if count % 2000 == 0:
            print(f"已处理 {count} / {len(meta)}")

csv_file.close()
print(f"\n全部推理完成！结果保存到：{out_csv_name}")
print("字段说明：")
print("true_bin_label：0=真实缺陷，1=真实正常")
print("pred_bin_label：0=预测缺陷，1=预测正常")
print("prob_defect：模型判定该图片为缺陷的置信度（0‑1）")