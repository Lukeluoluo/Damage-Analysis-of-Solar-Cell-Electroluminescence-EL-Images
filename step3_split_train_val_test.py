# step3_split_train_val_test.py
import os
import shutil
import random

# =====================路径配置=====================
dataset_root = r"D:\项目\太阳能电池的破损分析\dataset"
output_root = r"D:\项目\太阳能电池的破损分析\split_dataset"

classes = ["normal", "defect"]
subsets = ["train", "val", "test"]

# 划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 设置随机种子，每次划分结果固定，方便复现实验
random.seed(42)
# =================================================

# 创建输出目录 split_dataset/train、val、test下的normal/defect
for subset in subsets:
    for cls in classes:
        os.makedirs(os.path.join(output_root, subset, cls), exist_ok=True)


for cls in classes:
    src_dir = os.path.join(dataset_root, cls)
    img_list = [f for f in os.listdir(src_dir) if f.lower().endswith(".jpg")]
    random.shuffle(img_list)
    total = len(img_list)

    # 切分索引
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_imgs = img_list[:train_end]
    val_imgs = img_list[train_end:val_end]
    test_imgs = img_list[val_end:]

    # 复制文件
    for name in train_imgs:
        shutil.copy2(os.path.join(src_dir, name), os.path.join(output_root, "train", cls, name))
    for name in val_imgs:
        shutil.copy2(os.path.join(src_dir, name), os.path.join(output_root, "val", cls, name))
    for name in test_imgs:
        shutil.copy2(os.path.join(src_dir, name), os.path.join(output_root, "test", cls, name))

    print(f"【{cls}】总:{total} | train:{len(train_imgs)} | val:{len(val_imgs)} | test:{len(test_imgs)}")


print("\n数据集划分完成！")
print(f"划分后数据集根目录：{output_root}")
print("目录结构：")
print("split_dataset/")
print("├─train/ ─ normal, defect")
print("├─val/   ─ normal, defect")
print("└─test/  ─ normal, defect")