# step9_analyze_csv.py
import pandas as pd
import numpy as np

csv_path = "all_images_predict_result.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")

print("======= 全数据集(全部图片)总体统计 =======")
total = len(df)
# true_bin_label:0缺陷，1正常；pred_bin_label:0缺陷，1正常

# 正确预测
df["correct"] = df["true_bin_label"] == df["pred_bin_label"]
acc = df["correct"].mean()

# 混淆统计
# TP:真实缺陷(0)，预测缺陷(0)
tp = ((df["true_bin_label"] == 0) & (df["pred_bin_label"] == 0)).sum()
# FN:真实缺陷(0)，预测正常(1) 漏检
fn = ((df["true_bin_label"] == 0) & (df["pred_bin_label"] == 1)).sum()
# FP:真实正常(1)，预测缺陷(0) 误报
fp = ((df["true_bin_label"] == 1) & (df["pred_bin_label"] == 0)).sum()
# TN:真实正常(1)，预测正常(1)
tn = ((df["true_bin_label"] == 1) & (df["pred_bin_label"] == 1)).sum()

recall_defect = tp/(tp+fn)   #缺陷召回率
prec_defect = tp/(tp+fp)     #缺陷精确率

print(f"总样本数量：{total}")
print(f"总体准确率 Acc: {acc:.4f}")
print(f"缺陷召回率 Recall(缺陷检出): {recall_defect:.4f}")
print(f"缺陷精确率 Precision: {prec_defect:.4f}")
print(f"TP(缺陷正确检出) = {tp}")
print(f"FN(缺陷漏检)     = {fn}")
print(f"FP(正常误报)     = {fp}")
print(f"TN(正常正确识别) = {tn}")

print("\n======= 按原始缺陷类别分组统计召回率 =======")
# 每一种原始异常类别，统计：样本数、该类里面被成功识别为缺陷的数量、召回率
group = df[df["true_bin_label"]==0].groupby("true_anomaly_class")
stat_list = []
for cls_name, sub in group:
    total_cls = len(sub)
    hit = (sub["pred_bin_label"] == 0).sum()
    recall_cls = hit / total_cls
    stat_list.append([cls_name, total_cls, hit, recall_cls])

stat_df = pd.DataFrame(stat_list, columns=["缺陷类别","样本总数","正确检出数","召回率"])
stat_df = stat_df.sort_values("召回率")
print(stat_df.to_string(index=False))

print("\n======= 识别最差的3类缺陷（召回最低） =======")
print(stat_df.head(3).to_string(index=False))

# 保存统计结果到csv，放进报告素材
stat_df.to_csv("per_class_recall.csv", encoding="utf-8-sig", index=False)
print("\n每一类缺陷统计结果已保存至 per_class_recall.csv")