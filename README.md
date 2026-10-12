# InfraredSolarModules 光伏红外缺陷检测
> 项目用途：**学习向简单Demo**，基于ResNet18迁移学习实现太阳能电池红外图像二分类（正常 / 缺陷）。
> 使用公开数据集 `InfraredSolarModules`，完成数据集探查、数据整理、数据集划分、模型训练、评估、全数据集批量推理与统计分析。
> 本项目仅用于深度学习迁移学习练习，不用于工业直接部署。

## 项目目录结构
```
.
├── step1_dataset_stat.py          # 步骤1：数据集统计探查
├── step2_build_binary_dataset.py  # 步骤2：构建二分类数据集文件夹
├── step3_split_train_val_test.py  # 步骤3：分层划分训练/验证/测试集
├── step4_train.py                # 步骤4：基础版本模型训练
├── step6_model_optimization.py    # 步骤6：优化训练（加权损失，提升缺陷召回）
├── step7_test2.py                 # 步骤7：测试集模型评估（混淆矩阵、ROC‑AUC）
├── step8_predict_all_images.py   # 步骤8：全部原始图片批量推理输出CSV
├── step9_analyze_csv.py          # 步骤9：对推理结果做统计分析
├── best_solar.pth                 # 原始训练权重
├── best_solar_weighted.pth        # 加权损失优化后最优权重
├── all_images_predict_result.csv # 全部图片预测结果
├── per_class_recall.csv           # 各类缺陷召回统计
└── *.png                          # 生成的各类图表
```

## 环境依赖
Python >=3.10
```bash
pip install torch torchvision matplotlib seaborn scikit‑learn pandas
```

## 数据集准备
数据集项目：InfraredSolarModules
1. 下载数据集，解压到本地；
2. 修改脚本内路径，指向你的 `module_metadata.json` 和 `images` 文件夹。
- `module_metadata.json`：元数据，记录每张图片真实故障类别
- `images`：全部红外光伏组件图片

> 任务定义：二分类
> - normal：`No‑Anomaly` 无异常正常电池
> - defect：其余11种故障全部归为缺陷样本

## 运行步骤（按顺序执行）
### Step1：数据集统计探查 `step1_dataset_stat.py`
**做什么**
读取数据集元信息，统计全部图片数量、每一类故障样本数量，打印类别分布，绘制样本分布柱状图，同时校验图片文件是否缺失。

**怎么做**
```bash
python step1_dataset_stat.py
```

**输出结果**
1. 控制台打印：总图片数，每一类缺陷样本数量，正常/缺陷样本比例，缺失图片数量；
2. 生成图片 `step1_class_dist.png`：数据集类别分布柱状图；
> 现象：数据集共20000张，正常样本一万张，缺陷样本一万张，包含11种不同故障类型。

---

### Step2：构建二分类数据集文件夹 `step2_build_binary_dataset.py`
**做什么**
根据json元数据，将原图复制到新文件夹，分为`normal`和`defect`两个文件夹，完成二分类数据整理，不修改原始数据。

**怎么做**
```bash
python step2_build_binary_dataset.py
```

**输出结果**
```
dataset/
 ├─ normal/
 └─ defect/
```
控制台打印复制成功的图片数量。

---

### Step3：分层划分训练集、验证集、测试集 `step3_split_train_val_test.py`
**做什么**
分层抽样，保证训练、验证、测试集中正常/缺陷样本比例和原始数据集保持一致。
- train：70% 训练模型
- val：15% 训练过程中选择最优模型
- test：15% 完全独立，**只用于最终评估，训练阶段不使用**，模拟未知真实数据。

**怎么做**
```bash
python step3_split_train_val_test.py
```

**输出结果**
```
split_dataset/
 ├─ train/
 ├─ val/
 └─ test/
```
控制台打印每一类在三个子集的样本数量。

---

### Step4：基础版本模型训练 `step4_train.py`
**做什么**
使用ResNet18迁移学习，冻结主干网络，仅训练最后的全连接层；训练结束保存验证集精度最高模型 `best_solar.pth`；绘制训练loss、验证acc曲线。

**怎么做**
```bash
python step4_train.py
```

**输出结果**
1. 模型权重：`best_solar.pth`
2. 曲线图：`step4_train_curve.png`
> 基础版本效果：测试集准确率77%，缺陷召回0.68，缺陷漏检率较高。

---

### Step6：模型优化训练（加权损失）`step6_model_optimization.py`
**做什么**
针对光伏检测场景漏检风险高的问题，使用**加权交叉熵损失函数**，增大缺陷样本损失权重，惩罚缺陷漏判；同时加入简单图像增强；保存优化后的最优模型 `best_solar_weighted.pth`。

**怎么做**
```bash
python step6_model_optimization.py
```

**输出结果**
1. 模型权重：`best_solar_weighted.pth`
2. 训练曲线图片：`train_curve_weighted.png`
> 优化后验证集最高准确率92.77%。

---

### Step7：独立测试集评估 `step7_test2.py`
**做什么**
使用从未参与训练的test测试集评估优化后模型；输出混淆矩阵、分类报告、ROC曲线、AUC指标。

**怎么做**
修改脚本中`model_path = "best_solar_weighted.pth"`
```bash
python step7_test2.py
```

**输出结果**
1. 控制台：混淆矩阵、分类报告、AUC；
2. 图片 `step5_test_result.png`：混淆矩阵+ROC曲线图。

> 优化后测试集关键指标：
> - 总体准确率：92%
> - 缺陷召回率：0.91，漏检率下降至9.47%
> - AUC = 0.9728

> ROC曲线说明：横轴为正常样本误报率FPR，纵轴缺陷检出率TPR；曲线越靠近左上角代表模型区分能力越强；AUC为曲线下面积，0~1，越接近1效果越好。

---

### Step8：全部原始数据集批量推理 `step8_predict_all_images.py`
**做什么**
遍历全部20000张原始图片，加载训练好的模型推理，输出csv表格，记录图片ID、真实故障类别、真实标签、模型预测标签、缺陷置信概率。

**怎么做**
```bash
python step8_predict_all_images.py
```

**输出结果**
`all_images_predict_result.csv`
|字段|说明|
|----|----|
|image_id|图片ID|
|true_anomaly_class|原始故障类别|
|true_bin_label|真实标签：0缺陷，1正常|
|pred_bin_label|预测标签：0缺陷，1正常|
|prob_defect|判定为缺陷置信度0‑1|
|prob_normal|判定为正常置信度0‑1|

---

### Step9：推理结果统计分析 `step9_analyze_csv.py`
**做什么**
读取批量推理得到的CSV，计算全数据集总体准确率、漏检、误报；**按原始11种缺陷分组统计召回率**，找出最难识别的故障类型，为可行性分析提供数据支撑。

**怎么做**
```bash
python step9_analyze_csv.py
```

**输出结果**
1. 控制台打印总体指标、每一类缺陷召回率、识别最差的三类缺陷；
2. 文件 `per_class_recall.csv`：各类缺陷召回统计表。

## 项目可行性简要总结（可写进报告）
本项目基于ResNet18迁移学习完成太阳能电池红外图像二分类任务。经过加权损失优化，测试集总体准确率92%，缺陷召回率0.91，AUC达到0.9728。
模型可以用于海量光伏巡检图片的初步筛查，能够筛出大部分破损组件，交由人工复核；但部分微弱故障召回率偏低，存在少量漏检、误报，**不能直接完全替代人工检测，不建议直接用于工业全自动报警**。

## 改进方向
1. 解冻ResNet主干网络做完整微调；
2. 使用更强的数据增强；
3. 调整分类置信阈值，进一步权衡漏检与误报；
4. 升级为多分类任务，识别具体故障类型；
5. 导出ONNX用于部署。

## 注意事项
1. Windows环境，DataLoader的`num_workers`设置为0，否则报错；
2. CPU训练速度较慢，可以使用NVIDIA GPU加速；
3. 本项目仅用于学习练习，请勿直接用于生产环境。

