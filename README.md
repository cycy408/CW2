# INT101 Course work 2 — 加州房价预测

基于 scikit-learn `fetch_california_housing` 数据集的机器学习流水线项目，预测加州地区房屋中位价。

## 项目结构

```
CW2/
├── main.py                 # 主流程入口
├── data_prep.py            # 数据加载（pandas）与预处理
├── feature_selection.py    # 两种特征选择方法（纯 NumPy 实现）
├── models.py               # 随机森林训练与模型融合
├── evaluation.py           # MSE、MAE、R² 评估指标（纯 NumPy 实现）
├── analysis.py             # 异常值检测与分析
├── visualization.py        # 柱状图与地理热力图
├── advanced.py             # 进阶层：多模型训练与加权融合
├── output/
│   ├── mse_comparison.png                          # 所有模型 MSE 对比柱状图
│   ├── feature_selection_mse.png                   # 特征选择方法 MSE 对比
│   ├── advanced_mse_comparison.png                 # 进阶层融合模型对比
│   ├── geospatial_heatmap.png                      # 10×10 地理预测误差热力图（去异常值前）
│   ├── geospatial_heatmap_after_outlier_removal.png # 10×10 热力图（去异常值后）
│   ├── outlier_removal_comparison.png              # 去异常值前后 MSE/MAE 对比
│   └── baseline_results.txt                        # 数值结果导出
└── README.md
```

## 流水线总览

项目按三层结构组织：**基础层 → 进阶层 → 扩展层**。

---

### 第一层：基础层 — 数据预处理与基线模型

**data_prep.py — 数据准备**
1. **加载数据**：通过 `fetch_california_housing()` 获取原始 8 个特征，用 `pandas.DataFrame` 装载后转为 NumPy 数组
2. **特征工程**：构造两个比值特征
   - `RoomsPerHousehold = AveRooms / AveOccup`（户均房间数）
   - `BedroomsPerRoom = AveBedrms / AveRooms`（卧室占比）
   - 最终特征维度：8 → **10 个特征**
3. **划分数据集**：80% 训练 / 20% 测试，`random_state=42`
4. **缺失值填充**：用各列中位数填补 NaN
5. **标准化**：`StandardScaler`，使特征均值为 0、标准差为 1

**基线模型 — 线性回归**：
- 使用全部 10 个特征训练 `LinearRegression`
- 输出截距、各特征系数，以及 MSE、MAE、RMSE

**特征选择**（feature_selection.py）：
- **方法 A — Pearson 相关系数**：手动计算每个特征与目标值的 r 值，取 |r| 最大的 5 个
- **方法 B — F 检验**：由 r² 推导 F 统计量，公式 `F = (r² / (1 − r²)) × (n − 2)`，取 F 值最大的 5 个
- 输出两种方法选出的特征交集与差异

**基线模型 — 随机森林**：
- 超参数：`n_estimators=300`, `max_depth=20`, `n_jobs=-1`
- 分别用以下三组特征训练：
  - 全部 10 个特征
  - Pearson 选出的 5 个特征
  - F 检验选出的 5 个特征
- 将 3 组 RF 结果与线性回归基线对比 MSE
- 输出：`output/mse_comparison.png`、`output/feature_selection_mse.png`、`output/baseline_results.txt`

---

### 第二层：进阶层 — 模型融合

**advanced.py**（由 `main()` 调用，也可独立运行）

- 重新执行特征选择，训练四个模型：
  - 线性回归（全特征）
  - 随机森林（全特征）
  - 随机森林（Pearson 选出的 5 个特征）
  - 随机森林（F 检验选出的 5 个特征）
- **加权融合**（models.py 的 `fuse_predictions`）：对四个模型的预测值取加权平均，权重为 `[0.15, 0.35, 0.25, 0.25]`（LR / RF-all / RF-Pearson / RF-Freg）
- 对比融合模型与单一模型的 MSE、MAE、R²
- 输出：`output/advanced_mse_comparison.png`

---

### 第三层：扩展层 — 地理可视化与异常值处理

**地理空间热力图**（visualization.py）：
- 将测试集按经纬度划分为 10×10 网格
- 每个格子显示该区域的平均绝对预测误差
- 生成两张热力图，**使用统一色阶**（vmin=0, vmax 取自去异常值前），可直接对比：
  - `output/geospatial_heatmap.png` — 全量数据训练模型
  - `output/geospatial_heatmap_after_outlier_removal.png` — 去异常值后重训模型
- 用于发现模型在哪些地理位置表现较差，以及去异常值后的改善效果

**异常值分析**（analysis.py）：
1. **检测**：对目标值使用 IQR 方法（`factor=1.5`），标记异常高房价样本
2. **分析**：对比异常样本与正常样本在各特征上的均值差异
3. **移除与重训练**：剔除异常样本后重新划分数据集，用前两层表现最优的模型重新训练
4. **对比可视化**：生成去异常值前后 MSE/MAE 的分组柱状图 `output/outlier_removal_comparison.png`

---

## 输出文件一览

| 文件 | 说明 |
|------|------|
| `output/mse_comparison.png` | 四个基线模型的 MSE 柱状图对比 |
| `output/feature_selection_mse.png` | 两种特征选择方法（Pearson vs F 检验）的 MSE 对比 |
| `output/advanced_mse_comparison.png` | 单一模型与四模型加权融合的 MSE 对比 |
| `output/geospatial_heatmap.png` | 10×10 地理网格预测误差热力图（去异常值前） |
| `output/geospatial_heatmap_after_outlier_removal.png` | 10×10 热力图（去异常值后，统一色阶） |
| `output/outlier_removal_comparison.png` | 去异常值前后 MSE/MAE 分组柱状图 |
| `output/baseline_results.txt` | 各模型数值结果及线性回归系数 |

## 运行方式

```bash
python main.py
```

所有预处理、训练、评估、可视化步骤按顺序执行，生成的输出文件保存在 `output/` 目录下。

## 随机种子

`main.py` 顶部可通过 `RANDOM_SEED` 变量控制随机种子：`42` 为固定种子保证结果可复现，`None` 为每次随机。
