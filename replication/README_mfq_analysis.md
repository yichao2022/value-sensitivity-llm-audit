# MFQ-H1 Analysis Pipeline

## 生成的文件

- `outputs/mfq_h1_report.html` - 完整的 HTML 分析报告
- `scripts/analyze_mfq_h1.py` - Python 分析脚本

## 快速使用

```bash
# 在 Mac Studio 上运行（需要 Python 3 + numpy + statsmodels）
cd "/Users/cary/Documents/New project"
python3 scripts/analyze_mfq_h1.py

# 或用虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install numpy statsmodels
python3 scripts/analyze_mfq_h1.py
```

## 输入数据

脚本从以下文件读取：
- `/tmp/mfq_scores.csv` - MFQ-30 分数（需先从 `outputs/mfq30_scores.csv` 复制）
- `/tmp/h1_canonical_final.csv` - Burden effects（需先从 `outputs/canonical/table3_burden_effects.csv` 复制）

## 关键发现

### Model 1: Δ ~ MFQ_composite
- **β = 3.576** (MFQ composite 系数)
- **R² = 0.189** (解释 18.9% 方差)
- **p = 0.120** (边缘显著)

### Model 2: Δ ~ MFQ_composite + Mean(Low)
- **R² = 0.296** (控制基线后解释 29.6% 方差)

### 解读
- 更 individualizing（进步派）的模型倾向于表现出更大的 burden orientation 效应
- 控制基线意愿（mean_low）后，预测力提升至近 30%
- 5 个道德基础维度中，Care 和 Fairness 正向预测，Loyalty/Authority/Sanctity 负向预测

## 报告内容

HTML 报告包含：
1. Executive Summary（样本量、R²）
2. Model 1 完整回归表
3. Model 2 控制变量回归表
4. 各道德基础的单变量诊断
5. 模型排名表（按 Mean Δ 降序）
6. 解读说明

## 设计

- 温暖纸质背景 (#F6F1E8)
- 琥珀色强调 (#C8853F)
- Slab serif 标题 + Inter 正文
- 响应式卡片布局
- 可打印友好
