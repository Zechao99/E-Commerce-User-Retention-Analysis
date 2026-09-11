# E-commerce User Retention & RFM Analysis

基于 UCI Online Retail 真实交易数据的电商用户复购分析项目。完整走通 **数据清洗 → RFM 用户分群 → 复购行为分析 → 预测建模 → 业务建议** 的数据分析全链路。

## 项目背景

某英国在线零售商 2010.12–2011.12 的真实交易数据（54 万条原始记录）。业务问题：

- 哪些客户是高价值客户？如何差异化运营？
- 用户复购行为有什么规律？多久会复购？
- 能否预测用户未来是否会复购？哪些因素最重要？

## 核心结论

| 指标 | 数值 |
|---|---|
| 清洗后交易记录 | 397,884 条 |
| 独立客户数 | 4,338 |
| 总销售额 | 891 万英镑 |
| Top 20% 客户收入贡献 | **74.6%** |
| Champions（冠军客户）占比 | 22.2% 客户，贡献 65.2% 收入 |
| 90 天内复购率 | 65.6% |
| 180 天内复购率 | 88.3% |
| 首单到复购中位间隔 | 57 天 |
| 复购预测模型 AUC | 0.736（逻辑回归） |

### 关键发现

1. **客户集中度极高**：Champions 群体（22% 的客户）贡献了 65% 的收入，符合典型的帕累托分布。这部分客户是营收的基本盘，必须优先维护。
2. **复购窗口集中在前 90 天**：65.6% 的二次购买发生在首单后 90 天内，中位间隔 57 天。这意味着新用户上线后的前 3 个月是复购转化的黄金窗口。
3. **活跃天数是最强预测因子**：在复购预测模型中，用户活跃天数（active_days）的系数最大且为正——用户在更多不同日期上有购买行为，未来复购概率显著更高。
4. **品类广度促进复购**：购买过不同商品种类越多的用户，复购概率越高。交叉销售和品类推荐有明确的数据支撑。
5. **高客单价反而负向**：平均订单金额越高，复购概率略低——可能反映了高价低频商品的客户粘性较弱。

### 业务建议

| 客群 | 特征 | 建议策略 |
|---|---|---|
| **Champions** | 近期活跃、高频、高额 | VIP 专属服务、新品优先试用、会员积分翻倍 |
| **Loyal Customers** | 较活跃、高频 | 向上销售、提升客单价（推荐关联品类） |
| **At Risk (High Value)** | 沉睡但曾经高价值 | 定向召回优惠券、专属客户经理触达 |
| **Cannot Lose Them** | 长期沉睡、曾高价值 | 紧急挽回：大额折扣 + 人工回访 |
| **New Customers** | 新客、单次购买 | 前 90 天关键培育期：首单后 7/30/60 天自动化邮件触达 |
| **New Promising** | 新客但近期活跃 | 品类交叉推荐、引导第二单 |

## 分析框架

```
原始数据 (541,909 rows)
    │
    ▼
数据清洗（去缺失客户ID / 去取消订单 / 去异常值）
    │
    ▼
RFM 用户分群（Recency / Frequency / Monetary → 8 个客群）
    │
    ▼
复购行为分析（复购率 / 间隔分布 / 国家对比 / 月度趋势）
    │
    ▼
预测建模（逻辑回归 vs 决策树，预测未来 90 天复购）
    │
    ▼
业务建议与可视化报告
```

## 技术栈

- **数据处理**：Python / pandas / NumPy
- **统计建模**：scikit-learn（逻辑回归、决策树）
- **可视化**：Matplotlib / Seaborn
- **数据来源**：UCI Machine Learning Repository – Online Retail Dataset

## 项目结构

```
ecommerce-user-retention-analysis/
├── README.md                    # 本文件
├── requirements.txt             # Python 依赖
├── .gitignore
├── data/
│   ├── raw/                     # 原始数据（需自行下载，见下方说明）
│   └── processed/              # 清洗后数据、RFM 表
├── src/
│   ├── download_data.py         # 数据下载脚本
│   └── analysis.py              # 完整分析流水线
└── reports/
    ├── summary_stats.json       # 关键指标汇总
    └── figures/                 # 分析图表
        ├── 01_rfm_segments.png      # RFM 客群分布与收入贡献
        ├── 02_rfm_scatter.png       # RFM 三维散点图
        ├── 03_gap_distribution.png  # 复购间隔分布
        ├── 04_monthly_trend.png     # 月度收入与订单趋势
        ├── 05_feature_importance.png # 模型特征重要性
        └── 06_confusion_matrix.png  # 模型混淆矩阵
```

## 快速复现

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据
python src/download_data.py

# 3. 运行完整分析
python src/analysis.py
```

运行后所有图表和清洗数据会自动生成到 `reports/figures/` 和 `data/processed/`。

## 数据来源

UCI Machine Learning Repository: [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
> Dr. Daqing Chen, School of Engineering, London South Bank University, London SE1 0AA, UK
