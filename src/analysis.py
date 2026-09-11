"""
E-commerce User Retention & RFM Analysis
=========================================
Based on UCI Online Retail Dataset.
Steps:
  1. Data loading & cleaning
  2. RFM segmentation
  3. Repurchase behavior analysis
  4. Prediction model (repurchase likelihood)
  5. Visualization & report generation
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 120
PALETTE = sns.color_palette("viridis", 8)

# ============================================================
# 1. LOAD & CLEAN
# ============================================================
print("=" * 60)
print("STEP 1: Data Loading & Cleaning")
print("=" * 60)

df = pd.read_excel(os.path.join(RAW_DIR, "online_retail.xlsx"))
print(f"Raw data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Rename columns to snake_case for convenience
df.columns = ["invoice_no", "stock_code", "description", "quantity",
              "invoice_date", "unit_price", "customer_id", "country"]

# Convert types
df["invoice_date"] = pd.to_datetime(df["invoice_date"])
df["customer_id"] = df["customer_id"].astype(str).replace("nan", np.nan)

# --- Cleaning ---
# 1) Remove rows without CustomerID
before = len(df)
df = df.dropna(subset=["customer_id"])
print(f"\nDropped rows without CustomerID: {before} -> {len(df)} (removed {before-len(df)})")

# 2) Remove canceled invoices (InvoiceNo starts with 'C')
before = len(df)
df = df[~df["invoice_no"].astype(str).str.startswith("C")]
print(f"Removed canceled invoices: {before} -> {len(df)} (removed {before-len(df)})")

# 3) Keep only positive quantity & price
before = len(df)
df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
print(f"Removed non-positive quantity/price: {before} -> {len(df)} (removed {before-len(df)})")

# 4) Derived: total line amount
df["amount"] = df["quantity"] * df["unit_price"]

# 5) Date features
df["year"] = df["invoice_date"].dt.year
df["month"] = df["invoice_date"].dt.month
df["ym"] = df["invoice_date"].dt.to_period("M").astype(str)

print(f"\nFinal clean data shape: {df.shape}")
print(f"Date range: {df['invoice_date'].min()} ~ {df['invoice_date'].max()}")
print(f"Unique customers: {df['customer_id'].nunique()}")
print(f"Unique invoices: {df['invoice_no'].nunique()}")
print(f"Total revenue: {df['amount'].sum():,.2f} GBP")

# Save clean data
clean_path = os.path.join(PROC_DIR, "clean_data.csv")
df.to_csv(clean_path, index=False)
print(f"\nClean data saved to {clean_path}")

# ============================================================
# 2. RFM SEGMENTATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: RFM Segmentation")
print("=" * 60)

# Snapshot date = day after last purchase
snapshot_date = df["invoice_date"].max() + pd.Timedelta(days=1)

# Aggregate per customer
rfm = df.groupby("customer_id").agg(
    recency=("invoice_date", lambda x: (snapshot_date - x.max()).days),
    frequency=("invoice_no", "nunique"),
    monetary=("amount", "sum")
).reset_index()

print(f"RFM table shape: {rfm.shape}")
print(f"Recency stats:\n{rfm['recency'].describe()}")
print(f"\nFrequency stats:\n{rfm['frequency'].describe()}")
print(f"\nMonetary stats:\n{rfm['monetary'].describe()}")

# RFM scoring (1-5, higher = better)
# Recency: lower is better -> reverse scoring
rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
# Frequency & Monetary: higher is better
rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

# RFM combined score
rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
rfm["RFM_segment"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)

# Segment naming (standard RFM quadrants)
def assign_segment(row):
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3 and m >= 4:
        return "Loyal Customers"
    elif r >= 4 and f <= 2 and m >= 3:
        return "New Promising"
    elif r >= 3 and f <= 2 and m <= 2:
        return "New Customers"
    elif r <= 2 and f >= 3 and m >= 4:
        return "At Risk (High Value)"
    elif r <= 2 and f >= 3 and m <= 3:
        return "At Risk (Mid Value)"
    elif r <= 2 and f <= 2 and m >= 3:
        return "Cannot Lose Them"
    else:
        return "Others"

rfm["segment_name"] = rfm.apply(assign_segment, axis=1)

seg_summary = rfm.groupby("segment_name").agg(
    customers=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean")
).sort_values("avg_monetary", ascending=False)
seg_summary["pct_customers"] = seg_summary["customers"] / seg_summary["customers"].sum() * 100
seg_summary["pct_revenue"] = (seg_summary["avg_monetary"] * seg_summary["customers"]) / \
                              (seg_summary["avg_monetary"] * seg_summary["customers"]).sum() * 100

print("\nSegment summary:")
print(seg_summary.round(2))

# Save RFM table
rfm.to_csv(os.path.join(PROC_DIR, "rfm_table.csv"), index=False)
seg_summary.to_csv(os.path.join(PROC_DIR, "segment_summary.csv"))

# ---- Figure 1: RFM Segment Bar Chart ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

seg_summary_sorted = seg_summary.sort_values("customers", ascending=True)
axes[0].barh(seg_summary_sorted.index, seg_summary_sorted["customers"], color=PALETTE[2])
axes[0].set_xlabel("Number of Customers")
axes[0].set_title("Customer Distribution by RFM Segment")
for i, (v, p) in enumerate(zip(seg_summary_sorted["customers"], seg_summary_sorted["pct_customers"])):
    axes[0].text(v + 5, i, f"{v} ({p:.1f}%)", va="center", fontsize=9)

# Revenue share
seg_rev = seg_summary.sort_values("pct_revenue", ascending=True)
axes[1].barh(seg_rev.index, seg_rev["pct_revenue"], color=PALETTE[5])
axes[1].set_xlabel("Revenue Share (%)")
axes[1].set_title("Revenue Contribution by Segment")
for i, v in enumerate(seg_rev["pct_revenue"]):
    axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_rfm_segments.png"), bbox_inches="tight")
plt.close()
print("Saved: 01_rfm_segments.png")

# ---- Figure 2: RFM 3D scatter (R vs F vs M) ----
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(rfm["recency"], rfm["frequency"],
                     c=rfm["monetary"], s=20, alpha=0.6, cmap="viridis")
ax.set_xlabel("Recency (days)")
ax.set_ylabel("Frequency (orders)")
ax.set_title("RFM: Recency vs Frequency (color = Monetary)")
plt.colorbar(scatter, ax=ax, label="Monetary (GBP)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_rfm_scatter.png"), bbox_inches="tight")
plt.close()
print("Saved: 02_rfm_scatter.png")

# ============================================================
# 3. REPURCHASE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Repurchase Behavior Analysis")
print("=" * 60)

# Get first & last purchase date per customer
customer_dates = df.groupby("customer_id")["invoice_date"].agg(["min", "max", "count"])
customer_dates.columns = ["first_purchase", "last_purchase", "total_orders"]
customer_dates["lifetime_days"] = (customer_dates["last_purchase"] - customer_dates["first_purchase"]).dt.days

# Overall repurchase rate (>=2 orders = repeat)
total_customers = len(customer_dates)
repeat_customers = (customer_dates["total_orders"] >= 2).sum()
repeat_rate = repeat_customers / total_customers * 100
print(f"Total customers: {total_customers}")
print(f"Repeat customers (>=2 orders): {repeat_customers} ({repeat_rate:.1f}%)")

# Repurchase within 90 days
# For each customer, compute time between 1st and 2nd order
customer_orders = df.groupby("customer_id")["invoice_date"].apply(
    lambda x: sorted(x.dt.date.unique())
)

gap_days = []
for cid, dates in customer_orders.items():
    if len(dates) >= 2:
        gap = (pd.Timestamp(dates[1]) - pd.Timestamp(dates[0])).days
        gap_days.append(gap)

gap_days = np.array(gap_days)
print(f"\nCustomers with >=2 orders: {len(gap_days)}")
print(f"Avg days between 1st & 2nd order: {gap_days.mean():.1f} days")
print(f"Median days between 1st & 2nd order: {np.median(gap_days):.1f} days")

# 90-day repurchase rate
repurchase_90 = (gap_days <= 90).sum() / len(gap_days) * 100
repurchase_180 = (gap_days <= 180).sum() / len(gap_days) * 100
print(f"Repurchase within 90 days: {repurchase_90:.1f}%")
print(f"Repurchase within 180 days: {repurchase_180:.1f}%")

# Repurchase by country
country_stats = df.groupby("country")["customer_id"].nunique().reset_index()
country_stats.columns = ["country", "total_customers"]
country_repeat = df.groupby(["country", "customer_id"])["invoice_no"].nunique().reset_index()
country_repeat["is_repeat"] = country_repeat["invoice_no"] >= 2
country_rep = country_repeat.groupby("country")["is_repeat"].mean().reset_index()
country_rep.columns = ["country", "repurchase_rate"]
country_rep = country_rep.merge(country_stats, on="country")
country_rep = country_rep[country_rep["total_customers"] >= 50].sort_values("repurchase_rate", ascending=False)

print("\nTop countries by repurchase rate (>=50 customers):")
print(country_rep.head(10).round(3))

# ---- Figure 3: Gap days distribution ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(gap_days, bins=50, color=PALETTE[3], edgecolor="white", alpha=0.8)
ax.axvline(90, color="red", linestyle="--", label="90 days")
ax.axvline(180, color="orange", linestyle="--", label="180 days")
ax.set_xlabel("Days between 1st and 2nd order")
ax.set_ylabel("Number of Customers")
ax.set_title(f"Distribution of Time to First Repurchase (median={np.median(gap_days):.0f} days)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_gap_distribution.png"), bbox_inches="tight")
plt.close()
print("Saved: 03_gap_distribution.png")

# ---- Figure 4: Monthly revenue trend ----
monthly_rev = df.groupby("ym")["amount"].sum().reset_index()
monthly_orders = df.groupby("ym")["invoice_no"].nunique().reset_index()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(monthly_rev["ym"], monthly_rev["amount"] / 1000, "o-", color=PALETTE[0], label="Revenue (k GBP)")
ax1.set_xlabel("Month")
ax1.set_ylabel("Revenue (k GBP)", color=PALETTE[0])
ax1.tick_params(axis="x", rotation=45)

ax2 = ax1.twinx()
ax2.plot(monthly_orders["ym"], monthly_orders["invoice_no"], "s--", color=PALETTE[5], label="Orders")
ax2.set_ylabel("Number of Orders", color=PALETTE[5])

plt.title("Monthly Revenue & Order Trend")
fig.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_monthly_trend.png"), bbox_inches="tight")
plt.close()
print("Saved: 04_monthly_trend.png")

# ============================================================
# 4. PREDICTION MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Repurchase Prediction Model")
print("=" * 60)

# Build customer-level features
customer_features = df.groupby("customer_id").agg(
    total_orders=("invoice_no", "nunique"),
    total_amount=("amount", "sum"),
    avg_order_value=("amount", lambda x: x.sum() / df.loc[x.index, "invoice_no"].nunique()),
    avg_quantity=("quantity", "mean"),
    distinct_products=("stock_code", "nunique"),
    active_days=("invoice_date", lambda x: x.dt.date.nunique()),
    first_purchase=("invoice_date", "min"),
    last_purchase=("invoice_date", "max"),
).reset_index()

# More accurate avg_order_value
order_totals = df.groupby(["customer_id", "invoice_no"])["amount"].sum().reset_index()
aov = order_totals.groupby("customer_id")["amount"].mean().reset_index()
aov.columns = ["customer_id", "avg_order_value"]
customer_features = customer_features.drop(columns=["avg_order_value"]).merge(aov, on="customer_id")

# Recency
customer_features["recency_days"] = (snapshot_date - customer_features["last_purchase"]).dt.days
customer_features["customer_tenure"] = (customer_features["last_purchase"] - customer_features["first_purchase"]).dt.days

# Target: will customer repurchase within next 90 days?
# Use a split date: first 6 months as history, last 3 months as prediction window
split_date = df["invoice_date"].min() + pd.DateOffset(months=9)
print(f"Split date: {split_date}")

# Historical data (up to split)
hist = df[df["invoice_date"] <= split_date]
# Future data (after split)
future = df[df["invoice_date"] > split_date]

# Customers who had a future purchase = positive class
future_customers = set(future["customer_id"].unique())
print(f"Customers in history: {hist['customer_id'].nunique()}")
print(f"Customers who repurchased after split: {len(future_customers)}")

# Build features from history
hist_features = hist.groupby("customer_id").agg(
    total_orders=("invoice_no", "nunique"),
    total_amount=("amount", "sum"),
    active_days=("invoice_date", lambda x: x.dt.date.nunique()),
    distinct_products=("stock_code", "nunique"),
    first_purchase=("invoice_date", "min"),
    last_purchase=("invoice_date", "max"),
).reset_index()

hist_order_totals = hist.groupby(["customer_id", "invoice_no"])["amount"].sum().reset_index()
hist_aov = hist_order_totals.groupby("customer_id")["amount"].mean().reset_index()
hist_aov.columns = ["customer_id", "avg_order_value"]
hist_features = hist_features.merge(hist_aov, on="customer_id", how="left")

hist_features["recency_days"] = (split_date - hist_features["last_purchase"]).dt.days
hist_features["tenure_days"] = (hist_features["last_purchase"] - hist_features["first_purchase"]).dt.days
hist_features["orders_per_active_day"] = hist_features["total_orders"] / hist_features["active_days"].clip(lower=1)

# Target
hist_features["label"] = hist_features["customer_id"].isin(future_customers).astype(int)

print(f"\nLabel distribution:")
print(hist_features["label"].value_counts())
print(f"Repurchase rate in test window: {hist_features['label'].mean()*100:.1f}%")

# Prepare features
feature_cols = ["total_orders", "total_amount", "avg_order_value", "active_days",
                "distinct_products", "recency_days", "tenure_days", "orders_per_active_day"]

X = hist_features[feature_cols].copy()
# Fill NaN
X = X.fillna(0)
y = hist_features["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_prob)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"\nLogistic Regression: Accuracy={lr_acc:.3f}, AUC={lr_auc:.3f}")

# Model 2: Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_prob = dt.predict_proba(X_test)[:, 1]
dt_auc = roc_auc_score(y_test, dt_prob)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"Decision Tree: Accuracy={dt_acc:.3f}, AUC={dt_auc:.3f}")

# Use better model
best_model = lr if lr_auc >= dt_auc else dt
best_name = "Logistic Regression" if lr_auc >= dt_auc else "Decision Tree"
best_auc = max(lr_auc, dt_auc)
print(f"\nBest model: {best_name} (AUC={best_auc:.3f})")

# Feature importance (from logistic regression coefficients)
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)
print("\nFeature importance (Logistic Regression):")
print(coef_df.round(4))

# ---- Figure 5: Feature importance ----
fig, ax = plt.subplots(figsize=(9, 5))
coef_df_sorted = coef_df.sort_values("coefficient")
colors = [PALETTE[3] if c > 0 else PALETTE[0] for c in coef_df_sorted["coefficient"]]
ax.barh(coef_df_sorted["feature"], coef_df_sorted["coefficient"], color=colors)
ax.set_xlabel("Coefficient (positive = increases repurchase likelihood)")
ax.set_title(f"Feature Importance - Logistic Regression (AUC={lr_auc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_feature_importance.png"), bbox_inches="tight")
plt.close()
print("Saved: 05_feature_importance.png")

# ---- Figure 6: Confusion matrix ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, model_name, y_pred, y_true, auc_val in [
    (axes[0], "Logistic Regression", lr_pred, y_test, lr_auc),
    (axes[1], "Decision Tree", dt_pred, y_test, dt_auc),
]:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Repurchase", "Repurchase"],
                yticklabels=["No Repurchase", "Repurchase"])
    ax.set_title(f"{model_name}\nAUC={auc_val:.3f}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_confusion_matrix.png"), bbox_inches="tight")
plt.close()
print("Saved: 06_confusion_matrix.png")

# ============================================================
# 5. KEY FINDINGS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Key Findings Summary")
print("=" * 60)

# Pareto: top 20% customers contribution
customer_rev = rfm.sort_values("monetary", ascending=False).reset_index(drop=True)
customer_rev["cum_pct"] = customer_rev["monetary"].cumsum() / customer_rev["monetary"].sum() * 100
top20_pct = int(len(customer_rev) * 0.2)
top20_rev_pct = customer_rev["monetary"].head(top20_pct).sum() / customer_rev["monetary"].sum() * 100
print(f"Top 20% customers contribute: {top20_rev_pct:.1f}% of total revenue")

# Champions segment
champions_rev = seg_summary.loc["Champions", "pct_revenue"] if "Champions" in seg_summary.index else 0
champions_cust = seg_summary.loc["Champions", "pct_customers"] if "Champions" in seg_summary.index else 0
print(f"Champions segment: {champions_cust:.1f}% of customers, {champions_rev:.1f}% of revenue")

# Save summary stats
summary = {
    "total_customers": int(total_customers),
    "repeat_rate_pct": round(repeat_rate, 1),
    "repurchase_90d_pct": round(repurchase_90, 1),
    "repurchase_180d_pct": round(repurchase_180, 1),
    "median_gap_days": round(float(np.median(gap_days)), 1),
    "avg_gap_days": round(float(gap_days.mean()), 1),
    "top20_revenue_pct": round(top20_rev_pct, 1),
    "best_model": best_name,
    "best_auc": round(best_auc, 3),
}
print(f"\nSummary: {summary}")

import json
with open(os.path.join(REPORT_DIR, "summary_stats.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Figures saved to: {FIG_DIR}")
print(f"Processed data saved to: {PROC_DIR}")
