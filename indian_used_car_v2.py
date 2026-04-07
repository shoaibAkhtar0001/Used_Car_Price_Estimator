# ================================================================
#   INDIAN USED CAR PRICE PREDICTION — Google Colab Training Code
#   Dataset: indian_used_cars_dataset.csv  (already uploaded)
# ================================================================

# ── STEP 1: Install & Import ─────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

print("✅ All libraries imported successfully!")

# ── STEP 2: Load Dataset ─────────────────────────────────────────
df = pd.read_csv("indian_used_cars_dataset.csv")

print(f"\n📦 Dataset Shape : {df.shape}")
print(f"📋 Columns       : {list(df.columns)}")
print(f"\n🔍 First 5 rows:")
df.head()

# ── STEP 3: Basic EDA ────────────────────────────────────────────
print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Statistical Summary:")
df.describe()

# Check missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())

# Price distribution
print(f"\n💰 Price Range : ₹{df['Selling_Price_L'].min():.2f}L — ₹{df['Selling_Price_L'].max():.2f}L")
print(f"   Mean Price  : ₹{df['Selling_Price_L'].mean():.2f}L")
print(f"   Median Price: ₹{df['Selling_Price_L'].median():.2f}L")

# ── STEP 4: Visualizations ───────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Indian Used Car Dataset — EDA", fontsize=15, fontweight="bold")

# 1. Price Distribution
axes[0,0].hist(df["Selling_Price_L"].clip(upper=50), bins=40,
               color="#7c3aed", edgecolor="white")
axes[0,0].set_title("Selling Price Distribution")
axes[0,0].set_xlabel("₹ Lakhs")

# 2. Age vs Price
age_price = df.groupby("Age_Years")["Selling_Price_L"].median()
axes[0,1].plot(age_price.index, age_price.values,
               marker="o", color="#c4b5fd", lw=2)
axes[0,1].fill_between(age_price.index, age_price*0.9, age_price*1.1,
                        alpha=0.2, color="#7c3aed")
axes[0,1].set_title("Price vs Car Age (Depreciation)")
axes[0,1].set_xlabel("Age (Years)"); axes[0,1].set_ylabel("₹ Lakhs")

# 3. Fuel Type vs Price
fuel_price = df.groupby("Fuel_Type")["Selling_Price_L"].median().sort_values(ascending=False)
axes[0,2].bar(fuel_price.index, fuel_price.values, color="#4C72B0", edgecolor="white")
axes[0,2].set_title("Median Price by Fuel Type")
axes[0,2].set_ylabel("₹ Lakhs"); axes[0,2].tick_params(axis="x", rotation=30)

# 4. Owner Type vs Price
owner_order = ["First Owner","Second Owner","Third Owner","Fourth & Above Owner"]
owner_price = df.groupby("Owner")["Selling_Price_L"].median().reindex(
    [o for o in owner_order if o in df["Owner"].unique()])
axes[1,0].bar(owner_price.index, owner_price.values, color="#DD8452", edgecolor="white")
axes[1,0].set_title("Price by Owner Type")
axes[1,0].set_ylabel("₹ Lakhs"); axes[1,0].tick_params(axis="x", rotation=25)

# 5. Transmission vs Price
trans_price = df.groupby("Transmission")["Selling_Price_L"].median()
axes[1,1].bar(trans_price.index, trans_price.values, color="#55A868", edgecolor="white")
axes[1,1].set_title("Price by Transmission")
axes[1,1].set_ylabel("₹ Lakhs")

# 6. Km Driven vs Price
axes[1,2].scatter(df["Km_Driven"], df["Selling_Price_L"],
                  alpha=0.2, s=8, color="#8172B3")
axes[1,2].set_title("Km Driven vs Selling Price")
axes[1,2].set_xlabel("Km Driven"); axes[1,2].set_ylabel("₹ Lakhs")

plt.tight_layout()
plt.savefig("eda_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ EDA charts saved as eda_charts.png")

# ── STEP 5: Feature Engineering ──────────────────────────────────
print("\n⚙️  Encoding categorical features...")

# Identify categorical columns
cat_cols = []
for col in ["Brand", "Model", "Variant", "Fuel_Type",
            "Transmission", "Owner", "Seller_Type"]:
    if col in df.columns:
        cat_cols.append(col)

# Also handle Car_Name if Brand/Model/Variant not present
if "Car_Name" in df.columns and "Brand" not in df.columns:
    cat_cols.append("Car_Name")

print(f"   Categorical columns found: {cat_cols}")

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"   ✅ Encoded: {col} ({df[col].nunique()} unique values)")

# ── STEP 6: Define Features & Target ─────────────────────────────
# Auto-detect available feature columns
all_possible_features = {
    "Brand_enc", "Model_enc", "Variant_enc", "Car_Name_enc",
    "Year", "Age_Years", "Km_Driven",
    "Fuel_Type_enc", "Transmission_enc", "Owner_enc", "Seller_Type_enc",
    "Mileage_kmpl", "Engine_CC", "Max_Power_BHP", "Torque_Nm",
    "Seats", "Original_Price_L"
}

feature_cols = [f for f in all_possible_features if f in df.columns]
print(f"\n📌 Features selected ({len(feature_cols)}): {feature_cols}")

TARGET = "Selling_Price_L"
X = df[feature_cols]
y = df[TARGET]

print(f"\n🎯 Target column : {TARGET}")
print(f"   X shape       : {X.shape}")
print(f"   y shape       : {y.shape}")

# ── STEP 7: Train/Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\n✂️  Train size : {len(X_train)}  |  Test size : {len(X_test)}")

# ── STEP 8: Train Random Forest ───────────────────────────────────
print("\n🌲 Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("   ✅ Random Forest trained!")

# ── STEP 9: Train Gradient Boosting ───────────────────────────────
print("\n🚀 Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.07,
    max_depth=5,
    subsample=0.85,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print("   ✅ Gradient Boosting trained!")

# ── STEP 10: Ensemble Prediction ──────────────────────────────────
ens_pred = rf_pred * 0.45 + gb_pred * 0.55

# ── STEP 11: Evaluate Models ──────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n  📊 [{name}]")
    print(f"     MAE   : ₹{mae:.2f} Lakhs")
    print(f"     RMSE  : ₹{rmse:.2f} Lakhs")
    print(f"     R²    : {r2:.4f}  ({r2*100:.1f}% variance explained)")
    print(f"     MAPE  : {mape:.2f}%")
    return {"name":name, "MAE":mae, "RMSE":rmse, "R2":r2, "MAPE":mape}

print("\n" + "="*55)
print("   MODEL EVALUATION RESULTS")
print("="*55)
res_rf  = evaluate("Random Forest",     y_test, rf_pred)
res_gb  = evaluate("Gradient Boosting", y_test, gb_pred)
res_ens = evaluate("Ensemble (RF+GB)",  y_test, ens_pred)

# Best model summary
results_df = pd.DataFrame([res_rf, res_gb, res_ens])
print("\n📋 Comparison Table:")
print(results_df.to_string(index=False))

# ── STEP 12: Cross Validation ─────────────────────────────────────
print("\n🔄 Running 5-Fold Cross Validation (Random Forest)...")
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="r2")
print(f"   CV R² scores : {[round(s,4) for s in cv_scores]}")
print(f"   Mean R²      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── STEP 13: Confidence Intervals ─────────────────────────────────
print("\n📐 CONFIDENCE INTERVALS — Sample Predictions (90% CI)")
print(f"  {'#':<3} {'Actual':>10} {'Predicted':>11} {'CI Low':>9} {'CI High':>9} {'Width':>8}")
print(f"  {'─'*55}")

sample_X = X_test.iloc[:10]
tree_preds = np.array([t.predict(sample_X) for t in rf_model.estimators_])
ci_lo = np.percentile(tree_preds, 5,  axis=0)
ci_hi = np.percentile(tree_preds, 95, axis=0)
ci_mu = tree_preds.mean(axis=0)

for i in range(10):
    act = y_test.iloc[i]
    print(f"  {i+1:<3} ₹{act:>7.2f}L   ₹{ci_mu[i]:>7.2f}L  "
          f"₹{ci_lo[i]:>6.2f}L  ₹{ci_hi[i]:>6.2f}L  "
          f"±{(ci_hi[i]-ci_lo[i])/2:>5.2f}")

# ── STEP 14: Model Performance Plots ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Performance Dashboard", fontsize=14, fontweight="bold")

# Actual vs Predicted
lim = max(y_test.max(), ens_pred.max()) * 1.05
axes[0].scatter(y_test, ens_pred, alpha=0.3, s=10, color="#7c3aed")
axes[0].plot([0, lim], [0, lim], "r--", lw=2, label="Perfect Fit")
axes[0].set_title("Actual vs Predicted (Ensemble)")
axes[0].set_xlabel("Actual ₹ Lakhs"); axes[0].set_ylabel("Predicted ₹ Lakhs")
axes[0].legend()

# Residuals
resid = y_test.values - ens_pred
axes[1].scatter(ens_pred, resid, alpha=0.3, s=10, color="#DD8452")
axes[1].axhline(0, color="red", lw=2, linestyle="--")
axes[1].set_title("Residuals Plot")
axes[1].set_xlabel("Predicted ₹ Lakhs"); axes[1].set_ylabel("Residual")

# Feature Importance
fi = pd.Series(rf_model.feature_importances_,
               index=feature_cols).sort_values()
fi.tail(12).plot(kind="barh", ax=axes[2], color="#4C72B0")
axes[2].set_title("Top 12 Feature Importances")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("model_performance.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Performance plots saved as model_performance.png")

# ── STEP 15: Save Models ──────────────────────────────────────────
print("\n💾 Saving trained models...")

with open("rf_model.pkl",      "wb") as f: pickle.dump(rf_model,     f)
with open("gb_model.pkl",      "wb") as f: pickle.dump(gb_model,     f)
with open("le_dict.pkl",       "wb") as f: pickle.dump(le_dict,      f)
with open("feature_cols.pkl",  "wb") as f: pickle.dump(feature_cols, f)

print("   ✅ rf_model.pkl      — Random Forest")
print("   ✅ gb_model.pkl      — Gradient Boosting")
print("   ✅ le_dict.pkl       — Label Encoders")
print("   ✅ feature_cols.pkl  — Feature column names")

# ── STEP 16: Predict on a New Car ─────────────────────────────────
print("\n" + "="*55)
print("   🔮 PREDICT ON A NEW CAR EXAMPLE")
print("="*55)

def predict_new_car(
    brand="Hyundai", model_name="Creta", variant="SX",
    year=2020, km_driven=42000,
    fuel="Petrol", transmission="Automatic",
    owner="First Owner", seller="Individual",
    mileage=16.8, engine_cc=1497, max_power=113.0,
    torque=143, seats=5, original_price=14.99
):
    age = 2024 - year
    row = {}

    # Encode each categorical — handle unseen values gracefully
    for col, val in [("Brand", brand), ("Model", model_name),
                     ("Variant", variant), ("Fuel_Type", fuel),
                     ("Transmission", transmission),
                     ("Owner", owner), ("Seller_Type", seller)]:
        if col in le_dict:
            if val in le_dict[col].classes_:
                row[col+"_enc"] = le_dict[col].transform([val])[0]
            else:
                row[col+"_enc"] = 0  # unknown category fallback

    # Numeric
    row.update({
        "Year": year, "Age_Years": age, "Km_Driven": km_driven,
        "Mileage_kmpl": mileage, "Engine_CC": engine_cc,
        "Max_Power_BHP": max_power, "Torque_Nm": torque,
        "Seats": seats, "Original_Price_L": original_price,
    })

    # Handle Car_Name_enc if needed
    if "Car_Name_enc" in feature_cols and "Car_Name" in le_dict:
        car_name = f"{brand} {model_name} {variant} {year}"
        row["Car_Name_enc"] = (le_dict["Car_Name"].transform([car_name])[0]
                               if car_name in le_dict["Car_Name"].classes_ else 0)

    X_new = pd.DataFrame([row])[feature_cols]

    # Predict + CI
    tree_preds = np.array([t.predict(X_new)[0] for t in rf_model.estimators_])
    ci_lo = np.percentile(tree_preds, 5)
    ci_hi = np.percentile(tree_preds, 95)
    rf_p  = tree_preds.mean()
    gb_p  = gb_model.predict(X_new)[0]
    final = round(rf_p * 0.45 + gb_p * 0.55, 2)
    dep   = round((1 - final / original_price) * 100, 1)

    print(f"\n  🚗  {brand} {model_name} {variant} ({year})")
    print(f"  📌  {fuel} | {transmission} | {owner} | {km_driven:,} km")
    print(f"  💰  Predicted Price : ₹{final} Lakhs")
    print(f"  📊  90% CI         : ₹{round(ci_lo,2)}L  –  ₹{round(ci_hi,2)}L")
    print(f"  📉  Depreciation   : {dep}% from ₹{original_price}L")
    return final

# Run 3 example predictions
predict_new_car("Hyundai","Creta","SX",        2020, 42000,
                "Petrol","Automatic","First Owner","Individual",
                16.8, 1497, 113.0, 143, 5, 14.99)

predict_new_car("Maruti","Swift","VXI",         2018, 65000,
                "Petrol","Manual","Second Owner","Dealer",
                21.2, 1197, 83.0, 113, 5, 6.99)

predict_new_car("Toyota","Fortuner","4x4 AT",   2019, 55000,
                "Diesel","Automatic","First Owner","Trustmark Dealer",
                14.8, 2755, 170.0, 420, 7, 40.57)

print("\n" + "="*55)
print("✅  TRAINING COMPLETE — All models saved!")
print("="*55)
print("""
📁 Files in your Colab:
   indian_used_cars_dataset.csv  ← input dataset
   rf_model.pkl                  ← Random Forest model
   gb_model.pkl                  ← Gradient Boosting model
   le_dict.pkl                   ← Label encoders
   feature_cols.pkl              ← Feature names
   eda_charts.png                ← EDA visualizations
   model_performance.png         ← Model evaluation plots
""")
