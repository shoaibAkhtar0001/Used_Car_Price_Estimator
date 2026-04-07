"""
=======================================================================
  INDIAN USED CAR PRICE PREDICTOR — Streamlit App
  Run with: streamlit run app.py
=======================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Indian Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS (Modern Clean Dashboard)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5 {
    font-family: 'Outfit', sans-serif !important;
}

/* Metric Cards Hover & Shadow */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    border: 1px solid #f0f0f0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}
[data-testid="stMetricValue"] {
    color: #1e293b !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    color: #64748b !important;
}

/* Sidebar Button Styling */
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    padding: 0.75rem 1rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stButton>button:hover {
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4) !important;
    transform: translateY(-2px);
}

/* Modern Result Box */
.modern-result-box {
    background: linear-gradient(135deg, #ffffff, #f8fafc);
    border-left: 6px solid #2563eb;
    border-radius: 12px;
    padding: 2.5rem;
    margin: 1.5rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}
.result-label {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
    margin-top: 0;
}
.result-value {
    color: #0f172a;
    font-size: 4rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.5rem 0;
}
.result-ci {
    color: #475569;
    font-size: 1.1rem;
    font-weight: 500;
    margin: 0;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD AND PREPARE DATASET + MAPS
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("indian_used_cars_v2 (1).csv")
    
    # Ensure all necessary target columns
    if "Year" not in df.columns and "Age_Years" in df.columns:
        df["Year"] = 2024 - df["Age_Years"]
    
    brands = sorted(df["Brand"].astype(str).unique())
    
    brand_models = {
        b: sorted(df[df["Brand"]==b]["Model"].astype(str).unique()) for b in brands
    }
    
    # Map variants
    v_data = df.groupby(["Brand", "Model", "Variant"]).agg({
        "Fuel_Type": lambda x: x.mode()[0] if not x.empty else "Petrol",
        "Transmission": lambda x: x.mode()[0] if not x.empty else "Manual",
        "Engine_CC": "median",
        "Seats": "median",
        "Original_Price_L": "median",
        "Selling_Price_L": "count"
    }).reset_index()
    
    model_variants = {}
    for _, r in v_data.iterrows():
        b, m = str(r["Brand"]), str(r["Model"])
        if (b, m) not in model_variants:
            model_variants[(b,m)] = []
        model_variants[(b,m)].append((
            str(r["Variant"]), str(r["Fuel_Type"]), str(r["Transmission"]), 
            int(r["Engine_CC"]), int(r["Seats"]), round(r["Original_Price_L"], 2)
        ))
        
    return df, brands, brand_models, model_variants

df, brands, brand_models, model_variants = load_and_prep_data()

OWNER_LIST   = ["First Owner","Second Owner","Third Owner","Fourth & Above Owner"]
SELLER_LIST  = ["Individual","Dealer","Trustmark Dealer"]

def retained_value(age, base_price, owner):
    if base_price >= 40:   annual_dep = 0.055
    elif base_price >= 15: annual_dep = 0.065
    else:                  annual_dep = 0.080
    base_ret = 0.85 * ((1 - annual_dep) ** max(0, age - 1))
    owner_mult = {"First Owner":1.00,"Second Owner":0.90,
                  "Third Owner":0.82,"Fourth & Above Owner":0.72}
    return base_ret * owner_mult.get(owner, 0.90)

# ─────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Training model on real dataset...")
def train_model(dataframe):
    df_copy = dataframe.copy()
    cat_cols = ["Brand","Model","Variant","Fuel_Type","Transmission","Owner","Seller_Type"]
    
    # Encode categorical columns
    le_dict = {}
    for col in cat_cols:
        if col in df_copy.columns:
            le = LabelEncoder()
            df_copy[col+"_enc"] = le.fit_transform(df_copy[col].astype(str))
            le_dict[col] = le
            
    # Auto-detect available feature columns
    all_possible = ["Brand_enc","Model_enc","Variant_enc","Year","Age_Years",
                    "Km_Driven","Fuel_Type_enc","Transmission_enc","Owner_enc",
                    "Seller_Type_enc","Mileage_kmpl","Engine_CC","Max_Power_BHP",
                    "Torque_Nm","Seats","Original_Price_L"]
                    
    feature_cols = [f for f in all_possible if f in df_copy.columns]
    
    X = df_copy[feature_cols]
    y = df_copy["Selling_Price_L"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=300, max_depth=18,
                               min_samples_split=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.07,
                                   max_depth=5, subsample=0.85, random_state=42)
    gb.fit(X_train, y_train)
    
    rf_pred  = rf.predict(X_test)
    gb_pred  = gb.predict(X_test)
    ens_pred = rf_pred * 0.45 + gb_pred * 0.55
    metrics = {
        "r2":   round(r2_score(y_test, ens_pred)*100, 2),
        "mae":  round(mean_absolute_error(y_test, ens_pred), 2),
        "rmse": round(mean_squared_error(y_test, ens_pred)**0.5, 2),
        "mape": round(np.mean(np.abs((y_test - ens_pred)/y_test))*100, 2),
        "rows": len(df_copy),
    }
    return rf, gb, le_dict, feature_cols, metrics

def predict(rf, gb, le_dict, feature_cols,
            brand, model, variant, year, km_driven,
            fuel, trans, owner, seller,
            mileage, engine_cc, max_power, torque, seats, base_price):
    age = 2024 - year
    def enc(col, val):
        if col in le_dict:
            le = le_dict[col]
            if val in le.classes_:
                return le.transform([val])[0]
        return 0
    row = {
        "Brand_enc":       enc("Brand",        brand),
        "Model_enc":       enc("Model",        model),
        "Variant_enc":     enc("Variant",      variant),
        "Year":            year,
        "Age_Years":       age,
        "Km_Driven":       km_driven,
        "Fuel_Type_enc":   enc("Fuel_Type",    fuel),
        "Transmission_enc":enc("Transmission", trans),
        "Owner_enc":       enc("Owner",        owner),
        "Seller_Type_enc": enc("Seller_Type",  seller),
        "Mileage_kmpl":    mileage,
        "Engine_CC":       engine_cc,
        "Max_Power_BHP":   max_power,
        "Torque_Nm":       torque,
        "Seats":           seats,
        "Original_Price_L":base_price,
    }
    
    row_filtered = {k: v for k, v in row.items() if k in feature_cols}
    X_new = pd.DataFrame([row_filtered])[feature_cols]
    
    tree_preds = np.array([t.predict(X_new)[0] for t in rf.estimators_])
    ci_lo = np.percentile(tree_preds, 5)
    ci_hi = np.percentile(tree_preds, 95)
    rf_p  = tree_preds.mean()
    gb_p  = gb.predict(X_new)[0]
    final = rf_p * 0.45 + gb_p * 0.55
    final = max(0.5, min(final, base_price))
    ci_lo = max(0.5, ci_lo)
    ci_hi = min(ci_hi, base_price)
    return round(final,2), round(ci_lo,2), round(ci_hi,2)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
rf, gb, le_dict, feature_cols, metrics = train_model(df)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🚗 Indian Used Car Price Predictor")
st.markdown("ML-powered valuation engine trained on realistically derived Indian market entries.")



# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Data Explorer", "📈 Model Insights"])

# ════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════
with tab1:
    # ── SIDEBAR INPUTS ──
    st.sidebar.header("1. Car Selection")
    sel_brand = st.sidebar.selectbox("Brand", brands)
    
    models_for_brand = brand_models.get(sel_brand, [])
    sel_model = st.sidebar.selectbox("Model", models_for_brand)
    
    variants_for_model = model_variants.get((sel_brand, sel_model), [])
    variant_labels = [f"{v[0]} ({v[1]}, {v[2]})" for v in variants_for_model]
    
    if len(variant_labels) > 0:
        sel_variant_idx = st.sidebar.selectbox("Variant", range(len(variant_labels)), format_func=lambda i: variant_labels[i])
        sel_variant_data = variants_for_model[sel_variant_idx]
        variant_name, fuel, trans, engine_cc, seats, base_price = sel_variant_data
    else:
        variant_name, fuel, trans, engine_cc, seats, base_price = "Unknown", "Petrol", "Manual", 1000, 5, 5.0
        st.sidebar.warning("No variants found for this model.")

    st.sidebar.header("2. Condition & History")
    sel_year = st.sidebar.selectbox("Year of Purchase", list(range(2024, 2012, -1)))
    
    age_yr   = 2024 - sel_year
    max_km   = max(1000, age_yr * 25000)
    exp_km   = age_yr * 11000
    km_driven = st.sidebar.number_input("Km Driven", min_value=0, value=min(exp_km, max_km), step=500)
    
    max_owner_idx = min(age_yr, 3)
    owner_choices = OWNER_LIST[:max_owner_idx+1]
    owner = st.sidebar.selectbox("Owner Type", owner_choices)
    
    seller = st.sidebar.selectbox("Seller Type", SELLER_LIST)

    mileage_map = {"Petrol":(14,22),"Diesel":(18,27),"CNG":(22,30),
                   "Electric":(250,430),"Hybrid":(18,26),"LPG":(12,18)}
    m_lo, m_hi = mileage_map.get(fuel, (14,22))
    mileage = round((m_lo + m_hi) / 2, 1)

    if fuel == "Electric":
        max_power = round(np.random.uniform(80,150),1)
        torque    = int(max_power * 2.4)
    else:
        max_power = round(engine_cc * 0.068, 1)
        torque    = int(max_power   * 2.2)

    st.sidebar.divider()
    predict_btn = st.sidebar.button("🔮 Predict Engine", type="primary", use_container_width=True)
    
    # ── MAIN SCREEN OUTPUT ──
    st.subheader("Vehicle Specifications")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("Fuel Type", fuel)
    sc2.metric("Transmission", trans)
    sc3.metric("Engine CC", f"{engine_cc} cc" if engine_cc > 0 else "Electric")
    sc4.metric("Seats", seats)
    sc5.metric("Original Price", f"₹{base_price}L")
    
    st.divider()

    if predict_btn or 'price' not in st.session_state:
        price, ci_lo, ci_hi = predict(
            rf, gb, le_dict, feature_cols,
            sel_brand, sel_model, variant_name, sel_year,
            km_driven, fuel, trans, owner, seller,
            mileage, engine_cc, max_power, torque, seats, base_price
        )
        st.session_state['price'] = price
        st.session_state['ci_lo'] = ci_lo
        st.session_state['ci_hi'] = ci_hi
        st.session_state['base_price'] = base_price
        st.session_state['sel_year'] = sel_year

    if 'price' in st.session_state:
        price = st.session_state['price']
        ci_lo = st.session_state['ci_lo']
        ci_hi = st.session_state['ci_hi']
        base_price = st.session_state['base_price']
        sel_year = st.session_state['sel_year']
        
        dep_pct = round((1 - price/base_price)*100, 1) if base_price > 0 else 0
        
        st.markdown(f"""
        <div class="modern-result-box">
            <p class="result-label">ESTIMATED SELLING PRICE</p>
            <h1 class="result-value">₹ {price} Lakhs</h1>
            <p class="result-ci">90% Confidence Interval: ₹{ci_lo}L — ₹{ci_hi}L</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Price Breakdown")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Original Price", f"₹{base_price}L")
        b2.metric("Current Value", f"₹{price}L", f"-{dep_pct}% Depreciation")
        b3.metric("Accuracy Spread", f"±₹{round((ci_hi-ci_lo)/2,2)}L")
        b4.metric("Age at Sale", f"{2024-sel_year} years")
        
        ratio = price / base_price if base_price > 0 else 0.0
        st.progress(ratio, text=f"Retained {round(ratio*100, 1)}% of original value")

# ════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ════════════════════════════════════════════
with tab2:
    st.subheader("Dataset Overview")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Rows",     f"{len(df):,}")
    d2.metric("Brands",         df["Brand"].nunique())
    d3.metric("Unique Variants",df["Variant"].nunique())
    d4.metric("Year Range",     f"{df['Year'].min()}–{df['Year'].max()}")

    # Filters
    st.subheader("Filter & Explore")
    f1, f2, f3 = st.columns(3)
    with f1:
        filter_brand = st.multiselect("Filter by Brand", brands,
                                       default=["Maruti","Hyundai","Tata"])
    with f2:
        filter_fuel = st.multiselect("Filter by Fuel",
                                      df["Fuel_Type"].unique().tolist(),
                                      default=["Petrol","Diesel"])
    with f3:
        price_range = st.slider("Price Range (₹ Lakhs)",
                                 0.5, 70.0, (0.5, 30.0), 0.5)

    fdf = df.copy()
    if filter_brand: fdf = fdf[fdf["Brand"].isin(filter_brand)]
    if filter_fuel:  fdf = fdf[fdf["Fuel_Type"].isin(filter_fuel)]
    fdf = fdf[(fdf["Selling_Price_L"] >= price_range[0]) &
              (fdf["Selling_Price_L"] <= price_range[1])]

    st.caption(f"Showing {len(fdf):,} rows after filters")
    st.dataframe(
        fdf[["Brand","Model","Variant","Year","Age_Years","Km_Driven",
             "Fuel_Type","Transmission","Owner","Seller_Type",
             "Original_Price_L","Selling_Price_L"]]
        .sort_values("Selling_Price_L", ascending=False)
        .head(200)
        .reset_index(drop=True),
        width="stretch", height=380
    )

    # Charts
    st.subheader("Visual Analysis")
    ch1, ch2 = st.columns(2)

    with ch1:
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#0f0c29")
        ax.set_facecolor("#0f0c29")
        age_p = fdf.groupby("Age_Years")["Selling_Price_L"].median()
        ax.plot(age_p.index, age_p.values, color="#c4b5fd", lw=2.5, marker="o")
        ax.fill_between(age_p.index, age_p*0.90, age_p*1.10,
                        alpha=0.15, color="#7c3aed")
        ax.set_title("Price vs Age", color="white", fontsize=11)
        ax.set_xlabel("Age (Years)", color="#8888aa")
        ax.set_ylabel("Median ₹ Lakhs", color="#8888aa")
        ax.tick_params(colors="#8888aa")
        for spine in ax.spines.values(): spine.set_edgecolor("#333355")
        st.pyplot(fig); plt.close()

    with ch2:
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#0f0c29")
        ax.set_facecolor("#0f0c29")
        fp = fdf.groupby("Fuel_Type")["Selling_Price_L"].median().sort_values(ascending=False)
        bars = ax.bar(fp.index, fp.values, color="#7c3aed", edgecolor="#c4b5fd", linewidth=0.5)
        for bar in bars: bar.set_alpha(0.85)
        ax.set_title("Median Price by Fuel", color="white", fontsize=11)
        ax.set_ylabel("₹ Lakhs", color="#8888aa")
        ax.tick_params(colors="#8888aa", axis="x", rotation=30)
        ax.tick_params(colors="#8888aa", axis="y")
        for spine in ax.spines.values(): spine.set_edgecolor("#333355")
        st.pyplot(fig); plt.close()

    ch3, ch4 = st.columns(2)
    with ch3:
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#0f0c29")
        ax.set_facecolor("#0f0c29")
        op = fdf.groupby("Owner")["Selling_Price_L"].median().reindex(
             [o for o in OWNER_LIST if o in fdf["Owner"].unique()])
        colors = ["#c4b5fd","#a78bfa","#7c3aed","#5b21b6"]
        ax.bar(range(len(op)), op.values,
               color=colors[:len(op)], edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(op)))
        ax.set_xticklabels([o.replace(" Owner","") for o in op.index],
                            rotation=20, color="#8888aa", fontsize=9)
        ax.set_title("Price by Ownership", color="white", fontsize=11)
        ax.set_ylabel("₹ Lakhs", color="#8888aa")
        ax.tick_params(colors="#8888aa")
        for spine in ax.spines.values(): spine.set_edgecolor("#333355")
        st.pyplot(fig); plt.close()

    with ch4:
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#0f0c29")
        ax.set_facecolor("#0f0c29")
        fdf["Selling_Price_L"].clip(upper=50).hist(
            bins=40, ax=ax, color="#7c3aed", edgecolor="#c4b5fd", linewidth=0.4)
        ax.set_title("Price Distribution", color="white", fontsize=11)
        ax.set_xlabel("₹ Lakhs", color="#8888aa")
        ax.tick_params(colors="#8888aa")
        for spine in ax.spines.values(): spine.set_edgecolor("#333355")
        st.pyplot(fig); plt.close()

# ════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Accuracy",   f"{metrics['r2']}%")
    m2.metric("MAE",           f"₹{metrics['mae']}L")
    m3.metric("RMSE",          f"₹{metrics['rmse']}L")
    m4.metric("MAPE",          f"{metrics['mape']}%")

    fi_col, desc_col = st.columns([2,1])
    with fi_col:
        st.subheader("Feature Importances")
        fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
        fig, ax = plt.subplots(figsize=(7,5), facecolor="#0f0c29")
        ax.set_facecolor("#0f0c29")
        colors_fi = plt.cm.plasma(np.linspace(0.3, 0.9, len(fi.tail(12))))
        fi.tail(12).plot(kind="barh", ax=ax, color=colors_fi)
        ax.set_title("Top 12 Important Features", color="white", fontsize=11)
        ax.tick_params(colors="#8888aa", labelsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor("#333355")
        st.pyplot(fig); plt.close()

    with desc_col:
        st.subheader("How It Works")
        st.info("""
        **🤖 Ensemble Model**\n
        Combines Random Forest (45%) + Gradient Boosting (55%)\n\n
        **📐 Confidence Interval**\n
        Derived from spread of 300 individual decision trees\n\n
        **📉 Depreciation Logic**\n
        Budget cars: ~8%/yr | Premium cars: ~6.5%/yr | Luxury cars: ~5.5%/yr\n\n
        **🔑 Key Price Factors**\n
        Original price → Age → Km driven → Owner history
        """)

    st.subheader("Depreciation by Segment")
    fig, ax = plt.subplots(figsize=(10,4), facecolor="#0f0c29")
    ax.set_facecolor("#0f0c29")
    ages = np.arange(0, 13)
    for label, base, color in [
        ("Budget (₹4L)",   4,  "#c4b5fd"),
        ("Mid (₹12L)",     12, "#7c3aed"),
        ("Premium (₹25L)", 25, "#a78bfa"),
        ("Luxury (₹55L)",  55, "#5b21b6"),
    ]:
        prices = [base * retained_value(a, base, "First Owner") for a in ages]
        ax.plot(ages, prices, label=label, color=color, lw=2, marker="o", markersize=4)
    ax.set_xlabel("Car Age (Years)", color="#8888aa")
    ax.set_ylabel("₹ Lakhs", color="#8888aa")
    ax.set_title("Depreciation Curves by Segment", color="white", fontsize=11)
    ax.legend(facecolor="#1a1a2e", labelcolor="#c4b5fd", edgecolor="#333355")
    ax.tick_params(colors="#8888aa")
    for spine in ax.spines.values(): spine.set_edgecolor("#333355")
    st.pyplot(fig); plt.close()

