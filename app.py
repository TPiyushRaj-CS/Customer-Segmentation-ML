from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation Studio", page_icon="📊", layout="wide")
st.markdown("""
<style>
.main {background: #f7f9fc;}
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.hero {padding: 1.35rem 1.6rem; border-radius: 18px; background: linear-gradient(110deg,#e6f4f1,#eaf0ff); margin-bottom: 1rem;}
.hero h1 {margin:0; color:#172b4d; font-size:2rem;}
.hero p {margin:.4rem 0 0; color:#42526e;}
div[data-testid="stMetric"] {background:white; border:1px solid #e6eaf0; padding:15px; border-radius:14px;}
</style>
<div class="hero">
<h1>Customer Segmentation Studio</h1>
<p>Explore customer demographics and purchase behavior with K-Means clustering.</p>
</div>
""", unsafe_allow_html=True)

DEMO_PATH = Path(__file__).parent / "data" / "customer_data_demo.csv"

with st.sidebar:
    st.header("Project controls")
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    st.caption("Expected numeric features can include age, income, spending score, purchase frequency, and order value.")
    st.divider()
    st.markdown("**Methodology**")
    st.write("1. Validate and clean data")
    st.write("2. Scale selected numeric features")
    st.write("3. Compare cluster counts")
    st.write("4. Profile segments and export results")

try:
    raw = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(DEMO_PATH)
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

if raw.empty:
    st.error("The uploaded dataset has no rows.")
    st.stop()

st.subheader("1 · Data overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customer records", f"{len(raw):,}")
c2.metric("Columns", len(raw.columns))
c3.metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
c4.metric("Duplicate rows", f"{int(raw.duplicated().sum()):,}")
with st.expander("Preview dataset"):
    st.dataframe(raw.head(20), use_container_width=True)
with st.expander("Column summary"):
    st.dataframe(raw.describe(include="all").transpose(), use_container_width=True)

# Numeric feature choices, excluding ID-like columns by default
numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()
id_like = [c for c in numeric_cols if "id" in c.lower() or c.lower().endswith("_id")]
available_features = [c for c in numeric_cols if c not in id_like]
if len(available_features) < 2:
    st.warning("Please upload data with at least two numeric customer features.")
    st.stop()

st.subheader("2 · Configure segmentation")
left, right = st.columns([2,1])
with left:
    defaults = [c for c in available_features if any(k in c.lower() for k in ["income", "spending", "age", "purchase", "order", "frequency", "amount", "recency"])]
    if len(defaults) < 2:
        defaults = available_features[:min(3, len(available_features))]
    features = st.multiselect("Features used by K-Means", available_features, default=defaults[:5])
with right:
    k = st.slider("Number of segments (K)", min_value=2, max_value=min(10, max(2, len(raw)-1)), value=min(5, max(2, len(raw)-1)))
    random_state = st.number_input("Reproducibility seed", min_value=0, max_value=9999, value=42, step=1)

if len(features) < 2:
    st.info("Select at least two numeric features to build a meaningful segmentation.")
    st.stop()

# Cleaning: median imputation on selected features; drop exact duplicate rows
working = raw.drop_duplicates().copy()
for col in features:
    working[col] = pd.to_numeric(working[col], errors="coerce")
working = working.dropna(subset=features, how="all")
for col in features:
    working[col] = working[col].fillna(working[col].median())
if len(working) < k:
    st.error(f"Need at least {k} usable rows for K={k}; current rows: {len(working)}.")
    st.stop()

X = working[features].astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow and silhouette comparison
k_values = list(range(2, min(10, len(working)-1) + 1))
inertias, silhouettes = [], []
for kval in k_values:
    km = KMeans(n_clusters=kval, random_state=int(random_state), n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(float(km.inertia_))
    silhouettes.append(float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else -1)

model = KMeans(n_clusters=k, random_state=int(random_state), n_init=10)
working["Cluster"] = model.fit_predict(X_scaled)
# Make cluster IDs interpretable with neutral labels, not assumptions
working["Segment"] = working["Cluster"].map(lambda x: f"Segment {x+1}")
profile = working.groupby(["Cluster", "Segment"], as_index=False).agg(
    Customers=("Cluster", "size"),
    **{f"Average {col}": (col, "mean") for col in features}
)
profile["Customer Share (%)"] = (profile["Customers"] / len(working) * 100).round(1)

sil = silhouette_score(X_scaled, working["Cluster"]) if len(set(working["Cluster"])) > 1 else float("nan")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers analyzed", f"{len(working):,}")
m2.metric("Segments created", k)
m3.metric("Silhouette score", f"{sil:.3f}" if not np.isnan(sil) else "N/A")
m4.metric("Features selected", len(features))

st.subheader("3 · Cluster quality")
col_a, col_b = st.columns(2)
with col_a:
    fig = px.line(x=k_values, y=inertias, markers=True, labels={"x":"Number of clusters (K)", "y":"Inertia"}, title="Elbow method")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
with col_b:
    fig = px.line(x=k_values, y=silhouettes, markers=True, labels={"x":"Number of clusters (K)", "y":"Silhouette score"}, title="Silhouette score by K")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
st.caption("Use the elbow curve and silhouette score as decision aids; cluster quality also depends on whether the groups make business sense.")

st.subheader("4 · Visualize customer segments")
plot_x = st.selectbox("Horizontal axis", features, index=0)
plot_y = st.selectbox("Vertical axis", features, index=1)
fig = px.scatter(working, x=plot_x, y=plot_y, color="Segment",
                 hover_data=[c for c in ["CustomerID", "Gender", "Age"] if c in working.columns],
                 title="Customer map by selected features", opacity=.82)
fig.update_layout(template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(profile, x="Segment", y="Customers", color="Segment", title="Customers per segment", text="Customers")
    fig.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with c2:
    share_col = "Customer Share (%)"
    fig = px.pie(profile, names="Segment", values="Customers", hole=.45, title="Customer mix")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("5 · Segment profiles")
display_profile = profile.copy()
avg_cols = [c for c in display_profile.columns if c.startswith("Average ")]
display_profile[avg_cols] = display_profile[avg_cols].round(2)
st.dataframe(display_profile, use_container_width=True, hide_index=True)

st.subheader("6 · Business interpretation")
st.write("Treat segment labels as data-driven groups, not fixed customer identities. Compare each segment's averages with the overall average before assigning a business label.")
overall = working[features].mean()
for _, row in profile.iterrows():
    cl = int(row["Cluster"])
    group = working[working["Cluster"] == cl]
    strongest = []
    for feature in features:
        avg = group[feature].mean()
        base_avg = overall[feature]
        if base_avg != 0 and avg >= base_avg * 1.15:
            strongest.append(f"higher {feature}")
        elif base_avg != 0 and avg <= base_avg * .85:
            strongest.append(f"lower {feature}")
    summary = ", ".join(strongest[:3]) if strongest else "near the overall average on selected features"
    st.markdown(f"**{row['Segment']}** — {int(row['Customers'])} customers ({row['Customer Share (%)']}%): {summary}.")
st.info("Possible actions: tailor offers to high-value/high-frequency groups, test relevant bundles for moderate-spend groups, and use low-cost re-engagement campaigns for inactive or low-frequency groups. Validate recommendations against actual business outcomes.")

st.subheader("7 · Export")
csv_bytes = working.to_csv(index=False).encode("utf-8")
profile_bytes = profile.to_csv(index=False).encode("utf-8")
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download customers with segments (CSV)", data=csv_bytes, file_name="customers_with_segments.csv", mime="text/csv", use_container_width=True)
with col2:
    st.download_button("Download segment profile (CSV)", data=profile_bytes, file_name="segment_profile.csv", mime="text/csv", use_container_width=True)

st.caption("Demo data is synthetic and included for testing. For internship submission, use your approved dataset and review the generated insights before presenting them.")
