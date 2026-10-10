"""Command-line baseline for customer segmentation.
Run: python analyze_customers.py [optional/path/to/customer.csv]
"""
from pathlib import Path
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

root = Path(__file__).parent
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "customer_data_demo.csv"
df = pd.read_csv(source)
numeric = df.select_dtypes(include="number").columns.tolist()
features = [c for c in numeric if "id" not in c.lower() and not c.lower().endswith("_id")]
if len(features) < 2:
    raise ValueError("Need at least two numeric non-ID columns.")
clean = df.drop_duplicates().copy()
for col in features:
    clean[col] = pd.to_numeric(clean[col], errors="coerce")
clean = clean.dropna(subset=features, how="all")
for col in features:
    clean[col] = clean[col].fillna(clean[col].median())
scaled = StandardScaler().fit_transform(clean[features])
model = KMeans(n_clusters=5, random_state=42, n_init=10)
clean["Cluster"] = model.fit_predict(scaled)
clean["Segment"] = clean["Cluster"].map(lambda x: f"Segment {x+1}")
out = root / "outputs"
out.mkdir(exist_ok=True)
clean.to_csv(out / "customers_with_segments.csv", index=False)
profile = clean.groupby(["Cluster", "Segment"], as_index=False).agg(
    Customers=("Cluster", "size"),
    **{f"Average {c}": (c, "mean") for c in features}
)
profile["Customer Share (%)"] = (profile["Customers"] / len(clean) * 100).round(1)
profile.to_csv(out / "segment_profile.csv", index=False)
print(f"Rows analyzed: {len(clean)}")
print(f"Features: {features}")
print(f"Silhouette score: {silhouette_score(scaled, clean['Cluster']):.3f}")
print(profile.to_string(index=False))
print(f"Exports saved in: {out}")
