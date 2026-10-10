# Customer Segmentation Using K-Means Clustering

## 1. Project Overview
Customer segmentation is the process of dividing a customer base into groups with similar demographic characteristics or purchasing behavior. The purpose is to support more relevant marketing, product recommendations, customer engagement, and retention strategies.

## 2. Problem Statement
A business may treat every customer in the same way despite differences in spending levels, purchase frequency, and customer profiles. This project provides an analytical workflow to identify groups with similar numeric characteristics and summarize their distinguishing patterns.

## 3. Objectives
- Inspect and prepare customer data for analysis.
- Apply unsupervised machine learning using K-Means.
- Use the elbow method and silhouette score to compare candidate cluster counts.
- Visualize clusters and summarize each segment.
- Export customer-level segment assignments and segment-level profiles.

## 4. Tools and Technologies
Python, Pandas, NumPy, scikit-learn, Plotly, and Streamlit.

## 5. Methodology
### Data inspection and cleaning
The application checks dataset size, missing cells, and duplicate rows. Exact duplicate records are removed. Selected numeric features are converted to numeric values, rows with no usable selected features are excluded, and remaining missing values in selected features are filled with their median.

### Feature selection and scaling
The user chooses at least two numeric features. Identifier fields are excluded from the default feature list because customer IDs are labels rather than meaningful measures of similarity. `StandardScaler` standardizes selected features so measurements with larger numeric ranges do not automatically dominate the distance calculation.

### K-Means clustering
K-Means partitions observations into K groups by assigning each observation to its nearest cluster centroid and repeatedly updating centroids. The selected K is configurable in the dashboard.

### Model diagnostics
The elbow chart plots inertia across candidate K values. The silhouette score provides an additional view of separation and cohesion. Neither metric should be used alone: segment interpretability and business context are also important.

### Profiling and visualization
The dashboard compares average feature values, customer counts, and segment shares. A scatter plot allows users to inspect relationships between two selected features.

## 6. Business Use Cases
- Personalize campaigns according to spending and purchase patterns.
- Test loyalty benefits for valuable and frequent customers.
- Design product bundles for groups with similar preferences.
- Develop re-engagement experiments for low-frequency or inactive customers.
- Compare campaign performance across segments over time.

These are possible strategies, not guaranteed outcomes. They should be validated with actual campaign and revenue data.

## 7. Deliverables
The application can export:
1. A customer-level CSV containing the assigned cluster and segment label.
2. A segment-profile CSV containing customer counts, average feature values, and customer share.

## 8. Limitations
- The included dataset is synthetic demonstration data, not actual customer records.
- K-Means is sensitive to feature selection, scaling, outliers, and K.
- Numeric features are used directly; categorical fields are not automatically encoded for clustering.
- Cluster labels are arbitrary and must be interpreted by reviewing profile statistics.
- The analysis is descriptive and does not establish that a marketing action causes higher revenue.

## 9. Conclusion
This project demonstrates an end-to-end customer segmentation workflow, from data preparation and scaling through clustering, evaluation, visualization, and export. It provides a reusable starting point for customer analytics. Meaningful final conclusions require running it on a real, approved dataset and documenting the resulting profiles.

## 10. Reference
Tutorial provided with the internship task: https://youtu.be/-LGwdrajMZ0


## Tutorial-matching notebook
The project now includes `Analysis_Model.ipynb`, which follows the notebook-based workflow visible in the user's tutorial screenshots and is designed for the linked Kaggle marketing customer dataset. It adds total spending from product-category spend columns, derives age and children counts where possible, explores income and spending distributions, compares income by education/marital status, evaluates K-Means cluster counts, profiles segments, and exports CSV results. Download the Kaggle dataset separately and save it as `customer_segmentation.csv` next to the notebook before running all cells.
