# Customer Segmentation Studio

A complete, interactive customer analytics project using Python, Pandas, scikit-learn, K-Means, and Streamlit.

## Features
- Dataset upload (CSV)
- Data overview, missing-value and duplicate checks
- Numeric feature selection and median imputation
- StandardScaler preprocessing
- K-Means clustering with reproducible random seed
- Elbow-method inertia chart and silhouette-score comparison
- Interactive cluster scatter plot, segment size chart, and customer mix
- Segment profile table and cautious business interpretation
- Export customer-level segment labels and segment profiles to CSV

## Dataset
`data/customer_data_demo.csv` is a **synthetic demonstration dataset** created to test the app. It is not real company/customer data and should not be described as real findings. For your internship submission, replace it with a dataset approved by your internship supervisor or use the CSV uploader.

Expected useful fields include numeric demographic or behavioral features such as Age, Annual Income, Spending Score, Annual Purchases, Average Order Value, purchase frequency, monetary value, and recency. Keep a customer identifier for traceability, but do not include ID columns as clustering features.

## Run on Windows (PowerShell)
1. Extract the ZIP.
2. Open PowerShell in the `Customer_Segmentation_Project` folder.
3. Create a virtual environment (recommended):
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install libraries:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```powershell
   streamlit run app.py
   ```
6. Open the local URL shown in the terminal (usually `http://localhost:8501`).

If PowerShell blocks activation, run the install and app commands using your installed Python directly:
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Methodology
1. Remove exact duplicate rows.
2. Convert selected numeric features and fill missing selected feature values with the median.
3. Standardize selected features because K-Means uses distance.
4. Compare candidate K values using inertia (elbow method) and silhouette score.
5. Fit K-Means using the selected K and assign each record to a segment.
6. Profile segments and interpret differences against the overall average.

## Important interpretation notes
- Cluster numbers are arbitrary identifiers, not rankings.
- A silhouette score closer to 1 generally indicates better separation; values near 0 indicate overlap, and negative values may indicate poor assignments.
- The elbow method is a heuristic, not an automatic proof of the best K.
- The current app clusters selected numeric columns. Categorical variables such as Gender or Education are not automatically encoded into the model; they can be used for profiling separately.
- Business actions are hypotheses. Validate them using campaign results and business context.

## Files
- `app.py`: interactive Streamlit application
- `data/customer_data_demo.csv`: synthetic demo data
- `requirements.txt`: Python dependencies
- `PROJECT_REPORT.md`: project write-up suitable for editing
- `outputs/`: reserved for exported outputs

## Tutorial reference
The project follows the general workflow described in the supplied tutorial: data preparation, K-Means clustering, and interactive visualization. This implementation is independently structured and is not a pixel-for-pixel copy of the video.


## Tutorial-style notebook (important)
`Analysis_Model.ipynb` is the main notebook aligned to the screenshots you supplied. It uses the tutorial dataset schema (`Year_Birth`, `Education`, `Marital_Status`, `Income`, `MntWines`, `MntFruits`, `MntMeatProducts`, purchase counts, campaign fields, etc.) and includes:
- imports and dataset loading
- inspection and data cleaning
- engineered `Age`, `Total_Spending`, and `Total_Children`
- income/spending/age distributions
- correlation heatmap and income pivot table
- feature scaling, K-Means, elbow method, silhouette analysis
- segment visualizations, profiling, business interpretation, and CSV exports

### Download the correct dataset
Use the dataset link supplied for this internship:
https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering

Download `customer_segmentation.csv` from Kaggle and place it beside `Analysis_Model.ipynb`. The notebook expects that exact filename. The screenshots show a 2,240-row marketing customer dataset with columns such as `Year_Birth`, `MntWines`, `MntFruits`, `Income`, `NumWebPurchases`, and `NumStorePurchases`. If Kaggle downloads the file with a different filename, rename it to `customer_segmentation.csv`.

### Run the notebook
```powershell
python -m pip install -r requirements.txt
python -m jupyter notebook
```
Open `Analysis_Model.ipynb` in Jupyter, or open it in VS Code after installing the Jupyter extension, select your Python environment, then choose **Run All**.

The earlier `app.py` interactive dashboard remains included as an additional deliverable. The notebook is the closest match to the tutorial screenshots you provided.
