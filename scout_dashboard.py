# 🧠 FULL LDA PIPELINE FOR ATTACKING FULL-BACKS

# STEP 1: Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# STEP 2: Load data
file_path = "FB_ATT.xlsx"  # Make sure this file is in your Colab/working dir
df = pd.read_excel(file_path)

# STEP 3: Filter full-backs
fullback_keywords = ["RB", "RWB", "LB", "LWB"]
df_fullbacks = df[df["Position"].astype(str).str.contains('|'.join(fullback_keywords))].copy()

# STEP 4: Create labels - top 25% AFB score = attacking
threshold = df_fullbacks["Fullback Attacking AFB"].quantile(0.75)
df_fullbacks["Attacking Label"] = (df_fullbacks["Fullback Attacking AFB"] >= threshold).astype(int)

# STEP 5: Select features
features = [
    "Accurate progressive passes, %",
    "Deep completions per 90",
    "Crosses per 90",
    "Accurate through passes, %",
    "Key passes per 90",
    "Accurate forward passes, %",
    "Successful dribbles, %",
    "Assists per 90"
]
X = df_fullbacks[features].fillna(0)
y = df_fullbacks["Attacking Label"]

# STEP 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# STEP 7: Standardize & Train LDA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(X_train_scaled, y_train)

# STEP 8: Evaluate model
y_pred = lda.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 9: Check weights (feature importance)
weights = pd.Series(lda.coef_[0], index=features).sort_values(key=abs, ascending=False)
print("\nFeature Importances (LDA Weights):\n")
print(weights)

# STEP 10: Transform all data into LDA space
X_all_scaled = scaler.transform(X)
df_fullbacks["LDA1"] = lda.transform(X_all_scaled)

# STEP 11: Plot 1D scatter
import plotly.express as px

fig = px.scatter(
    df_fullbacks,
    x="LDA1",
    y=[0]*len(df_fullbacks),
    color=df_fullbacks["Attacking Label"].map({1: "Attacking", 0: "Not Attacking"}),
    hover_name="Player",
    title="LDA Projection of Full-Backs",
    width=900,
    height=400
)
fig.update_yaxes(visible=False)
fig.update_traces(marker=dict(size=10, opacity=0.8))
fig.show()


# STEP 12: Save result to CSV for dashboard
output_file = "attacking_fullbacks.csv"
df_fullbacks.to_csv(output_file, index=False)
print(f"\nSaved LDA output to {output_file}")