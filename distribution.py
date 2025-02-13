import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import kurtosis, skew, boxcox, shapiro, ks_2samp, genpareto, levene, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression

# === 1. Chargement et Audit Initial ===
print("\n🔍 Étape 1 : Chargement et audit des données\n")
df = pd.read_csv("SPY_returns.csv")
print(df.head())
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())

# Visualisation des valeurs manquantes
msno.heatmap(df)
plt.show()

# Boxplot pour détecter les outliers
df.boxplot(figsize=(10,5))
plt.xticks(rotation=90)
plt.show()

# Distribution des variables numériques
for col in df.select_dtypes(include=[np.number]).columns:
    print(f"{col}: skewness={skew(df[col])}, kurtosis={kurtosis(df[col])}")
    sns.kdeplot(df[col], shade=True, label=col)
plt.legend()
plt.show()

# === 2. Détection des Outliers ===
print("\n🚨 Étape 2 : Détection des Outliers\n")
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Z-Score
df["outlier_zscore"] = (np.abs(df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())) > 3).any(axis=1)

# IQR (Interquartile Range)
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df["outlier_iqr"] = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df["outlier_iso"] = iso_forest.fit_predict(df[numeric_cols]) == -1

# Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20)
df["outlier_lof"] = lof.fit_predict(df[numeric_cols]) == -1

# === 3. Transformation des Outliers ===
print("\n🔄 Étape 3 : Transformation des Outliers\n")
qt = QuantileTransformer(output_distribution="normal", random_state=42)
df_quantile = pd.DataFrame(qt.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Robust Scaling
scaler = RobustScaler()
df_robust_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# === 4. Tests Statistiques ===
print("\n📊 Étape 4 : Tests Statistiques\n")
for col in numeric_cols:
    stat, p = shapiro(df[col].dropna())
    print(f"Shapiro-Wilk Test ({col}): p-value={p}")

    stat, p = ks_2samp(df[col].dropna(), np.random.normal(0, 1, len(df[col].dropna())))
    print(f"Kolmogorov-Smirnov Test ({col}): p-value={p}")

    shape, loc, scale = genpareto.fit(df[col])
    print(f"Pareto Shape ({col}): {shape}")

# === 5. Gestion du Drift ===
print("\n⚠️ Étape 5 : Détection du Drift\n")
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

def compare_distributions(train, test, cols):
    results = []
    for col in cols:
        ks_p = ks_2samp(train[col].dropna(), test[col].dropna()).pvalue
        wass_dist = wasserstein_distance(train[col].dropna(), test[col].dropna())
        js_div = jensenshannon(train[col].dropna(), test[col].dropna())
        results.append({"Feature": col, "KS-Test p-value": ks_p, "Wasserstein Distance": wass_dist, "JS Divergence": js_div})
    return pd.DataFrame(results)

drift_results = compare_distributions(train_data, test_data, numeric_cols)
print(drift_results)

def detect_drift(drift_results):
    drift_results["Drift Alert"] = (drift_results["KS-Test p-value"] < 0.05) | (drift_results["Wasserstein Distance"] > 0.3) | (drift_results["JS Divergence"] > 0.2)
    return drift_results

drift_results = detect_drift(drift_results)
print(drift_results)

# === 6. Correction du Drift ===
print("\n✅ Étape 6 : Correction du Drift\n")
train_data["sample_weight"] = compute_sample_weight(class_weight="balanced", y=train_data[numeric_cols[0]])
model = LogisticRegression()
model.fit(train_data[numeric_cols], train_data[numeric_cols[0]], sample_weight=train_data["sample_weight"])
