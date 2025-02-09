import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data without column headers
file_path = "synthetic_returns.csv"
df = pd.read_csv(file_path, header=None, names=["Return"])

# Generate a dummy date column
df["Date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

# Extract returns
returns = df["Return"].values

# Statistical tests for white noise detection
adf_pvalue = adfuller(returns)[1]
ljung_pvalue = acorr_ljungbox(returns, lags=[10], return_df=True)['lb_pvalue'].values[0]
jb_pvalue = jarque_bera(returns)[1]

# Print test results
print("\n White Noise Test Results:")
print(f"ADF Test p-value: {adf_pvalue}")
print(f"Ljung-Box Test p-value: {ljung_pvalue}")
print(f"Jarque-Bera Test p-value: {jb_pvalue}")

# Check if the data deviates from white noise
if ljung_pvalue < 0.05 or adf_pvalue > 0.05:
    print("\n Deviation detected! Applying advanced ML models.")

    # Feature engineering
    df["Lag1"] = df["Return"].shift(1)
    df["Lag2"] = df["Return"].shift(2)
    df["SMA_5"] = df["Return"].rolling(window=5).mean()
    df["Volatility"] = df["Return"].rolling(window=5).std()
    df.dropna(inplace=True)

    # Create target variable (1 if next return is positive, 0 otherwise)
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)

    # Split data
    X = df[["Return", "Lag1", "Lag2", "SMA_5", "Volatility"]]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define ML models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale")
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

    # LSTM for deep learning
    X_train_LSTM = np.expand_dims(X_train.values, axis=1)
    X_test_LSTM = np.expand_dims(X_test.values, axis=1)

    lstm_model = Sequential([
        LSTM(50, activation="relu", input_shape=(1, X_train.shape[1])),
        Dense(1, activation="sigmoid")
    ])
    lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    lstm_model.fit(X_train_LSTM, y_train, epochs=10, batch_size=16, verbose=0)
    lstm_accuracy = lstm_model.evaluate(X_test_LSTM, y_test, verbose=0)[1]
    results["LSTM"] = lstm_accuracy

    # Display results
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
    print("\n ML Model Accuracy:\n", results_df)

    # Plot predictions
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"].iloc[-len(y_test):], y_test, label="Actual", color="blue")
    plt.plot(df["Date"].iloc[-len(y_test):], y_pred, label="Predictions (Random Forest)", linestyle="dashed", color="red")
    plt.title("Predictions vs Actual")
    plt.legend()
    plt.show()

else:
    print("\n No exploitable signal: The series appears to be pure white noise.")
