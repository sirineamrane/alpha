# 📌 Import des librairies
import pandas as pd
import numpy as np
import yfinance as yf
import ta  # Indicateurs techniques

# ✅ 1️⃣ Récupération des Données Historiques avec Yahoo Finance
def get_yahoo_data(symbol, start="2010-01-01", end="2024-02-09"):
    print(f"📡 Récupération des données historiques pour {symbol}...")
    df = yf.download(symbol, start=start, end=end)

    # Vérification des colonnes essentielles
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_columns):
        print(f"⚠️ Attention : Certaines colonnes sont manquantes pour {symbol}")
        return None

    # Utilisation du prix ajusté si disponible
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    df["Return"] = df["Close"].pct_change()  # Rendements quotidiens
    df.dropna(inplace=True)

    # Sauvegarde locale avec un nom sans caractères spéciaux
    filename = f"{symbol.replace('=', '_')}_historique.csv"
    df.to_csv(filename)
    print(f"✅ Données sauvegardées sous {filename}")

    return df

# Téléchargement des données
spy = get_yahoo_data("SPY")  # S&P 500
wti = get_yahoo_data("CL=F")  # Pétrole WTI
brent = get_yahoo_data("BZ=F")  # Pétrole Brent

# ✅ 2️⃣ Ajout des Indicateurs Techniques
def add_ta_indicators(df):
    if df is None:
        return None

    df = df.copy()

    # Vérification des colonnes nécessaires
    required_columns = ["High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_columns):
        print("❌ Erreur : certaines colonnes requises sont absentes.")
        return df

    # 🔥 Conversion et remplissage des valeurs manquantes
    df[required_columns] = df[required_columns].astype(float).ffill().bfill()

    # Vérification des NaN après remplissage
    if df[["High", "Low", "Close"]].isnull().values.any():
        print("⚠️ NaN détecté après ffill/bfill. Suppression...")
        df.dropna(subset=["High", "Low", "Close"], inplace=True)

    # 📌 Indicateurs de tendance
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    # 📌 Bandes de Bollinger
    df["Bollinger_Up"] = df["Close"].rolling(window=20).mean() + (df["Close"].rolling(window=20).std() * 2)
    df["Bollinger_Down"] = df["Close"].rolling(window=20).mean() - (df["Close"].rolling(window=20).std() * 2)

    # 📌 RSI (Correction définitive)
    df["RSI"] = pd.Series(ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi().values.ravel(), index=df.index)

    # 📌 MACD
    macd = ta.trend.MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # 📌 ATR
    df["ATR"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()

    # 📌 VWAP
    df["VWAP"] = ta.volume.VolumeWeightedAveragePrice(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
    ).volume_weighted_average_price()

    # 📌 OBV (On Balance Volume)
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"]
    ).on_balance_volume()

    # 📌 Momentum Williams %R
    df["Momentum"] = ta.momentum.WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], lbp=14
    ).williams_r()

    # 📌 Volatilité sur 10 jours
    df["Volatility_10d"] = df["Close"].pct_change().rolling(10).std()

    df.dropna(inplace=True)
    return df
