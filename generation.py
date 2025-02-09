import numpy as np
import pandas as pd

# 📌 Number of data points
num_samples = 50000  # Adjust this number if needed

# 📌 Generate synthetic daily returns (Gaussian distribution with slight drift)
np.random.seed(42)
returns = np.random.normal(loc=0.0005, scale=0.02, size=num_samples)  # Mean = 0.05%, Volatility = 2%

# 📌 Create a DataFrame
df = pd.DataFrame(returns, columns=["Return"])

# 📌 Save to CSV
df.to_csv("synthetic_returns.csv", index=False, header=False)

print(f"✅ Generated {num_samples} synthetic return data points and saved to 'synthetic_returns.csv'.")
