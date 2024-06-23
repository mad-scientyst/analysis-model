import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Load your dataset
data = pd.read_csv('datasets/A_data.csv')  # Replace with your actual data file
# Handling missing values
data.dropna(inplace=True)

# Normalize data
data['price'] = (data['price'] - data['price'].mean()) / data['price'].std()
# Creating lagged features
data['price_lag1'] = data['price'].shift(1)
data['price_lag2'] = data['price'].shift(2)
data.dropna(inplace=True)

# Target variable
data['target'] = np.where(data['price'].shift(-1) > data['price'], 1, 0)  # 1 for price up, 0 for price down
# Define features and target
features = ['price', 'price_lag1', 'price_lag2']
X = data[features]
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
# Simplified backtesting logic
data['predicted_signal'] = model.predict(X)
data['strategy_returns'] = data['target'] * data['predicted_signal'].shift(1)
cumulative_returns = (1 + data['strategy_returns']).cumprod()

# Evaluate backtest results
print("Cumulative Returns:", cumulative_returns.iloc[-1])
