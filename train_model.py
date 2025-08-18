import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Generate synthetic housing data
np.random.seed(42)
n_samples = 1000

# Generate features
sqft = np.random.normal(2000, 500, n_samples)
sqft = np.clip(sqft, 800, 5000)  # Reasonable range

bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                           p=[0.15, 0.15, 0.3, 0.15, 0.15, 0.05, 0.05])
age = np.random.randint(0, 50, n_samples)

# Create price with realistic relationships
base_price = (sqft * 100) + (bedrooms * 10000) + (bathrooms * 8000) - (age * 1000)
noise = np.random.normal(0, 15000, n_samples)
price = base_price + noise
price = np.clip(price, 50000, 800000)  # Reasonable price range

# Create DataFrame
data = pd.DataFrame({
    'sqft': sqft,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'price': price
})

print("Dataset Info:")
print(data.describe())

# Split the data
X = data[['sqft', 'bedrooms', 'bathrooms', 'age']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"MSE: {mse:,.2f}")
print(f"RMSE: {np.sqrt(mse):,.2f}")
print(f"R² Score: {r2:.4f}")

# Save the model
joblib.dump(model, 'model.pkl')
print("\nModel saved as 'model.pkl'")

# Save feature names for reference
feature_names = ['sqft', 'bedrooms', 'bathrooms', 'age']
joblib.dump(feature_names, 'feature_names.pkl')