import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("data.csv")

# Select features and target
X = df[['bedrooms', 'bathrooms', 'sqft_living',
        'sqft_lot', 'floors', 'waterfront',
        'view', 'condition', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated']]

y =df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model,("trained_model.pkl"))

print("Model saved successfully")
