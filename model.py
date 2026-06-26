import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("data.csv")

# Select features and target
X = df[['bedrooms', 'bathrooms', 'floors', 'sqft_living']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model saved successfully")
