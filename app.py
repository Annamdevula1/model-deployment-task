import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("trained_model.pkl")

st.title("House Price Prediction")

bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0.0)
sqft_living = st.number_input("Sqft Living", min_value=0)
sqft_lot = st.number_input("Sqft Lot", min_value=0)
floors = st.number_input("Floors", min_value=0)
waterfront = st.number_input("Waterfront (0 = No, 1 = Yes)", min_value=0, max_value=1)
view = st.number_input("View", min_value=0)
condition = st.number_input("Condition", min_value=0)
sqft_above = st.number_input("Sqft Above", min_value=0)
sqft_basement = st.number_input("Sqft Basement", min_value=0)
yr_built = st.number_input("Year Built", min_value=1800, max_value=2100)
yr_renovated = st.number_input("Year Renovated", min_value=0)

if st.button("Predict Price"):
    features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot,
                          floors, waterfront, view, condition,
                          sqft_above, sqft_basement,
                          yr_built, yr_renovated]])

    prediction = model.predict(features)

    st.success(f"Predicted House Price: {prediction[0]:.2f}")
