import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("trained_model.pkl")

# Title
st.title("🏠 House Price Prediction")

# Display House Image
st.image("HOUSE.jpeg", use_container_width=True)

# Description
st.write("Fill in the house details below and click 'Predict House Price'.")

# User Inputs
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Bathrooms", value=2.0)
sqft_living = st.number_input("Sqft Living", value=1800)
sqft_lot = st.number_input("Sqft Lot", value=5000)
floors = st.number_input("Floors", value=2)
waterfront = st.number_input("Waterfront (0 = No, 1 = Yes)", value=0)
view = st.number_input("View", value=2)
condition = st.number_input("Condition", value=3)
sqft_above = st.number_input("Sqft Above", value=1400)
sqft_basement = st.number_input("Sqft Basement", value=400)
yr_built = st.number_input("Year Built", value=2008)
yr_renovated = st.number_input("Year Renovated", value=0)

# Prediction
if st.button("Predict House Price"):
    features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot,
                          floors, waterfront, view, condition,
                          sqft_above, sqft_basement,
                          yr_built, yr_renovated]])

    prediction = model.predict(features)

    st.success(f"🏡 Predicted House Price: ₹ {prediction[0]:,.2f}")
