import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("trained_model.pkl")

st.title("House Price Prediction")

bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
sqft_living = st.number_input("Sqft Living")
sqft_lot = st.number_input("Sqft Lot")
floors = st.number_input("Floors")
waterfront = st.number_input("Waterfront")
view = st.number_input("View")
condition = st.number_input("Condition")
sqft_above = st.number_input("Sqft Above")
sqft_basement = st.number_input("Sqft Basement")
yr_built = st.number_input("Year Built")
yr_renovated = st.number_input("Year Renovated")

if st.button("Predict Price"):
    features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot,
                          floors, waterfront, view, condition,
                          sqft_above, sqft_basement,
                          yr_built, yr_renovated]])

    prediction = model.predict(features)

    st.write(prediction)
