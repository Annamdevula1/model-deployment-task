import streamlit as st
import joblib
import numpy as np

model = joblib.load('trained_model.pkl')

st.title("House Price Prediction")

bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
floors = st.number_input("Floors")
sqft_living = st.number_input("Sqft Living")


if st.button("Predict Price"):
    prediction = model.predict(
        np.array([[bedrooms, bathrooms, floors, sqft_living, sqft_above]])
    )

    st.success(f"Predicted House Price: {prediction[0]:.2f}")
