# python -m streamlit run main.py (for run streamlit app in browser)
# pip install scikit-learn
# pip install streamlit
# pip install joblib


import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


x_cut = joblib.load("x_cut.pkl")
x_color = joblib.load("x_color.pkl")
x_clarity = joblib.load("x_clarity.pkl")


st.title("diamond price prediction")

cut = st.selectbox(
    "Cut",
    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
)

color = st.selectbox(
    "color",
    ["E","I","J","D","H","F"]
)

clarity = st.selectbox(
    "clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

unnamed = st.number_input("Unnamed: 0", min_value=0, value=0)
carat = st.number_input("carat", min_value=0)
depth = st.number_input("Depth of diamond", min_value=0)
table = st.number_input("table", min_value=0)
length= st.number_input("Length", min_value=0)
height = st.number_input("height", min_value=0)
width = st.number_input("Width", min_value=0.0)
if st.button("Predict"):

    cut_enc = x_cut.transform([[cut]])[0]
    color_enc = x_color.transform([[color]])[0]
    clarity_enc = x_clarity.transform([[clarity]])[0]

    input_data = pd.DataFrame(
        [[unnamed, cut_enc, color_enc, clarity_enc, carat, depth, table, length, height, width]],
        columns=[
            "Unnamed: 0",
            "cut",
            "color",
            "clarity",
            "carat",
            "depth",
            "table",
            "length",
            "height",
            "width",
        ],
    )

    # If the scaler was fitted on a DataFrame, ensure column order matches
    if hasattr(scaler, "feature_names_in_"):
        try:
            input_data = input_data.loc[:, scaler.feature_names_in_]
        except Exception:
            # fallback: convert to numpy array preserving current order
            input_data = input_data.values
    else:
        input_data = input_data.values

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"💎 Predicted Diamond Price: ₹{prediction[0]:,.2f}")

