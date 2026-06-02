import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Life Expectancy Prediction",
    layout="wide"
)

st.title("Life Expectancy Prediction Dashboard")

linear_model = joblib.load("linear.pkl")
poly_model = joblib.load("poly.pkl")
ridge_model = joblib.load("ridge.pkl")

st.sidebar.header("Input Features")

adult_mortality = st.sidebar.slider(
    "Adult Mortality",
    1,
    800,
    200
)

bmi = st.sidebar.slider(
    "BMI",
    1.0,
    90.0,
    25.0
)

gdp = st.sidebar.slider(
    "GDP",
    0.0,
    120000.0,
    10000.0
)

alcohol = st.sidebar.slider(
    "Alcohol",
    0.0,
    20.0,
    5.0
)

input_df = pd.DataFrame({
    "Adult mortality":[adult_mortality],
    "BMI":[bmi],
    "GDP":[gdp],
    "Alcohol":[alcohol]
})


model_name = st.selectbox(
    "Choose Model",
    ["Linear","Poly","Ridge"]
)

if model_name == "Linear":
    model = linear_model
elif model_name == "Poly":
    model = poly_model
else:
    model = ridge_model

prediction = model.predict(input_df)[0]

st.markdown("## Predicted Life Expectancy")
st.success(f"{prediction:.2f} years")

st.header("Model Performance Comparison")

performance_df = pd.read_csv("performance.csv")

st.dataframe(
    performance_df,
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(8,5))

ax.bar(
    performance_df["Model"],
    performance_df["Test_R2"]
)

ax.set_title("Test R² Comparison")
ax.set_ylabel("R² Score")

st.pyplot(fig)
