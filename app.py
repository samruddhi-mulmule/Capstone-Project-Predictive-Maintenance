import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Machine Failure Predictor", page_icon="🛠️")


# ---------- Load the saved model ----------
@st.cache_resource
def load_model():
    with open("predictive_maintenance_model.pkl", "rb") as f:
        return pickle.load(f)


# ---------- Same steps as in the notebook (steps 4, 5 and 7) ----------
def prepare_input(raw, feature_columns, log_columns):
    data = raw.copy()

    # step 4: feature engineering
    data["Temp_diff"] = data["Process_temp"] - data["Air_temp"]
    data["Power"] = data["Torque"] * data["Rot_speed"] * 2 * np.pi / 60
    data["Torque_wear"] = data["Torque"] * data["Tool_wear"]

    # step 5: log transform
    for col in log_columns:
        data[col] = np.log1p(data[col])

    # step 7: one-hot encoding of Type
    for col in feature_columns:
        if col.startswith("Type_"):
            data[col] = (data["Type"] == col.split("_")[1]).astype(int)

    return data[feature_columns]


# ---------- Page ----------
st.title("🛠️ Machine Failure Predictor")
st.write(
    "Enter the sensor readings of a machine and the model will predict "
    "whether it is likely to fail."
)

try:
    info = load_model()
except FileNotFoundError:
    st.error("Model file 'predictive_maintenance_model.pkl' not found. "
             "Run the notebook first and keep the file in the same folder as app.py.")
    st.stop()

st.subheader("Machine readings")
col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox("Product quality type", ["L", "M", "H"],
                                help="L = Low, M = Medium, H = High quality variant")
    air_temp = st.number_input("Air temperature [K]", min_value=295.0, max_value=305.0,
                               value=300.0, step=0.1, format="%.1f",
                               help="Allowed range: 295.0 to 305.0")
    process_temp = st.number_input("Process temperature [K]", min_value=305.0, max_value=314.0,
                                   value=310.0, step=0.1, format="%.1f",
                                   help="Allowed range: 305.0 to 314.0")

with col2:
    rot_speed = st.number_input("Rotational speed [rpm]", min_value=1100, max_value=3000,
                                value=1500, step=10,
                                help="Allowed range: 1100 to 3000")
    torque = st.number_input("Torque [Nm]", min_value=3.0, max_value=80.0,
                             value=40.0, step=0.1, format="%.1f",
                             help="Allowed range: 3.0 to 80.0")
    tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=260,
                                value=100, step=1,
                                help="Allowed range: 0 to 260")

if process_temp <= air_temp:
    st.warning("Process temperature is normally higher than air temperature "
               "in the training data, so the result may be less reliable.")

if st.button("Predict", type="primary"):
    raw = pd.DataFrame([{
        "Type": machine_type,
        "Air_temp": air_temp,
        "Process_temp": process_temp,
        "Rot_speed": rot_speed,
        "Torque": torque,
        "Tool_wear": tool_wear,
    }])

    x = prepare_input(raw, info["feature_columns"], info["log_columns"])
    model = info["model"]
    failure_prob = model.predict_proba(x)[0, 1]
    prediction = model.predict(x)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error("⚠️ Machine is likely to FAIL")
    else:
        st.success("✅ Machine is likely to keep working (no failure)")

    st.metric("Model failure score", f"{failure_prob * 100:.1f}%")
    st.progress(float(failure_prob))
    st.caption(
        "The score is the model's confidence, not an exact real-world probability. "
        "A score above 50% is predicted as a failure."
    )

    with st.expander("Features created from your input"):
        st.dataframe(x.round(3))