import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import date
import os

# Load model and scalers
model = load_model("lstm_co2_predictor.h5", compile=False)
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

DATA_FILE = "user_data.csv"
TIME_STEPS = 14

# Emission factors
CO2_PER_LITER_DIESEL = 2.68      # kg CO₂ per liter diesel
CO2_PER_TONNE_COAL = 1.9         # kg CO₂ per tonne coal mined (example value)

# Initialize CSV if it doesn't exist
if not os.path.exists(DATA_FILE):
    df_init = pd.DataFrame(columns=["Date", "Fuel_Used_Liters", "Coal_Mined_Tonnes"])
    df_init.to_csv(DATA_FILE, index=False)

# Load user-entered data
df = pd.read_csv(DATA_FILE)

# App Title
st.title("Minelytics : The Intelligent Emission Analytics")

# 1. Daily Input Form
with st.form("daily_input"):
    st.subheader("Enter Today’s Data")
    fuel = st.number_input("Fuel Used (liters)", min_value=500.0, max_value=50000.0, step=100.0)
    coal = st.number_input("Coal Mined (tonnes)", min_value=1000.0, max_value=100000.0, step=500.0)
    submitted = st.form_submit_button("Submit")

if submitted:
    today = date.today().strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{ "Date": today, "Fuel_Used_Liters": fuel, "Coal_Mined_Tonnes": coal }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    st.success(f"✅ Data for {today} saved!")

# 2. Edit/Delete Last 30 Rows
st.subheader("📝 Edit or Delete Records")
editable_rows = df.tail(30).copy()
edited = st.data_editor(
    editable_rows,
    use_container_width=True,
    num_rows="dynamic",
    key="editor"
)

if st.button("💾 Save Changes"):
    df = df.drop(editable_rows.index)  # safer drop
    df = pd.concat([df, edited], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    st.success("✅ Edits and deletions saved successfully!")

# 3. Prediction Section
if len(df) >= TIME_STEPS:
    st.subheader("📈 Predicted CO₂ Emission for Tomorrow")

    last_14 = df[["Fuel_Used_Liters", "Coal_Mined_Tonnes"]].tail(TIME_STEPS).values
    X_input = scaler_x.transform(last_14).reshape(1, TIME_STEPS, 2)
    y_pred_scaled = model.predict(X_input)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]

    st.metric(label="CO₂ Emitted (kg)", value=f"{y_pred:,.2f}")

# 4. Trend & Summary Analysis
if len(df) >= TIME_STEPS:
    st.subheader("📊 CO₂ Emission Analysis & Trend")

    actual_vals = []
    predicted_vals = []

    for i in range(TIME_STEPS, len(df)):
        # Prepare input sequence for prediction
        seq = df[["Fuel_Used_Liters", "Coal_Mined_Tonnes"]].iloc[i - TIME_STEPS:i].values
        seq_scaled = scaler_x.transform(seq).reshape(1, TIME_STEPS, 2)
        pred_scaled = model.predict(seq_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        predicted_vals.append(pred)

        # Compute actual CO₂ (fuel + coal)
        fuel_co2 = df["Fuel_Used_Liters"].iloc[i] * CO2_PER_LITER_DIESEL
        coal_co2 = df["Coal_Mined_Tonnes"].iloc[i] * CO2_PER_TONNE_COAL
        actual = fuel_co2 + coal_co2
        actual_vals.append(actual)

    # Summary metrics
    st.markdown("### 📈 Last 30 Days CO₂ Stats")
    recent_actual = actual_vals[-30:] if len(actual_vals) >= 30 else actual_vals
    recent_predicted = predicted_vals[-30:] if len(predicted_vals) >= 30 else predicted_vals

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Actual (kg)", f"{(np.mean(recent_actual)/2):,.2f}")
    col2.metric("Avg Predicted (kg)", f"{np.mean(recent_predicted):,.2f}")
    col3.metric("Total Actual", f"{(np.sum(recent_actual)/2):,.2f} kg")

    # # Plot trend
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 4))
    # ax.plot([actual_vals[-30:]], label="Actual (Fuel + Coal)", linestyle="--")  # ÷2 applied
    # ax.plot([val/2 for val in predicted_vals[-30:]], label="Predicted (LSTM)", linewidth=2)
    # ax.set_xlabel("Days")
    # ax.set_ylabel("CO₂ Emitted (kg)")
    # ax.set_title("CO₂ Emission Trend – Last 30 Days")
    # ax.legend()
    # st.pyplot(fig)

