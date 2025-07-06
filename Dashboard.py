import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.title("💸 Dynamic Pricing Dashboard")
st.markdown("Analyze trends and predict ride prices using Linear Regression.")

# Load data
df = pd.read_csv("dynamic_pricing.csv")

# Feature and target selection
features = ["Number_of_Riders", "Number_of_Drivers", "Number_of_Past_Rides", "Average_Ratings", "Expected_Ride_Duration"]
target = "Historical_Cost_of_Ride"
X = df[features]
y = df[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# What-if scenario input
st.subheader("🔧 What-If Pricing Scenario")
with st.expander("Adjust Inputs to Predict Ride Cost"):
    col1, col2, col3 = st.columns(3)
    riders = col1.slider("Number of Riders", 10, 200, 50)
    drivers = col2.slider("Number of Drivers", 10, 200, 50)
    past_rides = col3.slider("Past Rides", 0, 100, 20)

    col4, col5 = st.columns(2)
    rating = col4.slider("Average Rating", 1.0, 5.0, 4.0)
    duration = col5.slider("Ride Duration (min)", 5, 180, 60)

    # Create input example
    example = pd.DataFrame([{
        "Number_of_Riders": riders,
        "Number_of_Drivers": drivers,
        "Number_of_Past_Rides": past_rides,
        "Average_Ratings": rating,
        "Expected_Ride_Duration": duration
    }])

    example_scaled = scaler.transform(example)
    predicted_price = lr.predict(example_scaled)[0]

    st.metric("💰 Predicted Ride Cost", f"₹{predicted_price:.2f}")

# Model performance
st.subheader("📈 Model Performance")
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"**R² Score:** {r2:.2f}")
st.write(f"**Mean Absolute Error:** ₹{mae:.2f}")

# Coefficient visualization
st.subheader("📉 Standardized Feature Impact (Linear Regression)")
coeff_df = pd.DataFrame({
    "Feature": features,
    "Standardized Coefficient": lr.coef_
}).sort_values("Standardized Coefficient", ascending=True)

fig = px.bar(
    coeff_df,
    x="Standardized Coefficient",
    y="Feature",
    orientation="h",
    title="Relative Impact of Features on Ride Price",
    color="Standardized Coefficient",
    color_continuous_scale="RdBu"
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("🔍 View Coefficient Values"):
    st.dataframe(coeff_df.style.format({"Standardized Coefficient": "{:.2f}"}))
