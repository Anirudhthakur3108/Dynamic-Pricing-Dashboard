import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("dynamic_pricing.csv")

# ----------------- Sidebar Filters -----------------
st.sidebar.title("🔍 Filter Options")
time_filter = st.sidebar.multiselect("Time of Booking", df["Time_of_Booking"].unique(), default=df["Time_of_Booking"].unique())
location_filter = st.sidebar.multiselect("Location Category", df["Location_Category"].unique(), default=df["Location_Category"].unique())
vehicle_filter = st.sidebar.multiselect("Vehicle Type", df["Vehicle_Type"].unique(), default=df["Vehicle_Type"].unique())

filtered_df = df[
    (df["Time_of_Booking"].isin(time_filter)) &
    (df["Location_Category"].isin(location_filter)) &
    (df["Vehicle_Type"].isin(vehicle_filter))
]

# ----------------- Main Dashboard -----------------
st.title("🚖 Dynamic Pricing Dashboard")

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Ride Cost", f"${filtered_df['Historical_Cost_of_Ride'].mean():.2f}")
col2.metric("Avg Duration (min)", f"{filtered_df['Expected_Ride_Duration'].mean():.1f}")
col3.metric("Rider-Driver Ratio", f"{(filtered_df['Number_of_Riders'].sum() / filtered_df['Number_of_Drivers'].sum()):.2f}")
col4.metric("Avg Rating", f"{filtered_df['Average_Ratings'].mean():.2f}")

# --- Cost by Time of Booking ---
st.subheader("💸 Cost Trends by Time of Booking")
fig = px.box(filtered_df, x="Time_of_Booking", y="Historical_Cost_of_Ride", color="Vehicle_Type")
st.plotly_chart(fig, use_container_width=True)

# --- Cost by Location and Vehicle ---
st.subheader("📍 Avg Cost by Location and Vehicle Type")
pivot_table = filtered_df.groupby(["Location_Category", "Vehicle_Type"])["Historical_Cost_of_Ride"].mean().reset_index()
fig2 = px.bar(pivot_table, x="Location_Category", y="Historical_Cost_of_Ride", color="Vehicle_Type", barmode="group")
st.plotly_chart(fig2, use_container_width=True)

# ----------------- What-If Scenario -----------------
st.subheader("🔧 What-If Pricing Scenario")

with st.expander("Adjust Inputs to Predict Ride Cost"):
    col1, col2, col3 = st.columns(3)
    riders = col1.slider("Number of Riders", 10, 200, 50)
    drivers = col2.slider("Number of Drivers", 10, 200, 50)
    past_rides = col3.slider("Past Rides", 0, 100, 20)

    col4, col5 = st.columns(2)
    rating = col4.slider("Average Rating", 1.0, 5.0, 4.0)
    duration = col5.slider("Ride Duration (min)", 5, 180, 60)

    # Encode example scenario
    example = pd.DataFrame({
        "Number_of_Riders": [riders],
        "Number_of_Drivers": [drivers],
        "Number_of_Past_Rides": [past_rides],
        "Average_Ratings": [rating],
        "Expected_Ride_Duration": [duration],
    })

    # Prepare training data
    features = ["Number_of_Riders", "Number_of_Drivers", "Number_of_Past_Rides", "Average_Ratings", "Expected_Ride_Duration"]
    model = LinearRegression()
    model.fit(df[features], df["Historical_Cost_of_Ride"])
    prediction = model.predict(example)[0]

    st.success(f"💰 Estimated Ride Cost: **₹{prediction:.2f}**")

    # ----------------- Coefficient Visualization -----------------
    st.subheader("📉 Linear Regression Coefficients")
    
    # Get coefficients
    coeff_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": lr.coef_
    }).sort_values("Coefficient", ascending=False)
    
    # Visualize
    fig = px.bar(
        coeff_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        title="Impact of Features on Predicted Price (₹)",
        color="Coefficient",
        color_continuous_scale="RdBu",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Optional: Display raw data
    with st.expander("🔍 View Coefficient Values"):
        st.dataframe(coeff_df.style.format({"Coefficient": "{:.2f}"}))

