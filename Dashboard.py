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
# ----------------- What-If Scenario -----------------
st.subheader("🔧 What-If Pricing Scenario")

with st.expander("Adjust Inputs to Predict Ride Cost (Model Comparison)"):
    col1, col2, col3 = st.columns(3)
    riders = col1.slider("Number of Riders", 10, 200, 50)
    drivers = col2.slider("Number of Drivers", 10, 200, 50)
    past_rides = col3.slider("Past Rides", 0, 100, 20)

    col4, col5 = st.columns(2)
    rating = col4.slider("Average Rating", 1.0, 5.0, 4.0)
    duration = col5.slider("Ride Duration (min)", 5, 180, 60)

    example = pd.DataFrame({
        "Number_of_Riders": [riders],
        "Number_of_Drivers": [drivers],
        "Number_of_Past_Rides": [past_rides],
        "Average_Ratings": [rating],
        "Expected_Ride_Duration": [duration],
    })

    # Model training setup
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    features = ["Number_of_Riders", "Number_of_Drivers", "Number_of_Past_Rides", "Average_Ratings", "Expected_Ride_Duration"]
    X = df[features]
    y = df["Historical_Cost_of_Ride"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(example)[0]
    y_pred_lr = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(example)[0]
    y_pred_rf = rf.predict(X_test)

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(example)[0]
    y_pred_xgb = xgb.predict(X_test)

    # Display predictions
    st.markdown("### 💡 Model Predictions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", f"₹{lr_pred:.2f}")
    col2.metric("Random Forest", f"₹{rf_pred:.2f}")
    col3.metric("XGBoost", f"₹{xgb_pred:.2f}")

    # Show performance metrics
    with st.expander("📈 Model Performance (on Test Data)"):
        perf = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest", "XGBoost"],
            "R² Score": [
                r2_score(y_test, y_pred_lr),
                r2_score(y_test, y_pred_rf),
                r2_score(y_test, y_pred_xgb),
            ],
            "MAE (₹)": [
                mean_absolute_error(y_test, y_pred_lr),
                mean_absolute_error(y_test, y_pred_rf),
                mean_absolute_error(y_test, y_pred_xgb),
            ]
        })
        st.dataframe(perf.style.format({"R² Score": "{:.2f}", "MAE (₹)": "{:.2f}"}))


