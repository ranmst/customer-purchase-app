import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("model/final_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

st.title("🛍️ Customer Purchase Prediction App")

st.sidebar.header("Customer Information")

# User inputs
age = st.sidebar.slider("Age", 18, 70, 30)
gender = st.sidebar.selectbox("Gender", [0, 1])
income = st.sidebar.number_input("Annual Income", min_value=5000, max_value=200000, value=50000)
num_purchases = st.sidebar.slider("Number of Purchases", 0, 50, 10)
product_category = st.sidebar.selectbox("Product Category", [0, 1, 2, 3, 4])
time_spent = st.sidebar.slider("Time Spent on Website (mins)", 0, 60, 20)
loyalty = st.sidebar.selectbox("Loyalty Program Member", [0, 1])
discounts = st.sidebar.slider("Discounts Availed", 0, 10, 2)

# Step 1: Create raw input in training column order
raw_input = pd.DataFrame([[
    age, gender, income, num_purchases, time_spent,
    loyalty, discounts
]], columns=[
    'Age', 'Gender', 'AnnualIncome', 'NumberOfPurchases',
    'TimeSpentOnWebsite', 'LoyaltyProgram', 'DiscountsAvailed'
])

# Step 2: Scale only numerical features (keep gender/loyalty untouched)
numerical_cols = ['Age', 'AnnualIncome', 'NumberOfPurchases', 'TimeSpentOnWebsite', 'DiscountsAvailed']
scaled_values = scaler.transform(raw_input[numerical_cols])
scaled_df = pd.DataFrame(scaled_values, columns=numerical_cols)

# Step 3: Build full input in correct order:

final_df = pd.DataFrame(columns=[
    'Age', 'Gender', 'AnnualIncome', 'NumberOfPurchases',
    'TimeSpentOnWebsite', 'LoyaltyProgram', 'DiscountsAvailed',
    'PC_0', 'PC_1', 'PC_2', 'PC_3', 'PC_4'
])

final_df.loc[0] = [
    scaled_df['Age'][0],
    raw_input['Gender'][0],
    scaled_df['AnnualIncome'][0],
    scaled_df['NumberOfPurchases'][0],
    scaled_df['TimeSpentOnWebsite'][0],
    raw_input['LoyaltyProgram'][0],
    scaled_df['DiscountsAvailed'][0],
    int(product_category == 0),
    int(product_category == 1),
    int(product_category == 2),
    int(product_category == 3),
    int(product_category == 4)
]

# Step 4: Predict
if st.button("Predict Purchase"):
    prediction = model.predict(final_df)[0]
    result = "✅ Will Purchase" if prediction == 1 else "❌ Will Not Purchase"
    st.subheader("Prediction Result")
    st.success(result)
    st.subheader("📁 Project File Structure")

# cd C:\Users\hp\Desktop\customer_purchase_app
# streamlit run streamlit_app.py