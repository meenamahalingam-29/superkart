import requests
import streamlit as st
import pandas as pd

st.title("SuperKart Sales Forecast")

st.subheader("Single Product Sales Prediction")

# Input fields
Product_Id = st.text_input("Product_Id", value="FDX07")
Product_Weight = st.number_input("Product_Weight", value=12.5)
Product_Sugar_Content = st.selectbox("Product_Sugar_Content", ["Regular", "Low sugar", "No sugar"])
Product_Allocated_Area = st.number_input("Product_Allocated_Area", value=0.07, format="%.3f")
Product_Type = st.selectbox("Product_Type", [
    'Baking Goods',
    'Breads',
    'Breakfast',
    'Canned',
    'Dairy',
    'Frozen Foods',
    'Fruits and Vegetables',
    'Hard Drinks',
    'Health and Hygiene',
    'Household',
    'Meat',
    'Seafood',
    'Snack Foods',
    'Soft Drinks',
    'Starchy Foods',
    'Others'
])
Product_MRP = st.number_input("Product_MRP", value=150.0)
Store_Id = st.selectbox("Store_Id", ["OUT001", "OUT002", "OUT003", "OUT004", "Other"])
Store_Size = st.selectbox("Store_Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store_Location_City_Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store_Type", ['Departmental Store','Food Mart','Supermarket Type1','Supermarket Type2', 'Other'])
Store_Age = st.number_input("Store_Age", value=20)

payload = {
    "Product_Id": Product_Id,
    "Product_Weight": float(Product_Weight),
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": float(Product_Allocated_Area),
    "Product_Type": Product_Type,
    "Product_MRP": float(Product_MRP),
    "Store_Id": Store_Id,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Store_Age": int(Store_Age)
}

if st.button("Predict Sales", type="primary"):
    response = requests.post(
        "https://upgraded-chainsaw-4qgwrjqpw7wjhj5pw-7860.app.github.dev/v1/predict",
        json=payload
    )
    if response.status_code == 200:
        result = response.json()
        prediction = result["Predicted_Product_Store_Sales_Total"]
        st.success(f"Predicted Product_Store_Sales_Total: {prediction}")
    else:
        st.error("Error in API request")

st.divider()

st.subheader("Batch Prediction")

file = st.file_uploader("Upload CSV file", type=["csv"])
if file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(
            "https://upgraded-chainsaw-4qgwrjqpw7wjhj5pw-7860.app.github.dev/v1/predictbatch",
            files={"file": file.getvalue()}
        )
        if response.status_code == 200:
            result = response.json()
            st.header("Batch Prediction Results")
            st.write(result)
        else:
            st.error("Error in API request")
