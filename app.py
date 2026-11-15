import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(

    page_title="Stockout Risk Prediction",

    page_icon="📦",

    layout="wide"
)


# =====================================================
# LOAD SAVED MODELS
# =====================================================

@st.cache_resource
def load_models():

    logistic_model = joblib.load(
        "models/logistic_model.joblib"
    )

    random_forest_model = joblib.load(
        "models/random_forest_model.joblib"
    )

    gradient_boosting_model = joblib.load(
        "models/gradient_boosting_model.joblib"
    )

    return (
        logistic_model,
        random_forest_model,
        gradient_boosting_model
    )


# =====================================================
# LOAD SAVED DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "models/processed_data.csv"
    )

    return df


# =====================================================
# LOAD RESULTS
# =====================================================

@st.cache_data
def load_results():

    return joblib.load(
        "models/model_results.joblib"
    )


# Load everything
logistic_model, random_forest_model, gradient_boosting_model = load_models()

df = load_data()

results = load_results()


# =====================================================
# TITLE
# =====================================================

st.title(
    "📦 Stockout Risk Prediction System"
)

st.write(
    "Machine Learning application for predicting "
    "product stockout risk."
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Choose Page",

    [
        "📊 Dataset",
        "🤖 Model Performance",
        "🔮 Stockout Prediction"
    ]
)


# =====================================================
# DATASET PAGE
# =====================================================

if page == "📊 Dataset":

    st.header(
        "📊 Dataset Overview"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with col3:

        st.metric(
            "Stockout Cases",
            int(
                df["Stockout_Risk"].sum()
            )
        )


    with col4:

        st.metric(
            "No Stockout",
            int(
                (df["Stockout_Risk"] == 0).sum()
            )
        )


    st.subheader(
        "Dataset Preview"
    )


    st.dataframe(

        df.head(20),

        use_container_width=True
    )


    st.subheader(
        "Stockout Risk Distribution"
    )


    risk_count = (
        df["Stockout_Risk"]
        .value_counts()
        .rename(
            index={
                0: "No Risk",
                1: "Risk"
            }
        )
    )


    st.bar_chart(
        risk_count
    )


    st.subheader(
        "Dataset Statistics"
    )


    st.dataframe(

        df.describe(),

        use_container_width=True
    )


# =====================================================
# MODEL PERFORMANCE PAGE
# =====================================================

elif page == "🤖 Model Performance":

    st.header(
        "🤖 Model Performance"
    )


    # Convert to percentage
    display_results = results.copy()


    for column in [

        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"

    ]:

        display_results[column] = (

            display_results[column] * 100

        ).round(2)


    st.dataframe(

        display_results,

        use_container_width=True
    )


    st.subheader(
        "Performance Comparison"
    )


    chart_data = results.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]


    st.bar_chart(
        chart_data
    )


    # =================================================
    # METRIC CARDS
    # =================================================

    st.subheader(
        "Model Metrics"
    )


    for _, row in results.iterrows():

        st.markdown(
            f"### {row['Model']}"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(

                "Accuracy",

                f"{row['Accuracy'] * 100:.2f}%"
            )


        with col2:

            st.metric(

                "Precision",

                f"{row['Precision'] * 100:.2f}%"
            )


        with col3:

            st.metric(

                "Recall",

                f"{row['Recall'] * 100:.2f}%"
            )


        with col4:

            st.metric(

                "F1 Score",

                f"{row['F1 Score'] * 100:.2f}%"
            )


# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🔮 Stockout Prediction":

    st.header(
        "🔮 Predict Stockout Risk"
    )


    st.write(
        "Enter the product information below."
    )


    # =================================================
    # INPUTS
    # =================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        category = st.selectbox(

            "Category",

            sorted(
                df["Category"]
                .dropna()
                .unique()
            )
        )


        region = st.selectbox(

            "Region",

            sorted(
                df["Region"]
                .dropna()
                .unique()
            )
        )


        inventory_level = st.number_input(

            "Inventory Level",

            min_value=0.0,

            value=float(
                df["Inventory Level"].median()
            )
        )


        units_sold = st.number_input(

            "Units Sold",

            min_value=0.0,

            value=float(
                df["Units Sold"].median()
            )
        )


        units_ordered = st.number_input(

            "Units Ordered",

            min_value=0.0,

            value=float(
                df["Units Ordered"].median()
            )
        )


    with col2:

        price = st.number_input(

            "Price",

            min_value=0.0,

            value=float(
                df["Price"].median()
            )
        )


        discount = st.number_input(

            "Discount",

            min_value=0.0,

            value=float(
                df["Discount"].median()
            )
        )


        promotion = st.number_input(

            "Promotion",

            min_value=0.0,

            value=float(
                df["Promotion"].median()
            )
        )


        competitor_pricing = st.number_input(

            "Competitor Pricing",

            min_value=0.0,

            value=float(
                df["Competitor Pricing"].median()
            )
        )


        demand = st.number_input(

            "Demand",

            min_value=0.0,

            value=float(
                df["Demand"].median()
            )
        )


    with col3:

        weather = st.selectbox(

            "Weather Condition",

            sorted(
                df["Weather Condition"]
                .dropna()
                .unique()
            )
        )


        seasonality = st.selectbox(

            "Seasonality",

            sorted(
                df["Seasonality"]
                .dropna()
                .unique()
            )
        )


        epidemic = st.number_input(

            "Epidemic",

            min_value=0.0,

            value=float(
                df["Epidemic"].median()
            )
        )


        year = st.number_input(

            "Year",

            min_value=2000,

            max_value=2100,

            value=int(
                df["Year"].median()
            )
        )


        month = st.number_input(

            "Month",

            min_value=1,

            max_value=12,

            value=int(
                df["Month"].median()
            )
        )


    # =================================================
    # CALCULATED FEATURES
    # =================================================

    inventory_demand_ratio = (

        inventory_level /

        (demand + 1)
    )


    sales_inventory_ratio = (

        units_sold /

        (inventory_level + 1)
    )


    price_difference = (

        price -

        competitor_pricing
    )


    day = int(
        df["Day"].median()
    )


    day_of_week = int(
        df["DayOfWeek"].median()
    )


    # =================================================
    # CREATE INPUT DATA
    # =================================================

    input_data = pd.DataFrame({

        "Category": [category],

        "Region": [region],

        "Inventory Level": [
            inventory_level
        ],

        "Units Sold": [
            units_sold
        ],

        "Units Ordered": [
            units_ordered
        ],

        "Price": [
            price
        ],

        "Discount": [
            discount
        ],

        "Weather Condition": [
            weather
        ],

        "Promotion": [
            promotion
        ],

        "Competitor Pricing": [
            competitor_pricing
        ],

        "Seasonality": [
            seasonality
        ],

        "Epidemic": [
            epidemic
        ],

        "Demand": [
            demand
        ],

        "Inventory_Demand_Ratio": [
            inventory_demand_ratio
        ],

        "Sales_Inventory_Ratio": [
            sales_inventory_ratio
        ],

        "Price_Difference": [
            price_difference
        ],

        "Year": [
            year
        ],

        "Month": [
            month
        ],

        "Day": [
            day
        ],

        "DayOfWeek": [
            day_of_week
        ]
    })


    # =================================================
    # MODEL SELECTION
    # =================================================

    model_name = st.selectbox(

        "Choose Prediction Model",

        [
            "Gradient Boosting",
            "Random Forest",
            "Logistic Regression"
        ]
    )


    if model_name == "Gradient Boosting":

        selected_model = gradient_boosting_model


    elif model_name == "Random Forest":

        selected_model = random_forest_model


    else:

        selected_model = logistic_model


    # =================================================
    # PREDICT
    # =================================================

    if st.button(

        "🔍 Predict Stockout Risk",

        type="primary"

    ):

        prediction = (

            selected_model
            .predict(input_data)
        )


        probability = (

            selected_model
            .predict_proba(
                input_data
            )[0][1]
        )


        st.subheader(
            "Prediction Result"
        )


        if prediction[0] == 1:

            st.error(
                "⚠️ STOCKOUT RISK"
            )


            st.metric(

                "Stockout Risk Probability",

                f"{probability * 100:.2f}%"
            )


            st.warning(

                "The model predicts that "
                "this product may face a "
                "stockout."
            )


        else:

            st.success(
                "✅ NO STOCKOUT RISK"
            )


            st.metric(

                "Stockout Risk Probability",

                f"{probability * 100:.2f}%"
            )


            st.info(

                "The model predicts that "
                "this product is not at "
                "high risk of stockout."
            )


# =====================================================
# FOOTER
# =====================================================

st.sidebar.markdown("---")

st.sidebar.write(
    "📦 Stockout Risk Prediction"
)

st.sidebar.write(
    "Powered by Machine Learning"
)