import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# 1. LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv("sales_data.csv")

print("Dataset loaded successfully!")


# =====================================================
# 2. DATA PREPROCESSING
# =====================================================

print("Preprocessing data...")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    by=["Store ID", "Product ID", "Date"]
).reset_index(drop=True)


# Next day inventory
df["Next_Day_Inventory"] = (
    df.groupby(
        ["Store ID", "Product ID"]
    )["Inventory Level"].shift(-1)
)


# Next day demand
df["Next_Day_Demand"] = (
    df.groupby(
        ["Store ID", "Product ID"]
    )["Demand"].shift(-1)
)


# Stockout risk
df["Stockout_Risk"] = (
    df["Next_Day_Inventory"]
    <
    df["Next_Day_Demand"]
).astype(int)


# Remove rows without next-day values
df = df.dropna(
    subset=[
        "Next_Day_Inventory",
        "Next_Day_Demand"
    ]
)


# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================

df["Inventory_Demand_Ratio"] = (
    df["Inventory Level"]
    /
    (df["Demand"] + 1)
)


df["Sales_Inventory_Ratio"] = (
    df["Units Sold"]
    /
    (df["Inventory Level"] + 1)
)


df["Price_Difference"] = (
    df["Price"]
    -
    df["Competitor Pricing"]
)


# Date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek


# =====================================================
# 4. FEATURES
# =====================================================

features = [

    "Category",
    "Region",

    "Inventory Level",
    "Units Sold",
    "Units Ordered",

    "Price",
    "Discount",

    "Weather Condition",
    "Promotion",

    "Competitor Pricing",
    "Seasonality",
    "Epidemic",

    "Demand",

    "Inventory_Demand_Ratio",
    "Sales_Inventory_Ratio",
    "Price_Difference",

    "Year",
    "Month",
    "Day",
    "DayOfWeek"
]


X = df[features]

y = df["Stockout_Risk"]


# =====================================================
# 5. FEATURE TYPES
# =====================================================

categorical_features = [

    "Category",
    "Region",
    "Weather Condition",
    "Seasonality"
]


numerical_features = [

    "Inventory Level",
    "Units Sold",
    "Units Ordered",

    "Price",
    "Discount",
    "Promotion",

    "Competitor Pricing",
    "Epidemic",
    "Demand",

    "Inventory_Demand_Ratio",
    "Sales_Inventory_Ratio",
    "Price_Difference",

    "Year",
    "Month",
    "Day",
    "DayOfWeek"
]


# =====================================================
# 6. PREPROCESSOR
# =====================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            StandardScaler(),
            numerical_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# =====================================================
# 7. TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# =====================================================
# 8. LOGISTIC REGRESSION
# =====================================================

print("\nTraining Logistic Regression...")

logistic_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


logistic_model.fit(
    X_train,
    y_train
)


logistic_predictions = (
    logistic_model.predict(X_test)
)


print(
    "Logistic Regression completed!"
)


# =====================================================
# 9. RANDOM FOREST
# =====================================================

print("\nTraining Random Forest...")

random_forest_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            RandomForestClassifier(

                n_estimators=100,

                random_state=42,

                class_weight="balanced"
            )
        )
    ]
)


random_forest_model.fit(
    X_train,
    y_train
)


random_forest_predictions = (
    random_forest_model.predict(X_test)
)


print(
    "Random Forest completed!"
)


# =====================================================
# 10. GRADIENT BOOSTING
# =====================================================

print("\nTraining Gradient Boosting...")

gradient_boosting_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",

            GradientBoostingClassifier(

                n_estimators=100,

                learning_rate=0.1,

                random_state=42
            )
        )
    ]
)


gradient_boosting_model.fit(
    X_train,
    y_train
)


gradient_boosting_predictions = (
    gradient_boosting_model.predict(X_test)
)


print(
    "Gradient Boosting completed!"
)


# =====================================================
# 11. CALCULATE METRICS
# =====================================================

def get_metrics(y_true, predictions):

    return {

        "Accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "Precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "F1 Score":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
    }


logistic_metrics = get_metrics(
    y_test,
    logistic_predictions
)


random_forest_metrics = get_metrics(
    y_test,
    random_forest_predictions
)


gradient_metrics = get_metrics(
    y_test,
    gradient_boosting_predictions
)


# =====================================================
# 12. SAVE RESULTS
# =====================================================

results = pd.DataFrame({

    "Model": [

        "Logistic Regression",

        "Random Forest",

        "Gradient Boosting"
    ],

    "Accuracy": [

        logistic_metrics["Accuracy"],

        random_forest_metrics["Accuracy"],

        gradient_metrics["Accuracy"]
    ],

    "Precision": [

        logistic_metrics["Precision"],

        random_forest_metrics["Precision"],

        gradient_metrics["Precision"]
    ],

    "Recall": [

        logistic_metrics["Recall"],

        random_forest_metrics["Recall"],

        gradient_metrics["Recall"]
    ],

    "F1 Score": [

        logistic_metrics["F1 Score"],

        random_forest_metrics["F1 Score"],

        gradient_metrics["F1 Score"]
    ]
})


# =====================================================
# 13. CREATE MODELS FOLDER
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)


# =====================================================
# 14. SAVE MODELS
# =====================================================

print("\nSaving models...")


joblib.dump(
    logistic_model,
    "models/logistic_model.joblib"
)


joblib.dump(
    random_forest_model,
    "models/random_forest_model.joblib"
)


joblib.dump(
    gradient_boosting_model,
    "models/gradient_boosting_model.joblib"
)


# Save results
joblib.dump(
    results,
    "models/model_results.joblib"
)


# Save feature information
joblib.dump(
    {
        "features": features,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features
    },
    "models/feature_info.joblib"
)


# =====================================================
# 15. SAVE PROCESSED DATA
# =====================================================

df.to_csv(
    "models/processed_data.csv",
    index=False
)


# =====================================================
# 16. DISPLAY RESULTS
# =====================================================

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")

print("\nModel Performance:")

print(
    results.round(4)
)

print("\nFiles saved inside models/ folder:")
print("✓ logistic_model.joblib")
print("✓ random_forest_model.joblib")
print("✓ gradient_boosting_model.joblib")
print("✓ model_results.joblib")
print("✓ feature_info.joblib")
print("✓ processed_data.csv")

print("\nYou can now run:")
print("streamlit run app.py")