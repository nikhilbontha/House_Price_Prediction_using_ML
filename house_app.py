import streamlit as st
import pandas as pd
import numpy as np

# TITLE

st.title(" Advanced House Price Prediction")

st.write("Linear Regression Model with Feature Engineering")

# LOAD DATA

@st.cache_data
def load_data():
    data = pd.read_csv("train.csv")
    return data

data = load_data()

st.subheader("Step 1 : Original Dataset")
st.write(data.head())


 # SELECT IMPORTANT COLUMNS


important_columns = [
    "SalePrice",
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "Neighborhood",
    "LotArea",
    "KitchenQual"
]

data = data[important_columns]

st.subheader("Step 2 : Selected Important Columns")
st.write(data.head())

neighborhood_list = sorted(data["Neighborhood"].dropna().unique())
kitchenqual_list = sorted(data["KitchenQual"].dropna().unique())

 # HANDLE MISSING VALUES
 
st.subheader("Step 3 : Handling Missing Values")

# numeric → median
num_cols = data.select_dtypes(include=["int64","float64"]).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())
st.write("Data after handling numeric missing values")
st.write(data[num_cols].head())

# categorical → mode
cat_cols = data.select_dtypes(include=["object"]).columns
data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])
st.write("Data after handling categorical missing values")
st.write(data[cat_cols].head())

st.write("Dataset after handling missing values")
st.write(data.head())

 # FEATURE ENGINEERING
 
st.subheader("Step 4 : Feature Engineering")

# Total Area
data["TotalArea"] = data["TotalBsmtSF"] + data["GrLivArea"]
st.write("Data of total area")
st.write(data[["TotalBsmtSF","GrLivArea","TotalArea"]].head())


# House Age
current_year = 2024
data["HouseAge"] = current_year - data["YearBuilt"]
st.write("Data of house age")
st.write(data[["YearBuilt","HouseAge"]].head())

st.write("Dataset after Feature Engineering")
st.write(data.head())

 # ENCODE CATEGORICAL VARIABLES


st.subheader("Step 5 : Encoding Categorical Features")

data = pd.get_dummies(data, columns=["Neighborhood","KitchenQual"], drop_first=True)
data = data.astype(int, errors='ignore')

st.write("Dataset after Encoding")
st.write(data.head())

# SPLIT FEATURES AND TARGET

st.subheader("Step 6 : Splitting Features and Target")

X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

st.write("Features")
st.write(X.head())

st.write("Target")
st.write(y.head())

# TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.subheader("Step 7 : Train Test Split")

st.write("Training Data Shape:", X_train.shape)
st.write("Testing Data Shape:", X_test.shape)


# TRAIN MODEL

st.subheader("Step 8 : Training Linear Regression Model")

model = LinearRegression()
model.fit(X_train, y_train)
st.write("Model Coefficients:", model.coef_)

st.success("Model Training Completed")

# MODEL EVALUATION

st.subheader("Step 9 : Model Evaluation")

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

st.write("RMSE :", rmse)
st.write("R² Score :", r2)

# PREDICTION SECTION

st.subheader("Predict House Price")

overallqual = st.slider("Overall Quality",1,10)
grlivarea = st.number_input("Living Area",500,5000)
garagecars = st.slider("Garage Cars",0,4)
basement = st.number_input("Basement Area",0,3000)
lotarea = st.number_input("Lot Area",1000,20000)
yearbuilt = st.number_input("Year Built",1900,2024)

neighborhood = st.selectbox(
    "Neighborhood",
    neighborhood_list
)

kitchenqual = st.selectbox(
    "Kitchen Quality",
    kitchenqual_list
)

# Feature engineering
totalarea = basement + grlivarea
houseage = 2024 - yearbuilt

# PREDICT BUTTON

if st.button("Predict Price"):

    # create dataframe with same columns used in training
    input_df = pd.DataFrame(columns=X.columns)

    # fill with zeros
    input_df.loc[0] = 0

    # fill numeric values
    input_df["OverallQual"] = overallqual
    input_df["GrLivArea"] = grlivarea
    input_df["GarageCars"] = garagecars
    input_df["TotalBsmtSF"] = basement
    input_df["LotArea"] = lotarea
    input_df["HouseAge"] = houseage
    input_df["TotalArea"] = totalarea

    # set categorical dummy values
    neigh_col = "Neighborhood_" + neighborhood
    if neigh_col in input_df.columns:
        input_df[neigh_col] = 1

    kit_col = "KitchenQual_" + kitchenqual
    if kit_col in input_df.columns:
        input_df[kit_col] = 1

    # prediction
    prediction = model.predict(input_df)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")