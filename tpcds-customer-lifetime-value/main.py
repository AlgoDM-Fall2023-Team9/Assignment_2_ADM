import streamlit as st
import pandas as pd
from snowflake.connector import connect
import json

# Ensure that your credentials are stored in creds.json
with open('creds.json') as f:
    data = json.load(f)
    USERNAME = data['user']
    PASSWORD = data['password']
    SF_ACCOUNT = data['account']

CONNECTION_PARAMETERS = {
    "account": SF_ACCOUNT,
    "user": USERNAME,
    "password": PASSWORD,
    "warehouse": "snowpark_opt_wh",
    "database": "TPCDS_XGBOOST",
    "schema": "demo",
}

# Connect to Snowflake
connection = connect(**CONNECTION_PARAMETERS)

# Streamlit interface
st.title("Customer Sales Prediction")

# Input fields
gender = st.selectbox("Gender", ["M", "F"])
marital_status = st.selectbox("Marital Status", ["D", "M", "S", "U", "W"])
credit_rating = st.selectbox("Credit Rating", ["Good", "High Risk", "Low Risk"])
education_status = st.selectbox("Education Status", ["2 yr Degree", "4 yr Degree", "Advanced Degree", "College", "Primary", "Secondary", "Unknown"])
birth_year = st.number_input("Birth Year", value=1990)
dependency_count = st.number_input("Dependency Count", value=1)
Aggregated_Income = st.number_input("Aggregated Income", value=20000)

# Error flag to track invalid inputs
input_error = False

if st.button("Predict"):
    # Define mappings for user inputs
    user_inputs = {
        "gender": gender,
        "marital_status": marital_status,
        "credit_rating": credit_rating,
        "education_status": education_status,
        "C_BIRTH_YEAR": birth_year,
        "CD_DEP_COUNT": dependency_count,
        "Aggregated Income": Aggregated_Income
    }

    # Define mappings for feature names
    feature_mappings = {
        "gender": {"F": "CD_GENDER_F", "M": "CD_GENDER_M"},
        "marital_status": {"D": "CD_MARITAL_STATUS_D", "M": "CD_MARITAL_STATUS_M", "S": "CD_MARITAL_STATUS_S", "U": "CD_MARITAL_STATUS_U", "W": "CD_MARITAL_STATUS_W"},
        "credit_rating": {"Good": "CD_CREDIT_RATING_GOOD", "High Risk": "CD_CREDIT_RATING_HIGHRISK", "Low Risk": "CD_CREDIT_RATING_LOWRISK"},
        "education_status": {"2 yr Degree": "CD_EDUCATION_STATUS_2YRDEGREE", "4 yr Degree": "CD_EDUCATION_STATUS_4YRDEGREE", "Advanced Degree": "CD_EDUCATION_STATUS_ADVANCEDDEGREE", "College": "CD_EDUCATION_STATUS_COLLEGE", "Primary": "CD_EDUCATION_STATUS_PRIMARY", "Secondary": "CD_EDUCATION_STATUS_SECONDARY", "Unknown": "CD_EDUCATION_STATUS_UNKNOWN"}
    }

    # Initialize input_data_dict with default values
    input_data_dict = {
        "CD_GENDER_F": 0,
        "CD_GENDER_M": 0,
        "CD_MARITAL_STATUS_D": 0,
        "CD_MARITAL_STATUS_M": 0,
        "CD_MARITAL_STATUS_S": 0,
        "CD_MARITAL_STATUS_U": 0,
        "CD_MARITAL_STATUS_W": 0,
        "CD_CREDIT_RATING_GOOD": 0,
        "CD_CREDIT_RATING_HIGHRISK": 0,
        "CD_CREDIT_RATING_LOWRISK": 0,
        "CD_EDUCATION_STATUS_2YRDEGREE": 0,
        "CD_EDUCATION_STATUS_4YRDEGREE": 0,
        "CD_EDUCATION_STATUS_ADVANCEDDEGREE": 0,
        "CD_EDUCATION_STATUS_COLLEGE": 0,
        "CD_EDUCATION_STATUS_PRIMARY": 0,
        "CD_EDUCATION_STATUS_SECONDARY": 0,
        "CD_EDUCATION_STATUS_UNKNOWN": 0,
        "C_BIRTH_YEAR": 0,
        "CD_DEP_COUNT": 0,
        "Aggregated_Income": 0
    }

    # Update input_data_dict based on user inputs
    for feature, value in user_inputs.items():
        if feature in feature_mappings and value in feature_mappings[feature]:
            input_data_dict[feature_mappings[feature][value]] = 1.0
        else:
            # Handle invalid inputs
            if feature == "C_BIRTH_YEAR" and (value < 1900 or value > 2023):  # Validate Birth Year
                input_error = True
                st.error("Invalid input for Birth Year. Please enter a valid year between 1900 and 2023.")
            elif feature == "Aggregated_Income" and (value < 0):
                input_error = True
                st.error("Invalid input for Total Sales. Please enter a non-negative value.")

    if not input_error:
        # Create a DataFrame from the input data
        input_data_df = pd.DataFrame([input_data_dict])

        # Call Snowflake UDF
        with connection.cursor() as cursor:
            input_data_values = ','.join(map(str, input_data_df.values[0]))
            cursor.execute(f"SELECT TPCDS_PREDICT_CLV({input_data_values})")
            prediction = cursor.fetchone()[0]

        st.write(f"Predicted Total Sales: {prediction}")
