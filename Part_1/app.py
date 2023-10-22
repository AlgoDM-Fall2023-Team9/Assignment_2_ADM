import datetime
import json
import pandas as pd
import re
from snowflake.snowpark import functions as F
from snowflake.snowpark import version as v
from snowflake.snowpark.session import Session

from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.preprocessing import KBinsDiscretizer, OneHotEncoder
from snowflake.ml.modeling.impute import SimpleImputer

from snowflake.snowpark.functions import col
import streamlit as st
import snowflake.connector.pandas_tools as sfpd

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime as dt
import io
import seaborn as sns

import snowflake.snowpark.dataframe

#Ensure that your credentials are stored in creds.json
with open('creds.json') as f:
    data = json.load(f)
    USERNAME = data['user']
    PASSWORD = data['password']
    SF_ACCOUNT = data['account']
  

CONNECTION_PARAMETERS = {
   "account": "xtjlspt-mwb21630",
   "user": "kimaya185",
   "password": "Par010101",
   "database":"AD_FORECAST_DEMO",
   "schema": "public"
}

session = Session.builder.configs(CONNECTION_PARAMETERS).create()

session.sql('USE WAREHOUSE AD_FORECAST_DEMO_WH').collect()

# Create a Snowflake connection
connection = snowflake.connector.connect(**CONNECTION_PARAMETERS)

# Set the current warehouse
connection.cursor().execute("USE WAREHOUSE AD_FORECAST_DEMO_WH")

# Define the start date
start_date = datetime.date(2022, 12, 6)

# Create Streamlit inputs
st.title('Anomaly Detection & Forecasting')

# User selection dropdown
selected_option = st.selectbox('Select an option:', ('Anomaly Detection', 'Forecasting'))

if selected_option == 'Anomaly Detection':
    st.header('Anomaly Detection')
    period = st.date_input("Select the date for anomaly detection", start_date)
    impression = st.number_input("Select the impression count", 1200)

    if st.button("Run Anomaly Model"):
        def fetch_query3(date, impression_count):
            query = f'''
            CALL impression_anomaly_detector!DETECT_ANOMALIES(
            INPUT_DATA => SYSTEM$QUERY_REFERENCE('select ''{date}''::timestamp as day, {impression_count} as impressions'),
            TIMESTAMP_COLNAME =>'day',
            TARGET_COLNAME => 'impressions'
            );
            '''
            return query

        query = fetch_query3(period, impression)
        result = session.sql(query).collect()
        st.write('Anomaly Detection Result:')
        st.write(result)
else:
    st.header("Time Series Data")
session.sql('CALL impressions_forecast!FORECAST(FORECASTING_PERIODS => 14)').collect()

forecast = '''
SELECT day AS ts, impression_count AS actual, NULL AS forecast, NULL AS lower_bound, NULL AS upper_bound 
FROM daily_impressions 
UNION ALL 
SELECT ts, NULL AS actual, forecast, lower_bound, upper_bound 
FROM TABLE(RESULT_SCAN(-1))
'''

result = session.sql(forecast)
print(type(result))
print(result)
df=result

# Convert it to a Pandas DataFrame

pandas_dataframe = result.toPandas()
df1=pandas_dataframe
st.write(df1)
df=df1

final_data=df1


# Function to create the time series graph
def create_time_series_plot(data):
    df = data
    df['TS'] = pd.to_datetime(df['TS'])

    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(x='TS', y='ACTUAL', data=df, label='Actual', color='blue')
    ax = sns.lineplot(x='TS', y='FORECAST', data=df, label='Forecast', color='green')


    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Time Series Graph')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    return plt

# Streamlit app
st.title("Time Series Data")

# Display the plot in Streamlit
st.pyplot(create_time_series_plot(final_data))