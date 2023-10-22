#Analomy
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

import snowflake.snowpark.dataframe

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
   "database":"AD_FORECAST_DEMO",
   "schema": "public"
}

session = Session.builder.configs(CONNECTION_PARAMETERS).create()

session.sql('USE WAREHOUSE AD_FORECAST_DEMO_WH').collect()

# Create a Snowflake connection
connection = snowflake.connector.connect(**CONNECTION_PARAMETERS)

# Set the current warehouse
connection.cursor().execute("USE WAREHOUSE AD_FORECAST_DEMO_WH")



import datetime

# Define the start date
start_date = datetime.date(2022, 12, 6)


# Create Streamlit inputs
st.title('Impression Anomaly Detection')
period = st.date_input("Select the date for anomaly detection", start_date)
impression = st.number_input("Select the impression count", 1200)

if st.button("Run Anomaly Model"):
    # Function to fetch data based on your query
    def fetch_query3(date, impression_count):
        query = f'''
        CALL impression_anomaly_detector!DETECT_ANOMALIES(
        INPUT_DATA => SYSTEM$QUERY_REFERENCE('select ''{date}''::timestamp as day, {impression_count} as impressions'),
        TIMESTAMP_COLNAME =>'day',
        TARGET_COLNAME => 'impressions'
        );
        '''
        return query

    # Call the function to retrieve the query
    query = fetch_query3(period, impression)

    # Execute the query with the Snowflake session
    result = session.sql(query).collect()

    # Display the result using Streamlit
    st.write('Anomaly Detection Result:')
    st.write(result)

# st.write("Anomaly dectecting for  12000 impression")
# query = '''
#  CALL impression_anomaly_detector!DETECT_ANOMALIES(
#   INPUT_DATA => SYSTEM$QUERY_REFERENCE('select ''2022-12-06''::timestamp as day, 12000 as impressions'),
#   TIMESTAMP_COLNAME =>'day',
#   TARGET_COLNAME => 'impressions'
# );
# '''
# impression_anomaly_detector = session.sql(query).collect()

# st.write(impression_anomaly_detector)

# st.write("Anomaly dectecting for  120000 impression")

# query = '''
#  CALL impression_anomaly_detector!DETECT_ANOMALIES(
#   INPUT_DATA => SYSTEM$QUERY_REFERENCE('select ''2022-12-06''::timestamp as day, 120000 as impressions'),
#   TIMESTAMP_COLNAME =>'day',
#   TARGET_COLNAME => 'impressions'
# );
# '''
# impression_anomaly_detector = session.sql(query).collect()

# st.write(impression_anomaly_detector)

# st.write("Anomaly dectecting for  60000 impression")

# query = '''
#  CALL impression_anomaly_detector!DETECT_ANOMALIES(
#   INPUT_DATA => SYSTEM$QUERY_REFERENCE('select ''2022-12-06''::timestamp as day, 60000 as impressions'),
#   TIMESTAMP_COLNAME =>'day',
#   TARGET_COLNAME => 'impressions'
# );
# '''
# impression_anomaly_detector = session.sql(query).collect()

# st.write(impression_anomaly_detector)