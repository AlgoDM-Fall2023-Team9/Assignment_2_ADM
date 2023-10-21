
import json
import pandas as pd
import streamlit as st
from snowflake.snowpark import functions as F
from snowflake.snowpark import version as v
from snowflake.snowpark.session import Session

from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.preprocessing import KBinsDiscretizer, OneHotEncoder
from snowflake.ml.modeling.impute import SimpleImputer

from snowflake.snowpark.functions import col
import snowflake.snowpark.dataframe
import snowflake.connector.pandas_tools as sfpd


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import io
import seaborn as sns

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

# session.sql("CREATE ROLE analyst").collect()


# session.sql('USE ROLE ACCOUNTADMIN').collect()

# session.sql("GRANT USAGE ON DATABASE AD_FORECAST_DEMO TO ROLE analyst").collect()

# session.sql("GRANT USAGE ON SCHEMA AD_FORECAST_DEMO.DEMO TO ROLE analyst").collect()

# session.sql("GRANT USAGE ON WAREHOUSE AD_FORECAST_DEMO_WH TO ROLE analyst").collect()

# session.sql("GRANT CREATE TABLE ON SCHEMA AD_FORECAST_DEMO.DEMO TO ROLE analyst").collect()

# session.sql("GRANT CREATE VIEW ON SCHEMA AD_FORECAST_DEMO.DEMO TO ROLE analyst").collect()

# session.sql("GRANT CREATE SNOWFLAKE.ML.FORECAST ON SCHEMA AD_FORECAST_DEMO.DEMO TO ROLE analyst").collect()

# session.sql("GRANT CREATE SNOWFLAKE.ML.ANOMALY_DETECTION ON SCHEMA AD_FORECAST_DEMO.DEMO TO ROLE analyst").collect()

# session.sql('CREATE DATABASE AD_FORECAST_DEMO').collect()

# session.sql('CREATE SCHEMA AD_FORECAST_DEMO.DEMO;').collect()

#session.sql(' CREATE OR REPLACE SNOWFLAKE.ML.FORECAST impressions_forecast(INPUT_DATA => SYSTEM$REFERENCE("TABLE", "daily_impressions"), TIMESTAMP_COLNAME => "day",TARGET_COLNAME =>"impression_count"   );').collect()
        
# session.sql('CREATE WAREHOUSE AD_FORECAST_DEMO_WH WITH WAREHOUSE_SIZE='XSmall'  ).collect()


session.sql('USE WAREHOUSE AD_FORECAST_DEMO_WH').collect()

session.sql('CALL impressions_forecast!FORECAST(FORECASTING_PERIODS => 14)').collect()

st.title("Time Series Data")

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
