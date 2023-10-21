import json
import pandas as pd
import sys

from snowflake.snowpark import functions as F
from snowflake.snowpark import version as v
from snowflake.snowpark.session import Session
import snowflake.connector

from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.preprocessing import KBinsDiscretizer, OneHotEncoder
from snowflake.ml.modeling.impute import SimpleImputer

import streamlit as st
import joblib
from snowflake.snowpark import functions as F
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")

# Ensure that your credentials are stored in creds.json
with open('creds.json') as f:
    data = json.load(f)
    USERNAME = data['user']
    PASSWORD = data['password']
    SF_ACCOUNT = data['account']
    SF_WH = data['warehouse']
    

CONNECTION_PARAMETERS = {
   "account": SF_ACCOUNT,
   "user": USERNAME,
   "password": PASSWORD,
}
session = Session.builder.configs(CONNECTION_PARAMETERS).create()


session.sql('CREATE DATABASE IF NOT EXISTS tpcds_xgboost').collect()
session.sql('CREATE SCHEMA IF NOT EXISTS tpcds_xgboost.demo').collect()
session.sql("create or replace warehouse FE_AND_INFERENCE_WH with warehouse_size='3X-LARGE'").collect()
session.sql("create or replace warehouse snowpark_opt_wh with warehouse_size = 'MEDIUM' warehouse_type = 'SNOWPARK-OPTIMIZED'").collect()
session.sql("alter warehouse snowpark_opt_wh set max_concurrency_level = 1").collect()
session.sql("CREATE OR REPLACE STAGE TPCDS_XGBOOST.DEMO.ML_MODELS").collect()
session.use_warehouse('FE_AND_INFERENCE_WH')
session.use_database('tpcds_xgboost')
session.use_schema('demo')


TPCDS_SIZE_PARAM = 10
SNOWFLAKE_SAMPLE_DB = 'SNOWFLAKE_SAMPLE_DATA' # Name of Snowflake Sample Database might be different...

if TPCDS_SIZE_PARAM == 100: 
    TPCDS_SCHEMA = 'TPCDS_SF100TCL'
elif TPCDS_SIZE_PARAM == 10:
    TPCDS_SCHEMA = 'TPCDS_SF10TCL'
else:
    raise ValueError("Invalid TPCDS_SIZE_PARAM selection")
    
store_sales = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.store_sales')
catalog_sales = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.catalog_sales') 
web_sales = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.web_sales') 
date = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.date_dim')
dim_stores = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.store')
customer = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.customer')
address = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.customer_address')
demo = session.table(f'{SNOWFLAKE_SAMPLE_DB}.{TPCDS_SCHEMA}.customer_demographics')




#DATA ENGIEERING

store_sales_agged = store_sales.group_by('ss_customer_sk').agg(F.sum('ss_sales_price').as_('total_sales'))
web_sales_agged = web_sales.group_by('ws_bill_customer_sk').agg(F.sum('ws_sales_price').as_('total_sales'))
catalog_sales_agged = catalog_sales.group_by('cs_bill_customer_sk').agg(F.sum('cs_sales_price').as_('total_sales'))
store_sales_agged = store_sales_agged.rename('ss_customer_sk', 'customer_sk')
web_sales_agged = web_sales_agged.rename('ws_bill_customer_sk', 'customer_sk')
catalog_sales_agged = catalog_sales_agged.rename('cs_bill_customer_sk', 'customer_sk')

total_sales = store_sales_agged.union_all(web_sales_agged)
total_sales = total_sales.union_all(catalog_sales_agged)
total_sales = total_sales.group_by('customer_sk').agg(F.sum('total_sales').as_('total_sales'))
customer = customer.select('c_customer_sk','c_current_hdemo_sk', 'c_current_addr_sk', 'c_customer_id', 'c_birth_year')
customer = customer.join(address.select('ca_address_sk', 'ca_zip'), customer['c_current_addr_sk'] == address['ca_address_sk'] )
customer = customer.join(demo.select('cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_credit_rating', 'cd_education_status', 'cd_dep_count'),
                                customer['c_current_hdemo_sk'] == demo['cd_demo_sk'] )
customer = customer.rename('c_customer_sk', 'customer_sk')
customer.limit(5).to_pandas()

final_df = total_sales.join(customer, on='customer_sk')
# Size of the final DF is around 95 Million.

final_df.count()


session.use_database('tpcds_xgboost')
session.use_schema('demo')
final_df.write.mode('overwrite').save_as_table('feature_store')

#FEATURE ENGINEERING 
session.use_warehouse('snowpark_opt_wh')
session.use_database('tpcds_xgboost')
session.use_schema('demo')
snowdf = session.table("feature_store")
snowdf = snowdf.drop(['CA_ZIP','CUSTOMER_SK', 'C_CURRENT_HDEMO_SK', 'C_CURRENT_ADDR_SK', 'C_CUSTOMER_ID', 'CA_ADDRESS_SK', 'CD_DEMO_SK'])


snowdf.limit(5).to_pandas()

# Define feature_cols based on your Snowflake feature engineering
feature_cols = ['C_BIRTH_YEAR', 'CD_GENDER', 'CD_MARITAL_STATUS', 'CD_CREDIT_RATING', 'CD_EDUCATION_STATUS', 'CD_DEP_COUNT']

# # Streamlit UI
# st.title('Customer Lifetime Value Prediction')
# st.header('Enter Customer Information:')

# total_sales = st.number_input('Total Sales')
# birth_year = st.number_input('Birth Year')
# gender = st.selectbox('Gender', ['M', 'F'])
# marital_status = st.selectbox('Marital Status', ['S', 'M', 'D', 'W'])
# credit_rating = st.selectbox('Credit Rating', ['Low', 'Good', 'Excellent'])
# education_status = st.selectbox('Education Status', ['High School', 'College', 'Advanced Degree'])
# dep_count = st.number_input('Number of Dependents')

# if st.button('Predict CLV'):
#     user_data = pd.DataFrame({
#         'TOTAL_SALES': [total_sales],
#         'C_BIRTH_YEAR': [birth_year],
#         'CD_GENDER': [gender],
#         'CD_MARITAL_STATUS': [marital_status],
#         'CD_CREDIT_RATING': [credit_rating],
#         'CD_EDUCATION_STATUS': [education_status],
#         'CD_DEP_COUNT': [dep_count]
#     })

#     # Feature engineering on user_data
#     user_data['C_BIRTH_YEAR'] = user_data['C_BIRTH_YEAR'].astype(float)
#     user_data['CD_DEP_COUNT'] = user_data['CD_DEP_COUNT'].astype(float)

#     # Imputation of Numeric Cols
#     my_imputer = SimpleImputer(input_cols=['C_BIRTH_YEAR', 'CD_DEP_COUNT'],
#                                output_cols=['C_BIRTH_YEAR', 'CD_DEP_COUNT'],
#                                strategy='median')
#     user_data = my_imputer.fit(user_data).transform(user_data)

#     # OHE of Categorical Cols
#     my_ohe_encoder = OneHotEncoder(input_cols=['CD_GENDER', 'CD_MARITAL_STATUS', 'CD_CREDIT_RATING', 'CD_EDUCATION_STATUS'],
#                                    output_cols=['CD_GENDER', 'CD_MARITAL_STATUS', 'CD_CREDIT_RATING', 'CD_EDUCATION_STATUS'],
#                                    drop_input_cols=True)
#     user_data = my_ohe_encoder.fit(user_data).transform(user_data)

#     # Prepare user data for prediction
#     user_data_for_prediction = user_data[feature_cols]

#     # Load your XGBoost model and predict
#     xgb_model = joblib.load('model.joblib.gz')  # Load your XGBoost model
#     predictions = xgb_model.predict(user_data_for_prediction)
    
#     st.subheader('Customer Lifetime Value Prediction:')
#     st.write(predictions[0])

# # Display some information about the features or model
# st.write('Feature Information:')
# st.write('C_BIRTH_YEAR: Customer birth year')
# st.write('CD_GENDER: Customer gender')
# st.write('CD_MARITAL_STATUS: Customer marital status')
# st.write('CD_CREDIT_RATING: Customer credit rating')
# st.write('CD_EDUCATION_STATUS: Customer education status')
# st.write('CD_DEP_COUNT: Number of dependents')

from snowflake.snowpark.functions import col

snowdf = snowdf.withColumn("C_BIRTH_YEAR", col("C_BIRTH_YEAR").cast("float"))
snowdf = snowdf.withColumn("CD_DEP_COUNT", col("CD_DEP_COUNT").cast("float"))


# Rest of your code for imputation and one-hot encoding
cat_cols = ['CD_GENDER', 'CD_MARITAL_STATUS', 'CD_CREDIT_RATING', 'CD_EDUCATION_STATUS']
num_cols = ['C_BIRTH_YEAR', 'CD_DEP_COUNT']

# Imputation of Numeric Cols
my_imputer = SimpleImputer(input_cols=num_cols,
                           output_cols=num_cols,
                           strategy='median')
sdf_prepared = my_imputer.fit(snowdf).transform(snowdf)

# OHE of Categorical Cols
my_ohe_encoder = OneHotEncoder(input_cols=cat_cols, output_cols=cat_cols, drop_input_cols=True)
sdf_prepared = my_ohe_encoder.fit(sdf_prepared).transform(sdf_prepared)

sdf_prepared.limit(5).to_pandas()


# Cleaning column names to make it easier for future referencing
import re

cols = sdf_prepared.columns
for old_col in cols:
    new_col = re.sub(r'[^a-zA-Z0-9_]', '', old_col) 
    new_col = new_col.upper()
    sdf_prepared = sdf_prepared.withColumnRenamed(old_col, new_col)


# Use Snowpark Optimized Warehouse
session.use_warehouse('snowpark_opt_wh')


# Prepare Data for modeling
feature_cols = sdf_prepared.columns
feature_cols.remove('TOTAL_SALES')
target_col = 'TOTAL_SALES'


# Save the train and test sets as time stamped tables in Snowflake
snowdf_train, snowdf_test = sdf_prepared.random_split([0.8, 0.2], seed=82) 
snowdf_train.write.mode("overwrite").save_as_table("tpcds_xgboost.demo.tpc_TRAIN")
snowdf_test.write.mode("overwrite").save_as_table("tpcds_xgboost.demo.tpc_TEST")



# Define the XGBRegressor and fit the model
xgbmodel = XGBRegressor(random_state=123, input_cols=feature_cols, label_cols=target_col, output_cols='PREDICTION')
xgbmodel.fit(snowdf_train)



# Score the data using the fitted xgbmodel
sdf_scored = xgbmodel.predict(snowdf_test)


print(sdf_scored.limit(5).to_pandas())


#Save predictions in Snowflake

session.use_database('tpcds_xgboost')
session.use_schema('demo')
sdf_scored.write.mode('overwrite').save_as_table('predictions')


snowdf_test = session.table('tpc_TEST')
# Predicting with sample dataset
sample_data = snowdf_test.limit(100)
sample_data.write.mode("overwrite").save_as_table("temp_test")
test_sdf = session.table('temp_test')

import joblib
import cachetools


xgb_file = xgbmodel.to_xgboost()
xgb_file


MODEL_FILE = 'model.joblib.gz'
joblib.dump(xgb_file, MODEL_FILE) # we are just pickling it locally first


# You can also save the pickled object into the stage we created earlier
session.file.put(MODEL_FILE, "@ML_MODELS", auto_compress=False, overwrite=True)


from snowflake.snowpark.functions import udf
import snowflake.snowpark.types as T


from cachetools import cached

@cached(cache={})
def load_model(model_path: str) -> object:
    from joblib import load
    model = load(model_path)
    return model

def udf_score_xgboost_model_vec_cached(df: pd.DataFrame) -> pd.Series:
    import os
    import sys
    # file-dependencies of UDFs are available in snowflake_import_directory
    IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
    import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
    model_name = 'model.joblib.gz'
    model = load_model(import_dir+model_name)
    df.columns = feature_cols
    scored_data = pd.Series(model.predict(df))
    return scored_data


udf_clv = session.udf.register(func=udf_score_xgboost_model_vec_cached, 
                               name="TPCDS_PREDICT_CLV", 
                               stage_location='@ML_MODELS',
                               input_types=[T.FloatType()]*len(feature_cols),
                               return_type = T.FloatType(),
                               replace=True, 
                               is_permanent=True, 
                               imports=['@ML_MODELS/model.joblib.gz'],
                               packages=['pandas',
                                         'xgboost',
                                         'joblib',
                                         'cachetools'], 
                               session=session)



test_sdf_w_preds = test_sdf.with_column('PREDICTED', udf_clv(*feature_cols))
test_sdf_w_preds.limit(2).to_pandas()



test_sdf_w_preds = test_sdf.with_column('PREDICTED',F.call_udf("TPCDS_PREDICT_CLV",
                                                               [F.col(c) for c in feature_cols]))
print(test_sdf_w_preds.limit(2).to_pandas())
