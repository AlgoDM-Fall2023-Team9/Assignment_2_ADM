# Assignment_2_ADM
## APPLICATION LINK: 

1)  https://forecastandanomaly.streamlit.app/
2)  https://finalslpy-66mmqw2rjdzbudjtzmknsu.streamlit.app/
# Project Title

**Ad Impression Forecasting and Anomaly Detection using Snowflake's ML-Powered Analysis**

## Overview

This project focuses on ad impression forecasting and anomaly detection using Snowflake's ML-Powered Analysis. It's designed to provide insights into ad campaigns, predict impressions, and detect anomalies during these campaigns for advertisers and media publishers.

## Prerequisites

Before getting started, ensure you have the following prerequisites:

- Access to a Snowflake account (or the Snowflake 30-day trial).
- Basic knowledge of SQL and database concepts.
- Python 3.x with required packages (numpy, pandas, streamlit, sklearn, LinearRegression).

## Repository Contents

- [analysis_code.py](analysis_code.py): Python script for ad impression forecasting and anomaly detection using Snowflake's ML-Powered Analysis.

## How to Use

1. Clone this repository to your local machine.
2. Store your Snowflake account credentials in a file named `creds.json`.
3. Run `analysis_code.py` to execute ad impression forecasting and anomaly detection.
4. Follow the on-screen instructions to select options and provide input data.

## Usage

This code offers the following functionality:

### Anomaly Detection

- Select a date and an impression count.
- The script uses Snowflake's Anomaly Detection capabilities to determine if the data point is an outlier, potentially indicating an anomaly in ad impressions.

### Forecasting

- Time series forecasting of ad impressions.
- View actual and forecasted data on a time series graph, providing insights into impression trends.

## Contributing

Contributions are welcome! Open issues, suggest improvements, or submit pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

Special thanks to the Snowflake team for their contributions to this field.

---

# Customer Lifetime Value Prediction with XGBoost in Snowflake

**2.1. Snowflake ML-Powered Analysis with XGBoost**

## Overview

This project demonstrates predictive modeling using Snowflake's ML-Powered Analysis with the XGBoost algorithm. It focuses on predicting Customer Lifetime Value (CLV) using a retail dataset. You can adapt this code for various machine learning tasks within the Snowflake platform.

## Prerequisites

- Access to a Snowflake account.
- Python environment with required dependencies.
- Sample retail dataset or access to Snowflake's TPC-DS dataset.

## Getting Started

1. Clone this repository.
2. Store your Snowflake credentials in a `creds.json` file.
3. Run the `analysis_code.py` script for data engineering, feature engineering, model training, and scoring.
4. Save and use the trained model in Snowflake via User-Defined Functions (UDFs).

## Acknowledgments

Thanks to the Snowflake team for their contributions to this field.

---

# Predict Customer Spend - Regression

**2.2. Predict Customer Spend - Regression**

## Overview

This project predicts customer spending using a Linear Regression model, Scikit-Learn, Snowpark, and Python UDFs. It helps an e-commerce retailer make data-driven decisions to improve either the mobile app or the website based on factors impacting customer spending.

## Prerequisites

- Snowpark for Python library v0.6.
- Snowflake account.
- Python libraries (scikit-learn, pandas, numpy, matplotlib, seaborn, streamlit, Jupyter).

## What You'll Learn

This tutorial covers data engineering, machine learning, and creating a Streamlit application. You will:

1. Use Snowpark for Python for feature engineering.
2. Train a machine learning model locally and deploy it as a Python UDF in Snowflake.
3. Visualize the model's predictions in a Streamlit app.

## Usage/Steps

For detailed steps and to run the project, refer to the [Hex notebook/app](https://hex.snowflake.com/notebooks/5v9soy/7#S21) or follow the instructions in the project's directory.

![ecommapp](ecommapp.png)

Enjoy exploring this regression model and understanding customer spending patterns!

---

# Advertising Spend and ROI Prediction

**2.3. Advertising Spend and ROI Prediction**

## Overview

This project involves data engineering, data analysis, and machine learning to train a Linear Regression model to predict Return On Investment (ROI) based on advertising spend budgets across different channels. The project leverages Snowpark for Python, Streamlit, and scikit-learn to create a powerful solution.

## Prerequisites

- Snowflake Account.
- Snowpark for Python.
- Streamlit.
- Python libraries (scikit-learn, pandas, numpy).

## Step-By-Step Guide

Follow these steps:

1. **Data Engineering**: Analyze and preprocess the dataset for machine learning.

2. **Machine Learning Model**: Train a Linear Regression model using scikit-learn.

3. **Interactive Web Application**: Create a Streamlit application to input ad spend budgets and visualize predicted ROI.

4. **Deployment**: Deploy the web application for users to explore ROI predictions.

For detailed instructions, code samples, and a step-by-step walkthrough, refer to the complete [QuickStart Guide](https://quickstarts.snowflake.com/guide/getting_started_with_dataengineering_ml_using_snowpark_python/index.html#0).

By following this guide, you'll gain experience in data engineering, machine learning, and application development, while also understanding the impact of advertising spend on ROI.


CODELABS: https://codelabs-preview.appspot.com/?file_id=1WR5ZXhYVMP-GxSyWELGjM2q52UGPN-7-YoIZSOOKi1s#3 

ZOOM RECORDING: https://northeastern.zoom.us/rec/share/yYJZcZP_HoDgtLO66SsbQWeUAzcTrHFIFjFtYYvffhnpIQPfzuLR7MtncJL4nD7k.Xpph_JBVrWFbbnsF?startTime=1697944398000

Passcode: mJ@br5j%

Contributions: 
Kimaya: 33%
Siddhesh: 33%
Shaurin: 34%
