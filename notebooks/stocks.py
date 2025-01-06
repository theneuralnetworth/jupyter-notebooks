# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stock Price Prediction System
# #### Team 1: Neural Networth

# %%
# Imports
import os
import pandas as pd
import json
from dotenv import load_dotenv
import kagglehub
import os
import matplotlib.pyplot as plt
import requests
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import numpy as np

# COMMON CONSTANTS
CUTOFF_DATE = "2004-11-26"
STOCKS_DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'stocks.csv')

# Load environment variables
load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
use_local_files = os.getenv('USE_LOCAL_FILES', 'true').lower() == 'true'

# %% [markdown]
# ## 1. Get the dataset
#
# In this section, we download the dataset from Kaggle that contains many files. We then merge all these files into a single dataset for further analysis.
#
# 1. Downloads dataset from Kaggle
# 1. Filters the dataset to only include Companies from S&P 500
# 1. Excludes records older than `CUTOFF_DATE` (default: 2004-11-26)
# 1. Merges all the files into a single dataset
# 1. Stores the dataset in the `STOCKS_DATASET` (default: `datasets/stocks.csv`) file

# %%

# KaggleHub won't download the dataset again if it already exists
directory = kagglehub.dataset_download("benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated")

files = []
for dirname, _, filenames in os.walk(directory):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

print(f"Downloaded {len(files)} files")

sp500_symbols = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets', 'sp500.csv'))
sp500_symbols = sp500_symbols.rename(columns={'Symbol': 'Symbol', 'GICS Sector': 'Sector'})

sp500_info = sp500_symbols[['Symbol', 'Sector']]
def is_sp500(filename):
    """Check if the filename is a SP500 stock"""
    symbol = os.path.basename(filename).split('.')[0]
    return symbol in sp500_info['Symbol'].values

files = list(filter(is_sp500, files))
print(f"Filtered down to {len(files)} files")

# Merge all the files into a single dataframe
dataframes = []

for f in files:
    df = pd.read_csv(f)
    df['Symbol'] = str(os.path.basename(f).split('.')[0])
    df = df[df['Date'] > CUTOFF_DATE]
    dataframes.append(df)

print("Merging all dataframes into one")
data = pd.concat(dataframes, ignore_index=True)
data = data.merge(sp500_info, how='left', on='Symbol')

data['sector'], sectors = pd.factorize(data['Sector'])
sectors = dict(enumerate(sectors))

# Save the dataframe
print("Saving the dataset to ", STOCKS_DATASET)
if not os.path.exists(STOCKS_DATASET) or input(f"Overwrite {STOCKS_DATASET}? [y/n] ").lower() == 'y':
    data.to_csv(STOCKS_DATASET, index=False)


# %%
print(sectors)
data.drop_duplicates(subset='sector')

# %% [markdown]
# ## 2. Exploratory Data Analysis
#
# ...

# %% [markdown]
# ## 3. Feature Engineering

# %% [markdown]
# ### 3.1 Economic indicators

# %% [markdown]
# #### 3.1.1 CPI

# %%
#Convert every date to datetime
data['Date'] = pd.to_datetime(data['Date'], utc=True)

#%%
## Add timeframe features

data['Quarter'] = data['Date'].dt.quarter
data['week_number'] = data['Date'].dt.isocalendar().week
data['week_day'] = data['Date'].dt.weekday
data['Month']  = data['Date'].dt.month

#%%
## Calculate new features
data["Daily Range"] = data["High"] - data["Low"]
data["Direction"] = (data["Close"] > data["Open"]).astype(int)
data["Percentage Change"] = ((data["Close"] - data["Open"]) / data["Open"]) * 100
data["Momentum_10"] = data["Close"] - data["Close"].shift(10)
data['Volume Change'] = data['Volume'].pct_change() * 100

# Daily Return
data['7-Day Return'] = (data['Close'] / data['Close'].shift(7) - 1) * 100  # 7 days
data['1-Month Return'] = (data['Close'] / data['Close'].shift(30) - 1) * 100  # 30 days
data['6-Month Return'] = (data['Close'] / data['Close'].shift(182) - 1) * 100  # ~6 months
data['1-Year Return'] = (data['Close'] / data['Close'].shift(365) - 1) * 100  # 1 year
#Future daily return (target)
data['Daily Return'] = data['Close'].pct_change()
data['Next Day Return'] = data['Daily Return'].shift(-1) * 100
data['Next Week Return'] = data['Daily Return'].shift(-7)
data['Next Month Return'] = data['Daily Return'].shift(-30)

data.info()
#%%

ALPHAVANTAGE_API_URL = f"https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={api_key}"
file_path = 'datasets/economic_indicators/cpi.json'

folder_path = os.path.dirname(file_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def get_cpi():
    response = requests.get(ALPHAVANTAGE_API_URL)
    data = response.json()
    cpi_monthly = data['data']
    return [{"date": cpi['date'], "value": float(cpi['value'])} for cpi in cpi_monthly]

if use_local_files and os.path.exists(file_path):
    with open(file_path, mode='r') as fp:
        cpi_monthly = json.load(fp)
else:
    cpi_monthly = get_cpi()
    with open(file_path, mode='w') as fp:
        json.dump(cpi_monthly, fp)

cpi_mapping = {
    entry['date'][:7]: float(entry['value'])
    for entry in cpi_monthly
}
data.info()

data['YearMonth'] = data['Date'].dt.strftime('%Y-%m')

data['CPI'] = data['YearMonth'].map(cpi_mapping)


data.drop(columns=['YearMonth'], inplace=True)

# %% [markdown]
# #### 3.1.2 Inflation

# %%
ALPHAVANTAGE_API_URL = f"https://www.alphavantage.co/query?function=INFLATION&apikey={api_key}"
file_path = 'datasets/economic_indicators/inflation.json'

folder_path = os.path.dirname(file_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def get_inflation():
    response = requests.get(ALPHAVANTAGE_API_URL)
    data = response.json()
    data_inflation = data['data']
    return [{"date": inflation['date'], "value": float(inflation['value'])} for inflation in data_inflation]

if use_local_files and os.path.exists(file_path):
    with open(file_path, mode='r') as fp:
        data_inflation = json.load(fp)
else:
    data_inflation = get_inflation()
    with open(file_path, mode='w') as fp:
        json.dump(data_inflation, fp)

# Create a mapping of inflation values by year
inflation_mapping = {item['date'][:4]: float(item['value']) for item in data_inflation}
# Add inflation value for 2024
inflation_mapping['2024'] = 2.6

# Add a new column for Inflation to the dataset
data['Inflation'] = None

# Apply the inflation value for each row based on the year
for index, row in data.iterrows():
    year = row['Date'].year
    if str(year) in inflation_mapping:
        data.at[index, 'Inflation'] = inflation_mapping[str(year)]

# Ensure 'Inflation' column is numeric
data['Inflation'] = pd.to_numeric(data['Inflation'], errors='coerce')

# Check for NaNs in the column after conversion
if data['Inflation'].isnull().any():
    print("Found NaNs in 'Inflation' column after conversion. Filling with a default value (e.g., median).")
    data['Inflation'].fillna(data['Inflation'].median(), inplace=True)

# %% [markdown]
# #### 3.1.3 Interest rates

# %%
ALPHAVANTAGE_API_URL = f"https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=monthly&apikey=${api_key}"
file_path = 'datasets/economic_indicators/interest_rates.json'

folder_path = os.path.dirname(file_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def get_interest_rates():
    response = requests.get(ALPHAVANTAGE_API_URL)
    data = response.json()
    rates = data['data']
    return [{"date": rate['date'], "value": float(rate['value'])} for rate in rates]

if use_local_files and os.path.exists(file_path):
    with open(file_path, mode='r') as fp:
        rates = json.load(fp)
else:
    rates = get_interest_rates()
    with open(file_path, mode='w') as fp:
        json.dump(rates, fp)

rates_df = pd.DataFrame(rates)

data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

rates_df['date'] = pd.to_datetime(rates_df['date'])
rates_df['Year'] = rates_df['date'].dt.year
rates_df['Month'] = rates_df['date'].dt.month

df_merged = pd.merge(data, rates_df, on=['Year', 'Month'], how='inner')

df_merged.drop(columns=['date', 'Year', 'Month'], inplace=True)
df_merged.rename(columns={'value': 'Interest Rate'}, inplace=True)

df_merged['Interest Rate'] = df_merged['Interest Rate'] / 100
data['Interest Rate'] = df_merged['Interest Rate']
# %% [markdown]
# #### 3.1.4 Real GDP 

# %%
ALPHAVANTAGE_API_URL = f"https://www.alphavantage.co/query?function=REAL_GDP&interval=quarterly&apikey={api_key}"
file_path = 'datasets/economic_indicators/real_gdp.json'

folder_path = os.path.dirname(file_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def get_real_gdp():
    response = requests.get(ALPHAVANTAGE_API_URL)
    data = response.json()
    real_gdp_quarterly = data['data']
    return [{"date": real_gdp['date'], "value": float(real_gdp['value'])} for real_gdp in real_gdp_quarterly]

if use_local_files and os.path.exists(file_path):
    with open(file_path, mode='r') as fp:
        real_gdp_quarterly = json.load(fp)
else:
    real_gdp_quarterly = get_real_gdp()
    with open(file_path, mode='w') as fp:
        json.dump(real_gdp_quarterly, fp)

real_gdp_mapping = {
    entry['date'][:4] + '-Q' + str((int(entry['date'][5:7]) - 1) // 3 + 1): entry['value']
    for entry in real_gdp_quarterly
}

def get_quarterly_gdp_from_quarter(row):
    quarter_key = row['Date'].strftime('%Y') + '-Q' + str(row['Quarter'])
    return real_gdp_mapping.get(quarter_key, None)

data['REAL_GDP'] = data.apply(get_quarterly_gdp_from_quarter, axis=1)

# %% [markdown]
# #### 3.1.5 Real GDP per capita

# %%
ALPHAVANTAGE_API_URL = f"https://www.alphavantage.co/query?function=REAL_GDP_PER_CAPITA&interval=quarterly&apikey={api_key}"
file_path = 'datasets/economic_indicators/real_gdp_per_capita.json'

folder_path = os.path.dirname(file_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def get_real_gdp_per_capita():
    response = requests.get(ALPHAVANTAGE_API_URL)
    data = response.json()
    real_gdp_per_capita_quarterly = data['data']
    return [{"date": real_gdp_per_capita['date'], "value": float(real_gdp_per_capita['value'])} for real_gdp_per_capita in real_gdp_per_capita_quarterly]

if use_local_files and os.path.exists(file_path):
    with open(file_path, mode='r') as fp:
        real_gdp_per_capita_quarterly = json.load(fp)
else:
    real_gdp_per_capita_quarterly = get_real_gdp_per_capita()
    with open(file_path, mode='w') as fp:
        json.dump(real_gdp_per_capita_quarterly, fp)


real_gdp_per_capita_mapping = {
    entry['date'][:4] + '-Q' + str((int(entry['date'][5:7]) - 1) // 3 + 1): entry['value']
    for entry in real_gdp_per_capita_quarterly
}

def get_quarterly_gdp_per_capita_from_quarter(row):
    quarter_key = row['Date'].strftime('%Y') + '-Q' + str(row['Quarter'])
    return real_gdp_per_capita_mapping.get(quarter_key, None)

data['REAL_GDP_PER_CAPITA'] = data.apply(get_quarterly_gdp_per_capita_from_quarter, axis=1)

# %% [markdown]
# #### 3.1.6. Future and Past Returns

# Ensure the data is sorted by symbol (company) and date
df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)

def calculate_returns(group):

    group['Daily Return'] = group['Close'].pct_change()

    group['Next Day Return'] = group['Daily Return'].shift(-1) * 100  # Next day

    group['7-Day Return'] = (group['Close'] / group['Close'].shift(7) - 1) * 100  # Last 7 days
    group['1-Month Return'] = (group['Close'] / group['Close'].shift(30) - 1) * 100  # Last 30 days
    group['1-Year Return'] = (group['Close'] / group['Close'].shift(365) - 1) * 100  # Last 365 days
    
    return group

# Apply the function to each company group
df = df.groupby('Symbol', group_keys=False).apply(calculate_returns).reset_index(drop=True)


#%%
# Correlation Matrix
def get_correlation_matrix():
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix')
    plt.show()

# get_correlation_matrix()

#%%
# Remove outliers~

data['Z-Score'] = (data['Daily Return'] - data['Daily Return'].mean()) / data['Daily Return'].std()
data_no_outliers = data[(data['Z-Score'] >= -2.5) & (data['Z-Score'] <= 2.5)]
data_no_outliers = data_no_outliers.drop(columns=['Z-Score'])
data = data_no_outliers

#%%
# Ensure there are no missing values before defining features and target
data = data.dropna()  # Ensure there are no missing values

# Define the target variable and features
target = 'Next Day Return'  # Example target variable
features = [
    'Quarter', 'week_number', 'week_day', 'CPI',
    'Daily Range', 'Volume', 'Momentum_10', 'Volume Change',
    'Percentage Change', '7-Day Return', '1-Month Return', '6-Month Return',
    '1-Year Return', 'Daily Return'
]

# Prepare the input data
inputData = data[features]
targetVariable = data[target]

# Check if inputData and targetVariable have the same length
if len(inputData) != len(targetVariable):
    raise ValueError("Mismatch between input data and target variable lengths.")

# Split the data into training and testing sets
input_train, input_test, target_train, target_test = train_test_split(inputData, targetVariable, test_size=0.2, random_state=42, shuffle=False)

#%%
# Define models to use
def get_ML_model(model_name, X_train, y_train):
    if model_name == 'RandomForest':
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
        )
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            objective='reg:squarederror',
            random_state=42,
            booster="dart"
        )
    elif model_name == 'LightGBM':
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
    else:
        raise ValueError(f"Model '{model_name}' is not recognized. Please choose 'RandomForest', 'XGBoost', or 'LightGBM'.")

    model.fit(X_train, y_train)

    return model

input_train = input_train.replace([np.inf, -np.inf], np.nan)
input_train = input_train.dropna()

model_name = 'XGBoost'
# model_name = 'RandomForest'
# model_name = 'LightGBM'
model = get_ML_model(model_name, input_train, target_train)


#%%
# Predict on the test set
target_pred = model.predict(input_test)

mse = mean_squared_error(target_test, target_pred)
r2 = r2_score(target_test, target_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-Squared: {r2:.4f}")
# %%
