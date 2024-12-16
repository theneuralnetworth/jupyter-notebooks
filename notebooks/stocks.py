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

# COMMON CONSTANTS
CUTOFF_DATE = "2004-11-26"
STOCKS_DATASET = "datasets/stocks.csv"

# Load environment variables
load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
use_local_files = os.getenv('USE_LOCAL_FILES', 'false').lower() == 'true'

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
import kagglehub

# KaggleHub won't download the dataset again if it already exists
directory = kagglehub.dataset_download("benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated")

files = []
for dirname, _, filenames in os.walk(directory):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

print(f"Downloaded {len(files)} files")

sp500_symbols = pd.read_csv('datasets/sp500.csv')
sp500_symbols = sp500_symbols.rename(columns={'Symbol': 'symbol', 'GICS Sector': 'sector'})

sp500_info = sp500_symbols[['symbol', 'sector']]
def is_sp500(filename):
    """Check if the filename is a SP500 stock"""
    symbol = os.path.basename(filename).split('.')[0]
    return symbol in sp500_info['symbol'].values

files = list(filter(is_sp500, files))
print(f"Filtered down to {len(files)} files")

# Merge all the files into a single dataframe
dataframes = []

for f in files:
    df = pd.read_csv(f)
    df['symbol'] = os.path.basename(f).split('.')[0]
    df = df[df['Date'] > CUTOFF_DATE]
    dataframes.append(df)

print("Merging all dataframes into one")
df = pd.concat(dataframes, ignore_index=True)
df = df.merge(sp500_info, how='left', on='symbol')

df['sector'], sectors = pd.factorize(df['sector'])
sectors = dict(enumerate(sectors))

# Save the dataframe
print("Saving the dataset to ", STOCKS_DATASET)
if not os.path.exists(STOCKS_DATASET) or input(f"Overwrite {STOCKS_DATASET}? [y/n] ").lower() == 'y':
    df.to_csv(STOCKS_DATASET, index=False)


# %%
print(sectors)
df.drop_duplicates(subset='sector')

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
