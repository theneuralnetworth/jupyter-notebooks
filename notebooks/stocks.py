# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Stock Price Prediction System
#
# ## Imports

# %%
import hashlib
# Ignore pandas warnings
import warnings
warnings.filterwarnings('ignore')
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import joblib
from fredapi import Fred

# COMMON CONSTANTS
CUTOFF_DATE = "2004-11-26"
STOCKS_DATASET = os.path.join(os.getcwd(), 'datasets', 'stocks.csv')

# Load environment variables
load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
fred_api_key = os.getenv("FRED_API_KEY")
use_local_files = os.getenv('USE_LOCAL_FILES', 'true').lower() == 'true'

# %% [markdown]
# ## Prepare the dataset
#
# In this section, we download the dataset from Kaggle that contains many files. We then merge all these files into a single dataset for further analysis.
#
# 1. Downloads dataset from Kaggle
# 1. Filters the dataset to only include Companies from S&P 500
# 1. Excludes records older than `CUTOFF_DATE` (default: 2004-11-26)
# 1. Merges all the files into a single dataset
# 1. Stores the dataset in the `STOCKS_DATASET` (default: `datasets/stocks.csv`) file

# %% [markdown]
# ### Merge all files into single dataset

# %%

# KaggleHub won't download the dataset again if it already exists
directory = kagglehub.dataset_download("benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated")

files = []
for dirname, _, filenames in os.walk(directory):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

print(f"Downloaded {len(files)} files")

sp500_symbols = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'sp500.csv'))
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

# One-Hot Encoding for Sector
data = pd.get_dummies(data, columns=['Sector'], prefix='Sector')

# Save the dataframe
print("Saving the dataset to ", STOCKS_DATASET)
if not os.path.exists(STOCKS_DATASET) or input(f"Overwrite {STOCKS_DATASET}? [Y/n] ").strip().lower() in {'y', ''}:
    data.to_csv(STOCKS_DATASET, index=False)

print("One-hot encoded columns:", [col for col in data.columns if col.startswith('Sector_')])

# %% [markdown]
# ### Adding new features
#
# #### Timeframe features

# %%
#Convert every date to datetime
data['Date'] = pd.to_datetime(data['Date'], utc=True)

## Add timeframe features
data['Quarter'] = data['Date'].dt.quarter
data['week_number'] = data['Date'].dt.isocalendar().week
data['week_day'] = data['Date'].dt.weekday
data['Month']  = data['Date'].dt.month

# %% [markdown]
# #### Technical indicators features
# Features like:
# * daily range, direction, percentage change, ...
# * moving averages, RSI, MACD, ...

# %%
## Calculate technical indicators
data["Daily Range"] = data["High"] - data["Low"]
data["Direction"] = (data["Close"] > data["Open"]).astype(int)
data["Percentage Change"] = ((data["Close"] - data["Open"]) / data["Open"]) * 100
data["Momentum_2"] = data["Close"] - data["Close"].shift(2)
data["Momentum_3"] = data["Close"] - data["Close"].shift(3)
data["Momentum_4"] = data["Close"] - data["Close"].shift(4)
data["Momentum_5"] = data["Close"] - data["Close"].shift(5)
data["Momentum_10"] = data["Close"] - data["Close"].shift(10)
data['Volume Change'] = data['Volume'].pct_change() * 100

# Simple Moving Average (SMA)
data["SMA_5"] = data["Close"].rolling(window=5).mean()
data["SMA_20"] = data["Close"].rolling(window=20).mean()

# Exponential Moving Average (EMA)
data["EMA_5"] = data["Close"].ewm(span=5, adjust=False).mean()
data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

# Remove rows where Volume Change is infinite
data = data[~np.isinf(data['Volume Change'])]

# Daily Return
data['Last Week Return'] = (data['Close'] / data['Close'].shift(7) - 1) * 100
#Future daily return (target)
data['Daily Return'] = data['Close'].pct_change()
data['Next Day Return'] = data['Daily Return'].shift(-1) * 100
data['Next Week Return'] = (data['Close'] / data['Close'].shift(-7) - 1) * 100
data['Next Month Return'] = (data['Close'] / data['Close'].shift(-30) - 1) * 100
data['Next Quarter Return'] = (data['Close'] / data['Close'].shift(-63) - 1) * 100 # excludes weekends

random_row = data.sample(n=1)
random_row

# %% [markdown]
# #### Inflation

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
    print("Found NaNs in 'Inflation' column after conversion. Filling with a default value (ie. median).")
    data['Inflation'].fillna(data['Inflation'].median(), inplace=True)

# %% [markdown]
# #### Economic Indicators from FRED (Federal Reserve Economic Data) API

# %%
api_key = 'fc61c0046f24fb037f982ee29ae1ab11'
fred = Fred(api_key = api_key)

# List of economic indicators to fetch from FRED API
economic_indicators = ["DTB4WK", "DTB3", "DTB6", "DGS5", "DGS10", "DAAA", "DBAA"]

features_fred = pd.DataFrame()

# Fetch data for each economic indicator and merge it into features_fred df
for indicator in economic_indicators :
    series_data = fred.get_series(indicator)
    series_data = series_data.reset_index()
    series_data.columns = ['Date', indicator]
    features_fred = pd.merge(features_fred, series_data, on='Date', how='outer') if not features_fred.empty else series_data

# Ensure the Date column has no timezone information to match with the FRED data
data['Date'] = data['Date'].dt.tz_localize(None)

# Convert the Date column in the FRED data to datetime format
features_fred['Date'] = pd.to_datetime(features_fred['Date'])

# Fill missing values with median for each economic indicator
for col in economic_indicators:
    features_fred[col].fillna(features_fred[col].median(), inplace=True)

data = pd.merge(data, features_fred, on='Date', how='left')

# Calculating interest rate spreads based on the indicators
data['TE1'] = data['DGS10'] - data['DTB4WK']
data['TE2'] = data['DGS10'] - data['DTB3']
data['TE3'] = data['DGS10'] - data['DTB6']
data['TE5'] = data['DTB3'] - data['DTB4WK']
data['TE6'] = data['DTB6'] - data['DTB4WK']

# %% [markdown]
# ## Feature importance analysis
# ### Correlation Matrix (numeric features only)

# %%
# Correlation Matrix
def get_correlation_matrix():
    numeric_columns = data.select_dtypes(include=[float, int])
    correlation_matrix = numeric_columns.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix')
    plt.show()

get_correlation_matrix()

# %% [markdown]
# ### Remove highly correlated features

# %%
data.drop(columns=['Adj Close', 'High', 'Low', 'Open', 'DTB3', 'DAAA', 'SMA_5',
          'SMA_20', 'EMA_5', 'EMA_20', 'Month', 'Momentum_5', 'Momentum_4',
          'Momentum_2', 'Next Month Return', 'DGS10', 'DTB6', 'DGS5', 'Daily Range',
          'Percentage Change', 'TE2', 'TE3', 'TE5'], inplace=True)

get_correlation_matrix()

# %% [markdown]
# ### Remove Outliers

# %%
data['Z-Score'] = (data['Daily Return'] - data['Daily Return'].mean()) / data['Daily Return'].std()
data_no_outliers = data[(data['Z-Score'] >= -2.5) & (data['Z-Score'] <= 2.5)]
data_no_outliers = data_no_outliers.drop(columns=['Z-Score'])
data = data_no_outliers

data['Z-Score'] = (data['Next Quarter Return'] - data['Next Quarter Return'].mean()) / data['Next Quarter Return'].std()
data_no_outliers = data[(data['Z-Score'] >= -2.5) & (data['Z-Score'] <= 2.5)]
data_no_outliers = data_no_outliers.drop(columns=['Z-Score'])
data = data_no_outliers

# %% [markdown]
# ### Drop missing values

# %%
# Ensure there are no missing values before defining features and target
data = data.dropna()  # Ensure there are no missing values
data.drop(columns="Symbol")
data = data.round(4)

# %% [markdown]
# ### Define targets and features

# %%
target = 'Next Day Return'
features = list(data.columns)
features.remove(target)
features.remove('Date')
features.remove('Symbol')
features
# %% [markdown]
# ### Split test and train data

# %%
input_data = data[features]
target_variable = data[target]

# Split the data into training and testing sets
input_train, input_test, target_train, target_test = train_test_split(input_data, target_variable, test_size=0.2, random_state=42, shuffle=False)

# %% [markdown]
# Reusable functions

# %%
from typing import Union
def predict(model, test_data, test_target, target_feature='Next Day Return') -> Union[float, float]:
    """A reusable function to test data and a model and predicts the target feature. It then prints our the following metrics:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Coefficient of Determination (R2)

    Args:
        model (_type_): the model used to predict
        test_data (_type_): the test data to predict
        test_target (_type_): the target data to predict
        target_feature (str, optional): The name of the target feature. Defaults to 'Next Day Return'.
    Returns:
        float, float, float: The Mean Absolute Error, Root Mean Squared Error and Coefficient of Determination (R2)
    """
    # Predict on the test set
    prediction = model.predict(test_data)

    mae = mean_absolute_error(target_test, prediction)
    rmse = np.sqrt(mean_squared_error(test_target, prediction))
    r2 = r2_score(test_target, prediction)

    plt.figure(figsize=(10, 6))
    plt.plot(test_target.index, test_target, label='Actual', alpha=0.8)
    plt.plot(test_target.index, prediction, label='Predicted', alpha=0.8)
    plt.legend()
    plt.title(f'Actual vs Predicted {target_feature}')
    plt.xlabel('Index')
    plt.ylabel(target_feature)
    plt.show()

    return mae, rmse, r2

# Computes a hash of the dataset so we can detect if the dataset has changed
def compute_data_hash(data):
    """Compute a hash for the dataset."""
    return hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values).hexdigest()

def has_data_changed(hash_file_path, current_hash):
    try:
        with open(hash_file_path, 'r') as f:
            stored_hash = f.read().strip()
        return stored_hash != current_hash
    except FileNotFoundError:
        return True



# %% [markdown]
# ## XGBoost

# %%
model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            objective='reg:squarederror',
            random_state=42,
            booster="gbtree"
        )

# Fit the model
model.fit(input_train, target_train, verbose=True)


# %% [markdown]
# ## Feature Importance
# ### Feature importance analysis

# %%
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
print(feature_importance_df)

plt.figure(figsize=(10, 8))
sns.barplot(
    x=feature_importance_df['Importance'], 
    y=feature_importance_df['Feature'], 
    palette="viridis"
)
plt.title('Feature Importance in the model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# %% [markdown]
# ### Permutation importance

# %%
from sklearn.inspection import permutation_importance

X = pd.DataFrame(input_test, columns=features)
result = permutation_importance(model, input_test, target_test, n_repeats=10, random_state=42)
for i in result.importances_mean.argsort()[::-1]:
    print(f"{X.columns[i]:<10} Importance: {result.importances_mean[i]:.4f}")


# %%
# Predict on the test set
unoptimized_mae, unoptimized_rmse, unoptimized_r2 = predict(model, input_test, target_test)
print("MAE:", unoptimized_mae)
print("RMSE:", unoptimized_rmse)
print("R2:", unoptimized_r2)

# %% [markdown]
# ### Randomized Hyperparameter tunning
#
# Uses RandomizedSearchCV to find the best hyperparameters for the XGBoost model.

# %%
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'n_estimators': randint(100, 1001), # extended range for more exploration
    'max_depth': randint(2, 11), # shallower trees
    'learning_rate': uniform(0.01, 0.19), # focus on smaller values for better generalization
    'subsample': uniform(0.6, 0.4), # smaller values to prevent overfitting
    'colsample_bytree': uniform(0.5, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0.1, 4)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=0,
    n_jobs=-1,
    n_iter=200,
    random_state=42,
    refit=True, # ---> This will refit the model, so we don't have to do it manually
    return_train_score=False,
    error_score='raise'
)

data_hash = compute_data_hash(data)
if has_data_changed('../datasets/data_hash.txt', data_hash):
    print("Data has changed or no file found. Refitting the model.")
    random_search.fit(input_train, target_train)
    joblib.dump(random_search, '../models/random_search.pkl')
    with open('../datasets/data_hash.txt', 'w') as f:
        f.write(data_hash)
    print("RandomizedSearchCV saved to disk.")
else:
    random_search = joblib.load('../models/random_search.pkl')
    print("Loaded RandomizedSearchCV from disk.")

print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", np.sqrt(-random_search.best_score_))

best_model = random_search.best_estimator_
y_pred = best_model.predict(input_test)
test_rmse = np.sqrt(mean_squared_error(target_test, y_pred))
print("Test RMSE:", test_rmse)

# Predict on the test set
optimized_mae, optimized_rsme, optimized_r2 = predict(best_model, input_test, target_test)
print(f"MAE: {optimized_mae:.4f}, diff: {optimized_mae - unoptimized_mae}")
print(f"RMSE: {optimized_rsme:.4f}, diff: {optimized_rsme - unoptimized_rmse}")
print(f"R2: {optimized_r2:.4f}, diff: {optimized_r2 - unoptimized_r2}")


# %% [markdown]
# ### Save model to file

# %%
def save_to_file(model, filepath: str):
    model_file = os.path.abspath(filepath)
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")


save_to_file(model, '../models/stocks_model.pkl')
