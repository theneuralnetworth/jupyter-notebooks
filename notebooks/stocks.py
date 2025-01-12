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
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import numpy as np
import joblib

# COMMON CONSTANTS
CUTOFF_DATE = "2004-11-26"
STOCKS_DATASET = os.path.join(os.getcwd(), 'datasets', 'stocks.csv')

# Load environment variables
load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
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
data['Next Week Return'] = (data['Close'] / data['Close'].shift(-7)) * 100

#data.info()

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
# ## Feature importance analysis
# ### Correlation Matrix (numeric features only)

# %%
# Correlation Matrix
def get_correlation_matrix():
    numeric_columns = data.select_dtypes(include=[float, int])
    correlation_matrix = numeric_columns.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix')
    plt.show()

get_correlation_matrix()

# %% [markdown]
# #### High Correlation features

# %%
threshold = 0.8
numeric_columns = data.select_dtypes(include=[float, int])
corr_matrix = numeric_columns.corr()

# Drop highly correlated features
high_corr = corr_matrix.abs() > threshold
upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1)
# Identify highly correlated features
to_drop = []
for i in range(corr_matrix.shape[0]):
    for j in range(i + 1, corr_matrix.shape[1]):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            # Add the column with the higher index to the drop list
            to_drop.append(corr_matrix.columns[j])

to_drop = list(set(to_drop))
to_drop


# %% [markdown]
# ### Remove Outliers

# %%
data['Z-Score'] = (data['Daily Return'] - data['Daily Return'].mean()) / data['Daily Return'].std()
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
features = [
    'Adj Close', 'Close', 'High', 'Low',
    'Open', 'Direction', 'Quarter', 'week_number',
    'week_day', 'Inflation', 'Daily Range',
    'Volume', 'Momentum_10', 'Volume Change',
    'Percentage Change', 'Last Week Return',
    'Daily Return', 'Sector_Communication Services',
    'Sector_Consumer Discretionary', 'Sector_Consumer Staples',
    'Sector_Energy', 'Sector_Financials', 'Sector_Health Care',
    'Sector_Industrials', 'Sector_Information Technology',
    'Sector_Materials', 'Sector_Real Estate', 'Sector_Utilities'
]
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
    - Squared Mean Squared Error
    - Coefficient of Determination (R2)

    Args:
        model (_type_): the model used to predict
        test_data (_type_): the test data to predict
        test_target (_type_): the target data to predict
        target_feature (str, optional): The name of the target feature. Defaults to 'Next Day Return'.
    Returns:
        float, float: The Squared Mean Squared Error and Coefficient of Determination (R2)
    """
    # Predict on the test set
    prediction = model.predict(test_data)

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

    return rmse, r2


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
feature_importance_df

# %% [markdown]
# ### Permutation importance

# %%
from sklearn.inspection import permutation_importance

X = pd.DataFrame(input_test, columns=features)
result = permutation_importance(model, input_test, target_test, n_repeats=10, random_state=42)
for i in result.importances_mean.argsort()[::-1]:
    print(f"{X.columns[i]:<10} Importance: {result.importances_mean[i]:.4f}")


# %% [markdown]
# ### Recursive Feature Elimination

# %%
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

rfe = RFE(estimator=model, n_features_to_select=5)  # Select top 5 features
rfe.fit(input_train, target_train)

# Get ranking of features
print("Ranking")
ranking = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_}).sort_values(by='Ranking')
print(ranking)
print("Support")
support = pd.DataFrame({'Feature': X.columns, 'Support': rfe.support_}).sort_values(by='Support', ascending=False)
print(support)

# Transforms the dataset to include only selected features
input_train_selected = rfe.transform(input_train)
input_test_selected = rfe.transform(input_test)


# %%
# Predict on the test set
unoptimized_rmse, unoptimized_r2 = predict(model, input_test, target_test)
print("RMSE:", unoptimized_rmse)
print("R2:", unoptimized_r2)


# %% [markdown]
# ### Save model to file

# %%
def save_to_file(model, filepath: str):
    model_file = os.path.abspath(filepath)
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

save_to_file(model, '../models/stocks_model.pkl')

# %% [markdown]
# ### Randomized Hyperparameter tunning
#
# Uses RandomizedSearchCV to find the best hyperparameters for the XGBoost model.

# %%
from sklearn.model_selection import RandomizedSearchCV
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

random_search.fit(input_train, target_train)

print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", np.sqrt(-random_search.best_score_))

best_model = random_search.best_estimator_
y_pred = best_model.predict(input_test)
test_rmse = np.sqrt(mean_squared_error(target_test, y_pred))
print("Test RMSE:", test_rmse)

# %%
# Predict on the test set
optimized_rsme, optimized_r2 = predict(best_model, input_test, target_test)
print(f"RMSE: {optimized_rsme:.4f}, diff: {optimized_rsme - unoptimized_rmse}")
print(f"R2: {optimized_r2:.4f}, diff: {optimized_r2 - unoptimized_r2}")

# %%
