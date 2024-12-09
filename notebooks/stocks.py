# %% [markdown]
# # Stock Price Prediction System
# #### Team 1: Neural Networth

# %%
# Imports
import os
import pandas as pd

# COMMON CONSTANTS
CUTOFF_DATE = "2004-11-26"
STOCKS_DATASET = "datasets/stocks.csv"

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

# Filter to only SP500 stocks
sp500_symbols = pd.read_csv('datasets/sp500.csv')['Symbol']
def is_sp500(filename):
    """Check if the filename is a SP500 stock"""
    symbol = os.path.basename(filename).split('.')[0]
    return symbol in sp500_symbols.values

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
df.shape

# Save the dataframe
print("Saving the dataset to ", STOCKS_DATASET)
if not os.path.exists(STOCKS_DATASET) or input(f"Overwrite {STOCKS_DATASET}? [y/n] ").lower() == 'y':
    df.to_csv(STOCKS_DATASET, index=False)


# %% [markdown]
# ## 2. Exploratory Data Analysis
#
# ...

# %% [markdown]
# ### 3. Feature Engineering
#
# 1. CPI, inflation, interest rates, etc.
# 1. growth past 7 days, month, year
# 1. growth next 7 days, month, year
# 1. sector/industry
#
