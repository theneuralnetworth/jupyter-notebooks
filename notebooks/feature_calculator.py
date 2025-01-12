import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

def get_inflation_data():
    """Fetch inflation data from Alpha Vantage or local file"""
    file_path = 'datasets/economic_indicators/inflation.json'
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as fp:
            data_inflation = json.load(fp)
    else:
        url = f"https://www.alphavantage.co/query?function=INFLATION&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        data_inflation = data['data']
        
        # Save for future use
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as fp:
            json.dump(data_inflation, fp)
    
    # Create mapping of inflation values by year
    inflation_mapping = {item['date'][:4]: float(item['value']) for item in data_inflation}
    inflation_mapping['2024'] = 2.6  # Current year default value
    
    return inflation_mapping

def calculate_features(stock_data, sector):
    """
    Calculate all necessary features for the prediction model
    
    Parameters:
    stock_data (dict): Dictionary with keys 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    sector (str): The sector of the stock (e.g., 'Information Technology')
    
    Returns:
    dict: Dictionary containing all required features for the prediction model
    """
    # Get economic indicators
    inflation_mapping = get_inflation_data()
    
    # Convert date to datetime if it's not already
    date = pd.to_datetime(stock_data.get('Date', datetime.now()))
    
    # Calculate basic features
    features = {
        'Adj Close': float(stock_data['Adj Close']),
        'Close': float(stock_data['Close']),
        'High': float(stock_data['High']),
        'Low': float(stock_data['Low']),
        'Open': float(stock_data['Open']),
        'Volume': float(stock_data['Volume']),
        
        # Time-based features
        'Quarter': date.quarter,
        'week_number': date.isocalendar()[1],
        'week_day': date.weekday(),
        
        # Price-based features
        'Daily Range': float(stock_data['High']) - float(stock_data['Low']),
        'Direction': 1 if float(stock_data['Close']) > float(stock_data['Open']) else 0,
        'Percentage Change': ((float(stock_data['Close']) - float(stock_data['Open'])) / float(stock_data['Open'])) * 100,
        
        # Historical features (if available)
        'Momentum_10': 0.0,  # This needs historical data
        'Volume Change': 0.0,  # This needs previous day's volume
        'Last Week Return': 0.0,  # This needs historical data
        'Daily Return': 0.0,  # This needs previous day's close
    }
    
    # Add inflation
    year = str(date.year)
    features['Inflation'] = inflation_mapping.get(year, inflation_mapping['2024'])
    
    # Add sector features (one-hot encoding)
    sectors = [
        'Communication Services', 'Consumer Discretionary', 'Consumer Staples',
        'Energy', 'Financials', 'Health Care', 'Industrials',
        'Information Technology', 'Materials', 'Real Estate', 'Utilities'
    ]
    
    for s in sectors:
        features[f'Sector_{s}'] = 1 if s == sector else 0
    
    return features

def calculate_features_with_history(stock_data_list, sector):
    """
    Calculate features including those that require historical data
    
    Parameters:
    stock_data_list (list): List of dictionaries containing daily stock data, ordered by date
    sector (str): The sector of the stock
    
    Returns:
    dict: Dictionary containing all required features for the prediction model
    """
    if not stock_data_list:
        raise ValueError("Stock data list is empty")
    
    # Get the most recent day's data
    current_data = stock_data_list[-1]
    features = calculate_features(current_data, sector)
    
    # Calculate features that require historical data
    if len(stock_data_list) > 10:
        features['Momentum_10'] = float(current_data['Close']) - float(stock_data_list[-11]['Close'])
    
    if len(stock_data_list) > 1:
        prev_volume = float(stock_data_list[-2]['Volume'])
        features['Volume Change'] = ((float(current_data['Volume']) - prev_volume) / prev_volume) * 100
        
        prev_close = float(stock_data_list[-2]['Close'])
        features['Daily Return'] = ((float(current_data['Close']) - prev_close) / prev_close) * 100
    
    if len(stock_data_list) > 7:
        week_ago_close = float(stock_data_list[-8]['Close'])
        features['Last Week Return'] = ((float(current_data['Close']) - week_ago_close) / week_ago_close) * 100
    
    return features

def get_stock_history(symbol):
    """
    Get the last two days of stock data for a given symbol from stocks.csv
    
    Parameters:
    symbol (str): The stock symbol (e.g., 'AAPL')
    
    Returns:
    tuple: (list of historical data dicts, sector string)
    """
    # Read the CSV file
    df = pd.read_csv('datasets/stocks.csv')
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for the specific symbol and sort by date
    stock_data = df[df['Symbol'] == symbol].sort_values('Date')
    
    if len(stock_data) == 0:
        raise ValueError(f"No data found for symbol {symbol}")
    
    # Get the sector
    sector = stock_data['Sector'].iloc[0]
    
    # Get the last two days of data
    # last_two_days = stock_data.tail(2)
    
    # Convert to list of dictionaries format
    historical_data = []
    # for _, row in last_two_days.iterrows():
    for _, row in stock_data.iterrows():
        historical_data.append({
            'Date': row['Date'].strftime('%Y-%m-%d'),
            'Open': float(row['Open']),
            'High': float(row['High']),
            'Low': float(row['Low']),
            'Close': float(row['Close']),
            'Adj Close': float(row['Adj Close']),
            'Volume': float(row['Volume'])
        })
    
    return historical_data, sector

# Example usage
if __name__ == "__main__":
    try:
        # Get historical data for a stock (e.g., 'A' for Agilent Technologies)
        historical_data, sector = get_stock_history('A')
        print(f"\nHistorical data for stock:")
        for day in historical_data:
            print(f"Date: {day['Date']}")
            print(f"Close: {day['Close']}")
            print(f"Volume: {day['Volume']}")
        print(f"Sector: {sector}")
        
        # Calculate features with the retrieved data
        features = calculate_features_with_history(historical_data, sector)
        print("\nCalculated features:")
        print(features)
        
    except ValueError as e:
        print(f"Error: {e}") 