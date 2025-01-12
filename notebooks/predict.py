import joblib
import pandas as pd
from feature_calculator import calculate_features_with_history, get_stock_history

def predict_stock_return(symbol):
    """
    Predict stock return using saved model
    
    Parameters:
    symbol (str): The stock symbol (e.g., 'AAPL')
    
    Returns:
    float: Predicted return percentage
    """
    # Get historical data and sector for the stock
    historical_data, sector = get_stock_history(symbol)
    
    # Calculate features
    features = calculate_features_with_history(historical_data, sector)
    
    # Load model
    model = joblib.load('../models/stocks_model.pkl')
    
    # Define the correct feature order
    feature_order = [
        'Adj Close', 'Close', 'High', 'Low', 'Open', 'Direction', 'Quarter',
        'week_number', 'week_day', 'Inflation', 'Daily Range',
        'Volume', 'Momentum_10', 'Volume Change', 'Percentage Change',
        'Last Week Return', 'Daily Return', 'Sector_Communication Services',
        'Sector_Consumer Discretionary', 'Sector_Consumer Staples', 'Sector_Energy',
        'Sector_Financials', 'Sector_Health Care', 'Sector_Industrials',
        'Sector_Information Technology', 'Sector_Materials', 'Sector_Real Estate',
        'Sector_Utilities'
    ]
    
    # Convert to DataFrame and reorder columns
    df = pd.DataFrame([features])
    df = df[feature_order]
    
    # Make prediction
    prediction = model.predict(df)
    
    return prediction[0]

# Example usage
if __name__ == "__main__":
    try:
        # Predict return for APPLE (AAPL)
        symbol = 'AAPL'
        prediction = predict_stock_return(symbol)
        print(f"\nPredicted next day return for {symbol}: {prediction:.2f}%")
    except ValueError as e:
        print(f"Error: {e}") 