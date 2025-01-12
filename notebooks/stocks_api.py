from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from feature_calculator import get_stock_history, calculate_features_with_history

app = FastAPI()

class request_body(BaseModel):
    stock_symbol: str

# Load the model once when starting the API
model = joblib.load('./models/stocks_model.pkl')

# Define the correct feature order that the model expects
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

@app.post('/predict')
def predict(data: request_body):
    try:
        # Get historical data and sector for the stock
        historical_data, sector = get_stock_history(data.stock_symbol)
        
        # Calculate features using the historical data
        features = calculate_features_with_history(historical_data, sector)
        
        # Convert features to DataFrame with correct column order
        pred_df = pd.DataFrame([features])[feature_order]
        
        # Make prediction
        prediction = model.predict(pred_df)[0].astype(float)
        
        return {
            'symbol': data.stock_symbol,
            'predicted_return': prediction,
            'current_price': historical_data[-1]['Close'],
            'prediction_date': historical_data[-1]['Date']
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")