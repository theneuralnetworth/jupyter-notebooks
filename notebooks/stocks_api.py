from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from feature_calculator import get_stock_history, calculate_features_with_history
import os

# Create FastAPI app
app = FastAPI()

# Get the base directory (parent of notebooks)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the request body model
class StockRequest(BaseModel):
    stock_symbol: str

# Load the model at startup
try:
    print("Loading model...")
    model_path = os.path.join('..', 'models', 'stocks_model.pkl')
    print(f"Looking for model at: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Define paths for data files
STOCKS_CSV_PATH = os.path.join('datasets', 'stocks.csv')
print(f"Stocks CSV path: {STOCKS_CSV_PATH}")
print(f"File exists: {os.path.exists(STOCKS_CSV_PATH)}")

# Define the feature order
feature_order = [
    'Adj Close', 'Close', 'High', 'Low', 'Open', 'Direction', 'Month', 'Quarter',
    'week_number', 'week_day', 'Inflation', 'Daily Range', 'Volume',
    'Momentum_2', 'Momentum_3', 'Momentum_4', 'Momentum_5',
    'Volume Change', 'Percentage Change', 'Last Week Return', 'Daily Return',
    'Sector_Communication Services', 'Sector_Consumer Discretionary',
    'Sector_Consumer Staples', 'Sector_Energy', 'Sector_Financials',
    'Sector_Health Care', 'Sector_Industrials', 'Sector_Information Technology',
    'Sector_Materials', 'Sector_Real Estate', 'Sector_Utilities'
]

@app.get("/test")
def test():
    return {"message": "API is working!"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(request: StockRequest):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        # Get the stock symbol from the request body
        symbol = request.stock_symbol
        print(f"Processing prediction request for {symbol}")
        
        # Get historical data and sector
        historical_data, sector = get_stock_history(symbol)
        print(f"Retrieved historical data for {symbol}, sector: {sector}")
        
        # Calculate features
        features = calculate_features_with_history(historical_data, sector)
        print("Features calculated successfully")
        
        # Prepare features for prediction
        pred_df = pd.DataFrame([features])[feature_order]
        
        # Make prediction
        prediction = model.predict(pred_df)[0].astype(float)
        print(f"Prediction made: {prediction:.2f}%")
        
        return {
            "symbol": symbol,
            "predicted_return": float(prediction),
            "current_price": float(historical_data[-1]['Close']),
            "prediction_date": historical_data[-1]['Date'],
            "sector": sector
        }
        
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))