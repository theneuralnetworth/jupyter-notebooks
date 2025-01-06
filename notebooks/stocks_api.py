from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# TODO: Add all the required input features
class request_body(BaseModel):
    stock_symbol: str

model = joblib.load('./stocks_model.pkl')

@app.post('/predict')
def predict(data: request_body):
    input_features  = {
        'stock_symbol': data.stock_symbol
    }
    pred_df = pd.DataFrame(input_features, index=[1])

    prediction = model.predict(pred_df)[0].astype(float)
    
    return {'value': prediction}