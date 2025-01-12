import uvicorn

if __name__ == "__main__":
    uvicorn.run("stocks_api:app", host="0.0.0.0", port=8000, reload=True)


# curl -X POST "http://localhost:8000/predict" \
#      -H "Content-Type: application/json" \
#      -d '{"stock_symbol": "AAPL"}'
