import uvicorn
import os
import sys

if __name__ == "__main__":
    # Get the absolute path to the notebooks directory
    notebooks_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the notebooks directory to Python path so it can find the stocks_api module
    sys.path.append(notebooks_dir)
    
    # Change to the notebooks directory to ensure relative paths work correctly
    os.chdir(notebooks_dir)
    
    print(f"Starting API server from directory: {notebooks_dir}")
    print("Make sure models/stocks_model.pkl exists in the parent directory")
    
    # Run the FastAPI application with explicit configuration
    config = uvicorn.Config(
        "stocks_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1
    )
    server = uvicorn.Server(config)
    server.run()


# curl -X POST "http://localhost:8000/predict" \
#      -H "Content-Type: application/json" \
#      -d '{"stock_symbol": "AAPL"}'
