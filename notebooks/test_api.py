import requests
import sys
import time

def test_prediction(symbol):
    base_url = "http://localhost:8000"
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to connect to API (attempt {attempt + 1}/{max_retries})...")
            health_response = requests.get(f"{base_url}/health")
            if health_response.status_code == 200:
                print("✓ API is running and healthy")
                break
            else:
                print(f"× API returned status code {health_response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("× Could not connect to API at http://localhost:8000")
            print("  Make sure to:")
            print("  1. Open a new terminal")
            print("  2. Navigate to the notebooks directory: cd notebooks")
            print("  3. Run the API server: python run_api.py")
            
            if attempt < max_retries - 1:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                print("× Failed to connect to API after multiple attempts")
                print("  Please make sure the API server is running")
                return
    
    # Make the prediction request
    try:
        print(f"\nMaking prediction request for {symbol}...")
        response = requests.post(
            f"{base_url}/predict",
            json={"stock_symbol": symbol}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Results:")
            print(f"Symbol: {result['symbol']}")
            print(f"Predicted Return: {result['predicted_return']:.2f}%")
            print(f"Current Price: ${result['current_price']:.2f}")
            print(f"Prediction Date: {result['prediction_date']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
        
    except Exception as e:
        print(f"Error making prediction request: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = "AAPL"
    
    test_prediction(symbol) 