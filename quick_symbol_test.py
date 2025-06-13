import json
import websocket
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_symbol(symbol_name, deriv_symbol):
    """Quick test of a single symbol."""
    api_token = 'V4FOmNuhORnxbHT'
    ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    
    result = {'success': False, 'data_count': 0}
    ws = None
    is_connected = False
    
    def on_open(ws):
        nonlocal is_connected
        logger.info(f"Connected for {symbol_name}")
        is_connected = True
        auth_request = {"authorize": api_token, "req_id": 1}
        ws.send(json.dumps(auth_request))
    
    def on_message(ws, message):
        nonlocal result
        try:
            data = json.loads(message)
            
            if 'authorize' in data and 'error' not in data:
                logger.info(f"Authorized for {symbol_name}")
                request = {
                    "ticks_history": deriv_symbol,
                    "count": 10,
                    "end": "latest",
                    "style": "candles",
                    "granularity": 60,
                    "req_id": 2
                }
                ws.send(json.dumps(request))
            
            elif 'candles' in data:
                candles = data['candles']
                if candles:
                    result['success'] = True
                    result['data_count'] = len(candles)
                    result['latest_price'] = candles[-1]['close']
                    logger.info(f"✓ {symbol_name} ({deriv_symbol}): {len(candles)} candles, price: {candles[-1]['close']}")
                else:
                    logger.warning(f"✗ {symbol_name}: No candles received")
                ws.close()
            
            elif 'error' in data:
                logger.error(f"✗ {symbol_name}: {data['error']['message']}")
                ws.close()
                
        except Exception as e:
            logger.error(f"Error for {symbol_name}: {str(e)}")
    
    def on_error(ws, error):
        logger.error(f"WebSocket error for {symbol_name}: {str(error)}")
    
    def on_close(ws, close_status_code, close_msg):
        nonlocal is_connected
        is_connected = False
    
    try:
        ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, 
                                  on_error=on_error, on_close=on_close)
        
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for result
        timeout = 15
        while timeout > 0 and not result['success'] and is_connected:
            time.sleep(0.5)
            timeout -= 0.5
        
        if ws:
            ws.close()
        
    except Exception as e:
        logger.error(f"Connection error for {symbol_name}: {str(e)}")
    
    return result

def main():
    """Test all four symbols quickly."""
    symbols_to_test = [
        ('crash300', 'CRASH300N'),
        ('boom300', 'BOOM300N'),
        ('crash1000', 'CRASH1000'),
        ('boom1000', 'BOOM1000')
    ]
    
    results = {}
    
    for symbol_name, deriv_symbol in symbols_to_test:
        logger.info(f"\nTesting {symbol_name}...")
        result = test_symbol(symbol_name, deriv_symbol)
        results[symbol_name] = result
        time.sleep(2)  # Brief pause between tests
    
    print("\n=== FINAL SYMBOL TEST RESULTS ===")
    working_symbols = []
    for symbol, result in results.items():
        status = "✓ WORKING" if result['success'] else "✗ FAILED"
        print(f"{symbol.upper()}: {status}")
        if result['success']:
            working_symbols.append(symbol)
            print(f"  Data points: {result['data_count']}")
            print(f"  Latest price: {result.get('latest_price', 'N/A')}")
    
    print(f"\nWorking symbols: {', '.join(working_symbols)}")
    return results

if __name__ == "__main__":
    main()