import json
import build
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

def get_trading_signal(predictions, threshold=0.001):
    """
    Convert predictions to trading signals with probability
    threshold: minimum price change to trigger buy/sell (default 0.001 = 0.1%)
    """
    if len(predictions) < 2:
        return {"signal": "HOLD", "probability": 0.0}
    
    # Calculate predicted price change
    price_change = predictions[-1] - predictions[0]
    price_change_percent = price_change / predictions[0]
    
    # Calculate probability based on price change magnitude
    probability = min(abs(price_change_percent) * 100, 100.0)  # Convert to percentage, cap at 100%
    
    if abs(price_change_percent) < threshold:
        return {"signal": "HOLD", "probability": probability}
    elif price_change_percent > 0:
        return {"signal": "BUY", "probability": probability}
    else:
        return {"signal": "SELL", "probability": probability}

def train_test_model(input_data):
    print(f"Received message: {input_data}")
    
    try:
        # Disable GPU usage entirely
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        file_name = input_data['FileName']
        if not input_data['Train']:
            num_bars = int(input_data['Bars'])
            pred = build.predict(file_name, num_bars)
            print(f"Predictions for next {num_bars} bars:", pred)
            signal_info = get_trading_signal(pred)
            responseJSON = {
                'Pred': pred,
                'Signal': signal_info['signal'],
                'Probability': f"{signal_info['probability']:.2f}%"
            }
            return json.dumps(responseJSON) + "\r\n"
                    
        data = input_data['Data']
        date = input_data['Time']    
        
        testSize = int(input_data['TestingPart'] / 100 * len(data))
        trainSize = len(data) - testSize
        
        # Train model using CPU only
        train = build.train(
            training_set=data[:trainSize], 
            date=date[:trainSize], 
            lr=input_data['LearningRate'],
            scale=input_data['Scale'], 
            epochs=input_data['Epochs'], 
            momentum=input_data['Momentum'],
            optimizer=input_data['Optimizer'], 
            loss=input_data['Loss'], 
            file_name=file_name,
            architecture=input_data['Architecture'], 
            cuda=False
        )
        print(train)

        test = build.test(
            testing_set=data[trainSize:], 
            date=date[trainSize:], 
            file_name=input_data['FileName']
        )
        print(test)
        
        evaluate = build.evaluate(file_name, data[trainSize:])
        print(evaluate)

        pred = build.predict(file_name, input_data['Bars'])
        print(pred)
        signal_info = get_trading_signal(pred)  # Get both signal and probability

        responseJSON = {
            'Eval': evaluate,
            'Pred': pred,
            'Signal': signal_info['signal'],
            'Probability': f"{signal_info['probability']:.2f}%"  # Add probability to response
        }

        return json.dumps(responseJSON) + "\r\n"
    except Exception as e:
        print(f"Error in train_test_model: {e}")
        raise

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            input_data = json.loads(post_data)
            response = train_test_model(input_data)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Invalid JSON')
        except BrokenPipeError:
            print("BrokenPipeError: Client closed the connection before the response was sent.")
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            try:
                self.wfile.write(str(e).encode('utf-8'))
            except BrokenPipeError:
                print("BrokenPipeError: Client closed the connection before the error message was sent.")
            print(f"Exception: {e}")

def run_server(server_class=HTTPServer, handler_class=RequestHandler, port=9090):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    print('Server started. Waiting for client...')
    server_thread.join()