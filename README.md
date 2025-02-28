## Components
- `server.py`: HTTP server handling MT4 requests
- `build.py`: Neural network model building and training
- `ForecastExpert.mq4`: MetaTrader 4 Expert Advisor

## Requirements
- Python 3.12+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- MetaTrader 4

## Setup
1. Install Python dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn
```

2. Start the prediction server:
```bash
python server.py
```

3. Install the Expert Advisor in MetaTrader 4:
   - Copy `ForecastExpert.mq4` to your MT4 Experts folder
   - Compile the EA in MetaTrader Editor
   - Add the EA to your chart

## Usage
1. The EA will automatically connect to the prediction server
2. Training data is sent to the server for model training
3. Real-time predictions are displayed on the chart
4. Trading signals appear as arrows:
   - Green Up Arrow: Buy signal
   - Red Down Arrow: Sell signal
   - Yellow Circle: Hold signal

## Configuration
Adjust the following parameters in the EA:
- Training period
- Prediction bars
- Model architecture (LSTM/GRU)
- Learning rate
- Training/testing split

## Author
GranatGas (granat.gas@gmail.com)

# MetaTrader Forecast

An AI-powered forex prediction system integrating deep learning with MetaTrader 4.

## Features
- Real-time price predictions using LSTM/GRU networks
- Trading signals (Buy/Sell/Hold) with confidence levels
- Visual indicators with customizable arrows
- REST API server for MT4 communication
