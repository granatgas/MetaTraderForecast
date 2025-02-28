import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, Input # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

def train(training_set, date, lr, scale, epochs, momentum, optimizer, loss, file_name, architecture, cuda):
    try:
        # Data preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = scaler.fit_transform(np.array(training_set).reshape(-1, 1))

        X_train = []
        y_train = []
        for i in range(60, len(training_set_scaled)):
            X_train.append(training_set_scaled[i-60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Building the RNN
        model = Sequential()

        if architecture == 0:  # LSTM
            model.add(Input(shape=(X_train.shape[1], 1)))
            model.add(LSTM(units=50, return_sequences=True, recurrent_activation='sigmoid'))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True, recurrent_activation='sigmoid'))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True, recurrent_activation='sigmoid'))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, recurrent_activation='sigmoid'))
            model.add(Dropout(0.2))
        elif architecture == 1:  # GRU
            model.add(Input(shape=(X_train.shape[1], 1)))
            model.add(GRU(units=50, return_sequences=True, recurrent_activation='sigmoid'))
            model.add(Dropout(0.2))
            model.add(GRU(units=50, return_sequences=True, recurrent_activation='sigmoid'))
            model.add(Dropout(0.2))
            model.add(GRU(units=50, return_sequences=True, recurrent_activation='sigmoid'))
            model.add(Dropout(0.2))
            model.add(GRU(units=50, recurrent_activation='sigmoid'))
            model.add(Dropout(0.2))
        elif architecture == 2:  # Bidirectional LSTM
            model.add(Input(shape=(X_train.shape[1], 1)))
            model.add(Bidirectional(LSTM(units=50, return_sequences=True, recurrent_activation='sigmoid')))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(units=50, return_sequences=True, recurrent_activation='sigmoid')))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(units=50, return_sequences=True, recurrent_activation='sigmoid')))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(units=50, recurrent_activation='sigmoid')))
            model.add(Dropout(0.2))
        elif architecture == 3:  # Bidirectional GRU
            model.add(Input(shape=(X_train.shape[1], 1)))
            model.add(Bidirectional(GRU(units=50, return_sequences=True, recurrent_activation='sigmoid')))
            model.add(Dropout(0.2))
            model.add(Bidirectional(GRU(units=50, return_sequences=True, recurrent_activation='sigmoid')))
            model.add(Dropout(0.2))
            model.add(Bidirectional(GRU(units=50, return_sequences=True, recurrent_activation='sigmoid')))
            model.add(Dropout(0.2))
            model.add(Bidirectional(GRU(units=50, recurrent_activation='sigmoid')))
            model.add(Dropout(0.2))

        model.add(Dense(units=1))

        # Compiling the RNN
        if optimizer == 0:
            opt = tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=momentum)
        elif optimizer == 1:
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
        elif optimizer == 2:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 3:
            opt = tf.keras.optimizers.Adagrad(learning_rate=lr)

        if loss == 0:
            loss_function = 'mean_squared_error'
        elif loss == 1:
            loss_function = 'mean_absolute_error'

        model.compile(optimizer=opt, loss=loss_function)

        # Fitting the RNN to the Training set
        model.fit(X_train, y_train, epochs=epochs, batch_size=32)

        # Save the model
        model.save(file_name + '.keras')

        return "Training completed"
    except Exception as e:
        print(f"Exception in train function: {e}")
        raise

def test(testing_set, date, file_name):
    try:
        # Convert testing_set to a numpy array and check length
        testing_set_array = np.array(testing_set, dtype=float)
        if testing_set_array.shape[0] < 61:
            raise ValueError("Not enough data for testing. Need at least 61 points.")
            
        scaler = MinMaxScaler(feature_range=(0, 1))
        testing_set_scaled = scaler.fit_transform(testing_set_array.reshape(-1, 1))

        X_test = []
        y_test = []
        for i in range(60, len(testing_set_scaled)):
            X_test.append(testing_set_scaled[i-60:i, 0])
            y_test.append(testing_set_scaled[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Log shapes for debugging
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Load the model
        model = tf.keras.models.load_model(file_name + '.keras')

        # Making predictions
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        # Calculate metric using original testing_set_array starting at index 60
        mse = mean_squared_error(testing_set_array[60:], predicted_stock_price)
        r2 = r2_score(testing_set_array[60:], predicted_stock_price)

        return f"Test MSE: {mse}, R2: {r2}"
    except Exception as e:
        print(f"Exception in test function: {e}")
        raise

def evaluate(file_name, testing_set):
    try:
        # Convert testing_set to a numpy array and check length as in test()
        testing_set_array = np.array(testing_set, dtype=float)
        if testing_set_array.shape[0] < 61:
            raise ValueError("Not enough data for evaluation. Need at least 61 points.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        testing_set_scaled = scaler.fit_transform(testing_set_array.reshape(-1, 1))

        X_eval = []
        y_eval = []
        for i in range(60, len(testing_set_scaled)):
            X_eval.append(testing_set_scaled[i-60:i, 0])
            y_eval.append(testing_set_scaled[i, 0])
        X_eval, y_eval = np.array(X_eval), np.array(y_eval)
        X_eval = np.reshape(X_eval, (X_eval.shape[0], X_eval.shape[1], 1))

        # Load the model
        model = tf.keras.models.load_model(file_name + '.keras')

        # Evaluate the model on the evaluation dataset
        evaluation = model.evaluate(X_eval, y_eval, verbose=0)

        return f"Evaluation: {evaluation}"
    except Exception as e:
        print(f"Exception in evaluate function: {e}")
        raise

def predict(file_name, bars):
    try:
        # Load the model
        model = tf.keras.models.load_model(file_name + '.keras')
        
        # Create input data for prediction
        # We need sequence of 60 points to predict next 'bars' points
        # For now, using zeros as placeholder - you might want to use actual historical data
        X_pred = np.zeros((1, 60, 1))  # Shape: (1, timesteps, features)
        
        # Predict future bars
        predictions = model.predict(X_pred)
        
        # Generate multiple predictions if bars > 1
        result = []
        for _ in range(int(bars)):
            pred = float(predictions[0, 0])  # Convert to float for JSON serialization
            result.append(pred)
            
            # Shift the sequence and add the prediction for next iteration
            X_pred = np.roll(X_pred, -1)
            X_pred[0, -1, 0] = pred
            
            # Get next prediction
            predictions = model.predict(X_pred)
            
        return result
    except Exception as e:
        print(f"Exception in predict function: {e}")
        raise