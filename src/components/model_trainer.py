import os
import sys

import numpy as np
import tensorflow as tf 
import keras

from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras.optimizers import Adam
from keras import callbacks


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.keras")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,val_array,test_array):
        try:
            logging.info("Convert series to supervised inputs and outputs")
            
            # Convert series to supervised inputs and outputs
            
            def to_supervised(data, n_input=24, n_out=1):
                # flatten data
                data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
                X, y = list(), list()
                in_start = 0
                
                # step over the entire history one time step at a time
                for _ in range(len(data)):
                    
                    # define the end of the input sequence
                    in_end = in_start + n_input
                    out_end = in_end + n_out
                    
                    # ensure we have enough data for this instance
                    if out_end <= len(data):
                        X.append(data[in_start:in_end, :])
                        y.append(data[in_end:out_end, 0])

                    # move along one time step
                    in_start += 1

                return np.array(X), np.array(y)
            
            # Prepare data to supervised input and output
            train_X, train_y = to_supervised(train_array, n_input=24)
            val_X, val_y = to_supervised(val_array, n_input =24)
            test_X, test_y = to_supervised(test_array, n_input =24)
            
            # fit a model
            def model_fit(train_X, train_y, val_X, val_y, batch_size):
                
                # Build the model
                tf.random.set_seed(7)
                np.random.seed(7)
                n_input, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
                                
                # Define the model
                model = Sequential()
                model.add(LSTM(200, return_sequences=False, activation='relu', input_shape=(n_input, n_features)))
                model.add(Dropout(0.1))
                model.add(Dense(100, activation='relu'))
                model.add(Dropout(0.1))
                model.add(Dense(50, activation='relu'))
                model.add(Dropout(0.1))
                model.add(Dense(n_outputs))

                # Compile the model
                model.compile(loss = 'mse', optimizer= Adam(learning_rate = 0.0005))

                es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4,
                                                      patience=50, restore_best_weights=True)

                
                # fit model
                model.fit(train_X, train_y, epochs=200, batch_size=batch_size, verbose=1, 
                          validation_data = (val_X,val_y),callbacks=[es_callback])

                return model
            
            def evaluate_models(train_X, train_y, val_X, val_y, test_X, test_y, batch_size):
                
                key = "batch size :" + str(batch_size)

                model = model_fit(train_X, train_y, val_X, val_y, batch_size)

                # make forecast
                yhat = model.predict(test_X)

                # actual observation
                test_y = test_y.reshape(-1,1)
                
                # get the target scaler object 
                target_scaler_path=os.path.join('artifacts',"target_scaler.pkl")
                target_scaler = load_object(file_path = target_scaler_path)
                
                # invert scaling for actual
                inv_test_y = target_scaler.inverse_transform(test_y)
                inv_test_y = inv_test_y.reshape(-1)
                
                # invert scaling for predictions
                inv_yhat = target_scaler.inverse_transform(yhat)
                inv_yhat = inv_yhat.reshape(-1)
                
                # estimate prediction error
                rmse = sqrt(mean_squared_error(inv_test_y,inv_yhat))
                mae = mean_absolute_error(inv_test_y, inv_yhat)
                R2=r2_score(inv_test_y, inv_yhat)
                
                print(f"{key}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {R2:.3f}")
    
                return (key, rmse, mae, R2, model)

        
            def grid_search(train_X, train_y, val_X, val_y, test_X, test_y, n_batch):
                results = []

                for batch_size in n_batch:
                    key, rmse, mae, R2, model = evaluate_models(train_X, train_y, val_X, val_y, test_X, test_y, batch_size)
                    results.append((key, rmse, mae, R2, model))
        

                # Sort results by RMSE in ascending order
                results.sort(key=lambda x: x[1])

                best_model_key, min_rmse, _, _, best_model = results[0]

                return best_model, min_rmse, results

            # Call grid search
            n_batch = [12, 24, 48]
            
            best_model, min_rmse, all_results = grid_search(train_X, train_y, val_X, val_y, test_X, test_y, n_batch)
            print("Best batch size:", min_rmse)
            
            # Save the best model
            best_model.save(self.model_trainer_config.trained_model_file_path)

            # Print all results sorted by RMSE
            print("\nAll Results Sorted by RMSE:")
            for key, rmse, mae, R2, model in all_results:
                print(f"{key}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {R2:.3f}")
                
            
            logging.info(f"Best model")

                       
        except Exception as e:
            raise CustomException(e,sys)
            
            
            
            