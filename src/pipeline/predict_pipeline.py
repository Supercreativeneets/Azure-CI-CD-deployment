import os
import sys

import numpy as np 
import pandas as pd
from tensorflow import keras

from src.exception import CustomException
from src.utils import load_object

from dataclasses import dataclass

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,DATE):
        try:
                          
            test_path = os.path.join('artifacts','test.csv')
            df = pd.read_csv(test_path, index_col=None, header=0)

            pred_date = pd.to_datetime(DATE)

            start_date = pred_date - pd.Timedelta(hours=24)

            df['date'] = pd.to_datetime(df['date'])  # Convert 'date' column to Timestamp

            extracted_df = df[(df['date'] >= start_date) & (df['date'] < pred_date)] 

            pred_df = extracted_df.drop(columns=['date','No','year','month','day','hour','SO2', 'O3','station'],axis=1)

            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            target_scaler_path = os.path.join('artifacts','target_scaler.pkl')
            feature_scaler_path = os.path.join('artifacts','feature_scaler.pkl')
            

            preprocessor = load_object(file_path=preprocessor_path)
            target_scaler = load_object(file_path=target_scaler_path)
            feature_scaler = load_object(file_path=feature_scaler_path)
            

            pred_array = preprocessor.transform(pred_df)

            target_pred_array = pred_array[:, :1]
            feature_pred_array = pred_array[:, 1:]

            target_scaled_array = target_scaler.transform(target_pred_array)
            feature_scaled_array = feature_scaler.transform(feature_pred_array)

            scaled_pred_arr = np.concatenate((target_scaled_array, feature_scaled_array), axis=1)

            final_pred_arr = scaled_pred_arr.reshape(1, 24, 22)
            
            model_path = os.path.join('artifacts','model.keras')
            print("Model path:", model_path)
            print("Does model file exist?", os.path.exists(model_path))
            model = keras.models.load_model(model_path)
           
            pred = model.predict(final_pred_arr)

            inv_pred = target_scaler.inverse_transform(pred)

            return inv_pred
            
        except Exception as e:
            raise CustomException(e,sys)