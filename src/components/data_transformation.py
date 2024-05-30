import sys
import os

import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
        
class TargetScalerConfig:
    target_scaler_obj_file_path=os.path.join('artifacts',"target_scaler.pkl")
    
class FeatureScalerConfig:
    feature_scaler_obj_file_path=os.path.join('artifacts',"feature_scaler.pkl")
    
class InterpolateImputer(BaseEstimator, TransformerMixin):
                def __init__(self, method='linear', limit_direction='forward', axis=0):
                    self.method = method
                    self.limit_direction = limit_direction
                    self.axis = axis
                def fit(self, X, y=None):
                    return self  # Nothing to do in fit
    
                def transform(self, X):
                    return X.interpolate(method=self.method, limit_direction=self.limit_direction, axis=self.axis)
            
class WindDirectionEncoder(BaseEstimator, TransformerMixin):
                def __init__(self, angle_mapping):
                    self.angle_mapping = angle_mapping
                    
                def fit(self, X, y=None):
                    return self  # Nothing to do in fit
    
                def transform(self, X):
                    # Mapping wind direction categories to angles
                    angles = np.array([self.angle_mapping[direction] for direction in X.squeeze()])
                    sin_Theta = np.sin(np.radians(angles))
                    cos_Theta = np.cos(np.radians(angles))
                    return np.column_stack((sin_Theta, cos_Theta))

            
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.target_scaler_config=TargetScalerConfig()
        self.feature_scaler_config=FeatureScalerConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ['PM2.5','PM2.5_Chan','PM2.5_Ding','PM2.5_Dong','PM2.5_Guan','PM2.5_Guch',
                                 'PM2.5_Huai','PM2.5_Nong','PM2.5_Shun','PM2.5_Tian','PM2.5_Wanl','PM2.5_Wans',
                                'PM10','NO2','CO','TEMP','PRES','DEWP','RAIN','WSPM' ]
            categorical_columns = ['wd']          
            
            angle_mapping = {'N': 0,
                             'NNE': 22.5,
                             'NE': 45,
                             'ENE': 67.5,
                             'E': 90,
                             'ESE': 112.5,
                             'SE': 135,
                             'SSE': 157.5,
                             'S': 180,
                             'SSW': 202.5,
                             'SW': 225,
                             'WSW': 247.5,
                             'W': 270,
                             'WNW': 292.5,
                             'NW': 315,
                             'NNW': 337.5,
                             'N': 360
                            }
            
                       
            # Create pipeline
            num_pipeline = Pipeline(steps=
                                    [('imputer', InterpolateImputer(method='linear', limit_direction='forward', axis=0))]
                                   )
            
                                      
            cat_pipeline=Pipeline(steps=
                                  [("imputer",SimpleImputer(strategy="most_frequent")),
                                   ("wd_encoder",WindDirectionEncoder(angle_mapping))]
                                 )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_target_scaler_object(self):
        
        target_scaler = MinMaxScaler(feature_range=(0,1))
            
        return target_scaler
    
    def get_feature_scaler_object(self):
        
        feature_scaler = MinMaxScaler(feature_range=(0,1))
            
        return feature_scaler
    
    def initiate_data_transformation(self,train_path,val_path,test_path):

        try:
            train_df=pd.read_csv(train_path, index_col=None, header=0)
            val_df=pd.read_csv(val_path, index_col=None, header=0)
            test_df=pd.read_csv(test_path, index_col=None, header=0)

            logging.info("Reading train, val and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            input_train_df=train_df.drop(columns=['date','No','year','month','day','hour','SO2', 'O3','station'],axis=1)
                        
            input_val_df=train_df.drop(columns=['date','No','year','month','day','hour','SO2', 'O3','station'],axis=1)
            
            input_test_df=test_df.drop(columns=['date','No','year','month','day','hour','SO2', 'O3','station'],axis=1)
            

            logging.info(
                f"Applying preprocessing object on training, validation and testing dataframe."
            )

            input_train_arr=preprocessing_obj.fit_transform(input_train_df)              
            input_val_arr=preprocessing_obj.transform(input_val_df)            
            input_test_arr=preprocessing_obj.transform(input_test_df)
            
            
            # Split and scale target and feature array
            
            target_train_arr=input_train_arr[ : , :1]
            feature_train_arr=input_train_arr[ : ,1:]
            
            target_val_arr=input_val_arr[ : , :1]
            feature_val_arr=input_val_arr[ : ,1:]
            
            target_test_arr=input_test_arr[ : , :1]
            feature_test_arr=input_test_arr[ : ,1:]
            
            logging.info("Obtaining target scaler object")
            target_scaler_obj=self.get_target_scaler_object()
            
            logging.info(
                f"Applying target scaler object on target train, val and test array."
            )
            target_scaled_train = target_scaler_obj.fit_transform(target_train_arr)
            target_scaled_val = target_scaler_obj.transform(target_val_arr) 
            target_scaled_test = target_scaler_obj.transform(target_test_arr)
            
            logging.info("Obtaining feature scaler object")
            feature_scaler_obj=self.get_feature_scaler_object()
            
            logging.info(
                f"Applying feature scaler object on feature train, val and test array."
            )
            feature_scaled_train = feature_scaler_obj.fit_transform(feature_train_arr)
            feature_scaled_val = feature_scaler_obj.transform(feature_val_arr) 
            feature_scaled_test = feature_scaler_obj.transform(feature_test_arr)

            # Concatenate scaled target and scaled feature array & finally re-structure into windows of 24 hrs
            
            scaled_train_arr = np.concatenate((target_scaled_train, feature_scaled_train), axis=1)
            scaled_val_arr = np.concatenate((target_scaled_val, feature_scaled_val), axis=1)
            scaled_test_arr = np.concatenate((target_scaled_test, feature_scaled_test), axis=1)
            
            train_arr = np.array(np.split(scaled_train_arr, len(scaled_train_arr) / 24))
            val_arr = np.array(np.split(scaled_val_arr, len(scaled_val_arr) / 24))
            test_arr = np.array(np.split(scaled_test_arr, len(scaled_test_arr) / 24))
            
            
            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            
            logging.info(f"Saved target scaler object.")

            save_object(

                file_path=self.target_scaler_config.target_scaler_obj_file_path,
                obj=target_scaler_obj

            )
            
            logging.info(f"Saved feature scaler object.")

            save_object(

                file_path=self.feature_scaler_config.feature_scaler_obj_file_path,
                obj=feature_scaler_obj

            )

            return (
                train_arr,
                val_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.target_scaler_config.target_scaler_obj_file_path,
                self.feature_scaler_config.feature_scaler_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
            
            