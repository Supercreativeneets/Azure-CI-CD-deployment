import os
import sys
import pandas as pd
import numpy as np
import glob
import datetime


from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    val_data_path: str=os.path.join('artifacts',"val.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            filepath = 'notebook/data'
            allfiles = sorted(glob.glob(filepath + "/*.csv"))

            stations = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 
                        'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']

            station_dfs = {}  # Dictionary to store DataFrames for each station

            for station in stations:
                station_dfs[station] = pd.DataFrame()  # Initialize an empty DataFrame for each station

            for file in allfiles:
                filename = file.split('/')[-1]  # Extract filename from the full file path
                for station in stations:
                    if station in filename:
                        df = pd.read_csv(file, index_col=None, header=0)
                        if station_dfs[station].empty:
                            station_dfs[station] = df
                        else:
                            station_dfs[station] = pd.concat([station_dfs[station], df], ignore_index=True)

            
            Aotizhongxin = station_dfs['Aotizhongxin']
            Changping = station_dfs['Changping']
            Dingling = station_dfs['Dingling']
            Dongsi = station_dfs['Dongsi']
            Guanyuan = station_dfs['Guanyuan']
            Gucheng = station_dfs['Gucheng']
            Huairou = station_dfs['Huairou']
            Nongzhanguan = station_dfs['Nongzhanguan']
            Shunyi = station_dfs['Shunyi']
            Tiantan = station_dfs['Tiantan']
            Wanliu = station_dfs['Wanliu']
            Wanshouxigong = station_dfs['Wanshouxigong']
                                        
            logging.info('Read the dataset as dataframe')
            
            # create a datetime column using the year,month,day and hour columns.
            years = Aotizhongxin['year'].values
            months = Aotizhongxin['month'].values
            days = Aotizhongxin['day'].values
            hours = Aotizhongxin['hour'].values
            full_date = []
            
            for i in range(Aotizhongxin.shape[0]):
                date_time = str(years[i])+'-'+str(months[i])+'-'+str(days[i])+' '+str(hours[i])+':'+str(0)
                full_date.append(date_time)
                
            dates = pd.to_datetime(full_date)
            dates = pd.DataFrame(dates,columns=['date'])
            Aotizhongxin = pd.concat([dates,Aotizhongxin],axis=1)                     
                                  
            # Extract PM2.5 column from other stations dataframes
            Changping_PM = Changping['PM2.5'].values.reshape(-1, 1)
            Dingling_PM = Dingling['PM2.5'].values.reshape(-1, 1)
            Dongsi_PM = Dongsi['PM2.5'].values.reshape(-1, 1)
            Guanyuan_PM = Guanyuan['PM2.5'].values.reshape(-1, 1)
            Gucheng_PM = Gucheng['PM2.5'].values.reshape(-1, 1)
            Huairou_PM = Huairou['PM2.5'].values.reshape(-1, 1)
            Nongzhanguan_PM = Nongzhanguan['PM2.5'].values.reshape(-1, 1)
            Shunyi_PM = Shunyi['PM2.5'].values.reshape(-1, 1)
            Tiantan_PM = Tiantan['PM2.5'].values.reshape(-1, 1)
            Wanliu_PM = Wanliu['PM2.5'].values.reshape(-1, 1)
            Wanshouxigong_PM = Wanshouxigong['PM2.5'].values.reshape(-1, 1)
            
            # Create a dataframe with PM2.5 column from other stations
            Otherstations_PM = np.hstack((Changping_PM, Dingling_PM, Dongsi_PM, Guanyuan_PM, Gucheng_PM, Huairou_PM, Nongzhanguan_PM,
                             Shunyi_PM, Tiantan_PM, Wanliu_PM, Wanshouxigong_PM)) # Horizontally stack columns

            other_PM_df = pd.DataFrame(Otherstations_PM, columns=['PM2.5_Chan','PM2.5_Ding','PM2.5_Dong','PM2.5_Guan','PM2.5_Guch',
                                                      'PM2.5_Huai','PM2.5_Nong','PM2.5_Shun','PM2.5_Tian','PM2.5_Wanl',
                                                      'PM2.5_Wans']) # create other PM2.5 dataframe
            
            # Concatenate Aotizhongxin dataframe with other_PM_df along the columns axis
            final_df = pd.concat([Aotizhongxin, other_PM_df], axis=1)
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
                     
            final_df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)

            logging.info("Train val test split initiated")

            # Split the dataset into train/test sets
            def Split_dataset(df):
                # Split into train_data (1096 days X 24 hr) and val_data (219 days X 24 hr), test_data (146 days X 24 hr )
                train_data, val_data, test_data = df.iloc[ :-8760, : ], df.iloc[-8760:-3504, : ], df.iloc[-3504: , : ]
                
                return train_data, val_data, test_data
            
            train_data, val_data, test_data = Split_dataset(final_df)
            
            train_set = pd.DataFrame(train_data)
            val_set = pd.DataFrame(val_data)
            test_set = pd.DataFrame(test_data)
                        

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            val_set.to_csv(self.ingestion_config.val_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__=="__main__":
    
    obj=DataIngestion()
    train_data,val_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,val_arr,test_arr,_,_,_=data_transformation.initiate_data_transformation(train_data,val_data,test_data)
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,val_arr,test_arr))