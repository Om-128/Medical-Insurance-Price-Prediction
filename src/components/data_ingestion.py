import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'data.csv')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):

        try:
            logging.info("Starting data ingestion process")

            #Load the dataset
            df = pd.read_csv('notebook/data/insurance.csv')
            logging.info("Dataset loaded successfully")

            #Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            df.to_csv(self.config.raw_data_path, index=False, header=True)

            ''' Split the dataset into training and testing sets'''
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            ''' Save the train and test datasets to their respective paths
            '''
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.initiate_data_ingestion()