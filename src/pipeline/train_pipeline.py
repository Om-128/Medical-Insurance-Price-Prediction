import os
import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

class TrainPipeline:
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def train_pipeline(self):
        try:
            logging.info("Training pipeline started")
            #Data Ingestion
            data_ingestion = DataIngestion(self.data_ingestion_config)
            train_data, test_data = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")

            #Data Transformation
            train_arr, test_arr = self.data_transformation.initiate_data_transformation(train_data, test_data)
            logging.info("Data transformation completed")

            #Model Trainer
            score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info("Model training completed")

            return score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    model_trainer = TrainPipeline()
    print(model_trainer.train_pipeline())