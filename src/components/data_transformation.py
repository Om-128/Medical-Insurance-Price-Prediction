import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


from src.utils import save_object

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_features = ['age', 'bmi', 'children']
            cat_features = ['sex', 'smoker', 'region']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns Scaling completed: {num_features}")
            logging.info(f"Categorical columns Encoding completed: {cat_features}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")

            preprocessor_obj = self.get_data_transformer_object()
            logging.info("Obtained preprocessor object")

            target_column_name = 'charges'

            logging.info("Splitting input and target feature from train and test dataframe")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            scaled_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            scaled_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("combine scaled input feature with target column")
            
            train_arr = np.c_[
                scaled_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                scaled_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            # Save the preprocessor object
            save_object(
                file_path = self.data_transformation_config.preprocessor_object_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr
            )


        except Exception as e:
            raise CustomException(e, sys)