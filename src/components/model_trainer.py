import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

# import ML Algorithms
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and testing input data")

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            ''' Define the models to be evaluated '''
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBoost": XGBRegressor()
            }

            '''     Hyperparameter tuning for the models    '''
            params = {
                    "Decision Tree": {
                        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                        "splitter": ["best", "random"],
                        },
                    "Random Forest": {
                        "n_estimators": [10, 50, 100],
                        "criterion": ["squared_error", "absolute_error"],
                    },
                    "Gradient Boosting": {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },  
                    "Linear Regression": {
                        "fit_intercept": [True, False],
                        "positive": [True, False]
                    },
                    "XGBoost": {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                    "CatBoost": {
                        "depth": [6, 8, 10],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "iterations": [30, 50, 100]
                    },
                    "K-Neighbors": {
                        "n_neighbors": [3, 5, 7, 9],
                        "weights": ["uniform", "distance"],
                        "p": [1, 2]
                    },
                    "AdaBoost": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1.0]
                    }
                }

            # Returns all models R2 score
            model_report: dict = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

        
            ''' To get the best model score from the dictionary '''
            best_model_score = max(sorted(model_report.values()))
            
            # Using the highest R2score we will get Key i.e. name of the model which has that score
            #list(model_report.keys()) => ['Linear Regression', 'K-Neighbors Regressor', 'Decision Tree', 'CatBoost Regressor', 'AdaBoost Regressor', 'Gradient Boosting Regressor', 'Random Forest', 'XGB Regressor']
            #list(model_report.values()) => [0.798, 0.856, 0.734, 0.876, 0.812, 0.892, 0.874, 0.881]
            #index(best_model_score) => index(0.856) will return 'K-Neighbors Regressor'
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            '''Get the best model object using best_model_name'''
            best_model = models[best_model_name]

            if(best_model_score < 0.6):
                raise CustomException("No best model found with sufficient accuracy")

            logging.info(f"Best model found for both training and testing data")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

