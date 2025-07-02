import os
import pandas as pd
import numpy as np
import joblib
from Src.logger import get_logger
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import lightgbm as lgb
from Src.custom_exception import CustomException
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from config.paths_config import *
from config.model_params import *
from utils.common_functions import *
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, train_path, test_path, output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path

        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading training data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading test data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(['booking_status'], axis=1)
            y_train = train_df['booking_status']

            X_test = test_df.drop(['booking_status'], axis=1)
            y_test = test_df['booking_status']

            logger.info("Data successfully split into features and labels.")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error loading or splitting data: {e}")
            raise CustomException("Failed to load and split data", e)

    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing LGBMClassifier")
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info("Starting hyperparameter tuning with RandomizedSearchCV")
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )

            random_search.fit(X_train, y_train)

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best hyperparameters: {best_params}")
            return best_lgbm_model
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException("Failed to train model", e)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating model performance on test set")
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Accuracy: {acc}")
            logger.info(f"Precision: {prec}")
            logger.info(f"Recall: {rec}")
            logger.info(f"F1 Score: {f1}")

            return {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            }
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException("Failed to evaluate model", e)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            logger.info("Saving model with joblib")
            joblib.dump(model, self.output_path)
            logger.info(f"Model saved at {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException("Failed to save model", e)

    def run(self):
        try:
            with mlflow.start_run():  # ✅ FIXED LINE
                logger.info("Model training pipeline started")

                # Log training and test datasets
                logger.info("Logging datasets to MLflow")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                # Data loading and training
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)

                # Evaluation
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)

                # Save model
                self.save_model(best_lgbm_model)

                # Log model and parameters
                logger.info("Logging model, parameters, and metrics to MLflow")
                mlflow.log_artifact(self.output_path)
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model training pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error in the model training pipeline: {e}")
            raise CustomException("Failed during model training pipeline", e)


if __name__ == "__main__":
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()
