import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # ✅ Handle case where y has multiple columns
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                logging.info(f"Detected multi-output classification: {y_train.shape[1]} labels")
                models = {
                    "Random Forest": MultiOutputClassifier(RandomForestClassifier()),
                    "Decision Tree": MultiOutputClassifier(DecisionTreeClassifier()),
                    "Gradient Boosting": MultiOutputClassifier(GradientBoostingClassifier()),
                    "KNN": MultiOutputClassifier(KNeighborsClassifier()),
                    "Logistic Regression": MultiOutputClassifier(LogisticRegression(max_iter=1000)),

                    "AdaBoost": MultiOutputClassifier(AdaBoostClassifier()),
                }
            else:
                logging.info("Detected single-output classification")
                models = {
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "KNN": KNeighborsClassifier(),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    
                    "AdaBoost": AdaBoostClassifier(),
                }

            # evaluate models
            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )
            if len(y_train.shape) == 1:  # Single-label classification
                le = LabelEncoder()
                all_y = np.concatenate([y_train, y_test])   # combine first
                le.fit(all_y)

                y_train = le.transform(y_train)
                y_test = le.transform(y_test)

            else:  # Multi-label case
                for col in range(y_train.shape[1]):
                    le = LabelEncoder()
                    all_y = np.concatenate([y_train[:, col], y_test[:, col]])
                    le.fit(all_y)

                    y_train[:, col] = le.transform(y_train[:, col])
                    y_test[:, col] = le.transform(y_test[:, col])

                y_train = np.array(y_train, dtype=int)
                y_test = np.array(y_test, dtype=int)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            preds = best_model.predict(X_test)

            # ✅ Choose metric based on type
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                acc = accuracy_score(y_test, preds)
                hloss = hamming_loss(y_test, preds)
                logging.info(f"Multi-output Accuracy: {acc}, Hamming Loss: {hloss}")
                return {"accuracy": acc, "hamming_loss": hloss}
            else:
                acc = accuracy_score(y_test, preds)
                logging.info(f"Single-output Accuracy: {acc}")
                return {"accuracy": acc}

        except Exception as e:
            logging.error(f"Error occurred while training models: {e}")
            raise CustomException(e, sys)
