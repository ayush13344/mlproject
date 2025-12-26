import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import accuracy_score, hamming_loss

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise e
  
def evaluate_model(X_train, y_train, X_test, y_test, models):    
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            # Train model
            model.fit(X_train, y_train)

            # Predict Training and Testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # ✅ Choose metric based on shape (single-label vs multi-label)
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                # Multi-label classification
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test, y_test_pred)

                # Optional: also log Hamming Loss for better insight
                hloss = hamming_loss(y_test, y_test_pred)
                print(f"{model_name}: Accuracy={test_score:.4f}, Hamming Loss={hloss:.4f}")

            else:
                # Single-label classification
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test, y_test_pred)
                print(f"{model_name}: Accuracy={test_score:.4f}")

            report[model_name] = test_score

        return report

    except Exception as e:
        raise e  
    

def load_object(file_path):   
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise e 