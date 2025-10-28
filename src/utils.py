import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=0)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            from sklearn.metrics import r2_score
            score = r2_score(y_test, y_pred)
            report[model_name] = score

        return report

    except Exception as e:
        raise CustomException(e, sys)
