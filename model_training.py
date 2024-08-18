"""
Trains model. Core functionality of the project.

Extend or tweak models in the models dictionary to experiment.
"""

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import RidgeCV, ElasticNetCV, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from data_processing import drop_missing_values, impute_missing_values_with_zero
import pandas as pd
import numpy as np
from utils import get_logger
import joblib
import os

# testing these
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import AdaBoostRegressor


logger = get_logger(__name__)


def train_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    position: str,
    save_models: bool = True,
):
    logger.debug(f"Training with features: {features}")
    df_train = impute_missing_values_with_zero(df_train.copy(), features)
    df_test = impute_missing_values_with_zero(df_test.copy(), features)

    models = {
        # Linear models
        "Ridge": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-6, 6, 13))
        ),
        "ElasticNet": make_pipeline(
            StandardScaler(), ElasticNetCV(cv=5, max_iter=10000)
        ),
        "BayesianRidge": make_pipeline(StandardScaler(), BayesianRidge()),
        # Other models
        "RandomForest": GridSearchCV(
            RandomForestRegressor(),
            param_grid={"max_depth": [3, 5, 10], "n_estimators": [50, 100]},
            cv=5,
        ),
        "GradientBoosting": GridSearchCV(
            GradientBoostingRegressor(),
            param_grid={"max_depth": [3, 5, 10], "n_estimators": [50, 100]},
            cv=5,
        ),
        "XGBoost": GridSearchCV(
            xgb.XGBRegressor(),
            param_grid={"max_depth": [3, 5, 10], "n_estimators": [50, 100]},
            cv=5,
        ),
        "LightGBM": GridSearchCV(
            lgb.LGBMRegressor(),
            param_grid={"max_depth": [3, 5, 10], "n_estimators": [50, 100]},
            cv=5,
        ),
        "CatBoost": GridSearchCV(
            cb.CatBoostRegressor(),
            param_grid={"max_depth": [3, 5, 10], "n_estimators": [50, 100]},
            cv=5,
        ),
        "AdaBoost": GridSearchCV(
            AdaBoostRegressor(),
            param_grid={"n_estimators": [50, 100]},
            cv=5,
        ),
    }

    kf = KFold(n_splits=5, shuffle=True)
    df_pos_train = df_train[df_train["position"] == position].copy()
    df_pos_test = df_test[df_test["position"] == position].copy()

    results = {}
    model_dir = "output/models/"
    os.makedirs(model_dir, exist_ok=True)

    for name, model in models.items():
        cv_scores = cross_val_score(
            model,
            df_pos_train[features],
            df_pos_train["fantasy_points_ppr"],
            cv=kf,
            scoring="neg_mean_squared_error",
        )

        model.fit(df_pos_train[features], df_pos_train["fantasy_points_ppr"])

        train_rmse = np.sqrt(
            mean_squared_error(
                df_pos_train["fantasy_points_ppr"],
                model.predict(df_pos_train[features]),
            )
        )
        test_rmse = np.sqrt(
            mean_squared_error(
                df_pos_test["fantasy_points_ppr"], model.predict(df_pos_test[features])
            )
        )
        cv_rmse = np.sqrt(-cv_scores.mean())

        results[name] = {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "cv_rmse": cv_rmse,
        }

        logger.info(
            f"Position: {position}, Model: {name}, Train RMSE: {train_rmse}, Test RMSE: {test_rmse}, CV RMSE: {cv_rmse}"
        )

        if save_models:
            model_path = os.path.join(model_dir, f"{position}_{name}.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Model {name} for position {position} saved to {model_path}")

    return results
