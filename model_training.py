from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import RidgeCV, ElasticNetCV, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from data_processing import drop_missing_values, impute_missing_values_with_zero
import pandas as pd
import numpy as np
from utils import get_logger
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


logger = get_logger(__name__)

def train_model(
    df_train: pd.DataFrame, df_test: pd.DataFrame, features: list[str], position: str
):
    logger.debug(f"Training with features: {features}")
    df_train = drop_missing_values(df_train.copy(), features)
    df_test = drop_missing_values(df_test.copy(), features)


    models = {
        "Ridge": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-6, 6, 13))
        ),
        "ElasticNet": make_pipeline(
            StandardScaler(), ElasticNetCV(cv=5, max_iter=10000)
        ),
        "RandomForestRegressor": GridSearchCV(
            RandomForestRegressor(max_depth=3), param_grid={"n_estimators": [50]}, cv=5
        ),
        "GradientBoostingRegressor": GridSearchCV(
            GradientBoostingRegressor(max_depth=3),
            param_grid={"n_estimators": [50], "learning_rate": [0.1]},
            cv=5,
        ),
        "XGBoost": GridSearchCV(
            XGBRegressor(),
            param_grid={"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.3]},
            cv=5,
        ),
        "LightGBM": GridSearchCV(
            LGBMRegressor(),
            param_grid={"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.3]},
            cv=5,
        ),
        "SVR": GridSearchCV(
            SVR(),
            param_grid={"C": [0.1, 1, 10], "epsilon": [0.1, 0.2]},
            cv=5,
        ),
        # "NeuralNetwork": GridSearchCV(
        #     MLPRegressor(max_iter=1000),
        #     param_grid={"hidden_layer_sizes": [(50,), (100,), (50, 50)], "alpha": [0.0001, 0.001]},
        #     cv=5,
        # )  
    }

    df_pos_train = df_train[df_train["position"] == position].copy()
    df_pos_test = df_test[df_test["position"] == position].copy()

    results = {}
    
    for name, model in models.items():
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
        cv_rmse = np.sqrt(
            -cross_val_score(
                model,
                df_train[features],
                df_train["fantasy_points_ppr"],
                cv=5,
                scoring="neg_mean_squared_error",
            ).mean()
        )

        results[name] = {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "cv_rmse": cv_rmse,
        }

    return results
