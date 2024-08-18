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


logger = get_logger(__name__)


def train_model(
    df_train: pd.DataFrame, df_test: pd.DataFrame, features: list[str], position: str
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
        # Tree-based models
        "RandomForestRegressor": GridSearchCV(
            RandomForestRegressor(),
            param_grid={"max_depth": [3, 5, 10], "n_estimators": [50, 100]},
            cv=5,
        ),
        "GradientBoostingRegressor": GridSearchCV(
            GradientBoostingRegressor(),
            param_grid={"max_depth": [3, 5, 10], "n_estimators": [50, 100]},
            cv=5,
        ),
    }

    kf = KFold(n_splits=5, shuffle=True)
    df_pos_train = df_train[df_train["position"] == position].copy()
    df_pos_test = df_test[df_test["position"] == position].copy()

    results = {}

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

        logger.debug(
            f"Position: {position}, Model: {name}, Train RMSE: {train_rmse}, Test RMSE: {test_rmse}, CV RMSE: {cv_rmse}"
        )

    return results
