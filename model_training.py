"""
Trains model. Core functionality of the project.

Extend or tweak models in the models dictionary to experiment.
"""

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import RidgeCV, ElasticNetCV, BayesianRidge, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
import catboost as cb
from data_processing import (
    drop_missing_values,
    impute_missing_values_with_zero,
    impute_missing_values_with_median,
)
from utils import get_logger
import pandas as pd
import numpy as np
import joblib
import os

from config import POSITIONS
from evaluation import calculate_position_rmse


logger = get_logger(__name__)


def train_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: list[str],
    save_models: bool = True,
) -> tuple[pd.DataFrame, str]:

    def generate_output_dir() -> str:
        timestamp: str = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        output_filepath: str = "output/" + timestamp + "/"
        os.makedirs(output_filepath, exist_ok=False)
        logger.info(f"Output directory created: {output_filepath}")
        return output_filepath

    output_filepath = generate_output_dir()

    model_performance_results = {}

    for position in POSITIONS:
        logger.info(f"Training for position {position}")
        model_performance_results[position] = train_models_by_position(
            df_train=df_train,
            df_test=df_test,
            feature_columns=feature_columns,
            position=position,
            filepath=output_filepath,
            save_models=save_models,
        )

    model_performance_results = calculate_position_rmse(model_performance_results)
    model_performance_results.to_csv(output_filepath + "/all_sets_rmse.csv")

    cv_model_perfomance_results = model_performance_results[
        [col for col in model_performance_results.columns if "cross_val" in col]
    ]
    cv_model_perfomance_results.to_csv(output_filepath + "/cross_val_rmse.csv")

    return model_performance_results, output_filepath


def train_models_by_position(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: list[str],
    position: str,
    filepath: str,
    save_models: bool = True,
):
    logger.debug(f"[{position}] Selected features: {feature_columns}")

    # Defaults to filling with median. Other options for missing values are zero and mean.
    df_train = impute_missing_values_with_zero(df_train.copy(), feature_columns)
    df_test = impute_missing_values_with_zero(df_test.copy(), feature_columns)

    models = {
        # Remove these three prob
        # "Ridge": make_pipeline(
        #     StandardScaler(), RidgeCV(alphas=np.logspace(-6, 6, 13))
        # ),
        # "Lasso": make_pipeline(
        #     StandardScaler(),
        #     LassoCV(cv=5, max_iter=20000, alphas=np.logspace(-6, 6, 50)),
        # ),
        # "BayesianRidge": make_pipeline(StandardScaler(), BayesianRidge()),
        # **
        # "ElasticNet": make_pipeline(
        #     StandardScaler(), ElasticNetCV(cv=5, max_iter=10000)
        # ),
        # "GradientBoosting": GridSearchCV(
        #     GradientBoostingRegressor(),
        #     param_grid={
        #         "max_depth": [3, 5, 10],
        #         "n_estimators": [50, 100],
        #         "learning_rate": [0.01, 0.1],
        #     },
        #     cv=5,
        # ),
        # "CatBoost": GridSearchCV(
        #     cb.CatBoostRegressor(),
        #     param_grid={"max_depth": [3, 5, 10], "n_estimators": [50, 100]},
        #     cv=5,
        # ),
        "RandomForest": GridSearchCV(
            RandomForestRegressor(),
            param_grid={"max_depth": [3, 5, 10], "n_estimators": [50, 100]},
            cv=5,
        ),
        # **
        # "KNN": GridSearchCV(
        #     KNeighborsRegressor(),
        #     param_grid={"n_neighbors": [3, 5, 10]},
        #     cv=5,
        # ),
    }

    logger.debug(f"[{position}] Training with models: {models.keys()}")

    kf = KFold(n_splits=5, shuffle=True)
    train_data_by_position = df_train[df_train["position"] == position].copy()
    test_data_by_position = df_test[df_test["position"] == position].copy()

    performance_results = {}

    for model_name, model in models.items():
        cv_scores = cross_val_score(
            model,
            train_data_by_position[feature_columns],
            train_data_by_position["fantasy_points_ppr"],
            cv=kf,
            scoring="neg_mean_squared_error",
        )

        model.fit(
            train_data_by_position[feature_columns],
            train_data_by_position["fantasy_points_ppr"],
        )

        train_rmse = np.sqrt(
            mean_squared_error(
                train_data_by_position["fantasy_points_ppr"],
                model.predict(train_data_by_position[feature_columns]),
            )
        )
        test_rmse = np.sqrt(
            mean_squared_error(
                test_data_by_position["fantasy_points_ppr"],
                model.predict(test_data_by_position[feature_columns]),
            )
        )
        cross_val_rmse = np.sqrt(-cv_scores.mean())

        performance_results[model_name] = {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "cross_val_rmse": cross_val_rmse,
        }

        logger.info(
            f"Position: {position}, Model: {model_name}, Train RMSE: {train_rmse}, Test RMSE: {test_rmse}, Cross-Validation RMSE: {cross_val_rmse}"
        )

        if save_models:
            model_path = os.path.join(filepath, model_name + ".joblib")
            joblib.dump(model, model_path)
            logger.info(
                f"Model {model_name} for position {position} saved to {model_path}"
            )

    return performance_results
