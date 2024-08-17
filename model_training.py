from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import RidgeCV, ElasticNetCV, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import pandas as pd
import numpy as np


def drop_missing_values(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.dropna(subset=features)
    return df


def impute_missing_values(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    imputer = SimpleImputer(strategy="mean")
    df[features] = imputer.fit_transform(df[features])
    return df


def train_model(
    df_train: pd.DataFrame, df_test: pd.DataFrame, features: list[str], position: str
) -> dict[str, dict[str, float]]:
    df_train = drop_missing_values(df_train, features)
    df_test = drop_missing_values(df_test, features)

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
            param_grid={"n_estimators": [50], "learning_rate": [0.3]},
            cv=5,
        ),
    }

    df_pos_train = df_train[df_train["Pos"] == position]
    df_pos_test = df_test[df_test["Pos"] == position]

    results = {}
    for name, model in models.items():
        model.fit(df_pos_train[features], df_pos_train["FD points"])
        train_rmse = np.sqrt(
            mean_squared_error(
                df_pos_train["FD points"], model.predict(df_pos_train[features])
            )
        )
        test_rmse = np.sqrt(
            mean_squared_error(
                df_pos_test["FD points"], model.predict(df_pos_test[features])
            )
        )
        cv_rmse = np.sqrt(
            -cross_val_score(
                model,
                df_train[features],
                df_train["FD points"],
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

    # if(est == "GradientBoostingRegressor"):
    #     n_estimators = [50]
    #     learning_rate = [0.1]
    #     param_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
    #     grid_search = GridSearchCV(GradientBoostingRegressor(max_depth=3), param_grid, cv=5)
    #     grid_search.fit(df_pos_train[features], df_pos_train['FD points'])

    # elif(est == "RandomForestRegressor"):
    #     n_estimators = [50]
    #     param_grid = {'n_estimators': n_estimators}
    #     grid_search = GridSearchCV(RandomForestRegressor(max_depth=3), param_grid, cv=5)
    #     grid_search.fit(df_pos_train[features], df_pos_train['FD points'])

    # elif(est == "ElasticNet"):
    #     grid_search = ElasticNetCV().fit(df_pos_train[features], df_pos_train['FD points'])

    # elif(est == "BayesianRidge"):
    #     alpha_1 = [1e-6, 1e-5, 1e-7]
    #     alpha_2 = [1e-6, 1e-5, 1e-7]
    #     lambda_1 = [1e-6, 1e-5, 1e-7]
    #     lambda_2 = [1e-6, 1e-5, 1e-7]
    #     param_grid = {'alpha_1': alpha_1, 'alpha_2':alpha_2, 'lambda_1':lambda_1, 'lambda_2':lambda_2}
    #     grid_search = GridSearchCV(BayesianRidge(), param_grid, cv=5)
    #     grid_search.fit(df_pos_train[features], df_pos_train[target])

    # elif(est == "Ridge"):
    #     grid_search = RidgeCV().fit(df_pos_train[features], df_pos_train['FD points'])

    # elif(est == "SVM"):
    #     C = [50]
    #     gamma = [0.3]
    #     param_grid = {'C': C, 'gamma': gamma}
    #     grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    #     grid_search.fit(df_pos_train[features], df_pos_train['FD points'])
