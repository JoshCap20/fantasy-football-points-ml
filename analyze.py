import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_logger
from sklearn.model_selection import learning_curve

logger = get_logger(__name__)


class FantasyFootballAnalysis:
    def __init__(self, df: pd.DataFrame, path: str = "output/"):
        self.path = path
        self.df = df

        if self.df.index.name != "Position":
            self.df.set_index("Position", inplace=True)

        self.models = self.df["Model"].unique()

    def autolabel(self, rects, ax):
        """Attach a text label above each bar displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    def plot_rmse_by_model_and_position(self):
        """Plot CV RMSE by model and position in a single grouped bar chart."""

        fig, ax = plt.subplots(figsize=(14, 7))

        positions = self.df.index.unique()
        num_positions = len(positions)
        num_models = len(self.models)
        bar_width = 0.8 / num_models

        indices = np.arange(num_positions)

        for i, model in enumerate(self.models):
            data = self.df[self.df["Model"] == model]
            ax.bar(
                indices + i * bar_width,
                data["cross_val_rmse"],
                width=bar_width,
                label=model,
                alpha=0.7,
            )

        ax.set_xticks(indices + bar_width * (num_models - 1) / 2)
        ax.set_xticklabels(positions, rotation=20, ha="right", fontsize=10)

        ax.set_ylabel("CV RMSE")
        ax.set_title("Cross-Validation RMSE by Model and Position", fontsize=16, pad=20)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.path, "cv_rmse_comparison_by_model.png"), dpi=300)
        plt.show()

    def plot_rmse_distribution(self):
        """Plot RMSE distribution by model and metric."""
        metrics = ["train_rmse", "test_rmse", "cross_val_rmse"]
        melted_df = pd.melt(
            self.df,
            id_vars=["Model"],
            value_vars=metrics,
            var_name="Metric",
            value_name="RMSE",
        )

        plt.figure(figsize=(14, 8))
        sns.boxplot(x="Model", y="RMSE", hue="Metric", data=melted_df)
        plt.title("Aggregated RMSE Distribution by Model and Metric")
        plt.ylabel("RMSE")
        plt.xlabel("Model")
        plt.legend(loc="upper right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, "rmse_distribution_by_model.png"), dpi=300)
        plt.show()

    def plot_feature_correlation_heatmap(self, df_features: pd.DataFrame):
        """Plot a heatmap of feature correlations."""
        correlation_matrix = df_features.corr()
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, "feature_correlation_heatmap.png"), dpi=300)
        plt.show()

    def plot_learning_curve(self, estimator, X, y, cv=None):
        """Plot a learning curve for the given estimator."""
        plt.figure(figsize=(10, 6))
        plt.title(f"Learning Curve: {type(estimator).__name__}")
        plt.xlabel("Training examples")
        plt.ylabel("RMSE")

        train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring="neg_root_mean_squared_error",
        )

        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)

        plt.grid()
        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )

        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.path, f"learning_curve_{type(estimator).__name__}.png"),
            dpi=300,
        )
        plt.show()

    def plot_rmse_across_years(self, rmse_across_years: dict):
        """Plot RMSE comparison across different years."""
        plt.figure(figsize=(14, 8))
        for year, rmse in rmse_across_years.items():
            plt.plot(self.models, rmse, label=f"RMSE for {year}")

        plt.title("Comparison of RMSE Across Years")
        plt.ylabel("RMSE")
        plt.xlabel("Model")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.path, "rmse_comparison_across_years.png"), dpi=300
        )
        plt.show()


def run_analysis(df: pd.DataFrame | None = None, path: str | None = None):
    if path:
        analysis = FantasyFootballAnalysis(df=df, path=path)
    else:
        analysis = FantasyFootballAnalysis(df=df)

    analysis.plot_rmse_by_model_and_position()
    analysis.plot_rmse_distribution()
    # Example usage:
    # df_features should be a DataFrame containing the features for heatmap visualization
    # analysis.plot_feature_correlation_heatmap(df_features)
    # Example for learning curve:
    # analysis.plot_learning_curve(estimator, X, y, cv=5)
    # Example for RMSE across years:
    # analysis.plot_rmse_across_years(rmse_across_years)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fantasy Football Analysis")
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to the CSV file containing RMSE data",
        required=True,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    run_analysis(df)
