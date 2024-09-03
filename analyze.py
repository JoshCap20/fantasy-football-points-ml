import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from typing import Optional

class FantasyFootballAnalysis:
    def __init__(self, df: pd.DataFrame | None = None, path="output/"):
        self.path = path
        self.df = df if df is not None else self._load_latest_rmse()
        print("Columns in the DataFrame:", self.df.columns)
        self.df.set_index("Model", inplace=True)
        self.base_models = sorted(set(model.split("_")[0] for model in self.df.index))

    def _load_latest_rmse(self) -> pd.DataFrame:
        """Load the most recent RMSE file from the output directory."""
        rmse_files = [
            f for f in os.listdir(self.path) if f.startswith("rmse_") and f.endswith(".csv")
        ]
        if not rmse_files:
            raise FileNotFoundError("No RMSE files found in the output directory.")
        latest_rmse_file = max(rmse_files, key=lambda x: x.split("_")[1].split(".")[0])
        return pd.read_csv(os.path.join(self.path, latest_rmse_file))

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
        """Plot RMSE by model and position."""
        num_positions = len(self.df.columns)
        fig, axes = plt.subplots(num_positions, 1, figsize=(14, num_positions * 5))

        if num_positions == 1:
            axes = [axes]

        colors = {"train": "blue", "test": "orange", "cv": "green"}

        for ax, position in zip(axes, self.df.columns):
            data = {
                base_model: self.df.loc[
                    [
                        f"{base_model}_train_rmse",
                        f"{base_model}_test_rmse",
                        f"{base_model}_cross_val_rmse",
                    ],
                    position,
                ].values
                for base_model in self.base_models
            }

            ind = np.arange(len(self.base_models))
            width = 0.2
            for i, (base_model, values) in enumerate(data.items()):
                rects_train = ax.bar(
                    ind[i] - width,
                    values[0],
                    width,
                    color=colors["train"],
                    label="Train" if i == 0 else "",
                )
                rects_test = ax.bar(
                    ind[i],
                    values[1],
                    width,
                    color=colors["test"],
                    label="Test" if i == 0 else "",
                )
                rects_cv = ax.bar(
                    ind[i] + width,
                    values[2],
                    width,
                    color=colors["cv"],
                    label="CV" if i == 0 else "",
                )

                self.autolabel(rects_train, ax)
                self.autolabel(rects_test, ax)
                self.autolabel(rects_cv, ax)

            ax.set_ylabel("RMSE")
            ax.set_title(f"RMSE by Model for Position: {position}", fontsize=14, pad=20)
            ax.set_xticks(ind)
            ax.set_xticklabels(self.base_models, rotation=20, ha="right", fontsize=10)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.join(self.path, "position_rmse_comparison_by_model.png"), dpi=300)
        plt.show()

    def plot_rmse_distribution(self):
        """Plot RMSE distribution by model and metric."""
        boxplot_data = [
            {
                "Model": base_model,
                "Metric": metric,
                "RMSE": self.df.loc[f"{base_model}_{metric}", :].values,
            }
            for base_model in self.base_models
            for metric in ["train_rmse", "test_rmse", "cv_rmse"]
        ]

        boxplot_df = pd.DataFrame(boxplot_data)
        plt.figure(figsize=(14, 8))
        sns.boxplot(x="Model", y="RMSE", hue="Metric", data=boxplot_df.explode("RMSE"))
        plt.title("RMSE Distribution by Model and Metric")
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
        plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")

        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, f"learning_curve_{type(estimator).__name__}.png"), dpi=300)
        plt.show()

    def plot_rmse_across_years(self, rmse_across_years: dict):
        """Plot RMSE comparison across different years."""
        plt.figure(figsize=(14, 8))
        for year, rmse in rmse_across_years.items():
            plt.plot(self.base_models, rmse, label=f"RMSE for {year}")

        plt.title("Comparison of RMSE Across Years")
        plt.ylabel("RMSE")
        plt.xlabel("Model")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, "rmse_comparison_across_years.png"), dpi=300)
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
        "--csv", type=str, help="Path to the CSV file containing RMSE data"
    )
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        run_analysis(df)
    else:
        run_analysis()
