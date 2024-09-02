import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve


class FantasyFootballAnalysis:
    def __init__(self, path="output/"):
        self.path = path
        self.df = pd.read_csv(self.path + "position_rmse.csv")
        self.df.set_index("Model", inplace=True)
        self.base_models = sorted(set(model.split("_")[0] for model in self.df.index))

    def autolabel(self, rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                1.05 * height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    # def find_best_estimator(self):
    #     min_rmse_by_position = {}
    #     print("The best estimator for each position with the lowest RMSE is:\n")
    #     for position in self.df.columns[1:]:
    #         min_rmse = self.df[position].min()
    #         min_rmse_index = self.df[position].idxmin()
    #         min_rmse_algo = self.df.iloc[min_rmse_index, 0]
    #         min_rmse_by_position[position] = min_rmse
    #         print(f"{position}: {min_rmse_algo} with RMSE: {min_rmse:.3f}")
    #     return min_rmse_by_position

    def plot_rmse_by_model_and_position(self):
        num_positions = len(self.df.columns)
        fig, axes = plt.subplots(num_positions, 1, figsize=(14, num_positions * 5))

        if num_positions == 1:
            axes = [axes]

        colors = {"train": "blue", "test": "orange", "cv": "green"}

        for ax, position in zip(axes, self.df.columns):
            data = {}
            for base_model in self.base_models:
                data[base_model] = self.df.loc[
                    [
                        f"{base_model}_train_rmse",
                        f"{base_model}_test_rmse",
                        f"{base_model}_cv_rmse",
                    ],
                    position,
                ].values

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
        plt.savefig(self.path + "position_rmse_comparison_by_model.png", dpi=300)
        plt.show()

    def plot_rmse_distribution(self):
        boxplot_data = []
        for base_model in self.base_models:
            for metric in ["train_rmse", "test_rmse", "cv_rmse"]:
                boxplot_data.append(
                    {
                        "Model": base_model,
                        "Metric": metric,
                        "RMSE": self.df.loc[f"{base_model}_{metric}", :].values,
                    }
                )

        boxplot_df = pd.DataFrame(boxplot_data)
        plt.figure(figsize=(14, 8))
        sns.boxplot(x="Model", y="RMSE", hue="Metric", data=boxplot_df.explode("RMSE"))
        plt.title("RMSE Distribution by Model and Metric")
        plt.ylabel("RMSE")
        plt.xlabel("Model")
        plt.legend(loc="upper right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.path + "rmse_distribution_by_model.png", dpi=300)
        plt.show()

    def plot_feature_correlation_heatmap(self, train_df):
        correlation_matrix = train_df.corr()
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(self.path + "feature_correlation_heatmap.png", dpi=300)
        plt.show()

    def plot_learning_curve(
        self,
        estimator,
        title,
        X,
        y,
        cv=None,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    ):
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("RMSE")

        train_sizes, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
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
        plt.savefig(self.path + f"learning_curve_{title}.png", dpi=300)
        plt.show()

    def plot_rmse_across_years(self, rmse_across_years):
        plt.figure(figsize=(14, 8))
        for year, rmse in rmse_across_years.items():
            plt.plot(self.base_models, rmse, label=f"RMSE for {year}")

        plt.title("Comparison of RMSE Across Years")
        plt.ylabel("RMSE")
        plt.xlabel("Model")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.path + "rmse_comparison_across_years.png", dpi=300)
        plt.show()


def run_analysis():
    analysis = FantasyFootballAnalysis()
    # analysis.find_best_estimator()
    # analysis.compare_rmse()
    analysis.plot_rmse_by_model_and_position()
    analysis.plot_rmse_distribution()
    # Assume `train_df` is the DataFrame with your features for the heatmap#
    # analysis.plot_feature_correlation_heatmap(train_df)# Assume `estimator`, `X`, and `y` are def ined for the learning curve#
    # analysis.plot_learning_curve(estimator, "ModelName", X, y)# Assume `rmse_across_years` is a dict of year: RMSE list#
    # analysis.plot_rmse_across_years(rmse_across_years)


if __name__ == "__main__":
    run_analysis()
