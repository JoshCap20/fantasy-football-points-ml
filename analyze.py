"""
Standalone script to visually analyze the results of the model training and testing.

Run this script after running main.py to generate the necessary results files.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Define a function to annotate bars in bar plots
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            1.05 * height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )


# Load the data
path = "results/"
pred = pd.read_csv(path + "position_rmse.csv")
fantasy = pd.read_csv(path + "fantasy_rmse.csv")

# Find and print the best estimator for each position
print("The best estimator for each position with the lowest RMSE is:\n")

min_rmse_by_position = {}
for position in pred.columns[1:]:
    min_rmse = pred[position].min()
    min_rmse_index = pred[position].idxmin()
    min_rmse_algo = pred.iloc[min_rmse_index, 0]
    min_rmse_by_position[position] = min_rmse
    print(f"{position}: {min_rmse_algo} with RMSE: {min_rmse:.3f}")

# Compare prediction RMSE to FantasyData RMSE
Fantasy_RMSE = []
for position in pred.columns[1:]:
    Fantasy_RMSE.append(
        float(fantasy.loc[fantasy["Pos"] == position, "diff"].values[0])
    )

# Bar plot to compare prediction RMSE to Fantasydata.com
positions = pred.columns[1:]
ind = np.arange(len(positions))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(
    ind, list(min_rmse_by_position.values()), width, color="r", label="Prediction"
)
rects2 = ax.bar(ind + width, Fantasy_RMSE, width, color="y", label="Fantasydata.com")

# Set labels and titles
ax.set_ylabel("RMSE")
ax.set_ylim([0, 13])
ax.set_title("RMSE Comparison by Position")
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(positions)
ax.legend()

# Annotate bars
autolabel(rects1, ax)
autolabel(rects2, ax)

plt.show()

# Now, plot RMSE by model for each position, grouping by model name and splitting into train, test, and cv
df = pd.read_csv(path + "position_rmse.csv")
df.set_index("Model", inplace=True)

# Extract unique base model names (before the underscore)
base_models = sorted(set(model.split("_")[0] for model in df.index))

# Determine the number of positions
num_positions = len(df.columns)

# Create subplots: one row per position
fig, axes = plt.subplots(num_positions, 1, figsize=(12, num_positions * 4))

# If there's only one subplot, axes won't be an array, so we make it one
if num_positions == 1:
    axes = [axes]

# Define consistent colors for train, test, and cv
colors = {"train": "blue", "test": "orange", "cv": "green"}

# Loop through each position and create a subplot for each
for ax, position in zip(axes, df.columns):
    # Prepare data for plotting
    data = {}
    for base_model in base_models:
        data[base_model] = df.loc[
            [
                f"{base_model}_train_rmse",
                f"{base_model}_test_rmse",
                f"{base_model}_cv_rmse",
            ],
            position,
        ].values

    # Plot each base model's train, test, and cv as grouped bars
    ind = np.arange(len(base_models))
    width = 0.2
    rects_train = []
    rects_test = []
    rects_cv = []
    for i, (base_model, values) in enumerate(data.items()):
        rects_train.append(
            ax.bar(
                ind[i] - width,
                values[0],
                width,
                color=colors["train"],
                label="Train" if i == 0 else "",
            )
        )
        rects_test.append(
            ax.bar(
                ind[i],
                values[1],
                width,
                color=colors["test"],
                label="Test" if i == 0 else "",
            )
        )
        rects_cv.append(
            ax.bar(
                ind[i] + width,
                values[2],
                width,
                color=colors["cv"],
                label="CV" if i == 0 else "",
            )
        )

        # Annotate bars
        autolabel(rects_train[-1], ax)
        autolabel(rects_test[-1], ax)
        autolabel(rects_cv[-1], ax)

    # Set up the subplot
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE by Model for Position: {position}")
    ax.set_xticks(ind)
    ax.set_xticklabels(base_models, rotation=45, ha="right")

# Set a single legend for all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
