# Fantasy Football Points Prediction ML Model

Fantasy is right around the corner so I'll go ahead and open source this for others to use and contribute to. You can change what years are used for training and testing data by changing the `TRAIN_YEARS` and `TEST_YEARS` variables in the `config.py` file. It will automatically scrape the data for the years you specify and train the models.

## Perfomance

### RMSE by Position grouped by Model

![RMSE for each Position by Model](./output/position_rmse_comparison_by_model.png)

### RSME Distribution by Model

![RMSE Distribution by Model](./output/rmse_distribution_by_model.png)

## Models

1. **Elastic Net Regularization** - Elastic Net combines both L1 and L2 penalties of Lasso and Ridge, respectively, making it effective for handling datasets with multicollinearity and for selecting correlated features. This hybrid approach balances feature selection and regularization, reducing overfitting while retaining important predictors.

2. **Random Forest Regressor** - An ensemble learning method that constructs multiple decision trees during training and outputs the average prediction of the individual trees. Its ensemble approach enhances robustness and reduces the likelihood of overfitting, making it reliable for datasets with high variance or noise.

3. **Gradient Boosting Regressor** - This model builds an ensemble of weak learners, typically decision trees, sequentially, where each tree corrects the errors of the previous one. It is highly effective for capturing complex, non-linear relationships, often resulting in superior predictive performance on structured data.

4. **CatBoost Regressor** - A gradient boosting algorithm specifically optimized for categorical features, offering excellent performance with minimal hyperparameter tuning. Its ability to handle categorical data natively and its automatic feature combination make it particularly strong in real-world datasets with mixed data types.

5. **K-Nearest Neighbors Regressor** - A non-parametric model that predicts the target value based on the average of the nearest k neighbors in the feature space. It is intuitive and effective for capturing local patterns in the data, though its performance can degrade with high-dimensional data or noisy features.

*Each position has separate groups of models.*

New models can easily be added and compared by adding them to the models dictionary in `model_training.py/train_model()`. The perfomance will also be outputted in the results csv. New features in the data can be added by adding them in `feature_engineering.py/add_features().`

## Running the Model

To run the model, first install the required packages:

```bash
pip install -r requirements.txt
```

Then run the following command:

```bash
python main.py
```

The model will train and test on the years specified in the `config.py` file. The results will be outputed to the `output` folder.

## Validation - DEPRECATED

*While this still works, it is automatically ran after the main file is ran now.*  

After running the main file, the model's accuracy is outputed by position and model in results/rmse.csv. The RMSE is calculated by taking the square root of the mean of the squared differences between the predicted and actual values. The RMSE is used to determine the accuracy of the models.

To visualize the results of the models like the above, run the following command:

```bash
python analyze.py
```

## TODO

- Combine models for each position into one model for each position.

## Data

Data is dynamically scraped for input years from the nfl_data_py package. The data is then cleaned, aggregated, and transformed to be used in the models.
