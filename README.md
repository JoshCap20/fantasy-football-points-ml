# Fantasy Football Points Prediction ML Model

Fantasy is right around the corner so I'll go ahead and open source this for others to use and contribute to. You can change what years are used for training and testing data by changing the `TRAIN_YEARS` and `TEST_YEARS` variables in the `config.py` file. It will automatically scrape the data for the years you specify and train the models.

## Perfomance

### RMSE by Position grouped by Model

![RMSE for each Position by Model](./results/position_rmse_comparison_by_model.png)

### RSME Distribution by Model

![RMSE Distribution by Model](./results/rmse_distribution_by_model.png)

## Models

1. **Ride Regression** - Ridge regression is similar to linear regression however it contains a penalty term which increases as the feature coefficients increase.
2. **Bayesian Ridge Regression** - Bayesian ridge regression is similar to ridge regression however it includes information about the features to determine the penalty weight.
3. **Elastic Net Regularization** - Elastic net regularization applies a weighted average of the ridge regression and lasso regression penalties. 
4. **Random Forest** - Random forest is a tree-based machine learning algorithm which splits on randomly generated selection features in an attempt to prevent over-fitting.
5. **Gradient Boosting** - Gradient Boosting is also a tree-based method which learns from previous performance mistakes. A grid search was performed to optimize the parameters within the model.

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

The model will train and test on the years specified in the `config.py` file. The results will be outputed to the `results` folder.

## Validation

After running the main file, the model's accuracy is outputed by position and model in results/position_rmse.csv. The RMSE is calculated by taking the square root of the mean of the squared differences between the predicted and actual values. The RMSE is used to determine the accuracy of the models.

To visualize the results of the models like the above, run the following command:

```bash
python analyze.py
```

## Data

Data is dynamically scraped for input years from the nfl_data_py package. The data is then cleaned, aggregated, and transformed to be used in the models.
