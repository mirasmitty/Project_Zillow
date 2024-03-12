
<img style="float: right;" src="https://github.com/mirasmitty/Project_Zillow/blob/main/Resources/DealOrNoDeal.png">
Team Members:

*	Stephanie Santiago 
*	Molleigh Hughes 
*	Hidy Vengalil 
*	Tasha Christensen 
*	Miranda Smith

# `Deal or No Deal?` 
## Zillow Price Cut Prediction Introduction 

This machine learning project develops three distinct models aimed at predicting the percentage of price reduction that a listed house in Detroit is likely to undergo based on its duration on the market (in weeks). The analysis utilizes publicly accessible Zillow data, which has been previously processed by Zillow and is updated on a weekly basis. The focus of the models is on 315 weeks of data specifically from Detroit, identified by the ID 394532. However, it was found that two of the provided weeks lacked data & therefore, were deemed as Null. After removing the weeks of April 11, 2020, & September 9, 2021, the machine learning models were trained using 313 weeks of data from Detroit.
* [Advanced-Regression-Analysis] reference found on GitHub and available for other's to learn from according to (2021) MIT License

[Advanced-Regression-Analysis]: https://github.com/tatha04/Housing-Prices-Advanced-Regression-Techniques/blob/main/Housing.ipynb


## Zillow's Data Description
- `Days to Pending`: This metric measures the duration it takes for homes in a specific area to transition from being listed for sale to pending status on Zillow.com. It reflects the average or median number of days it took for homes that entered pending status within the reported week. Unlike the previous "Days on Zillow" metric, Days to Pending excludes the time period during which a home is under contract before it is sold.
- `Share of Listings With a Price Cut`: This metric calculates the proportion of distinct properties that have a final list price at the conclusion of the month lower than the initial list price at the beginning of the month. It is determined by dividing the number of unique properties with a reduced list price by the total number of unique properties that had an active listing at any time during the month.
- `Price Cuts`: This data presents the average and median reduction in listing prices within a specified region over a certain timeframe. It is represented both in monetary terms ($) and as a **percentage (%) of the original list price**.

## Machine Learning Models

In our analysis, we employed the following machine learning models:

- **Random Forest Regressor** (RFR): a versatile machine-learning approach that leverages the predictions of numerous decision trees to mitigate overfitting and enhance accuracy. RFR is a supervised learning algorithm and can be applied to both classification and regression. 
- **GradientBoosting Regressor** (GBR): a machine learning algorithm that works by sequentially fitting multiple decision trees, where each tree corrects the errors of the previous one. GBR is well-suited for regression tasks, aiming to predict continuous numerical values. This iterative process gradually improves the model's predictive accuracy, and combines the predictive power of multiple weak learners (decision trees) to create a robust and accurate predictive model. 
- **Extreme Gradient Boosting Regressor** (XGBR):
- **GridSearchCV** (model = XGBRegressor):

These models were compared, and it was determined that XGB performed the worst among them, which is why we used a best fit for that model by using GridSearchCV.


## Getting Started

- Using Google Collaborator & installing dependencies:
    - `import pandas as pd`
    - `import numpy as np`
    - `from sklearn.preprocessing import StandardScaler`
    - `from sklearn.model_selection import train_test_split`
    - `from sklearn.ensemble import RandomForestRegressor`
    - `import matplotlib.pyplot as plt`
    - `from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_error`
    - `import seaborn as sns`
    - `from sklearn.preprocessing import OneHotEncoder`
    - `from sklearn.ensemble import GradientBoostingRegressor`
    - `from xgboost import XGBRegressor`
    - `from sklearn.model_selection import GridSearchCV` (for XGBRegressor)

## Steps: 

1. We loaded the data by reading in the csv file, during which, we designated the `Weeks of pending` as a datetime object using Pandas, because the column was originally stored as strings. 

2. After dropping the null values from the dataframe, we defined the features used for prediction (`x =weeks of pending , Days to Pending`) and the output we want to predict in a our machine learning models (`y = Mean Price Reduction Percetage`).
 
3. We then used the `train_test_split` function from skscikit-learn library to split the features (`x`) and the target variable (`y`) into training and testing sets. The resulting variables `x_train`, `x_test`, `y_train`, and `y_test` store the training & testing sets for both features and the target variable. 
    * After this step, we use `x_train` and `y_train` for training our machine learning models, and `x_test` and `y_test` for evaluating its performance on unseen data. This design is used to assess how well the model generalizes to new, unseen observations.

4. After creating the Standar Scaler, fitting the Standard Scaler, and scaling the data, we used the `RandomForestRegressor` function from the scikit-learn library to train our Random Forest Regressor model.
    * Then, we extract the feature importances from a trained Random Forest model (`rf_model`) using the `feature_importances` attribute 
        * `Feature importance`: a measure of the contribution of each feature to the model's predictive performance

5. The Feature Importances are visualized using a horizontal bar plot. The features are shown on the y-axis and their importances are represented by the length of the bars. This visualization provides insights into which features are more influential in making predictions with the Random Forest Regressor model. 
<p align="center">
<img src="https://github.com/mirasmitty/Project_Zillow/blob/main/Resources/Features%20Importances%20Horizontal%20Plot.png" width="600" height="400" border="10"/>
</p>

6. Next, we imported seaborn to build a Heatmap of our "missing values"; however, because we previously filtered out our null values, we did not see any missing values in our dataset. This provides reasurrance that our data is ready for our other machine learning models. 
<p align="center">
<img src="https://github.com/mirasmitty/Project_Zillow/blob/main/Resources/Seaborn%20HeatMap%20Missing%20Values.png" width="800" height="450" border="10"/>
</p>

7. Preprocessing for our data was completed using the `OneHotEncoder` function from the scikit-learn library to encode the categorical features as binary vector of 0s and 1s, where each position in the vector corresponds to a unique category.

8. After presenting the `RandomForestRegressor` again, the relevant metrics for RFR Model are as follows:
    * The score is 0.9995553744147414.
    * The r2 is 0.9995553744147414.
    * The mean squared error is 1.3095197551880405e-08.
    * The root mean squared error is 0.00011443424990744863.
    * The standard deviation is 0.005426987263266367.
    * The error is 5.7804842222217705e-05 
    * The mean absolute error is 0.00011443424990744863.
        * The results of the RFR Model metrics reveal important information about our RFR Model: 
            * <samp>Because R-squared values range from 0 to 1, where 1 indicates a perfect fit, a score of 0.9995553744147414 indicates that the model explains approximately 99.96% of the variance in the target variable.</samp>
            * <samp>A very small (MSE) mean squared error (1.3095197551880405e-08) indicates that, on average, the model's predictions are very close to the actual values.</samp>
            * <samp>The standard deviation is a measure of the amount of variation in a set of values. In this context, it's associated with the residuals (the difference between predicted and actual values). A smaller standard deviation (0.005426987263266367) indicates less variability in the residuals.</samp>
* <ins>RFR Summary</ins>: RFR Model appears to be performing exceptionally well, with high accuracy and precision in predicting the target variable. 


9. We then used the `GradientBoostingRegressor` function from the scikit-learn library and the relevant metrics for GBR Model are as follows:
    * The score is 0.9995463581799123.
    * The r2 is 0.9995463581799123.
    * The mean squared error is 1.3360745419963808e-08.
    * The root mean squared error is 0.00011558869070961833.
    * The standard deviation is 0.005426987263266367.
    * The error is 6.293934425514186e-05 
        * The results of the GBR Model metrics reveal important information about our GBR model: 
            * <samp>The score (R-squared) is a measure of how well the regression model explains the variance in the target variable. A high R-squared value, 0.9995463581799123, indicates that the model explains a large portion of the variance in the target variable, approximately 99.95%. </samp>
            * <samp>The root mean squared error is the square root of the mean sqaured error and is expressed in the same units as the target variable. A small RMSE (0.00011558869070961833) suggests that, on average, the model's predictions are close to the actual values.</samp>
            * <samp>A small error value (6.293934425514186e-05 = 0.00006293934425514186) suggests that, on average, the model's predictions are very close to the actual values</samp>
* <ins>GBR Summary</ins>: the outcome metrics for the GradientBoostingRegressor Model indicate excellent performance. The model is explaining a large proportion  of the variance in the target variable, approximately 99.95%, and the predictions are very accurate. 

10. We then used the `ExtremeGradientBoosting` function from the scikit-learn library and the relevant metrics for XGBR Model are as follows:
    * The score is 0.998024236380367.
    * The r2 is 0.998024236380367.
    * The mean squared error is 5.8190567013516395e-08.
    * The root mean squared error is 0.00024122721035056636.
    * The standard deviation is 0.005426987263266367.
    * The error is 0.00013350071454126876 
        * The results of the XGBR Model metrics reveal important information about our XGBR model: 
            * <samp>A high R-squared value, 0.998024236380367, indicates that the model explains a large portion of the varaince in the target variable, approximately 99.80%. Interestingly, this is our lowest scoring model. </samp>
* <ins>XGBR Summary</ins>: the outcome metrics for the XGBRegressor Model indicate excellent performance. The model is explaining a large proportion  of the variance in the target variable, approximately 99.80%, and the predictions are very accurate. The plot below is a visualization of the RandomForestRegressor, GradientBoostingRegressor, and ExtremeGradientBoosting models all in one chart. 
<p align="center">
<img src="https://github.com/mirasmitty/Project_Zillow/blob/main/Resources/RFR_GBR_XGBR_Plot.png" width="600" height="500" border="10"/>
</p>

11. Finally, using Grid Search Cross Validation, to perform a hyperparameter tuning for the XGBoost Regressor model; the code imports the `GridSearchCV` class from scikit-learn & the `XGBRegressor` class from the XGBoost library. The evaluation metric to optimize during the grid search, in this case, the "negative mean absolute error", is set as the parameter `scoring`. The negative mean absolute error is used, because of conventional optimization and consistency. The following "best fit model" negates the negative mean absolute error values, and because lower values are desirable, the optimization seeks to minimize the error. 
<p align="center">
<img src="https://github.com/mirasmitty/Project_Zillow/blob/main/Resources/ALL_MODELS_PLOT.png" width="600" height="500" border="10"/>
</p>

* *Negative mean absolute error* is a performance metric used in regression problems, typically with machine learning. The mean absolute error (*MAE*) is a measure of the average absolute differences between the predicted and actual values. The negative version is used in certain libraries like scikit-learn for optimization purposes, as it aligns with the convention that high values are better for the optimization process. 
    * Mean Absolute Error (*MAE*): is calculated as the average of the absolute differences between predicted and actual values.
        *  $MAE = {1 \over n} \sum{^n_{i=1}} |y_i - \hat{y}_i|$ , where $n$ is the number of samples, $y_i$ is the actual value and $\hat{y}_i$ is the predicted value.
    * Negative Mean Absolute Error (*NMAE*): 
        * In some optimization frameworks like scikit-learn's GridSearchCV, the goal is to maximize the scoring metric during hyperparameter tuning.
            * Since lower *MAE* values are desired (indicating better model performance), negating the *MAE* makes it consistent with the optimization goal. The larger the negated *MAE*, the better the model performance. 

    

## Results:
<p align="center">
<img src="" width="600" height="500" border="10"/>
</p>



---
