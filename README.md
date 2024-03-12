Zillow Data Description
- `Days to Pending`: How long it takes homes in a region to change to pending status on Zillow.com after first being shown as for sale. The reported figure indicates the number of days (mean or median) that it took for homes that went pending during the week being reported, to go pending. This differs from the old “Days on Zillow” metric in that it excludes the in-contract period before a home sells.
- `Share of Listings With a Price Cut`: The number of unique properties with a list price at the end of the month that’s less than the list price at the beginning of the month, divided by the number of unique properties with an active listing at some point during the month.
- `Price Cuts`: The mean and median price cut for listings in a given region during a given time period, expressed as both dollars ($) and as a percentage (%) of list price.

---

# Project Zillow Introduction 

This machine learning project trains 3 different models to predict how much of a price cut (%) a listed house in Detroit will receive based on how long it’s been on the market (weeks of pending)? This project analyzes publicly avaliable Zillow data (previously smoothed by Zillow), and collected on a weekly basis. Our models specifically focus on 315 weeks of data from Detroit (394532 = ID); however, two of the provided weeks did not have any data, and thus, deemed as Null. After remvoing the weeks of April 11, 2020 and September 9, 2021, the machine learning models used 313 weeks of Detroit data. 

## Zillow Data Description

### Days to Pending
How long it takes homes in a Detroit to change to pending status on Zillow.com after first being shown as for sale. The reported figure indicates the mean number of days that it took for homes that went pending during the week being reported, to go pending.

### Price Cuts
- CThe mean price cut for listings in a Detroit during a given time period, expressed as a percentage (%) of list price.

### Time
The data was collected weekly across 315 weeks, allowing us to measure data across season and time. 


## Machine Learning Models

In our analysis, we employed the following machine learning models:

- Random Forest Regressor (RFR)
- GradientBoosting Regressor (GBR)
- Extreme Gradient Boosting (XGB)
- GridSearchCV (model = XGBRegressor)

These models were compared, and it was determined that XGB performed the worst among them, which is why we used a best fit for that model, in order to create a.

## Usage

Outline how users can utilize your project. Include examples and code snippets to guide them.


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

## Steps & Results: 

1. We loaded the data by reading in the csv file, during which, we designated the `Weeks of pending` as a datetime object using Pandas, because the column was originally stored as strings. 

2. After dropping the null values from the dataframe, we defined the features used for prediction (`x = Mean Price Reduction Percetage`) and the output we want to predict in a our machine learning models (`y = Days to Pending`).

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
<img src="https://github.com/mirasmitty/Project_Zillow/blob/main/Resources/Seaborn%20HeatMap%20Missing%20Values.png" width="500" height="450" border="10"/>
</p>

7. Preprocessing for our data was completed using the `OneHotEncoder` function from the scikit-learn library to encode the categorical features as binary vector of 0s and 1s, where each position in the vector corresponds to a unique category.

8. After presenting the `RandomForestRegressor` again, the relevant metrics for RFR Model are as follows:
    * The score is 0.9995553744147414.
    * The r2 is 0.9995553744147414.
    * The mean squared error is 1.3095197551880405e-08.
    * The root mean squared error is 0.00011443424990744863.
    * The standard deviation is 0.005426987263266367.
    * The error is 5.7804842222217705e-05 

9. We then used the `GradientBoostingRegressor` function from the scikit-learn library and the relevant metrics for GBR Model are as follows:
    * The score is 0.9995463581799123.
    * The r2 is 0.9995463581799123.
    * The mean squared error is 1.3360745419963808e-08.
    * The root mean squared error is 0.00011558869070961833.
    * The standard deviation is 0.005426987263266367.
    * The error is 6.293934425514186e-05 

10. We then used the `XGBRegressor` function from the scikit-learn library and the relevant metrics for XGB Model are as follows:
    * The score is 0.998024236380367.
    * The r2 is 0.998024236380367.
    * The mean squared error is 5.8190567013516395e-08.
    * The root mean squared error is 0.00024122721035056636.
    * The standard deviation is 0.005426987263266367.
    * The error is 0.00013350071454126876 

11. We then used the `GridSearchCV` function from the scikit-learn library to optimize the hyperparameters of the XGB model, because it was our lowest-scoring model with a 99.8% accuracy. The best fit 


## Results

Highlight key findings or results from your analysis.

[Advanced-Regression-Analysis-Reference]

[Advanced-Regression-Analysis-Reference]: https://github.com/tatha04/Housing-Prices-Advanced-Regression-Techniques/blob/main/Housing.ipynb

---
