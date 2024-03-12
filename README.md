# Project_Zillow

All data is smoothed and collected on a weekly basis

Zillow Data Description
- `Days to Pending`: How long it takes homes in a region to change to pending status on Zillow.com after first being shown as for sale. The reported figure indicates the number of days (mean or median) that it took for homes that went pending during the week being reported, to go pending. This differs from the old “Days on Zillow” metric in that it excludes the in-contract period before a home sells.
- `Share of Listings With a Price Cut`: The number of unique properties with a list price at the end of the month that’s less than the list price at the beginning of the month, divided by the number of unique properties with an active listing at some point during the month.
- `Price Cuts`: The mean and median price cut for listings in a given region during a given time period, expressed as both dollars ($) and as a percentage (%) of list price.

---

# Project Zillow

Can we train our machine learning models to predict how much of a price cut (%) a listed house in Detroit will receive based on how long it’s been on the market? This project focuses on analyzing publicly avaliable Zillow data (which has been previously smoothed by Zillow), and collected on a weekly basis. The models specifically focus on 315 weeks of data from Detroit (394532 = ID); however, two weeks on this data were designated as Null. Therefore, after remvoing the week of April 11, 2020 & September 9, 2021, the machine learning models used 313 weeks of Detroit data. 

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

These models were compared, and it was determined that XGB performed the worst among them, which is why we used a best fit for that model, in order to create a.

## Getting Started

- Using Google Collaborator installing dependencies:
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

- Any other essential steps:

## Usage

Outline how users can utilize your project. Include examples and code snippets to guide them.

## Results

Highlight key findings or results from your analysis.

[Advanced-Regression-Analysis-Reference]

[Advanced-Regression-Analysis-Reference]: https://github.com/tatha04/Housing-Prices-Advanced-Regression-Techniques/blob/main/Housing.ipynb

---
