Copy---
layout: default
title: "Time Series Market Extreme Prediction"
permalink: /Time-Series-Reg
---
# Project Overview

This project aims to develop a trading strategy capable of predicting corruent day market price range. Specifically, The goal is to forecast potential upward and downward deviations range from the market's opening price within a single trading session.The primary objective is to apply quantitative modeling techniques and machine learning methods to intraday price movements, rather than to achieve high predictive accuracy.

# Table of Contents
1. [Introduction](#introduction)  
2. [Data Acquisition](#data-acquisition)  
3. [Data Pre-processing](#data-preprocessing)  
4. [Time Series Regression Model](#time-series-regression-model)  
5. [Feature Engineering](#feature-engineering)  
6. [Strategy Return and Performance](#strategy-return-and-performance)  
7. [Future Work](#future-work)  

## <a id="introduction"></a> Introduction

This project seeks to create a trading strategy that can predict the market price range for the current day. Specifically, the goal is to forecast potential upward and downward deviations from the market's opening price within a single trading session. The primary focus is not on achieving high predictive accuracy but rather on applying quantitative modeling techniques and machine learning methods to analyze intraday price movements. By doing so, the project aims to provide valuable insights into the potential price ranges that traders can expect throughout the day.

To achieve this, data from the SPDR Gold Trust ETF (GLD) is utilized, as it offers a widely traded and accessible representation of gold price movements. GLD allows investors to gain exposure to gold price dynamics without engaging in the complexities of trading futures contracts or handling physical gold. By focusing on GLD, the project leverages a liquid asset that closely mirrors gold prices, aligning with the objective of forecasting high and low deviations from the market's opening price. This analysis involves sourcing, preprocessing, modeling, and evaluating financial data using a suite of libraries, ultimately enhancing the interpretability of the results and aiding in the communication of key insights.


```python
# Importing necessary libraries 
import yfinance as yf
import pandas as pd
import numpy as np
import pyfolio as pf 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid')

# To ignore unwanted warnings
import warnings
warnings.filterwarnings("ignore")
```

## <a id="data-acquisition"></a> Data Acquisition


```python
# Fetching historical data
ticker ='GLD' 
GLD = yf.download(ticker, start= '2018-10-10', end= '2024-10-10')
GLD_df = GLD.drop(['Adj Close', 'Volume'], axis=1)
# Ensuring data is Fetched
if GLD_df.empty: 
    raise exception("Failed to download data from ticker")

# Plotting GLD chart
plt.figure(figsize=(10,6))
plt.plot(GLD_df['Close'], label='Gold ETF',color='blue')
plt.title(' SPDR Gold Trust ETF (GLD) ')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
```

    [*********************100%%**********************]  1 of 1 completed
    


    
![png](output_5_1.png)
    



```python
GLD_df.isna().sum()
```




    Open     0
    High     0
    Low      0
    Close    0
    dtype: int64



## <a id="data-preprocessing"></a> Data Pre-processing

The OHLCV DataFrame we're using contains time series data with fields for Open, High, Low, Close, Volume, and Adjusted Close prices. After removing the Volume and Adjusted Close columns, we need to prepare the remaining data for modeling. This will involve data preprocessing and manipulation, as well as the creation of key predictive indicators, which are outlined as follows:  

###### Custom Indicators
To calculate custom indicators, we start with:
- **STD\_U** (Upward price movement) for the day, defined as:
  $$ \text{STD_U} = \text{Open} - \text{High} $$
- **STD\_D** (Downward price movement) for the day, defined as:
  $$ \text{STD_D} = \text{Open} - \text{Low} $$

###### Customise Moving Averages (MA)
We add 3-day, 15-day, and 60-day moving averages, which help us analyze both short-term and long-term price trends:
$$ \text{MA}_n = \frac{1}{n} \sum_{i=1}^{n} \text{Price}_i $$

where \( n \) is the number of days (3, 15, or 60).

###### Price Change Indicator
To observe daily opening momentum, we calculate the difference between today’s open and yesterday’s open:
$$ \Delta \text{Open} = \text{Open}_{\text{today}} - \text{Open}_{\text{yesterday}} $$

###### Correlation Indicator
Using the Pandas correlation function, we add a correlation metric to see the relationship between today’s closing price and the 3-day moving average.

###### Overnight Change
Finally, we calculate overnight price changes by comparing today’s open with the previous day’s close:
$$ \text{Overnight Change} = \text{Open}_{\text{today}} - \text{Close}_{\text{yesterday}} $$



```python
# Calculate 3, 15, and 60 days MA of close prices
GLD_df['MA_3'] = GLD_df['Close'].shift(1).rolling(window=3).mean()
GLD_df['MA_15'] = GLD_df['Close'].shift(1).rolling(window=15).mean()
GLD_df['MA_60'] = GLD_df['Close'].shift(1).rolling(window=60).mean()

# Calculate the correlation between 3 days MA and trading Session Close prices
GLD_df['Corr'] = GLD_df['Close'].shift(1).rolling(window=10).corr(GLD_df['MA_3'].shift(1))

# Calculate the difference between High & Open, Open & Low
GLD_df['Std_U'] = GLD_df['High']-GLD_df['Open']
GLD_df['Std_D'] = GLD_df['Open']-GLD_df['Low']

# Calculate the difference between `Open` & previous day's `Open`, and `Open` & previous day's `Close`
GLD_df['OD'] = GLD_df['Open']-GLD_df['Open'].shift(1)
GLD_df['OL'] = GLD_df['Open']-GLD_df['Close'].shift(1)

GLD_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>MA_3</th>
      <th>MA_15</th>
      <th>MA_60</th>
      <th>Corr</th>
      <th>Std_U</th>
      <th>Std_D</th>
      <th>OD</th>
      <th>OL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-10-03</th>
      <td>244.720001</td>
      <td>245.910004</td>
      <td>243.690002</td>
      <td>245.490005</td>
      <td>244.776667</td>
      <td>241.913999</td>
      <td>230.678666</td>
      <td>0.717001</td>
      <td>1.190002</td>
      <td>1.029999</td>
      <td>-0.889999</td>
      <td>-0.940002</td>
    </tr>
    <tr>
      <th>2024-10-04</th>
      <td>245.000000</td>
      <td>246.690002</td>
      <td>244.050003</td>
      <td>245.000000</td>
      <td>245.586670</td>
      <td>242.524666</td>
      <td>231.114167</td>
      <td>0.578269</td>
      <td>1.690002</td>
      <td>0.949997</td>
      <td>0.279999</td>
      <td>-0.490005</td>
    </tr>
    <tr>
      <th>2024-10-07</th>
      <td>244.580002</td>
      <td>244.820007</td>
      <td>243.809998</td>
      <td>244.169998</td>
      <td>245.383336</td>
      <td>242.946000</td>
      <td>231.476667</td>
      <td>0.291136</td>
      <td>0.240005</td>
      <td>0.770004</td>
      <td>-0.419998</td>
      <td>-0.419998</td>
    </tr>
    <tr>
      <th>2024-10-08</th>
      <td>243.789993</td>
      <td>244.039993</td>
      <td>240.630005</td>
      <td>242.369995</td>
      <td>244.886668</td>
      <td>243.313333</td>
      <td>231.827666</td>
      <td>-0.514362</td>
      <td>0.250000</td>
      <td>3.159988</td>
      <td>-0.790009</td>
      <td>-0.380005</td>
    </tr>
    <tr>
      <th>2024-10-09</th>
      <td>241.160004</td>
      <td>241.839996</td>
      <td>240.639999</td>
      <td>241.050003</td>
      <td>243.846664</td>
      <td>243.648666</td>
      <td>232.136666</td>
      <td>-0.373007</td>
      <td>0.679993</td>
      <td>0.520004</td>
      <td>-2.629990</td>
      <td>-1.209991</td>
    </tr>
  </tbody>
</table>
</div>



##### Creating X and y data set 
we will feed input datasets and yU and yDfor feeding into the machine learning linear regression model that we are going to build. The model has two dependent variables, $ y_U $ (for upward deviation) and $ y_D $ (for downward deviation), and a set of independent variables $ X_t $. The independent variables include the features:

$$
X_t = \left[ \text{MA}_{3,t}, \text{MA}_{15,t}, \text{MA}_{60,t}, OD_t, OL_t, \text{Corr}_t \right]
$$

where each $ X_t $ contains the values of the predictors (Moving Averages, Operational Metrics, etc.) at time $ t $, and $ \text{Corr}_t $ represents the correlation at time $ t $, which can be written as:

## <a id="time-series-regression-model"></a> Time Series Regression Model


For predicting the upward deviation $ y_{U,t} $ at time \( t \), we can use a linear regression model:

$$
y_{U,t} = \beta_0 + \beta_1 \cdot \text{MA}_{3,t} + \beta_2 \cdot \text{MA}_{15,t} + \beta_3 \cdot \text{MA}_{60,t} + \beta_4 \cdot OD_t + \beta_5 \cdot OL_t + \beta_6 \cdot \text{Corr}_t + \epsilon_{U,t}
$$

Similarly, for the downward deviation $ y_{D,t} $ at time \( t \), the model would be:

$$
y_{D,t} = \alpha_0 + \alpha_1 \cdot \text{MA}_{3,t} + \alpha_2 \cdot \text{MA}_{15,t} + \alpha_3 \cdot \text{MA}_{60,t} + \alpha_4 \cdot OD_t + \alpha_5 \cdot OL_t + \alpha_6 \cdot \text{Corr}_t + \epsilon_{D,t}
$$

where:
- $ X_t $ is the feature vector at time $ t $,
- $ \beta $ and $ \alpha $ are the learned coefficients for each model (upward and downward deviation, respectively),
- $ \epsilon_{U,t} $ and $ \epsilon_{D,t} $ are the error terms for each prediction.



```python
# Independent variable 
X = GLD_df[['Open','MA_3','MA_15','MA_60','OD','OL','Corr',]]
X.isna().sum()
```




    Open     0
    MA_3     0
    MA_15    0
    MA_60    0
    OD       0
    OL       0
    Corr     0
    dtype: int64




```python
# Depenedent variable for upward deviation 
yU = GLD_df['Std_U']
# Dependent varibale for downward deviation 
yD = GLD_df['Std_D']
```


```python
GLD_df.isna().sum()
```




    Open     0
    High     0
    Low      0
    Close    0
    MA_3     0
    MA_15    0
    MA_60    0
    Corr     0
    Std_U    0
    Std_D    0
    OD       0
    OL       0
    dtype: int64



We have 60 NaN values in `MA_60`, 15 NaN values in `MA_15`,13 NaN values in `Corr` and 3 NaN values in `MA_3` etc. Now we will simply drop all the NaN values using `dropna'. 


```python
# Dropping all the NaN
GLD_df.dropna(inplace=True)

# Checking for NaN values
GLD_df.isna().sum()
```




    Open     0
    High     0
    Low      0
    Close    0
    MA_3     0
    MA_15    0
    MA_60    0
    Corr     0
    Std_U    0
    Std_D    0
    OD       0
    OL       0
    dtype: int64



## <a id="feature-engineering"></a> Feature Engineering

In this section we will apply Scaling, Imputation And Pipeline Integration which are necessary for modelling. 

To ensure that the model can learn effectively from all features, we standardized the dataset and handled missing data through imputation. First, we observed that certain features had variances significantly larger than others, which could dominate the objective function and impair the model's ability to learn correctly. To prevent this, we applied the Standard Scaler function, which centered the data (by reducing the mean to zero) and scaled it (by dividing each entry by the standard deviation, making it equal to one). This standardization is crucial for predictive models, especially for linear regression, where the model performs best when the feature magnitudes are similar.

For the missing values in the dataset, we chose to replace them with the most frequent value (mode) rather than the mean or median. The mean can distort the distribution and underestimate the standard deviation, leading to potential errors. The median alters the data's mean, which might not be desirable in some cases. The mode, however, preserves the distribution's integrity while addressing missing values, which is why we used it. This approach was implemented in the pipeline using the SimpleImputer function with the strategy set to 'most_frequent'. We then combined this step with the scaling and regression steps in a pipeline for seamless processing.


```python
# Setting steps of the pipeline
steps = [
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),  
    ('scaler', StandardScaler()),                
    ('linear', LinearRegression())]
# Defining the pipeline 
pipeline = Pipeline(steps)
```

#### Hyperparameters

In machine learning, certain parameters, known as hyperparameters, cannot be estimated directly from the training data but are essential for optimizing model performance. For our linear regression model, we focus on the intercept as a key hyperparameter. By tuning this hyperparameter, we aim to enhance the model's accuracy and overall performance.

To determine whether the intercept should be included, we use the fit_intercept function. This boolean function allows us to decide if the model should compute an intercept (value of 1) or omit it (value of 0), based on which configuration yields the best results. This simple yet crucial step helps fine-tune the model for better predictive accuracy.


```python
# Using intercept of the linear model as a hyperparameter 
parameters = {'linear__fit_intercept': [0, 1]}
```


```python
GLD_df.isna().sum()
```




    Open     0
    High     0
    Low      0
    Close    0
    MA_3     0
    MA_15    0
    MA_60    0
    Corr     0
    Std_U    0
    Std_D    0
    OD       0
    OL       0
    dtype: int64



#### Grid Search Cross-Validation
 To assess the model’s performance in real-world scenarios and mitigate overfitting, we implement cross-validation. We utilize the GridSearchCV function, an efficient tool for performing exhaustive search over specified hyperparameters, ensuring that the model generalizes well.

We set cv=5 for the grid search, meaning the data will be split into five subsets for cross-validation, providing a robust performance evaluation by averaging the results across these rounds. We opt for GridSearchCV over RandomizedSearchCV due to the relatively smaller number of features in our model, making a grid search more efficient. Additionally, we use TimeSeriesSplit for partitioning the training data, ensuring that temporal dependencies are respected when splitting the data. This comprehensive approach helps fine-tune model parameters, like the intercept, to achieve optimal performance.


```python
# Spliting the time series for Grid Search Cross Validation
kf = TimeSeriesSplit(n_splits=5)

# Defining reg as a variable for Gridserech function contatining pipeline, Hyperparameter 
reg = GridSearchCV(pipeline, parameters, cv=kf)
```

#### Split Train and Test Data

Now, we will split data into train and test data sets. 

1. First, 70% of data is used for training and the remaining data for testing.
2. Fit the training data to a grid search function.


```python
Splitting_ratio = 0.7
# Splitting the data into two parts
# Using int to make sure integer number comes out.
split = int(Splitting_ratio*len(GLD_df))

# Defining train dataset
X_train = X[:split]
yU_train = yU[:split]
yD_train = yD[:split]

# Defining test data
X_test = X[split:]
```

#### Predicting Market High and Low 

we will apply the linear regression model to the training dataset and use it to predict upward deviations in the test dataset. These predictions will serve as the foundation for developing a trading application. Specifically, we will predict deviations in the high and low prices relative to the opening price. To generate actual market predictions, we will add and subtract these deviations from the open price, respectively, to obtain the predicted market highs and lows.

We will create two new columns in the dataframe, max_u and max_d, to store the predicted values. However, we must address a key practical consideration: while the linear regression model can generate negative predictions, it is not realistic for the high of the day to be lower than the opening price or for the low to be higher. To account for this, we will trim the predictions to ensure they fall within feasible ranges. Finally, we will calculate the predicted high (p_h) by adding max_u to the open price, and the predicted low (p_l) by subtracting max_d from the open price, populating these values in new columns. This process bridges the model's predictions to actionable trading insights.



```python
# fit model
reg.fit(X_train, yU_train)

# Print best parameter
print(reg.best_params_)
```

    {'linear__fit_intercept': 1}
    


```python
# Predict the upward deviation
yU_predict = reg.predict(X_test)
```


```python
# Fit the model
reg.fit(X_train, yD_train)

# Print best parameter
print(reg.best_params_)

# Predict the downward deviation
yD_predict = reg.predict(X_test)
```

    {'linear__fit_intercept': 1}
    

Now we will create `yU_predict` and `yD_predict` columns in the `X_test`.Formulas for upward deviation and downward deviation are given by:

Upward deviation  = High - Open

Downward deviation = Open - Low

It is clear from the above two formulas that upward and downward deviation can not be negative. So, we replace negative values with zero.



```python
# Create new column in X_test
X_test['yU_predict'] = yU_predict
X_test['yD_predict'] = yD_predict

# Assign zero to all the negative predicted values to take into account real life conditions
X_test.loc[X_test['yU_predict']< 0, 'yU_predict'] = 0
X_test.loc[X_test['yD_predict']< 0, 'yD_predict'] = 0
```

##### Generating Market High and Low Predictions for Trading
Using predictions from our linear regression model, we calculate market high and low values for trading. We first predict deviations in high and low prices relative to the open, then store these in max_u and max_d columns. Adding max_u to the open price provides the predicted high (p_h), while subtracting max_d gives the predicted low (p_l). To maintain realistic values, we trim predictions, ensuring the high is never below and the low never above the open. This adjustment aligns predictions with practical market behavior, enhancing the trading model's reliability.


```python
# Add open values in ['yU_predict'] to get the predicted high column
X_test['P_H'] = X_test['Open']+X_test['yU_predict'].shift(1)

# Subtract ['yD_predict'] values in open to get the predicted low column.
X_test['P_L'] = X_test['Open']-X_test['yD_predict'].shift(1)

# Print tail of GLD-df dataframe
X_test.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>MA_3</th>
      <th>MA_15</th>
      <th>MA_60</th>
      <th>OD</th>
      <th>OL</th>
      <th>Corr</th>
      <th>yU_predict</th>
      <th>yD_predict</th>
      <th>P_H</th>
      <th>P_L</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-10-03</th>
      <td>244.720001</td>
      <td>244.776667</td>
      <td>241.913999</td>
      <td>230.678666</td>
      <td>-0.889999</td>
      <td>-0.940002</td>
      <td>0.717001</td>
      <td>1.256133</td>
      <td>1.977312</td>
      <td>245.950930</td>
      <td>242.783379</td>
    </tr>
    <tr>
      <th>2024-10-04</th>
      <td>245.000000</td>
      <td>245.586670</td>
      <td>242.524666</td>
      <td>231.114167</td>
      <td>0.279999</td>
      <td>-0.490005</td>
      <td>0.578269</td>
      <td>1.293902</td>
      <td>1.944456</td>
      <td>246.256133</td>
      <td>243.022688</td>
    </tr>
    <tr>
      <th>2024-10-07</th>
      <td>244.580002</td>
      <td>245.383336</td>
      <td>242.946000</td>
      <td>231.476667</td>
      <td>-0.419998</td>
      <td>-0.419998</td>
      <td>0.291136</td>
      <td>1.288878</td>
      <td>1.955655</td>
      <td>245.873904</td>
      <td>242.635546</td>
    </tr>
    <tr>
      <th>2024-10-08</th>
      <td>243.789993</td>
      <td>244.886668</td>
      <td>243.313333</td>
      <td>231.827666</td>
      <td>-0.790009</td>
      <td>-0.380005</td>
      <td>-0.514362</td>
      <td>1.256892</td>
      <td>1.939375</td>
      <td>245.078871</td>
      <td>241.834338</td>
    </tr>
    <tr>
      <th>2024-10-09</th>
      <td>241.160004</td>
      <td>243.846664</td>
      <td>243.648666</td>
      <td>232.136666</td>
      <td>-2.629990</td>
      <td>-1.209991</td>
      <td>-0.373007</td>
      <td>1.305928</td>
      <td>1.967981</td>
      <td>242.416896</td>
      <td>239.220629</td>
    </tr>
  </tbody>
</table>
</div>



Here we add the `Close`, `High`, and `Low` columns from `gold_prices` because we will need all these columns to calculate strategy returns in the following notebook.
We are using the split function to get only the test part of the `gold_prices`.


```python
# Copy columns from GLD-df to X_test
X_test[['Close', 'High', 'Low']] = GLD_df[['Close', 'High', 'Low']][split:]
X_test.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>MA_3</th>
      <th>MA_15</th>
      <th>MA_60</th>
      <th>OD</th>
      <th>OL</th>
      <th>Corr</th>
      <th>yU_predict</th>
      <th>yD_predict</th>
      <th>P_H</th>
      <th>P_L</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-10-03</th>
      <td>244.720001</td>
      <td>244.776667</td>
      <td>241.913999</td>
      <td>230.678666</td>
      <td>-0.889999</td>
      <td>-0.940002</td>
      <td>0.717001</td>
      <td>1.256133</td>
      <td>1.977312</td>
      <td>245.950930</td>
      <td>242.783379</td>
      <td>245.490005</td>
      <td>245.910004</td>
      <td>243.690002</td>
    </tr>
    <tr>
      <th>2024-10-04</th>
      <td>245.000000</td>
      <td>245.586670</td>
      <td>242.524666</td>
      <td>231.114167</td>
      <td>0.279999</td>
      <td>-0.490005</td>
      <td>0.578269</td>
      <td>1.293902</td>
      <td>1.944456</td>
      <td>246.256133</td>
      <td>243.022688</td>
      <td>245.000000</td>
      <td>246.690002</td>
      <td>244.050003</td>
    </tr>
    <tr>
      <th>2024-10-07</th>
      <td>244.580002</td>
      <td>245.383336</td>
      <td>242.946000</td>
      <td>231.476667</td>
      <td>-0.419998</td>
      <td>-0.419998</td>
      <td>0.291136</td>
      <td>1.288878</td>
      <td>1.955655</td>
      <td>245.873904</td>
      <td>242.635546</td>
      <td>244.169998</td>
      <td>244.820007</td>
      <td>243.809998</td>
    </tr>
    <tr>
      <th>2024-10-08</th>
      <td>243.789993</td>
      <td>244.886668</td>
      <td>243.313333</td>
      <td>231.827666</td>
      <td>-0.790009</td>
      <td>-0.380005</td>
      <td>-0.514362</td>
      <td>1.256892</td>
      <td>1.939375</td>
      <td>245.078871</td>
      <td>241.834338</td>
      <td>242.369995</td>
      <td>244.039993</td>
      <td>240.630005</td>
    </tr>
    <tr>
      <th>2024-10-09</th>
      <td>241.160004</td>
      <td>243.846664</td>
      <td>243.648666</td>
      <td>232.136666</td>
      <td>-2.629990</td>
      <td>-1.209991</td>
      <td>-0.373007</td>
      <td>1.305928</td>
      <td>1.967981</td>
      <td>242.416896</td>
      <td>239.220629</td>
      <td>241.050003</td>
      <td>241.839996</td>
      <td>240.639999</td>
    </tr>
  </tbody>
</table>
</div>



## <a id="strategy-return-and-performance"></a> Strategy Return and Performance

##### Signal Generation

We will use the predicted high and predicted low values to determine whether to buy or sell GLD ETF the next day.

We will sell GLD ETF when
1. The actual high value is greater than the predicted high value. 
2. The actual low value is greater than the predicted low value.

We will buy GLD ETF when 
1. The actual high value is less than the predicted high value.  
2. The actual low value is less than the predicted low value.



```python
X_test['Signal'] = 0

# When selling, assigining Signal value as -1
X_test.loc[(X_test['High'] > X_test['P_H']) & 
            (X_test['Low'] > X_test['P_L']), 'Signal'] = -1
# When buying, assigning Signal value as 1
X_test.loc[(X_test['High'] < X_test['P_H']) &
            (X_test['Low'] < X_test['P_L']), 'Signal'] = 1
```

##### Algorithmic Buy/Sell Signal Generation Using Predicted High and Low Prices
This algorithm generates buy or sell signals based on predicted high and low prices, assuming that the market will correct any deviations from these levels. If the market goes above the predicted high without falling below the predicted low, we treat this as an overbought signal, indicating a likely correction. If the market’s actual high and low stay below our predicted levels, we consider it oversold, signaling a potential buying opportunity.

To execute this strategy, we’ll first calculate daily returns based on buying at the previous day's close and selling at the following day's close. Since our predictions are available at the day's open, we assume no extreme changes just before the close, allowing trades near the end of the day. We’ll then create columns, RET1 for returns and SIGNAL for signals. A sell signal (-1) is triggered when market highs and lows exceed predictions; a buy signal (1) occurs when they fall below predictions. Returns align with market returns on buy days and inversely on sell days. Finally, we’ll plot cumulative returns to assess strategy performance, comparing it to market trends.

### Strategy Returns
Here, we will compute the GLD returns and strategy returns.


```python
# Calculating the GLD (ETF) return
X_test['GLD_returns'] =  X_test['Close'].pct_change() 

# Computing Strategy returns
X_test['Strategy_returns'] = X_test['GLD_returns'] * (X_test['Signal'].shift(1))

X_test.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>MA_3</th>
      <th>MA_15</th>
      <th>MA_60</th>
      <th>OD</th>
      <th>OL</th>
      <th>Corr</th>
      <th>yU_predict</th>
      <th>yD_predict</th>
      <th>P_H</th>
      <th>P_L</th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Signal</th>
      <th>GLD_returns</th>
      <th>Strategy_returns</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-10-03</th>
      <td>244.720001</td>
      <td>244.776667</td>
      <td>241.913999</td>
      <td>230.678666</td>
      <td>-0.889999</td>
      <td>-0.940002</td>
      <td>0.717001</td>
      <td>1.256133</td>
      <td>1.977312</td>
      <td>245.950930</td>
      <td>242.783379</td>
      <td>245.490005</td>
      <td>245.910004</td>
      <td>243.690002</td>
      <td>0</td>
      <td>-0.000692</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2024-10-04</th>
      <td>245.000000</td>
      <td>245.586670</td>
      <td>242.524666</td>
      <td>231.114167</td>
      <td>0.279999</td>
      <td>-0.490005</td>
      <td>0.578269</td>
      <td>1.293902</td>
      <td>1.944456</td>
      <td>246.256133</td>
      <td>243.022688</td>
      <td>245.000000</td>
      <td>246.690002</td>
      <td>244.050003</td>
      <td>-1</td>
      <td>-0.001996</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2024-10-07</th>
      <td>244.580002</td>
      <td>245.383336</td>
      <td>242.946000</td>
      <td>231.476667</td>
      <td>-0.419998</td>
      <td>-0.419998</td>
      <td>0.291136</td>
      <td>1.288878</td>
      <td>1.955655</td>
      <td>245.873904</td>
      <td>242.635546</td>
      <td>244.169998</td>
      <td>244.820007</td>
      <td>243.809998</td>
      <td>0</td>
      <td>-0.003388</td>
      <td>0.003388</td>
    </tr>
    <tr>
      <th>2024-10-08</th>
      <td>243.789993</td>
      <td>244.886668</td>
      <td>243.313333</td>
      <td>231.827666</td>
      <td>-0.790009</td>
      <td>-0.380005</td>
      <td>-0.514362</td>
      <td>1.256892</td>
      <td>1.939375</td>
      <td>245.078871</td>
      <td>241.834338</td>
      <td>242.369995</td>
      <td>244.039993</td>
      <td>240.630005</td>
      <td>1</td>
      <td>-0.007372</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>2024-10-09</th>
      <td>241.160004</td>
      <td>243.846664</td>
      <td>243.648666</td>
      <td>232.136666</td>
      <td>-2.629990</td>
      <td>-1.209991</td>
      <td>-0.373007</td>
      <td>1.305928</td>
      <td>1.967981</td>
      <td>242.416896</td>
      <td>239.220629</td>
      <td>241.050003</td>
      <td>241.839996</td>
      <td>240.639999</td>
      <td>0</td>
      <td>-0.005446</td>
      <td>-0.005446</td>
    </tr>
  </tbody>
</table>
</div>



#### Plot the GLD Returns and Strategy Returns
Here we will plot `gld_returns` and `test_dataset` in one plot for comparison.


```python
plt.figure(figsize =(10, 6))

# Ploting GLD returns
plt.plot(((X_test['GLD_returns'][:]+1).cumprod()),
         color = 'black', label = 'GLD_Returns')

# Ploting Sterategy return 
plt.plot(((X_test['Strategy_returns'][:]+1).cumprod()),
         color = 'Green', label =  'Strategy_return')


# x-labeling
plt.xlabel('Date', fontsize=12)

# y-labeling
plt.ylabel('Returns', fontsize=12)

# Titlename
plt.title('Comparing GLD and Strategy Returns', fontsize=14)
plt.legend()
plt.show()
```


    
![png](output_42_0.png)
    


## <a id="future-work"></a> Future Work

From this point, the strategy requires quantitative market research to optimize and fine-tune the model. Further refinement and adjustments will be necessary to ensure the strategy's robustness and adaptability to various market conditions. This notebook will be periodically updated as new insights and improvements are made.
