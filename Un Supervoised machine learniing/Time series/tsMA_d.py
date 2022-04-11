# Jesus is my Saviour!
#https://towardsdatascience.com/time-series-from-scratch-moving-averages-ma-theory-and-implementation-a01b97b60a18

import pandas as pd
# plotting data - matplotlib
from matplotlib import pyplot as plt
# time series - statsmodels 
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# adf test
import statsmodels.api as sm

from statsmodels.tsa.arima_model import ARMA
# holt winters 
# single exponential smoothing
#from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
#from statsmodels.tsa.holtwinters import ExponentialSmoothing
# evaluation metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error

import numpy as np
from matplotlib import rcParams
from cycler import cycler

rcParams['figure.figsize'] = 18, 5
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.prop_cycle'] = cycler(color=['#365977'])
rcParams['lines.linewidth'] = 2.5

df = pd.read_csv("C:/Users/Dr Vinod/Desktop/ts201121/airline_passengers.csv",parse_dates = True, index_col= 'month')
df.info()
df.head()

df.index #as datetime64[ns] !

# Plot
plt.title('Airline passengers dataset', size=20)
plt.plot(df);

#plt.plot(df['passengers'], color='red') # right! 

# Calculcate
df['MA3'] = df['passengers'].rolling(window=3).mean()
df['MA6'] = df['passengers'].rolling(window=6).mean()
df['MA12'] = df['passengers'].rolling(window=12).mean()

# Plot
plt.title('Airline passengers moving averages', size=20)
plt.plot(df['passengers'], label='Original')
plt.plot(df['MA3'], color='gray', label='MA3')
plt.plot(df['MA6'], color='orange', label='MA6')
plt.plot(df['MA12'], color='red', label='MA12')
plt.legend();

##___________________________ forecasting is not possible by above method
## need to use ARMA models 
from statsmodels.tsa.arima_model import ARMA

# Train/test split
df_train = df[:-24]
df_test = df[-24:]

# Train the model
model = ARMA(df_train['passengers'], order=(0, 1))
results = model.fit()
predictions = results.forecast(steps=24)
predictions_df = pd.DataFrame(index=df_test.index, data=predictions[0])

# Plot
plt.title('Airline passengers MA(1) predictions', size=20)
plt.plot(df_train['passengers'], label='Training data')
plt.plot(df_test['passengers'], color='gray', label='Testing data')
plt.plot(predictions_df, color='orange', label='Predictions')
plt.legend();

# Differencing
df['Diff'] = df['passengers'].diff(1)
df = df.dropna() # 1st value to be dropped 

# Train/test split
df_train = df[:-24]
df_test = df[-24:]

# Train the model
model = ARMA(df_train['Diff'], order=(0, 1))
results = model.fit()
predictions = results.forecast(steps=24)
predictions_df = pd.DataFrame(index=df_test.index, data=predictions[0])

# Plot
plt.title('Differenced airline passengers MA(1) predictions', size=20)
plt.plot(df_train['Diff'], label='Training data')
plt.plot(df_test['Diff'], color='gray', label='Testing data')
plt.plot(predictions_df, color='orange', label='Predictions')
plt.legend();































































