# Jesus!
#https://medium.com/analytics-vidhya/python-code-on-holt-winters-forecasting-3843808a9873
# dataframe opertations - pandas
import pandas as pd
# plotting data - matplotlib
from matplotlib import pyplot as plt
# time series - statsmodels 
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# adf test
import statsmodels.api as sm
# holt winters 
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# evaluation metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv("C:/Users/Dr Vinod/Desktop/DataSets1/international-airline-passengers.csv",parse_dates = True, index_col= 'Month')
df.info()
df.head()

df= df.rename(columns = {'International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60': 'passengers'})

# plotting the original data
df[['passengers']].plot(title='Passengers Data')

df.tail() # last row is starnge! 
df = df.drop(labels = 'International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60', axis = 0)

df.index #oops....as object!

df.index = pd.to_datetime(df.index)

df.index # after converting to datetime!

decompose_result = seasonal_decompose(df['passengers'], model='multiplicative')
decompose_result.plot();

# adf test
import statsmodels.api as sm
seasonal_decompose(df['passengers']).plot(); # additive model

result = sm.tsa.stattools.adfuller(df['passengers'])
result

# Set the frequency of the date time index as Monthly start as indicated by the data
df.index.freq = 'MS'
# Set the value of Alpha and define m (Time Period)
m = 12
alpha = 1/(2*m) 

# Single HWES 

df['HWES1'] = SimpleExpSmoothing(df['passengers']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
df[['passengers','HWES1']].plot(title='Holt Winters Single Exponential Smoothing');

# Double HWES

df['HWES2_ADD'] = ExponentialSmoothing(df['passengers'],trend='add').fit().fittedvalues
df['HWES2_MUL'] = ExponentialSmoothing(df['passengers'],trend='mul').fit().fittedvalues
df[['passengers','HWES2_ADD','HWES2_MUL']].plot(title='HWSE-Double: Additive and Multiplicative Trend');


# Triple HWES

df['HWES3_ADD'] = ExponentialSmoothing(df['passengers'],trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
df['HWES3_MUL'] = ExponentialSmoothing(df['passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues

df[['passengers','HWES3_ADD','HWES3_MUL']].plot(title='HWSE-Triple: Additive and Multiplicative Seasonality');

#___________forecasting

df1 = pd.read_csv("C:/Users/Dr Vinod/Desktop/ts201121/airline_passengers.csv",parse_dates = True, index_col= 'month')
df1.info()
df1.head()
df1.dtypes
df1.index #as datetime64[ns] !

# freq of index as months 
df1.index.freq = 'MS'
# Split into train and test set
train_airline = df1[:120]
test_airline = df1[120:]

# fit and forecast
fitted_model = ExponentialSmoothing(train_airline['passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
## ignore warning and error!
test_predictions = fitted_model.forecast(24)

#plot all together
train_airline['passengers'].plot(legend=True,label='TRAIN')
test_airline['passengers'].plot(legend=True,label='TEST',figsize=(6,4))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Train,Test and Predicted Test using Holt Winters')

# plot only test and predictions
test_airline['passengers'].plot(legend=True,label='TEST',figsize=(9,6))
test_predictions.plot(legend=True,label='PREDICTION');

from sklearn.metrics import mean_absolute_error,mean_squared_error
print(f'Mean Absolute Error = {mean_absolute_error(test_airline,test_predictions)}')
print(f'Mean Squared Error = {mean_squared_error(test_airline,test_predictions)}')




























