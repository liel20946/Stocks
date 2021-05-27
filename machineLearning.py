from typing import Counter
import numpy as np
from numpy.lib.function_base import average
from numpy.ma import count
from scipy import stats
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime as dt


#data = yf.download(tickers='AMZN', period='1y', interval='1h') #IN CASE WE WANT TO CHANGE THE INTERVAL CAN BE USED THAT WAY

tickerData = yf.Ticker('AAL')
startDate = '2005-10-02' # as strings
endDate = '2017-07-28' # as strings

# Create historic data dataframe and fetch the data for the dates given. 
data= tickerData.history(start = startDate, end = endDate)

# Define string format
date_change = '%Y-%m-%d'

# Create a new date column from the index
data['DateN'] = data.index

# Perform the date type change
data['DateN']= pd.to_datetime(data['DateN'], format = date_change)

"""
#declare figure
fig = go.Figure()


#Candlestick
fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name = 'market data'))

# Add titles
fig.update_layout(
    title='AAL live share price evolution',
    yaxis_title='Stock Price (USD per Shares)')

# X-Axes
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=6, label=" 6m", step="month", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=7, label="week", step="day", stepmode="backward"),
            dict(step="all")
        ])
    )
)

#Show
fig.show()

#diffrent way of showing
plt.figure(figsize = (18,9))
plt.plot(range(data.shape[0]),(data['Low']+data['High'])/2.0)
plt.xticks(range(0,data.shape[0],500),data['DateN'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()
"""



high_prices = data.loc[:,'High']
low_prices = data.loc[:,'Low']
mid_prices = (high_prices+low_prices)/2.0

train_data = mid_prices[:2500]
test_data = mid_prices[2500:]

scaler = MinMaxScaler()
train_data = train_data.values.reshape(-1,1)
test_data = test_data.values.reshape(-1,1)

# Train the Scaler with training data and smooth data
smoothing_window_size = 800
for di in range(0,2400,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# You normalize the last bit of remaining data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)

# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(2500):
  EMA = gamma*train_data[ti] + (1-gamma)*EMA
  train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data,test_data],axis=0)


window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []


for pred_idx in range(window_size,N):

    if pred_idx >= N:
            date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
            date = data['DateN'].iloc[pred_idx]

    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)


print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

plt.figure(figsize = (18,9))
plt.plot(range(all_mid_data.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()


























"""
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

x = numpy.mean(speed)

print(x)
"""

"""

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()

"""

"""

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)
"""


