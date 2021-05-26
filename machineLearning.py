import numpy as np
from scipy import stats
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go


#data = yf.download(tickers='AMZN', period='1y', interval='1h') #IN CASE WE WANT TO CHANGE THE INTERVAL CAN BE USED THAT WAY

tickerData = yf.Ticker('AMZN')
startDate = '2015-04-01' # as strings
endDate = '2021-05-25' # as strings

# Create historic data dataframe and fetch the data for the dates given. 
data= tickerData.history(start = startDate, end = endDate)

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
    title='Uber live share price evolution',
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

#print(df.head())



#branch test
#test 2
#test
#lat test

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


