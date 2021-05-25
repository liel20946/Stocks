import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd


tickerSymbol = "AMZN"

# Choose date range - format should be 'YYYY-MM-DD' 
startDate = '2015-04-01' # as strings
endDate = '2021-05-25' # as strings

# Create ticker yfinance object
tickerData = yf.Ticker(tickerSymbol)

# Create historic data dataframe and fetch the data for the dates given. 
df = tickerData.history(start = startDate, end = endDate)

# Print statement showing the download is done

# Show what the first 5 rows of the data frame
# Note the dataframe has:
#   - Date (YYY-MM-DD) as an index
#   - Open (price the stock started as)
#   - High (highest price stock reached that day)
#   - Low (lowest price stock reached that day)
#   - Close (price the stock ended the day as)
#   - Volume (how many shares were traded that day)
#   - Dividends (any earnings shared to shareholders)
#   - Stock Splits (any stock price changes)

print('-----------------------')
print('Done!')
print(df.head())



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


