from typing import Counter
import datetime as dt
import numpy as np
from numpy.lib.function_base import average
from numpy.ma import count
from scipy import stats
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

#1 day prediction with standart averaging
def one_day_prediction(train_data, test_data):
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1,1)
    test_data = test_data.reshape(-1,1)

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


    window_size = 50
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []


    for pred_idx in range(window_size,N):

        if pred_idx >= N:
                date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
                date = df['DateN'].iloc[pred_idx]

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
########################################### LTSM  ###########################################
# Make sure that the number of rows in the dataset = train rows + test rows
def predict_more_days(train_data):
    model = Sequential()
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))
    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = 100, batch_size = 32)


    # Getting the predicted stock price of 2017
    dataset_train = df.iloc[:800, 1:2]
    dataset_test = df.iloc[800:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(120, 519):
        X_test.append(inputs[i-120:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(X_test.shape)
    # (459, 60, 1)


    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    
    testdta_size = int(len(dataset_test))
    predict_size = int(len(predicted_stock_price))

    # Visualising the results
    plt.plot(range(0,testdta_size),dataset_test.values, color = 'red', label = 'Real TESLA Stock Price')
    plt.plot(range(0,predict_size),predicted_stock_price, color = 'blue', label = 'Predicted TESLA Stock Price')
    plt.xticks(np.arange(0,459,50))
    plt.title('TESLA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('TESLA Stock Price')
    plt.legend()
    plt.show()


def isLeak(T_shape, train_shape, test_shape):
    return not(T_shape[0] == (train_shape[0] + test_shape[0]))

#visualzing in a diffrent way
def visualize(data):
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


def main():
    #data = yf.download(tickers='AMZN', period='1y', interval='1h') #IN CASE WE WANT TO CHANGE THE INTERVAL CAN BE USED THAT WAY

    tickerData = yf.Ticker('AMZN')
    startDate = '2005-10-02' # as strings
    endDate = '2017-07-28' # as strings

    # Create historic data dataframe and fetch the data for the dates given. 
    df= tickerData.history(start = startDate, end = endDate)

    # Define string format
    date_change = '%Y-%m-%d'

    # Create a new date column from the index
    df['DateN'] = df.index

    # Perform the date type change
    df['DateN']= pd.to_datetime(df['DateN'], format = date_change)

    high_prices = df.loc[:,'High']
    low_prices = df.loc[:,'Low']
    mid_prices = (high_prices+low_prices)/2.0


    scaler = MinMaxScaler(feature_range=(0,1))

    train_size = int(len(mid_prices)*0.8)
    test_size = int(len(mid_prices) - train_size)


    train_data = mid_prices[0:train_size]
    test_data = mid_prices[train_size:len(mid_prices)]
    train_data = train_data.values.reshape(-1,1)
    test_data = test_data.values.reshape(-1,1)

    scaler.fit(train_data[:,:])
    train_data = scaler.transform(train_data[:,:])


    train_data = train_data.reshape(-1)
    test_data = scaler.transform(test_data).reshape(-1)

    # Roughly one month of trading assuming 5 trading days per week
    window_size = 20
    df_shape = df.shape
    train_shape = train_data.shape
    test_shape = test_data.shape


    print(isLeak(df_shape, train_shape, test_shape))

        # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(120, 800):
        X_train.append(train_data[i-120:i])
        y_train.append(train_data[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #(740, 60, 1)
    one_day_prediction(train_data, test_data)

main()
