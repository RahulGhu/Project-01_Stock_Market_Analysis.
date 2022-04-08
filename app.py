
# numerical and computational libraries
import pandas as pd
import numpy as np
from numpy import array

from keras.models import load_model   # using this model we not need to load epoch and other models 
# Stock data downloading libraries
import yfinance as yf

import tensorflow as tf

# Visualization library
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
# for web app 
import streamlit as st
# for date and time 
from datetime import date
# for graph plotting lib.
from plotly import graph_objs as go
from fbprophet import Prophet
# for graph plotting lib.
from fbprophet.plot import plot_plotly
#from pandas import Period



#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
#import pandas_datareader as data
#from bs4 import BeautifulSoup
#from sklearn.preprocessing import MinMaxScaler



START = "2011-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title(" Stock Trend Prediction")
stocks = ('TATAMOTORS.NS', 'RELIANCE.NS', 'SBIN.NS','INFY.NS','IRCTC.NS')
data = st.selectbox('Select Stock for prediction : ', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


#@st.cache
#@st.cache(suppress_st_warning=True)
#@st.cache(allow_output_mutation=True)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(data)
data_load_state.text('Loading data... done!')
st.subheader('Last Five Days Records')
st.write(data.tail())

#data.reset_index(inplace = True)

# Describing data
st.subheader("Data from 2011 to Till the date")
st.write(data.describe())


# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'\n \n Forecast plot for  {n_years}  years ')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


#Visualizations
st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
#sns.lineplot(x='year',y='Close',data=data.Close)
#plt.xlabel('year')
#plt.ylabel('month')

st.pyplot(fig)


#Visualizations
st.subheader("Closing Price vs Time chart with 100MA")  # MA is moving avrg
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

#Visualizations
st.subheader("Closing Price vs Time chart with 100MA & 200MA")  # MA is moving avrg
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(data.Close)
st.pyplot(fig)

# droping the cols date
data.drop(['Date'],inplace = True, axis = 1)

# rescaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
data = scaler.fit_transform(np.array(data).reshape(-1,1))


## spliting the data set in to train and test split
training_size = int(len(data)*0.65)
test_size = len(data)-training_size
train_data, test_data = data[0:training_size,:],data[training_size:len(data),:1]

import numpy
# it convert an array of values into a dataset matrix
def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step - 1):
        a = dataset[i:i+time_step, 0]     ### ## i = 0,1,2,3.......
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


# reshape into X = t, t+1, t+2, t+3 and y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test =  create_dataset(test_data, time_step)


# reshape input to be [samples, time steps, features0] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test =  X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


#Load my model from python program
model = load_model('keras_model.h5')


# Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)



# Plotting 
# shift train predictions for plotting
fig = plt.figure(figsize=(12,6))
look_back=100
trainPredictPlot = numpy.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(data)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(data)-1, :] = test_predict
# plot baseline and predictions

st.subheader("Plot baseline and Predictions")
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
st.pyplot(fig)
# Blue color actual data
# Orange color for train data 
# Green color test data output which is i predicted



# previous 100 days data for pred. and reshaping it.
x_input=test_data[len(test_data)-100:].reshape(1,-1)

# converting this data to the list 
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
print(lst_output)

# for plotting graph
day_new=np.arange(1,101)   # indexes inside this days means 100 day pred
day_pred=np.arange(101,131) # 131 bcoz u hv to pred. next 30 days

# Actual 30 day predicted graph
st.subheader("Next 30 day prediction")
fig = plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(data[(len(data)-100):]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.pyplot(fig)
#st.plt.show()
# orange is predicted graph for next 30 days






