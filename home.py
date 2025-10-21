import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split   #for training and testing models
from sklearn.linear_model import LinearRegression

#loading dataset from csv file
data=pd.read_csv("home_dataset.csv")

#Extracting feartures and target values
house_size=data['HouseSize'].values
house_price=data['HousePrice'].values

#visualization 
plt.scatter(house_size,house_price,marker='o',color='blue')
plt.title('House Prices vs. House size')
plt.xlabel('House Size(sq.ft)')
plt.ylabel('House Price($)')
plt.show()

#slitting data into test and train sets 
x_train,x_test,y_train,y_test=train_test_split(house_price,house_size,test_size=0.2,random_state=42)
#Setting a random_state value fixes the random seed, so every time you run the code with the same data and random_state, you'll get the same split.
#test size 0.2 splits data into such that 80% will train the data and 20$ will test the algo

#reshaping for numpy
x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)

#training 
model=LinearRegression()
model.fit(x_train,y_train)

predictions=model.predict(x_test)

plt.scatter(x_test,y_test,marker='o',color='blue',label='Actual prices')
plt.plot(x_test,predictions,color='red',linewidth=2 ,label='Predicted prices')
plt.title('Sample Property Price Prediction with Linear Regression')
plt.xlabel('House size(sq.ft)')
plt.ylabel('House Prices (million $)')
plt.legend()
plt.show()