import pandas as pd
from sklearn import linear_model


df = pd.read_csv('train_data.csv', header=0)


train_data=df.ix[0:500,]
validate_data=df.ix[500:1000,]


# In[42]:



#create a linear regression model
lr=linear_model.LinearRegression()
trainx=train_data.ix[:,2:]
trainy=train_data.ix[:,1]

#fit the linear model.
lr.fit(trainx,trainy)
#access the coeffcients using
print("w values for the linear model",lr.coef_)

testx=validate_data.ix[:,2:]
testy=validate_data.ix[:,1]
ypredicted=lr.predict(testx)
predicted_values=pd.DataFrame(ypredicted)