import pandas as pd
from sklearn import linear_model

df = pd.read_csv('train_data.csv', header=0)
df_test = pd.read_csv('test_data.csv', header=0)
# Gather Train data
train_data=df.ix[0:,]
# Gather validate data
validate_data=df_test.ix[0:,]

trainx=train_data.ix[:,1:59]
trainy=train_data.ix[:,60]

testx=validate_data.ix[:,1:59]

#Linear Regression
#-----------------
# create a linear regression model
lr = linear_model.LinearRegression()
# lr = linear_model.LogisticRegression()
#fit the linear model.
lr.fit(trainx,trainy)
ypredicted=lr.predict(testx)
predicted_values=pd.DataFrame(ypredicted)

predicted_values.to_csv(path_or_buf="output.csv", index_label=["id"], header=["shares"])
