import pandas as pd
from sklearn import linear_model
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV


df = pd.read_csv('train_data.csv', header=0)

# Gather Train data
train_data=df.ix[0:500,]
# Gather validate data
validate_data=df.ix[501:1000,]

trainx=train_data.ix[:,1:59]
trainy=train_data.ix[:,60]

testx=validate_data.ix[:,1:59]
testy=validate_data.ix[:,60]

#Linear Regression
#-----------------
# create a linear regression model
lr = linear_model.LinearRegression()
#fit the linear model.
lr.fit(trainx,trainy)
#access the coeffcients using
print("w values for the linear model",lr.coef_)
ypredicted=lr.predict(testx)
predicted_values=pd.DataFrame(ypredicted)
print predicted_values

#SVR Regression
#--------------
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# y_rbf = svr_rbf.fit(trainx, trainy).predict(testx)
# print y_rbf

# svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
#                    param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                                "gamma": [0.1]})
# svr.fit(trainx, trainy)
# y_svr = svr.predict(testx)
# print type(y_svr)
# resultdf= pd.DataFrame(y_svr)
# print  resultdf
index = ['x', 'y']
predicted_values.to_csv(path_or_buf="output.csv", index=index)


# # look at the results
# plt.scatter(trainx, trainy, c='k', label='data')
# plt.hold('on')
# plt.plot(trainx, y_rbf, c='g', label='RBF model')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()