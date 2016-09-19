import scipy
import numpy as np
import pandas as pd
from sklearn import linear_model


trainingInputData = np.loadtxt("train_data.csv", skiprows=1, usecols=range(2, 60), delimiter=",")
predictTarget = np.loadtxt("train_data.csv", skiprows=1, usecols=[60], delimiter=",")
testData = np.loadtxt("test_data.csv", skiprows=1, usecols=range(2, 60), delimiter=",")

#Delete the columns
trainingInputData = scipy.delete(trainingInputData, scipy.s_[52:], axis=1)
testData = scipy.delete(testData, scipy.s_[52:], axis=1)
trainingInputData = scipy.delete(trainingInputData, scipy.s_[6:12], axis=1)
testData = scipy.delete(testData, scipy.s_[6:12], axis=1)

#Delete the rows
trainx = trainingInputData[0:20000, ]
trainy = predictTarget[0:20000, ]

#Add the square and cube columns in training data
trainingFinalData = []
for i in range(0, len(trainx)):
    tmpRow = []
    for j in range(len(trainx[i])):
        tmpRow.append(trainx[i][j])
        tmpRow.append(trainx[i][j] ** 2)
    trainingFinalData.append(tmpRow)

#Add the square and cube columns in test data
testDataList = []
for i in range(0, len(testData)):
    tmpRow = []
    for j in range(len(testData[i])):
        tmpRow.append(testData[i][j])
        tmpRow.append(testData[i][j] ** 2)
    testDataList.append(tmpRow)




# Ridge Regression
# -----------------
# create a Ridge regression model
lr = linear_model.Ridge(alpha=10)
# fit the linear model.
lr.fit(trainingFinalData, trainy)
ypredicted = lr.predict(testDataList)
predicted_values = pd.DataFrame(ypredicted)
predicted_values[predicted_values.columns[0]]=predicted_values[predicted_values.columns[0]].astype(int)
predicted_values.to_csv(path_or_buf="output.csv", index_label=["id"], header=["shares"],dtype=int)