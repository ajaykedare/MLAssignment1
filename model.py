import pandas as pd
from sklearn import linear_model

df = pd.read_csv('train_data.csv', header=0)
df_testData = pd.read_csv('test_data.csv', header=0)

columnValues= ['url','timedelta','n_tokens_title','n_tokens_content','n_unique_tokens','n_non_stop_words','n_non_stop_unique_tokens','num_hrefs','num_self_hrefs','num_imgs','num_videos','average_token_length','num_keywords','data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world','kw_min_min','kw_max_min','kw_avg_min','kw_min_max','kw_max_max','kw_avg_max','kw_min_avg','kw_max_avg','kw_avg_avg','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess','weekday_is_monday','weekday_is_tuesday','weekday_is_wednesday','weekday_is_thursday','weekday_is_friday','weekday_is_saturday','weekday_is_sunday','is_weekend','LDA_00','LDA_01','LDA_02','LDA_03','LDA_04','global_subjectivity','global_sentiment_polarity','global_rate_positive_words','global_rate_negative_words','rate_positive_words','rate_negative_words','avg_positive_polarity','min_positive_polarity','max_positive_polarity','avg_negative_polarity','min_negative_polarity','max_negative_polarity','title_subjectivity','title_sentiment_polarity','abs_title_subjectivity','abs_title_sentiment_polarity','shares']
dfColumns = df.columns

def dropColumnInplace(dataFrame,columnName):
    dataFrame.drop(dfColumns[columnValues.index(columnName)], axis=1,inplace=True)

def dropColumn(dataFrame,columnName):
    return dataFrame.drop(dfColumns[columnValues.index(columnName)], axis=1)

columnsToDrop=['url']
for column in columnsToDrop:
    dropColumnInplace(df,column)
    dropColumnInplace(df_testData,column)

# Gather Train data
train_data=df.ix[0:20000,]
# Gather validate data
validate_data= df_testData.ix[0:,]

trainx=train_data.ix[:,:len(df.columns)-2]
trainy=train_data.ix[:,len(df.columns)-1]

testx=validate_data.ix[:,:len(df_testData.columns)-1]

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