#encoding : utf-8
#使用lstm模型预测时间序列数据
import numpy as np
import os
import re
import pandas as pd
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# array=[71.,71.,70.,73.,76. ,70. ,67. ,80. ,78., 74. ,67. ,70., 81., 77., 70. ,78., 70., 74.,
#  75., 76., 72., 72., 78., 82., 74., 78., 75., 78., 79., 72., 79., 76., 77., 73., 68., 74.,
#  70., 71., 68., 89., 77., 78., 76., 74., 80., 88., 76., 74., 77., 79., 85., 76., 75., 80.,
#  82., 83., 73., 78., 79., 80., 84., 87., 82., 82., 67., 69., 70., 82., 75., 79. ,83., 81.,
#  80., 98., 86., 87., 91., 90., 91., 82., 86., 90. ,83. ,82.]
#
# print(type(array))
# dataset=np.array(array)
# dataset = dataset.astype('float32')
#
# # plt.plot(dataset)
# # plt.show()
# dataset=np.array(dataset).reshape(-1,1)
# print(dataset.shape)
#
#
# def create_dataset(dataset, look_back):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back - 1):
#         a = dataset[i:(i + look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return np.array(dataX), np.array(dataY)
#
#
# # fix random seed for reproducibility
# np.random.seed(7)
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
#
# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# look_back = 20
# print(train_size,test_size)
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# predict=dataset[len(dataset)-look_back:len(dataset),:]
# #print(train, test )
# # use this function to prepare the train and test datasets for modeling
#
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# predict_X=[]
# predict_X.append(predict[:,0])
# predict_X=np.array(predict_X)
# print(trainX.shape,predict_X.shape)
# # reshape input to be [samples, time steps, features]
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# predict_X=np.reshape(predict_X, (predict_X.shape[0], 1, predict_X.shape[1]))
# model = Sequential()
# model.add(LSTM(16, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# predict_y = model.predict(predict_X)
#
# #print(trainPredict ,testPredict ,predict_y)
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# predict_y =scaler.inverse_transform(predict_y)
# #预测最后一次考试成绩
#
#
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
#
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
# print(type(predict_y))
# sub_test = pd.read_csv('./test_s1/submission_s1.csv')
# #sub_test.rename(columns={'pred':'score'},inplace = True)
# sub_test.iloc[2]['pred'] = predict_y
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# fix random seed for reproducibility
np.random.seed(7)
df_train = pd.read_csv('./train_s1/exam_score.csv')
print(len(df_train))
df_test = pd.read_csv('./test_s1/submission_s1.csv')

course_class = pd.read_csv('./train_s1/course.csv')
student = pd.read_csv('./train_s1/student.csv')
sub_test = pd.read_csv('./test_s1/submission_s1.csv')
print(len(student))
#all_know = pd.read_csv('/home/kesci/input/smart_edu7557/all_knowledge.csv')
student_list=student['student_id']
for i in student_list:
    for k in range(1,9):
        course='course'+str(k)
        print(course)
        col=[j for j in df_train.index if df_train.iloc[j]['student_id']==i and df_train.iloc[j]['course']==course]
        score=df_train.loc[col]['score']
        dataset=score.values
        dataset=np.array(dataset)
        dataset = dataset.astype('float32')
        dataset=np.array(dataset).reshape(-1,1)
        #数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        # split into train and test sets
        look_back = 6
        train_size = int(len(dataset))
        train = dataset[0:train_size,:]
        predict=dataset[len(dataset)-look_back:len(dataset),:]
        #use this function to prepare the train and test datasets for modeling
        trainX, trainY = create_dataset(train, look_back)
        predict_X=[]
        predict_X.append(predict[:,0])
        predict_X=np.array(predict_X)
        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        predict_X=np.reshape(predict_X, (predict_X.shape[0], 1, predict_X.shape[1]))
        model = Sequential()
        model.add(LSTM(16, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
        trainPredict = model.predict(trainX)
        predict_y = model.predict(predict_X)
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        predict_y =scaler.inverse_transform(predict_y)#预测出最后一次考试成绩
        col=[j for j in sub_test.index if sub_test.iloc[j]['student_id']==i and sub_test.iloc[j]['course']==course]
        sub_test.iloc[col]['pred'] = predict_y[0][0]
        print(predict_y)
        #查看训练得分
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
print(sub_test)
sub_test.to_csv('./submisson3.csv',index=None)