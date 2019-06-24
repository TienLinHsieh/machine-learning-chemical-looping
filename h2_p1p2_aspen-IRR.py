# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:36:36 2019

@author: yrret
"""
import glob
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

##opening tensorboard
#cd model directory then
#tensorboard --logdir=.

##Set current directory to .py file location
os.chdir(os.path.dirname(sys.argv[0]))

##Read and consolidate all data source files
extension  = 'xlsx'
all_filenames = [i for i in glob.glob('Fe29*.{}'.format(extension))]

df_all = pd.DataFrame()

for file in all_filenames:
    df = pd.read_excel(file, sheet_name = 'Fe29-32')
    df['Data_source'] = file[:-5]
    df_all = df_all.append(df)

##cleaning variable names
import re
var_name_list_clr = []
for name in list(df_all):
    a = re.sub("[\(\[].*?[\)\]]", "", name) ##delete parenthesis and context within
    var_name_list_clr.append(a.translate(str.maketrans('','',' _'))) ##delete space and underscore
    

df_all.columns = var_name_list_clr
    
##inputs and outputs
y_val = df_all['20yearIRR']
x_data = df_all.drop(['20yearIRR','Datasource'], axis=1)
#x_data = df_all.drop(['Unnamed:0','error','20yearIRR','Datasource','STInT','H2Mass','AirInT','NGInT'], axis=1)

##train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_val, test_size=0.30, random_state=101)

##normalize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(data=scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(data=scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

##transform features to numeric columns
feat_dict = {} ##create a dictionary 

for variable_name in list(X_train): ##define all variables as numeric columns
    feat_dict[variable_name] = tf.feature_column.numeric_column('{}'.format(variable_name))
feat_columns = feat_dict.values()


##Create the input funtion for the estimator project
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,
                                                 y=y_train,
                                                 batch_size=10,
                                                 num_epochs=100,
                                                 shuffle=True)

##Create estimator model. (Dense Neural Network Regressor)
model = tf.estimator.DNNRegressor(hidden_units=[19,19], feature_columns = feat_columns, model_dir='./model_opt-adam')  ##model_dir='./tmp/tf'

##Train model 
model.train(input_fn=input_func,steps = 500)

##Create prediction input function and use .predict method off the estimator model to create a list of 
##predictions on test data
predict_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test,
                                                         batch_size = 10,
                                                         num_epochs = 1,
                                                         shuffle =False)
pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)

### Calculate root-mean-square error.
final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])
from sklearn.metrics import mean_squared_error
RMSE_value = mean_squared_error(y_test, final_preds)**0.5
## Check difference
difference =[]
i=0
for e in list(y_test):
    difference.append(e-final_preds[i])
    i+=1

##show weights
fig, axarr = plt.subplots(1,2)
weights0 = model.get_variable_value('dnn/hiddenlayer_0/kernel')
axarr[0].imshow(weights0, cmap='gray')
weights1 = model.get_variable_value('dnn/hiddenlayer_1/kernel')
axarr[1].imshow(weights1, cmap='gray')
