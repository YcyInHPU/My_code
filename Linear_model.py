# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:32:46 2022

@author: ycy
"""

from sklearn import datasets, linear_model,svm
import pandas as pd
import numpy as np
import joblib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.feature_selection import f_regression
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import time
start = time.time()


path = r'E:\XM\20220820全国水库库容模型\ML建模\ML_record_text.xlsx'
path1 = r'E:\XM\20220820全国水库库容模型\ML建模\ML_last.xlsx'

# sheet_name='Factors'
sheet_name='Normalization'
all_data = pd.read_excel(path,sheet_name)
# All parameters
#vars = ['lake_maxdepth_m', 'lake_area_km2', 'lake_perimeter_km', 'lake_shorelinedevfactor', 'lake_mbgconhull_length_km',
#        'lake_mbgconhull_width_km', ' Mean_Dem_buffer100', 'Ele_dif_buffer100_Lake_min', 'Slope_Dem_buffer100']

# Part parameters
# vars = ['A_km2', 'P_km', 'Cr', 'SDI', 'D', 'Slope_100', 'Tmp', 'Pre', 'STRA', 'CLAS', 'FLOW', 'Basin', 'Terrain', 'Climate']
# vars = ['A_km2', 'P_km', 'Cr', 'SDI', 'D', 'Slope_100', 'Tmp', 'Pre', 'STRA', 'CLAS', 'FLOW', 'Basin', 'Terrain', 'Climate'] #14因子
vars = ['Log10_A', 'Log10_P', 'Cr', 'SDI', 'D', 'Slope_100', 'Slope_200','Slope_500','Tmp', 'Pre', 'STRA', 'CLAS', 'FLOW', 'Basin', 'Terrain', 'Climate'] #16因子
# vars = ['Log10_A', 'Log10_P', 'Cr',  'D', 'Slope_100', 'Tmp', 'Pre'] #7因子
# vars = ['Log10_A', 'Log10_P', 'Cr', 'SDI', 'D', 'Slope_100', 'Tmp', 'Pre'] #8因子
# vars = ['Log10_A', 'Log10_P','Slope_100'] #3因子
# Predict lake mean depth
#keys = ['lake_meandepth_m']

# Predict lake volume
# keys = ['V_km3']
keys = ['Log10_V']
X = all_data[vars].values
Y = all_data[keys].values   
sta = np.zeros(shape=(500,11))

# sheet_name1 = 'last2'
# all_data1 = pd.read_excel(path,sheet_name1)
# X_last = all_data1[vars].values

def calErr(y0,y1):
    alpha=y1-y0
    beta=(y1-y0)/y0
    n=y0.shape[0]
    R2=np.corrcoef(y0,y1)[0,1]**2
    R2t=metrics.r2_score(y0,y1)
    RMSE=np.sqrt(np.sum(alpha**2)/n)
    MAPE=np.mean(np.abs(beta)*100) 
    MNB=np.mean((beta)*100) 
    NRMS=np.std(beta)*100
    UPD=np.mean(np.abs(alpha)/np.abs(y0+y1))*2*100
    
    RMSE_log=np.sqrt(np.sum((np.log10(y0)-np.log10(y1))**2)/n)
    URMSE=np.sqrt(np.sum(((y0-y1)/0.5/(y0+y1))**2)/n)*100
    Err=np.array([R2,RMSE,MAPE,MNB,n])
    return Err

print("running...")

for i in range(0,100):
     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
     #index0=y_test.index
     train_x_reg=pd.DataFrame(X_train)
     train_x_reg.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\Linear\Volume_key_variables_500\train_test_split/train_x_ '+ str(i) +'.csv',encoding='gb2312',index=None)
     train_y_reg=pd.DataFrame(y_train)
     train_y_reg.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\Linear\Volume_key_variables_500\train_test_split/train_y_ '+ str(i) +'.csv',encoding='gb2312',index=None)
     test_x_reg=pd.DataFrame(X_test)
     test_x_reg.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\Linear\Volume_key_variables_500\train_test_split/test_x_ '+ str(i) +'.csv',encoding='gb2312',index=None)
     test_y_reg=pd.DataFrame(y_test)
     test_y_reg.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\Linear\Volume_key_variables_500\train_test_split/test_y_ '+ str(i) +'.csv',encoding='gb2312',index=None)
     
     # The training process of ML机器的训练过程
     # model=svm.SVR(kernel='rbf', C=1)
     # model.fit(X_train,y_train)
     model = LinearRegression()
     model.fit(X_train,y_train)
     print(model.coef_)
     print(model.intercept_)
     #model_path = 'xgb_test'+'_sat.json'
     #joblib.dump(model, "pima.joblib.dat")
     #loaded_model = joblib.load("pima.joblib.dat")
     #model.save_model(r'D:\TEST\SVM\model/save_ '+ str(i) + '_model.xgb')
     joblib.dump(model, r'E:\XM\20220820全国水库库容模型\TEST\Linear\Volume_key_variables_500\model/save_ '+ str(i) + '_model.m')
     
     #Predict the test set and training set based on trained model根据训练好的模型预测测试集和训练集
     ans_test = model.predict(X_test)
     result_pre_test=pd.DataFrame(ans_test)
     result_pre_test.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\Linear\Volume_key_variables_500\result_pre/Linear_pre_test_ '+ str(i) +'.csv',encoding='gb2312',index=None)
     ans_train = model.predict(X_train)
     result_pre_train=pd.DataFrame(ans_train)
     result_pre_train.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\Linear\Volume_key_variables_500\result_pre/Linear_pre_train_ '+ str(i) +'.csv',encoding='gb2312',index=None)
     # ans_last = model.predict(X_last)
     # result_pre_last=pd.DataFrame(ans_last)
     # result_pre_last.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\Linear\Volume_key_variables_500\result_pre/Linear_pre_last_ '+ str(i) +'.csv',encoding='gb2312',index=None)
     
     # plt.scatter(X_train,y_train,color = 'teal',edgecolors = 'black',label = 'Horsepower vs. Miles_per_Gallon')
     # plt.plot(X_train,model.predict(X_train),color = 'orange',label = 'Linear regressor')
     # plt.title('Linear Regression')
     # plt.legend()
     # plt.show()
     
     # R2
     num1=ans_train.shape[0]
     num2=ans_test.shape[0]
     ans_train = ans_train.reshape(num1,1)
     ans_test = ans_test.reshape(num2,1)
     R2_train=pearsonr(y_train[:,0],ans_train[:,0])[0]**2
     R2_test=pearsonr(y_test[:,0],ans_test[:,0])[0]**2
     
     err_train=calErr(y_train,ans_train)
     mape_train=err_train[2]
     err_test=calErr(y_test,ans_test)
     mape_test=err_test[2]
     
     sta[i,0] = i
     sta[i,1] = R2_train
     sta[i,2] = mape_train
     sta[i,3] = metrics.mean_absolute_error(y_train,ans_train)
     sta[i,4] = np.sqrt(metrics.mean_squared_error(y_train,ans_train))
     sta[i,5] = metrics.mean_squared_error(y_train,ans_train)
     sta[i,6] = R2_test
     sta[i,7] = mape_test
     sta[i,8] = metrics.mean_absolute_error(y_test,ans_test)
     sta[i,9] = np.sqrt(metrics.mean_squared_error(y_test,ans_test))
     sta[i,10] = metrics.mean_squared_error(y_test,ans_test)
     print(i)

#Scatter plot of predicted and measured values预测值和测量值的散点图
validation_sta=pd.DataFrame(sta)
validation_sta.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\Linear\Volume_key_variables_500\validation_sta/Linear_key_variables_validation_volume.csv',encoding='gb2312',index=None)



print("All done")
end = time.time()
print ("程序运行时间{:.2f}分钟".format((end-start)/60.0))