import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
#from metrics import performance
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import metrics
#from sklearn.metrics import performance
from scipy import stats
from scipy.stats import pearsonr
import time
start = time.time()

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

"""Main"""
# Import sample data
# xls_file = r'D:\TEST\Samples.xlsx'
# all_data = pd.read_excel(xls_file, 'Train')
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
# vars = ['Log10_A', 'Log10_P', 'Cr', 'SDI', 'D', 'Slope_100', 'Slope_200','Slope_500','Tmp', 'Pre', 'STRA', 'CLAS', 'FLOW', 'Basin', 'Terrain', 'Climate'] #16因子
vars = ['Log10_A', 'Log10_P', 'Cr',  'D', 'Slope_100', 'Tmp', 'Pre'] #7因子
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

print("running...")
for i in range(0,10):          
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)
    
    #Export training dataset and test dataset
    train_x_reg=pd.DataFrame(X_train)
    train_x_reg.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\DNN\Volume_all_variables_500\train_test_split/train_x_ '+ str(i) +'.csv',encoding='gb2312',index=None)
    train_y_reg=pd.DataFrame(y_train)
    train_y_reg.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\DNN\Volume_all_variables_500\train_test_split/train_y_ '+ str(i) +'.csv',encoding='gb2312',index=None)
    test_x_reg=pd.DataFrame(X_test)
    test_x_reg.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\DNN\Volume_all_variables_500\train_test_split/test_x_ '+ str(i) +'.csv',encoding='gb2312',index=None)
    test_y_reg=pd.DataFrame(y_test)
    test_y_reg.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\DNN\Volume_all_variables_500\train_test_split/test_y_ '+ str(i) +'.csv',encoding='gb2312',index=None)
       
    # scaler the data
    x_scaler = RobustScaler()
    y_scaler = MinMaxScaler()
    x_t1 = x_scaler.fit_transform(X_train)
    y_t1 = y_scaler.fit_transform(np.log(y_train))
    x_t2 = x_scaler.transform(X_test)
    y_t2 = y_scaler.transform(np.log(y_test))

    joblib.dump(x_scaler, r'E:\XM\20220820全国水库库容模型\TEST\DNN\Volume_all_variables_500\model/DNN_ag_x_scaler_ '+ str(i) +'.json')
    joblib.dump(y_scaler, r'E:\XM\20220820全国水库库容模型\TEST\DNN\Volume_all_variables_500\model/DNN_ag_y_scaler_ '+ str(i) +'.json')
    
    # train a model
    # add the model parameter
    n_input = X_train.shape[1]
    n_output = y_train.shape[1]
    hiddenLayer = [128]*17
    max_epoches = 800
    batch_size = 512
    model_path = r'E:\XM\20220820全国水库库容模型\TEST\DNN\Volume_all_variables_500\model/DNN_save_ '+ str(i) +'_sat.h5'
    
# -------------------------------DNN model-------------------------------
    model = tf.keras.Sequential()
    # input layer
    model.add(tf.keras.Input(shape=(n_input, )))
    model.add(layers.Dense(hiddenLayer[0], activation='relu'))
    # model.add(layers.BatchNormalization())
    # hidden layers
    for item in hiddenLayer:
        model.add(layers.Dense(item, activation='relu'))
        # model.add(layers.BatchNormalization())
    # output layer
    model.add(layers.Dense(n_output, activation='linear'))
    opti = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    # opti = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opti, loss='mse', metrics=['mae'])
    
    # train the model
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    check_pointer = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,monitor='val_loss',
                                                       save_best_only=True,save_freq=100)
    
    hist = model.fit(x_t1, y_t1, epochs=max_epoches,batch_size=batch_size, shuffle=True,
                     callbacks=[early_stopper],validation_split=0.1,verbose=1)
    model.save(model_path)
    
    
    
    
    # predict and inverse trainsform
    #y_hat1 = model.predict(X_train)
    y_hat1 = np.exp(y_scaler.inverse_transform(model.predict(x_t1)))
    result_pre_train=pd.DataFrame(y_hat1)
    result_pre_train.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\DNN\Volume_all_variables_500\result_pre/DNN_pre_train_ '+ str(i) +'.csv',encoding='gb2312',index=None)
    #y_hat2 = model.predict(X_test)
    y_hat2 = np.exp(y_scaler.inverse_transform(model.predict(x_t2)))
    result_pre_test=pd.DataFrame(y_hat2)
    result_pre_test.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\DNN\Volume_all_variables_500\result_pre/DNN_pre_test_ '+ str(i) +'.csv',encoding='gb2312',index=None)
    
    num1=y_hat1.shape[0]
    num2=y_hat2.shape[0]
    y_hat1 = y_hat1.reshape(num1,1)
    y_hat2 = y_hat2.reshape(num2,1)
    R2_train=pearsonr(y_train[:,0],y_hat1[:,0])[0]**2
    R2_test=pearsonr(y_test[:,0],y_hat2[:,0])[0]**2
    
    err_train=calErr(y_train,y_hat1)
    mape_train=err_train[2]
    err_test=calErr(y_test,y_hat2)
    mape_test=err_test[2]
    
    sta[i,0] = i
    sta[i,1] = R2_train
    sta[i,2] = mape_train
    sta[i,3] = metrics.mean_absolute_error(y_train,y_hat1)
    sta[i,4] = np.sqrt(metrics.mean_squared_error(y_train,y_hat1))
    sta[i,5] = metrics.mean_squared_error(y_train,y_hat1)
    sta[i,6] = R2_test
    sta[i,7] = mape_test
    sta[i,8] = metrics.mean_absolute_error(y_test,y_hat2)
    sta[i,9] = np.sqrt(metrics.mean_squared_error(y_test,y_hat2))
    sta[i,10] = metrics.mean_squared_error(y_test,y_hat2)
    print(i)


#Scatter plot of predicted and measured values
validation_sta=pd.DataFrame(sta)
validation_sta.to_csv(r'E:\XM\20220820全国水库库容模型\TEST\DNN\Volume_all_variables_500\validation_sta/dnn_all_variables_validation_volume.csv',encoding='gb2312',index=None)


print("All done")
end = time.time()
print ("程序运行时间{:.2f}分钟".format((end-start)/60.0))