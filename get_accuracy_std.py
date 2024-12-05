#coding=utf-8
import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
from functools import reduce
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from matplotlib import font_manager
import warnings
from matplotlib.pyplot import MultipleLocator

warnings.filterwarnings("ignore")

# 导入数据集
data=pd.read_csv('D:/yanjiu/6data/xgboost/water level_new/1920/3.csv')

#y_test为真实水位
y_test = data['source']

#y_pre为预测水位
y_pre = data['pre']

#y为1961-2020年份
year=data['year']

#y_std标准差
y_std = np.std(y_pre,ddof=1)


#输出机器学习精度
'''
print("STD:",y_std)
print("MSE:",metrics.mean_squared_error(y_test[49:118],y_pre[49:118])) 
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test[49:118],y_pre[49:118])))
print("MAE:",metrics.mean_absolute_error(y_test[49:118],y_pre[49:118]))
print("R2：", metrics.r2_score(y_test[49:118],y_pre[49:118]))
'''

#设置缩放
plt.figure(figsize=(12,7))

#设置字体
plt.rcParams["font.family"] = "Times New Roman"
#my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/timesbi.ttf")

#设置坐标轴名称
plt.xlabel(u'Year',size=30)
plt.ylabel(u'Water level (m)',size=28)

#设置x，y范围
x_major_locator=MultipleLocator(20)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(1)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(1920,2020)
plt.xticks(fontsize=26)
plt.ylim(455,458)
plt.yticks(fontsize=26)



#作图，1961-2020预测值为蓝色实线，真实值为红色点
plt.plot(year,y_test,'o',lw=1,label='Measured',color='r',alpha=0.9)
plt.plot(year,y_pre,lw=3.5,label='Predicted',color='dodgerblue')



#绘制1倍标准差范围
plt.fill_between(year,y_pre,y_pre+y_std,facecolor='lightskyblue')
plt.fill_between(year,y_pre-y_std,y_pre,facecolor='lightskyblue')

 #绘制2倍标准差范围
plt.fill_between(year,y_pre+y_std,y_pre+2*y_std,facecolor='powderblue')
plt.fill_between(year,y_pre-2*y_std,y_pre-y_std,facecolor='powderblue')

#添加图例
plt.rcParams.update({'font.size': 26})     #设置图例字体大小
plt.legend(loc=0,frameon=False,ncol=2)

#添加文本

plt.text(1923,457.9,
    '(r)',
	fontsize=40,
	verticalalignment="top",
	horizontalalignment="left")


#设置图框线粗细
bwith = 2 #边框宽度设置为2
TK = plt.gca()#获取边框
TK.spines['bottom'].set_linewidth(bwith)
TK.spines['left'].set_linewidth(bwith)
TK.spines['top'].set_linewidth(bwith)
TK.spines['right'].set_linewidth(bwith)

#出图
plt.savefig(r"D:\yanjiu\6data\xgboost\Uncertainly range\new\3_test_r.eps", dpi=1200)
plt.show()
