from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xg
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import *
import pickle
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime
import xgboost as xg
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Lasso
#from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
col_s ='300%_Tensile_Orth_66'#'elongation_Orth_66'#'strength_Orth_66'#'300%_Tensile_Orth_66'#'MH_Orth_66'#'300%_Tensile_Orth_66'#'300_Tensile(Mpa)_Orth_251'#'MH(dN.m)_Orth_167'#' 'ML(dN.m)'#'MH(dN.m)'
#col_s ='300_Tensile(Mpa)_Orth_251'#'MH(dN.m)_Orth_167'#' 'ML(dN.m)'#'MH(dN.m)'
# 从文件中读取数据
#data = pd.read_excel('data.xlsx')
#data = pd.read_excel(f'./ver1/results/Column_Based/Orth_Property/6.605_Orth_predictions_{col_s}.xlsx')#,index_col='number')
#data = pd.read_excel(f'./ver1/results/Column_Based/Orth_Property/7.605_Orth_predictions_{col_s}.xlsx')#,index_col='number')
#data = pd.read_excel(f'./ver1/results/Column_Based/Orth_Property/3.015_Orth_predictions_{col_s}.xlsx')#,index_col='number')
data_1=pd.read_excel('./train_all_f2.2_new_ranking.xlsx')
#data = np.loadtxt(f'./ver1/results/Column_Based/Orth_Property/3.015_Orth_predictions_{col_s}.txt',skiprows=1)#,index_col='number')
#data = pd.read_excel(f'./ver1/results/Column_Based/Orth_Property/5.605_Orth_predictions_MH(dN.m)_3k.xlsx')#,index_col='number')
a = 3+1
model_x = xg.XGBRegressor()
param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05,0.75, 0.10],
        'n_estimators': [100,200,300,500],
        'reg_lambda': [0, 0.1, 1, 5, 10],
        'reg_alpha': [0, 0.1, 1, 5, 10]
          }
# 提取自变量 X 和目标变量 y
X = data_1.iloc[:, 1:].values  # 排除第一行和第一列作为自变量
y = data_1.iloc[:, 0].values   # 第一行作为目标变量
model = GridSearchCV(model_x, param_grid, cv=5, n_jobs=-1)
# 创建随机森林回归模型
model_x.fit(X, y)
rf = RandomForestRegressor()

# 拟合模型
rf.fit(X, y)

# 提取特征重要性
importances = rf.feature_importances_
#importances = model_x.feature_importances_

# 对重要性进行排序
indices = np.argsort(importances)[::-1]

# 获取自变量的名称作为 x 坐标标签，并按照特征重要性排序
x_labels = data_1.columns[1:][indices]

# 获取图的名称作为 y 坐标标签
y_label = data_1.columns[0]
for i,v in enumerate(importances):
    print(f'Feature: {x_labels[i]} -- Importance: {v}')

# 可视化特征重要性
from matplotlib.colors import TABLEAU_COLORS

plt.figure()
plt.title(y_label)
#colors = [plt.cm.Pastel1(x) for x in importances]
colors =  plt.cm.rainbow(np.linspace(0, 1, len(importances)))#'skyblue'#plt.cm.cool(importances[indices])  
plt.bar(x_labels, importances[indices], color=colors, align="center")
plt.xlabel('x_Labels')  # 设置 x 轴标签
plt.ylabel("Feature Importance")  # 设置 y 轴标签为图的名称
plt.xticks(rotation=60)  # 旋转 x 轴标签
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)
plt.show()
#plt.tight_layout()