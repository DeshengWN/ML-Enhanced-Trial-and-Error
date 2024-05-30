import gplearn
print("gplearn verion is {}".format(gplearn.__version__))
from gplearn.genetic import SymbolicRegressor
from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
from sklearn.metrics import r2_score as R2
import matplotlib.pyplot as plt
import numpy as np
from joblib import *
import pickle
import os
import datetime
from sklearn.model_selection import train_test_split
#
#data_1=np.loadtxt('./1.1SR/300generation/rawdata/train_ten_2.txt')
data_1=np.loadtxt('./1.1SR/300generation/rawdata/train_all.txt')
#data_2=np.loadtxt('./test2_all.txt')
# data2=np.loadtxt('./data/result_p.txt')#delimiter=',')
clum=len(data_1)
row=len(data_1[1,:])
results1=[]
#a=0# 0:tem; 1:300; 2:硬度; 3:强度; 4:抽出力; 5: 断伸变化率
#
for a in range(2,3):
 y_data = data_1[:,a]#
 X_data = data_1[:,4:]
#num_t=54
#y_test = data_1[num_t:num_t+1+num_p,a]#
#X_test = data_1[num_t:num_t+1+num_p,1:]
 file1=open(f"./1.1SR/300generation/rawdata/liu_results/{a}_data2_yin-results.txt",mode="a")
 file2=open(f"./1.1SR/300generation/rawdata/liu_results/{a}_data2_yin-Equation_{a}.txt",mode="a")
 file3=open(f"./1.1SR/300generation/rawdata/liu_results/{a}_data2_yin-hyperparameter_{a}.txt",mode="a")
 file4=open(f"./1.1SR/300generation/rawdata/liu_results/{a}_data2_yin-data_{a}.txt",mode="a")
 print(f"Y{a}  ",file=file1)
 print(f"Y{a} ",file=file2)
 print(f"Y{a} ",file=file3)
 num_folds = 5
 test_size = 0.15  # 选取十分之15的数据作为预测集
# 交叉验证循环
 ct = 0
 for iteration in range(2,4):#(num_folds):

  
  fold =  iteration
  print(f'Iteration {iteration + 1}/{num_folds}')
    
    # 划分训练集和测试集
  X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=iteration)
  #test_indices = list(X_test.index)
  test_indices = np.where(np.isin(y_data, y_test))[0]
  print('a',a,y_test,file=file4)
  print('a',a,X_test,file=file4)
  print('a',a,'iteration',iteration,test_indices,file=file4)
#y += np.random.rand(64)  #给数据加点噪声
#y = np.array(y).reshape(64,1)
#y += np.random.rand(64)  #给数据加点噪声 
#y = np.array(y).reshape(64,1)

  for i in range(0,10):
   for j in range(0,5):
     for t in range(0,5):
       for cc in range(0,6):
         #for id in range(1,3):
           ct=ct+1
           #if id == 0:idp=[2,6]
           #elif id == 1:idp=[4,8]
           #elif id == 2:idp=[6,10]
           est_gp = SymbolicRegressor(population_size=5000,#000,#*id,
                           generations=40, stopping_criteria=0.01,
                           function_set=('add','sub','mul','div','sqrt'),#'max','min','sin','cos'，'abs',),##),#,'log'),
                           p_crossover=0.3+0.05*i, p_subtree_mutation=(1-(0.3+0.05*i))/3+0.01*j,
                           p_hoist_mutation=(1-(0.3+0.05*i))/3+0.01*t,#(1-(0.3+0.05*i))/3+0.01*t,
                           p_point_mutation=1-((0.3+0.05*i+(1-(0.3+0.05*i))/3+0.01*j)+(1-(0.3+0.05*i))/3+0.01*t),#((1-(0.3+0.05*i))/3+0.01*t)),
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.0005+cc*0.0005, 
                           #random_state=cc
                           #init_depth=idp
                           )
 # 打印当前时间
           time1 = datetime.datetime.now()
# 打印按指定格式排版的时间
           time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           est_gp.fit(X, y)
           y_ = est_gp.predict(X)
           y_test_ = est_gp.predict(X_test)
           score=R2(y,y_)  
           score_test=R2(y_test,y_test_)          
           print(ct,score,score_test,time2,file=file1)
           print(ct,score,score_test,time2,file=file2)
           p_crossover=0.3+0.05*i
           p_subtree_mutation=(1-(0.3+0.05*i))/3+0.01*j
           p_hoist_mutation=(1-(0.3+0.05*i))/3+0.01*t
           p_point_mutation=1-((0.3+0.05*i+(1-(0.3+0.05*i))/3+0.01*j)+((1-(0.3+0.05*i))/3+0.01*t))
           random_state=cc
           #init_depth=idp
           print(f"pc={p_crossover},ps={p_subtree_mutation},ph={p_hoist_mutation},pp={p_point_mutation},rs={random_state}",file=file3)
           print(est_gp,file=file2)
           print(ct,score,score_test,time2)
 #fold=1
 #pc=0.5+0.01*i
#

 #if R2(y,y_)>0.0:
 # filename = f'saved_models/all_models_{a}_2/model-{a}_{fold + j}.pkl'
    #with open('my_model.pkl', 'wb') as output:
  #with open(filename, 'wb') as output:
   #  pickle.dump(est_gp, output, pickle.HIGHEST_PROTOCOL)

  print(f'\n Saved model with score {score:.2f} at {time2}\n')
 
 
"""ad1 = np.array(y_).reshape(1,num_t)
for i in range(0,num_t):
    
    print(ad1[0,i])
print("========================")       
ad2 = np.array(y_test_).reshape(1,num_p)
for i in range(0,num_p):
    print(ad2[0,i]) """

""" print(R2(y_test,y_test_))
plt.scatter(y,y_,color='g')
plt.plot(y,y,color="red",linewidth=3.0,linestyle="-")
#plt.legend(["Data","func"],loc=0)
plt.scatter(y_test,y_test_,color='b')
plt.legend(["Train","Data","Predict"],loc=0)
plt.title("GP")
plt.show()
     """

