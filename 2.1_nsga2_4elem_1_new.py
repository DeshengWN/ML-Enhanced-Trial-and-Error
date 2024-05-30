import numpy as np
from pymoo.core.problem import ElementwiseProblem
from math import *
#
def divd(b,a):
  if a==0.0:
     c=1.0
  if a!= 0.0 :
     c=(b)/(a)
  return c 
def sqr(a):
 if a<0.0:
     c=sqrt(abs(a))
 if a>=0:
     c=sqrt(a)
 return c   
""" def divd(b,a):
  if a==0.0:
     c=1.0
  if a!= 0.0 :
     c=(b)/(a)
  return c 
def sqr(a):
 if a<0.0:
     c=1.0
 if a>=0:
     c=sqrt(a)
 return c     """ 
#f0 试底温升
#f0 试底温升
def fy0(x0,x1,x2,x3,x4,x5,x6,x7,x8):
    #z0=x0-sqr(x0)-x2-x3+x4+3*x6-2*x8+divd(x5,x7*x7)+divd(x1,x8)#x_1-√(x_1 )-x_3+x_5+2x_7-2x_9 〖+x_6/x_8+x〗_7 √(x_3-√(2x_7 )) ,  
    z0=x0-sqr(x0)-x2+x4+2*x6-2*x8+divd(x5,x7)+x6*sqr(x5-sqr(2*x6))
    z0 = x0+x5/x7 -x2 +x4 +3*x6-x7-x8
    z0 = x0+x5/x7-x2-x3*(x7-x6)+2*x6-2*x7-2*x8
    return z0
##f1 300定伸
def fy1(x0,x1,x2,x3,x4,x5,x6,x7,x8):
    z1=x7*x8+0.24*x0-sqr(x4+1.536)-divd(x3,x7)
    #z1 = 0.2*x0 +x7*x8-0.2*x4-x3/x7-0.1
    return z1  
##f2——losso
def fy2_ls(x1,x2,x3,x4,x5,x6,x7,x8,x9):
    z2= 0.5450334432576264  + 0.3236801884756626 * x1 + 0.35321074265506963 * x2  -0.5315757051288267 * x3 + 0.03179375912533936 * x5 + 2.2995438945347555 * x6 + 4.341061991987665 * x7 + 3.2677768337119106 * x8 + 0.010875773730096383 * x9 + 27.39651062915685
    #z2=0.545+0.32268*x0+0.35321*x1+0.53158*x2+0.03179*x4+2.2995*x5+4.3141*x6+3.2678*x7+0.0109*x8+27.3965
    return z2
##f2 硬度
def fy2_1sr(x0,x1,x2,x3,x4,x5,x6,x7,x8):
    z2=40+sqr(x0*sqr(x0*x8*(x6+x7)))
    #0.5450334432576264  + 0.3236801884756626 * x1 + 0.35321074265506963 * x2  -0.5315757051288267 * x3 + 0.03179375912533936 * x5 + 2.2995438945347555 * x6 + 4.341061991987665 * x7 + 3.2677768337119106 * x8 + 0.010875773730096383 * x9 + 27.39651062915685
    #z2=0.545+0.32268*x0+0.35321*x1+0.53158*x2+0.03179*x4+2.2995*x5+4.3141*x6+3.2678*x7+0.0109*x8+27.3965
    return z2
def fy2(x0,x1,x2,x3,x4,x5,x6,x7,x8):
    z2=(x6+x7)*(x2+sqr(x1))-sqr(x0)+0.737*(x0+sqr(sqr(x8))/0.025)
    z2=5*(sqr(x0)+x7+x8+sqr(sqr(sqr(sqr(x0))))-x8/(sqr(x0)*x6)+sqr(sqr(x0)+sqr(x1)))
    z2=5*(sqr(x0)+sqr(sqr(x0))+sqr(x6)+x7+x8)
    z2=5*(sqr(x0)+x7+x8+sqr(2*sqr(x0)+3*sqr(x6)-2*x8))
    return z2
##f5 断伸变化率
def fy5(x0,x1,x2,x3,x4,x5,x6,x7,x8):
    z5=sqr(x6)*x2*x8-x2*x8*x8-x7*sqr(x0)-0.178*x0
    return z5
""" def divd(b,a):
 if a==0:
   c=1.0
 elif a!= 0 :
  c=b/a
  return float(c) 
def sqr(a):
 if a<0:
   c=1.0
 elif a>=0:
  c=sqrt(a)
  return float(c)  """
 # 

class MyProblem(ElementwiseProblem):

 def __init__(self):
  data_ip1=np.loadtxt('./Data_Fig/NSGA2/input_f.txt',skiprows=1)#delimiter=',')
# data2=np.loadtxt('./data/result_p.txt')#delimiter=',')
  precision=2
  x1_max=round(np.max(data_ip1[:,0]),precision)
  x1_min=round(np.min(data_ip1[:,0]),precision)
  x2_max=round(np.max(data_ip1[:,1]),precision)
  x2_min=round(np.min(data_ip1[:,1]),precision)
  x3_max=round(np.max(data_ip1[:,2]),precision)
  x3_min=round(np.min(data_ip1[:,2]),precision)
  x4_max=round(np.max(data_ip1[:,3]),precision)
  x4_min=round(np.min(data_ip1[:,3]),precision)
  x5_max=round(np.max(data_ip1[:,4]),precision)
  x5_min=round(np.min(data_ip1[:,4]),precision)
  x6_max=round(np.max(data_ip1[:,5]),precision)
  x6_min=round(np.min(data_ip1[:,5]),precision)
  x7_max=round(np.max(data_ip1[:,6]),precision)
  x7_min=round(np.min(data_ip1[:,6]),precision)
  x8_max=round(np.max(data_ip1[:,7]),precision)
  x8_min=round(np.min(data_ip1[:,7]),precision)
  x9_max=round(np.max(data_ip1[:,8]),precision)
  x9_min=round(np.min(data_ip1[:,8]),precision)
  x3_min = 0.5
  
  super().__init__(n_var=9,
                         n_obj=4,
                         n_ieq_constr=4,
                         xl=np.array([x1_min,x2_min,x3_min,x4_min,x5_min,x6_min,x7_min,x8_min,x9_min]),
                         xu=np.array([x1_max,x2_max,x3_max,x4_max,x5_max,x6_max,x7_max,x8_max,x9_max]))

 def _evaluate(self, x, out, *args, **kwargs):
        #f1 = 100 * (x[0]**2 + x[1]**2)
        f1= round(fy0(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]),2)#x[0]-x[3]-x[1]
        f2=-round(fy1(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]),2)#x[0]-x[3]-x[1]
        f3= round(fy2(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]),2)#x[1]+x[3]+x[2]
        f4=-round(fy5(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]),2)#x[0]+x[1]
        #f5=-round(fy5(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]),2)#x[0]+x[1]
        #f6=-round(fy6(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]),2)#x[2]-x[3]#x[2]
# 
        g1 = fy0(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8])-25
        g2 = 10-fy1(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8])
        g3 = fy2(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8])-68
        g4 = -20-fy5(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8])

        out["F"] = [f1, f2, f3, f4]#, f5, f6]#, f4]
        #out["G"] = []#[g1, g2, g3, g4]
#        
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
#for icc in range(10):
algorithm = NSGA2(
    pop_size=200,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9,  eta=15),
    mutation=PM(prob=0.1, eta=10),
    eliminate_duplicates=True
)
problem = MyProblem()
#
# 
from pymoo.termination import get_termination

termination = get_termination("n_gen",4000)
#
from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=False,
               verbose=True)

X = res.X
F = res.F
LG=len(res.F)
print(X)
print(F)
file1=open("./Data_Fig/2.1_predict_components.txt",mode="a")
file2=open("./Data_Fig/2.1_predict_properties.txt",mode="a")
file3=open("./M1/2.1_predict_all_result.txt",mode="a")
print( F,file=file3)
print( X,file=file1)
#
#
precision = 5
data_ip2=np.loadtxt('./Data_Fig/NSGA2/input_properties.txt',skiprows=1)#delimiter=',')
data_ip3=np.loadtxt('./Data_Fig/NSGA2/expect_proper.txt',skiprows=1)#delimiter=',')
y1_max=round(np.max(data_ip2[:,0]),precision)
y1_min=round(np.min(data_ip2[:,0]),precision)
y2_max=round(np.max(data_ip2[:,1]),precision)
y2_min=round(np.min(data_ip2[:,1]),precision)
y3_max=round(np.max(data_ip2[:,2]),precision)
y3_min=round(np.min(data_ip2[:,2]),precision)
y4_max=round(np.max(data_ip2[:,3]),precision)
y4_min=round(np.min(data_ip2[:,3]),precision)
y5_max=round(np.max(data_ip2[:,4]),precision)
y5_min=round(np.min(data_ip2[:,4]),precision)
y6_max=round(np.max(data_ip2[:,5]),precision)
y6_min=round(np.min(data_ip2[:,5]),precision)
result=[]
results=[]
num=1
y1_expect=round((data_ip3[num,0]),precision)
y2_expect=round((data_ip3[num,1]),precision)
y3_expect=round((data_ip3[num,2]),precision)
y4_expect=round((data_ip3[num,3]),precision)
y5_expect=round((data_ip3[num,4]),precision)
y6_expect=round((data_ip3[num,5]),precision)
res_plot=[]

file0=open("./Data_Fig/1st_optimal_data.txt",mode="a")
j=0
 #print(LG)
max_f3=np.max(F[:,2])
for i in range(0,LG):
  print(i, F[i,0],' ',-F[i,1],' ',F[i,2],' ',-F[i,3],file=file0)#round(100*F[i,2]/max_f3,2),' ',-F[i,3],file=file0)
  x1=round(X[i][0],5)
  x2=round(X[i][1],5)
  x3=round(X[i][2],5)
  x4=round(X[i][3],5)
  x5=round(X[i][4],5)
  x6=round(X[i][5],5)
  x7=round(X[i][6],5)
  x8=round(X[i][7],5)
  x9=round(X[i][8],5)
  f1=round(fy0(x1,x2,x3,x4,x5,x6,x7,x8,x9),5)
  f2=round(fy1(x1,x2,x3,x4,x5,x6,x7,x8,x9),5)
  f3=round(fy2(x1,x2,x3,x4,x5,x6,x7,x8,x9),5)
  f4=round(fy5(x1,x2,x3,x4,x5,x6,x7,x8,x9),5)
  #f5=round(fy5(x1,x2,x3,x4,x5,x6,x7,x8,x9),5)
  #f6=round(fy6(x1,x2,x3,x4,x5,x6,x7,x8,x9),5)
  results.append([x1,x2,x3,x4,x5,x6,x7,x8,x9,f1,f2,f3,f4])
#res_plot=np.append(F[:][0],F[:][1],axis=1)
#res_plot=np.append(res_plot,-F[:][2],axis=1)
#res_plot=np.append(res_plot,-F[:][3],axis=1)
#上线界限（if else），暂时关闭
#  if f1 < 0.3* y1_min:
#    y1 = round(y1_min*0.3,5)
#  else:
  y1 = round(f1,5)
#  if f2 > 2.0* y2_max:
#    y2 = round(2.0 * y2_max,5)
#  else:
  y2 = f2
#  if f3 < 0.3* y1_min:
#    y3 = round(0.3*y3_min,5)
#  else:
  y3 = f3
#  if f4 > 2.0* y4_max:
#    y4 = round(2.0 * y4_max,5)
#  else:
  y4 = f4 
#  if f5 > 2.0* y5_max:
#    y5 = round(2.0 * y5_max,5)
#  else:
  #y5 = f5   
#  if f6 > 0.7* y6_max:
#    y6 = round(0.7 * y6_max,5)
#  else:
  #y6 = f6
  if y1 < 30:#1 * y1_expect:
      if y2 > 8:#1.05 * y2_expect:
        if y3 < 78:#1 * y3_expect:
          if y4 > -30:#1 * y6_expect:
            #if y5 > 0.7 * y5_expect:
              #if y6 > 0.7 * y6_expect: 
                 j=j+1  
                 results.append([x1,x2,x3,x4,x5,x6,x7,x8,x9,y1,y2,y3,y4])#,y5,y6])       
                 print(i, F[i,0],' ',-F[i,1],' ',F[i,2],' ',-F[i,3])                 
#print(res.F)
#print(res.X)
#print(results) """
 #LG=j
 #print(LG)
file1=open("./Data_Fig/2.1_predict_components.txt",mode="a")
file2=open("./Data_Fig/2.1_predict_properties.txt",mode="a")
file3=open("./Data_Fig/2.1_predict_all_result.txt",mode="a")
for i in range(0,LG):
  print(i,'%0.5f' %results[i][0],'%0.5f' %results[i][1],
        '%0.5f' %results[i][2],'%0.5f' %results[i][3],
        '%0.5f' %results[i][4],'%0.5f' %results[i][5],
        '%0.5f' %results[i][6],'%0.5f' %results[i][7],
        '%0.5f' %results[i][8],file=file1)
for i in range(0,LG):
  print('%0.0f' %results[i][9],'%0.0f' %results[i][10],'%0.0f' %results[i][11],
        '%0.0f' %results[i][12],#'%0.0f' %results[i][13],'%0.0f' %results[i][14],
        file=file2)
for i in range(0,LG):
  print(i,'%0.2f' %results[i][0],'%0.2f' %results[i][1],'%0.2f' %results[i][2],
        '%0.2f' %results[i][3],'%0.2f' %results[i][4],'%0.2f' %results[i][5],
        '%0.2f' %results[i][6],'%0.2f' %results[i][7],'%0.2f' %results[i][8],
        '%0.2f' %results[i][9],'%0.2f' %results[i][10],'%0.2f' %results[i][11],
        '%0.2f' %results[i][12],#'%0.2f' %results[i][13],'%0.2f' %results[i][14],
        file=file3)
#print(F)
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions
#
#ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
#plot = Scatter(tight_layout=True)
#plot.add(res.F,s=10,edgecolor='#c82423',color='k')#'#2878B5')
#plot.show() 