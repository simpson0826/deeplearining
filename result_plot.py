import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
########################MLP
df1 = pd.read_csv('./DOE/DOE_result_MLP.csv')  
df4 = pd.read_csv('./result/result_MLP.csv')
df1.columns
df4.columns
result = pd.concat([df1, df4], axis=1)
result_mean=result.groupby([result['lookback'],result['step'],result['unit1'],result['unit2'],
                result['act1'],result['act2'],result['optimizer'],result['batchSize'],
                result['drop1'],result['drop2']]).aggregate(['mean','std'])
result_mean
result_mean.to_csv('./result/result_mean_MLP.csv')
result.to_csv('./result/result_DOEconcant_MLP.csv',index=0)

#mean變化
for i in df1.columns:
    locals()['result_'+i]=result['Training_mae'].groupby([result[i]]).aggregate(['mean','std'])
    plt.plot(locals()['result_'+i].index,locals()['result_'+i]['mean']) #畫線    
    plt.title("%s mean plot"%(i))
    plt.xticks(locals()['result_'+i].index) #設定x軸刻度 
    plt.show() #顯示繪製的圖形
    
# boxplot
for i in df1.columns:
    locals()['result_'+i]=result[[i,"Training_mae"]]
    boxplot = locals()['result_'+i].boxplot(by=i)
    
########################GRU
df1 = pd.read_csv('./DOE/DOE_result_GRU.csv')  
df4 = pd.read_csv('./result/result_GRU.csv')
df1.columns
df4.columns
result = pd.concat([df1, df4], axis=1)
result_mean=result.groupby([result['lookback'],result['step'],result['unit1'],result['unit2'],
                result['act1'],result['act2'],result['optimizer'],result['batchSize'],
                result['drop1'],result['drop2'],result['dropRnn1'],result['dropRnn2']]).aggregate(['mean','std'])
result_mean
result_mean.to_csv('./result/result_mean_GRU.csv')
result.to_csv('./result/result_DOEconcant_GRU.csv',index=0)
########################concant1
df1 = pd.read_csv('./DOE/DOE_result_concant1.csv')  
df4 = pd.read_csv('./result/result_concant1.csv')
df1.columns
df4.columns
result = pd.concat([df1, df4], axis=1)
result_mean=result.groupby([result['lookback'],result['step'],result['unit1'],result['unit2'],
                result['act1'],result['act2'],result['optimizer'],result['filterSize'],
                result['poolSize'],result['batchSize'],result['drop2'],result['dropRnn2']]).aggregate(['mean','std'])
result_mean
result_mean.to_csv('./result/result_mean_concant1.csv')
result.to_csv('./result/result_DOEconcant_concant1.csv',index=0)
########################concant2
df1 = pd.read_csv('./DOE/DOE_result_concant2.csv')  
df4 = pd.read_csv('./result/result_concant2.csv')
df1.columns
df4.columns
result = pd.concat([df1, df4], axis=1)
result_mean=result.groupby([result['lookback'],result['step'],result['unit1'],result['unit2'],result['unit3'],
                result['act1'],result['act2'],result['act3'],result['optimizer'],result['filterSize'],
                result['poolSize'],result['batchSize'],result['drop2'],result['drop3'],result['dropRnn2']]).aggregate(['mean','std'])
result_mean
result_mean.to_csv('./result/result_mean_concant2.csv')
result.to_csv('./result/result_DOEconcant_concant2.csv',index=0)