
# coding: utf-8

# * 队伍:for the dream

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
from time import time
BeginTime = time()

#读取数据
path = 'data'

test_correlation = pd.read_csv(path+'/test_correlation.csv')
train_correlation = pd.read_csv(path+'/train_correlation.csv')
all_correlation = pd.merge(train_correlation,test_correlation,how='left')
TargetID = all_correlation['Unnamed: 0']


test_fund_return =  pd.read_csv(path+'/test_fund_return.csv')
train_fund_return =  pd.read_csv(path+'/train_fund_return.csv')
all_fund_return = pd.merge(train_fund_return,test_fund_return,how='left')


test_fund_benchmark_return =  pd.read_csv(path+'/test_fund_benchmark_return.csv')
train_fund_benchmark_return =  pd.read_csv(path+'/train_fund_benchmark_return.csv')
all_fund_benchmark_return = pd.merge(train_fund_benchmark_return,test_fund_benchmark_return,how='left')


test_index_return = pd.read_csv(path+'/test_index_return.csv',encoding='GBK',index_col=0)
train_index_return =  pd.read_csv(path+'/train_index_return.csv',encoding='GBK',index_col=0)
index_return = pd.concat([train_index_return,test_index_return],axis=1)

#根据TargetID把基金对拆分为两列ID，分别为基金1和基金2 
Target1 = TargetID.map(lambda x:x.split('-')[0])
Target2 = TargetID.map(lambda x:x.split('-')[1])
SplitID = pd.concat([Target1,Target2],axis=1)
SplitID.columns = ['Target1','Target2']


#根据评分规则，定义验证函数
from sklearn.metrics import mean_absolute_error  
def model_metrics(ypred,ytrue):
    msum = 0;
    mcount = 0;
    for i in range(len(ypred)):
        msum += abs((ypred[i]-ytrue[i]) / (1.5-ytrue[i]));
        mcount +=1;
    mae = mean_absolute_error(ytrue,ypred);
    metrics_result = ((2/(2+mae+msum/mcount))**2);
    return metrics_result

#定义xgboost模型
def Xgb_To_Pred(Xtrain,label,val,Xtest,params):

    DMtrain = xgb.DMatrix(np.array(Xtrain),label);
    DMtest = xgb.DMatrix(np.array(Xtest));
    DMval = xgb.DMatrix(np.array(val));
    
    best_round=params['nrounds'];
    clf = xgb.train(params,DMtrain,best_round);
    
    return clf.predict(DMtest),clf.predict(DMval)


#定义lgboost模型
def Lgb_To_Pred(Xtrain,label,val,Xtest,params):
    
    Dtrain = lgb.Dataset(np.array(Xtrain),label);
    
    best_round=params['nrounds'];
    clf = lgb.train(params,Dtrain,best_round);
      
    return clf.predict( Xtest ),clf.predict( val ),clf.feature_importance()

#定义IdData函数：根据输入的数据集和起止时间，提取基金1和基金2的数据作为特征

def IdData(DataSet,StartTime,EndTime):
    
    DataID = DataSet[DataSet.columns[0]]
    Data   = DataSet[DataSet.columns[StartTime:EndTime]]
    
    FundData = pd.concat((DataID,Data),axis=1)
    FundData.rename(columns={FundData.columns[0]:"Target1"},inplace=True)
    Target1  = pd.merge(SplitID,FundData,how = 'left')      
    FundData.rename(columns={FundData.columns[0]:"Target2"},inplace=True)
    Target2 = pd.merge(SplitID,FundData,on = 'Target2',how = 'left')
    
    Target1 = Target1[Target1.columns[2:]]
    Target2 = Target2[Target2.columns[2:]]
    Target1.columns=range(0,Target1.shape[1])
    Target2.columns=range(0,Target2.shape[1])
    return Target1,Target2


#从相关性计算结果表中提取与TargetID相对应的数据作为特征
#因为相关性计算结果表是n*n的矩阵，我们按顺序取对角线左下区域的相关性数据。
def GetCorr(q):
    for j in range(test_fund_return.shape[0]):
        if j ==0:
            trainr = q[j][j+1:];
        else:
            x = q[j][j+1:];
            trainr = np.hstack([trainr,x]);
    return trainr


#计算各基金对Index的相关性，并计算基金对之间的曼哈顿距离之和作为特征
def GetIndexCorr(Data,StartTime,EndTime):
    a = pd.concat([Data[Data.columns[StartTime:EndTime]].T,index_return[index_return.columns[StartTime:EndTime]].T],axis=1)
    b = a.corr()[-35:]
    c = b[b.columns[:-35]].T
    d = c.rank(axis=1,ascending=False)
    e = pd.concat([all_fund_return['Unnamed: 0'],c],axis=1)
    A,B = IdData(e,1,None)
    return abs(A-B).sum(axis=1)

#计算数据集的平均值，25%、50%、75%分位值，作为特征之一
def Describe(data,StartTime,EndTime):
    a = data[data.columns[StartTime:EndTime]].T
    b= a.mean()
    c = a.quantile(0.25)
    d = a.quantile(0.5)
    e = a.quantile(0.75)
    return np.vstack([b,c,d,e]).T


#提取第一层训练集特征共5组特征:
#1、特征分别为基金对的fund_return相关性\benchmark_return相关性\fund_return累计值的相关性\fund_return累计值的曼哈顿距离\fund_return相关性

def GetFeature(StartTime,EndTime): 
    
    Date = all_fund_return.columns[StartTime:EndTime]
    FRData = all_fund_return[Date].T ;
    FRCorr = GetCorr(FRData.corr()) ;#计算并提取各基金对的fund_return相关性
    FRCumCor = GetCorr(FRData.cumsum(axis=1).corr())#计算并提取各基金对的fund_return累计值的相关性
    
    BRData = all_fund_benchmark_return[Date].T ;
    BRData = BRData.corr() ;
    BRCorr = GetCorr(BRData) ;#计算并提取各基金对的benchmark_return相关性
    
    Target1FR,Target2FR = IdData(all_fund_return,StartTime,EndTime)
    A,B = Target1FR.cumsum(axis=1), Target2FR.cumsum(axis=1)
    FRCum = abs(A[A.columns[-1]]-B[B.columns[-1]])#计算并提取各基金对fund_return累计值的曼哈顿距离
    TargetCor = (Target1FR.T).corrwith(Target2FR.T)#计算并提取各基金对fund_return相关性
    
    return np.vstack([FRCorr,FRCumCor,BRCorr,FRCum,TargetCor]).T
    

#第二层训练集特征：
#第二层特征为基金对的fund_return的曼哈顿距离求和
#定义函数：融合第一层预测结果和第二次训练集特征

Feature2date = [5,30,60,90]  #第二层训练集的统计时间段，分别为5天、30天、60天、90天

def StackFeature2(date,StackData,StartTime,EndTime):
    for i in tqdm(date):
        
        Target1FR,Target2FR = IdData(all_fund_return,-i+StartTime,EndTime)
        
        MDTargetFR = abs(Target1FR-Target2FR).sum(axis=1)     #计算基金1、2 fund_return的曼哈顿距离并求和   
        
        StackData = np.vstack([StackData,MDTargetFR])
        
    return StackData.T


#定义函数：根据给定时间间隔和次数，叠加特征集，并增加一组特征：计算基金对相关性的平均值，25%、50%、75%分位值。

def StackFeature(StartTime,EndTime,times):
    for i in tqdm(range(times)):        
        if i ==0:
            xtrain = GetFeature(StartTime,EndTime) ;
            TCorrDes = Describe(all_correlation,1,None)#计算基金对相关性的 平均值，25%、50%、75%分位值
            xtrain = np.hstack([TCorrDes,xtrain])
        else:
            DayF = StartTime-day*(i+1)
            StackTrain = GetFeature(DayF,EndTime) ;
            
            xtrain = np.hstack([xtrain,StackTrain]) ;

    return xtrain


#根据给的的时间段和叠加次数，叠加训练集以增加训练集的数据量

def StackTrain(EndTime,Time,long):
    for i in range(Time):
        
        Stacktrain = StackFeature(-day+EndTime,EndTime,times) #生成训练集
        StackTarget = all_correlation[all_correlation.columns[EndTime+60-i]] #生成训练集对应的目标集
                                      
        if i == 0 :
            TrainData = Stacktrain
            TrainTarget = StackTarget
        else:
            TrainData = np.vstack([TrainData,Stacktrain])  #叠加训练集
            TrainTarget = np.hstack([TrainTarget,StackTarget])  #叠加训练集对应的目标集
        
    return TrainData,TrainTarget


# # 生成第一层训练、预测数据

#1、定义训练目标和验证集目标
trainday=-62#训练集日期
valday=-61#验证集日期
testday=-61  #用于线下测试集，用于模型验证，
ytrain = all_correlation[all_correlation.columns[trainday+60]] ;
test_val1 = all_correlation[all_correlation.columns[valday+60]]
test_val2 = all_correlation[all_correlation.columns[testday+60]]#用于线下测试集，用于模型验证，

#设定:间隔每20天提取一次FRCorr,FRCumCor,BRCorr,FRCum,FRCorr特征，即0-20，0-40……0-400天的数据，生成训练、验证、测试数据集
#加上基金对相关性的 平均值，25%、50%、75%分位值共1004列特征
day=20
times=20
#xtrain = StackFeature(-day+trainday,trainday,times) ;
xval1 = StackFeature(-day+valday,valday,times) ;
xtest = StackFeature(-day,None,times) ;



#叠加训练集以增加训练集的数据量
xtrain,ytrain=StackTrain(trainday,10,1)


# # 开始第一层训练、预测


#xgb预测
xgb_params = {
    #'tree_method':"gpu_hist",
    'objective': 'reg:linear',
    'learning_rate': 0.3,
    'max_depth': 1,
    'subsample':1,
    'colsample_bytree':0.06,
    'alpha':50,
    'lambda':5,
    'nrounds':2100
}

xgby,xgbval = Xgb_To_Pred(xtrain,ytrain,xval1,xtest,xgb_params)

model_metrics(xgbval,test_val1),model_metrics(xgby,test_val2)

#lgb预测
lgb_params = {
   # 'device':'gpu',
    'application':'regression_l1',
    'metric':'mae',
    'seed': 0,
    'learning_rate':0.04,
    'max_depth':1,
    'feature_fraction':0.5,
    'lambda_l1':1,
    'nrounds':900
}
lgby,lgbval,q = Lgb_To_Pred(xtrain,ytrain,xval1,xtest,lgb_params)

model_metrics(lgbval,test_val1),model_metrics(lgby,test_val2)


# # 融合第一层预测结果和第二层特征，生成第二层训练、测试集

#第一层预测结果融合
strain=np.vstack([xgbval,lgbval]);
stest=np.vstack([lgby,lgby]);

#第一次预测结果和第二层特征融合
strain = StackFeature2(Feature2date,strain,valday,valday)
stest = StackFeature2(Feature2date,stest,0,None)


# # 开始第二层训练、预测

lgbs_params = {
   # 'device':'gpu',
    'application':'regression_l1',
    'seed':0,
    'learning_rate': 0.02,
    'max_depth':1,
    'feature_fraction':0.8,
    'nrounds':1400
}


y_pred,yval,q = Lgb_To_Pred(strain,test_val1,strain,stest,lgbs_params,)
print("The prediction had almost complited and It takes about " + str(time()-BeginTime) + 'second')

model_metrics(yval,test_val1),model_metrics(y_pred,test_val2)

df = pd.DataFrame({'ID':TargetID,'value':y_pred})


# In[ ]:


df.to_csv('For The Dream.csv',index=None)

