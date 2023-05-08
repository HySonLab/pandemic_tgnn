import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
import networkx as nx
import numpy as np
import scipy.sparse as sp


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBRegressor

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('prophet')
datas = collect_data_files('prophet')

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

            
def arima(ahead,start_exp,n_samples,labels):
    var = []
    y_pred = []
    y_true = []
    for idx in range(ahead):
        var.append([])

    error= np.zeros(ahead)
    count = 0
    for test_sample in range(start_exp,n_samples-ahead):#
        print(test_sample)
        count+=1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            if(sum(ds.iloc[:,1])==0):
                yhat = [0]*(ahead)
            else:
                try:
                    fit2 = ARIMA(ds.iloc[:,1].values, order=(2, 0, 2)).fit()
                except:
                    fit2 = ARIMA(ds.iloc[:,1].values, order=(1, 0, 0)).fit()
                yhat = abs(fit2.predict(start = test_sample , end = (test_sample+ahead-1) ))
            y_me = labels.iloc[j,test_sample:test_sample+ahead]
            e =  abs(yhat - y_me.values)
            err += e
            error += e
            y_pred.append(yhat)
            y_true.append(y_me.values)

        for idx in range(ahead):
            var[idx].append(err[idx])
    return error, var, y_pred, y_true



def gaussian_reg_time(start_exp,n_samples,labels,i_ahead,rand_seed=0):
    y_pred_mat = np.zeros((0))
    y_true_mat = np.zeros((0))

    for test_sample in range(start_exp,n_samples-i_ahead):#
        print(test_sample)
        y_pred_arr = np.zeros((0))
        y_true_arr = np.zeros((0))
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            X_train = np.empty((0, 1))
            y_train = np.empty((0))
            for k in range(ds.shape[0]):
                X_train = np.append(X_train, [[k]], axis=0)
                y_train = np.append(y_train, ds.iloc[k, 1])

            if(sum(ds.iloc[:,1])==0):
                yhat = np.array([0])
            elif(X_train.shape[0]==0):
                continue
            else:
                reg = GaussianProcessRegressor(kernel=RBF()).fit(X_train, y_train)
                yhat = reg.predict([[test_sample+i_ahead-1]])
            y_me = labels.iloc[j,test_sample+i_ahead-1]
            y_pred_arr = np.append(y_pred_arr, yhat)
            y_true_arr = np.append(y_true_arr, y_me)
        y_pred_mat = np.append(y_pred_mat, y_pred_arr)
        y_true_mat = np.append(y_true_mat, y_true_arr)

    return y_pred_mat, y_true_mat



def lin_reg_time(start_exp,n_samples,labels,i_ahead):
    y_pred_mat = np.zeros((0))
    y_true_mat = np.zeros((0))

    for test_sample in range(start_exp,n_samples-i_ahead):#
        print(test_sample)
        y_pred_arr = np.zeros((0))
        y_true_arr = np.zeros((0))
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            X_train = np.empty((0, 1))
            y_train = np.empty((0))
            for k in range(ds.shape[0]):
                X_train = np.append(X_train, [[k]], axis=0)
                y_train = np.append(y_train, ds.iloc[k, 1])

            if(sum(ds.iloc[:,1])==0):
                yhat = np.array([0])
            elif(X_train.shape[0]==0):
                continue
            else:
                reg = LinearRegression().fit(X_train, y_train)
                yhat = reg.predict([[test_sample+i_ahead-1]])
            y_me = labels.iloc[j,test_sample+i_ahead-1]
            y_pred_arr = np.append(y_pred_arr, yhat)
            y_true_arr = np.append(y_true_arr, y_me)
        y_pred_mat = np.append(y_pred_mat, y_pred_arr)
        y_true_mat = np.append(y_true_mat, y_true_arr)

    return y_pred_mat, y_true_mat



def rand_forest_time(start_exp,n_samples,labels,i_ahead,rand_seed=0):
    y_pred_mat = np.zeros((0))
    y_true_mat = np.zeros((0))

    for test_sample in range(start_exp,n_samples-i_ahead):#
        print(test_sample)
        y_pred_arr = np.zeros((0))
        y_true_arr = np.zeros((0))
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            X_train = np.empty((0, 1))
            y_train = np.empty((0))
            for k in range(ds.shape[0]):
                X_train = np.append(X_train, [[k]], axis=0)
                y_train = np.append(y_train, ds.iloc[k, 1])

            if(sum(ds.iloc[:,1])==0):
                yhat = np.array([0])
            elif(X_train.shape[0]==0):
                continue
            else:
                reg = RandomForestRegressor().fit(X_train, y_train)
                yhat = reg.predict([[test_sample+i_ahead-1]])
            y_me = labels.iloc[j,test_sample+i_ahead-1]
            y_pred_arr = np.append(y_pred_arr, yhat)
            y_true_arr = np.append(y_true_arr, y_me)
        y_pred_mat = np.append(y_pred_mat, y_pred_arr)
        y_true_mat = np.append(y_true_mat, y_true_arr)

    return y_pred_mat, y_true_mat



def xgboost(start_exp,n_samples,labels,i_ahead,rand_seed=0):
    y_pred_mat = np.zeros((0))
    y_true_mat = np.zeros((0))

    for test_sample in range(start_exp,n_samples-i_ahead):#
        print(test_sample)
        y_pred_arr = np.zeros((0))
        y_true_arr = np.zeros((0))
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            X_train = np.empty((0, 1))
            y_train = np.empty((0))
            for k in range(ds.shape[0]):
                X_train = np.append(X_train, [[k]], axis=0)
                y_train = np.append(y_train, ds.iloc[k, 1])

            if(sum(ds.iloc[:,1])==0):
                yhat = np.array([0])
            elif(X_train.shape[0]==0):
                continue
            else:
                reg = XGBRegressor(n_estimators=1000).fit(X_train, y_train)
                yhat = reg.predict([[test_sample+i_ahead-1]])
            y_me = labels.iloc[j,test_sample+i_ahead-1]
            y_pred_arr = np.append(y_pred_arr, yhat)
            y_true_arr = np.append(y_true_arr, y_me)
        y_pred_mat = np.append(y_pred_mat, y_pred_arr)
        y_true_mat = np.append(y_true_mat, y_true_arr)

    return y_pred_mat, y_true_mat



def prophet(ahead, start_exp, n_samples, labels):
    var = []
    y_pred = []
    y_true = []
    for idx in range(ahead):
        var.append([])

    error= np.zeros(ahead)
    count = 0
    for test_sample in range(start_exp,n_samples-ahead):#
        print(test_sample)
        count+=1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample].reset_index()
            ds.columns = ["ds","y"]
            #with suppress_stdout_stderr():
            m = Prophet(interval_width=0.95)
            m.fit(ds)
            future = m.predict(m.make_future_dataframe(periods=ahead))
            yhat = future["yhat"].tail(ahead)
            y_me = labels.iloc[j,test_sample:test_sample+ahead]
            e =  abs(yhat-y_me.values).values
            err += e
            error += e
            y_pred.append(yhat)
            y_true.append(y_me.values)
        for idx in range(ahead):
            var[idx].append(err[idx])
            
    return error, var, y_pred, y_true
            
            
            
            
class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super(MPNN_LSTM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)
        
        self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear( nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)#self.batch_size
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)#self.batch_size*self.n_nodes
        
        x = self.relu(self.conv1(x, adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = torch.cat(lst, dim=1)
        
        #--------------------------------------
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes
        
        x, (hn1, cn1) = self.rnn1(x)
        
        
        out2, (hn2,  cn2) = self.rnn2(x)
        
        x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)
        skip = skip.reshape(skip.size(0),-1)
                
        x = torch.cat([x,skip], dim=1)
        #--------------------------------------
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
        
        
        return x
 



class MPNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MPNN, self).__init__()
        #self.n_nodes = n_nodes
    
        #self.batch_size = batch_size
        self.nhid = nhid
        
        
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid) 
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.fc1 = nn.Linear(nfeat+2*nhid, nhid )
        self.fc2 = nn.Linear(nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
        lst.append(x)
        
        x = self.relu(self.conv1(x,adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        
        x = torch.cat(lst, dim=1)
                                   
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x)).squeeze() # 
        
        x = x.view(-1)
        
        return x

    
    
    
class BiLSTM(nn.Module):
    def __init__(self, nfeat, nhid, n_nodes, window, dropout,batch_size, recur):
        super().__init__()
        self.nhid = nhid
        self.n_nodes = n_nodes
        self.nout = n_nodes
        self.window = window
        self.nb_layers= 4
        
        self.nfeat = nfeat 
        self.recur = recur
        self.batch_size = batch_size
        self.lstm = nn.LSTM(nfeat, self.nhid, num_layers=self.nb_layers, bidirectional=True, dropout=0.2)
    
        self.linear = nn.Linear(nhid*2, nhid)
        self.linear2 = nn.Linear(nhid, self.nout)
        self.cell = ( nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).to(device)),requires_grad=True))
        
        
    def forward(self, adj, features):
        #adj is 0 here
        features = features.view(self.window,-1, self.n_nodes)#.view(-1, self.window, self.n_nodes, self.nfeat)
        
        
        #------------------
        if(self.recur):
            try:
                lstm_out, (hc,self.cell) = self.lstm(features,(torch.zeros(self.nb_layers,self.batch_size,self.nhid).cuda(),self.cell)) 
                # = (hc,cn)
            except:
                hc = torch.zeros(self.nb_layers,features.shape[1],self.nhid).cuda()                 
                cn = self.cell[:,0:features.shape[1],:].contiguous().view(2,features.shape[1],self.nhid)
                lstm_out, (hc,cn) = self.lstm(features,(hc,cn)) 
        else:
        #------------------
            lstm_out, (hc,cn) = self.lstm(features)#, self.hidden_cell)#self.hidden_cell 
        
        lstm_out = self.linear(lstm_out)
        predictions = self.linear2(lstm_out)#.view(self.window,-1,self.n_nodes)#.view(self.batch_size,self.nhid))#)
        return predictions[-1].view(-1)
