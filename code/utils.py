import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import ceil
import glob
import unidecode 
from datetime import date, timedelta

from sklearn import preprocessing

import os
    
    
    
def read_meta_datasets(window,rand_weight=False):
    os.chdir("../data")
    meta_labs = []
    meta_graphs = []
    meta_features = []
    meta_y = []

    #------------------ Italy
    os.chdir("Italy")
    labels = pd.read_csv("italy_labels.csv")
    del labels["id"]
    labels = labels.set_index("name")

    sdate = date(2020, 2, 24)
    edate = date(2020, 4, 24)
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    
    
    Gs = generate_graphs_tmp(dates,"IT",rand_weight) 
    #labels = labels[,:]
    labels = labels.loc[list(Gs[0].nodes()),:]
    labels = labels.loc[:,dates]    
     
    meta_labs.append(labels)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    meta_graphs.append(gs_adj)

    features = generate_new_features(Gs ,labels ,dates ,window )

    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    meta_y.append(y)

    
    
    #------------------------- Spain
    os.chdir("../Spain")
    labels = pd.read_csv("spain_labels.csv")

    labels = labels.set_index("name")

    sdate = date(2020, 3, 12)
    edate = date(2020, 5, 12)
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    
    
    
    Gs = generate_graphs_tmp(dates,"ES",rand_weight)# 
    labels = labels.loc[list(Gs[0].nodes()),:]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10
   
    meta_labs.append(labels)

    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    meta_graphs.append(gs_adj)

    features = generate_new_features(Gs ,labels ,dates ,window )

    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    meta_y.append(y)

    
    
    #---------------- Britain
    os.chdir("../England")
    labels = pd.read_csv("england_labels.csv")
    #del labels["id"]
    labels = labels.set_index("name")

    sdate = date(2020, 3, 13)
    edate = date(2020, 5, 12)
    #Gs = generate_graphs(dates)
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]

    
    Gs = generate_graphs_tmp(dates,"EN",rand_weight)
    
    labels = labels.loc[list(Gs[0].nodes()),:]
    #print(labels.shape)
    labels = labels.loc[:,dates]    
    
    meta_labs.append(labels)

    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]
    meta_graphs.append(gs_adj)

    features = generate_new_features(Gs ,labels ,dates ,window)
    meta_features.append(features)

    y = list()
    nodes_without_labels = set()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    meta_y.append(y)

    #--- series of graphs and their respective dates
    #targets = produce_targets(dates, 'cases')
    
    
    #---------------- France
    os.chdir("../France")
    labels = pd.read_csv("france_labels.csv")
    #del labels["id"]
    labels = labels.set_index("name")

    sdate = date(2020, 3, 10)
    edate = date(2020, 5, 12)
    
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10

    
    
    Gs = generate_graphs_tmp(dates,"FR",rand_weight)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]
    
    meta_labs.append(labels)

    meta_graphs.append(gs_adj)

    features = generate_new_features(Gs ,labels ,dates ,window)

    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    meta_y.append(y)
    
    
    #---------------- New Zealand
    os.chdir("../NewZealand")
    labels = pd.read_csv("newzealand_labels.csv")
    #del labels["id"]
    labels = labels.set_index("name")

    sdate = date(2022, 3, 4)
    edate = date(2022, 9, 4)
    
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10

    
    
    Gs = generate_graphs_tmp(dates,"NZ",rand_weight)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]
    
    meta_labs.append(labels)

    meta_graphs.append(gs_adj)

    features = generate_new_features(Gs ,labels ,dates ,window, age_group=True, group_num=9)

    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    print("data Y: {}".format(np.array(y).shape))

    meta_y.append(y)



    #---------------- New Zealand (with age groups)
    os.chdir("../NewZealand")
    labels = pd.read_csv("newzealand_labels.csv")
    #del labels["id"]
    labels = labels.set_index("name")

    sdate = date(2022, 3, 4)
    edate = date(2022, 9, 4)
    
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10

    
    
    Gs = generate_graphs_tmp(dates,"NZ",rand_weight)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]
    
    meta_labs.append(labels)

    meta_graphs.append(gs_adj)

    features = generate_new_features(Gs ,labels ,dates ,window, age_group=True, group_num=10)

    meta_features.append(features)

    labels = pd.read_csv("NZ_newcase_labels_grouped.csv")
    labels = labels.set_index("name")
    for date1 in dates:
        labels[date1] = labels[date1].apply(lambda x: np.fromstring(
            x.replace('\n','')
            .replace('[','')
            .replace(']','')
            .replace('  ',' '), sep=','
        ))
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    
    print("data Y group: {}".format(np.array(y).shape))

    meta_y.append(y)
    
    os.chdir("../../code")

    return meta_labs, meta_graphs, meta_features, meta_y
    
    

def generate_graphs_tmp(dates,country,rand_weight=False,dum=False):
    Gs = []
    for date in dates:
        d = pd.read_csv("graphs/"+country+"_"+date+".csv",header=None)
        G = nx.DiGraph()
        if dum:
            nodes = set(['auckland', 'capital_and_coast_hutt', 'waikato'])  # special nz lim
        else:
            nodes = set(d[0].unique()).union(set(d[1].unique()))  
        nodes = sorted(nodes)
        G.add_nodes_from(nodes)

        if rand_weight:
            for node_start in list(G.nodes):
                for node_end in list(G.nodes):
                    G.add_edge(node_start, node_end, weight=1)
        else:
            for row in d.iterrows():
                G.add_edge(row[1][0], row[1][1], weight=row[1][2])

        Gs.append(G)
        
    return Gs






def generate_new_features(Gs, labels, dates, window=7, scaled=False, economic=False, econ_feat=0, age_group=False, group_num=0):
    """
    Generate node features
    Features[1] contains the features corresponding to y[1]
    e.g. if window = 7, features[7]= day0:day6, y[7] = day7
    if the window reaches before 0, everything is 0, so features[3] = [0,0,0,0,day0,day1,day2], y[3] = day3
    """
    features = list()
    
    labs = labels.copy()
    nodes = Gs[0].nodes()
    
    labs_arr = []
    if age_group:
        for i in range(group_num):
            age_labs = pd.read_csv("NZ_newcase_labels_group_{}.csv".format(i))
            age_labs = age_labs.set_index("name")
            age_labs = age_labs.loc[:,dates] 
            age_labs = age_labs.loc[list(Gs[0].nodes()),:]
            labs_arr.append(age_labs)
  

    #--- one hot encoded the region
    
    #print(n_departments)
    for idx,G in enumerate(Gs):
        #  Features = population, coordinates, d past cases, one hot region
        
        if economic: # adding additional elements equal length of economic features
            H = np.zeros([G.number_of_nodes(),window+econ_feat*6])
            econ_data = pd.read_csv("gdp_2020_dhb_mmnorm.csv")
            econ_data = econ_data.set_index("name")
            econ_data["ARR"] = econ_data["COMBINED"].apply(lambda x: np.fromstring(
                x.replace('\n','')
                .replace('[','')
                .replace(']','')
                .replace('  ',' '), sep=' '
            ))
        elif age_group:
            H = np.zeros([G.number_of_nodes(),window+window*group_num])
        else: 
            H = np.zeros([G.number_of_nodes(),window])

        me = labs.loc[:, dates[:(idx)]].mean(1)
        sd = labs.loc[:, dates[:(idx)]].std(1)+1

        ### enumarate because H[i] and labs[node] are not aligned
        for i,node in enumerate(G.nodes()):
            # print("i: {}".format(i))
            # print("node: {}".format(node))
            #---- Past cases      
            if(idx < window):# idx-1 goes before the start of the labels
                if(scaled):
                    #me = np.mean(labs.loc[node, dates[0:(idx)]]
                    H[i,(window-idx):(window)] = (labs.loc[node, dates[0:(idx)]] - me[node])/ sd[node]
                else:
                    H[i,(window-idx):(window)] = labs.loc[node, dates[0:(idx)]]

            elif idx >= window:
                if(scaled):
                    H[i,0:(window)] =  (labs.loc[node, dates[(idx-window):(idx)]] - me[node])/ sd[node]
                else:
                    H[i,0:(window)] = labs.loc[node, dates[(idx-window):(idx)]]

            if economic:
                # do some kind of matching here so that the economic data is appended to the second dimension of H
                # currently the shape of H (each feature vector at timestep) is (20, 7) aka (#regions, #time_feature)
                # get the economic data vectors from another file, then build a dictionary (keys are region names), then match and append to the feature vector of each corresponding region
                # perform matching inside the for loop since the variable 'node' is exactly the name of the region to be matched
                econ_row = econ_data.loc[node]["ARR"]
                H[i,(window):(window+econ_feat)] = np.fromstring(econ_row)
                H[i,(window+econ_feat):(window+econ_feat*2)] = np.power(np.fromstring(econ_row),2) / np.max(np.power(np.fromstring(econ_row),2))
                H[i,(window+econ_feat*2):(window+econ_feat*3)] = np.power(np.fromstring(econ_row),3) / np.max(np.power(np.fromstring(econ_row),3))
                H[i,(window+econ_feat*3):(window+econ_feat*4)] = np.sqrt(np.fromstring(econ_row)) / np.max(np.sqrt(np.fromstring(econ_row)))
                H[i,(window+econ_feat*4):(window+econ_feat*5)] = np.sin(np.fromstring(econ_row)) / np.max(np.sin(np.fromstring(econ_row)))
                H[i,(window+econ_feat*5):(window+econ_feat*6)] = np.cos(np.fromstring(econ_row)) / np.max(np.cos(np.fromstring(econ_row)))

        if age_group:
            for n in range(1,group_num+1):
                age_lab = labs_arr[n-1]
                me = age_lab.loc[:, dates[:(idx)]].mean(1)
                sd = age_lab.loc[:, dates[:(idx)]].std(1)+1

                ### enumarate because H[i] and labs[node] are not aligned
                for i,node in enumerate(G.nodes()):
                    #---- Past cases      
                    if(idx < window):# idx-1 goes before the start of the labels
                        if(scaled):
                            #me = np.mean(labs.loc[node, dates[0:(idx)]]
                            H[i,(window*(n+1)-idx):(window*(n+1))] = (age_lab.loc[node, dates[0:(idx)]] - me[node])/ sd[node]
                        else:
                            H[i,(window*(n+1)-idx):(window*(n+1))] = age_lab.loc[node, dates[0:(idx)]]

                    elif idx >= window:
                        if(scaled):
                            H[i,(window*n):(window*(n+1))] =  (age_lab.loc[node, dates[(idx-window):(idx)]] - me[node])/ sd[node]
                        else:
                            H[i,(window*n):(window*(n+1))] = age_lab.loc[node, dates[(idx-window):(idx)]]
      
        features.append(H)
        
    return features






def generate_new_batches(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample):
    """
    Generate batches for graphs for MPNN
    """

    N = len(idx)
    n_nodes = Gs[0].shape[0]
    #n_nodes = Gs[0].number_of_nodes()
  
    adj_lst = list()
    features_lst = list()
    y_lst = list()

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes
        step = n_nodes*graph_window

        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))

        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)

        #fill the input for each batch
        for e1,j in enumerate(range(i, min(i+batch_size, N) )):
            val = idx[j]

            # Feature[10] containes the previous 7 cases of y[10]
            for e2,k in enumerate(range(val-graph_window+1,val+1)):
                
                adj_tmp.append(Gs[k-1].T)  
                # each feature has a size of n_nodes
                features_tmp[(e1*step+e2*n_nodes):(e1*step+(e2+1)*n_nodes),:] = features[k]#-features[val-graph_window-1]
            
            
            if(test_sample>0):
                #--- val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                    
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                        
                        
            else:
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
        
        adj_tmp = sp.block_diag(adj_tmp)
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_lst, features_lst, y_lst





def generate_new_batches_group(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample, num_group=10):
    """
    Generate batches for graphs for MPNN
    """

    N = len(idx)
    n_nodes = Gs[0].shape[0]
    #n_nodes = Gs[0].number_of_nodes()
  
    adj_lst = list()
    features_lst = list()
    y_lst = list()

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes
        step = n_nodes*graph_window

        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))

        y_tmp = np.zeros(((min(i+batch_size, N)-i)*n_nodes,num_group))

        #fill the input for each batch
        for e1,j in enumerate(range(i, min(i+batch_size, N) )):
            val = idx[j]

            # Feature[10] containes the previous 7 cases of y[10]
            for e2,k in enumerate(range(val-graph_window+1,val+1)):
                
                adj_tmp.append(Gs[k-1].T)  
                # each feature has a size of n_nodes
                features_tmp[(e1*step+e2*n_nodes):(e1*step+(e2+1)*n_nodes),:] = features[k]#-features[val-graph_window-1]
            
            
            if(test_sample>0):
                #--- val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                    
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                        
                        
            else:
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
        
        adj_tmp = sp.block_diag(adj_tmp)
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_lst, features_lst, y_lst




def generate_batches_lstm(n_nodes, y, idx, window, shift, batch_size, device,test_sample):
    """
    Generate batches for graphs for the LSTM
    """
    N = len(idx)
    features_lst = list()
    y_lst = list()
    adj_fake = list()
    
    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*n_nodes*1
        #step = n_nodes#*window
        step = n_nodes*1

        adj_tmp = list()
        features_tmp = np.zeros((window, n_nodes_batch))#features.shape[1]))
        
        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)
        
        for e1,j in enumerate(range(i, min(i+batch_size, N))):
            val = idx[j]
            
            # keep the past information from val-window until val-1
            for e2,k in enumerate(range(val-window,val)):
               
                if(k==0): 
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.zeros([n_nodes])#features#[k]
                else:
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.array(y[k])#.reshape([n_nodes,1])#

            if(test_sample>0):
                # val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                        
            else:
         
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]       
         
        adj_fake.append(0)
        
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append( torch.FloatTensor(y_tmp).to(device))
        
    return adj_fake, features_lst, y_lst




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

