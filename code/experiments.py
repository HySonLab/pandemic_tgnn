#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import networkx as nx
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from math import ceil

import itertools
import pandas as pd


from utils import generate_new_features, generate_new_batches, AverageMeter,generate_batches_lstm, read_meta_datasets
from models import MPNN_LSTM, BiLSTM, MPNN, prophet, arima, lin_reg_time, rand_forest_time, gaussian_reg_time, xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        

    
def train(epoch, adj, features, y):
    optimizer.zero_grad()
    output = model(adj, features)
    loss_train = F.mse_loss(output, y)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train



def test(adj, features, y):    
    output = model(adj, features)
    loss_test = F.mse_loss(output, y)
    return output, loss_test

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7,
                        help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='Size of window for graphs in MPNN LSTM.')
    parser.add_argument('--recur',  default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=28,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=21,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10,
                        help='Seperator for validation and train set.')
    parser.add_argument('--rand-seed', type=int, default=0,
                        help="Specify the random seeds for reproducibility.")
    
    args = parser.parse_args()
    # torch.manual_seed(args.rand_seed)
    # random.seed(args.rand_seed)
    # np.random.seed(args.rand_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    
    meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets(args.window)
    
    
    for country in ["NZ"]:#,",
        if(country=="IT"):
            idx = 0

        elif(country=="ES"):
            idx = 1

        elif(country=="EN"):
            idx = 2

        elif(country=="FR"):
            idx = 3
	
        else:
            idx = 4
            
            
        labels = meta_labs[idx]
        gs_adj = meta_graphs[idx]
        features = meta_features[idx]
        y = meta_y[idx]
        n_samples= len(gs_adj)
        nfeat = meta_features[0][0].shape[1]
        
        n_nodes = gs_adj[0].shape[0]
        print(n_nodes)
        if not os.path.exists('../results'):
            os.makedirs('../results')
        if not os.path.exists('../Checkpoints'):
            os.makedirs('../Checkpoints')
        if not os.path.exists('../Predictions'):
            os.makedirs('../Predictions')

        
        for args.model in ["RAND_FOREST","GAUSSIAN_REG","XGBOOST"]:#
            
            if(args.model=="PROPHET"):

                error, var, y_pred, y_true = prophet(args.ahead,args.start_exp,n_samples,labels)
                count = len(range(args.start_exp,n_samples-args.ahead))
                for idx,e in enumerate(error):
                    fw = open("../results/results_"+country+"_baseline.csv","a")
                    fw.write("PROPHET,"+str(idx)+",{:.5f}".format(e/(count*n_nodes))+",{:.5f}".format(np.std(var[idx]))+",{:.5f}".format(mean_absolute_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False))+",{:.5f}".format(r2_score(y_true, y_pred))+"\n")
                    fw.close()
                continue


            if(args.model=="ARIMA"):

                error, var, y_pred, y_true = arima(args.ahead,args.start_exp,n_samples,labels)
                count = len(range(args.start_exp,n_samples-args.ahead))

                for idx,e in enumerate(error):
                    fw = open("../results/results_"+country+"_baseline.csv","a")
                    fw.write("ARIMA,"+str(idx)+",{:.5f}".format(e/(count*n_nodes))+",{:.5f}".format(np.std(var[idx]))+",{:.5f}".format(mean_absolute_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False))+",{:.5f}".format(r2_score(y_true, y_pred))+"\n")
                    fw.close()
                continue

            
            if(args.model=="LIN_REG"):

                for shift_time in range(args.ahead):
                    y_pred, y_true = lin_reg_time(args.start_exp,n_samples,labels,shift_time)
                    fw = open("../results/results_"+country+"_baseline.csv","a")
                    fw.write("LIN_REG,"+str(shift_time)+",{:.5f}".format(mean_absolute_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False))+",{:.5f}".format(r2_score(y_true, y_pred))+"\n")
                    fw.close()

                continue


            if(args.model=="RAND_FOREST"):

                for shift_time in range(args.ahead):
                    y_pred, y_true = rand_forest_time(args.start_exp,n_samples,labels,shift_time,rand_seed=args.rand_seed)
                    fw = open("../results/results_"+country+"_baseline.csv","a")
                    fw.write("RAND_FOREST,"+str(shift_time)+",{:.5f}".format(mean_absolute_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False))+",{:.5f}".format(r2_score(y_true, y_pred))+"\n")
                    fw.close()

                continue


            if(args.model=="GAUSSIAN_REG"):

                for shift_time in range(args.ahead):
                    y_pred, y_true = gaussian_reg_time(args.start_exp,n_samples,labels,shift_time,rand_seed=args.rand_seed)
                    fw = open("../results/results_"+country+"_baseline.csv","a")
                    fw.write("GAUSSIAN_REG,"+str(shift_time)+",{:.5f}".format(mean_absolute_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False))+",{:.5f}".format(r2_score(y_true, y_pred))+"\n")
                    fw.close()

                continue


            if(args.model=="XGBOOST"):

                for shift_time in range(args.ahead):
                    y_pred, y_true = xgboost(args.start_exp,n_samples,labels,shift_time,rand_seed=args.rand_seed)
                    fw = open("../results/results_"+country+"_baseline.csv","a")
                    fw.write("XGBOOST,"+str(shift_time)+",{:.5f}".format(mean_absolute_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False))+",{:.5f}".format(r2_score(y_true, y_pred))+"\n")
                    fw.close()

                continue


            prediction_set = np.empty((args.ahead, n_nodes), np.float64)
            truth_set = np.empty((args.ahead, n_nodes), np.float64)
			#---- predict days ahead , 0-> next day etc.
            for shift in list(range(0,args.ahead)):

                result = []
                y_pred = np.empty((n_nodes, 0), dtype=int)
                y_true = np.empty((n_nodes, 0), dtype=int)
                exp = 0

                for test_sample in range(args.start_exp,n_samples-shift):#
                    exp+=1
                    print(test_sample)

                    #----------------- Define the split of the data
                    idx_train = list(range(args.window-1, test_sample-args.sep))
                    
                    idx_val = list(range(test_sample-args.sep,test_sample,2)) 
                    # print('idx_val: {}'.format(idx_val))
                                     
                    idx_train = idx_train+list(range(test_sample-args.sep+1,test_sample,2))
                    # print('idx_train: {}'.format(idx_train))


                    #--------------------- Baselines
                    if(args.model=="AVG"):
                        avg = labels.iloc[:,:test_sample-1].mean(axis=1)
                        targets_lab = labels.iloc[:,test_sample+shift]
                        error = np.sum(abs(avg - targets_lab))/n_nodes
                        y_pred = np.append(y_pred, avg.to_numpy().reshape(-1,1), axis=1)
                        y_true = np.append(y_true, targets_lab.to_numpy().reshape(-1,1), axis=1)
                        print(error)
                        result.append(error)
                        continue        
                        
                    
                    if(args.model=="LAST_DAY"):
                        win_lab = labels.iloc[:,test_sample-1]
                        targets_lab = labels.iloc[:,test_sample+shift]#:(test_sample+1)]
                        error = np.sum(abs(win_lab - targets_lab))/n_nodes#/avg)
                        if(not np.isnan(error)):
                            result.append(error)
                            y_pred = np.append(y_pred, win_lab.to_numpy().reshape(-1,1), axis=1)
                            y_true = np.append(y_true, targets_lab.to_numpy().reshape(-1,1), axis=1)
                        else:
                            exp-=1
                        continue   

                    
                    if(args.model=="AVG_WINDOW"):
                        win_lab = labels.iloc[:,(test_sample-args.window):test_sample]
                        targets_lab = labels.iloc[:,test_sample+shift]#:
                        error = np.sum(abs(win_lab.mean(1) - targets_lab))/n_nodes
                        if(not np.isnan(error)):
                            result.append(error)
                            y_pred = np.append(y_pred, win_lab.mean(1).to_numpy().reshape(-1,1), axis=1)
                            y_true = np.append(y_true, targets_lab.to_numpy().reshape(-1,1), axis=1)
                        else:
                            exp-=1
                        continue   


                    if(args.model=="BiLSTM"):
                        lstm_features = 1*n_nodes
                        adj_train, features_train, y_train = generate_batches_lstm(n_nodes, y, idx_train, args.window, shift,  args.batch_size,device,test_sample)
                        adj_val, features_val, y_val = generate_batches_lstm(n_nodes, y, idx_train, args.window, shift, args.batch_size,device,test_sample)
                        adj_test, features_test, y_test = generate_batches_lstm(n_nodes, y, [test_sample],  args.window, shift,  args.batch_size,device,test_sample)


                    elif(args.model=="MPNN_LSTM"):
                        adj_train, features_train, y_train = generate_new_batches(gs_adj, features, y, idx_train, args.graph_window, shift, args.batch_size,device,test_sample)
                        adj_val, features_val, y_val = generate_new_batches(gs_adj, features, y, idx_val, args.graph_window,  shift,args.batch_size, device,test_sample)
                        adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y,  [test_sample], args.graph_window,shift, args.batch_size, device,test_sample)

                    else:
                        adj_train, features_train, y_train = generate_new_batches(gs_adj, features, y, idx_train, 1,  shift,args.batch_size,device,test_sample)
                        adj_val, features_val, y_val = generate_new_batches(gs_adj, features, y, idx_val, 1,  shift,args.batch_size,device,test_sample)
                        adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y,  [test_sample], 1,  shift,args.batch_size, device,-1)


                    n_train_batches = ceil(len(idx_train)/args.batch_size)
                    n_val_batches = 1
                    n_test_batches = 1


                    #-------------------- Training
                    # Model and optimizer
                    stop = False#
                    while(not stop):#
                        if(args.model=="BiLSTM"):

                            model = BiLSTM(nfeat=lstm_features, nhid=args.hidden, n_nodes=n_nodes, window=args.window, dropout=args.dropout,batch_size = args.batch_size, recur=args.recur).to(device)

                        elif(args.model=="MPNN_LSTM"):

                            model = MPNN_LSTM(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout).to(device)

                        elif(args.model=="MPNN"):

                            model = MPNN(nfeat=nfeat, nhid=args.hidden, nout=1, dropout=args.dropout).to(device)

                        optimizer = optim.Adam(model.parameters(), lr=args.lr)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

                        #------------------- Train
                        best_val_acc= 1e8
                        val_among_epochs = []
                        train_among_epochs = []
                        stop = False

                        for epoch in range(args.epochs):    
                            start = time.time()

                            model.train()
                            train_loss = AverageMeter()

                            # Train for one epoch
                            for batch in range(n_train_batches):
                                output, loss = train(epoch, adj_train[batch], features_train[batch], y_train[batch])
                                train_loss.update(loss.data.item(), output.size(0))

                            # Evaluate on validation set
                            model.eval()

                            #for i in range(n_val_batches):
                            output, val_loss = test(adj_val[0], features_val[0], y_val[0])
                            val_loss = float(val_loss.detach().cpu().numpy())


                            # Print results
                            if(epoch%50==0):
                                print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),"val_loss=", "{:.5f}".format(val_loss), "time=", "{:.5f}".format(time.time() - start))

                            train_among_epochs.append(train_loss.avg)
                            val_among_epochs.append(val_loss)

                            if(epoch<30 and epoch>10):
                                if(len(set([round(val_e) for val_e in val_among_epochs[-20:]])) == 1 ):
                                    #stuck= True
                                    stop = False
                                    break

                            if( epoch>args.early_stop):
                                if(len(set([round(val_e) for val_e in val_among_epochs[-50:]])) == 1):#
                                    print("break")
                                    #stop = True
                                    break

                            stop = True


                            #--------- Remember best accuracy and save checkpoint
                            if val_loss < best_val_acc:
                                best_val_acc = val_loss
                                torch.save({
                                    'state_dict': model.state_dict(),
                                    'optimizer' : optimizer.state_dict(),
                                }, '../Checkpoints/model_best_{}_shift{}_{}.pth.tar'.format(args.model, shift, country))

                            scheduler.step(val_loss)


                    print("validation")  
                    #print(best_val_acc)     
                    #---------------- Testing
                    test_loss = AverageMeter()

                    #print("Loading checkpoint!")
                    checkpoint = torch.load('../Checkpoints/model_best_{}_shift{}_{}.pth.tar'.format(args.model, shift, country))
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    model.eval()

                    #error= 0
                    #for batch in range(n_test_batches):
                    output, loss = test(adj_test[0], features_test[0], y_test[0])

                    if(args.model=="BiLSTM"):
                        o = output.view(-1).cpu().detach().numpy()
                        l = y_test[0].view(-1).cpu().numpy()
                    else:
                        o = output.cpu().detach().numpy()
                        l = y_test[0].cpu().numpy()

	                # average error per region
                    error = np.sum(abs(o-l))/n_nodes
                    y_pred = np.append(y_pred, o.reshape(-1,1), axis=1)
                    y_true = np.append(y_true, l.reshape(-1,1), axis=1)
			
                    # Print results
                    print("test error=", "{:.5f}".format(error))
                    result.append(error)

                print("{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+",{:.5f}".format(  np.sum(labels.iloc[:,args.start_exp:test_sample].mean(1))))
                print("Aux metrics: {:.5f}".format(mean_absolute_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False))+",{:.5f}".format(r2_score(y_true, y_pred)))

                if args.model == "BiLSTM":
                    fw = open("../results/results_"+country+"_LSTM.csv","a")
                else:
                    fw = open("../results/results_"+country+"_baselines.csv","a")

                # Output save metrics and specific predictions
                fw.write(str(args.model)+","+str(shift)+",{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+",{:.5f}".format(mean_absolute_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred))+",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False))+",{:.5f}".format(r2_score(y_true, y_pred))+"\n")
                fw.close()
                np.savetxt("../Predictions/predict_{}_shift{}_{}.csv".format(args.model, shift, country), y_pred, fmt="%.5f", delimiter=',')
                np.savetxt("../Predictions/truth_{}_shift{}_{}.csv".format(args.model, shift, country), y_true, fmt="%.5f", delimiter=',')