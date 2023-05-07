#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import networkx as nx
import numpy as np


import torch
import torch.nn.functional as F
import torch.optim as optim

from math import ceil

import itertools
import pandas as pd


from utils_nz import generate_new_features, generate_new_batches, AverageMeter,generate_batches_lstm, read_meta_datasets
from models_multiresolution import MGNN, ATMGNN, TMGNN
        

    
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
    parser.add_argument('--batch-size', type=int, default=64,
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
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10,
                        help='Seperator for validation and train set.')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    
    meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets(args.window)
    
    
    for idx in range(9):#,",          
            
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

        
        for args.model in ["ATMGNN"]:#
			#---- predict days ahead , 0-> next day etc.
            for shift in list(range(0,args.ahead)):

                result = []
                exp = 0
                fw = open("../results/results_"+"NZ_group_"+str(idx)+".csv","a")

                for test_sample in range(args.start_exp,n_samples-shift):#
                    exp+=1
                    print(test_sample)

                    #----------------- Define the split of the data
                    idx_train = list(range(args.window-1, test_sample-args.sep))
                    
                    idx_val = list(range(test_sample-args.sep,test_sample,2)) 
                                     
                    idx_train = idx_train+list(range(test_sample-args.sep+1,test_sample,2))

                    #--------------------- Baselines
                    if(args.model=="ATMGNN" or args.model=="TMGNN"):
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
                        if(args.model=="ATMGNN"):

                            model = ATMGNN(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout, nhead=1).to(device)

                        elif(args.model=="TMGNN"):

                            model = TMGNN(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout).to(device)

                        elif(args.model=="MGNN"):

                            model = MGNN(nfeat=nfeat, nhid=args.hidden, nout=1, dropout=args.dropout).to(device)

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
                                #print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg), "time=", "{:.5f}".format(time.time() - start))
                                print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),"val_loss=", "{:.5f}".format(val_loss), "time=", "{:.5f}".format(time.time() - start))

                            train_among_epochs.append(train_loss.avg)
                            val_among_epochs.append(val_loss)

                            #print(int(val_loss.detach().cpu().numpy()))

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
                                }, 'model_best_{}__group{}_shift{}.pth.tar'.format(args.model, idx, shift))

                            scheduler.step(val_loss)


                    print("validation")  
                    #print(best_val_acc)     
                    #---------------- Testing
                    test_loss = AverageMeter()

                    #print("Loading checkpoint!")
                    checkpoint = torch.load('model_best_{}__group{}_shift{}.pth.tar'.format(args.model, idx, shift))
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    model.eval()

                    #error= 0
                    #for batch in range(n_test_batches):
                    output, loss = test(adj_test[0], features_test[0], y_test[0])

                    o = output.cpu().detach().numpy()
                    l = y_test[0].cpu().numpy()

	            # average error per region
                    error = np.sum(abs(o-l))/n_nodes
			
                    # Print results
                    print("test error=", "{:.5f}".format(error))
                    result.append(error)


                print("{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+",{:.5f}".format(  np.sum(labels.iloc[:,args.start_exp:test_sample].mean(1))))

                fw.write(str(args.model)+","+str(shift)+",{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+"\n")
                fw.close()

