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
from datetime import date, timedelta

from utils import generate_new_features, generate_new_batches, AverageMeter,generate_batches_lstm, read_meta_datasets, generate_graphs_tmp
from models import MPNN_LSTM, BiLSTM, MPNN
from models_multiresolution import MGNN, ATMGNN, TMGNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def output_val(gs_adj, features, y, model, checkpoint_name, shift):

    adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y, [args.eval_start], args.graph_window, shift, args.batch_size,device,-1)

    checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
    output = model(adj_test[0], features_test[0])

    if (args.model=="LSTM"):
        o = output.view(-1).cpu().detach().numpy()
        l = y_test[0].view(-1).cpu().numpy()
    else:
        o = output.cpu().detach().numpy()
        l = y_test[0].cpu().numpy()

    return o, l


def output_val_autoreg(y_act, model, checkpoint_name, shift, rand_weight=False):

    prediction_set_inner = np.empty((args.ahead, n_nodes), np.float64)
    truth_set_inner = np.empty((args.ahead, n_nodes), np.float64)

    checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    # Test set NZ, October data
    # This initialization should be outside of the loop
    os.chdir("../data/NewZealand")
    labels = pd.read_csv("newzealand_labels.csv")
    labels = labels.set_index("name")
    labels_copy = labels.copy()
    labels_truth = labels.copy()
    os.chdir("../../code")

    sdate = date(2022, 9, 4)
    edate = date(2022, 11, 4)

    #--- replacing the predicted day data with the predict vector

    for n in range(0, args.ahead):
        print("Evaluating shift {} at autoreg time {} with rand-weights {}...".format(shift, n, rand_weight))
    #--- series of graphs and their respective dates
        delta = edate - sdate
        dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
        dates = [str(date) for date in dates]
        labels = labels_copy.loc[:,dates]    #labels.sum(1).values>10
        
        #--- generate graphs from data format
        os.chdir("../data/NewZealand")
        Gs = generate_graphs_tmp(dates,"NZ")
        gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

        labels = labels.loc[list(Gs[0].nodes()),:]
        features = generate_new_features(Gs, labels, dates, args.window, economic=False, econ_feat=21)
        os.chdir("../../code")

        y = list()
        for i,G in enumerate(Gs):
            y.append(list())
            for node in G.nodes():
                y[i].append(labels.loc[node,dates[i]])

        adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y_act, [args.eval_start+n], args.graph_window, shift, args.batch_size,device,-1)
        output = model(adj_test[0], features_test[0])

        if (args.model=="LSTM"):
            o = output.view(-1).cpu().detach().numpy()
            l = y_test[0].view(-1).cpu().numpy()
        else:
            o = output.cpu().detach().numpy()
            l = y_test[0].cpu().numpy()

        #--- replacing the predicted day data with the predict vector
        to_be_replace_with = np.array(output.cpu().detach().numpy())
        replacement_value = np.array(y_test[0].cpu().numpy())
        past_seven = []
        for cur_date in labels_copy:
            past_seven.append(cur_date)
            if (labels_copy[cur_date].to_numpy() == replacement_value).all():
                labels_copy[cur_date] = to_be_replace_with
                if n > 4:
                    labels_copy[past_seven[len(past_seven) - 5]] = labels_truth[past_seven[len(past_seven) - 5]]

        prediction_set_inner[n] = o
        truth_set_inner[n] = l

    return prediction_set_inner, truth_set_inner


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
    parser.add_argument('--eval-start', type=int, default=7,
                        help='Start day offset for evaluation on new data.')
    parser.add_argument('--rand-weights', type=bool, default=False,
                        help="True or False. Enable ablation where weights in the adjacency matrix are shuffled.")

    args = parser.parse_args()
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
        if not os.path.exists('../eval'):
            os.makedirs('../eval')


        for args.model in ["MPNN_LSTM"]:
            prediction_set = np.empty((args.ahead, n_nodes), np.float64)
            truth_set = np.empty((args.ahead, n_nodes), np.float64)

            for shift in list(range(0,args.ahead)):

                result = []
                y_pred = np.empty((n_nodes, 0), dtype=int)
                y_true = np.empty((n_nodes, 0), dtype=int)

                print("Evaluating {} at shift {}...".format(args.model, shift))

                #-------------------- Initializing
                # Model and optimizer
                if(args.model=="LSTM"):
                    model = BiLSTM(nfeat=1*n_nodes, nhid=args.hidden, n_nodes=n_nodes, window=args.window, dropout=args.dropout,batch_size = args.batch_size, recur=args.recur).to(device)

                elif(args.model=="MPNN_LSTM"):
                    model = MPNN_LSTM(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout).to(device)

                elif(args.model=="MPNN"):
                    model = MPNN(nfeat=nfeat, nhid=args.hidden, nout=1, dropout=args.dropout).to(device)

                elif(args.model=="ATMGNN"):
                    model = ATMGNN(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout, nhead=1).to(device)

                elif(args.model=="TMGNN"):
                    model = TMGNN(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout).to(device)

                elif(args.model=="MGNN"):
                    model = MGNN(nfeat=nfeat, nhid=args.hidden, nout=1, dropout=args.dropout).to(device)

                optimizer = optim.Adam(model.parameters(), lr=args.lr)

                #---------------- Testing
                prediction_set, truth_set = output_val_autoreg(y_act=meta_y[5], model=model, checkpoint_name='../Checkpoints/model_best_{}_shift{}_{}_RW_True.pth.tar'.format(args.model, shift, country), shift=shift, rand_weight=args.rand_weights)
                np.savetxt("../eval/predict_{}_autoreg_shift{}_randWeights{}_disconnected_p7.csv".format(args.model, shift, args.rand_weights), prediction_set, fmt="%.5f", delimiter=',')
                np.savetxt("../eval/truth_{}_autoreg_shift{}_randWeights{}_disconnected_p7.csv".format(args.model, shift, args.rand_weights), truth_set, fmt="%.5f", delimiter=',')