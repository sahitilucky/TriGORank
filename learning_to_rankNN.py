
import argparse
import random
from format_data import get_XY, get_iter
import numpy as np
import torch.nn as nn
import torch
import logging
import os
import sys
import pyltr
import torch.nn.functional as F
from lambdarankNN.metrics import NDCG
from lambdarankNN.early_stopping import EarlyStopping
#https://github.com/haowei01/pytorch-examples
#https://discuss.pytorch.org/t/getting-error-fluctuations-in-training-accuracy-while-implementing-ranknet-in-pytorch/38215
#https://github.com/airalcorn2/RankNet/blob/master/lambdarank.py

logger = logging.getLogger(__name__)
logging.getLogger("format_data").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',datefmt='%d-%b-%y %H:%M:%S')
#filename='lamdarank.log',

ndcg_10_train = pyltr.metrics.NDCG(k=10)
ndcg_100_train = pyltr.metrics.NDCG(k=100)
ndcg_10_val = pyltr.metrics.NDCG(k=10)
ndcg_100_val = pyltr.metrics.NDCG(k=100)
ndcg_10_test = pyltr.metrics.NDCG(k=10)
ndcg_100_test = pyltr.metrics.NDCG(k=100)
ndcg_30_test = pyltr.metrics.NDCG(k=30)
ndcg_200_test = pyltr.metrics.NDCG(k=200)

"""
LambdaRank:
From RankNet to LambdaRank to LambdaMART: An Overview
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf
ListWise Rank
1. For each query's returned document, calculate the score Si, and rank i (forward pass)
    dS / dw is calculated in this step
2. Without explicit define the loss function L, dL / dw_k = Sum_i [(dL / dS_i) * (dS_i / dw_k)]
3. for each document Di, find all other pairs j, calculate lambda:
    for rel(i) > rel(j)
    lambda += - N / (1 + exp(Si - Sj)) * (gain(rel_i) - gain(rel_j)) * |1/log(pos_i+1) - 1/log(pos_j+1)|
    for rel(i) < rel(j)
    lambda += - N / (1 + exp(Sj - Si)) * (gain(rel_i) - gain(rel_j)) * |1/log(pos_i+1) - 1/log(pos_j+1)|
    and lambda is dL / dS_i
4. in the back propagate send lambda backward to update w
to compare with RankNet factorization, the gradient back propagate is:
    pos pairs
    lambda += - 1/(1 + exp(Si - Sj))
    neg pairs
    lambda += 1/(1 + exp(Sj - Si))
to reduce the computation:
    in RankNet
    lambda = sigma * (0.5 * (1 - Sij) - 1 / (1 + exp(sigma *(Si - Sj)))))
    when Rel_i > Rel_j, Sij = 1:
        lambda = -sigma / (1 + exp(sigma(Si - Sj)))
    when Rel_i < Rel_j, Sij = -1:
        lambda = sigma  / (1 + exp(sigma(Sj - Si)))
    in LambdaRank
    lambda = sigma * (0.5 * (1 - Sij) - 1 / (1 + exp(sigma *(Si - Sj))))) * |delta_NDCG|
"""

def get_lambda_cost(Y, Y_pred, ideal_dcg, args):
    N = 1.0 / ideal_dcg.maxDCG(Y)
    # compute the rank order of each document
    (sorted_scores, sorted_idxs) = Y.sort(dim=0, descending=True)
    doc_ranks = torch.zeros(Y.shape[0], 1).to(dtype=args.precision, device=args.device)
    doc_ranks[sorted_idxs] = 1 + torch.arange(Y.shape[0]).view((Y.shape[0], 1)).to(args.device).float()
    doc_ranks = doc_ranks.view((Y.shape[0], 1))

    Y_tensor = Y.view(-1, 1)
    if args.ndcg_gain_in_train == "exp2":
        gain_diff = torch.pow(2.0, Y_tensor) - torch.pow(2.0, Y_tensor.t())
    elif args.ndcg_gain_in_train == "identity":
        gain_diff = Y_tensor - Y_tensor.t()
    else:
        raise ValueError("ndcg_gain method not supported yet {}".format(args.ndcg_gain_in_train))
    decay_diff = 1.0 / torch.log2(doc_ranks + 1.0) - 1.0 / torch.log2(doc_ranks.t() + 1.0)
    delta_ndcg = torch.abs(N * gain_diff * decay_diff)

    rel_diff = Y_tensor - Y_tensor.t()
    pos_pairs = (rel_diff > 0).type(torch.FloatTensor)
    neg_pairs = (rel_diff < 0).type(torch.FloatTensor)
    Sij = pos_pairs - neg_pairs
    num_pairs = pos_pairs.sum() + neg_pairs.sum()
    diff = Y_pred - Y_pred.t()
    # Sij ∈{0,±1} be defined to be 1 if document i has been labeled to be more relevant than document j,
    # −1 if document i has been labeled to be less relevant than document j, and 0 if they have the same label.
    pos_pairs_score_diff = 1.0 + torch.exp(args.sigma * diff)  # Pij
    lambda_update = args.sigma * (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
    lambda_update = torch.sum(lambda_update, 1, keepdim=True)
    assert lambda_update.shape == Y_pred.shape

    """
    Cost: formula in https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
    C = 0.5 * (1 - S_ij) * sigma * (si - sj) + log(1 + exp(-sigma * (si - sj)))
    when S_ij = 1:  C = log(1 + exp(-sigma(si - sj)))
    when S_ij = -1: C = log(1 + exp(-sigma(sj - si)))
    sigma can change the shape of the curve
    logsigmoid(x) = log(1 / (1 + exp(-x))) equivalent to log(1 + exp(-x))
    """
    Sij = pos_pairs - neg_pairs
    diff_pairs = pos_pairs + neg_pairs
    C = 0.5 * (1 - Sij) * args.sigma * diff - F.logsigmoid(- args.sigma * diff)
    C = C * diff_pairs
    cost = torch.sum(C, (0, 1))
    if cost.item() == float('inf') or np.isnan(cost.item()):
        import pdb; pdb.set_trace()
    return cost, lambda_update, Y_pred, num_pairs



def train(train_loader, valid_loader, net, scheduler, optimizer, early_stopping, args):
    n_rel = 512
    ideal_dcg = NDCG(n_rel, args.ndcg_gain_in_train)

    for i in range(args.num_epochs):
        total_cost, total_num_pairs = 0, 0
        net.train()
        all_y, all_pred = [], []
        for count, (ids, X, Y) in enumerate(train_loader):#.generate_batch_per_query():
            if torch.sum(Y) == 0: # all negatives, cannot learn useful signal
                continue
            y_pred = net(X)
            all_y.extend(Y.data.cpu().numpy())
            all_pred.extend(y_pred.detach().data.cpu().numpy())
            cost, lambda_update, y_pred, num_pairs = get_lambda_cost(Y, y_pred, ideal_dcg, args)
            total_cost += cost
            total_num_pairs += num_pairs
            #gradient
            net.zero_grad()
            y_pred.backward(lambda_update)
            optimizer.step()
            #scheduler.step()
        avg_cost = total_cost / float(total_num_pairs)
        logger.info("training dataset at epoch {}, total queries: {}, cost:{},  total_pairs {}".format(i, count, avg_cost, total_num_pairs))
        #logger.info('Train Phase evaluate NDCG@10:{:.5f}'.format(ndcg_10_train.calc_mean(np.ones(len(all_y)), np.array(all_y), np.array(all_pred))))
        #logger.info('Train Phase evaluate NDCG@100:{:.5f}'.format(ndcg_100_train.calc_mean(np.ones(len(all_y)), np.array(all_y), np.array(all_pred))))

        if (i + 1) % args.log_interval == 0:
            total_val_cost, total_val_pairs = 0, 0
            all_yval, all_predval = [], []
            for c, (_, X_val, Y_val) in enumerate(valid_loader):
                if torch.sum(Y_val) == 0: continue
                net.eval()
                yval_pred = net(X_val)
                all_yval.extend(Y_val)
                all_predval.extend(yval_pred)
                val_cost, _, y_pred, num_pairs = get_lambda_cost(Y_val,  yval_pred, ideal_dcg, args)
                total_val_cost += val_cost
                total_val_pairs += num_pairs
            avg_val_cost = total_val_cost / float(total_val_pairs)
            early_stopping(avg_val_cost)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            #logger.info("Eval at epoch {}, total queries: {}, cost:{},  total_pairs {}".format(i, c, avg_val_cost, total_val_pairs))
            #logger.info('Eval Phase evaluate NDCG@10:{:.5f}'.format(ndcg_10_val.calc_mean(np.ones(len(all_yval)), np.array(all_yval), np.array(all_predval))))
            #logger.info('Eval Phase evaluate NDCG@100:{:.5f}'.format(ndcg_100_val.calc_mean(np.ones(len(all_yval)), np.array(all_yval), np.array(all_predval))))


    all_yval, all_predval = [], []
    for c, (_, X_val, Y_val) in enumerate(valid_loader):
        if torch.sum(Y_val) == 0: continue
        net.eval()
        yval_pred = net(X_val)
        all_yval.extend(Y_val)
        all_predval.extend(yval_pred)
    #logger.info('finish training Eval Phase evaluate NDCG@10:{:.5f}'.format(ndcg_10_val.calc_mean(np.ones(len(all_yval)), np.array(all_yval), np.array(all_predval))))
    #logger.info('finish training Eval Phase evaluate NDCG@100:{:.5f}'.format(ndcg_100_val.calc_mean(np.ones(len(all_yval)), np.array(all_yval), np.array(all_predval))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Genetic triple mutation ltr NN models")
    parser.add_argument('-dataset', default='data/triple_fitness.tsv', type=str, help="path to dataset")
    parser.add_argument('-dataset_double', default='data/double_fitness.tsv', type=str, help="path to dataset")
    parser.add_argument('-labeling', type=str, default='', choices=['binarize', 'topk'], help="binarize: make targets 0/1, topk: top k as relevant (k user set)")
    # Training
    parser.add_argument("-optim", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument("-lr", type=float, default=0.1)
    parser.add_argument("-batch_size",type=int, default=32, help="batch size")
    parser.add_argument("-num_epochs",type=int, default=5)
    parser.add_argument("-patience",type=int, default=10)
    # Model
    parser.add_argument("-leaky_relu",  action='store_true', default=False)
    parser.add_argument("-structure",  type=list, default=[64, 32]) #change
    parser.add_argument("-sigma", type=float, default=1.0) #change
    parser.add_argument("-ndcg_gain_in_train", type=str, default="exp2", choices=["exp2","identity"])
    #Misc
    parser.add_argument('-eval_k', default="10,30,100,1000", type=str, help="list of k cutoff points (n_rel) for evaluation. First one is also used for training")
    parser.add_argument("-double_precision", action='store_true', default=False)
    parser.add_argument('-seed', type=int, default=1, help="seed")
    parser.add_argument("-log_interval", type=int, default=1)
    parser.add_argument("-gpu", type=int, default=-1)
    parser.add_argument("-save_dir",type=str, default='saving', help="directory for saving results and models")
    args = parser.parse_args()
    args.precision = torch.float64 if args.double_precision else torch.float32
    args.device = "cuda:{}".format(args.gpu) if args.gpu !=-1 else "cpu"
    #set seed
    print ('coming here 1')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.eval_k = [int(i) for i in args.eval_k.split(',')]
    #data
    (ids_train, X_train, y_train, train_dict), (ids_val, X_val, y_val, val_dict), (ids_test, X_test, y_test, test_dict) = get_XY(args.dataset, args.dataset_double, val_size=0.3, test_size=0.3, labeling=args.labeling)
    logger.info('Train relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_train >= 1), sum(y_train < 1),sum(y_train >= 1) / float(sum(y_train >= 1) + sum(y_train < 1))))
    logger.info('Val relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_val >= 1), sum(y_val < 1),sum(y_val >= 1) / float(sum(y_val >= 1) + sum(y_val < 1))))
    logger.info('Test relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_test >= 1), sum(y_test < 1), sum(y_test >= 1) / float(sum(y_test >= 1) + sum(y_test < 1))))

    train_iter =  get_iter(ids_train, X_train, y_train, device=args.device, batch_size=args.batch_size, precision=args.precision)
    val_iter = get_iter(ids_val, X_val, y_val, device=args.device, batch_size=args.batch_size, precision=args.precision)
    test_iter = get_iter(ids_test, X_test, y_test, device=args.device, batch_size=args.batch_size, precision=args.precision)
    #model
    #an object of type nn.Sequential has a forward() method
    modules = []
    for i in range(len(args.structure)):
        if i==0:
            modules.append(nn.Linear(X_train.shape[1], args.structure[i]))
            modules.append(nn.LeakyReLU() if args.leaky_relu else nn.ReLU())
        else:
            modules.append(nn.Linear(args.structure[i-1], args.structure[i]))
            modules.append(nn.LeakyReLU() if args.leaky_relu else nn.ReLU())
        if i == len(args.structure) - 1:
            modules.append(nn.Linear(args.structure[i], 1))
    net = nn.Sequential(*modules)
    # net = torch.nn.Sequential(
    #     nn.Linear(X_train.shape[1], 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 32),
    #     nn.ReLU(),
    #     nn.Linear(32, 1),
    # )

    #net = LambdaRank(lambdarank_structure, leaky_relu=args.leaky_relu, sigma=args.sigma, double_precision=args.double_precision)
    net.to(args.device)
    #net.apply(net.init_weights)
    logger.info(net)
    print ('coming here 2')
    #saving dir
    net_name = 'lamdarank-scale{}.pt'.format(args.sigma)
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    ckptfile = os.path.join(args.save_dir, net_name)
    logger.info("checkpoint dir:{}".format(ckptfile))
    #sys.stdout = open(os.path.join(args.save_dir, 'out.txt'), 'w')
    #optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) if args.optim == "adam" else torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    logger.info(optimizer)
    #train
    print ('coming here 3')
    train(train_iter, val_iter, net, scheduler, optimizer, early_stopping, args)
    print ('coming here 4')
    all_yt, all_predt = [], []
    for c, (_, X_test, Y_test) in enumerate(test_iter):
        net.eval()
        yval_pred = net(X_test)
        all_yt.extend(Y_test)
        all_predt.extend(yval_pred)
    print ('coming here 5')    
    logger.info('finish training Test Phase evaluate NDCG@10:{:.5f}'.format(ndcg_10_test.calc_mean(np.ones(len(all_yt)), np.array(all_yt), np.array(all_predt))))
    logger.info('finish training Test Phase evaluate NDCG@100:{:.5f}'.format(ndcg_100_test.calc_mean(np.ones(len(all_yt)), np.array(all_yt), np.array(all_predt))))
    print('NDCG @top10: ', ndcg_10_test.calc_mean(np.ones(len(all_yt)), np.array(all_yt), np.array(all_predt)))
    print('NDCG @top30: ', ndcg_30_test.calc_mean(np.ones(len(all_yt)), np.array(all_yt), np.array(all_predt)))
    print('NDCG @top100: ', ndcg_100_test.calc_mean(np.ones(len(all_yt)), np.array(all_yt), np.array(all_predt)))
    print('NDCG @top200: ', ndcg_200_test.calc_mean(np.ones(len(all_yt)), np.array(all_yt), np.array(all_predt)))
    # save the model
    '''logger.info('save to {}'.format(ckptfile))
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
    }, ckptfile)'''


