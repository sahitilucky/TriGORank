import lightgbm as lgb
import pyltr
import numpy as np
from sklearn.model_selection import StratifiedKFold
import itertools
from ranklib_ltr import *
import copy
import sys
import argparse
import random
import os

def filter_small_from_large(y_true, y_predicted, true_fitness_scores, LTR_test_file, LTR_small_test_file):
    small_data_triplets = {}
    with open(LTR_small_test_file, "r") as infile:
        for line in infile:
            triplet = line.strip().split("#")[1]
            triplet = triplet.split("triplet=")[1]
            small_data_triplets[triplet] = 1
    y_true_new, y_predicted_new, true_fitness_scores_new, triplet_predict_scores = [], [], [], []
    with open(LTR_test_file, "r") as infile:
        idx = 0
        for line in infile:
            triplet = line.strip().split("#")[1]
            triplet = triplet.split("triplet=")[1]
            try:
                there = small_data_triplets[triplet]
                y_true_new += [y_true[idx]]
                y_predicted_new += [y_predicted[idx]]
                true_fitness_scores_new += [true_fitness_scores[idx]]
                triplet_predict_scores += [(triplet, y_predicted[idx])]
            except KeyError:
                pass
            idx += 1
    print (len(y_true_new), len(y_predicted_new), len(true_fitness_scores_new))
    print ("NUMBER OF ONES: ", sum(y_true_new))
    return y_true_new, y_predicted_new, true_fitness_scores_new, triplet_predict_scores

def true_order(infile_valid, triplet_data):
    data, data_dict = read_data(triplet_data)
    true_scores = []
    with open(infile_valid, "r") as infile:
        for line in infile:
            triplet = line.strip().split("#")[1]
            triplet = triplet.split("triplet=")[1]
            triplet_target_score = data_dict[triplet][0]
            true_scores += [triplet_target_score]
    return true_scores

def partition(indexlist, n):
    random.shuffle(indexlist)
    return [indexlist[i::n] for i in range(n)]

def calc_prec(y_predicted, true_scores):
    true_top_k = sorted(list(range(len(true_scores))), key=lambda k: true_scores[k], reverse=True)
    predicted_top_k = sorted(list(range(len(y_predicted))), key=lambda k: y_predicted[k], reverse=True)
    for k in [5,10,30,50]:
        precision = len(list(set(predicted_top_k[:k]).intersection(true_top_k[:k])))
        precision = float(precision)/float(k)
        print ("Prec@", k , precision)

def convert(input_filename, out_data_filename, out_query_filename, out_query_filename2):
    input = open(input_filename, "r")
    output_feature = open(out_data_filename, "w")
    output_query = open(out_query_filename, "w")
    output_query2 = open(out_query_filename2, "w")
    cur_cnt = 0
    cur_doc_cnt = 0
    last_qid = -1
    while True:
        line = input.readline()
        if not line:
            break
        line = line.split("#")[0]
        tokens = line.split(' ')
        tokens[-1] = tokens[-1].strip()
        label = tokens[0]
        qid = int(tokens[1].split(':')[1])
        if qid != last_qid:
            if cur_doc_cnt > 0:
                output_query.write(str(cur_doc_cnt) + '\n')
                output_query2.write(str(cur_doc_cnt) + '\n')
                cur_cnt += 1
            cur_doc_cnt = 0
            last_qid = qid
        cur_doc_cnt += 1
        output_feature.write(label + ' ')
        output_feature.write(' '.join(tokens[2:]) + '\n')
    output_query.write(str(cur_doc_cnt) + '\n')
    output_query2.write(str(cur_doc_cnt) + '\n')

    input.close()
    output_query.close()
    output_feature.close()
    output_query2.close()

def split_LTR_data_into_queries(LTR_train_file, LTR_new_train_file):
    with open(LTR_train_file, "r") as infile:
        index = 0
        label_indeces = {}
        feature_data = {}
        for line in infile:
            line = line.strip()
            target_label = int(line.split(" ")[0])
            try:
                label_indeces[target_label] += [index]
            except:
                label_indeces[target_label] = [index]
            feature_data[index] = " ".join(line.split(" ")[2:])
            index += 1


    no_of_queries = max(int(index/2000),5)
    query_id_data = [[] for x in range(no_of_queries)]
    sorted_labels = list(label_indeces.keys())
    for label in label_indeces:
        index_partitions = partition(label_indeces[label], no_of_queries)
        for idx,p in enumerate(index_partitions):
            query_id_data[idx] += [(label, p)]

    #test_queries = random.sample(list(range(no_of_queries)), int(no_of_queries/5))
    train_queries = list(range(no_of_queries))

    with open(LTR_new_train_file, "w") as outfile:
        for qid in train_queries:
            no_of_docs = sum([len(p) for (label, p) in query_id_data[qid]])
            for (label, p) in query_id_data[qid]:
                for index in p:
                    outfile.write(str(label) + " ")
                    outfile.write("qid:" + str(qid) + " ")
                    outfile.write(feature_data[index] + "\n")

def split_train_test_indeces_queries(filename):
    queryids = {}
    with open(filename, "r") as infile:
        index = 0
        for line in infile:
            qid = line.split(" ")[1].split("qid:")[1]
            try:
                queryids[qid] += [index]
            except:
                queryids[qid] = [index]
            index += 1
    qids = list(queryids.keys())
    qids_partitions = partition(qids,5)
    partitions_indeces = list(range(5))
    train_test_indeces = []
    for i in partitions_indeces:
        test_indeces = []
        for qid in qids_partitions[i]:
            test_indeces += queryids[qid]
        test_index = np.array(test_indeces)
        train_indeces = []
        for j in (partitions_indeces[0:i] + partitions_indeces[i+1:5]):
            for qid in qids_partitions[j]:
                train_indeces += queryids[qid]
        train_indeces = np.array(train_indeces)
        train_test_indeces += [(train_indeces,test_index)]
    return train_test_indeces

if __name__== "__main__":
    parser = argparse.ArgumentParser("LTR models with Lightbgm")
    parser.add_argument('-LTR_train_file', type=str)
    parser.add_argument('-LTR_test_file',
                        type=str)
    parser.add_argument('-libsvm_train_file', type=str)
    parser.add_argument('-libsvm_test_file', type=str)  # 10,30,100,200 #new200
    parser.add_argument('-perfstats_outfile', type=str)
    parser.add_argument('-modelname', type=str)
    parser.add_argument('-split_queries', type=int)
    parser.add_argument('-LTR_small_test_file', type=str, default = "")

    args = parser.parse_args()
    split_queries = args.split_queries
    print ("SPLIT QUERIES: ", args.split_queries)
    '''
    LTR_train_file = "data/LTR_Trigenic_smalldata_stratified_fold1_train_data.txt"
    LTR_test_file = "data/LTR_Trigenic_smalldata_stratified_fold1_test_data.txt"
    libsvm_train_file = "data/libsvm_smalldata.train"
    libsvm_test_file = "data/libsvm_smalldata.test"
    '''

    LTR_train_file, LTR_test_file, libsvm_train_file, libsvm_test_file = args.LTR_train_file, args.LTR_test_file, args.libsvm_train_file, args.libsvm_test_file
    train_instances = len([line for line in open(LTR_train_file, "r")])
    test_instances = len([line for line in open(LTR_test_file, "r")])
    if train_instances > 10000:
        split_queries = 1
    if split_queries == 1:
        LTR_new_train_file = LTR_train_file + ".split_queries"
        split_LTR_data_into_queries(LTR_train_file, LTR_new_train_file)
        train_test_indeces = split_train_test_indeces_queries(LTR_new_train_file)
        print (train_test_indeces)
        LTR_train_file = LTR_new_train_file
        libsvm_train_file = libsvm_train_file + ".split_queries"
    #if test_instances > 10000:
    LTR_new_test_file = LTR_test_file + ".split_queries"
    split_LTR_data_into_queries(LTR_test_file, LTR_new_test_file)
    LTR_test_file = LTR_new_test_file
    libsvm_test_file = libsvm_test_file + ".split_queries"

    convert(LTR_train_file, libsvm_train_file, libsvm_train_file + ".query", libsvm_train_file + ".group")
    convert(LTR_test_file, libsvm_test_file, libsvm_test_file + ".query", libsvm_test_file + ".group")
    infile_train = libsvm_train_file
    infile_valid = libsvm_test_file
    train_data = lgb.Dataset(infile_train)
    valid_data = lgb.Dataset(infile_valid)
    train_group_size = [l.strip("\n") for l in open(infile_train + ".query")]
    valid_group_size = [l.strip("\n") for l in open(infile_valid + ".query")]
    train_data.set_group(train_group_size)
    valid_data.set_group(valid_group_size)
    if split_queries == 0:
        with open(infile_train) as infile:
            y = []
            X = []
            for line in infile:
                y += [int(line.split(" ")[0])]
                x = line.split(" ")[1:]
                x[-1] = x[-1].strip()
                #x = [float(token.split(':')[1]) for token in x]
                X = X + [x]
        X = np.array(X)
        y = np.array(y)
        split_kfold = StratifiedKFold(n_splits=5, shuffle=True)
        train_test_indeces = split_kfold.split(X,y)
        train_test_indeces_2 = []
        for train_index, test_index in train_test_indeces:
            train_test_indeces_2 += [(train_index, test_index)]
        train_test_indeces = train_test_indeces_2
        print ("coming here...")

    # Parameters:
    num_leaves = [5,10,50,100,200]
    min_data_in_leaf = [1,5,10,50]
    max_depth = [3,5,7,10]
    learning_rate = [0.01,0.05,0.1]
    #num_leaves = [200]
    #min_data_in_leaf = [50]
    #max_depth = [10]
    #learning_rate = [0.1]
    tuning_params = list(itertools.product(num_leaves,min_data_in_leaf,max_depth,learning_rate))
    params_list = []
    with open(libsvm_train_file+".query") as infile:
        max_k = max([int(l.strip()) for l in infile])
    for (a,b,c,d) in tuning_params:
        params_list += [{
            "task": "train",
            "num_leaves": a,
            "min_data_in_leaf": b,
            "max_depth": c,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [max_k],
            "learning_rate": d,
            "num_threads": 2
        }]
    #"min_sum_hessian_in_leaf": 100

    if not os.path.exists("lightbgm_models/" + args.modelname):
        #TRAINING
        best_performance = 0
        param_performance = {}
        tuninig_params_performances = []
        for idx,param in enumerate(params_list):
            #train_test_indeces_copy = copy.deepcopy(train_test_indeces)
            model = lgb.cv(param, train_data, num_boost_round=50, folds = iter(train_test_indeces), eval_train_metric=True, return_cvbooster = True)#keep_training_booster = True)
            #param_performance[tuning_params[idx]] =  [model["valid ndcg@1-mean"][-1], model["valid ndcg@3-mean"][-1], model["valid ndcg@5-mean"][-1], model["valid ndcg@10-mean"][-1], model["valid ndcg@30-mean"][-1]]
            print (param, "coming here..")
            print (list(model.keys()))
            for x in model:
                if ("valid ndcg@" in x) and ("mean" in x):
                    tuninig_params_performances += [(param, model[x][-1])]
                    print (model[x][-1])
                    if model[x][-1] > best_performance:
                        best_model = model["cvbooster"]
                        best_param = param
                        best_performance = model[x][-1]

        #for tuning_param in tuning_params:
        #    print (tuning_param, param_performance[tuning_param])
        #model = lgb.Booster(model_str = model)
        sys.stdout = open(args.perfstats_outfile, "w")
        for param_perf in tuninig_params_performances:
            print (param_perf)
        print ("BEST PERFORMANCE: ", best_performance)
        print ("BEST PARAMS: ", best_param)

        # Saving
        model = lgb.train(best_param, train_data, num_boost_round=50, keep_training_booster = True)
        model.save_model("lightbgm_models/" + args.modelname, num_iteration= model.best_iteration)

    #if not os.path.exists(args.perfstats_outfile):
    sys.stdout = open(args.perfstats_outfile, "w")
    #PREDICTING
    #loading
    model = lgb.Booster(model_file = "lightbgm_models/" + args.modelname)
    y_predicted = model.predict(infile_valid)
    #y_predicted = y_predicted.tolist()

    if "smalldata" in LTR_test_file:
        triplet_data = "../../../srv/local/work/DeepMutationsData/data/Trigenic_data_with_features/Trigenic_smalldata.txt"
    elif "largedata" in LTR_test_file:
        triplet_data = "../../../srv/local/work/DeepMutationsData/data/Trigenic_data_with_features/Trigenic_largedata.txt"
    true_fitness_scores = true_order(LTR_test_file, triplet_data)
    y_true = []
    with open(infile_valid, "r") as infile:
        for line in infile:
            y_true += [float(line.split(" ")[0])]
    print (len(y_predicted))
    calc_prec(y_predicted, true_fitness_scores)
    print (true_fitness_scores[:20])
    print (y_predicted[:20])
    print (y_true[:20])
    for K in [5,10,30]:
        ndcg_train = pyltr.metrics.NDCG(k=K)
        print ("NDCG@", K, ndcg_train.calc_mean(np.ones(len(y_true)), np.array(y_true), np.array(y_predicted)))

    if args.LTR_small_test_file != "":
        LTR_small_test_file = args.LTR_small_test_file
        print ("RESULT OF Small test set from the large test set")
        y_true,y_predicted,true_fitness_scores,triplet_predict_scores  = filter_small_from_large(y_true, y_predicted, true_fitness_scores, LTR_test_file, LTR_small_test_file)
        calc_prec(y_predicted, true_fitness_scores)
        print(true_fitness_scores[:20])
        print(y_predicted[:20])
        print(y_true[:20])
        for K in [5, 10, 30]:
            ndcg_train = pyltr.metrics.NDCG(k=K)
            print("NDCG@", K, ndcg_train.calc_mean(np.ones(len(y_true)), np.array(y_true), np.array(y_predicted)))

        triplet_predict_scores = sorted(triplet_predict_scores, key = lambda l:l[1], reverse = True)
        for i in range(100):
            print (triplet_predict_scores[i])