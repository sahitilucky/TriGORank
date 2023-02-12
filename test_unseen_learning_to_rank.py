import pandas as pd
import argparse
import copy
import os
import csv
import pickle
import random
import re
import pyltr
import numpy as np
from itertools import chain
import multiprocessing
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
pd.set_option('display.max_colwidth', -1)
#from ensemble_regressor import VotingRegression
#from learning_to_rank import get_features
from ontology import *
from onto_path_graph import Ontology_graph
from utils import write_relevance_judgments_topranked, precision_recall_k, get_trigenic_data

def validate_ensemble(args, save_dir, labeling=''):
    train_set, test_set = get_features(args.dataset, args.dataset_double, save_dir=save_dir, labeling=labeling)
    ids_test, X_test, y_test, test_dict, qids = test_set
    y_pred = voting_model.predict(X_test)
    y_test_temp = copy.deepcopy(y_test)
    y_test_temp = sorted(range(len(y_test_temp)), reverse=True, key=lambda i: y_test_temp[i])
    pr_lines = []
    for k in evalk:
        print('\nEvaluation for top-{} relevant items:'.format(k))
        topitems = y_test_temp[:k]
        y_test = np.array([1 if i in topitems else 0 for i in range(len(y_test_temp))])
        print('Test relevant:{}, irrelevant:{}, percentage:{}'.format(sum(y_test >= 1), sum(y_test < 1),
                                                                      sum(y_test >= 1) / float(
                                                                          sum(y_test >= 1) + sum(y_test < 1))))
        rel_jud = os.path.join(save_dir, 'relevance_judgments', 'test' + str(k) + '.csv')
        rel_res = os.path.join(save_dir, 'result_ranked', 'test' + str(k) + '.csv')
        write_relevance_judgments_topranked(ids_test, y_pred, rel_res)
        write_relevance_judgments_topranked(ids_test, y_test, rel_jud)

        top_file = os.path.join(save_dir, 'topranked', '.'.join(['test' + str(k), 'csv']))
        writer = csv.writer(open(top_file, 'w'), delimiter='\t')
        precision, recall, correct, total, relids, topranked = precision_recall_k(y_pred, y_test, k, ids_test, X_test,
                                                                                  writer=writer)
        print("Our model correct:{} total:{} precision:{} recall:{}".format(correct, total, precision, recall))

        ndcg_k = pyltr.metrics.NDCG(k=k)
        print('NDCG Random ranking:', ndcg_k.calc_mean_random(qids, y_test))
        print('NDCG Our model:', ndcg_k.calc_mean(qids, y_test, y_pred))

        process_str = "perl ireval.pl -j " + rel_jud + " < " + rel_res
        print('\n')
        stream = os.popen(process_str).read()
        print(stream)

        start_reading = False
        prec, rec = [], []
        for line in stream.split('\n'):
            if line == 'Interpolated precsion at recalls:':
                start_reading = True
            elif line == 'Non-interpolated precsion at docs:':
                start_reading = False
            elif start_reading:
                pr = line.split('=')[-1].strip()
                point = line.split('=')[0].split('at')[-1].strip()
                prec.append(float(pr))
                rec.append(float(point))
        print('precision', len(prec), prec)
        print('recall', len(rec), rec)
        pr_lines.append((prec, rec, 'test.top' + str(k)))

    plt.clf()
    for precision, recall, label in pr_lines:
        plt.step(recall, precision, where='post', label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="best")
    plt.savefig(os.path.join(save_dir, '.'.join(['PRcurve', 'ensemble', 'png'])), facecolor='white',
                edgecolor='none', bbox_inches="tight")

def get_unseen_trigenic(fname_triple, fname_double, dirname_unseen, unseen_finalfile, model):
    if not os.path.exists(dirname_unseen):
        os.makedirs(dirname_unseen)
        create_unseen_trigenic(fname_double, fname_triple, dirname_unseen)
    rank_unseen_trigenic_all(dirname_unseen, unseen_finalfile, model)

def rank_unseen_trigenic_all_2(dirname_unseen, filename, model_filename, triple_training_dataset, offset, noof_files = 100):
    ids = ['query1', 'query2', "arr"]
    triplet = get_trigenic_data(fname_triple = triple_training_dataset)
    triplet["sorted_ids"] = triplet[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    training_data_ids = {id:1 for id in triplet["sorted_ids"].values.tolist()}
    print ("Sample training data ids:", list(training_data_ids.keys())[:10])
    GO = Ontology()
    GO_graph = Ontology_graph(GO.go)
    model = pickle.load(open(model_filename, "rb"))
    print('Number of files:', len(os.listdir(dirname_unseen)))
    files = [f for f in os.listdir(dirname_unseen) if f.endswith('.pkl')]
    files.sort()
    files = files[offset : offset + noof_files]
    print ("Number of files:", len(files))
    #print (files[:10])
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    positives = pool.map(partial(rank_unseen_file_2, dirname_unseen=dirname_unseen, model=model, GO = GO, training_data_ids = training_data_ids), files)
    positives = list(chain.from_iterable(positives))
    pool.close()
    print('Number of positives:', len(positives))
    count = 0
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for x in sorted(zip(positives), reverse=True, key=lambda pair: pair[0]):
            #score, id_q1, id_q2, id_a, item = x[0] #dont know why [0]
            writer.writerow(list(x[0]))
            count += 1
            if count % 1000000 == 0: print(count)
    print ("writing to csv done...")
    triples = pd.read_csv(filename, sep='\t', header=None) #nrows=10000,
    triples.drop_duplicates(subset=[2, 3, 4], inplace=True)
    triples.to_csv(filename, sep='\t')

def rank_unseen_file_2(f, dirname_unseen, model, GO, training_data_ids):
    # TODO: remove double ids when rerun
    X_unseen, _, unseen_ids_l = pickle.load(open(os.path.join(dirname_unseen, f), 'rb'))
    ids = ['query3', 'query1', 'query2']
    fitness_features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score', 'query1_query2_score']
    unseen_ids = pd.DataFrame(X_unseen, index = unseen_ids_l.index,columns = fitness_features)
    unseen_ids['query3'] =  unseen_ids_l['array'].values.tolist()
    unseen_ids['query1'] =  unseen_ids_l['query1'].values.tolist()
    unseen_ids['query2'] =  unseen_ids_l['query2'].values.tolist()
    unseen_ids['ids'] = unseen_ids[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    unseen_ids.drop_duplicates(subset=['ids'], inplace=True)
    print ("Before unseen ids: ", unseen_ids.shape, len(training_data_ids))
    unseen_ids["not_in_training_data"] = unseen_ids['ids'].apply(lambda x: check_for_in_training(training_data_ids,x))
    unseen_ids = unseen_ids[unseen_ids["not_in_training_data"] == 1]
    print("After removing training instances unseen ids: ", unseen_ids.shape)
    term2genes, gene_idx_features = make_features_adv(GO)
    unseen_ids['has_onto_features'] = unseen_ids['ids'].apply(lambda x: has_gene_features(x,gene_idx_features))
    unseen_ids = unseen_ids[unseen_ids['has_onto_features']==1]
    unseen_ids['onto'] = unseen_ids['ids'].apply(lambda x: return_onto_features(x, term2genes, gene_idx_features, GO.go))
    X_unseen = np.array(unseen_ids[fitness_features].values.tolist()).astype(np.float32)
    #X_unseen_onto = np.array(unseen_ids["onto"].values.tolist()).astype(np.float32)
    print ("New size: ", X_unseen.shape)
    #print ("Doing hstack.........")
    #X_unseen = np.hstack((X_unseen,X_unseen_onto))
    print ("Predicting model...........")
    y_pred = model.predict(X_unseen)
    pos = []
    sl_no = 0
    for y, id_q1, id_q2, id_a, x in zip(y_pred,unseen_ids.query1, unseen_ids.query2,unseen_ids.query3, X_unseen):
        if y>=1:
            sl_no += 1
            items = [y, sl_no, id_q1, id_q2, id_a]
            #items.extend(x)
            pos.append(tuple(items))
    print(f, len(y_pred), len(pos))
    return pos

def check_for_in_training(training_data_ids,x):
    if x in training_data_ids:
        return 0
    else:
        return 1

def rank_unseen_trigenic_all(dirname_unseen, filename, model):
    print('Number of files:', len(os.listdir(dirname_unseen)))
    files = [f for f in os.listdir(dirname_unseen) if f.endswith('.pkl')]
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    positives = pool.map(partial(rank_unseen_file, dirname_unseen=dirname_unseen, model=model), files)
    positives = list(chain.from_iterable(positives))
    pool.close()
    print('Number of positives:', len(positives))
    count = 0
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for x in sorted(zip(positives), reverse=True, key=lambda pair: pair[0]):
            #score, id_q1, id_q2, id_a, item = x[0] #dont know why [0]
            writer.writerow(list(x[0]))
            count += 1
            if count % 1000000 == 0: print(count)
    triples = pd.read_csv(filename, sep='\t', header=None) #nrows=10000,
    triples.drop_duplicates(subset=[1, 2, 3], inplace=True)
    triples.to_csv("final"+filename, sep='\t')


def rank_unseen_file(f, dirname_unseen, model):
    # TODO: remove double ids when rerun
    X_unseen, _, unseen_ids = pickle.load(open(os.path.join(dirname_unseen, f), 'rb'))
    y_pred = model.predict(X_unseen)
    pos = []
    for y, id_q1, id_q2, id_a, x in zip(y_pred,unseen_ids.query1, unseen_ids.query2,unseen_ids.array, X_unseen):
        if y>=1:
            items = [y, id_q1, id_q2, id_a]
            items.extend(x)
            pos.append(tuple(items))
    print(f, len(y_pred), len(pos))
    return pos

#TODO: parallelize this process, e.g. pool.map(partial(curate_unseen, triple_ids=triple_ids), combinations(ids, 3))
def create_unseen_trigenic(fname_double, fname_triple, dirname_unseen):
    features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score', 'query1_query2_score']
    ids = ['array', 'query1', 'query2']
    triple_known = pd.read_csv(fname_triple, sep='\t')
    triple_known.drop(['triple_score'], axis=1, inplace=True)
    triple_known.rename(columns={'arr': 'array'}, inplace=True)
    triple_known.drop(['Unnamed: 0'], axis=1, inplace=True)
    triple_known.drop_duplicates(subset=ids, keep=False, inplace=True)
    triple_known.dropna(inplace=True)

    double = pd.read_csv(fname_double, sep='\t')
    double.drop(['Unnamed: 0'], axis=1, inplace=True)
    double.drop_duplicates(subset=['query', 'array'], keep=False, inplace=True)
    print('Nrounds', len(double.array.unique()))
    for count, val in enumerate(double.array.unique()):
        start = time.time()
        df1 = pd.merge(double[double.array == val], double, on='array')
        df1.drop(['array_score_y'], axis=1, inplace=True)
        df1.rename(columns={'query_x': 'query1', 'double_score_x': 'query1_query3_score',
                            'query_score_x': 'query1_score', 'array_score_x': 'query3_score',
                            'query_y': 'query2', 'double_score_y': 'query2_query3_score',
                            'query_score_y': 'query2_score'}, inplace=True)
        #one column missing (query1-query2_score), fix it with heuristic below. Might not be getting all possible combinations
        check = copy.deepcopy(double)
        check.rename(columns={'query': 'query1', 'array': 'query2'}, inplace=True)
        check.drop(['query_score', 'array_score'], axis=1, inplace=True)
        step1 = df1.merge(check, left_on=['query1', 'query2'], right_on=['query1', 'query2'])
        check = copy.deepcopy(double)
        check.rename(columns={'query': 'query2', 'array': 'query1'}, inplace=True)
        check.drop(['query_score', 'array_score'], axis=1, inplace=True)
        step2 = df1.merge(check, left_on=['query1', 'query2'], right_on=['query1', 'query2'])
        final = pd.concat([step1, step2])
        final.rename(columns={'double_score':'query1_query2_score'}, inplace=True)
        final.drop_duplicates(subset=ids, keep=False, inplace=True)
        final.dropna(inplace=True)
        #Remove the ones we have in the training set
        unseen = pd.concat([final, triple_known], sort=True)
        unseen.drop_duplicates(keep=False)
        #unseen['ids'] = unseen[ids].apply(lambda x: '_'.join(x), axis=1) #takes too much time
        X_unseen = unseen[features].to_numpy().astype(np.float32)
        end = time.time()
        print('Round:{},{} X_unseen {}'.format(end-start, count, X_unseen.shape))
        #TODO: remove double ids when rerun
        pickle.dump((X_unseen,unseen[ids], unseen[ids]), open(os.path.join('data/unseen_trigenic', val+'.pkl'), 'wb'))
        #final.to_csv(os.path.join('data/unseen_trigenic', val+'.tsv'), sep='\t')


def create_unseen_trigenic_new(fname_double, fname_triple, dirname_unseen):
    features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score', 'query1_query2_score']
    ids = ['array', 'query1', 'query2']
    triple_known = pd.read_csv(fname_triple, sep='\t')
    triple_known.drop(['triple_score'], axis=1, inplace=True)
    triple_known.rename(columns={'arr': 'array'}, inplace=True)
    triple_known.drop(['Unnamed: 0'], axis=1, inplace=True)
    triple_known.drop_duplicates(subset=ids, keep=False, inplace=True)
    triple_known.dropna(inplace=True)

    double = pd.read_csv(fname_double, sep='\t')
    double.drop(['Unnamed: 0'], axis=1, inplace=True)
    double.drop_duplicates(subset=['query', 'array'], keep=False, inplace=True)
    print('Nrounds', len(double.array.unique()))
    for count, val in enumerate(double.array.unique()):
        start = time.time()
        df1 = pd.merge(double[double.array == val], double, on='array')
        df1.drop(['array_score_y'], axis=1, inplace=True)
        df1.rename(columns={'query_x': 'query1', 'double_score_x': 'query1_query3_score',
                            'query_score_x': 'query1_score', 'array_score_x': 'query3_score',
                            'query_y': 'query2', 'double_score_y': 'query2_query3_score',
                            'query_score_y': 'query2_score'}, inplace=True)
        #one column missing (query1-query2_score), fix it with heuristic below. Might not be getting all possible combinations
        check = copy.deepcopy(double)
        check.rename(columns={'query': 'query1', 'array': 'query2'}, inplace=True)
        check.drop(['query_score', 'array_score'], axis=1, inplace=True)
        step1 = df1.merge(check, left_on=['query1', 'query2'], right_on=['query1', 'query2'])
        check = copy.deepcopy(double)
        check.rename(columns={'query': 'query2', 'array': 'query1'}, inplace=True)
        check.drop(['query_score', 'array_score'], axis=1, inplace=True)
        step2 = df1.merge(check, left_on=['query1', 'query2'], right_on=['query1', 'query2'])
        final = pd.concat([step1, step2])
        final.rename(columns={'double_score':'query1_query2_score'}, inplace=True)
        final.drop_duplicates(subset=ids, keep=False, inplace=True)
        final.dropna(inplace=True)
        #Remove the ones we have in the training set
        unseen = pd.concat([final, triple_known], sort=True)
        unseen.drop_duplicates(keep=False)
        #unseen['ids'] = unseen[ids].apply(lambda x: '_'.join(x), axis=1) #takes too much time
        X_unseen = unseen[features].to_numpy().astype(np.float32)
        end = time.time()
        print('Round:{},{} X_unseen {}'.format(end-start, count, X_unseen.shape))
        #TODO: remove double ids when rerun
        pickle.dump((X_unseen,unseen[ids], unseen[ids]), open(os.path.join('data/unseen_trigenic', val+'.pkl'), 'wb'))
        #final.to_csv(os.path.join('data/unseen_trigenic', val+'.tsv'), sep='\t')



def check_seen_unseen_pairs(triple_training_dataset, unseen_finalfile):
    features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score',
                'query1_query2_score']
    target = 'triple_score'
    ids = ['query1', 'query2', "arr"]
    triplet = get_trigenic_data(fname_triple=triple_training_dataset)
    triplet["sorted_ids"] = triplet[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    Unique_boolean = triplet.duplicated(subset = ["sorted_ids", "triple_score"]).any()
    print("Any duplicated: ", Unique_boolean)
    filename = unseen_finalfile
    all_unseen_ids = []
    with open(filename, "r") as infile:
        i = 0
        for line in infile:
            if i!=0:
                unseen_id = [line.strip().split("\t")[3],line.strip().split("\t")[4],line.strip().split("\t")[5]]
                all_unseen_ids += [("_".join(sorted(list(unseen_id))), line.strip().split("\t")[1])]
            i += 1
    intersection = []
    sorted_ids_list = list(triplet["sorted_ids"])
    print(sorted_ids_list[:10])
    print (all_unseen_ids[:10])
    for idx,unseen_id in enumerate(all_unseen_ids):
        unseen_id, score = unseen_id[0],unseen_id[1]
        #print ((triplet["sorted_ids"] == unseen_id).any())
        if (triplet["sorted_ids"] == unseen_id).any():
            intersection += [unseen_id]
            selected_row = triplet[triplet["sorted_ids"] == id]
            if selected_row.duplicated(subset=["sorted_ids"]).any():
                print(selected_row[["sorted_ids", "triple_score"]])
        if idx%10000 == 0:
            print (idx)
    '''
    for id in sorted_ids_list:
        selected_row = triplet[triplet["sorted_ids"] == id]
        if selected_row.duplicated(subset=["sorted_ids"]).any():
            print(selected_row[["sorted_ids", "triple_score"]])
    '''
    print (intersection)

def check_all_seen_unseen_data(triple_training_dataset, unseen_finalfile, outfilename):
    ids = ['query1', 'query2', "arr"]
    training_triplet = get_trigenic_data(fname_triple=triple_training_dataset)
    training_triplet["sorted_ids"] = training_triplet[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    Unique_boolean = training_triplet.duplicated(subset=["sorted_ids", "triple_score"]).any()
    print("Any duplicated: ", Unique_boolean)

    triple = pd.read_csv('../../../srv/local/work/DeepMutationsData/data/Trigenic data/aao1729_Data_S1.tsv', sep='\t')
    triple = triple.rename(columns={'Query strain ID': 'query', 'Array strain ID': 'array',
                                    'Combined mutant fitness': 'triple_score'})

    triple = triple[['query', 'array', 'triple_score']]
    triple['query'] = triple['query'].str.split('_', expand=True)[0]
    triple['array'] = triple['array'].str.split('_', expand=True)[0]
    triple[['query1', 'query2']] = triple['query'].str.split('+', expand=True)
    triple.drop(['query'], axis=1, inplace=True)
    ids = ['array', 'query1', 'query2']
    #triple.drop(['Unnamed: 0'], axis=1, inplace=True)
    triple.drop_duplicates(subset=ids, keep=False, inplace=True)
    triple.dropna(inplace=True)
    triple["sorted_ids"] = triple[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    Unique_boolean = triple.duplicated(subset = ["sorted_ids", "triple_score"]).any()
    print ("Any duplicated: ", Unique_boolean)
    all_unseen_ids = []
    filename = unseen_finalfile
    with open(filename, "r") as infile:
        i = 0
        for line in infile:
            if i != 0:
                unseen_id = [line.strip().split("\t")[3], line.strip().split("\t")[4], line.strip().split("\t")[5]]
                all_unseen_ids += [("_".join(sorted(list(unseen_id))), line.strip().split("\t")[1])]
            i += 1
    intersection = []
    intersection2 = []
    sorted_ids_list = list(triple["sorted_ids"])
    print(sorted_ids_list[:10])
    sorted_ids_list = {i:1 for i in sorted_ids_list}
    print (len(sorted_ids_list), len(all_unseen_ids))
    sorted_ids_training_list = list(training_triplet["sorted_ids"])
    print (sorted_ids_training_list[:10])
    sorted_ids_training_list = {i:1 for i in sorted_ids_training_list}
    print (all_unseen_ids[:10])
    with open(outfilename, "w") as outfile:
        for idx,unseen_id in enumerate(all_unseen_ids[:400000]):
            unseen_id, score = unseen_id[0], unseen_id[1]
            #print((triple["sorted_ids"] == unseen_id).any())
            try:
                there = sorted_ids_training_list[unseen_id]
                intersection2 += [unseen_id]
                selected_row = training_triplet[training_triplet["sorted_ids"] == unseen_id]
                if selected_row.duplicated(subset=["sorted_ids"]).any():
                    print(selected_row[["sorted_ids", "triple_score"]])
            except KeyError:
                pass

            try:
                there = sorted_ids_list[unseen_id]
                intersection += [unseen_id]
                selected_row = triple[triple["sorted_ids"] == unseen_id]
                if selected_row.duplicated(subset=["sorted_ids"]).any():
                    print(selected_row[["sorted_ids", "triple_score"]])
            except KeyError:
                outfile.write(unseen_id + " " + str(score) + "\n")

            if idx%10000 == 0:
                print (idx)
    '''
    with open("aao1729_Data_S1.txt", "w") as outfile:
        for id in sorted_ids_list:
            selected_row = triple[triple["sorted_ids"]==id]
            if selected_row.duplicated(subset = ["sorted_ids"]).any():
                print (selected_row[["sorted_ids", "triple_score"]])
            outfile.write(str(id) + "\n")
    '''
    print (intersection2)
    print(intersection)

def merge_ranked_lists(filenames, outfilename):
    filename1, filename2 = filenames[0], filenames[1]
    all_unseen_ids = {}
    with open(filename1, "r") as infile:
        i = 0
        for line in infile:
            if i != 0:
                unseen_id = line.strip().split(" ")[0]
                all_unseen_ids[unseen_id] = float(line.strip().split(" ")[1])
            i += 1
    with open(filename2, "r") as infile:
        i = 0
        for line in infile:
            if i != 0:
                unseen_id = line.strip().split(" ")[0]
                all_unseen_ids[unseen_id] =  float(line.strip().split(" ")[1])
            i += 1
    all_unseen_ids = sorted(all_unseen_ids.items(), key = lambda l:l[1], reverse=True)
    with open(outfilename, "w") as outfile:
        for (triplet,score) in all_unseen_ids:
            outfile.write(triplet + " " + str(score) + "\n")
    return

if __name__== "__main__":
    parser = argparse.ArgumentParser("Genetic triple mutation ltr models")
    parser.add_argument('-dataset', default='../../../srv/local/work/DeepMutationsData/data/triple_fitness.tsv', type=str, help="path to dataset")
    parser.add_argument('-dataset_double', default='../../../srv/local/work/DeepMutationsData/data/double_fitness.tsv', type=str, help="path to dataset")
    parser.add_argument('-dirname_unseen', default='../../../srv/local/work/DeepMutationsData/data/unseen_trigenic/', type=str, help="path to dataset")
    parser.add_argument('-unseen_finalfile', default='unseen_trigenic_ranked.txt', type=str, help="path to dataset")
    parser.add_argument('-seed', type=int, default=1, help="seed")
    parser.add_argument('-validate', type=int, default=0, choices=[0,1], help="perform validation on ensemble")
    parser.add_argument('-produce', type=int, default=1, choices=[0,1], help="perform validation on ensemble")

    parser.add_argument('-models', type=str, help="model file names, comma separated")


    args = parser.parse_args()

    #args.model = "New_experiments/onto_exps_all_tune_train_200_2/RF/all/model200all.pkl"
    #args.model = "New_experiments/onto_exps_intersct_go_graph_top10_fs_train_100/RF/baseline/model100all.pkl"
    #args.model = "New_experiments/baseline_intersct_ws_go_terms_top10_stratify_fold1/RF/baseline/model160all.pkl"
    args.model = "New_experiments/baseline_intersct_ws_go_terms_top10_stratify_fold1/RF/baseline+intersct_ws/model160all.pkl"

    args.unseen_finalfile = "New_experiments/unseen_trigenic_stratified_fold1_baseline+intersct_ws_model_first_200.txt"
    rank_unseen_trigenic_all_2(args.dirname_unseen, args.unseen_finalfile, args.model, args.dataset, offset=0, noof_files = 200)

    args.unseen_finalfile = "New_experiments/unseen_trigenic_stratified_fold1_baseline+intersct_ws_model_first_200.txt"
    new_unseen_finalfile = "New_experiments/unseen_trigenic_stratified_fold1_baseline+intersct_ws_model_first_200_remove_seens.txt"
    #rank_unseen_trigenic_all_2(args.dirname_unseen, args.unseen_finalfile, args.model, args.dataset, offset = 100)
    #check_seen_unseen_pairs(args.dataset, args.unseen_finalfile)
    check_all_seen_unseen_data(args.dataset, args.unseen_finalfile, new_unseen_finalfile)

    '''
    args.unseen_finalfile = "New_experiments/unseen_trigenic_old_data_onto_model_first_100_2.txt"
    new_unseen_finalfile = "New_experiments/unseen_trigenic_old_data_onto_model_first_100_2_remove_seens.txt"
    #rank_unseen_trigenic_all_2(args.dirname_unseen, args.unseen_finalfile, args.model, args.dataset, offset = 100)
    #check_seen_unseen_pairs(args.dataset, args.unseen_finalfile)
    check_all_seen_unseen_data(args.dataset, args.unseen_finalfile, new_unseen_finalfile)
    '''
    #merge_ranked_lists(["New_experiments/unseen_trigenic_old_data_onto_model_first_200_2_remove_seens.txt", "New_experiments/unseen_trigenic_old_data_onto_model_first_100_200_2_remove_seens.txt"], "New_experiments/unseen_trigenic_old_data_onto_model_first_200_2_remove_seens.txt" )

    '''
    #"saving10/MLP/model10.pkl,saving10/RF/model10.pkl,saving30/MLP/model30.pkl,saving30/RF/model30.pkl,"
    #"saving100/MLP/model100.pkl,saving100/RF/model100.pkl,saving200/MLP/model200.pkl,saving200/RF/model200.pkl"
    args.models = "saving10/RF/model10.pkl,saving30/RF/model30.pkl,saving100/RF/model100.pkl,saving200/RF/model200.pkl"
    #args.models = "saving10/MLP/model10.pkl,saving30/MLP/model30.pkl,saving100/MLP/model100.pkl,saving200/MLP/model200.pkl"
    random.seed(args.seed)
    np.random.seed(args.seed)

    models, evalk = [], []
    for model_filename in args.models.split(','):
        if os.path.exists(model_filename):
            model = pickle.load(open(model_filename, 'rb'))
            evalk.append(int(re.sub("[^0-9]", "", model_filename.split('/')[-1])))
            if 'n_jobs' in model.get_params().keys():
                model.set_params(n_jobs=1)
            models.append((model_filename,model))
    voting_model = VotingRegression(models)
    save_dir = 'ensemble' + str(len(evalk))
    evalk = set(evalk)
    if args.validate: validate_ensemble(args, save_dir=save_dir, labeling='')
    if args.produce: get_unseen_trigenic(args.dataset, args.dataset_double, args.dataset_unseen, args.unseen_finalfile, voting_model)
    '''