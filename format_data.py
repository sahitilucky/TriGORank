import pandas as pd
import os
import re
from collections import Counter
pd.set_option('display.max_colwidth', -1)
import pandas as pd
import numpy as np
import logging
import torch
import time
import copy
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
pd.options.mode.chained_assignment = None
from sklearn.utils import shuffle
from ontology import *#make_features, return_list_genes, make_features_adv, return_onto_features,read_go_term_sims,gene_semantic_sims
from Genesemsims import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
#http://science.sciencemag.org/content/suppl/2018/04/18/360.6386.eaao1729.DC1
#http://boonelab.ccbr.utoronto.ca/supplement/costanzo2016/
from onto_path_graph import Ontology_graph
from sklearn.model_selection import KFold
from go_term_feature_selection import go_term_feature_slctn
from utils import get_trigenic_data
import random
logger = logging.getLogger(__name__)


def get_iter(ids, X, y, device, batch_size=64, shuffle=False, precision=torch.float32):
    dataset = TensorDataset(torch.from_numpy(ids), torch.from_numpy(X).to(dtype=precision, device=device), torch.from_numpy(y).to(dtype=precision, device=device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_XY_data(fname_triple, fname_double, onto_filename, onto_features=True):
    target = 'triple_score'
    if "triple_fitness.tsv" in fname_triple:
        ids = ['query1', 'query2', "arr"]
        features = ['query1_score', 'query2_score', 'array_score', 'query1_array_score', 'query2_array_score',
                    'query1_query2_score']
    elif "triple_fitness_large.tsv" in fname_triple:
        ids = ['query1', 'query2', "query3"]
        features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score',
                    'query1_query2_score']
    feat_type = np.float32
    if not os.path.isfile(fname_triple) or not os.path.isfile(fname_double):
        if "triple_fitness.tsv" in fname_triple:
            triple, double = get_data()
        elif "triple_fitness_large.tsv" in fname_triple:
            triple, double = get_data_large()
        triple.to_csv(fname_triple, sep='\t')
        double.to_csv(fname_double, sep='\t')

    triple = pd.read_csv(fname_triple, sep='\t')
    triple.drop(['Unnamed: 0'], axis=1, inplace=True)
    triple = triple.groupby(by=ids).agg("min").reset_index()
    triple.drop_duplicates(subset=ids, inplace=True)
    triple.dropna(inplace=True)
    triple = shuffle(triple)
    print("Number of triplet instances: ", triple.shape[0])
    triple['sorted_ids'] = triple[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    # triple['ids'] = triple[['query1', 'query2']].apply(lambda x: '_'.join(x), axis=1)
    '''
    triple['sorted_ids'] = triple[ids].apply(lambda x: "_".join(list(sorted(x))),axis=1)
    triple.drop_duplicates(subset=["sorted_ids"], inplace=True)
    print("Number of training instances, testing instances: ", triple.shape[0])
    fname_triple_old = '../../../srv/local/work/DeepMutationsData/data/triple_fitness.tsv'
    triple_old = pd.read_csv(fname_triple_old, sep='\t')
    triple_old.drop(['Unnamed: 0'], axis=1, inplace=True)
    triple_old.drop_duplicates(subset=['query1', 'query2', "arr"], inplace=True)
    triple_old.dropna(inplace=True)
    triple_old = shuffle(triple_old)
    triple_old['ids'] = triple_old[['query1', 'query2']].apply(lambda x: '_'.join(x), axis=1)
    triple_old['sorted_ids'] = triple_old[['query1', 'query2','arr']].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    triple_old.drop_duplicates(subset = ["sorted_ids"], inplace=True)
    print("Number of training instances, testing instances: ", triple_old.shape[0])
    '''
    # Calculate statistics
    logger.info('Mean: %.3f' % np.mean(triple['triple_score']))
    logger.info('Variance: %.3f' % np.var(triple['triple_score']))
    logger.info('Standard Deviation: %.3f' % np.std(triple['triple_score']))
    logger.info('====================================')

    triple['rank'] = triple[target].rank(ascending=1, method='first')
    triple['ids'] = triple[ids].apply(lambda x: '_'.join(x), axis=1)
    # do feature selection using the train and apply it to test, these features will be in ["go term features"]

    GO = Ontology(onto_filename)
    GO_graph = Ontology_graph(GO.go)
    triplets = triple['ids'].to_numpy(copy="True")
    GO_graph.make_triplet_graphs(GO, triplets)
    if onto_features:
        term2genes, gene_idx_features = make_features_adv(GO)
        go_terms_sims = read_go_term_sims()
        triple['onto'] = triple['ids'].apply(lambda x: return_onto_features(x, term2genes, gene_idx_features, GO.go))

        triple['sem_sims'] = triple['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, "all"))
        similarities = ["Wang", "Jiang", "Lin", "Resnik", "Rel", "GOGO"]
        for sim_name in similarities:
            triple[sim_name] = triple['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, sim_name))

        num_regulates = 4
        num_obsolete = 0
        num_part_of = 4
        num_is_a = 4
        num_lcs = 4
        triple['regulates'] = triple['onto'].apply(lambda x: x[:num_regulates])
        triple['obsolete_features'] = triple['onto'].apply(lambda x: x[num_regulates:num_regulates + num_obsolete])
        triple['part_of_features'] = triple['onto'].apply(
            lambda x: x[num_regulates + num_obsolete:num_regulates + num_obsolete + num_part_of])
        triple['is_a_features'] = triple['onto'].apply(lambda x: x[
                                                               num_regulates + num_obsolete + num_part_of:num_regulates + num_obsolete + num_part_of + num_is_a])
        triple['part_of_lcs_similarity'] = triple['onto'].apply(
            lambda x: x[num_obsolete + num_part_of + num_is_a:num_obsolete + num_part_of + num_is_a + num_lcs])

    y = triple[target].to_numpy().astype(np.float32)
    ids_triple = triple['rank'].to_numpy().astype(np.int)
    triple_dict = dict(zip(triple['rank'], triple['ids']))

    train, test, selected_GO_term_names = go_term_feature_slctn(triple, y, None, GO, GO_graph)
    return (ids_triple, y, triple_dict, triple), selected_GO_term_names


def get_XY_train_test_data(fname_triple, fname_double, onto_filename, val_size=None, test_size=0.2, labeling='', onto_features=True, train_ids_file = None, test_ids_file = None):
    #features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score','query1_query2_score']
    start_time = time.time()
    target = 'triple_score'
    if "triple_fitness.tsv" in fname_triple:
        ids = ['query1', 'query2', "arr"]
        features = ['query1_score', 'query2_score', 'array_score', 'query1_array_score', 'query2_array_score','query1_query2_score']
    elif "triple_fitness_large.tsv" in fname_triple:
        ids = ['query1', 'query2', "query3"]
        features = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score','query1_query2_score']
    feat_type = np.float32
    if not os.path.isfile(fname_triple) or not os.path.isfile(fname_double):
        if "triple_fitness.tsv" in fname_triple:
            triple, double= get_data()
        elif "triple_fitness_large.tsv" in fname_triple:
            triple, double = get_data_large()
        triple.to_csv(fname_triple, sep='\t')
        double.to_csv(fname_double, sep='\t')
    print("Time taken to finish get features: ", time.time() - start_time)

    start_time = time.time()
    triple = pd.read_csv(fname_triple, sep='\t')
    triple.drop(['Unnamed: 0'], axis=1, inplace=True)
    triple = triple.groupby(by=ids).agg("min").reset_index()
    triple.drop_duplicates(subset=ids, inplace=True)
    triple.dropna(inplace=True)
    triple = shuffle(triple)
    print("Number of triplet instances: ", triple.shape[0])
    triple['sorted_ids'] = triple[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    print("Time taken to finish get features: ", time.time() - start_time)
    #triple['ids'] = triple[['query1', 'query2']].apply(lambda x: '_'.join(x), axis=1)
    '''
    triple['sorted_ids'] = triple[ids].apply(lambda x: "_".join(list(sorted(x))),axis=1)
    triple.drop_duplicates(subset=["sorted_ids"], inplace=True)
    print("Number of training instances, testing instances: ", triple.shape[0])
    fname_triple_old = '../../../srv/local/work/DeepMutationsData/data/triple_fitness.tsv'
    triple_old = pd.read_csv(fname_triple_old, sep='\t')
    triple_old.drop(['Unnamed: 0'], axis=1, inplace=True)
    triple_old.drop_duplicates(subset=['query1', 'query2', "arr"], inplace=True)
    triple_old.dropna(inplace=True)
    triple_old = shuffle(triple_old)
    triple_old['ids'] = triple_old[['query1', 'query2']].apply(lambda x: '_'.join(x), axis=1)
    triple_old['sorted_ids'] = triple_old[['query1', 'query2','arr']].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    triple_old.drop_duplicates(subset = ["sorted_ids"], inplace=True)
    print("Number of training instances, testing instances: ", triple_old.shape[0])
    '''
    # Calculate statistics
    start_time = time.time()
    logger.info('Mean: %.3f' % np.mean(triple['triple_score']))
    logger.info('Variance: %.3f' % np.var(triple['triple_score']))
    logger.info('Standard Deviation: %.3f' % np.std(triple['triple_score']))
    logger.info('====================================')

    start_time = time.time()
    if (train_ids_file!=None) and (test_ids_file!=None):
        train_gene_ids = []
        with open(train_ids_file, "r") as infile:
            for line in infile:
                gene_ids = line.strip().split("_")
                train_gene_ids += ["_".join(list(sorted(gene_ids)))]
        test_gene_ids = []
        with open(test_ids_file, "r") as infile:
            for line in infile:
                gene_ids = line.strip().split("_")
                test_gene_ids += ["_".join(list(sorted(gene_ids)))]
        train = triple[triple["sorted_ids"].isin(train_gene_ids)]
        test = triple[triple["sorted_ids"].isin(test_gene_ids)]
        print ("Number of training instances, testing instances: ", train_ids_file, train.shape[0], test_ids_file, test.shape[0])
    else:
        if labeling.lower().strip() == 'transfer':
            test = triple[triple['triple_score'] >= 1]
            train = triple[triple['triple_score'] < 1]
        else:
            train, test = train_test_split(triple, test_size=test_size)
        #train, test = train_test_split(triple, test_size=test_size, stratify=triple[['query1', 'query2']])
    if val_size: train, val = train_test_split(train, test_size=val_size)

    train['rank'] = train[target].rank(ascending=1, method='first')
    train['ids'] = train[ids].apply(lambda x: '_'.join(x), axis=1)
    test['ids'] = test[ids].apply(lambda x: '_'.join(x), axis=1)

    #triple["rank"] = triple[target].rank(ascending=1, method='first')
    #triple["reverserank"] = triple[target].rank(ascending=0, method='first')
    triple['ids'] = triple[ids].apply(lambda x: '_'.join(x), axis=1)
    #do feature selection using the train and apply it to test, these features will be in ["go term features"]
    print("Time taken to finish get features: ", time.time() - start_time)

    start_time = time.time()
    GO = Ontology(onto_filename)
    GO_graph = Ontology_graph(GO.go)
    triplets = triple['ids'].to_numpy(copy="True")
    GO_graph.make_triplet_graphs(GO, triplets)
    if (val_size):
        val['ids'] = val[ids].apply(lambda x: '_'.join(x), axis=1)
    print("Time taken to finish get features: ", time.time() - start_time)

    if onto_features:
        start_time = time.time()
        term2genes, gene_idx_features = make_features_adv(GO)
        go_terms_sims = read_go_term_sims()
        train['onto'] = train['ids'].apply(lambda x: return_onto_features(x, term2genes, gene_idx_features, GO.go))
        #train['onto'] = pad_sequences(train['onto'], maxlen=maxlen, padding='post', value=0).tolist()
        test['onto'] = test['ids'].apply(lambda x: return_onto_features(x, term2genes, gene_idx_features, GO.go))
        #test['onto'] = pad_sequences(test['onto'], maxlen=maxlen, padding='post', value=0).tolist()
        if val_size:
            val['onto'] = val['ids'].apply(lambda x: return_onto_features(x, term2genes, gene_idx_features, GO.go))
            #val['onto'] = pad_sequences(val['onto'], maxlen=maxlen, padding='post', value=0).tolist()
        print("Time taken to finish get features: ", time.time() - start_time)
        #train['onto'] = train['onto'].apply(lambda x: x[:-4])
        #test['onto'] = test['onto'].apply(lambda x: x[:-4])
        start_time = time.time()
        train['sem_sims'] = train['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, "all"))
        test['sem_sims'] = test['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, "all"))
        if val_size:
            #val['onto'] = val['onto'].apply(lambda x: x[:-4])
            val['sem_sims'] = val['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, "all"))
        similarities = ["Wang", "Jiang", "Lin", "Resnik", "Rel", "GOGO"]
        for sim_name in similarities:
            train[sim_name] = train['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, sim_name))
            test[sim_name] = test['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, sim_name))
            if val_size:
                val[sim_name] = val['ids'].apply(lambda x: gene_semantic_sims(x, term2genes, go_terms_sims, sim_name))
        print("Time taken to finish get features: ", time.time() - start_time)

        start_time = time.time()
        num_regulates = 4
        num_obsolete = 0
        num_part_of = 4
        num_is_a = 4
        num_lcs = 4
        train['regulates'] = train['onto'].apply(lambda x: x[:num_regulates])
        train['obsolete_features'] = train['onto'].apply(lambda x: x[num_regulates:num_regulates+num_obsolete])
        train['part_of_features'] = train['onto'].apply(lambda x: x[num_regulates+num_obsolete:num_regulates+num_obsolete+num_part_of])
        train['is_a_features'] = train['onto'].apply(lambda x: x[num_regulates+num_obsolete+num_part_of:num_regulates+num_obsolete+num_part_of+num_is_a])
        train['part_of_lcs_similarity'] = train['onto'].apply(lambda x: x[num_obsolete+num_part_of+num_is_a:num_obsolete+num_part_of+num_is_a+num_lcs])
        test['regulates'] = test['onto'].apply(lambda x: x[:num_regulates])
        test['obsolete_features'] = test['onto'].apply(lambda x: x[num_regulates:num_regulates+num_obsolete])
        test['part_of_features'] = test['onto'].apply(lambda x: x[num_regulates+num_obsolete:num_regulates+num_obsolete+num_part_of])
        test['is_a_features'] = test['onto'].apply(lambda x: x[num_regulates+num_obsolete+num_part_of:num_regulates+num_obsolete+num_part_of+num_is_a])
        test['is_a_features'] = test['onto'].apply(lambda x: x[num_regulates+num_obsolete+num_part_of:num_regulates+num_obsolete+num_part_of+num_is_a])
        test['part_of_lcs_similarity'] = test['onto'].apply(lambda x: x[num_obsolete+num_part_of+num_is_a:num_obsolete+num_part_of+num_is_a+num_lcs])
        if val_size:
            val['regulates'] = val['onto'].apply(lambda x: x[:num_regulates])
            val['obsolete_features'] = val['onto'].apply(lambda x: x[num_regulates:num_regulates+num_obsolete])
            val['part_of_features'] = val['onto'].apply(lambda x: x[num_regulates+num_obsolete:num_regulates+num_obsolete+num_part_of])
            val['is_a_features'] = val['onto'].apply(lambda x: x[num_regulates+num_obsolete+num_part_of:num_regulates+num_obsolete+num_part_of+num_is_a])
            val['part_of_lcs_similarity'] = val['onto'].apply(lambda x: x[num_obsolete+num_part_of+num_is_a:num_obsolete+num_part_of+num_is_a+num_lcs])
        print("Time taken to finish get features: ", time.time() - start_time)

        #features += ['onto']
        #feat_type = np.int


    y_train = train[target].to_numpy().astype(np.float32)
    ids_train = train['rank'].to_numpy().astype(np.int)
    # TODO: discuss about the duplicates!
    #logger.info(train[train.duplicated(['ids'])])
    train_dict = dict(zip(train['rank'], train['ids']))
    logger.info('y_train {}'.format(y_train.shape))

    test['rank'] = test[target].rank(ascending=1, method='first')
    #test['ids'] = test[ids].apply(lambda x: '_'.join(x), axis=1)
    y_test = test[target].to_numpy().astype(np.float32)
    ids_test = test['rank'].to_numpy().astype(np.int)
    test_dict = dict(zip(test['rank'], test['ids']))
    logger.info('y_test {}'.format(y_test.shape))

    start_time = time.time()
    train_k = None
    if labeling.lower().strip() == 'binarize':
        y_train = np.array([1 if i >= 1 else 0 for i in y_train])
        #y_test = np.array([1 if i >= 1 else 0 for i in y_test])
        print ("Number of positive samples in train, test: ", np.sum(y_train))
    elif labeling.lower().strip() == 'above_1_topk':
        number_of_ones = np.sum(np.array([1 if i >= 1 else 0 for i in y_train]))
        train_k = number_of_ones
    elif labeling.lower().strip().startswith('top'):
        k = int(labeling.lower().strip().split('top')[-1])
        topitems = sorted(range(len(y_train)), key=lambda i: y_train[i])[-k:]
        logger.info('y_train top items:{}'.format(','.join([str(y_train[i]) for i in topitems])))
        print ('y_train top items:{}'.format(','.join([str(y_train[i]) for i in topitems])))
        y_train = np.array([1 if i in topitems else 0 for i in range(len(y_train))])
        #topitems = sorted(range(len(y_test)), key=lambda i: y_test[i])[-k:]
        #logger.info('y_test top items:{}'.format(','.join([str(y_test[i]) for i in topitems])))
        #y_test = np.array([1 if i in topitems else 0 for i in range(len(y_test))])
    print("Time taken to finish get features: ", time.time() - start_time)

    if val_size:
        val['rank'] = val[target].rank(ascending=1, method='first')
        #val['ids'] = val[ids].apply(lambda x: '_'.join(x), axis=1)
        y_val = val[target].to_numpy().astype(np.float32)
        ids_val = val['rank'].to_numpy().astype(np.int)
        val_dict = dict(zip(val['rank'], val['ids']))
        logger.info('y_val {}'.format(y_val.shape))
        if labeling.lower().strip() == 'binarize':
            y_val = np.array([1 if i >= 1 else 0 for i in y_val])
        elif labeling.lower().strip().startswith('top'):
            k = int(labeling.lower().strip().split('top')[-1])
            topitems = sorted(range(len(y_val)), key=lambda i: y_val[i])[-k:]
            y_val = np.array([1 if i in topitems else 0 for i in range(len(y_val))])

    start_time = time.time()
    train, test, selected_GO_term_names = go_term_feature_slctn(train, y_train, test, GO, GO_graph)
    print("Time taken to finish get features: ", time.time() - start_time)
    # Calculate statistics
    logger.info('====================================')
    logger.info('y_train Mean: %.3f' % np.mean(y_train))
    logger.info('y_train Variance: %.3f' % np.var(y_train))
    logger.info('y_train Standard Deviation: %.3f' % np.std(y_train))
    logger.info('====================================')
    if val_size:
        logger.info('y_val Mean: %.3f' % np.mean(y_val))
        logger.info('y_val Variance: %.3f' % np.var(y_val))
        logger.info('y_val Standard Deviation: %.3f' % np.std(y_val))
        logger.info('====================================')
    logger.info('y_test Mean: %.3f' % np.mean(y_test))
    logger.info('y_test Variance: %.3f' % np.var(y_test))
    logger.info('y_test Standard Deviation: %.3f' % np.std(y_test))
    logger.info('====================================')

    if val_size: return (ids_train, y_train, train_dict, train) , (ids_val, y_val, val_dict, val), (ids_test, y_test, test_dict, test)
    return (ids_train, y_train, train_dict, train), (ids_test, y_test, test_dict, test), train_k, selected_GO_term_names

def get_top_k_GO_terms(train, test, k=None):
    test_go_graph_fs = None
    if k!=None:
        train_go_fs = train['go_terms_TS_'].apply(lambda x: x[:k])
        if test is not None:
            test_go_graph_fs = test['go_terms_TS_'].apply(lambda x: x[:k])
        train_go_fs = np.array(train_go_fs.values.tolist()).astype(np.float32)
        if test is not None:
            test_go_graph_fs = np.array(test_go_graph_fs.values.tolist()).astype(np.float32)
    else:
        train_go_fs = train['go_terms_TS_']
        if test is not None:
            test_go_graph_fs = test['go_terms_TS_']
        train_go_fs = np.array(train_go_fs.values.tolist()).astype(np.float32)
        if test is not None:
            test_go_graph_fs = np.array(test_go_graph_fs.values.tolist()).astype(np.float32)
        print ("Number of go term features: ", train_go_fs.shape)
    return train_go_fs,test_go_graph_fs

def get_data():
    logger.info('Creating dataset')
    # get the triple fitness
    triple = pd.read_csv('../../../srv/local/work/DeepMutationsData/data/Trigenic data/aao1729_Data_S1.tsv', sep='\t')
    triple = triple.rename(columns={'Query strain ID': 'query', 'Array strain ID': 'array',
                                    'Combined mutant fitness': 'triple_score'})

    triple = triple[['query',  'array',  'triple_score']]
    triple['query'] = triple['query'].str.split('_', expand=True)[0]
    triple['array'] = triple['array'].str.split('_', expand=True)[0]
    triple[['query1','query2']] = triple['query'].str.split('+', expand=True)
    triple.drop(['query'], axis=1, inplace=True)

    DIR = '../../../srv/local/work/DeepMutationsData/data/Pair-wise interaction format'
    frames_single, frames_double = [], []
    for file in os.listdir(DIR):
        if(file.endswith('.txt')): #and  os.path.getsize(os.path.join(DIR, file))< 563662389):
            data = pd.read_csv(os.path.join(DIR, file), sep='\t')
            data = data.rename(columns={'Query Strain ID': 'query', 'Query single mutant fitness (SMF)': 'single_score',
                                        'Array Strain ID': 'array', 'Double mutant fitness': 'double_score'})

            #get the single fitness
            single = data.copy()
            single = single[['query', 'single_score']]
            single['query'] = single['query'].str.split('_', expand=True)[0]
            single.drop_duplicates(subset=['query'], inplace=True)
            single.dropna(inplace=True)
            frames_single.append(single)

            #get the double fitness
            double = data.copy()
            double = double[['query', 'array', 'double_score']]
            double['query'] = double['query'].str.split('_', expand=True)[0]
            double['array'] = double['array'].str.split('_', expand=True)[0]
            double.drop_duplicates(subset=['query', 'array'], inplace=True)
            double.dropna(inplace=True)
            frames_double.append(double)

    #Map single fitness values
    single = pd.concat(frames_single)
    triple = triple.merge(single, left_on=['query1'], right_on=['query'])
    triple.rename(columns={'single_score': 'query1_score'}, inplace=True)
    triple.drop(['query'], axis=1, inplace=True)

    triple = triple.merge(single, left_on=['query2'], right_on=['query'])
    triple.rename(columns={'single_score': 'query2_score'}, inplace=True)
    triple.drop(['query'], axis=1, inplace=True)

    triple = triple.merge(single, left_on=['array'], right_on=['query'])
    triple.rename(columns={'single_score': 'array_score'}, inplace=True)
    triple.drop(['query'], axis=1, inplace=True)

    double = pd.concat(frames_double)
    #Match query1 -> query and array -> array
    triple = pd.merge(triple, double, left_on=['query1', 'array'], right_on=['query', 'array'])
    triple.rename(columns={'double_score': 'query1_array_score'}, inplace=True)
    triple.drop(['query'], axis=1, inplace=True)

    triple = triple.merge(double, left_on=['query2', 'array'], right_on=['query', 'array'])
    triple.rename(columns={'double_score': 'query2_array_score'}, inplace=True)
    triple.drop(['query'], axis=1, inplace=True)

    triple.rename(columns={'array': 'arr'}, inplace=True)
    triple = triple.merge(double, left_on=['query1', 'query2'], right_on=['query', 'array'])
    triple.rename(columns={'double_score': 'query1_query2_score'}, inplace=True)
    triple.drop(['query', 'array'], axis=1, inplace=True)
    triple.drop_duplicates(subset=['arr', 'query1', 'query2'], keep=False, inplace=True)
    logger.info(list(triple.columns.values))

    double = double.merge(single, left_on=['query'], right_on=['query'])
    double.rename(columns={'single_score': 'query_score'}, inplace=True)
    double = double.merge(single, left_on=['array'], right_on=['query'])
    double.rename(columns={'single_score': 'array_score', 'query_x':'query'}, inplace=True)
    double.drop(['query_y'], axis=1, inplace=True)
    double.drop_duplicates(subset=['array', 'query'], keep=False, inplace=True)
    return triple, double


def get_data_large(fname_double_raw):
    logger.info('Creating dataset')
    # get the triple fitness
    triple = pd.read_csv('../../../srv/local/work/DeepMutationsData/data/Trigenic data/aao1729_Data_S1.tsv', sep='\t')
    triple = triple.rename(columns={'Query strain ID': 'query', 'Array strain ID': 'array',
                                    'Combined mutant fitness': 'triple_score'})

    triple = triple[['query',  'array',  'triple_score']]
    triple['query'] = triple['query'].str.split('_', expand=True)[0]
    triple['query3'] = triple['array'].str.split('_', expand=True)[0]
    triple[['query1','query2']] = triple['query'].str.split('+', expand=True)
    triple.drop(['query'], axis=1, inplace=True)
    triple.drop(['array'], axis=1, inplace=True)
    triple["sorted_ids"] = triple[['query1','query2', 'query3']].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    triple.drop(['query1', 'query2', 'query3'], axis=1, inplace=True)
    triple["query1"] = triple['sorted_ids'].apply(lambda x: x.split("_")[0])
    triple["query2"] = triple['sorted_ids'].apply(lambda x: x.split("_")[1])
    triple["query3"] = triple['sorted_ids'].apply(lambda x: x.split("_")[2])
    triple.drop(['sorted_ids'], axis=1, inplace=True)
    print("Triple data group by started...")
    triple = triple.groupby(by=["query1","query2","query3"]).agg("min").reset_index()
    DIR = '../../../srv/local/work/DeepMutationsData/data/Pair-wise interaction format'
    frames_single, frames_double = [], []
    print ("Triple data loaded ready...")
    for file in os.listdir(DIR):
        if(file.endswith('.txt')): #and  os.path.getsize(os.path.join(DIR, file))< 563662389):
            data = pd.read_csv(os.path.join(DIR, file), sep='\t')
            data = data.rename(columns={'Query Strain ID': 'query', 'Query single mutant fitness (SMF)': 'single_score',
                                        'Array Strain ID': 'array', 'Array SMF': 'array_single_score', 'Double mutant fitness': 'double_score'})

            #get the single fitness
            single = data.copy()
            single = single[['query', 'single_score']]
            single['query'] = single['query'].str.split('_', expand=True)[0]
            single2 = data.copy()
            single2 = single2[['array', 'array_single_score']]
            single2['array'] = single2['array'].str.split('_', expand=True)[0]
            single2.rename(columns={'array': 'query'}, inplace=True)
            single2.rename(columns={'array_single_score': 'single_score'}, inplace=True)
            all_singles = [single, single2]
            single = pd.concat(all_singles)
            single = single.groupby(by=["query"]).agg("min").reset_index()
            #single.drop_duplicates(subset=['query'], inplace=True)
            single.dropna(inplace=True)
            frames_single.append(single)

            #get the double fitness
            double = data.copy()
            double = double[['query', 'array', 'double_score']]
            double['query'] = double['query'].str.split('_', expand=True)[0]
            double['array'] = double['array'].str.split('_', expand=True)[0]
            double.dropna(inplace=True)
            frames_double.append(double)
            print ("single and double data loading...")

    #Map single fitness values
    single = pd.concat(frames_single)
    single = single.groupby(by=["query"]).agg("min").reset_index()
    single.dropna(inplace=True)
    double = pd.concat(frames_double)
    double["sorted_ids"] = double[["query", "array"]].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    double.drop(['query', 'array'], axis=1, inplace=True)
    double["query"] = double['sorted_ids'].apply(lambda x: x.split("_")[0])
    double["array"] = double['sorted_ids'].apply(lambda x: x.split("_")[1])
    double.drop(['sorted_ids'], axis=1, inplace=True)
    print ("double dataset group by started...")
    double = double.groupby(by=["query","array"]).agg("min").reset_index()
    double.dropna(inplace=True)
    print ("double data ready....")
    double.to_csv(fname_double_raw, sep='\t')
    triple = triple.merge(single, left_on=['query1'], right_on=['query'])
    triple.rename(columns={'single_score': 'query1_score'}, inplace=True)
    triple.drop(['query'], axis=1, inplace=True)

    triple = triple.merge(single, left_on=['query2'], right_on=['query'])
    triple.rename(columns={'single_score': 'query2_score'}, inplace=True)
    triple.drop(['query'], axis=1, inplace=True)

    triple = triple.merge(single, left_on=['query3'], right_on=['query'])
    triple.rename(columns={'single_score': 'query3_score'}, inplace=True)
    triple.drop(['query'], axis=1, inplace=True)

    print ("single score merging done...")

    #Match query1 -> query and array -> array
    triple = pd.merge(triple, double, left_on=['query1', 'query3'], right_on=['query', 'array'])
    triple.rename(columns={'double_score': 'query1_query3_score'}, inplace=True)
    triple.drop(['query', 'array'], axis=1, inplace=True)

    triple = triple.merge(double, left_on=['query2', 'query3'], right_on=['query', 'array'])
    triple.rename(columns={'double_score': 'query2_query3_score'}, inplace=True)
    triple.drop(['query', 'array'], axis=1, inplace=True)

    #triple.rename(columns={'array': 'arr'}, inplace=True)
    triple = triple.merge(double, left_on=['query1', 'query2'], right_on=['query', 'array'])
    triple.rename(columns={'double_score': 'query1_query2_score'}, inplace=True)
    triple.drop(['query', 'array'], axis=1, inplace=True)

    print ("triple dataset merging done....")
    triple = triple.groupby(by=["query1", "query2", "query3"]).agg("min").reset_index()
    #triple.drop_duplicates(subset=['query1', 'query2', 'query3'], keep = False, inplace=True)
    logger.info(list(triple.columns.values))

    double = double.merge(single, left_on=['query'], right_on=['query'])
    double.rename(columns={'single_score': 'query_score'}, inplace=True)
    double = double.merge(single, left_on=['array'], right_on=['query'])
    double.rename(columns={'single_score': 'array_score', 'query_x':'query'}, inplace=True)
    double.drop(['query_y'], axis=1, inplace=True)
    print ("double dataset merging done....")
    #double.drop_duplicates(subset=['array', 'query'], inplace=True)
    print ("Number of instances: ", triple.shape[0], double.shape[0])
    return triple, double

def training_dataset_analysis():
    triple = pd.read_csv('../../../srv/local/work/DeepMutationsData/data/Trigenic data/aao1729_Data_S1.tsv', sep='\t')
    triple = triple.rename(columns={'Query strain ID': 'query', 'Array strain ID': 'array',
                                    'Combined mutant fitness': 'triple_score'})

    triple = triple[['query', 'array', 'triple_score']]
    triple["query_g2_prefix"] = triple['query'].apply(lambda x: get_prefix(x.split("+")[1]))
    triple["array_prefix"] = triple['array'].apply(lambda x: get_prefix(x))
    print (triple.groupby(['query_g2_prefix']).size())
    print (triple.groupby(['array_prefix']).size())
    triple["query2"] = triple['query'].apply(lambda x: x.split("+")[1])
    query2s = triple.query2.unique().tolist()
    arrays = triple.array.unique().tolist()
    all_unique_genes = {g:1 for g in query2s}
    all_unique_genes2 = {g:1 for g in arrays}
    all_unique_genes.update(all_unique_genes2)
    arrays = [get_prefix(i) for i in arrays]
    query2s = [get_prefix(i) for i in query2s]
    all_unique_genes = list(all_unique_genes.keys())
    all_unique_genes = [get_prefix(i) for i in all_unique_genes]
    print (Counter(arrays))
    print (Counter(query2s))
    print (Counter(all_unique_genes))
    #allq2_categories = triple.groupby(['query_g2_prefix']).size()
    #allarray_categories = triple.groupby(['array_prefix']).size().tolist()
    #print ("allq2_categories: ", allq2_categories)
    #print("allarray_categories: ", allarray_categories)
    DIR = '../../../srv/local/work/DeepMutationsData/data/Pair-wise interaction format'
    frames_single, frames_double = [], []
    alldigenicq_categories = {}
    alldigenicarray_categories = {}
    frames_double = []
    for file in os.listdir(DIR):
        if (file.endswith('.txt')) and (file!="digenic_data_sample.txt"):  # and  os.path.getsize(os.path.join(DIR, file))< 563662389):
            data = pd.read_csv(os.path.join(DIR, file), sep='\t')
            data = data.rename(columns={'Query Strain ID': 'query', 'Query single mutant fitness (SMF)': 'single_score',
                                        'Array Strain ID': 'array', 'Double mutant fitness': 'double_score'})

            data['query_prefix'] = data['query'].apply(lambda x: get_prefix(x))
            data['array_prefix'] = data['array'].apply(lambda x: get_prefix(x))
            #q_categories = data.groupby(['query_prefix']).size().tolist()
            #array_categories = data.groupby(['array_prefix']).size().tolist()
            double = data.copy()
            double = double[['query', 'array', 'double_score']]
            examples = double[double["query"].isin(["YPL240C+YMR186W_y14084"])]
            print (examples)
            frames_double.append(double)
            print ("Dataset name: ", file)
            #print ("Query categories: ", q_categories)
            #print ("Array categories: ", array_categories)
            print (data.groupby(['query_prefix']).size())
            print (data.groupby(['array_prefix']).size())
            data["query2"] = data['query'].apply(lambda x: x.split("+")[1] if (len(x.split("+"))>1) else x)
            #data["query2"] = data['query'].apply(lambda x: x.split("+")[1])
            query2s = data.query2.unique().tolist()
            arrays = data.array.unique().tolist()
            all_unique_genes = {g: 1 for g in query2s}
            all_unique_genes2 = {g: 1 for g in arrays}
            all_unique_genes.update(all_unique_genes2)
            arrays = [get_prefix(i) for i in arrays]
            query2s = [get_prefix(i) for i in query2s]
            all_unique_genes = list(all_unique_genes.keys())
            all_unique_genes = [get_prefix(i) for i in all_unique_genes]
            print(Counter(arrays))
            print(Counter(query2s))
            print(Counter(all_unique_genes))
    frames_double = pd.concat(frames_double)
    print ("All dataset")
    frames_double["query2"] = frames_double['query'].apply(lambda x: x.split("+")[1] if (len(x.split("+"))>1) else x)
    query2s = frames_double.query2.unique().tolist()
    arrays = frames_double.array.unique().tolist()
    all_unique_genes = {g: 1 for g in query2s}
    all_unique_genes2 = {g: 1 for g in arrays}
    all_unique_genes.update(all_unique_genes2)
    arrays = [get_prefix(i) for i in arrays]
    query2s = [get_prefix(i) for i in query2s]
    all_unique_genes = list(all_unique_genes.keys())
    all_unique_genes = [get_prefix(i) for i in all_unique_genes]
    print(Counter(arrays))
    print(Counter(query2s))
    print(Counter(all_unique_genes))

def split_train_test_set(k=5):
    #K for the K-fold split
    ids = ["query1", "query2", "arr"]
    filenames = ["data/Trigenic_smalldata_fold1_", "data/Trigenic_smalldata_fold2_", "data/Trigenic_smalldata_fold3_", "data/Trigenic_smalldata_fold4_", "data/Trigenic_smalldata_fold5_"]
    triple = get_trigenic_data()
    triple['sorted_ids'] = triple[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    X = triple["sorted_ids"].tolist()
    X = np.array(X)
    kf = KFold(n_splits=k, random_state=1, shuffle = False)
    kf.get_n_splits(X)
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        with open(filenames[i]+"train_ids.txt", "w") as outfile:
            X_train = X_train.tolist()
            for train_id in X_train:
                outfile.write(train_id + "\n")
        with open(filenames[i] + "test_ids.txt", "w") as outfile:
            X_test = X_test.tolist()
            for test_id in X_test:
                outfile.write(test_id + "\n")
        i = i +1

def split_train_test_set_stratify(k=5):
    #K for the K-fold split
    ids = ["query1", "query2", "query3"]
    filenames = ["data/Trigenic_largedata_stratified_fold1_", "data/Trigenic_largedata_stratified_fold2_", "data/Trigenic_largedata_stratified_fold3_", "data/Trigenic_largedata_stratified_fold4_", "data/Trigenic_largedata_stratified_fold5_"]
    triple = get_trigenic_data(fname_triple="DeepMutationsData/data/triple_fitness_large.tsv", ids = ids)
    triple['sorted_ids'] = triple[ids].apply(lambda x: "_".join(list(sorted(x))), axis=1)
    top_200 = triple[triple["reverserank"]<=1000]
    bottom_remaining = triple[triple["reverserank"]>1000]
    X_positive = np.array(top_200["sorted_ids"].tolist())
    X_negative = np.array(bottom_remaining["sorted_ids"].tolist())
    #print (X_positive)
    #print (X_negative[:100])
    kf = KFold(n_splits=k, random_state=1, shuffle = True)
    kf.get_n_splits(X_positive)
    kf.get_n_splits(X_negative)
    X_neg_trains, X_neg_tests = [],[]
    for train_index, test_index in kf.split(X_negative):
        X_neg_train, X_neg_test = X_negative[train_index], X_negative[test_index]
        X_neg_trains += [X_neg_train]
        X_neg_tests += [X_neg_test]
    X_pos_trains, X_pos_tests = [], []
    for train_index, test_index in kf.split(X_positive):
        X_pos_train, X_pos_test = X_positive[train_index], X_positive[test_index]
        X_pos_trains += [X_pos_train]
        X_pos_tests += [X_pos_test]

    for i in range(5):
        X_train = X_pos_trains[i].tolist() + X_neg_trains[i].tolist()
        X_test = X_pos_tests[i].tolist() + X_neg_tests[i].tolist()
        random.shuffle(X_train)
        random.shuffle(X_test)
        with open(filenames[i]+"train_ids.txt", "w") as outfile:
            for train_id in X_train:
                outfile.write(train_id + "\n")
        with open(filenames[i] + "test_ids.txt", "w") as outfile:
            for test_id in X_test:
                outfile.write(test_id + "\n")


def get_prefix(x):
    if "_" in x:
        prefix = x.split("_")[1]
        #if re.sub(r'[0-9]+', '', prefix) == "y":
        #    print (x)
        return re.sub(r'[0-9]+', '', prefix)
    else:
        return ""


def write_train_test_datasets(train_set,train_outfilename, selected_GO_terms_names):
    (ids_train, y_train, train_dict, train) = train_set

    if "smalldata" in train_outfilename:
        fitness_features1 = ['query1_score', 'query2_score', 'array_score', 'query1_array_score', 'query2_array_score',
                        'query1_query2_score']
    elif "largedata" in train_outfilename:
        fitness_features1 = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score',
                             'query2_query3_score', 'query1_query2_score']

    fitness_features2 = ['query1_score', 'query2_score', 'query3_score', 'query1_query3_score', 'query2_query3_score','query1_query2_score']
    onto_feature_names = ["regulates_12_intersection", "regulates_13_intersection", "regulates_23_intersection","regulates_123_intersection"]
    onto_feature_names += ["partof_12_intersection", "partof_13_intersection", "partof_23_intersection","partof_123_intersection"]
    onto_feature_names += ["isa_12_intersection", "isa_13_intersection", "isa_23_intersection", "isa_123_intersection"]
    onto_feature_names += ["wusim_12_intersection", "wusim_13_intersection", "wusim_23_intersection","wusim_123_intersection"]
    semsim_feature_names = []
    semsims_sims = ["Wang", "Jiang", "Lin", "Resnik", "Rel", "GOGO"]
    semsim_pairs = ["12","23","13"]
    semsim_onto_parts = ["mf","bp","cc"]
    semsim_combines = ["avg", "bma"]
    for s in semsims_sims:
        for p in semsim_pairs:
            for o in semsim_onto_parts:
                for c in semsim_combines:
                    name = s + p + "_" + o + "_" + c
                    semsim_feature_names += [name]

    top_k = None
    all_feature_names = fitness_features2 + onto_feature_names + [s[1] for s in selected_GO_terms_names] + semsim_feature_names
    #train dataset here
    y_train = y_train.tolist()
    train_sorted_ids = train["sorted_ids"].values.tolist()
    train_features_set = train[fitness_features1].values.tolist()
    train_onto_features_set = train["onto"].values.tolist()
    sim_names = ["Wang", "Jiang", "Lin", "Resnik", "Rel", "GOGO"]
    train_semsim_features_set = {}
    for sim_name in sim_names:
        if sim_name in train.columns:
            train_semsim_features_set[sim_name] = train[sim_name].values.tolist()
    train_go_terms,_ = get_top_k_GO_terms(train, None, k = top_k)
    train_go_terms_feature_set = train_go_terms.tolist()
    print ("Number of GO term features: ", len(train_go_terms_feature_set[0]))
    with open(train_outfilename,"w") as outfile:
        outfile.write("target\tsorted_ids\t")
        for feature_name in all_feature_names:
            outfile.write(feature_name + "\t")
        outfile.write("\n")
        for idx, feature_set in enumerate(train_features_set):
            outfile.write(str(y_train[idx]) + "\t")
            outfile.write(train_sorted_ids[idx]+ "\t")
            for j in train_features_set[idx]:
                outfile.write(str(j) + "\t")
            for j in train_onto_features_set[idx]:
                outfile.write(str(j) + "\t")
            for j in train_go_terms_feature_set[idx]:
                outfile.write(str(j) + "\t")
            for sim_name in sim_names:
                if sim_name in train.columns:
                    for j in train_semsim_features_set[sim_name][idx]:
                        outfile.write(str(j) + "\t")
            outfile.write("\n")

def write_train_test_datasets_main():
    dataset = '../../../srv/local/work/DeepMutationsData/data/triple_fitness.tsv'
    dataset_double = '../../../srv/local/work/DeepMutationsData/data/double_fitness.tsv'
    onto_features = 1
    dir_prefix = "../../../srv/local/work/DeepMutationsData/data/Trigenic_data_with_features/"
    train_outfilename = dir_prefix + "Trigenic_smalldata_3.txt"
    triple_set, selected_GO_terms_names = get_XY_data(dataset, dataset_double, onto_features)
    write_train_test_datasets(triple_set, train_outfilename, selected_GO_terms_names)


if __name__== "__main__":
    #training_dataset_analysis()
    '''
    fname_triple = '../../../srv/local/work/DeepMutationsData/data/triple_fitness_large_2.tsv'
    fname_double = '../../../srv/local/work/DeepMutationsData/data/double_fitness_large_2.tsv'
    fname_double_raw = '../../../srv/local/work/DeepMutationsData/data/double_fitness_raw_large_2.tsv'
    triple,double = get_data_large(fname_double_raw)
    triple.to_csv(fname_triple, sep='\t')
    double.to_csv(fname_double, sep='\t')
    '''
    #split_train_test_set_stratify()
    #write_train_test_datasets_main()
    split_train_test_set_stratify()