import pandas as pd
import os
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
from ontology_2 import  make_features, return_list_genes
from tensorflow.keras.preprocessing.sequence import pad_sequences
#http://science.sciencemag.org/content/suppl/2018/04/18/360.6386.eaao1729.DC1
#http://boonelab.ccbr.utoronto.ca/supplement/costanzo2016/

logger = logging.getLogger(__name__)


def get_iter(ids, X, y, device, batch_size=64, shuffle=False, precision=torch.float32):
    dataset = TensorDataset(torch.from_numpy(ids), torch.from_numpy(X).to(dtype=precision, device=device), torch.from_numpy(y).to(dtype=precision, device=device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_XY(fname_triple, fname_double, val_size=None, test_size=0.2, labeling='', onto_features=False, maxlen=100):
    features = ['query1_score', 'query2_score', 'array_score', 'query1_array_score', 'query2_array_score', 'query1_query2_score']
    target = 'triple_score'
    ids = ['arr', 'query1', 'query2']
    feat_type = np.float32

    if not os.path.isfile(fname_triple) or not os.path.isfile(fname_double):
        triple, double = get_data()
        triple.to_csv(fname_triple, sep='\t')
        double.to_csv(fname_double, sep='\t')

    triple = pd.read_csv(fname_triple, sep='\t')
    triple.drop(['Unnamed: 0'], axis=1, inplace=True)
    triple.drop_duplicates(subset=ids, keep=False, inplace=True)
    triple.dropna(inplace=True)
    triple = shuffle(triple)
    triple['ids'] = triple[['query1', 'query2']].apply(lambda x: '_'.join(x), axis=1)

    # Calculate statistics
    logger.info('Mean: %.3f' % np.mean(triple['triple_score']))
    logger.info('Variance: %.3f' % np.var(triple['triple_score']))
    logger.info('Standard Deviation: %.3f' % np.std(triple['triple_score']))
    logger.info('====================================')

    if labeling.lower().strip() == 'transfer':
        test = triple[triple['triple_score'] >= 1]
        train = triple[triple['triple_score'] < 1]
    else:
        train, test = train_test_split(triple, test_size=test_size, stratify=triple['ids'])
        #train, test = train_test_split(triple, test_size=test_size, stratify=triple[['query1', 'query2']])
    if val_size: train, val = train_test_split(train, test_size=val_size, stratify=train['ids'])

    #check if we split the data correctly
    #for index, row in test.iterrows():
    #    if (row['query1'] in train and row['query2'] in train and row['arr'] in train): print('Duplicate!', row)
    # Split the data into training/testing sets

    #make ranks
    train['rank'] = train[target].rank(ascending=1, method='first')
    train['ids'] = train[ids].apply(lambda x: '_'.join(x), axis=1)

    if onto_features:
        gene_idx_features = make_features()
        train['onto'] = train['ids'].apply(lambda x: return_list_genes(x,gene_idx_features))
        train['onto'] = pad_sequences(train['onto'], maxlen=maxlen, padding='post', value=0).tolist()
        test['onto'] = test['ids'].apply(lambda x: return_list_genes(x, gene_idx_features))
        test['onto'] = pad_sequences(test['onto'], maxlen=maxlen, padding='post', value=0).tolist()
        if val_size:
            val['onto'] = val['ids'].apply(lambda x: return_list_genes(x, gene_idx_features))
            val['onto'] = pad_sequences(val['onto'], maxlen=maxlen, padding='post', value=0).tolist()
        features = 'onto'
        feat_type = np.int

    X_train = np.array(train[features].values.tolist()).astype(feat_type)
    y_train = train[target].to_numpy().astype(np.float32)
    ids_train = train['rank'].to_numpy().astype(np.int)
    # TODO: discuss about the duplicates!
    #logger.info(train[train.duplicated(['ids'])])
    train_dict = dict(zip(train['rank'], train['ids']))
    logger.info('X_train {}, y_train {}'.format(X_train.shape, y_train.shape))

    test['rank'] = test[target].rank(ascending=1, method='first')
    test['ids'] = test[ids].apply(lambda x: '_'.join(x), axis=1)
    X_test = np.array(test[features].values.tolist()).astype(feat_type)
    y_test = test[target].to_numpy().astype(np.float32)
    ids_test = test['rank'].to_numpy().astype(np.int)
    test_dict = dict(zip(test['rank'], test['ids']))
    logger.info('X_test {}, y_test {}'.format(X_test.shape, y_test.shape))

    if labeling.lower().strip() == 'binarize':
        y_train = np.array([1 if i >= 1 else 0 for i in y_train])
        y_test = np.array([1 if i >= 1 else 0 for i in y_test])
    elif labeling.lower().strip().startswith('top'):
        k = int(labeling.lower().strip().split('top')[-1])
        topitems = sorted(range(len(y_train)), key=lambda i: y_train[i])[-k:]
        logger.info('y_train top items:{}'.format(','.join([str(y_train[i]) for i in topitems])))
        y_train = np.array([1 if i in topitems else 0 for i in range(len(y_train))])
        #topitems = sorted(range(len(y_test)), key=lambda i: y_test[i])[-k:]
        #logger.info('y_test top items:{}'.format(','.join([str(y_test[i]) for i in topitems])))
        #y_test = np.array([1 if i in topitems else 0 for i in range(len(y_test))])

    if val_size:
        val['rank'] = val[target].rank(ascending=1, method='first')
        val['ids'] = val[ids].apply(lambda x: '_'.join(x), axis=1)
        X_val = np.array(val[features].values.tolist()).astype(feat_type)
        y_val = val[target].to_numpy().astype(np.float32)
        ids_val = val['rank'].to_numpy().astype(np.int)
        val_dict = dict(zip(val['rank'], val['ids']))
        logger.info('X_val {}, y_val {}'.format(X_val.shape, y_val.shape))
        if labeling.lower().strip() == 'binarize':
            y_val = np.array([1 if i >= 1 else 0 for i in y_val])
        elif labeling.lower().strip().startswith('top'):
            k = int(labeling.lower().strip().split('top')[-1])
            topitems = sorted(range(len(y_val)), key=lambda i: y_val[i])[-k:]
            y_val = np.array([1 if i in topitems else 0 for i in range(len(y_val))])

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

    if val_size: return (ids_train, X_train, y_train, train_dict), (ids_val, X_val, y_val, val_dict), (ids_test, X_test, y_test, test_dict)
    return (ids_train, X_train, y_train, train_dict), (ids_test, X_test, y_test, test_dict)


def get_data():
    logger.info('Creating dataset')
    # get the triple fitness
    triple = pd.read_csv('data/Trigenic data/aao1729_Data_S1.tsv', sep='\t')
    triple = triple.rename(columns={'Query strain ID': 'query', 'Array strain ID': 'array',
                                    'Combined mutant fitness': 'triple_score'})

    triple = triple[['query',  'array',  'triple_score']]
    triple['query'] = triple['query'].str.split('_', expand=True)[0]
    triple['array'] = triple['array'].str.split('_', expand=True)[0]
    triple[['query1','query2']] = triple['query'].str.split('+', expand=True)
    triple.drop(['query'], axis=1, inplace=True)

    DIR = 'data/Pair-wise interaction format'
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
