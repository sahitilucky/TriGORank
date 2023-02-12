import pandas as pd
import argparse
import copy
import os
import csv
import sys
import pickle
import random
import numpy as np
from format_data import get_XY
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from lamdamart import Lambda
from collections import  defaultdict
from itertools import combinations, permutations, chain
import pyltr
from torch.backends import cudnn
from torch import cuda
import torch
import multiprocessing
from functools import partial
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('seaborn') #or plt.xkcd()
from sklearn.utils import shuffle
import time
pd.set_option('display.max_colwidth', -1)
from utils import ndcg_scoring, parameter_tuning, get_model, write_all_instances, write_relevance_judgments_topranked, precision_recall_k, learning_curves, get_data_for_analysis
from ontology import make_features_adv,load_ontologydata_2
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from ontology import *


def make_dictionary(go_term_dicts):
    pop_gos = {}
    for go_terms in go_term_dicts:
        for go_term in go_terms:
            try:
                pop_gos[go_term] += 1 
            except:
                pop_gos[go_term] = 1 
    return pop_gos

def load_bigontology():
    treeSourceUrl = 'http://chianti.ucsd.edu/~kono/ci/data/collapsed_go.no_IGI.propagated.small_parent_tree'
    # Load the tree data
    #term2genes - gene term in the data to gene_id in GO.
    treeColNames = ['parent', 'child', 'type', 'in_tree']
    tree = pd.read_csv(treeSourceUrl, delimiter='\t', names=treeColNames)
    term2genes = {}
    gene_regulates = {}
    gene_neg_regulates = {}
    gene_pos_regulates = {}
    gene_is_a = {}
    gene_part_of = {}
    for row in tree.itertuples():
        t = row[3]
        if t == 'gene':
            term = row[2]
            terms = []
            if term in term2genes.keys():
                terms = term2genes[term]
            terms.append(row[1])
            term2genes[term] = terms
        elif t == 'regulates':
            term2 = row[2]
            term1 = row[1]
            try:
                gene_regulates[term1] += [term2]
            except:
                gene_regulates[term1] = [term2]                
        elif t == 'negatively_regulates':
            term2 = row[2]
            term1 = row[1]
            try:
                gene_neg_regulates[term1] += [term2]
            except:
                gene_neg_regulates[term1] = [term2] 
            try:
                gene_regulates[term1] += [term2]
            except:
                gene_regulates[term1] = [term2]               
        elif t == 'positively_regulates':
            term2 = row[2]
            term1 = row[1]
            try:
                gene_pos_regulates[term1] += [term2]
            except:
                gene_pos_regulates[term1] = [term2]
            try:
                gene_regulates[term1] += [term2]
            except:
                gene_regulates[term1] = [term2]
        # "is-a" and "part-of" relations from the treesourceurl 
        elif t == 'is_a':
            term2 = row[2]
            term1 = row[1]
            try:
                gene_is_a[term1] += [term2]
            except:
                gene_is_a[term1] = [term2]
        elif t == 'part_of':
            term2 = row[2]
            term1 = row[1]
            try:
                gene_part_of[term1] += [term2]
            except:
                gene_part_of[term1] = [term2]

    for term1 in gene_regulates:
        gene_regulates[term1] = list(set(gene_regulates[term1]))

    for term1 in gene_is_a:
        gene_is_a[term1] = list(set(gene_is_a[term1]))

    for term1 in gene_part_of:
        gene_part_of[term1] = list(set(gene_part_of[term1]))

    return gene_regulates,gene_neg_regulates,gene_pos_regulates,gene_is_a,gene_part_of

def plot_set_length_dist_intersection(triple, relation_name):
    set_len_dist1t = triple[['triple_score', relation_name]].to_numpy(copy=True)
    max_triplet_score = np.max(triple['triple_score'].to_numpy(copy=True))
    avg_triple_score = {}
    for x in set_len_dist1t:
        try:
            avg_triple_score[x[1]] += [x[0]]
        except:
            avg_triple_score[x[1]] = [x[0]]

    for x in avg_triple_score:
        if len(avg_triple_score[x]) != 0:
            avg_triple_score[x] = float(sum(avg_triple_score[x]))/float(len(avg_triple_score[x]))
        else:
            avg_triple_score[x] = 0
    avg_triple_score = sorted(avg_triple_score.items(), key = lambda l:l[0])
    plt.plot([x[0] for x in avg_triple_score], [x[1] for x in avg_triple_score])
    plt.xlabel('Number of intersection with only path 2 and not one')
    plt.ylabel('Average triple fitness score')
    plt.ylim([0.0, max([x[1] for x in avg_triple_score]) ])
    plt.xlim([0.0, max([x[0] for x in avg_triple_score])])
    plt.title(relation_name + ' analysis')
    plt.legend(loc="best")
    plt.savefig(relation_name + "_triple_inter.jpg")
    plt.clf()

def get_discriminative_terms(common_go_terms1t, common_go_terms1b):
    dis_pop_go_terms, dis_unpop_go_terms = {},{}
    pop_gos = make_dictionary(common_go_terms1t)
    unpop_gos = make_dictionary(common_go_terms1b)
    all_intrst_gos = list(set(list(pop_gos.keys()) + list(unpop_gos.keys())))
    for go in all_intrst_gos:
        if go in pop_gos:
            p_go_pop = float(pop_gos[go])/float(len(common_go_terms1t))  
        else:
            p_go_pop = 0
        if go in unpop_gos:
            p_go_unpop = float(unpop_gos[go])/float(len(common_go_terms1b))  
        else:
            p_go_unpop = 0
        p_pop = float(len(common_go_terms1t))/float(len(common_go_terms1b)+len(common_go_terms1t))
        p_pop_go = float(p_go_pop*p_pop)/float((p_go_pop*p_pop) + (p_go_unpop*(1-p_pop)))
        p_unpop_go = float(p_go_unpop*(1-p_pop))/float((p_go_pop*p_pop) + (p_go_unpop*(1-p_pop)))
        dis_pop_go_terms[go] = p_pop_go
        dis_unpop_go_terms[go] = p_unpop_go
    dis_pop_go_terms = sorted(dis_pop_go_terms.items(), key = lambda l:l[1], reverse=True)
    dis_unpop_go_terms = sorted(dis_unpop_go_terms.items(), key = lambda l:l[1], reverse=True)
    return dis_pop_go_terms, dis_unpop_go_terms        
    #Frequency of a go term: P(go in triplet): no.og triplets it is in P(go|+ve) , P(+ve|go) = P(go|+ve)P(+ve)/ (P(go|+ve)P(+ve) + P(go|-ve)P(-ve)), P(go|+ve) = P(go in +ve)



def common_onto_terms(gene_terms, term2genes, gene_idx_features):
    terms = gene_terms.split('_')
    regulates1 = gene_idx_features[terms[0]]["regulates"]
    regulates2 = gene_idx_features[terms[1]]["regulates"]
    regulates3 = gene_idx_features[terms[2]]["regulates"]
    s1 = set(regulates1).intersection(set(regulates2))
    s2 = set(regulates2).intersection(set(regulates3))
    s3 = set(regulates1).intersection(set(regulates3))
    regulates123 = list(s1.intersection(set(regulates3)))

    parentterms1 = list(set([x for gene_value in gene_idx_features[terms[0]]["assoc_goobjs"] for x in gene_value["part_of"]]))
    parentterms2 = list(set([x for gene_value in gene_idx_features[terms[1]]["assoc_goobjs"] for x in gene_value["part_of"]]))
    parentterms3 = list(set([x for gene_value in gene_idx_features[terms[2]]["assoc_goobjs"] for x in gene_value["part_of"]]))
    s1 = set(parentterms1).intersection(set(parentterms2))
    s2 = set(parentterms2).intersection(set(parentterms3))
    s3 = set(parentterms1).intersection(set(parentterms3))
    partof123 = list(s1.intersection(set(parentterms3)))
    
    parentterms1 = list(set([x for gene_value in gene_idx_features[terms[0]]["assoc_goobjs"] for x in gene_value["is_a"]]))
    parentterms2 = list(set([x for gene_value in gene_idx_features[terms[1]]["assoc_goobjs"] for x in gene_value["is_a"]]))
    parentterms3 = list(set([x for gene_value in gene_idx_features[terms[2]]["assoc_goobjs"] for x in gene_value["is_a"]]))
    s1 = set(parentterms1).intersection(set(parentterms2))
    s2 = set(parentterms2).intersection(set(parentterms3))
    s3 = set(parentterms1).intersection(set(parentterms3))
    isa123 = list(s1.intersection(set(parentterms3)))

    return [regulates123, partof123, isa123]

def get_go_features(regulates, relation_name, go, treesourcego):
    regulates_list = []
    count = 0 
    for x in regulates:
        try:
            regulates_list += go[x][relation_name]
            #print ('Coming here: ', x)
        except KeyError:
            try:
                regulates_list += treesourcego[x]
                count += 1
                #print ('Coming here: ', x, count, treesourcego[x])
            except KeyError:
                #print ("No relation present:")
                pass
    return set(regulates_list)

def path2_regulates(gene_terms, term2genes, gene_idx_features, term2gofeatures, go, treesourcego, path_length=2):
    
    terms = gene_terms.split('_')
    regulates1 = gene_idx_features[terms[0]]["regulates"]
    regulates2 = gene_idx_features[terms[1]]["regulates"]
    regulates3 = gene_idx_features[terms[2]]["regulates"]

    regulates11 = get_go_features(regulates1, "regulates", go, treesourcego)
    regulates22 = get_go_features(regulates2, "regulates", go, treesourcego)
    regulates33 = get_go_features(regulates3, "regulates", go, treesourcego)

    #regulates11 = set([y for x in regulates1 for y in go[x]["regulates"]])
    #regulates22 = set([y for x in regulates2 for y in go[x]["regulates"]])
    #regulates33 = set([y for x in regulates3 for y in go[x]["regulates"]])

    regulates1 = dict(list(zip(regulates1, range(len(regulates1)))))
    regulates2 = dict(list(zip(regulates2, range(len(regulates2)))))
    regulates3 = dict(list(zip(regulates3, range(len(regulates3)))))

    regulates11 = regulates11.difference(regulates1)
    regulates22 = regulates22.difference(regulates2)
    regulates33 = regulates33.difference(regulates3)
    
    s1 = set(regulates11).intersection(set(regulates22))
    s2 = set(regulates22).intersection(set(regulates33))
    s3 = set(regulates11).intersection(set(regulates33))
    regulates112233 = list(s1.intersection(set(regulates33)))

    s1 = set(regulates1).intersection(set(regulates2))
    s2 = set(regulates2).intersection(set(regulates3))
    s3 = set(regulates1).intersection(set(regulates3))
    regulates123 = list(s1.intersection(set(regulates3)))

    return (regulates112233, len(regulates112233), regulates123, len(regulates123))

def pasth2_is_a_part_of(gene_terms, term2genes, gene_idx_features, term2gofeatures, go, treesourcego, relation_name, path_length=2):
    #term2genes, term2gofeatures, go =load_ontologydata_2('../../../srv/local/work/DeepMutationsData/data/goslim_yeast.obo')
    terms = gene_terms.split('_')
    regulates1 = list(set([x for gene_value in gene_idx_features[terms[0]]["assoc_goobjs"] for x in gene_value[relation_name]]))
    regulates2 = list(set([x for gene_value in gene_idx_features[terms[1]]["assoc_goobjs"] for x in gene_value[relation_name]]))
    regulates3 = list(set([x for gene_value in gene_idx_features[terms[2]]["assoc_goobjs"] for x in gene_value[relation_name]]))

    treesourcego = {}
    regulates11 = get_go_features(regulates1, relation_name, go, treesourcego)
    regulates22 = get_go_features(regulates2, relation_name, go, treesourcego)
    regulates33 = get_go_features(regulates3, relation_name, go, treesourcego)

    regulates1 = dict(list(zip(regulates1, range(len(regulates1)))))
    regulates2 = dict(list(zip(regulates2, range(len(regulates2)))))
    regulates3 = dict(list(zip(regulates3, range(len(regulates3)))))

    regulates11 = regulates11.difference(regulates1)
    regulates22 = regulates22.difference(regulates2)
    regulates33 = regulates33.difference(regulates3)
    
    s1 = set(regulates11).intersection(set(regulates22))
    s2 = set(regulates22).intersection(set(regulates33))
    s3 = set(regulates11).intersection(set(regulates33))
    regulates112233 = list(s1.intersection(set(regulates33)))

    s1 = set(regulates1).intersection(set(regulates2))
    s2 = set(regulates2).intersection(set(regulates3))
    s3 = set(regulates1).intersection(set(regulates3))
    regulates123 = list(s1.intersection(set(regulates3)))

    return (regulates112233, len(regulates112233), regulates123, len(regulates123))

    #[for p in parentterms1 for x in go[p]["part_of"]]

def get_longer_path_analysis(triple,term2genes, gene_idx_features, top_k = 100):
    term2genes, term2gofeatures, go = load_ontologydata_2('../../../srv/local/work/DeepMutationsData/data/goslim_yeast.obo')
    for go_obj in go:
        relations = ["regulates"]
        for relation in relations:
            new_ps = []
            for parent in go[go_obj][relation]:
                if go.get(parent):
                    new_ps += [parent]
            go[go_obj][relation] = new_ps
    gene_regulates, gene_is_a, gene_part_of = {},{},{}
    #gene_regulates,gene_neg_regulates,gene_pos_regulates,gene_is_a,gene_part_of = load_bigontology()
    #Analysing the Common terms in the intersection and length distribution for higher and lower fitness scores
    triple["path2_regs"],triple["path2_regs_num"],triple["path1_regs"],triple["path1_regs_num"] = zip(*triple['ids'].apply(lambda x: path2_regulates(x, term2genes, gene_idx_features, term2gofeatures, go, gene_regulates, path_length=2)))
    triple["path2_is_a"],triple["path2_is_a_num"],triple["path1_is_a"],triple["path1_is_a_num"] = zip(*triple['ids'].apply(lambda x: pasth2_is_a_part_of(x, term2genes, gene_idx_features, term2gofeatures, go, gene_is_a, "is_a", path_length=2)))
    triple["path2_part_of"],triple["path2_part_of_num"],triple["path1_part_of"],triple["path1_part_of_num"] = zip(*triple['ids'].apply(lambda x: pasth2_is_a_part_of(x, term2genes, gene_idx_features, term2gofeatures, go, gene_part_of, "part_of", path_length=2)))

    bottomterms = triple[triple['rank'] <= top_k]
    topterms = triple[triple['reverserank'] <= top_k]

    print ("Number of genes: " , len(go.keys()))
    print ("Number of gene regulates: ", len(gene_regulates.keys()))
    relation_names = [("regulates intersections w path len 2", "path2_regs"), ("regulates intersections w path len 1", "path1_regs"), ("is a intersections w path len 2", "path2_is_a"), ("is a intersections w path len 1", "path1_is_a"), ("partof intersections w path len 2", "path2_part_of"), ("partof intersections w path len 1", "path1_part_of")]
    for i in range(len(relation_names)):
        print ("Relation: {} term name:{}".format(relation_names[i][0], relation_names[i][1]))
        #Get Common GO terms and their frequency in top and bottom 
        common_go_terms1t = topterms[relation_names[i][1]].to_numpy(copy=True)
        common_go_terms1b = bottomterms[relation_names[i][1]].to_numpy(copy=True)
        pop_gos = make_dictionary(common_go_terms1t)
        pop_gos = sorted(pop_gos.items(), key= lambda l:l[1], reverse=True)
        unpop_gos = make_dictionary(common_go_terms1b)
        unpop_gos = sorted(unpop_gos.items(), key= lambda l:l[1], reverse=True)
        pop_gos = name_go_terms(pop_gos[:100], go)
        unpop_gos = name_go_terms(unpop_gos[:100], go)
        print ("Popular and unpopular GO Terms by their frequency")
        print (pop_gos[:100])
        print (unpop_gos[:100])
        dis_pop_gos, dis_unpop_gos = get_discriminative_terms(common_go_terms1t, common_go_terms1b)
        dis_pop_gos = name_go_terms(dis_pop_gos[:100], go)
        dis_unpop_gos = name_go_terms(dis_unpop_gos[:100], go)
        print ("Popular and unpopular GO Terms by their discriminativeness")
        print (dis_pop_gos)
        print (dis_unpop_gos)
        print ("=============================================")
        #Computing intersection set length distribution
        #plot_set_length_dist_intersection(triple, relation_names[i][1] + "_num")


if __name__== "__main__":
    parser = argparse.ArgumentParser("Genetic triple mutation ltr models")
    parser.add_argument('-dataset', default='../../../srv/local/work/DeepMutationsData/data/triple_fitness.tsv', type=str, help="path to dataset")
    parser.add_argument('-dataset_double', default='../../../srv/local/work/DeepMutationsData/data/double_fitness.tsv', type=str, help="path to dataset")
    #parser.add_argument("-save_dir", type=str, default='onto_exps', help="directory for saving results and models")
    parser.add_argument('-onto_filename', default='../../../srv/local/work/DeepMutationsData/data/goslim_yeast.obo', type=str, help="ontology features filename")
    
    args = parser.parse_args()
    #data_analysis(args.dataset, args.dataset_double, args.onto_filename)
    #gene_regulates,gene_neg_regulates,gene_pos_regulates,gene_is_a,gene_part_of = load_bigontology()
    go = Ontology()
    triple = get_trigenic_data()
    get_longer_path_analysis(triple, go, top_k =100)





