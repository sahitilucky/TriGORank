import pandas as pd
pd.options.mode.chained_assignment = None
from collections import defaultdict
import random
import csv
import itertools
import numpy as np
from lcs_sim_utils import *
from gensim.models.keyedvectors import KeyedVectors

class Ontology():
    """
    Contains Ontology objects. 
    """
    def __init__(self, filename):
        self.go = None
        self.treesource_go = None
        self.term2genes = None
        self.term2goobjs = None
        self.read_all_ontology_data(filename)
    
    def read_all_ontology_data(self, filename):
        '''
        Read all ontology data and store, read ontology in URL and store in treesource_go, read ontology in .obo and store in go.
        term2goterms, term2goobjs also read from treesource_go but will filter go terms that are in go.
        Parameters
        ----------
        filename
        Returns
        -------
        '''
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

        treesource_go = {}
        
        treesource_genefeatures = [gene_regulates,gene_neg_regulates,gene_pos_regulates,gene_is_a,gene_part_of]
        relation_names = ["regulates", "neg_regulates", "pos_regulates", "is_a", "part_of"]
        for i in range(5):
            for term1 in treesource_genefeatures[i]:
                treesource_genefeatures[i][term1] = list(set(treesource_genefeatures[i][term1]))
                try:
                    treesource_go[term1][relation_names[i]] = treesource_genefeatures[i][term1]
                except KeyError:
                    treesource_go[term1] = {}
                    treesource_go[term1][relation_names[i]] = treesource_genefeatures[i][term1]
        
        go = get_gene_ontology(filename)

        for key,value in gene_regulates.items():
            gene_info_id = go.get(key)
            if gene_info_id: go[key]['regulates'] += value

        for key,value in gene_neg_regulates.items():
            gene_info_id = go.get(key)
            if gene_info_id: go[key]['regulates'] += value

        for key,value in gene_pos_regulates.items():
            gene_info_id = go.get(key)
            if gene_info_id: go[key]['regulates'] += value

        for key,value in gene_is_a.items():
            gene_info_id = go.get(key)
            if gene_info_id: go[key]['is_a'] += value

        for key,value in gene_part_of.items():
            gene_info_id = go.get(key)
            if gene_info_id: go[key]['part_of'] += value

        for key in term2genes:
            term2genes[key] = list(set(term2genes[key]))

        for go_obj in go:
            relations = ["part_of", "is_a", "regulates"]
            for relation in relations:
                new_ps = []
                for parent in go[go_obj][relation]:
                    if go.get(parent):
                        new_ps += [parent]
                go[go_obj][relation] = new_ps

        for key in term2genes:
            term2genes[key] = [v for v in term2genes[key] if v in go]

        term2gofeatures = defaultdict(list)
        for key,value in term2genes.items():
            gene_ontology = []
            for v in value:
                gene_value = go.get(v)
                if gene_value: gene_ontology += [gene_value]
            term2gofeatures[key] = gene_ontology
        '''
        term2parentgofeatures = defaultdict(list)
        for key,value in term2genes.items():
            gene_part_of = []
            for v in value:
                gene_obj = go.get(v)
                if gene_obj: gene_part_of += gene_obj['part_of']
            term2parentgofeatures[key] = gene_part_of
        '''
        
        self.treesource_go = treesource_go
        self.go = go
        self.term2genes = term2genes
        self.term2goobjs = term2gofeatures
        return


def get_gene_ontology(filename):
    '''
    Reading Gene Ontology from OBO Formatted file
    '''
    go = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'namespace':
                    obj['namespace'] = l[1]
                elif l[0] == 'def':
                    obj['def'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
                elif l[0] == 'relationship':
                    if 'part_of' in l[1]:
                        obj['part_of'].append(l[1].split('part_of ')[1].split(' ! ')[0])
                    if 'has_part' in l[1]:
                        obj['part_of'].append(l[1].split('has_part ')[1].split(' ! ')[0])
                    if 'regulates' in l[1]:
                        obj['regulates'].append(l[1].split('regulates ')[1].split(' ! ')[0])

                    
    if obj is not None:
        go[obj['id']] = obj
    for go_id in list(go.keys()):
        if go[go_id]['is_obsolete']:
            del go[go_id]
    for go_id, val in go.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in go:
                if 'children' not in go[p_id]:
                    go[p_id]['children'] = set()
                go[p_id]['children'].add(go_id)
    return go
'''
def load_ontologydata_2(filename):
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
        elif t == 'positively_regulates':
            term2 = row[2]
            term1 = row[1]
            try:
                gene_pos_regulates[term1] += [term2]
            except:
                gene_pos_regulates[term1] = [term2]
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

    go = get_gene_ontology(filename)
    for key,value in gene_regulates.items():
        gene_info_id = go.get(key)
        if gene_info_id: go[key]['regulates'] += value

    for key,value in gene_neg_regulates.items():
        gene_info_id = go.get(key)
        if gene_info_id: go[key]['regulates'] += value

    for key,value in gene_pos_regulates.items():
        gene_info_id = go.get(key)
        if gene_info_id: go[key]['regulates'] += value

    for key,value in gene_is_a.items():
        gene_info_id = go.get(key)
        if gene_info_id: go[key]['is_a'] += value

    for key,value in gene_part_of.items():
        gene_info_id = go.get(key)
        if gene_info_id: go[key]['part_of'] += value

    for key in term2genes:
        term2genes[key] = list(set(term2genes[key]))

    for go_obj in go:
        relations = ["part_of", "is_a"]
        for relation in relations:
            new_ps = []
            for parent in go[go_obj][relation]:
                if go.get(parent):
                    new_ps += [parent]
            go[go_obj][relation] = new_ps

    term2gofeatures = defaultdict(list)
    for key,value in term2genes.items():
        gene_ontology = []
        for v in value:
            gene_value = go.get(v)
            if gene_value: gene_ontology += [gene_value]
        term2gofeatures[key] = gene_ontology

    return term2genes, term2gofeatures, go
'''


def make_features_adv(GO):
    '''
    Preparing data to compute ontology features  of the model, gene_idx_features object is constructed to use for the features
    preparing data to compute LCS feature using the ontology object, for every gene only few associated go terms are used to compute LCS similarity.
    Input: Ontology object
    Return: gene to selected go terms(dict)
    '''
    #term2genes, term2gofeatures, go = load_ontologydata_2(filename)

    go = calculate_depths(GO.go)

    gene_idx_features =  {}
    for key, values in GO.term2genes.items():
        gene_idx_features[key] = dict()
        gene_idx_list = []
        for v in values:
            gene_value = go.get(v)
            if gene_value: gene_idx_list += [v]
        gene_idx_features[key]["assoc_goterms"] = gene_idx_list

    for key, values in GO.term2genes.items():
        regulate_ids = []
        for v in values:
            try:
                there = GO.treesource_go[v]['regulates']
                regulate_ids += GO.treesource_go[v]['regulates']
            except KeyError:
                pass

        '''    
        for v in values:
            gene_value = go.get(v)
            if gene_value: regulate_ids += gene_value['regulates'] 
        #print ('Regaulate ids:', regulate_ids)
        '''
        gene_idx_features[key]["regulates"] = list(set(regulate_ids))
    
    for key in gene_idx_features:
        values = gene_idx_features[key]["assoc_goterms"]
        min_depth_go_terms = []
        min_depth = 100000000
        for gene_id in values:
            if go[gene_id]["part_of_depth"] < min_depth:
                min_depth = go[gene_id]["part_of_depth"]
                min_depth_go_terms = [gene_id]
            elif go[gene_id]["part_of_depth"] == min_depth:
                min_depth_go_terms += [gene_id]
        gene_idx_features[key]["selected_assoc_goterms"] = min_depth_go_terms
    
    for key in gene_idx_features:
        gene_idx_features[key]["assoc_goobjs"] = GO.term2goobjs[key]
        
    return GO.term2genes, gene_idx_features

def set_intersections(go_terms1, go_terms2, go_terms3):
    '''
    set intersection of go terms of some relation of the three genes -  considers four intersection - 12, 23, 12, 123
    Parameters
    ----------
    go_terms1
    go_terms2
    go_terms3
    Returns
    -------
    '''
    s1 = set(go_terms1).intersection(set(go_terms2))
    s2 = set(go_terms1).intersection(set(go_terms3))
    s3 = set(go_terms2).intersection(set(go_terms3))
    s123 = list(s1.intersection(set(go_terms3)))
    return [len(s1),len(s2),len(s3),len(s123)]

def has_gene_features(gene_terms, gene_idx_features):
    terms = gene_terms.split('_')
    try:
        there = gene_idx_features[terms[0]]
        there = gene_idx_features[terms[1]]
        there = gene_idx_features[terms[2]]
    except KeyError:
        return 0
    return 1

def return_onto_features(gene_terms, term2genes, gene_idx_features, go):
    '''
    Makes Intersection features of reg, part_if, is_a relations , Obsolete features, LCS wu palmaer's similarity features for a triplet of genes
    Input:  Ontology object, triplet, 
    Returns: features (list)
    '''

    terms = gene_terms.split('_')
    features = []
    #regulates1 = list(set([x for gene_value in gene_idx_features[terms[0]]["assoc_goobjs"] for x in gene_value["regulates"]]))
    #regulates2 = list(set([x for gene_value in gene_idx_features[terms[1]]["assoc_goobjs"] for x in gene_value["regulates"]]))
    #regulates3 = list(set([x for gene_value in gene_idx_features[terms[2]]["assoc_goobjs"] for x in gene_value["regulates"]]))
    regulates1 = list(
        set([x for x in gene_idx_features[terms[0]]["regulates"]]))
    regulates2 = list(
        set([x for x in gene_idx_features[terms[1]]["regulates"]]))
    regulates3 = list(
        set([x for x in gene_idx_features[terms[2]]["regulates"]]))

    features += set_intersections(regulates1, regulates2, regulates3)
    
    #features += list(map(lambda l: 1 if (len(gene_idx_features[terms[l]]["assoc_goterms"])==0) else 0, range(3)))
    
    # part-of intersections
    #GO terms intersections
    parentterms1 = list(set([x for gene_value in gene_idx_features[terms[0]]["assoc_goobjs"] for x in gene_value["part_of"]]))
    parentterms2 = list(set([x for gene_value in gene_idx_features[terms[1]]["assoc_goobjs"] for x in gene_value["part_of"]]))
    parentterms3 = list(set([x for gene_value in gene_idx_features[terms[2]]["assoc_goobjs"] for x in gene_value["part_of"]]))
    features += set_intersections(parentterms1, parentterms2, parentterms3)

    #is-a intersections
    #GO terms intersections
    parentterms1 = list(set([x for gene_value in gene_idx_features[terms[0]]["assoc_goobjs"] for x in gene_value["is_a"]]))
    parentterms2 = list(set([x for gene_value in gene_idx_features[terms[1]]["assoc_goobjs"] for x in gene_value["is_a"]]))
    parentterms3 = list(set([x for gene_value in gene_idx_features[terms[2]]["assoc_goobjs"] for x in gene_value["is_a"]]))
    features += set_intersections(parentterms1, parentterms2, parentterms3)
    
    #use selected go terms and lcs based similarity
    wu_palmers_sims1 = []
    wu_palmers_sims2 = []
    wu_palmers_sims3 = []
    ccs_list12 = []
    ccs_list23 = []
    ccs_list13 = []
    for gene_id1 in gene_idx_features[terms[0]]["selected_assoc_goterms"]:
        for gene_id2 in gene_idx_features[terms[1]]["selected_assoc_goterms"]:
            sim, ccs = wu_palmers_sim(gene_id1, gene_id2, go, "part_of")
            wu_palmers_sims1 += [sim]
            ccs_list12 += [ccs]

    for gene_id1 in gene_idx_features[terms[0]]["selected_assoc_goterms"]:        
        for gene_id3 in gene_idx_features[terms[2]]["selected_assoc_goterms"]:
            sim, ccs = wu_palmers_sim(gene_id1, gene_id3, go, "part_of")
            wu_palmers_sims2 += [sim]
            ccs_list13 += [ccs]

    for gene_id2 in gene_idx_features[terms[1]]["selected_assoc_goterms"]:        
        for gene_id3 in gene_idx_features[terms[2]]["selected_assoc_goterms"]:
            sim, ccs = wu_palmers_sim(gene_id2, gene_id3, go, "part_of")
            wu_palmers_sims3 += [sim]
            ccs_list23 += [ccs]

    wu_palmers_sims123 = []  
    ccs_list12 = list(set([c for ccs in ccs_list12 for c in ccs]))
    for ccs in ccs_list12:
        for gene_id3 in gene_idx_features[terms[2]]["selected_assoc_goterms"]:
            sim, triple_ccs = wu_palmers_sim(ccs, gene_id3, go, "part_of")
            wu_palmers_sims123 += [sim]

    wu_palmers_sims231 = []
    ccs_list23 = list(set([c for ccs in ccs_list23 for c in ccs]))
    for ccs in ccs_list23:
        for gene_id1 in gene_idx_features[terms[2]]["selected_assoc_goterms"]:
            sim, triple_ccs = wu_palmers_sim(ccs, gene_id1, go, "part_of")
            wu_palmers_sims231 += [sim]

    wu_palmers_sims132 = []
    ccs_list13 = list(set([c for ccs in ccs_list13 for c in ccs]))
    for ccs in ccs_list13:
        for gene_id2 in gene_idx_features[terms[2]]["selected_assoc_goterms"]:
            sim, triple_ccs = wu_palmers_sim(ccs, gene_id2, go, "part_of")
            wu_palmers_sims132 += [sim]

    if (ccs_list12) == []:
        wu_palmers_sims123 += [0]
    if (ccs_list23) == []:
        wu_palmers_sims231 += [0]
    if (ccs_list13) == []:
        wu_palmers_sims132 += [0]

    avg_sim123 = float(sum(wu_palmers_sims123)) / float(len(wu_palmers_sims123))
    avg_sim123 += float(sum(wu_palmers_sims231))/float(len(wu_palmers_sims231))
    avg_sim123 += float(sum(wu_palmers_sims132))/float(len(wu_palmers_sims132))
    avg_sim123 = float(avg_sim123)/float(3)

    avg_sim1,avg_sim2,avg_sim3 = float(sum(wu_palmers_sims1))/float(len(wu_palmers_sims1)), float(sum(wu_palmers_sims2))/float(len(wu_palmers_sims2)), float(sum(wu_palmers_sims3))/float(len(wu_palmers_sims3))
    features += [avg_sim1,avg_sim2,avg_sim3,avg_sim123]
    
    return features

def get_all_GO_term_pairs(go, fname_triple):
    '''
    writes all GO term pairs, Gene term pairs,... data into output files, data is obtained from the triplets data, gene annotations. 
    Input: all triplet instances, Ontology
    Returns: none
    '''
    
    treeSourceUrl = 'http://chianti.ucsd.edu/~kono/ci/data/collapsed_go.no_IGI.propagated.small_parent_tree'
    # Load the tree data
    treeColNames = ['parent', 'child', 'type', 'in_tree']
    tree = pd.read_csv(treeSourceUrl, delimiter='\t', names=treeColNames)
    term2genes = {}
    for row in tree.itertuples():
        t = row[3]
        if t == 'gene':
            term = row[2]
            terms = []
            if term in term2genes.keys():
                terms = term2genes[term]
            terms.append(row[1])
            term2genes[term] = terms
            continue

    ids = ['arr', 'query1', 'query2']
    triple = pd.read_csv(fname_triple, sep='\t')
    triple.drop(['Unnamed: 0'], axis=1, inplace=True)
    triple.drop_duplicates(subset=ids, keep=False, inplace=True)
    triple.dropna(inplace=True)

    gene_pairs_dict = {}
    gene_pairs1 = triple[['arr','query1']].values.tolist()
    for pair in gene_pairs1:
        pair.sort()
        gene_pairs_dict[(pair[0],pair[1])] = 1
    gene_pairs2 = triple[['arr','query2']].values.tolist()
    for pair in gene_pairs2:
        pair.sort()
        gene_pairs_dict[(pair[0],pair[1])] = 1
    gene_pairs3 = triple[['query1','query2']].values.tolist()
    for pair in gene_pairs3:
        pair.sort()
        gene_pairs_dict[(pair[0],pair[1])] = 1

    print ("Number of gene pairs: ", len(gene_pairs_dict))
    with open("gene_pairs_data/gene_pairs.txt", "w") as outfile:
        for pair in gene_pairs_dict:    
            outfile.write(pair[0] + "\t" + pair[1] + "\n")

    term2genes_pairs = {}
    for pair in gene_pairs_dict:
        term2genes_pairs[pair[0]] = list(set([term for term in term2genes[pair[0]] if term in go]))
        term2genes_pairs[pair[1]] = list(set([term for term in term2genes[pair[1]] if term in go]))

    with open("gene_pairs_data/gene_2_go_terms_small.txt", "w") as outfile:
        term2genes_pairs = term2genes_pairs.items()
        for gene,go_terms_list in term2genes_pairs:    
            outfile.write(gene + "\t")
            for i in range(len(go_terms_list)-1):
                outfile.write(go_terms_list[i] + "\t")
            outfile.write(go_terms_list[len(go_terms_list)-1] + "\n")

    term2genes_pairs = {}
    for pair in gene_pairs_dict:
        term2genes_pairs[pair[0]] = list(set([term for term in term2genes[pair[0]]]))
        term2genes_pairs[pair[1]] = list(set([term for term in term2genes[pair[1]]]))
    with open("gene_pairs_data/gene_2_go_terms.txt", "w") as outfile:
        term2genes_pairs = term2genes_pairs.items()
        for gene,go_terms_list in term2genes_pairs:    
            outfile.write(gene + "\t")
            for i in range(len(go_terms_list)-1):
                outfile.write(go_terms_list[i] + "\t")
            outfile.write(go_terms_list[len(go_terms_list)-1] + "\n")

    i = 0
    all_go_terms_dict = {}
    for pair in gene_pairs_dict:
        go_terms0 = term2genes[pair[0]]
        go_terms1 = term2genes[pair[1]]
        go_terms0 = [term for term in go_terms0 if term in go]
        go_terms1 = [term for term in go_terms1 if term in go]
        go_term_pairs = list(itertools.product(go_terms0, go_terms1))
        for pair in go_term_pairs:
            pair_list = [pair[0], pair[1]]
            pair_list.sort()
            all_go_terms_dict[(pair_list[0],pair_list[1])] =  1
        i += 1
        #if (i%10000)==0:
        #    print (i)
    
    print ("Number of gene term pairs: ", len(all_go_terms_dict))
    with open("gene_pairs_data/go_term_pairs_small.txt", "w") as outfile:
        for pair in all_go_terms_dict:
            outfile.write(pair[0] + "\t" + pair[1] + "\n")
    return

def get_GO_initial_features(GO):
    # TODO: get Initial feature embeddings for each GO term, write to a file for HINE

    model = KeyedVectors.load_word2vec_format('path/to/GoogleNews-vectors-negative300.bin', binary=True)
    go = GO.go
    word_not_found = {}
    for id in go:
        info = [go["name"], go["namespace"], go["def"]]
        for text in info:
            name_vector = []
            name = pre_process_text(text)
            for word in name.split():
                try:
                    name_vector.append(model[word])
                except KeyError:
                    word_not_found[word] = 1
            vector += [name_vector]
        go["name_embedding"] = vector
    print ("Word not present in the model")
    for word in word_not_found:
        print ("Word: ", word)
    write_all_name_embeddings(go)

def write_all_name_embeddings(go):
    with open("initial_features.txt", "w") as outfile:
        noof_term = len(go)
        outfile.write()
def pre_process_text(text):
    return

if __name__== "__main__":
    gene_idx_features = make_features()
    #get_all_GO_term_pairs(go)