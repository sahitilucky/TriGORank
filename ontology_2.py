import pandas as pd
pd.options.mode.chained_assignment = None
from collections import defaultdict


def get_gene_ontology(filename):
    # Reading Gene Ontology from OBO Formatted file
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
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
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

def load_ontologydata(filename):
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
    a = get_gene_ontology(filename)
    gene_ontology_parentfeatures = defaultdict(list)
    for key,value in term2genes.items():
        gene_ontology = []
        for v in value:
            gene_values = a.get(v)
            if gene_values: gene_ontology.append(gene_values)
        gene_ontology_parentfeatures[key] = gene_ontology
    return term2genes, gene_ontology_parentfeatures


def make_features(filename='../../../Data/work/DeepMutationsData/data/goslim_yeast.obo'):
    term2genes, gene_ontology_parentfeatures = load_ontologydata(filename)

    #make index
    all_terms = []
    for values in term2genes.values():
        for value in values:
            all_terms.append(value)
    all_terms = list(set(all_terms))
    all_terms = {k: v for v, k in enumerate(all_terms)}
    print('Number of unique ids:', len(all_terms))
    #make feature indices
    gene_idx_features =  defaultdict(list)
    for key, values in term2genes.items():
        gene_idx_list = []
        for value in values:
            gene_idx_list.append(all_terms[value])
        gene_idx_features[key] = sorted(gene_idx_list)
    return gene_idx_features

def return_list_genes(ids, gene_idx_features):
    ids = ids.split('_')
    unique_geneids = []
    for id in ids:
        unique_geneids.extend(gene_idx_features[id])
    unique_geneids = sorted(list(set(unique_geneids)))
    return unique_geneids

if __name__== "__main__":
    gene_idx_features = make_features()