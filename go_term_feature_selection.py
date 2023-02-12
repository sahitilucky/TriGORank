#To select distinctive GO terms and add then as features.
from onto_path_graph import Ontology_graph
from ontology import *
from utils import *
from sklearn.feature_selection import SelectKBest, chi2
def bayes_go_term_feature_selection():
    GO_graph = Ontology_graph()
    triplets = get_trigenic_data()
    '''
    input is the top, bottom statistics of the go terms in, select based on P(+ve|GO_term), P(-ve|GO_term), chi square test or something
    Returns
    -------
    list of GO terms to be added as features to the model
    '''

def triplet_GO_graph_features(gene_terms, GO_graph, go_term_ids):
    nodes_frequency = triplet_go_frequency(GO_graph, gene_terms)
    feature_list = []
    for go_term in go_term_ids:
        try:
            feature_list += [nodes_frequency[go_term]]
        except KeyError:
            feature_list += [float(0)]
    return feature_list

def triplet_go_frequency(GO_graph, triplet):
    total_pairs = GO_graph.triplet_total_pairs[triplet]
    nodes_frequency = {}
    all_paths = 0
    for (x, y) in total_pairs:
        paths = GO_graph.all_paths_terms[(x, y)]
        all_paths += len(paths)
        for path in paths:
            for node in path:
                try:
                    nodes_frequency[node] += 1
                except KeyError:
                    nodes_frequency[node] = 1
    for node in nodes_frequency:
        nodes_frequency[node] = float(nodes_frequency[node])/float(all_paths)
    return nodes_frequency

def go_term_feature_slctn(train, y_train, test, GO, GO_graph):
    go_term_ids = GO_graph.nodes_list
    print ("Number of terms in GO graph" , len(go_term_ids), type(go_term_ids))
    train["go_terms_TS_"] = train['ids'].apply(lambda x: triplet_GO_graph_features(x, GO_graph, go_term_ids))
    if test is not None:
        test["go_terms_TS_"] = test['ids'].apply(lambda x: triplet_GO_graph_features(x, GO_graph, go_term_ids))
    X_train =  np.array(train["go_terms_TS_"].values.tolist()).astype(np.float32)
    y_train = y_train.astype('int')
    print (X_train.shape, y_train.shape)
    total_go_term_features = len(go_term_ids)
    print ("Number of go terms and X_train shape: ", len(go_term_ids), X_train.shape)
    selectkboost = SelectKBest(chi2, k= total_go_term_features).fit(X_train, y_train)
    features_scores = selectkboost.scores_.tolist()
    index = 0
    features_scores_list = []
    print ("before feature scores:", len(features_scores))
    for f in features_scores:
        #print (f, type(f))
        if math.isnan(f):
            pass
        else:
            features_scores_list += [(float(f),index)]
        index += 1
    print ("after feature scores:", len(features_scores_list))
    features_scores_list = sorted(features_scores_list, key=lambda l:l[0], reverse = True)
    #print (features_scores_list)
    selected_indices = [index[1] for index in features_scores_list]
    '''
    selected_indices = sorted(list(zip(features_scores, range(len(features_scores)))), key=lambda l:l[0], reverse=True)
    #selected_indices = [index[1] for index in selected_indices]
    selected_indices2 = selectkboost.get_support(indices = True)
    selected_indices2 = selected_indices2.tolist()
    print (selected_indices)
    print (selected_indices2)
    selectkboost = SelectKBest(chi2, k=10).fit(X_train, y_train)
    selected_indices2 = selectkboost.get_support(indices=True)
    selected_indices2 = selected_indices2.tolist()
    print(selected_indices2)
    '''
    selected_go_terms = [go_term_ids[i] for i in selected_indices]
    train['go_terms_TS_'] = train['go_terms_TS_'].apply(lambda x: [x[y] for y in selected_indices])
    if test is not None:
        test['go_terms_TS_'] = test['go_terms_TS_'].apply(lambda x: [x[y] for y in selected_indices])
    #print ("SELECTED GO TERMS FROM TRIPLET SUBGRAPHS: ", selected_indices)
    #print (selected_go_terms)
    #print ("SELECTED GO TERM NAMES: " , name_go_terms_list(selected_go_terms, GO.go))
    print ("no of go terms: ", len(selected_go_terms))
    go_term_names = name_go_terms_list(selected_go_terms, GO.go)
    print ("GO term names: ", len(go_term_names))
    return train, test, name_go_terms_list(selected_go_terms, GO.go)

def go_term_feature_slctn_data(train_data, test_data, GO, GO_graph):
    go_term_ids = GO_graph.nodes_list
    print (go_term_ids, type(go_term_ids))
    X_train = []
    y_train = []
    print(len(train_data[0][2]))
    for (sorted_id,target,feature_values) in train_data:
        go_term_features = feature_values[22:]
        X_train += [go_term_features]
        y_train += [target]
    X_train = np.array(X_train)
    X_train = X_train.astype(np.float32)
    y_train = np.array(y_train)
    y_train = y_train.astype(np.float32)
    total_go_term_features = 156 #len(go_term_ids)
    print (X_train.shape, y_train.shape)
    selectkboost = SelectKBest(chi2, k=total_go_term_features).fit(X_train, y_train)
    features_scores = selectkboost.scores_.tolist()
    index = 0
    features_scores_list = []
    for f in features_scores:
        print (f, type(f))
        if math.isnan(f):
            pass
        else:
            features_scores_list += [(float(f),index)]
        index += 1
    print (features_scores_list)
    features_scores_list = sorted(features_scores_list, key=lambda l:l[0], reverse = True)
    print (features_scores_list)
    selected_indices = [index[1] for index in features_scores_list]
    selected_go_terms = [go_term_ids[i] for i in selected_indices]

    train_data_new = []
    for (sorted_id, target, feature_values) in train_data:
        go_features = feature_values[22:]
        go_features = [go_features[y] for y in selected_indices]
        train_data_new += [(sorted_id, target, feature_values[:22] + go_features)]

    test_data_new = []
    for (sorted_id, target, feature_values) in test_data:
        go_features = feature_values[22:]
        go_features = [go_features[y] for y in selected_indices]
        test_data_new += [(sorted_id, target, feature_values[:22] + go_features)]
    print ("SELECTED GO TERMS FROM TRIPLET SUBGRAPHS: ", selected_indices)
    print (selected_go_terms)
    print ("SELECTED GO TERM NAMES: " , name_go_terms_list(selected_go_terms, GO.go))
    return train_data_new, test_data_new, name_go_terms_list(selected_go_terms, GO.go)



