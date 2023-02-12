import os
from ontology import *
from onto_path_graph import Ontology_graph
from sklearn.model_selection import KFold
from go_term_feature_selection import go_term_feature_slctn_data

def read_data(filename):
    data = []
    with open(filename, "r") as infile:
        i = 0
        for line in infile:
            if i==0:
                i = i + 1
                continue
            else:
                i = i +1
                all_features = line.strip().split("\t")
                target = all_features[0]
                sorted_id = all_features[1]
                feature_values = [float(x) for x in all_features[2:]]
                data += [(sorted_id,float(target),feature_values)]
    data_dict = {}
    for (sorted_id,target,feature_values) in data:
        data_dict[sorted_id] = (target,feature_values)
    return data,data_dict
    #data = sorted(data, key = lambda l:l[1], reverse=True)
    #data = [(d[0],1,d[2]) for d in data[:topk]] + [(d[0],0,d[2]) for d in data[topk:]]


def make_ltr_input(outfilename, data):
    with open(outfilename, "w") as outfile:
        for d in data:
            outfile.write(str(d[1]) + " " + "qid:0" + " ")
            feature_id = 1
            for feature in d[2][:22]:
                outfile.write(str(feature_id) + ":"+ str(feature) + " ")
                feature_id = feature_id + 1
            outfile.write("#triplet=" + d[0])
            outfile.write("\n")

def train_test_split_small_data(train_ids_file, test_ids_file, data_dict):
    print ("total number fo intances: ", len(data_dict))
    train_gene_ids = []
    train_data = []
    with open(train_ids_file, "r") as infile:
        for line in infile:
            gene_ids = line.strip().split("_")
            train_gene_ids += ["_".join(list(sorted(gene_ids)))]
            sorted_id = "_".join(list(sorted(gene_ids)))
            train_data += [(sorted_id,  data_dict[sorted_id][0], data_dict[sorted_id][1])]
    test_gene_ids = []
    test_data = []
    with open(test_ids_file, "r") as infile:
        for line in infile:
            gene_ids = line.strip().split("_")
            test_gene_ids += ["_".join(list(sorted(gene_ids)))]
            sorted_id = "_".join(list(sorted(gene_ids)))
            test_data += [(sorted_id, data_dict[sorted_id][0], data_dict[sorted_id][1])]

    #train = triple[triple["sorted_ids"].isin(train_gene_ids)]
    #test = triple[triple["sorted_ids"].isin(test_gene_ids)]
    #print ("Number of training instances, testing instances: ", train_ids_file, train.shape[0], test_ids_file, test.shape[0])
    #train, test = train_test_split(triple, test_size=test_size, stratify=triple[['query1', 'query2']])
    return train_data,test_data

def make_ltr_data(triplet_data, train_ids_file,test_ids_file, train_out_file,test_out_file):
    data, data_dict = read_data(triplet_data)
    train_data,test_data = train_test_split_small_data(train_ids_file, test_ids_file, data_dict)
    #train_data,test_data, selected_GO_term_data = go_term_feature_slctn_data(train_data,test_data,GO, GO_graph)
    train_data = ltr_with_multiple_labels(train_data)
    test_data = ltr_with_multiple_labels(test_data)
    '''
    train_data = sorted(train_data, key=lambda l: l[1], reverse=True)
    train_data = [(d[0],1,d[2]) for d in train_data[:1000]] + [(d[0],0,d[2]) for d in train_data[1000:]]
    test_data = sorted(test_data, key=lambda l: l[1], reverse=True)
    test_data = [(d[0], 1, d[2]) for d in test_data[:200]] + [(d[0], 0, d[2]) for d in test_data[200:]]
    '''
    '''
    with open("data/selected_go_terms_order.txt","w") as outfile:
        for term in selected_GO_term_data:
            outfile.write(term[0] + "\t" + term[1] + "\n")
    '''
    train_filename = train_out_file
    test_filename = test_out_file
    make_ltr_input(train_filename, train_data)
    make_ltr_input(test_filename, test_data)
    #os.system("java -jar RankLib/RankLib.jar -train " + train_filename +" -test " + test_filename + " -tvs 0.8 -ranker 6 -metric2t NDCG@160 -metric2T NDCG@40 -save ranklibmodels/sample_model_basline_intersct.m")

def ltr_with_multiple_labels(data_list):
    data = sorted(data_list, key=lambda l: l[1], reverse=True)
    maximum_value = data[0][1]
    minimum_value = data[-1][1]
    if minimum_value < 0:
        print (minimum_value)
        print ("target values less than zeros!")
    range_boundaries = np.linspace(0, 1, 10,endpoint=False).tolist()
    range_boundaries = [round(elem, 2) for elem in range_boundaries]
    if maximum_value > 1:
        l = np.arange(1, maximum_value, 0.1).tolist()
        l = [round(elem, 2) for elem in l]
    range_boundaries = range_boundaries + l
    max_relevance_label = len(range_boundaries)-1
    relevance_labels = [round(elem, 2) for elem in list(range(len(range_boundaries)))]
    new_data = []
    for d in data:
        if int(d[1]/0.1) <= 0:
            new_data += [(d[0], relevance_labels[0], d[2])]
        else:
            new_data += [(d[0], relevance_labels[int(d[1]/0.1)], d[2])]
    return new_data

def remove_test_indeces(file0, file1, file2):
    filters = {}
    with open(file1, "r") as infile:
        for line in infile:
            filters[line.strip()] = 1
    print (filters)

    adds = {}
    with open(file0, "r") as infile:
        for line in infile:
            adds[line.strip()] = 1
    print(adds)

    file2_triplets = {}
    with open(file2, "r") as infile:
        for line in infile:
            triplet = line.strip()
            try:
                there = filters[triplet]
            except KeyError:
                file2_triplets[triplet] = 1

    for triplet in adds:
        try:
            there = file2_triplets[triplet]
        except KeyError:
            file2_triplets[triplet] = 1

    with open(file2 + ".filtered" , "w") as outfile:
        for triplet in file2_triplets:
            outfile.write(triplet + "\n")



if __name__== "__main__":
    triplet_data = "../../../srv/local/work/DeepMutationsData/data/Trigenic_data_with_features/Trigenic_largedata.txt"
    train_ids_file = "data/Trigenic_largedata_stratified_fold2_train_ids.txt"
    test_ids_file ="data/Trigenic_largedata_stratified_fold2_test_ids.txt"
    train_out_file = "data/LTR_Trigenic_largedata_stratified_fold2_train_more_labels.txt"
    test_out_file = "data/LTR_Trigenic_largedata_stratified_fold2_test_more_labels.txt"
    make_ltr_data(triplet_data, train_ids_file,test_ids_file, train_out_file,test_out_file)

    train_ids_file = "data/Trigenic_largedata_stratified_fold3_train_ids.txt"
    test_ids_file = "data/Trigenic_largedata_stratified_fold3_test_ids.txt"
    train_out_file = "data/LTR_Trigenic_largedata_stratified_fold3_train_more_labels.txt"
    test_out_file = "data/LTR_Trigenic_largedata_stratified_fold3_test_more_labels.txt"
    make_ltr_data(triplet_data, train_ids_file, test_ids_file, train_out_file, test_out_file)

    train_ids_file = "data/Trigenic_largedata_stratified_fold4_train_ids.txt"
    test_ids_file = "data/Trigenic_largedata_stratified_fold4_test_ids.txt"
    train_out_file = "data/LTR_Trigenic_largedata_stratified_fold4_train_more_labels.txt"
    test_out_file = "data/LTR_Trigenic_largedata_stratified_fold4_test_more_labels.txt"
    make_ltr_data(triplet_data, train_ids_file, test_ids_file, train_out_file, test_out_file)

    train_ids_file = "data/Trigenic_largedata_stratified_fold5_train_ids.txt"
    test_ids_file = "data/Trigenic_largedata_stratified_fold5_test_ids.txt"
    train_out_file = "data/LTR_Trigenic_largedata_stratified_fold5_train_more_labels.txt"
    test_out_file = "data/LTR_Trigenic_largedata_stratified_fold5_test_more_labels.txt"
    make_ltr_data(triplet_data, train_ids_file, test_ids_file, train_out_file, test_out_file)


    '''
    remove_test_indeces("data/Trigenic_smalldata_stratified_fold2_train_ids.txt",
                        "data/Trigenic_smalldata_stratified_fold2_test_ids.txt",
                        "data/Trigenic_largedata_stratified_fold2_train_ids.txt")
    remove_test_indeces("data/Trigenic_smalldata_stratified_fold2_test_ids.txt", "data/Trigenic_smalldata_stratified_fold2_train_ids.txt", "data/Trigenic_largedata_stratified_fold2_test_ids.txt")

    remove_test_indeces("data/Trigenic_smalldata_stratified_fold3_train_ids.txt",
                        "data/Trigenic_smalldata_stratified_fold3_test_ids.txt",
                        "data/Trigenic_largedata_stratified_fold3_train_ids.txt")
    remove_test_indeces("data/Trigenic_smalldata_stratified_fold3_test_ids.txt",
                        "data/Trigenic_smalldata_stratified_fold3_train_ids.txt",
                        "data/Trigenic_largedata_stratified_fold3_test_ids.txt")

    remove_test_indeces("data/Trigenic_smalldata_stratified_fold4_train_ids.txt",
                        "data/Trigenic_smalldata_stratified_fold4_test_ids.txt",
                        "data/Trigenic_largedata_stratified_fold4_train_ids.txt")
    remove_test_indeces("data/Trigenic_smalldata_stratified_fold4_test_ids.txt",
                        "data/Trigenic_smalldata_stratified_fold4_train_ids.txt",
                        "data/Trigenic_largedata_stratified_fold4_test_ids.txt")

    remove_test_indeces("data/Trigenic_smalldata_stratified_fold5_train_ids.txt",
                        "data/Trigenic_smalldata_stratified_fold5_test_ids.txt",
                        "data/Trigenic_largedata_stratified_fold5_train_ids.txt")
    remove_test_indeces("data/Trigenic_smalldata_stratified_fold5_test_ids.txt",
                        "data/Trigenic_smalldata_stratified_fold5_train_ids.txt",
                        "data/Trigenic_largedata_stratified_fold5_test_ids.txt")
    '''

