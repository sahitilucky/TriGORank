from ranklib_ltr import *

triplet_data_small = "../../../srv/local/work/DeepMutationsData/data/Trigenic_data_with_features/Trigenic_smalldata.txt"
triplet_data_large = "../../../srv/local/work/DeepMutationsData/data/Trigenic_data_with_features/Trigenic_largedata.txt"

data, data_dict = read_data(triplet_data_small)
data_large, data_dict_large = read_data(triplet_data_large)

for triplet in data_dict:
    triplet_score = data_dict[triplet][0]
    triplet_score_large = data_dict_large[triplet][0]
    if triplet_score != triplet_score_large:
        print ("Score not match", triplet, triplet_score, triplet_score_large)
    else:
        print ("match", triplet, triplet_score, triplet_score_large)
        print (data_dict[triplet][1], data_dict_large[triplet][1])