#Read GO term similarity between two GO terms
def read_go_term_sims(considerNA = False):
    '''
    read go term similarities from the file.
    Parameters
    ----------
    considerNA - NA is considered as zero if True, otherwise it is ignored totally.

    Returns
    -------

    '''
    go_term_sims = {}
    with open("gene_pairs_data/go_term_pairs_small_sims.txt", "r") as infile:
            for line in infile:
                if considerNA:
                    go_sims = line.strip().split("\t")
                    sims = [float(sim) if (sim != "NA") else 0 for sim in go_sims[2:]]               
                else:
                    go_sims = line.strip().split("\t")
                    sims = [float(sim) if (sim != "NA") else "NA" for sim in go_sims[2:]]
                go_term_sims[(go_sims[0],go_sims[1])] = {}
                go_term_sims[(go_sims[0],go_sims[1])]["Wang"] = [sims[0] , sims[5], sims[10]]
                go_term_sims[(go_sims[0],go_sims[1])]["Jiang"] = [sims[1], sims[6], sims[11]]
                go_term_sims[(go_sims[0],go_sims[1])]["Lin"] = [sims[2], sims[7], sims[12]]
                go_term_sims[(go_sims[0],go_sims[1])]["Resnik"] = [sims[3], sims[8], sims[13]]
                go_term_sims[(go_sims[0],go_sims[1])]["Rel"] = [sims[4], sims[9], sims[14]]

    with open("gene_pairs_data/go_term_pairs_small_GOGO_sim.txt", "r") as infile:
        for line in infile:
            go_sims = line.strip().split(" ")
            try:
                there = go_term_sims[(go_sims[0],go_sims[1])]
            except:
                go_term_sims[(go_sims[0],go_sims[1])] = {}
            if "Error:not_in_the_same_ontology" in line:
                go_term_sims[(go_sims[0],go_sims[1])]["GOGO"] = ["NA" , "NA", "NA"]
            else:
                go_sims = line.strip().split(" ")
                ont = go_sims[2]
                sim = float(go_sims[3])         
                if ont == "BPO":    
                    go_term_sims[(go_sims[0],go_sims[1])]["GOGO"] = ["NA" , sim, "NA"]
                elif ont == "MFO":
                    go_term_sims[(go_sims[0],go_sims[1])]["GOGO"] = [sim, "NA", "NA"]    
                elif ont == "CCO":
                    go_term_sims[(go_sims[0],go_sims[1])]["GOGO"] = ["NA", "NA", sim]
    return go_term_sims

    
def combine_sims_bma_avg(gene_pairs, term2genes, go_term_sims, sim_name):
    '''
    combine go terms similarities between 2 genes using different methods - bma, avg,
    Parameters
    ----------
    gene_pairs
    term2genes
    go_term_sims
    sim_name

    Returns
    -------

    '''
    features = []
    for i in range(3):
        sim_matrix = [[] for j in range(len(term2genes[gene_pairs[0]]))]
        sim_matrix2 = [[] for j in range(len(term2genes[gene_pairs[1]]))]
        for idx,go_term1 in enumerate(term2genes[gene_pairs[0]]):
            for idx2,go_term2 in enumerate(term2genes[gene_pairs[1]]):
                pairs = [go_term1,go_term2]
                pairs.sort()
                #print (pairs[0],pairs[1])
                #try:
                if go_term_sims[(pairs[0],pairs[1])][sim_name][i] != "NA":
                    sim_matrix[idx] += [go_term_sims[(pairs[0],pairs[1])][sim_name][i]] 
                    sim_matrix2[idx2] += [go_term_sims[(pairs[0],pairs[1])][sim_name][i]]
                #except KeyError:
                #    print (pairs[0],pairs[1])
        all_similarities = [s for sim in sim_matrix for s in sim]
        if len(all_similarities)!=0: 
            avg_sim = float(sum(all_similarities))/float(len(all_similarities))
            max_sim = float(max(all_similarities))/float(len(all_similarities))
            bma_sims1 = [max(sim) for sim in sim_matrix if len(sim)!=0]
            bma_sims2 = [max(sim) for sim in sim_matrix2 if len(sim)!=0]
            bma_sim = float(sum(bma_sims1 + bma_sims2))/float(len(bma_sims1 + bma_sims2))
        else:
            bma_sim = 0
            avg_sim = 0
            max_sim = 0 
        features += [avg_sim,bma_sim]
    return features

def combine_sims(gene_pairs, term2genes, go_term_sims):
    '''
    combine go terms similarity between 2 genes using different methods  for all five types of similarity.
    Parameters
    ----------
    gene_pairs
    term2genes
    go_term_sims

    Returns
    -------

    '''
    all_features = []
    sim_name_features = {}
    similarities = ["Wang", "Jiang", "Lin", "Resnik", "Rel", "GOGO"]
    for sim_name in similarities:
        features = combine_sims_bma_avg(gene_pairs, term2genes, go_term_sims, sim_name)
        sim_name_features[sim_name] = features
        all_features += features
        #sim_matrix = np.array(sim_matrix)
        #max_val = np.amax(sim_matrix)
        #avg_val = np.average(sim_matrix)
        #rcmax_val = np.average(np.amax(sim_matrix, axis = 0)) + np.average(np.amax(sim_matrix, axis = 1)) 
        #BMA = np.average( np.concatenate( (np.amax(sim_matrix, axis = 0), np.amax(sim_matrix, axis = 1))))
    return all_features, sim_name_features

def gene_semantic_sims(gene_terms, term2genes, go_term_sims, sim_name):
    '''
    Compute gene semantic similarity between the triplet of genes and return those features
    Parameters
    ----------
    gene_terms: triplet of genes
    term2genes: gene term to GO terms annotation
    go_term_sims: similarity between GO terms
    sim_name: similarity name

    Returns
    -------

    '''
    terms = gene_terms.split('_')
    all_features1, sim_name_features1 = combine_sims([terms[0],terms[1]], term2genes, go_term_sims)
    all_features2, sim_name_features2 = combine_sims([terms[1],terms[2]], term2genes, go_term_sims)
    all_features3, sim_name_features3 = combine_sims([terms[0],terms[2]], term2genes, go_term_sims)

    #gogo_features = combine_sims_bma_avg([terms[0],terms[1]], term2genes, go_term_gogosims)
    #gogo_features = combine_sims_bma_avg([terms[1],terms[2]], term2genes, go_term_gogosims)
    #gogo_features = combine_sims_bma_avg([terms[0],terms[2]], term2genes, go_term_gogosims)
    
    if sim_name == "all":
        features = all_features1 + all_features2 + all_features3 
    else:
        features = sim_name_features1[sim_name] + sim_name_features2[sim_name] + sim_name_features3[sim_name] 
    return features