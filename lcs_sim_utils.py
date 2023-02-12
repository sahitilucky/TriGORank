import random
'''
All functions needed to compute LCS features
'''

def calculate_depths(go):
    '''
    All functions needed to compute LCS features
    Parameters
    ----------
    go
    Returns
    -------
    '''
    #remove 2-length cycle
    for go_id in go:
        for p_id in go[go_id]["part_of"]:
            if go_id in go[p_id]["part_of"]:
                if len(go[go_id]["part_of"]) > len(go[p_id]["part_of"]):
                    go[go_id]["part_of"].remove(p_id)
                elif len(go[go_id]["part_of"]) < len(go[p_id]["part_of"]):
                    go[p_id]["part_of"].remove(go_id)    
                else:
                    r = random.random()
                    if r>=0.5: go[go_id]["part_of"].remove(p_id)
                    else: go[p_id]["part_of"].remove(go_id)

    for go_id in go:
        go[go_id]['part_of_depth'] = -1
    for go_id in go:
        #print (go_id, go[go_id]["part_of"])
        go[go_id]['part_of_depth'] = get_depth(go_id, go)
    return go


def wu_palmers_sim(go_id1, go_id2, go, relation_name):
    '''
    Wu palmers similarity.
    Parameters
    ----------
    go
    Returns
    -------
    '''
    if go_id1==go_id2:
        return 1,[go_id2]
    N1, N2 = go[go_id1][relation_name+"_depth"], go[go_id2][relation_name+"_depth"]
    if (N1+N2)==0:
        return (0,[])
    min_value,ccs_list = common_ancester([go_id1], [go_id2], go, relation_name)
    N = min_value
    return (float(2*N)/float(N1+N2),ccs_list)

def common_ancester(go_id1s, go_id2s, go, relation_name):
    parents1 = [p_id for go_id1 in go_id1s for p_id in go[go_id1][relation_name]]
    parents2 = [p_id for go_id2 in go_id2s for p_id in go[go_id2][relation_name]]
    if len(set(go_id2s).intersection(go_id1s))!=0:
        cs = list(set(go_id2s).intersection(go_id1s))
        min_value = min([go[c][relation_name+"_depth"] for c in cs])
        ccs = list(filter(lambda l: go[l][relation_name+"_depth"] == min_value, cs))
        return min_value,ccs
    elif len(set(go_id2s).intersection(parents1))!=0:
        cs = list(set(go_id2s).intersection(parents1))
        min_value = min([go[c][relation_name+"_depth"]  for c in cs])
        ccs = list(filter(lambda l: go[l][relation_name+"_depth"] == min_value, cs))
        return min_value,ccs
    elif len(set(go_id1s).intersection(parents2))!=0:
        cs = list(set(go_id1s).intersection(parents2))
        min_value = min([go[c][relation_name+"_depth"]  for c in cs])
        ccs = list(filter(lambda l: go[l][relation_name+"_depth"] == min_value, cs))        
        return min_value,ccs
    elif parents1 == [] and parents2 == []:
        ccs = []
        return 0,ccs
    elif parents1 == []:
        return common_ancester(go_id1s, parents2, go, relation_name)
    elif parents2 == []:
        return common_ancester(parents1, go_id2s, go, relation_name)
    else:
        return common_ancester(parents1, parents2, go, relation_name)


def get_depth(go_id, go): 
    if go[go_id]["part_of_depth"] != -1:
        #print ("{} Depth returning here 2: {} parents: {}".format(go_id, go[go_id]["part_of_depth"], go[go_id]["part_of"]))
        return go[go_id]["part_of_depth"]
    elif len(go[go_id]["part_of"])==0:
        #print ("{} Depth returning here 1: {} parents: {}".format(go_id, 0, go[go_id]["part_of"]))
        return 1
    else:
        p_depths = []
        p_list = go[go_id]["part_of"][:]
        for p_id_idx in range(len(p_list)):
            #print (p_id_idx)
            p_id = p_list[p_id_idx]
            #print ("{} parents: {}".format(p_id, go[p_id]["part_of"]))
            if go_id in go[p_id]["part_of"]:
                continue
            p_d = get_depth(p_id,go)
            #print ("{} Depth: {} parents: {}".format(p_id, p_d, go[p_id]["part_of"]))
            #print ("{} ".format(p_list))
            if go[p_id]["part_of_depth"] == -1:
                go[p_id]["part_of_depth"] = p_d
            p_depths += [p_d]
        #print ("{} Depth returning here 3: {}".format(go_id, min(p_depths)+1))
        return min(p_depths)+1