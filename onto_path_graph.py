import pandas as pd
import argparse
import copy
import os
import csv
import sys
import pickle
import random
import numpy as np
from collections import  defaultdict
from itertools import combinations, permutations, chain
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
from utils import ndcg_scoring, parameter_tuning, get_model, precision_recall_k, learning_curves,get_trigenic_data
from ontology import * #Ontology, set_intersections
from tensorflow.keras.preprocessing.sequence import pad_sequences
import networkx as nx 
import matplotlib.pyplot as plt 
import itertools

def name_go_terms_dict(go_dict, go):
    pop_go_terms = []
    for go_term in go_list:
        try:
            pop_go_terms += [((go_term, go[go_term]["name"]), go_dict[go_term])]
        except KeyError:
            pop_go_terms += [((go_term, "NA"), go_list[go_term])] 
    return dict(pop_go_terms)


class find_all_paths():
    def __init__(self, G):
        self.G = G
        self.All_paths = []

    '''A recursive function to print all paths from 'u' to 'd'. 
    visited[] keeps track of vertices in current path. 
    path[] stores actual vertices and path_index is current 
    index in path[]'''
    def getAllPathsUtil(self, u, d, visited, path):
  
        # Mark the current node as visited and store in path 
        visited[u]= True
        path.append(u) 
  
        # If current vertex is same as destination, then print 
        # current path[] 
        if u == d: 
            self.All_paths += [path[:]] 
        else: 
            # If current vertex is not destination 
            # Recur for all the vertices adjacent to this vertex
            neighbours = self.G.neighbors(u) 
            for i in neighbours: 
                if visited[i]== False: 
                    self.getAllPathsUtil(i, d, visited, path) 
                      
        # Remove current vertex from path[] and mark it as unvisited 
        path.pop() 
        visited[u]= False
   
   
    # Prints all paths from 's' to 'd' 
    def getAllPaths(self, s, d): 
  
        # Mark all the vertices as not visited 
        visited = {n:False for n in self.G.nodes()}
        self.All_paths = []
        # Create an array to store paths 
        path = [] 
  
        # Call the recursive helper function to print all paths 
        self.getAllPathsUtil(s, d, visited, path)

        return self.All_paths

    def get_shortest_paths(self,paths):
        paths = [(path,len(path)) for path in paths]
        paths = sorted(paths, key = lambda l :l[1], reverse=False)
        if len(paths)==0:
            return [path[0] for path in paths]
        shortest_length = paths[0][1]
        shortest_paths = []
        for path in paths:
            if (path[1]==shortest_length):
                shortest_paths += [path[0]]
        return shortest_paths

class Path():
    def __init__(self):
        self.source = None
        self.dest = None
        self.undirectedpath = None
        self.directedpath = None

class Ontology_graph():
    def __init__(self, go):
        self.undirectedG, self.multidigraphG = self.create_graph(go)
        self.triplet_subgraph = {}
        self.triplet_multigraph = {}
        self.all_paths_terms, self.all_directedpaths_terms = {}, {}
        self.triplet_total_pairs = {}

    def write_graph_network_HINE(self):
        something = self.triplet_multigraph.edges()

    def create_graph(self, go):
        '''
        Create a 1)undirected and 2)multi edge type directed graph for the gene ontology.
        Parameters
        ----------
        go
        Returns
        -------
        '''
        # Creating undirected Graph and Multi graph from the ontology
        print("Creating graph...")
        for go_obj in go:
            relations = ["regulates"]
            for relation in relations:
                new_ps = []
                for parent in go[go_obj][relation]:
                    if go.get(parent):
                        new_ps += [parent]
                go[go_obj][relation] = new_ps

        added_multi_edges = {}
        undirectedG = nx.Graph()
        multidigraphG = nx.MultiDiGraph()
        for go_term in go:
            relations = ["reg", "is_a", "pt_of"]
            for relation_name in relations:
                if relation_name == 'reg':
                    regulates_list = go[go_term]["regulates"]
                elif relation_name == 'pt_of':
                    regulates_list = go[go_term]["part_of"]
                else:
                    regulates_list = go[go_term][relation_name]
                for go_b in regulates_list:
                    undirectedG.add_edge(go_term, go_b)
                    try:
                        there = added_multi_edges[(go_term, go_b, relation_name)]
                    except KeyError:
                        multidigraphG.add_edge(go_term, go_b, relation=relation_name)
                        added_multi_edges[(go_term, go_b, relation_name)] = 1
                    try:
                        there = added_multi_edges[(go_term, go_b, relation_name + "_rv")]
                    except KeyError:
                        multidigraphG.add_edge(go_b, go_term, relation=relation_name + "_rv")
                        added_multi_edges[(go_term, go_b, relation_name + "_rv")] = 1
        node_labels = {}
        count = 0
        for node in multidigraphG.nodes():
            node_labels[node] = count
            count += 1
        nx.draw_networkx(G=multidigraphG, labels=node_labels)
        plt.savefig("graphs/Total_multigraph" + ".png")
        plt.clf()
        print("Graph done....")
        print("Graph details: no.of nodes {}\n no.of edges:{}\n".format(undirectedG.__len__(), undirectedG.size()))
        self.nodes_list = list(undirectedG.nodes)
        self.edges_list = list(undirectedG.edges)
        self.nodes_list.sort()
        return undirectedG, multidigraphG

    def get_triplet_go_terms(self, term2genes, triple):
        term1, term2, term3 = triple.split('_')[0], triple.split('_')[1], triple.split('_')[2]
        go_terms1 = term2genes[term1]
        go_terms2 = term2genes[term2]
        go_terms3 = term2genes[term3]
        #print("Number of go terms: ", len((go_terms1)), len((go_terms2)), len((go_terms3)))
        #print("Number of set_intersections terms: ", set_intersections(go_terms1, go_terms2, go_terms3))
        go_term_pairs1 = list(itertools.product(go_terms1, go_terms2))
        go_term_pairs1 = [(x, y) for (x, y) in go_term_pairs1 if x != y]
        go_term_pairs2 = list(itertools.product(go_terms2, go_terms3))
        go_term_pairs2 = [(x, y) for (x, y) in go_term_pairs2 if x != y]
        go_term_pairs3 = list(itertools.product(go_terms1, go_terms3))
        go_term_pairs3 = [(x, y) for (x, y) in go_term_pairs3 if x != y]
        go_term_pairs = [go_term_pairs1, go_term_pairs2, go_term_pairs3]
        total_pairs = {}
        for i in range(len(go_term_pairs)):
            for (x, y) in go_term_pairs[i]:
                if (y, x)    in total_pairs:
                    pass
                else:
                    total_pairs[(x, y)] = 1
        return total_pairs, go_terms1, go_terms2, go_terms3

    def make_triplet_graphs(self, GO, triplets):
        '''
        Creating a subgraph for a triplet using the total graph, make a undirected and directed multi edge graph.
        Find all undirected and directed paths between all pairs of GO terms in the triplet.
        Parameters
        ----------
        triplets

        Returns
        -------

        '''
        term2genes = GO.term2genes
        undirectedG, multidigraphG = self.undirectedG, self.multidigraphG
        # triplets = np.concatenate((top_triplets,bottom_triplets), axis=0)

        #all_paths_terms = {}
        #all_directedpaths_terms = {}
        #triplet_subgraph = {}
        #triplet_multigraph = {}
        #triplet_total_pairs = {}
        for triple in triplets:
            '''Creating Triplet subgraph, 
            Step1: Get all GO terms(nodes) of all genes in triplet 
            Step2: get all pairs of nodes 
            Step3: Find all paths between all pairs of go terms
            Step4: Add edges of all paths to a new graph
            Step5: Add edges from GO to goterms, this will be new subgraph'''
            if triple in self.triplet_subgraph:
                continue
            #print("Triplet: ", triple)
            term1, term2, term3 = triple.split('_')[0], triple.split('_')[1], triple.split('_')[2]
            '''
            go_terms1 = term2genes[term1]
            go_terms2 = term2genes[term2]
            go_terms3 = term2genes[term3]
            print("Number of go terms: ", len((go_terms1)), len((go_terms2)), len((go_terms3)))
            print("Number of set_intersections terms: ", set_intersections(go_terms1, go_terms2, go_terms3))
            go_term_pairs1 = list(itertools.product(go_terms1, go_terms2))
            go_term_pairs1 = [(x, y) for (x, y) in go_term_pairs1 if x != y]
            go_term_pairs2 = list(itertools.product(go_terms2, go_terms3))
            go_term_pairs2 = [(x, y) for (x, y) in go_term_pairs2 if x != y]
            go_term_pairs3 = list(itertools.product(go_terms1, go_terms3))
            go_term_pairs3 = [(x, y) for (x, y) in go_term_pairs3 if x != y]
            go_term_pairs = [go_term_pairs1, go_term_pairs2, go_term_pairs3]
            # GET ALL PATHS BETWEEN ALL GOterms
            total_pairs = {}
            for i in range(len(go_term_pairs)):
                for (x, y) in go_term_pairs[i]:
                    if (y, x) in total_pairs:
                        pass
                    else:
                        total_pairs[(x, y)] = 1
            '''
            total_pairs, go_terms1, go_terms2, go_terms3 = self.get_triplet_go_terms(GO.term2genes, triple)
            #print("Number of total pairs: ", len(total_pairs))
            count = 0
            # print ("Triplet Finding paths...")
            f = find_all_paths(undirectedG)
            for (x, y) in total_pairs:
                try:
                    there = self.all_paths_terms[(x, y)]
                except KeyError:
                    paths = f.getAllPaths(x, y)
                    self.all_paths_terms[(x, y)] = f.get_shortest_paths(paths)
                    count += 1
                # if (count%50)==0: print("{} no.of paths done: ".format(count))

            # MAKE SUBGRAPH for triplets, ADD EDGES OF ALL PATHS

            # ADDING ALL EDGES FROM Gene TO GO TERMS
            added_multi_edges = {}
            new_subgraph = nx.Graph()
            new_multidi_subgraph = nx.MultiDiGraph()
            for go_term1 in go_terms1:
                new_subgraph.add_edge(term1, go_term1)
                try:
                    there = added_multi_edges[(term1, go_term1, "GO_ann")]
                except KeyError:
                    new_multidi_subgraph.add_edge(term1, go_term1, relation="GO_ann")
                    added_multi_edges[(term1, go_term1, "GO_ann")] = 1
            for go_term2 in go_terms2:
                new_subgraph.add_edge(term2, go_term2)
                try:
                    there = added_multi_edges[(term2, go_term2, "GO_ann")] = 1
                except KeyError:
                    new_multidi_subgraph.add_edge(term2, go_term2, relation="GO_ann")
                    added_multi_edges[(term2, go_term2, "GO_ann")] = 1
            for go_term3 in go_terms3:
                new_subgraph.add_edge(term3, go_term3)
                try:
                    there = added_multi_edges[(term3, go_term3, "GO_ann")] = 1
                except KeyError:
                    new_multidi_subgraph.add_edge(term3, go_term3, relation="GO_ann")
                    added_multi_edges[(term3, go_term3, "GO_ann")] = 1

            # ADDING EDGES between GO terms FROM ALL PATHS
            for (x, y) in total_pairs:
                directed_paths = []
                directed_paths_rv = []
                for path in self.all_paths_terms[(x, y)]:
                    if len(path) == 1: continue
                    directed_path = []
                    directed_path_rv = []
                    for i in range(len(path) - 1):
                        new_subgraph.add_edge(path[i], path[i + 1])
                        attribute_dict = multidigraphG.get_edge_data(path[i], path[i + 1])
                        attribute_dict2 = multidigraphG.get_edge_data(path[i + 1], path[i])
                        try:
                            there = added_multi_edges[(path[i], path[i + 1], attribute_dict[0]["relation"])]
                        except KeyError:
                            new_multidi_subgraph.add_edge(path[i], path[i + 1], relation=attribute_dict[0]["relation"])
                            added_multi_edges[(path[i], path[i + 1], attribute_dict[0]["relation"])] = 1
                        try:
                            there = added_multi_edges[(path[i + 1], path[i], attribute_dict2[0]["relation"])]
                        except KeyError:
                            new_multidi_subgraph.add_edge(path[i + 1], path[i], relation=attribute_dict2[0]["relation"])
                            added_multi_edges[(path[i + 1], path[i], attribute_dict2[0]["relation"])] = 1

                        # if len(attribute_dict)!=1:
                        #    print ("LENGTH NOT EQUAL TO 1")

                        # if len(attribute_dict2)!=1:
                        #    print ("LENGTH NOT EQUAL TO 1")

                        directed_path += [(path[i], path[i + 1], attribute_dict[0]["relation"])]
                        directed_path_rv += [(path[i + 1], path[i], attribute_dict2[0]["relation"])]
                    directed_path_rv = directed_path_rv[::-1]
                    directed_paths += [directed_path]
                    directed_paths_rv += [directed_path_rv]
                try:
                    there = self.all_directedpaths_terms[(x, y)]
                except KeyError:
                    self.all_directedpaths_terms[(x, y)] = directed_paths
                try:
                    there = self.all_directedpaths_terms[(y, x)]
                except KeyError:
                    self.all_directedpaths_terms[(y, x)] = directed_paths_rv

            self.triplet_subgraph[triple] = new_subgraph
            self.triplet_multigraph[triple] = new_multidi_subgraph
            self.triplet_total_pairs[triple] = total_pairs
        # pickle.dump(triplet_subgraph, open("triplet_subgraphs.p","wb"))
        #self.triplet_subgraph = triplet_subgraph
        #self.triplet_multigraph = triplet_multigraph
        #self.all_paths_terms, self.all_directedpaths_terms = all_paths_terms, all_directedpaths_terms
        #self.triplet_total_pairs = triplet_total_pairs
        return
        #return triplet_subgraph, triplet_multigraph, all_paths_terms, all_directedpaths_terms, top_triplets, bottom_triplets


def node_edge_analysis(total_pairs, all_paths_terms, all_directedpaths_terms):
    '''
    Compute node,edge,path frequencies in a triplet by using all the paths between all pairs of go temrs of the triplet.
    Parameters
    ----------
    total_pairs
    all_paths_terms
    all_directedpaths_terms

    Returns
    -------

    '''
    nodes_frequency = {}
    edge_frequency = {}
    path_frequency = {}
    for (x,y) in total_pairs:
        paths = all_paths_terms[(x,y)]
        for path in paths:
            for node in path:
                try:
                    nodes_frequency[node] += 1
                except KeyError:
                    nodes_frequency[node] = 1
        paths1 = all_directedpaths_terms[(x,y)]
        for path in paths1:
            relation_path = []
            for edge in path:
                relation_path += [edge[2]] 
                try:
                    edge_frequency[edge[2]] += 1  
                except KeyError:
                    edge_frequency[edge[2]] = 1  
            try:
                path_frequency[tuple(relation_path)] += 1 
            except KeyError:
                path_frequency[tuple(relation_path)] = 1 
        paths2 = all_directedpaths_terms[(y,x)]
        for path in paths2:
            relation_path = []
            for edge in path:
                relation_path += [edge[2]]
                try:
                    edge_frequency[edge[2]] += 1  
                except KeyError:
                    edge_frequency[edge[2]] = 1 
            try:
                path_frequency[tuple(relation_path)] += 1 
            except KeyError:
                path_frequency[tuple(relation_path)] = 1  
    return [nodes_frequency,edge_frequency, path_frequency]

def get_frequencies(triplet_VE_stats, triplets, go):
    '''
    Agrregate node, edge, path statsitics over a set of triplets.
    Parameters
    ----------
    triplet_VE_stats
    triplets
    Returns
    -------
    '''
    nodes_frequency = {}
    edges_frequency = {}
    paths_frequency = {}
    for triple in triplets:
        for node in triplet_VE_stats[triple][0]:
            try:
                nodes_frequency[node] += triplet_VE_stats[triple][0][node]
            except KeyError:
                nodes_frequency[node] = triplet_VE_stats[triple][0][node]
        for edge in triplet_VE_stats[triple][1]:
            try:
                edges_frequency[edge] += triplet_VE_stats[triple][1][edge]
            except:
                edges_frequency[edge] = triplet_VE_stats[triple][1][edge]
        for path in triplet_VE_stats[triple][2]:
            try:
                paths_frequency[path] += triplet_VE_stats[triple][2][path]
            except:
                paths_frequency[path] = triplet_VE_stats[triple][2][path]
    nodes_frequency = name_go_terms_dict(nodes_frequency, go)
    return nodes_frequency,edges_frequency,paths_frequency

def get_VE_global_stats(triplet_VE_stats, top_triplets, bottom_triplets, all_triplets, filename_suffix, go):
    '''
    Find node, edge, path frequencies statistics for top triplets and bottom triplets separately and write them in a file.
    Parameters
    ----------
    triplet_VE_stats
    top_triplets
    bottom_triplets
    all_triplets
    go

    Returns
    -------

    '''
    top_nodes_frequency,top_edges_frequency,top_paths_frequency = get_frequencies(triplet_VE_stats, top_triplets, go)
    bottom_nodes_frequency,bottom_edges_frequency,bottom_paths_frequency= get_frequencies(triplet_VE_stats, bottom_triplets, go)
    all_nodes_frequency, all_edges_frequency, all_paths_frequency = get_frequencies(triplet_VE_stats, all_triplets, go)

    top_triplet_stats = [top_nodes_frequency,top_edges_frequency,top_paths_frequency]
    bottom_triplet_stats = [bottom_nodes_frequency,bottom_edges_frequency,bottom_paths_frequency]
    all_nodes_stats = [all_nodes_frequency, all_edges_frequency, all_paths_frequency]
    filenames = ["top_bottom_nodes_" + filename_suffix, "top_bottom_edges_" + filename_suffix, "top_bottom_paths_" + filename_suffix]

    for i in range(len(top_triplet_stats)):
        all_common_nodes = set(list(top_triplet_stats[i].keys())+ list(bottom_triplet_stats[i].keys()))
        node_keys = sorted(top_triplet_stats[i].items(), key = lambda l:l[1], reverse=True)
        node_keys = [node[0] for node in node_keys]
        remaining_nodes = []
        for node in all_common_nodes:
            try:
                top_triplet_stats[i][node]
            except KeyError:
                remaining_nodes += [node]
        all_common_nodes = node_keys+remaining_nodes

        print ("NODE TOP AND BOTTOM FREQUENCIES writing....")
        with open("graphs/" + filenames[i] + ".csv", "w") as outfile:
            for node in all_common_nodes:
                node_string = "_".join([n for n in node])
                if (node in top_triplet_stats[i]) and (node in bottom_triplet_stats[i]):
                    outfile.write(node_string + "," + str(top_triplet_stats[i][node]) + "," + str(bottom_triplet_stats[i][node]) + "\n")
                elif node in top_triplet_stats[i]:
                    outfile.write(node_string + "," + str(top_triplet_stats[i][node]) + "," + str(0) + "\n")
                elif node in bottom_triplet_stats[i]:
                    outfile.write(node_string + "," + str(0) + "," + str(bottom_triplet_stats[i][node]) + "\n")
            

        #frequency normalization
        for node in bottom_triplet_stats[i]:
            bottom_triplet_stats[i][node] = float(bottom_triplet_stats[i][node])/float(all_nodes_stats[i][node])
        for node in top_triplet_stats[i]:
            top_triplet_stats[i][node] = float(top_triplet_stats[i][node])/float(all_nodes_stats[i][node])

        print ("NODE NORMALIZED TOP AND BOTTOM FREQUENCIES writing....")
        with open("graphs/" + filenames[i] + "_normalized.csv", "w") as outfile:
            for node in all_common_nodes:
                node_string = "_".join([n for n in node])
                if (node in top_triplet_stats[i]) and (node in bottom_triplet_stats[i]):
                    outfile.write(node_string + "," + str(top_triplet_stats[i][node]) + "," + str(bottom_triplet_stats[i][node]) + "\n")
                elif node in top_triplet_stats[i]:
                    outfile.write(node_string + "," + str(top_triplet_stats[i][node]) + "," + str(0) + "\n")
                elif node in bottom_triplet_stats[i]:
                    outfile.write(node_string + "," + str(0) + "," + str(bottom_triplet_stats[i][node]) + "\n")

def plot_triplet_graphs(triplet_subgraph, triplets, prefix):
    '''
    plotting graphs for the given triplets lists.
    PLOT TRIPLET GRAPH: step1: labels nodes from 1 to n and GO1, GO2, GO3. step2: Plot the graph with labels
    step3: write the mapping from nodes labels to node names in a file
    Parameters
    ----------
    triplet_subgraph
    triplets

    Returns
    -------

    '''
    triplets = triplets.tolist()
    discriminative_terms_list = []#['invasive growth in response to glucose limitation', 'structural molecule activity', 'extracellular region', 'vacuole', 'generation of precursor metabolites and energy','DNA-templated transcription, initiation','cellular response to DNA damage stimulus','mitochondrion organization','pseudohyphal growth','sporulation']
    for k,triple in enumerate(triplets):
        subgraph = triplet_subgraph[triple]
        term1,term2,term3 = triple.split('_')[0], triple.split('_')[1], triple.split('_')[2]
        genes_list = [term1, term2, term3]
        nodes = subgraph.nodes()
        term1,term2,term3 = triple.split('_')[0], triple.split('_')[1], triple.split('_')[2]
        node_labels = {}
        '''
        node_labels[term1] = "GO1"
        node_labels[term2] = "GO2"
        node_labels[term3] = "GO3"
        count = 4
        for node in nodes:
            if node in genes_list:
                pass
            else:
                node_labels[node] = count
            count += 1
        '''
        node_labels[term1],node_labels[term2], node_labels[term3] = term1,term2,term3
        count = 1
        for node in nodes:
            if node in genes_list:
                pass
            else:
                if node in go:
                    if go[node]["name"] in discriminative_terms_list:
                        node_labels[node] = go[node]["name"]
                    else:
                        node_labels[node] = count
                else:
                    node_labels[node] = count
            count += 1

        with open("graphs/" + prefix + str(k) + "_tripletgraph_labels_" + triple + ".txt", "w") as outfile:
            outfile.write("TRIPLET NAME: {}\n".format(triple))
            outfile.write("Subgraph details: no.of nodes {} no.of edges:{}\n".format(subgraph.__len__(),subgraph.size()))
            for node in node_labels:
                node_name = go[node]["name"] if node in go else "NA"
                outfile.write("Node:{} Label:{} Node name:{}\n".format(node, node_labels[node], node_name))

        node_list = subgraph.nodes()
        node_sizes = [800 if x in genes_list else 300 for x in node_list]
        nx.draw_networkx(G = subgraph, labels = node_labels, with_labels = True)
        plt.savefig("graphs/" + prefix + str(k) + "_triplegraph_" + triple + ".png")
        plt.clf()

#def compute_relation_paths(triplet_multigraph, all_directedpaths_terms, top_triplets, bottom_triplets):
def compute_centrality(triplet_subgraph, top_triplets, bottom_triplets):
    #Compute Average Node Centralities of top triplet subgraphs and bottom triplet subgraphs. Get nodes(GO temrms) with high centralities. 
    total_node_centrality = {}
    total_node_centrality2 = {}
    with open("graphs/node_centralities.txt", "w") as outfile:
        for triple in top_triplets:
            node_centrality = nx.degree_centrality(triplet_subgraph[triple])
            outfile.write("TOP TRIPLET NAME: {}, node_centrality top 30: {}\n".format(triple, sorted(node_centrality.items(), key = lambda l:l[1], reverse=True)[:30]))
            for node in node_centrality:
                try:
                    total_node_centrality[node] += [node_centrality[node]]
                except KeyError:
                    total_node_centrality[node] = [node_centrality[node]]
        for triple in bottom_triplets:
            node_centrality = nx.degree_centrality(triplet_subgraph[triple])             
            outfile.write("BOTTOM TRIPLET NAME: {}, node_centrality top 30: {}\n".format(triple, sorted(node_centrality.items(), key = lambda l:l[1], reverse=True)[:30]))
            for node in node_centrality:
                try:
                    total_node_centrality2[node] += [node_centrality[node]]
                except KeyError:
                    total_node_centrality2[node] = [node_centrality[node]]
        avg_node_centrality,avg_node_centrality2 = {},{}
        for node in total_node_centrality:
            avg_node_centrality[node] = float(sum(total_node_centrality[node]))/float(len(total_node_centrality[node]))
        for node in total_node_centrality2:
            avg_node_centrality2[node] = float(sum(total_node_centrality2[node]))/float(len(total_node_centrality2[node]))
        avg_node_centrality = sorted(avg_node_centrality.items(), key = lambda l:l[1], reverse=True)
        avg_node_centrality2 = sorted(avg_node_centrality2.items(), key = lambda l:l[1], reverse=True)
        for node,avg_cent in avg_node_centrality[:30]:
            outfile.write("TOP TRIPLET Node centrality: {}, {}, {}, {}\n".format(node, go[node]["name"] if node in go else "NA", avg_cent, total_node_centrality[node]))
        for node,avg_cent in avg_node_centrality2[:30]:
            outfile.write("BOTTOM TRIPLET Node centrality: {}, {}, {}, {}\n".format(node, go[node]["name"] if node in go else "NA", avg_cent, total_node_centrality2[node]))
    return total_node_centrality, total_node_centrality2

if __name__== "__main__":
    parser = argparse.ArgumentParser("Genetic triple mutation ltr models")
    parser.add_argument('-dataset', default='../../../srv/local/work/DeepMutationsData/data/triple_fitness.tsv', type=str, help="path to dataset")
    parser.add_argument('-dataset_double', default='../../../srv/local/work/DeepMutationsData/data/double_fitness.tsv', type=str, help="path to dataset")
    #parser.add_argument("-save_dir", type=str, default='onto_exps', help="directory for saving results and models")
    parser.add_argument('-onto_filename', default='../../../srv/local/work/DeepMutationsData/data/goslim_yeast.obo', type=str, help="ontology features filename")
    args = parser.parse_args()  

    GO = Ontology()
    #term2genes, term2gofeatures, go = load_ontologydata_2('../../../srv/local/work/DeepMutationsData/data/goslim_yeast.obo')
    trigenic_data = get_trigenic_data()
    print ("Number of triplets: ",len(trigenic_data.index))
    topk = 1
    top_triplet = trigenic_data[trigenic_data['reverserank'] <= topk] 
    bottom_triplet = trigenic_data[trigenic_data['rank'] <= topk] 
    top_triplets = top_triplet['ids'].to_numpy(copy="True")
    bottom_triplets = bottom_triplet['ids'].to_numpy(copy="True")
    triplets = trigenic_data['ids'].to_numpy(copy="True")

    GO_graph = Ontology_graph(GO.go)
    GO_graph.make_triplet_graphs(GO, triplets)
    triplet_VE_stats = {}
    for triple in triplets:
        triplet_VE_stats[triple] = node_edge_analysis(GO_graph.triplet_total_pairs[triple], GO_graph.all_paths_terms, GO_graph.all_directedpaths_terms)

    #get_VE_global_stats(triplet_VE_stats, top_triplets, bottom_triplets, triplets, "top" + str(topk), GO.go)
    #triplet_subgraph,triplet_multigraph,all_paths_terms,all_directedpaths_terms,top_triplets, bottom_triplets =  get_triplet_graph(GO, trigenic_data, triplets, top_triplets, bottom_triplets)
    plot_triplet_graphs(GO_graph.triplet_subgraph, top_triplets, "top")
    #plot_triplet_graphs(GO_graph.triplet_subgraph, bottom_triplets, "bottom")
    #total_node_centrality, total_node_centrality2 = compute_centrality(triplet_subgraph, top_triplets, bottom_triplets)
    #compute_common_GO_terms(triplet_subgraph, top_triplets, bottom_triplets)
    #compute_common_relations(triplet_multigraph, top_triplets, bottom_triplets)
    #compute_relation_paths(triplet_multigraph, all_directedpaths_terms, top_triplets, bottom_triplets)


        









        