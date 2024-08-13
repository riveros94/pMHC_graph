from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph
from graphein.protein.edges.distance import add_distance_threshold, compute_distmat
from graphein.protein.features.nodes.dssp import rsa, secondary_structure
from graphein.protein.config import DSSPConfig
from graphein.protein.subgraphs import extract_surface_subgraph, extract_subgraph, extract_subgraph_from_secondary_structure
from functools import partial
import networkx as nx
import matplotlib.pyplot as plt
from utils import check_identity, compute_atom_distance, build_contact_map, find_contact_residues, coefficient_of_variation, check_identity_same_chain, check_neighborhoods, create_sphere_residue, add_sphere_residues, create_subgraph_with_neighbors, check_cross_positions, graph_message_passing, check_similarity
from itertools import combinations, compress
import plotly.graph_objects as go
import numpy as np
from itertools import product
import pandas as pd
from pathlib import Path
from os import path
import argparse

'''
#This script takes two pMHC structures as input and returns the common subgraphs, which will also be mapped to the structure

#The input PDB structures must be unbound, without the TCR, only the pMHC structure

#original work of a similar algorithm can be found in Protein−Protein Binding-Sites Prediction by Protein Surface Structure Conservation Janez Konc and Dušanka Janežič, 2006

#Graphein was tested inside a conda enviroment

#Graphein was not computing correctly the centoids when multiple chains were included, I correction I made was replace
#["residue_number", "chain_id", "residue_name", "insertion"] by
#["chain_id", "residue_number", "residue_name", "insertion"] in anaconda3/envs/graphein/lib/python3.8/site-packages/graphein/protein/graphs.py

#I found some incompabilities when running dssp with graphein, so I had to made a correction in my graphein version:
#In anaconda3/envs/graphein/lib/python3.8/site-packages/graphein/protein/features/nodes/dssp.py
#I changed #pdb_file, DSSP=executable, dssp_version=dssp_version to 
#pdb_file, DSSP=executable
'''


#command example
#python3 --molA_path /home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_mage3_5brz_renumber.pdb --molB_path /home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_titin_5bs0_renumber.pdb --interface_list /home/helder/Projects/pMHC_graphs/interface_MHC_unique.csv --centroid_threshold 10 --run_name teste_sim --association_mode similarity --output_path output5

################################################################################################
parser = argparse.ArgumentParser(description='Building common subgraphs')
parser.add_argument('--molA_path', type=str, default='',
                    help='Path to the first PDB file.')
parser.add_argument('--molB_path', type=str, default='',
                    help='Path to the first PDB file.')
parser.add_argument('--interface_list', type=str, default='',
                    help='File with a canonical list of MHC residues at the interface with TCR. No header needed for this file.')
parser.add_argument('--centroid_threshold', type=int, default=10,
                    help="Distance threshold for building the molA and molB interface graphs")
parser.add_argument('--run_name', type=str, default='test',
                    help='Name for storing results in the output folder')
parser.add_argument('--association_mode', type=str, default='identity',
                    help='Mode for creating association nodes. Identify or similarity.')                                        
parser.add_argument('--output_path', type=str, default='~/',
                    help='Path to store output results.')
args = parser.parse_args()
################################################################################################


#### inputs 
#List of MHC interface residues with TCR from the non-redundant TCR:pMHC analysis
inter_list = pd.read_csv(args.interface_list, header=None)[0].to_list()
#Threshold for centroid distance
centroid_threshold=args.centroid_threshold
#Path to Mol A
molA_path = args.molA_path
#Path to Mol B
molB_path = args.molB_path
#output folder
output_path = args.output_path
#Path to full common subgraph
path_full_subgraph = path.join(output_path,f"full_association_graph_{args.run_name}.png")
################################

#check if output folder exists, otherwise create it 
Path(args.output_path).mkdir(parents=True, exist_ok=True)

#Initialize the protein graph config
config = ProteinGraphConfig(edge_construction_functions=[partial(add_distance_threshold, long_interaction_threshold=0, threshold=centroid_threshold)],
                            graph_metadata_functions=[rsa, secondary_structure], 
                            dssp_config=DSSPConfig(),
                            granularity="centroids") #using CB granularity we are missing the Gly residues!!!

#Build general pMHC interface graph for MolA
g1 = construct_graph(config=config, path=molA_path)
lista_peptide = extract_subgraph(g1, chains=['C'], return_node_list = True)
lista_mhc = extract_subgraph(g1, sequence_positions=inter_list, return_node_list = True)
lista_mhc = [i for i in lista_mhc if i[0] == 'A']
all_solv_exposed = extract_subgraph(g1, rsa_threshold = 0.1, return_node_list = True)
lista_all = lista_peptide + lista_mhc
lista_peptide_helix_exposed = [i for i in all_solv_exposed if i in lista_all]
s_g1 = extract_subgraph(g1, node_list = lista_peptide_helix_exposed, return_node_list = False)

#Build general pMHC interface graph for MolB
g2 = construct_graph(config=config, path=molB_path)
lista_peptide = extract_subgraph(g2, chains=['C'], return_node_list = True)
lista_mhc = extract_subgraph(g2, sequence_positions=inter_list, return_node_list = True)
lista_mhc = [i for i in lista_mhc if i[0] == 'A']
all_solv_exposed = extract_subgraph(g2, rsa_threshold = 0.1, return_node_list = True)
lista_all = lista_peptide + lista_mhc
lista_peptide_helix_exposed = [i for i in all_solv_exposed if i in lista_all]
s_g2 = extract_subgraph(g2, node_list = lista_peptide_helix_exposed, return_node_list = False)

#Create a contact map and a list with residue names order as the contact map
#We do not need to do this, since graphein already build internally the distance matrix considering the centroid
#This matrix can be obtained from: graph.graph["pdb_df"]
#The contact matrix can be obtained from: graph.graph["dist_mat"]
#Usually this dist_mat is not updated in related to the pdb_df after graph subsets, so it is important to double check it 
#Instead of using the dist_mat we can build it again using compute_distmat(graph.graph["pdb_df"])
contact_map1, residue_map1, residue_map1_all = build_contact_map(molA_path) #_all means with resname
contact_map2, residue_map2, residue_map2_all = build_contact_map(molB_path)

#Run the cartesian product to create the association graphs
#This can be substituted for just getting the combinations of node names, no need to build the original graphs!
M = nx.cartesian_product(s_g1, s_g2)

if args.association_mode == 'identity':
    #Filter the pair of nodes based on identity, same chain and neighborhoods
    select_ident = [i for i in list(M) if check_identity(i) and check_identity_same_chain(i) and check_neighborhoods(i, contact_map1, residue_map1_all, contact_map2, residue_map2_all)]
elif args.association_mode == 'similarity':
    message_passing_molA = graph_message_passing(s_g1, '/home/helder/Projects/pMHC_graphs/atchley_aa.csv', use_degree=False, norm_features=False)
    message_passing_molB = graph_message_passing(s_g2, '/home/helder/Projects/pMHC_graphs/atchley_aa.csv', use_degree=False, norm_features=False)
    #Filter the pair of nodes 
    select_ident = [i for i in list(M) if check_similarity(i, message_passing_molA, message_passing_molB, threshold=0.95) and check_identity_same_chain(i)]
else:
    print(f'Mode {args.association_mode} is not a valid mode')
    quit

#Create the association graphs
paired_graphs = [list(pair) for pair in combinations(select_ident, 2)]

#Filter out pairs of pairs (edges) based on cross positions
#For instance, if an association node has a pair with same positions 169 and 169 and its paired association node has a pair with different positions (169 and 170, for instance), it does not make sense to connect these nodes
paired_graphs_filtered = [i for i in paired_graphs if check_cross_positions(i)]

#To build the edges between association nodes, we first calculate the d1 distance, then d2 distance and then the d1d2 ratio
#calculate d1 distance
d1 = [find_contact_residues(contact_map1, residue_map1, (i[0][0].split(':')[0],int(i[0][0].split(':')[2])),  (i[1][0].split(':')[0],int(i[1][0].split(':')[2]))) for i in paired_graphs_filtered]
#calculate d2 distance
d2 = [find_contact_residues(contact_map2, residue_map2, (i[0][1].split(':')[0],int(i[0][1].split(':')[2])),  (i[1][1].split(':')[0],int(i[1][1].split(':')[2]))) for i in paired_graphs_filtered]
#calculate d1d2 ratio distance using coefficient of variation
d1d2 = [coefficient_of_variation(a,b) for a,b in zip(d1,d2)]
#generate the pair of connected nodes indicating the edges
new_nodes_edges = [paired_graphs_filtered[n] for n,m in enumerate(d1) if d1[n] < 10 and d2[n] < 10 and d1[n] > 0 and d2[n] > 0 and d1d2[n] < 15 and (paired_graphs_filtered[n][0][0] != paired_graphs_filtered[n][1][1] and paired_graphs_filtered[n][0][1] != paired_graphs_filtered[n][1][0])]

# Build the new graph
G_sub = nx.Graph()
# Add nodes
for sublist in new_nodes_edges:
    for edge in sublist:
        G_sub.add_node(edge[0])
        G_sub.add_node(edge[1])
# Add edges
for sublist in new_nodes_edges:
    G_sub.add_edge(sublist[0], sublist[1])

# Remove nodes with no edges
G_sub.remove_nodes_from(list(nx.isolates(G_sub)))

# Add chain as attribute to color the nodes 
for nodes in G_sub.nodes:
    if nodes[0].startswith('A') and nodes[1].startswith('A'):
        G_sub.nodes[nodes]['chain_id'] = 'red'
    elif nodes[0].startswith('C') and nodes[1].startswith('C'):
        G_sub.nodes[nodes]['chain_id'] = 'blue'
    else:
        G_sub.nodes[nodes]['chain_id'] = None
node_colors = [G_sub.nodes[node]['chain_id'] for node in G_sub.nodes]

# Draw the full cross-reactive subgraph
nx.draw(G_sub, with_labels=True, node_color=node_colors, node_size=50, font_size=6)
#plt.show()
plt.savefig(path_full_subgraph)
plt.clf()

# Build all possible common TCR interface pMHC subgraphs centered at the peptide nodes 
count_pep_nodes = 0
for nodes in G_sub.nodes:
    if nodes[0].startswith('C') and nodes[1].startswith('C'): #i.e. peptide nodes
        count_pep_nodes += 1
        #print(nodes)
        bfs_subgraph = create_subgraph_with_neighbors(s_g1, s_g2, G_sub, nodes, 20)
        for nodes2 in bfs_subgraph.nodes:
            if nodes2[0].startswith('A') and nodes2[1].startswith('A'):
                bfs_subgraph.nodes[nodes2]['chain_id'] = 'red'
            elif nodes2[0].startswith('C') and nodes2[1].startswith('C'):
                bfs_subgraph.nodes[nodes2]['chain_id'] = 'blue'
            else:
                bfs_subgraph.nodes[nodes2]['chain_id'] = None
        #check whether nodes meet the requirements to be a TCR interface
        number_peptide_nodes = len([i for i in bfs_subgraph.nodes if i[0].startswith('C')])
        if bfs_subgraph.number_of_nodes() >= 14 and nx.diameter(bfs_subgraph) >= 3 and number_peptide_nodes >=3:
            node_colors = [bfs_subgraph.nodes[node]['chain_id'] for node in bfs_subgraph.nodes]
            nx.draw(bfs_subgraph, with_labels=True, node_color=node_colors)
            plt.savefig(path.join(output_path,f'plot_bfs_{nodes[0]}_{args.run_name}.png'))
            plt.clf()

            #write PDB with the common subgraphs as spheres
            get_node_names = list(bfs_subgraph.nodes())
            node_names_molA = [i[0] for i in get_node_names]
            node_names_molB = [i[1] for i in get_node_names]

            #create PDB with spheres
            add_sphere_residues(node_names_molA, molA_path, node_names_molB, molB_path, output_path, nodes[0])
            #add_sphere_residues(node_names_molB, path_protein2, path_protein2.split('.pdb')[0]+'_spheres.pdb')
        else:
            print(f"The subgraph centered at the {nodes[0]} node does not satisfies the requirements")
if count_pep_nodes == 0:
    print(f'No peptide nodes were found in the association graph. No subgraph will be generated.')