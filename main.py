from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_threshold
from graphein.protein.features.nodes.dssp import rsa, secondary_structure
from graphein.protein.config import DSSPConfig
from functools import partial
import pandas as pd
from pathlib import Path
from os import path
import argparse
from graph import *


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

# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')

#command example
#python3 --molA_path /home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_mage3_5brz_renumber.pdb --molB_path /home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_titin_5bs0_renumber.pdb --interface_list /home/helder/Projects/pMHC_graphs/interface_MHC_unique.csv --centroid_threshold 10 --run_name teste_sim --association_mode similarity --output_path output5

################################################################################################
parser = argparse.ArgumentParser(description='Building common subgraphs')
parser.add_argument('--mols_paths', type=list_of_strings, default='',
                    help='Path list of PDB files.')
# parser.add_argument('--molB_path', type=str, default='',
#                     help='Path to the first PDB file.')
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

### functions
def get_exposed_residues_mhc(graph, inter_list, rsa_threshold = 0.1, chains_peptide = ["C"], chain_mhc = "A"):
    graph.create_subgraph(name= "peptide_list", chains=chains_peptide)
    graph.create_subgraph(name="mhc_list", sequence_positions=inter_list)
    graph.filter_subgraph(subgraph_name="mhc_list", filter_func= lambda i: i[0] == chain_mhc)
    graph.create_subgraph(name="all_solv_exposed", rsa_threshold = rsa_threshold)
    graph.join_subgraph(name="all_list", graphs_name=["peptide_list", "mhc_list"])
    lista_peptide_helix_exposed = graph.filter_subgraph(subgraph_name="all_solv_exposed", name="peptide_helix_exposed_list", filter_func = lambda i: i in graph.subgraphs["all_list"], return_node_list=True)
    graph.create_subgraph(name = "s_graph", node_list=lista_peptide_helix_exposed, return_node_list= False)

    return graph.get_subgraph("s_graph")


#### inputs 
#List of MHC interface residues with TCR from the non-redundant TCR:pMHC analysis
inter_list = pd.read_csv(args.interface_list, header=None)[0].to_list()
#Threshold for centroid distance
centroid_threshold=args.centroid_threshold
# List of paths
mols_paths = args.mols_paths

#output folder
output_path = args.output_path
#Path to full common subgraph
path_full_subgraph = path.join(output_path,f"full_association_graph_{args.run_name}.png")
################################

#check if output folder exists, otherwise create it 
Path(output_path).mkdir(parents=True, exist_ok=True)

#Initialize the protein graph config
config = ProteinGraphConfig(edge_construction_functions=[partial(add_distance_threshold, long_interaction_threshold=0, threshold=centroid_threshold)],
                            graph_metadata_functions=[rsa, secondary_structure], 
                            dssp_config=DSSPConfig(),
                            granularity="centroids") #using CB granularity we are missing the Gly residues!!!


graphs = []

for mol_path in mols_paths:
    #Build general pMHC interface graph for MolA
    g = Graph(config=config, graph_path=mol_path)
    s_g = get_exposed_residues_mhc(g, inter_list=inter_list, rsa_threshold= 0.1, chains_peptide=["C"], chain_mhc="A")
    graphs.append((s_g, mol_path))

# G = AssociatedGraph(graphA = s_g1, molA_path = molA_path, graphB = s_g2, molB_path=molB_path, output_path=output_path, path_full_subgraph=path_full_subgraph,association_mode=args.association_mode, run_name= args.run_name,interface_list=inter_list, centroid_threshold=10)
G = AssociatedGraph(graphs= graphs, output_path=output_path, path_full_subgraph=path_full_subgraph, association_mode=args.association_mode, run_name= args.run_name, interface_list=inter_list, centroid_threshold=args.centroid_threshold)
G_sub = G.associated_graph

G.draw_graph(show = True)

G.grow_subgraph_bfs()