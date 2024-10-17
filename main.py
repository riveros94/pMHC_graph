from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_threshold
from functools import partial
import pandas as pd
from pathlib import Path
import os
from os import path
import argparse
from graph.graph import *
from config.graph_config import make_graph_config
from io_utils.pdb_io import list_pdb_files, get_user_selection
import logging, sys
import json
import matplotlib



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

def none_or_float(value):
    if value == 'None':
        return None
    return float(value)

#command example
#python3 --molA_path /home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_mage3_5brz_renumber.pdb --molB_path /home/helder/Projects/pMHC_graphs/pdbs_teste/pmhc_titin_5bs0_renumber.pdb --interface_list /home/helder/Projects/pMHC_graphs/interface_MHC_unique.csv --centroid_threshold 10 --run_name teste_sim --association_mode similarity --output_path output5

### functions

def parser_args():
    parser = argparse.ArgumentParser(description='Building common subgraphs')
    parser.add_argument('--mols_path', type=str, default='',
                        help='Path with PDB input files.')
    # parser.add_argument('--interface_list', type=str, default='',
                        # help='File with a canonical list of MHC residues at the interface with TCR. No header needed for this file.')
    parser.add_argument('--centroid_threshold', type=int, default=10,
                        help="Distance threshold for building the molA and molB interface graphs")
    parser.add_argument('--run_name', type=str, default='test',
                        help='Name for storing results in the output folder')
    parser.add_argument('--association_mode', type=str, default='identity',
                        help='Mode for creating association nodes. Identify or similarity.')                                        
    parser.add_argument('--output_path', type=str, default='~/',
                        help='Path to store output results.')
    parser.add_argument('--neighbor_similarity_cutoff', type=float, default=0.95,
                        help="Threshold for neighbor's similarity ")
    parser.add_argument('--residues_similarity_cutoff', type=float, default=0.95,
                        help="Threshold for residue's similarity")
    parser.add_argument('--factors_path', type=str, default=None,
                        help="Factors for calculating the residue similarity ")
    parser.add_argument('--rsa_filter', type=none_or_float, default=0.1,
                        help="Threshold for filter residues by RSA")
    parser.add_argument('--rsa_similarity_threshold', type=float, default=0.90,
                        help="Threshold for make an associate graph using RSA similarity")
    parser.add_argument('--residues_lists', type=str, default=None,
                        help="Path to Json file which contains the pdb residues")
    parser.add_argument('--debug', type=bool, default=False,
                        help="Activate debug mode")
    args = parser.parse_args()
    
    return args

def get_exposed_residues(graph: Graph, rsa_filter = 0.1, params=None):
    all_residues = []
    merge_graphs = []
    
    if rsa_filter is None:
        graph.create_subgraph(name="exposed_residues")
    else:
        graph.create_subgraph(name="exposed_residues", rsa_threshold = rsa_filter)
        
    if "chains" not in params.keys() and "residues" not in params.keys():

        return graph.get_subgraph(name="exposed_residues")

    if "chains" in params.keys():
        if not isinstance(params["chains"], list):
            print(f"`chains` key must have a list. I received: {params['chains']}, {type(params['chains'])}")
            exit(0)
        
        graph.create_subgraph(name= "selected_chains", chains=params["chains"])
        merge_graphs.append("selected_chains")
        
    if "residues" in params.keys():
        if not isinstance(params["residues"], dict):
            print(f"`dict` key must have a list. I received: {params['dict']}, {type(params['dict'])}")
            exit(0)
        
        residues_dict = params["residues"]
        
        for chain in residues_dict:
            if not isinstance(residues_dict[chain], list):
                print(f"The key {chain} in residues isn't a list")
                exit(0)
            
            all_residues += residues_dict[chain]
            
        graph.create_subgraph(name="all_residues", sequence_positions=all_residues)
        merge_graphs.append("all_residues")
    
    graph.join_subgraph(name="merge_list", graphs_name=merge_graphs)
    
    exposed_nodes = graph.filter_subgraph(subgraph_name="exposed_residues", name="selected_exposed_residues", filter_func= lambda i: i in graph.subgraphs["merge_list"], return_node_list=True)
    graph.create_subgraph(name = "s_graph", node_list=exposed_nodes, return_node_list=False)
    
    return graph.get_subgraph("s_graph")

def get_exposed_residues_mhc(graph: Graph, inter_list, rsa_filter = 0.1, chains_peptide = ["C"], chain_mhc = "A"):
    graph.create_subgraph(name= "peptide_list", chains=chains_peptide)
    graph.create_subgraph(name="mhc_list", sequence_positions=inter_list)
    graph.filter_subgraph(subgraph_name="mhc_list", filter_func= lambda i: i[0] == chain_mhc)
    graph.create_subgraph(name="all_solv_exposed", rsa_threshold = rsa_filter)
    graph.join_subgraph(name="all_list", graphs_name=["peptide_list", "mhc_list"])
    lista_peptide_helix_exposed = graph.filter_subgraph(subgraph_name="all_solv_exposed", name="peptide_helix_exposed_list", filter_func = lambda i: i in graph.subgraphs["all_list"], return_node_list=True)
    graph.create_subgraph(name = "s_graph", node_list=lista_peptide_helix_exposed, return_node_list= False)

    return graph.get_subgraph("s_graph")

def main():
    args = parser_args()

    #### Inputs 

    # List of MHC interface residues with TCR from the non-redundant TCR:pMHC analysis
    # inter_list = pd.read_csv(args.interface_list, header=None)[0].to_list()
    
    centroid_threshold=args.centroid_threshold
    rsa_filter = args.rsa_filter
    rsa_similarity_threshold = args.rsa_similarity_threshold
    neighbor_similarity_cutoff = args.neighbor_similarity_cutoff
    
    debug = args.debug
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    if debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        log = logging.getLogger("CRSProtein") 
        log.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        log = logging.getLogger("CRSProtein") 
        log.setLevel(logging.INFO)
    
    with open(args.residues_lists, "r") as f:
        residues_lists = json.load(f) 


    # List of paths
    mols_path = args.mols_path
    mols_files = list_pdb_files(mols_path)
    if not mols_files:
        return
        
    selected_files, reference_graph = get_user_selection(mols_files, mols_path)

    output_path = args.output_path
    #Path to full common subgraph
    path_full_subgraph = path.join(output_path,f"full_association_graph_{args.run_name}.png")
    ################################

    #check if output folder exists, otherwise create it 
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Initialize the protein graph config
    config = make_graph_config(centroid_threshold=centroid_threshold)
    
    graphs = []

    for mol_path in selected_files:
        g = Graph(config=config, graph_path=mol_path[0])
        params = residues_lists[mol_path[1]]
        
        if "base" in params.keys():
            base = params["base"]
            try:
                params = residues_lists[base]
            except KeyError as e:
                raise KeyError(f"I wasn't able to find the template for {base}. Error message: {e}")            
    
        # s_g = get_exposed_residues_mhc(g, inter_list=inter_list, rsa_filter = rsa_filter, chains_peptide=["C"], chain_mhc="A")
        s_g = get_exposed_residues(graph=g, rsa_filter=rsa_filter, params=params )
        graphs.append((s_g, mol_path[0]))

    G = AssociatedGraph(graphs=graphs, reference_graph= reference_graph, output_path=output_path, path_full_subgraph=path_full_subgraph, association_mode=args.association_mode, factors_path=args.factors_path, run_name= args.run_name, centroid_threshold=centroid_threshold, residues_similarity_cutoff=args.residues_similarity_cutoff, neighbor_similarity_cutoff=neighbor_similarity_cutoff, rsa_similarity_threshold=rsa_similarity_threshold)
    # G_sub = G.associated_graph

    G.draw_graph(show = True)
    # print(G.associated_graph.nodes())

    G.grow_subgraph_bfs()

if __name__ == "__main__":
    
    import os

    # Verificar o número de threads OpenMP
    num_threads = os.environ.get('OMP_NUM_THREADS', None)
    if num_threads:
        print(f"Usando {num_threads} threads OpenMP.")
    else:
        print("A variável OMP_NUM_THREADS não está definida.")

    main()
    