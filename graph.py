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
from typing import Callable, Any, Union, Optional 


class AssociatedGraph:
    def __init__(self, graphA, graphB, output_path, path_full_subgraph, association_mode = "identity",  interface_list = None, centroid_threshold = 10):
        if interface_list:
            self.interface_list = pd.read_csv(interface_list, header=None)[0].to_list()
            
        self.graphA = graphA
        self.graphB = graphB
        
        self.output_path = output_path
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.path_full_subgraph = path_full_subgraph
        
        pass
    
class Graph:
    def __init__(self, graph_path, config = None):
        if config:
            self.config = config
        else:
            self.config = ProteinGraphConfig(edge_construction_functions=[partial(add_distance_threshold, long_interaction_threshold=0, threshold=centroid_threshold)],
                            graph_metadata_functions=[rsa, secondary_structure], 
                            dssp_config=DSSPConfig(),
                            granularity="centroids")

            
        self.graph = construct_graph(config=config, path=graph_path)
        
        self.subgraphs = {}
    
    def get_subgraph(self, name:str):
        if name not in self.subgraphs.keys():
            print(f"Can't find {name} in subgraph")
        else:
            return self.subgraphs[name]
    
    def create_subgraph(self, name: str, node_list: list = [], return_node_list: bool = False, **args):
        if name in self.subgraphs.keys():
            print(f"You already have this subgraph created. Use graph.delete_subraph({name}) before creating it again.")
        elif not node_list:
            self.subgraphs[name] =  extract_subgraph(g = self.graph, **args)
            print(f"Subgraph {name} created with success!")
        elif node_list:
            self.subgraphs[name] = self.graph.subgraph(node_list)
        
        if return_node_list:
            return self.subgraphs[name].nodes
        
    def delete_subraph(self, name: str):
        if name not in  self.subgraphs.keys():
            del self.subgraphs[name]
        else:
            print(f"{name} isn't in.subgraphs")
            
    def filter_subgraph(self, 
            subgraph_name: str,
            filter_func: Union[Callable[..., Any], str],
            name: Union[str, None] = None, 
            return_node_list: bool = False):
           
        nodes = [i for i in self.subgraphs[subgraph_name].nodes if filter_func(i)]
        if name:
            self.subgraphs[name] = self.subgraphs[subgraph_name].subgraph(nodes)
        else:
            self.subgraphs[subgraph_name] = self.subgraphs[subgraph_name].subgraph(nodes)
        
        if return_node_list:
            return self.subgraphs[name].nodes if name != None else self.subgraphs[subgraph_name].nodes
        
    def join_subgraph(self, name: str, graphs_name: list, return_node_list: bool = False):
        if name in self.subgraphs.keys():
            print(f"You already have this subgraph created. Use graph.delete_subraph({name}) before creating it again.")
        elif set(graphs_name).issubset(self.subgraphs.keys()):
            
            self.subgraphs[name] = self.graph.subgraph([node for i in graphs_name for node in self.subgraphs[i].nodes])
            if return_node_list:
                return self.subgraphs[name].nodes
        else:
            print(f"Some of your subgraph isn't in the subgraph list")
        
        
        
        
    
        

