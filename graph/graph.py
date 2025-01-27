from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold
from graphein.protein.features.nodes.dssp import rsa, secondary_structure
from graphein.protein.config import DSSPConfig
from graphein.protein.subgraphs import extract_subgraph
from functools import partial
from time import time
import networkx as nx
import matplotlib.pyplot as plt
from utils.tools import association_product,build_contact_map, add_sphere_residues, create_subgraph_with_neighbors
import plotly.graph_objects as go
from pathlib import Path
from os import path
from typing import Callable, Any, Union, Optional, List, Tuple
from operator import itemgetter
import numpy as np

class AssociatedGraph:
    def __init__(self, graphs: List[Tuple], reference_graph: Union[str, int, None], output_path: str, path_full_subgraph: str, run_name: str, 
                 association_mode: str = "identity", centroid_threshold: int = 10, neighbor_similarity_cutoff: float = 0.95, 
                 rsa_similarity_threshold: float = 0.95, residues_similarity_cutoff: float = 0.95, factors_path: Union[str, None] = None):

        self.graphs = graphs
        self.output_path = output_path
        self.run_name = run_name
        self.association_mode = association_mode
        self.centroid_threshold = centroid_threshold
        self.neighbor_similarity_cutoff = neighbor_similarity_cutoff
        self.rsa_similarity_threshold = rsa_similarity_threshold
        self.factors_path = factors_path
        self.residues_similarity_cutoff = residues_similarity_cutoff
        self.reference_graph = reference_graph
        self.graphsList = None
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.path_full_subgraph = path_full_subgraph
        
        self.associated_graph = self._construct_graph(graphs=self.graphs, reference_graph=self.reference_graph, association_mode=self.association_mode, centroid_threshold=self.centroid_threshold, residues_similarity_cutoff=self.residues_similarity_cutoff, neighbor_similarity_cutoff=self.neighbor_similarity_cutoff, rsa_similarity_threshold=self.rsa_similarity_threshold, factors_path=self.factors_path )
        
    def create_graphs_list(self, graphs: List[Tuple]) -> List[Tuple]:

        graphsList = []
        for item in graphs:
            residue_depth = item[0].residue_depth
            contact_map, residue_map, residue_map_all = build_contact_map(item[1])
            rsa_map = np.array(item[0].graph["dssp_df"]["rsa"])
            graphsList.append((item[0], item[1], contact_map, residue_map, residue_map_all, rsa_map, residue_depth ))

        return graphsList
        
    def build_associated_graph(self, graphsList: List[Tuple], reference_graph: Union[str, int, None], association_mode: str, centroid_threshold: int, neighbor_similarity_cutoff: float, rsa_similarity_threshold: float, residues_similarity_cutoff: float, factors_path: Union[str, None]):
        
        if isinstance(reference_graph, int):
            graphsList[0], graphsList[reference_graph] = graphsList[reference_graph], graphsList[0]
            
        elif isinstance(reference_graph, str):

            index = next((i for i, x in enumerate(graphsList) if x[1] == reference_graph), None)

            if index is not None:
                graphsList[0], graphsList[index] = graphsList[index], graphsList[0]
            else:
                graphsList = sorted(graphsList, key=lambda x: len(x[0].nodes()))
        elif reference_graph is None:
            graphsList = sorted(graphsList, key=lambda x: len(x[0].nodes()))
        
        self.graphsList = graphsList
        # contact_maps = [graph[2] for graph in graphsList]

        # graphs = [graph[0] for graph in graphsList]
        # # residue_maps = [graph[3] for graph in graphsList]
        # residue_maps_all = [graph[4] for graph in graphsList]    
        # rsa_maps = [graph[5] for graph in graphsList]
        # residue_depths = [graph[6] for graph in graphsList]
        # nodes_graphs = [list(graph.nodes()) for graph in graphs]

        start = time()
        print(f"Vou iniciar o produto cartesiano")
        
        M = association_product(graphsList=self.graphsList, association_mode = association_mode, factors_path=factors_path, centroid_threshold = centroid_threshold, residues_similarity_cutoff=residues_similarity_cutoff, neighbor_similarity_cutoff = neighbor_similarity_cutoff, rsa_similarity_threshold=rsa_similarity_threshold)
        # M = association_product(graphs=graphs, association_mode = association_mode, factors_path=factors_path, nodes_graphs = nodes_graphs, contact_maps = contact_maps, residue_maps_all = residue_maps_all, rsa_maps=rsa_maps, centroid_threshold = centroid_threshold, residues_similarity_cutoff=residues_similarity_cutoff, neighbor_similarity_cutoff = neighbor_similarity_cutoff, rsa_similarity_threshold=rsa_similarity_threshold)
        
        end = time()
        print(f"Tempo para produto cartesiano: {end - start}")
        
        return M
    
    def _construct_graph(self, graphs: list, reference_graph: Union[str, int, None],  association_mode: str, centroid_threshold: int, neighbor_similarity_cutoff: float, rsa_similarity_threshold: float, residues_similarity_cutoff: float, factors_path: Union[str, None]):
        # Create a contact map and a list with residue names order as the contact map
        # We do not need to do this, since graphein already build internally the distance matrix considering the centroid
        # This matrix can be obtained from: graph.graph["pdb_df"]
        # The contact matrix can be obtained from: graph.graph["dist_mat"]
        # Usually this dist_mat is not updated in related to the pdb_df after graph subsets, so it is important to double check it 
        # Instead of using the dist_mat we can build it again using compute_distmat(graph.graph["pdb_df"])
        
        graphsList= self.create_graphs_list(graphs)
        associated_graph = self.build_associated_graph(graphsList = graphsList, reference_graph= reference_graph, association_mode = association_mode, centroid_threshold = centroid_threshold, factors_path=factors_path, residues_similarity_cutoff=residues_similarity_cutoff, neighbor_similarity_cutoff= neighbor_similarity_cutoff, rsa_similarity_threshold = rsa_similarity_threshold)

        return associated_graph
    
    def add_spheres(self):
        ...
        
    def draw_graph(self, show = True, save = True):
        if not show and not save:
            print("You are not saving or viewing the graph. Please leave at least one of the parameters as true.")
        else:
            node_colors = [self.associated_graph.nodes[node]['chain_id'] for node in self.associated_graph.nodes]
            # Draw the full cross-reactive subgraph
            nx.draw(self.associated_graph, with_labels=True, node_color=node_colors, node_size=50, font_size=6)
            
            if show:
                plt.show()
            if save:
                plt.savefig(self.path_full_subgraph)
                plt.clf()
    
                print(f"GraphAssociated's plot saved in {self.path_full_subgraph}")
  
    def grow_subgraph_bfs(self):
        # Build all possible common TCR interface pMHC subgraphs centered at the peptide nodes 
        count_pep_nodes = 0
        G_sub = self.associated_graph
        for nodes in G_sub.nodes:
            if nodes[0].startswith('C') and nodes[1].startswith('C'): #i.e. peptide nodes
                count_pep_nodes += 1
                #print(nodes)
                bfs_subgraph = create_subgraph_with_neighbors(self.graphs, G_sub, nodes, 20)
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
                    plt.savefig(path.join(self.output_path,f'plot_bfs_{nodes[0]}_{self.run_name}.png'))
                    plt.clf()

                    #write PDB with the common subgraphs as spheres
                    get_node_names = list(bfs_subgraph.nodes())
                    list_node_names = []
                    for n in range(len(self.graphs)):
                        node_names = [i[n] for i in get_node_names]
                        list_node_names.append(node_names)
  
                    #create PDB with spheres
                    add_sphere_residues(self.graphs, list_node_names, self.output_path, nodes[0])
                    #add_sphere_residues(node_names_molB, path_protein2, path_protein2.split('.pdb')[0]+'_spheres.pdb')
                else:
                    print(f"The subgraph centered at the {nodes[0]} node does not satisfies the requirements")
        if count_pep_nodes == 0:
            print(f'No peptide nodes were found in the association graph. No subgraph will be generated.')
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

            
        self.graph = construct_graph(config=self.config, path=graph_path)
        
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
        
        
        
        
    
        

