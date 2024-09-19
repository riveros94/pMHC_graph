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


class AssociatedGraph:
    def __init__(self, graphs: List[Tuple], output_path: str, path_full_subgraph: str, run_name: str, association_mode: str = "identity",  interface_list: Union[None, list] = None, centroid_threshold: int = 10):
        if interface_list:
            self.interface_list = interface_list
        else:
            self.interface_list = None
        
        self.graphs = graphs
        self.output_path = output_path
        self.run_name = run_name
        self.association_mode = association_mode
        self.centroid_threshold = centroid_threshold
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.path_full_subgraph = path_full_subgraph
        
        self.associated_graph = self._construct_graph(self.graphs, self.association_mode, self.centroid_threshold )
        
    def _build_multiple_contact_map(self, graphs: List[Tuple]) -> List[Tuple]:

        contact_residues_map = []
        for item in graphs:
            contact_map, residue_map, residue_map_all = build_contact_map(item[1])
            contact_residues_map.append((item[0], item[1], contact_map, residue_map, residue_map_all))

        return contact_residues_map
        
    def _gen_associated_graph(self, graphsList: List[Tuple], association_mode: str, centroid_threshold: int):
        
        graphsList = sorted(graphsList, key=lambda x: len(x[0].nodes()))
        
        contact_maps = [graph[2] for graph in graphsList]
    
        graphs = [graph[0] for graph in graphsList]
        residue_maps = [graph[3] for graph in graphsList]
        residue_maps_all = [graph[4] for graph in graphsList]       
 
        """
        def sort_key(node):
            chain, residue, number = node.split(':')
            return (chain, residue, int(number))
        
        sorted_nodes_graphs = [sorted(graph.nodes(), key=sort_key) for graph in graphs]
        """
        
        nodes_graphs = [list(graph.nodes()) for graph in graphs]

        start = time()
        print(f"Vou iniciar o produto cartesiano")
        
        M = association_product(graphs, association_mode=association_mode, nodes_graphs=nodes_graphs, contact_maps=contact_maps, residue_maps_all=residue_maps_all, centroid_threshold=centroid_threshold)
        
        end = time()
        print(f"Tempo para produto cartesiano: {end - start}")
        
        return M
    
    def _construct_graph(self, graphs, association_mode, centroid_threshold):
        # Create a contact map and a list with residue names order as the contact map
        # We do not need to do this, since graphein already build internally the distance matrix considering the centroid
        # This matrix can be obtained from: graph.graph["pdb_df"]
        # The contact matrix can be obtained from: graph.graph["dist_mat"]
        # Usually this dist_mat is not updated in related to the pdb_df after graph subsets, so it is important to double check it 
        # Instead of using the dist_mat we can build it again using compute_distmat(graph.graph["pdb_df"])
        
        
        contact_residues_maps = self._build_multiple_contact_map(graphs)
        
        contact_residues_maps_sorted = sorted(contact_residues_maps, key=itemgetter(1)) 
               
        associated_graph = self._gen_associated_graph(contact_residues_maps_sorted, association_mode, centroid_threshold)

        return associated_graph
    
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
        
        
        
        
    
        

