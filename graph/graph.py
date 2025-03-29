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
from typing import Callable, Any, Union, Optional, List, Tuple, Dict
from operator import itemgetter
import numpy as np
import logging

log = logging.getLogger("CRSProtein")


class Graph:
    """Represents a protein structure graph."""

    def __init__(self, graph_path: str, config: Optional[ProteinGraphConfig] = None):
        """
        Initialize a Graph instance.

        :param graph_path: Path to the PDB file.
        :param config: Custom configuration for the protein graph.
        """
        self.config = config or ProteinGraphConfig(
            edge_construction_functions=[partial(add_distance_threshold, long_interaction_threshold=0, threshold=10)],
            graph_metadata_functions=[rsa, secondary_structure],
            dssp_config=DSSPConfig(),
            granularity="centroids"
        )
        self.graph = construct_graph(config=self.config, path=graph_path)
        self.subgraphs: Dict[str, nx.Graph] = {}

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

class AssociatedGraph:
    """Handles the association of multiple protein graphs."""
    
    def __init__(self, 
                graphs: List[Tuple],  
                reference_graph: Optional[Union[str, int]] = None,
                output_path: str = ".",
                run_name: str = "",
                association_mode: str = "identity",
                association_config: Optional[Dict[str, Any]] = None):
        """
        Initialize an AssociatedGraph instance with a reduced configuration.
        
        :param graphs: List of tuples (Graph instance, raw_data) from preprocessing.
        :param reference_graph: Identifier for the reference graph.
        :param output_path: Where to save results.
        :param run_name: Unique run identifier.
        :param association_mode: "identity" or "similarity".
        :param association_config: Dictionary with keys:
                - centroid_threshold (float)
                - neighbor_similarity_cutoff (float)
                - rsa_similarity_threshold (float)
                - depth_similarity_threshold (float)
                - residues_similarity_cutoff (float)
                - angle_diff (float)
                - checks (dict)
                - factors_path (str or None)
        """
        default_config = {
            "centroid_threshold": 10.0,
            "neighbor_similarity_cutoff": 0.95,
            "rsa_similarity_threshold": 0.95,
            "depth_similarity_threshold": 0.95,
            "residues_similarity_cutoff": 0.95,
            "angle_diff": 20.0,
            "checks": {"neighbors": True, "rsa": True, "depth": True},
            "factors_path": None
        }
        self.association_config = default_config.copy()
        if association_config:
            self.association_config.update(association_config)
        
        self.graphs = graphs
        self.reference_graph = reference_graph
        self.output_path = Path(output_path)
        self.run_name = run_name
        
        self.graph_data = self._prepare_graph_data()
        
        result = association_product(graph_data=self.graph_data,
                                    association_mode=association_mode,
                                    config=self.association_config)
        
        if result is not None:
            self.__dict__.update(result)
            self.associated_graphs = result["AssociatedGraph"]
        else:
            self.associated_graphs = None

    def _prepare_graph_data(self) -> List[dict]:
        """
        For each (Graph, raw) tuple, build a dictionary with the necessary data:
            - "graph": The Graph instance.
            - "contact_map": Output of build_contact_map(raw)[0].
            - "residue_map_all": Output of build_contact_map(raw)[2].
            - "rsa": np.array(g.graph["dssp_df"]["rsa"]).
            - "residue_depth": g.residue_depth.
        """
        graph_data = []
        for g, raw in self.graphs:
            contact_map, residue_map, residue_map_all = build_contact_map(raw)
            data = {
                "graph": g,
                "contact_map": contact_map,
                "residue_map": residue_map,
                "residue_map_all": residue_map_all,
                "rsa": np.array(g.graph["dssp_df"]["rsa"]),
                "residue_depth": g.residue_depth
            }
            graph_data.append(data)
        return graph_data
    
    
    def add_spheres(self):
        ...
        
    def draw_graph(self, show = True, save = True):
        if not show and not save:
            log.info("You are not saving or viewing the graph. Please leave at least one of the parameters as true.")
            return 
        if isinstance(self.associated_graphs, list):
            for i, graph in enumerate(self.associated_graphs):

                node_colors = [graph.nodes[node]['chain_id'] for node in graph.nodes]
                # Draw the full cross-reactive subgraph
                nx.draw(graph, with_labels=True, node_color=node_colors, node_size=50, font_size=6)
                
                if show:
                    plt.show()
                if save:
                    if i == 0:
                        plt.savefig(path.join(self.output_path, "Associated Graph Base.png"))
                    else:

                        plt.savefig(path.join(self.output_path, f"Associated Graph {i}.png"))
                    plt.clf()
        
                    log.info(f"{i} GraphAssociated's plot saved in {self.output_path}")
        else:
            log.warning(f"I cant draw the graph because it's {self.associated_graphs}")
  
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
