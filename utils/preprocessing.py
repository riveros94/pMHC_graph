from graph.graph import Graph
import logging
import json
from pathlib import Path
from io_utils.pdb_io import list_pdb_files, get_user_selection
from config.parse_configs import make_graph_config
from classes.classes import StructureSERD
from collections import defaultdict
from os import path
from Bio import PDB
import time
import numpy as np
import networkx as nx
from typing import Tuple, Dict, List

from memory_profiler import profile

logger = logging.getLogger("Preprocessing")


def remove_water_from_pdb(source_file, dest_file):
    """Remove water molecules (HOH) from a PDB file and save the cleaned version."""
    
    if path.exists(dest_file):
        logger.debug(f"The file {dest_file} already exists.")
    else:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", source_file)
        
        io = PDB.PDBIO()
        io.set_structure(structure)
        
        class NoWaterSelect(PDB.Select):
            def accept_residue(self, residue):
                return residue.get_resname() != "HOH"
        
        io.save(dest_file, select=NoWaterSelect())
        logger.debug(f"Saved cleaned PDB file: {dest_file}")

# @profile
def get_exposed_residues(graph: Graph, rsa_filter=0.1, depth_filter=10, selection_params=None) -> nx.Graph:
    selection_params = selection_params or {}
    
    if rsa_filter is None:
        graph.create_subgraph(name="exposed_residues")
    else:
        graph.create_subgraph(name="exposed_residues", rsa_threshold=rsa_filter)
    
    if not any(key in selection_params for key in ("chains", "residues")):
        expo = graph.get_subgraph(name="exposed_residues")
        if expo is not None:
            return expo
        raise Exception("I didn't find any nodes that passes in your filter")        
 
    merge_keys = []
    
    if "chains" in selection_params:
        if not isinstance(selection_params["chains"], list):
            logger.warning(f"`chains` key must have a list. Received: {selection_params['chains']}, {type(selection_params['chains'])}")
            raise ValueError("Invalid chains parameter")
        
        graph.create_subgraph(name="selected_chains", chains=selection_params["chains"])
        merge_keys.append("selected_chains")
    
    if "residues" in selection_params:
        if not isinstance(selection_params["residues"], dict):
            logger.warning(f"`residues` key must be a dictionary. Received: {selection_params['residues']}, {type(selection_params['residues'])}")
            raise ValueError("Invalid residues parameter")
        
        residue_positions = []
        for chain, positions in selection_params["residues"].items():
            if not isinstance(positions, list):
                logger.warning(f"Value for chain '{chain}' in residues must be a list. Received: {positions}")
                raise ValueError("Invalid residues list")
            residue_positions.extend(positions)
        
        graph.create_subgraph(name="all_residues", sequence_positions=residue_positions)
        merge_keys.append("all_residues")
    
    graph.join_subgraph(name="merge_list", graphs_name=merge_keys)
    
    exposed_nodes = graph.filter_subgraph(
        subgraph_name="exposed_residues",
        name="selected_exposed_residues",
        filter_func=lambda node: node in graph.subgraphs["merge_list"],
        return_node_list=True
    )

    if exposed_nodes:
        if depth_filter:
            valid_nodes = graph.depth.loc[graph.depth["ResidueDepth"] <= depth_filter]
            valid_nodes_list = valid_nodes["ResNumberChain"].tolist()

            def isValid(node):
                node_split = node.split(":")
                resChain = str(node_split[2]) + node_split[0] 
 
                if resChain in valid_nodes_list:
                    return True
                return False
        
            exposed_nodes[:] = [node for node in exposed_nodes if isValid(node)]

        graph.create_subgraph(name="s_graph", node_list=exposed_nodes, return_node_list=False)
        
        s_graph = graph.get_subgraph("s_graph")
        if s_graph is not None:        
            return s_graph
    
    raise Exception("I didn't find any nodes that passes in your filter")        

# @profile
def calculate_residue_depth(pdb_file_path, serd_config=None):
    default_config = {
        "vdw": None,
        "step": 0.6,
        "probe": 1.4,
        "type": "SES",
        "keep_only_interface": False,
        "ignore_backbone": False,
        "metric": "minimum"
    }
    
    config = defaultdict(lambda: None, default_config)
    
    if isinstance(serd_config, str):
        with open(serd_config, "r") as f:
            config.update(json.load(f))
    elif isinstance(serd_config, dict):
        config.update(serd_config)
    
    print("Loading structure")
    structure = StructureSERD(vdw=config["vdw"])
    structure.load(pdb_file_path)
     
    print("Creating model surface")
    structure.model_surface(
        type=config["type"],
        step=config["step"],
        probe=config["probe"]
    )
    
    print("Calculating depth")
    depth = structure.residue_depth(
        metric=config["metric"],
        keep_only_interface=config["keep_only_interface"],
        ignore_backbone=config["ignore_backbone"]
    )
    depth_value = depth["ResidueDepth"]
    depth["ResidueDepth"] = (depth_value - np.min(depth_value)) / (np.max(depth_value) - np.min(depth_value))
    
    return depth


def create_graphs(args) -> List[Tuple]:
    with open(args.residues_lists, "r") as f:
        residues_data = json.load(f)
    
    pdb_directory = args.folder_path
    
    if not pdb_directory:
        raise Exception("You must provide the path for PDB folder")
    
    if not args.files_name:
        pdb_files = list_pdb_files(pdb_directory)
        
        # print(pdb_files)
        selected_files = get_user_selection(pdb_files, pdb_directory)
        selected_files = [
            {"input_path": file_pair[0], "name": file_pair[1]} for file_pair in selected_files
        ]
        # if reference_graph:
        #     reference_graph = {"input_path": reference_graph[0], "name": reference_graph[1]}
    else:
        file_names = args.files_name.split(',')
        selected_files = []
        # reference_graph = None
        for fname in file_names:
            file_info = {"input_path": path.join(pdb_directory, fname), "name": fname}
            selected_files.append(file_info)
            # if args.reference_graph == fname:
            #     reference_graph = file_info
            # else:
            #     selected_files.append(file_info)
    
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    graph_config = make_graph_config(centroid_threshold=args.centroid_threshold)
    
    graphs = []
    for file_info in selected_files:
        cleaned_path = path.join(pdb_directory, file_info["name"].replace('.pdb', '_nOH.pdb'))
        
        remove_water_from_pdb(file_info["input_path"], cleaned_path)
        file_info["input_path"] = cleaned_path
        
        graph_instance = Graph(config=graph_config, graph_path=file_info["input_path"])
        print("Before calculating the Depth")    
        start_time = time.time()
        depth = calculate_residue_depth(pdb_file_path=file_info["input_path"], serd_config=args.serd_config)
        logger.debug(f"Depth calculated in {time.time() - start_time} seconds") 

        depth["ResNumberChain"] = depth["ResidueNumber"].astype(str) + depth["Chain"]

        graph_instance.depth = depth

        selection_params = residues_data.get(file_info["name"], {})
        if "base" in selection_params:
            selection_params = residues_data.get(selection_params["base"], {})
        
        subgraph = get_exposed_residues(
            graph=graph_instance,
            rsa_filter=args.rsa_filter,
            depth_filter=args.depth_filter,
            selection_params=selection_params
        )

        subgraph.graph["depth"] = depth
         
        graphs.append((subgraph, file_info["input_path"]))
    
    return graphs #, reference_graph
