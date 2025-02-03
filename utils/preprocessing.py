from graph.graph import Graph
import logging
import json
from pathlib import Path
from io_utils.pdb_io import list_pdb_files, get_user_selection
from config.parse_configs import make_graph_config
from classes.classes import StructureSERD
from collections import defaultdict
from os import path

log = logging.getLogger("Preprocessing")

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
            log.warning(f"`chains` key must have a list. I received: {params['chains']}, {type(params['chains'])}")
            exit(0)
        
        graph.create_subgraph(name= "selected_chains", chains=params["chains"])
        merge_graphs.append("selected_chains")
        
    if "residues" in params.keys():
        if not isinstance(params["residues"], dict):
            log.warning(f"`dict` key must have a list. I received: {params['dict']}, {type(params['dict'])}")
            exit(0)
        
        residues_dict = params["residues"]
        
        for chain in residues_dict:
            if not isinstance(residues_dict[chain], list):
                log.warning(f"The value {residues_dict[chain]} of chain '{chain}' in residues isn't a list")
                exit(0)
            
            all_residues += residues_dict[chain]
            
        graph.create_subgraph(name="all_residues", sequence_positions=all_residues)
        merge_graphs.append("all_residues")
    
    graph.join_subgraph(name="merge_list", graphs_name=merge_graphs)
    
    exposed_nodes = graph.filter_subgraph(subgraph_name="exposed_residues", name="selected_exposed_residues", filter_func= lambda i: i in graph.subgraphs["merge_list"], return_node_list=True)
    graph.create_subgraph(name = "s_graph", node_list=exposed_nodes, return_node_list=False)
    
    return graph.get_subgraph("s_graph")

def calculate_residue_depth(mol_path, serd_config=None):
    config_serd = defaultdict(lambda: None, {
        "vdw": None,
        "step": 0.6,
        "probe": 1.4,
        "type": "SES",
        "keep_only_interface": True,
        "ignore_backbone": True,
        "metric": "minimum"
    })

    if serd_config:
        with open(serd_config, "r") as f:
            config_serd.update(json.load(f))

    structure = StructureSERD(vdw=config_serd["vdw"])
    structure.load(mol_path)
    structure.model_surface(
        type=config_serd["type"], step=config_serd["step"], probe=config_serd["probe"]
    )
    residue_depth = structure.residue_depth(
                metric=config_serd["metric"],
                keep_only_interface=config_serd["keep_only_interface"],
                ignore_backbone=config_serd["ignore_backbone"],
            )
    
    return residue_depth


def create_graphs(args):
    with open(args.residues_lists, "r") as f:
        residues_lists = json.load(f) 

    if args.folder_path and not args.files_name:
        mols_files = list_pdb_files(args.folder_path)
        if not mols_files:
            raise Exception(f"I didn't find any file in: {args.folder_path}")
            
        selected_files, reference_graph = get_user_selection(mols_files, args.folder_path)
        
    elif args.folder_path and args.files_name:
        files_name = args.files_name.split(',')
        pdb_dir = args.folder_path
        selected_files = []
        reference_graph = None
        for file_name in files_name:
            if args.reference_graph == file_name:
                reference_graph = (path.join(pdb_dir, file_name), file_name)
            else:
                selected_files.append((path.join(pdb_dir, file_name), file_name))

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    config = make_graph_config(centroid_threshold=args.centroid_threshold)
    
    graphs = []
    for mol_path in selected_files:
        g = Graph(config=config, graph_path=mol_path[0])

        residue_depth = calculate_residue_depth(mol_path=mol_path[0], serd_config=args.serd_config)

        params = residues_lists.get(mol_path[1], {})
        if "base" in params:
            params = residues_lists.get(params["base"], {})

        s_g = get_exposed_residues(graph=g, rsa_filter=args.rsa_filter, params=params)
        s_g.residue_depth = residue_depth

        graphs.append((s_g, mol_path[0]))

    return graphs, reference_graph