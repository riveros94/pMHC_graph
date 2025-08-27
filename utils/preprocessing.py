from graph.graph import Graph
import logging
import json
from pathlib import Path
from io_utils.pdb_io import list_pdb_files, get_user_selection
from SERD_Addon.classes import StructureSERD
from collections import defaultdict
from os import path
from Bio import PDB
import time
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional, Any, Union
from memory_profiler import profile
from core.config import make_default_config
from core.tracking import save

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

def get_exposed_residues(graph: Graph, rsa_filter=0.1, depth_filter: Union[float, None]=10.0, selection_params=None) -> nx.Graph:
    selection_params = selection_params or {}
    
    if rsa_filter is None:
        graph.create_subgraph(name="exposed_residues")
    else:
        graph.create_subgraph(name="exposed_residues", rsa_threshold=rsa_filter)
    
    # print(f"Neighbors after rsa filter: {[node for node in graph.get_subgraph(name='exposed_residues').neighbors('C:ASP:4')]}")
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

def _is_within(child: Path, parent: Path) -> bool:
    """True se child está dentro de parent (ou igual)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

def _name_contains(fname: str, cond: Any) -> bool:
    """Suporta string ou lista de strings para 'file_name_contains'."""
    if cond is None:
        return True
    if isinstance(cond, str):
        return cond in fname
    try:
        return any(s in fname for s in cond)
    except Exception:
        return False

def _merge_constraints(base: Dict[str, Any], add: Dict[str, Any]) -> Dict[str, Any]:
    """Une listas de cadeias e posições de resíduos."""
    out: Dict[str, Any] = {"chains": [], "residues": {}}
    for src in (base or {}, add or {}):
        # chains
        chains = src.get("chains") or []
        out["chains"].extend([c for c in chains if c not in out["chains"]])
        # residues
        residues = src.get("residues") or {}
        for ch, positions in residues.items():
            lst = out["residues"].setdefault(ch, [])
            for p in positions:
                if p not in lst:
                    lst.append(p)
    return out

def load_manifest(manifest_path: Optional[str]) -> Dict[str, Any]:
    if not manifest_path:
        return {}
    with open(manifest_path, "r") as f:
        data = json.load(f)

    data.setdefault("inputs", [])
    data.setdefault("constrains", {})
    return data

def resolve_selection_params_for_file(
    file_path: Path, manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Encontra e mescla todos os 'constrains' aplicáveis a file_path,
    com base em regras definidas em manifest['inputs'].
    """
    if not manifest:
        return {}

    fname = file_path.name
    fpath_str = str(file_path.resolve())
    merged: Dict[str, Any] = {}

    for rule in manifest.get("inputs", []):
        paths: List[str] = rule.get("path") or []
        hit_path = False
        for p in paths:
            p_obj = Path(p)
            if any(ch in p for ch in "*?[]"): 
                if fnmatch.fnmatch(fpath_str, str(p)):
                    hit_path = True
                    break
            elif p_obj.exists() and p_obj.is_dir():
                if _is_within(file_path, p_obj):
                    hit_path = True
                    break
            else:
                try:
                    if file_path.resolve() == p_obj.resolve():
                        hit_path = True
                        break
                except Exception:
                    if str(p) in fpath_str:
                        hit_path = True
                        break

        if not hit_path:
            continue

        for c in (rule.get("constrains") or []):
            name = c.get("name")
            if not name:
                continue
            if not _name_contains(fname, c.get("file_name_contains")):
                continue
            spec = manifest.get("constrains", {}).get(name, {})
            merged = _merge_constraints(merged, spec)

    return merged


# ---------------------- create_graphs (mantendo sua seleção) ----------------------

def create_graphs(args) -> List[Tuple]:
    # Load the manifest for dealing with files
    manifest = load_manifest(getattr(args, "manifest", None))

    pdb_directory = args.folder_path
    if not pdb_directory:
        raise Exception("You must provide the path for PDB folder")

    if not args.files_name:
        pdb_files = list_pdb_files(pdb_directory)
        selected_files = get_user_selection(pdb_files, pdb_directory)
  
        selected_files = [
            {"input_path": pair[0], "name": pair[1]} for pair in selected_files
        ]
    else:
        file_names = args.files_name.split(",")
        selected_files = [
            {"input_path": str(Path(pdb_directory) / fname), "name": fname}
            for fname in file_names
        ]

    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    graph_config = make_default_config(
        centroid_threshold=args.centroid_threshold,
        granularity=args.centroid_granularity,
        exclude_waters=args.exclude_waters,
    )

    graphs: List[Tuple] = []
    i = 1
    for file_info in selected_files:
        orig_path = Path(file_info["input_path"]).resolve()
        graph_path: Path

        if args.exclude_waters:
            cleaned_name = file_info["name"]
            if cleaned_name.endswith(".pdb.gz"):
                cleaned_name = cleaned_name[:-7] + "_nOH.pdb"
            elif cleaned_name.endswith(".pdb"):
                cleaned_name = cleaned_name[:-4] + "_nOH.pdb"
            else:
                cleaned_name = cleaned_name + "_nOH.pdb"
            cleaned_path = (Path(args.folder_path or ".") / cleaned_name).resolve()
            remove_water_from_pdb(str(orig_path), str(cleaned_path))
            graph_path = cleaned_path
        else:
            graph_path = orig_path

        graph_instance = Graph(config=graph_config, graph_path=str(graph_path))

        if args.check_depth:
            start_time = time.time()
            depth = calculate_residue_depth(
                pdb_file_path=str(graph_path),
                serd_config=args.serd_config,
            )
            logger.debug(f"Depth calculated in {time.time() - start_time} seconds")
            depth["ResNumberChain"] = depth["ResidueNumber"].astype(str) + depth["Chain"]
            graph_instance.depth = depth
        else:
            args.depth_filter = None
            depth = None

        # Resolve constraints
        selection_params = resolve_selection_params_for_file(orig_path, manifest)

        # Extrai subgrafo (expostos + constraints resolvidos)
        subgraph = get_exposed_residues(
            graph=graph_instance,
            rsa_filter=args.rsa_filter,
            depth_filter=args.depth_filter,
            selection_params=selection_params or {},  # sempre dict
        )

        subgraph.graph["depth"] = depth

        save("create_graphs", f"{graph_path.stem}_subgraph", subgraph)

        graphs.append((subgraph, str(orig_path)))

    return graphs