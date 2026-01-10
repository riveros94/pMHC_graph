from graph.graph import Graph
import logging
import json
from pathlib import Path
from io_utils.pdb_io import list_pdb_files, get_user_selection
from collections import defaultdict
from os import path
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO
import time
import numpy as np
import networkx as nx
from typing import Tuple, List, Dict, Optional, Any, Union
from memory_profiler import profile
from core.config import make_default_config
from core.tracking import save
import gemmi
import os

logger = logging.getLogger("Preprocessing")


def remove_water_from_pdb(source_file, dest_file):
    """Remove water molecules from a PDB or mmCIF file and save the cleaned version safely."""
    
    if os.path.exists(dest_file):
        logger.debug(f"The file {dest_file} already exists.")
        return

    suffix = source_file.lower()
    is_cif = suffix.endswith((".cif", ".mmcif", ".mcif"))

    st = gemmi.read_structure(source_file)

    for model in st:
        model.remove_waters()

    if is_cif:
        doc = st.make_mmcif_document()
        with open(dest_file, "w") as f:
            f.write(doc.as_string())
    else:
        st.write_pdb(dest_file)

    logger.debug(f"Saved cleaned structure without waters: {dest_file}")
    
def get_exposed_residues(graph: Graph, rsa_filter=0.1, depth_filter: Union[float, None]=10.0, selection_params=None) -> nx.Graph:
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

def _infer_is_dir_or_file(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if p.exists():
        return "dir" if p.is_dir() else "file"
    typical_exts = [".pdb", ".pdb.gz", ".cif", ".ent", ".mmcif"]
    return "file" if any(path_str.endswith(ext) for ext in typical_exts) else "dir"

def list_struct_files(folder: Path, extensions: List[str]) -> List[Path]:
    exts = set(extensions or [".pdb", ".pdb.gz", ".cif"])
    files: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file():
            for ext in exts:
                if str(p.name).endswith(ext):
                    files.append(p)
                    break
    files.sort()
    return files

def collect_selected_files_from_manifest(manifest):
    results = []

    for rule in manifest.get("inputs", []):
        path_val = rule.get("path")
        if not path_val:
            continue

        paths = path_val.split(",") if isinstance(path_val, str) else list(path_val)
        extensions = rule.get("extensions") or [".pdb", ".pdb.gz", ".cif"]
        enable_tui = bool(rule.get("enable_tui", False))

        for p_str in paths:
            kind = _infer_is_dir_or_file(p_str)
            p = Path(p_str).expanduser().resolve()

            if kind == "file":
                results.append({"input_path": str(p), "name": p.name})
            else:
                if enable_tui:
                    # TUI no MESMO formato que você usava
                    names = list_pdb_files(str(p), extensions=extensions)
                    selected = get_user_selection(names, str(p))
                    for full_path, fname in selected:
                        results.append({"input_path": str(Path(full_path).resolve()),
                                        "name": fname})
                else:
                    # modo não interativo: pega todos do diretório (sem recursão, estilo original)
                    if not p.exists() or not p.is_dir():
                        continue
                    for fname in sorted(os.listdir(p)):
                        if any(fname.endswith(ext) for ext in extensions):
                            full = (p / fname).resolve()
                            results.append({"input_path": str(full), "name": fname})

    # dedup
    seen = set()
    out = []
    for it in results:
        if it["input_path"] not in seen:
            seen.add(it["input_path"])
            out.append(it)
    return out

def resolve_selection_params_for_file(file_path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    if not manifest:
        return {}
    fname = file_path.name
    fpath_str = str(file_path.resolve())
    merged: Dict[str, Any] = {}
    for rule in manifest.get("inputs", []):
        paths = rule.get("path")
        if not paths:
            continue
        if isinstance(paths, str):
            paths = paths.split(",")

        hit_path = False
        for p in paths:
            p_obj = Path(p).expanduser()
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

        for c in (rule.get("selectors") or []):
            name = c.get("name")
            if not name:
                continue
            if not _name_contains(fname, c.get("file_name_contains")):
                continue
            spec = manifest.get("selectors", {}).get(name, {})
            merged = _merge_constraints(merged, spec)
    return merged

def create_graphs(manifest: Dict) -> List[Tuple]:

    S = manifest["settings"]

    output_path = Path(S["output_path"]).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    selected_files = collect_selected_files_from_manifest(manifest)
    if not selected_files:
        raise Exception("Nenhum arquivo selecionado a partir do manifest")

    graph_config = make_default_config(
        edge_threshold=S["edge_threshold"],
        granularity=S["node_granularity"],
        exclude_waters=S["exclude_waters"],
        dssp_acc_array=S["rsa_table"]
    )

    graphs: List[Tuple] = []
    start = time.perf_counter()
    for file_info in selected_files:
        orig_path = Path(file_info["input_path"]).resolve()
        if S["exclude_waters"]:
            cleaned_name = file_info["name"]
            if cleaned_name.endswith(".pdb.gz"):
                cleaned_name = cleaned_name[:-7] + "_nOH.pdb"
            elif cleaned_name.endswith(".pdb"):
                cleaned_name = cleaned_name[:-4] + "_nOH.pdb"
            elif cleaned_name.endswith(".cif"):
                cleaned_name = cleaned_name[:-4] + "_nOH.cif"
            else:
                cleaned_name = cleaned_name + "_nOH.pdb"
            cleaned_path = (orig_path.parent / cleaned_name).resolve()
            remove_water_from_pdb(str(orig_path), str(cleaned_path))
            graph_path = cleaned_path
        else:
            graph_path = orig_path

        graph_instance = Graph(config=graph_config, graph_path=str(graph_path))

        depth = None
        selection_params = resolve_selection_params_for_file(orig_path, manifest)

        subgraph = get_exposed_residues(
            graph=graph_instance,
            rsa_filter=S.get("rsa_filter"),
            depth_filter=S.get("depth_filter") if S.get("depth_check") else None,
            selection_params=selection_params or {},
        )

        subgraph.graph["depth"] = depth

        save("create_graphs", f"{graph_path.stem}_subgraph", subgraph)
        graphs.append((subgraph, str(orig_path)))
    end = time.perf_counter()

    logger.debug(f"Took {end - start:.6f} seconds to create graphs")
    return graphs
