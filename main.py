import json
import logging
import os
import shutil
import sys
from os import path
from cli.cli_parser import parse_args
from graph.graph import AssociatedGraph
from utils.preprocessing import create_graphs
from itertools import combinations
from pathlib import Path
from typing import Dict, Any

from core.tracking import init_tracker

def load_manifest(manifest_path: str) -> Dict[str, Any]:
    if not manifest_path:
        return {}

    with open(manifest_path, "r") as f:
        data = json.load(f)

    data.setdefault("settings", {})
    data.setdefault("inputs", [])
    data.setdefault("selectors", {})

    S = data["settings"]
    S.setdefault("run_name", "test")
    S.setdefault("run_mode", "all")
    S.setdefault("output_path", "./outputs")

    os.makedirs(S["output_path"], exist_ok=True)
    
    shutil.copy2(manifest_path, S["output_path"]+"/manifest.json")
    S.setdefault("debug", False)
    S.setdefault("track_steps", False)
    S.setdefault("rsa_table", "Wilke")

    S.setdefault("edge_threshold", 8.5)
    S.setdefault("close_tolerance", 1.0)
    S.setdefault("node_granularity", "all_atoms")
    S.setdefault("exclude_waters", True)

    S.setdefault("triad_rsa", False)
    S.setdefault("check_depth", False)

    S.setdefault("rsa_filter", 0.1)
    S.setdefault("depth_filter", 10.0)

    S.setdefault("distance_std_threshold", 3.0)
    S.setdefault("distance_diff_threshold", 1.0)

    S.setdefault("rsa_bin_width", 0.2)
    S.setdefault("distance_bin_width", 2.0)
    S.setdefault("depth_bins", 5)

    S.setdefault("dynamic_distance_classes", False)
    S.setdefault("distance_bins", 5)
    # S.setdefault("distance_bins", 5)

    S.setdefault("serd_config", None)
    S.setdefault("max_chunks", 5)

    S.setdefault("filter_triads_by_chain", None)
    S.setdefault("classes", {})


    return data

def run_association_task(graphs, output_path, run_name, association_config, log):
    """
    Helper function to run the association logic for a specific set of graphs
    and a specific output directory.
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    task_config = association_config.copy()
    task_config["output_path"] = str(output_path)

    log.info(f"Starting run '{run_name}' in mode: {task_config['run_mode']} with {len(graphs)} graphs.")
    log.info(f"Output directory: {output_path}")

    G = AssociatedGraph(
        graphs=graphs,
        output_path=str(output_path),
        run_name=run_name,
        association_config=task_config,
    )

    if G.associated_graphs is None:
        log.warning(f"No associated graphs found for {run_name}.")
        return

    log.debug(f"Drawing Graph for {run_name}")
    G.draw_graph_interactive(show=False, save=True)
    # G.draw_graph(show=False, save=True)
    
    G.create_pdb_per_protein()
    G.align_all_frames()

    # log.debug("Growing Subgraph")
    # try:
    #     G.grow_subgraph_bfs()
    # except Exception as e:
    #     log.error(f"Unable to grow subgraphs with BFS. Error: {e}")

    graph_data = dict()
    for j, comps in enumerate(G.associated_graphs):
        graph_data[j] = {"comp": j, "frames": {}}
        for i in range(len(comps[0])):
            nodes = list(comps[0][i].nodes)
            edges = list(comps[0][i].edges)
            neighbors = {
                str(node): [str(neighbor) for neighbor in comps[0][i].neighbors(node)]
                for node in nodes
            }
            graph_data[j]["frames"][i] = {
                "nodes": nodes,
                "edges": edges,
                "neighbors": neighbors
            }

    output_json = path.join(output_path, f"graph_{run_name}.json")
    with open(output_json, "w") as f:
        json.dump(graph_data, f, indent=4)
    log.info(f"Graph data saved to {output_json}")

def main():
    args = parse_args() 
    manifest = load_manifest(args.manifest)
    S = manifest["settings"] 
    
    base_run_name = S["run_name"]
    base_output_path = S["output_path"]
    run_mode = S.get("run_mode", "all") 

    init_tracker(
        root="CrossSteps",
        outdir=base_run_name,
        enabled=S.get("track_steps", False),
        prefer_npy_for_ndarray=True,
        add_timestamp_prefix=False,
    )

    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if S.get("debug", False) else logging.INFO
    )
    log = logging.getLogger("CRSProtein")
    log.setLevel(logging.DEBUG if S.get("debug", False) else logging.INFO)

    checks = {
        "depth": S.get("check_depth"),
        "rsa":   S.get("triad_rsa"),
    }

    # Load all graphs once
    graphs = create_graphs(manifest)

    base_association_config = {
        "run_mode":                 run_mode,
        "edge_threshold":           S.get("edge_threshold"),
        "distance_std_threshold":   S.get("distance_std_threshold"),
        "distance_diff_threshold":  S.get("distance_diff_threshold"),
        "rsa_filter":               S.get("rsa_filter"),
        "depth_filter":             S.get("depth_filter"),
        "rsa_bin_width":            S.get("rsa_bin_width"),
        "depth_bins":               S.get("depth_bins"),
        "distance_bin_width":       S.get("distance_bin_width"),
        "close_tolerance":          S.get("close_tolerance"),
        "checks":                   checks,
        "exclude_waters":           S.get("exclude_waters"),
        "classes":                  S.get("classes", {}),
        "max_chunks":               S.get("max_chunks"),
        "rsa_table":                S.get("rsa_table", "Wilke"),
        "dynamic_distance_classes": S.get("dynamic_distance_classes", False),
        "distance_bins":            S.get("distance_bins", 5),
        "filter_triads_by_chain":   S.get("filter_triads_by_chain", None),
        # "output_path": passed dynamically
    }

    if run_mode == "all":
        target_dir = path.join(base_output_path, "ALL")
        run_association_task(
            graphs=graphs,
            output_path=target_dir,
            run_name=base_run_name, # Keep original name or modify if preferred
            association_config=base_association_config,
            log=log
        )

    elif run_mode == "pair":
        pair_base_dir = path.join(base_output_path, "PAIR")
        
        for g1, g2 in combinations(graphs, 2):
            name1 = Path(g1[1]).stem
            name2 = Path(g2[1]).stem
            
            name1 = name1.replace("_nOH", "")
            name2 = name2.replace("_nOH", "")

            pair_folder_name = f"{name1}_vs_{name2}"
            target_dir = path.join(pair_base_dir, pair_folder_name)
            
            pair_run_name = f"{base_run_name}_{name1}_{name2}"

            run_association_task(
                graphs=[g1, g2],
                output_path=target_dir,
                run_name=pair_run_name,
                association_config=base_association_config,
                log=log
            )
            
    else:
        log.error(f"Unknown run_mode: {run_mode}. Please use 'all' or 'pair'.")

if __name__ == "__main__":
    log = logging.getLogger("CRSProtein")
    num_threads = os.environ.get('OMP_NUM_THREADS', None)
    if num_threads:
        log.info(f"Using {num_threads} threads OpenMP.")
    else:
        log.info("The variable `OMP_NUM_THREADS` is not defined.")
    main()
