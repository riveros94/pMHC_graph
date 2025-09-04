import json
import logging
import os
import sys
from os import path
from cli.cli_parser import parse_args
from graph.graph import AssociatedGraph
from utils.preprocessing import create_graphs
from typing import Dict, Any

from core.tracking import init_tracker

def load_manifest(manifest_path: str) -> Dict[str, Any]:
    if not manifest_path:
        return {}

    with open(manifest_path, "r") as f:
        data = json.load(f)

    data.setdefault("settings", {})
    data.setdefault("inputs", [])
    data.setdefault("constrains", {})

    S = data["settings"]
    S.setdefault("run_name", "test")
    S.setdefault("output_path", "./outputs")
    S.setdefault("debug", False)
    S.setdefault("track_steps", False)

    S.setdefault("centroid_threshold", 8.5)
    S.setdefault("centroid_granularity", "all_atoms")
    S.setdefault("exclude_waters", True)

    S.setdefault("check_rsa", True)
    S.setdefault("check_depth", True)

    S.setdefault("rsa_filter", 0.1)
    S.setdefault("depth_filter", 10.0)

    S.setdefault("distance_diff_threshold", 2.0)

    S.setdefault("rsa_bins", 5)
    S.setdefault("depth_bins", 5)
    S.setdefault("distance_bins", 5)

    S.setdefault("serd_config", None)

    S.setdefault("classes", {})


    return data

def main():
    # 1) CLI mínima: só --manifest
    args = parse_args()  # parse_args() agora só tem --manifest

    # 2) Carrega manifest unificado
    manifest = load_manifest(args.manifest)
    S = manifest["settings"]  # atalhos
    run_name = S["run_name"]
    output_path = S["output_path"]

    # 3) Tracker e logging baseados em settings
    init_tracker(
        root="CrossSteps",
        outdir=run_name,
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

    # 4) Checks
    checks = {
        "depth": S.get("check_depth"),
        "rsa":   S.get("check_rsa"),
    }

    # 5) Constrói grafos a partir do manifest
    graphs = create_graphs(manifest)

    # 6) Config de associação vinda de settings
    association_config = {
        "centroid_threshold":          S.get("centroid_threshold"),
        "distance_diff_threshold":     S.get("distance_diff_threshold"),
        "rsa_filter":                  S.get("rsa_filter"),
        "depth_filter":                S.get("depth_filter"),
        "rsa_bins":                    S.get("rsa_bins"),
        "depth_bins":                  S.get("depth_bins"),
        "distance_bins":               S.get("distance_bins"),
        "checks":                      checks,
        "exclude_waters":              S.get("exclude_waters"),
        "classes": S.get("classes", {}),
    }

    # 7) Roda a associação
    G = AssociatedGraph(
        graphs=graphs,
        output_path=output_path,
        run_name=run_name,
        association_config=association_config,
    )

    if G.associated_graphs is None:
        return

    log.debug("Drawing Graph")
    G.draw_graph_interactive(show=False, save=True)
    G.draw_graph(show=False, save=True)

    # G.create_pdb_per_protein()
    # G.align_all_frames()


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


if __name__ == "__main__":
    log = logging.getLogger("CRSProtein")
    num_threads = os.environ.get('OMP_NUM_THREADS', None)
    if num_threads:
        log.info(f"Using {num_threads} threads OpenMP.")
    else:
        log.info("The variable `OMP_NUM_THREADS` is not defined.")
    main()
