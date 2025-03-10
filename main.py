import argparse
import json
import logging
import os
import sys
from os import path

from cli.cli_parser import parse_args
from graph.graph import AssociatedGraph
from utils.preprocessing import create_graphs


def main():
    args = parse_args()

    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.debug else logging.INFO
    )
    log = logging.getLogger("CRSProtein")
    log.setLevel(logging.DEBUG if args.debug else logging.INFO)

    checks = {
        "depth": args.check_depth,
        "rsa": args.check_rsa,
        "angles": args.check_angles,
        "neighbors": args.check_neighbors
    }

    graphs, reference_graph = create_graphs(args)

    association_config = {
        "centroid_threshold": args.centroid_threshold,
        "neighbor_similarity_cutoff": args.neighbor_similarity_cutoff,
        "rsa_similarity_threshold": args.rsa_similarity_threshold,
        "depth_similarity_threshold": args.depth_similarity_threshold,
        "residues_similarity_cutoff": args.residues_similarity_cutoff,
        "distance_diff_threshold": args.distance_diff_threshold,
        "angle_diff": args.angle_diff,
        "checks": checks,
        "factors_path": args.factors_path
    }

    G = AssociatedGraph(
        graphs=graphs,
        reference_graph=reference_graph,
        output_path=args.output_path,
        run_name=args.run_name,
        association_mode=args.association_mode,
        association_config=association_config
    )

    if G.associated_graph is None:
        return

    log.debug("Drawing Graph")
    G.draw_graph(show=True)

    log.debug("Growing Subgraph")
    try:
        G.grow_subgraph_bfs()
    except Exception as e:
        log.error(f"Unable to grow subgraphs with BFS. Error: {e}")

    # Prepare and save the output graph data as JSON.
    graph_data = {
        "nodes": [str(node) for node in G.associated_graph.nodes],
        "edges": [(str(u), str(v)) for u, v in G.associated_graph.edges],
        "neighbors": {
            str(node): [str(neighbor) for neighbor in G.associated_graph.neighbors(node)]
            for node in G.associated_graph.nodes
        }
    }

    output_json = path.join(args.output_path, f"graph_{args.run_name}.json")
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
