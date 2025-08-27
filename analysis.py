import json
import logging
import os
import sys
from os import path
import warnings
warnings.filterwarnings("ignore")
from cli.cli_parser import parse_args
from graph.graph import AssociatedGraph
from utils.preprocessing import create_graphs
import pandas as pd

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
 
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

args = dict2obj({
        "folder_path": "Analysis/selected_strs_renumber/without_TCR",
        "residues_lists": "jsons/residues_lists.json",
        "serd_config": None,
        "files_name": None,
        "output_path": None,
        "run_name": None,
        "check_depth": False,
        "check_rsa": True,
        "centroid_threshold": 8.5,
        "distance_diff_threshold": 3,
        "depth_filter": None,
        "depth_bins": 3,
        "rsa_filter": 0.1,
        "rsa_bins": 3,
        "distance_bins": 3,
        "factors_path": "resources/atchley_aa.csv",
        "classes_path": None})

checks = {
        "depth": args.check_depth,
        "rsa": args.check_rsa,
}

association_config = {
    "centroid_threshold": args.centroid_threshold,
    "distance_diff_threshold": args.distance_diff_threshold,
    "rsa_filter": args.rsa_filter,
    "depth_filter": args.depth_filter,
    "rsa_bins": args.rsa_bins,
    "depth_bins": args.depth_bins,
    "distance_bins": args.distance_bins,
    "checks": checks,
    "factors_path": args.factors_path,
    "classes_path": args.classes_path
}

diffMHC_diffPep = pd.read_csv("Analysis/crossreact_processed_diff_MHC_diff_pep_helder.csv")
diffMHC_SamePep = pd.read_csv("Analysis/crossreact_processed_diff_MHC_same_pep_helder.csv")
sameMHC_diffPep = pd.read_csv("Analysis/crossreact_processed_same_MHC_diff_pep_helder.csv")
rawCross = pd.read_csv("Analysis/crossreact_tcrs_dedup_noncanonicalclassv2_samuel.csv")

crossDf = rawCross[["TCR_ID", "TCR_pair_id", "MHC_allele_id", "peptide_id", "pMHC_id", "Reference"]]

TCR_pair_id = crossDf.groupby('TCR_pair_id')
Graphs = {}
for pair_id, group in TCR_pair_id:
    print(f"\n--- TCR_pair_id: {pair_id} ---")
    args.files_name = ",".join([f"noTCR_{name.lower()}.trunc.fit_renum.pdb" for name in group["Reference"]])
    args.run_name = pair_id
    args.output_path = f"Analysis/CrossGraphs/{pair_id}/All"

    graphs = create_graphs(args)

    G = AssociatedGraph(
        graphs=graphs,
        output_path=args.output_path,
        run_name=args.run_name,
        association_config=association_config
    )
    G.draw_graph(show=False, save=True)
    Graphs[pair_id] = G
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

    output_json = path.join(args.output_path, f"graph_{args.run_name}.json")
    with open(output_json, "w") as f:
        json.dump(graph_data, f, indent=4)
    print(f"Graph data saved to {output_json}")

