from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_threshold
from functools import partial
import pandas as pd
import os
from os import path
import argparse
from graph.graph import *
from config.parse_configs import make_graph_config
from io_utils.pdb_io import list_pdb_files, get_user_selection
import logging, sys
import json
import numpy
import matplotlib
from typing import Dict, List, Optional
from classes.classes import Structure
from collections import defaultdict
from cli.cli_parser import parse_args
from utils.preprocessing import create_graphs

'''
#This script takes two pMHC structures as input and returns the common subgraphs, which will also be mapped to the structure

#The input PDB structures must be unbound, without the TCR, only the pMHC structure

#original work of a similar algorithm can be found in Protein−Protein Binding-Sites Prediction by Protein Surface Structure Conservation Janez Konc and Dušanka Janežič, 2006

#Graphein was tested inside a conda enviroment

#Graphein was not computing correctly the centoids when multiple chains were included, I correction I made was replace
#["residue_number", "chain_id", "residue_name", "insertion"] by
#["chain_id", "residue_number", "residue_name", "insertion"] in anaconda3/envs/graphein/lib/python3.8/site-packages/graphein/protein/graphs.py

#I found some incompabilities when running dssp with graphein, so I had to made a correction in my graphein version:
#In anaconda3/envs/graphein/lib/python3.8/site-packages/graphein/protein/features/nodes/dssp.py
#I changed #pdb_file, DSSP=executable, dssp_version=dssp_version to 
#pdb_file, DSSP=executable
'''

#command example
### functions

def main():
    args = parse_args()
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if args.debug else logging.INFO)
    log = logging.getLogger("CRSProtein")
    log.setLevel(logging.DEBUG if args.debug else logging.INFO)

    graphs, reference_graph= create_graphs(args)

    G = AssociatedGraph(
        graphs=graphs,
        reference_graph=reference_graph,
        output_path=args.output_path,
        path_full_subgraph=path.join(args.output_path, f"full_association_graph_{args.run_name}.png"),
        association_mode=args.association_mode,
        factors_path=args.factors_path,
        run_name=args.run_name,
        centroid_threshold=args.centroid_threshold,
        residues_similarity_cutoff=args.residues_similarity_cutoff,
        neighbor_similarity_cutoff=args.neighbor_similarity_cutoff,
        rsa_similarity_threshold=args.rsa_similarity_threshold,
        depth_similarity_threshold=args.depth_similarity_threshold,
        angle_diff=args.angle_diff,
    )

    if G.associated_graph is None:
        return
    
    G.draw_graph(show = True)

    try:
        G.grow_subgraph_bfs()
    except Exception as e: 
        log.error(f"I wasn't able to grow subgraphs with bfs. Error message:\n{e}")

if __name__ == "__main__":

    # Verify the number of threads OpenMP
    num_threads = os.environ.get('OMP_NUM_THREADS', None)
    if num_threads:
        log.info(f"Using {num_threads} threads OpenMP.")
    else:
        log.info("The variable `OMP_NUM_THREADS` it's not defined.")

    main()
    