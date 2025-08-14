# core/pipeline.py
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional
from pathlib import Path
import pandas as pd
import networkx as nx

# Importa o builder que criei anteriormente
from core.pdb_graph_builder import PDBGraphBuilder, GraphBuildConfig
from core.config import ProteinGraphConfig

def build_graph_with_config(pdb_path: str, config: ProteinGraphConfig) -> nx.Graph:
    builder_cfg = GraphBuildConfig(
        chains=config.chains,
        include_waters=config.include_waters,
        residue_distance_cutoff=config.residue_distance_cutoff,
        water_distance_cutoff=config.water_distance_cutoff,
        compute_rsa=config.compute_rsa,
        store_distance_matrix=True,        # para ter NxN
        rsa_method="dssp",                 # <<< usar DSSP como no Graphein
        dssp_exec="mkdssp",                # ou "dssp" no seu ambiente
        dssp_acc_array="Sander",           # ou "Wilke" / "Miller"
    )
    builder = PDBGraphBuilder(pdb_path, builder_cfg)
    built = builder.build_graph()
    G = built.graph

    # contexto para metadados
    residue_map = {nid: res for nid, res in built.residue_index}
    ctx = {
        "pdb_path": pdb_path,
        "structure": builder.structure,
        "residue_map": residue_map,
        "dssp_config": config.dssp_config,
        "config": config,
    }

    for fn in (config.edge_construction_functions or []):
        G = fn(G, **ctx)
    for fn in (config.node_metadata_functions or []):
        G = fn(G, **ctx)
    for fn in (config.edge_metadata_functions or []):
        G = fn(G, **ctx)
    for fn in (config.graph_metadata_functions or []):
        G = fn(G, **ctx)

    # anexa artefatos úteis ao grafo
    G.graph["residue_labels"]    = [nid for nid, _ in built.residue_index]
    G.graph["water_labels"]      = [nid for nid, _ in built.water_index]
    G.graph["distance_matrix"]   = built.distance_matrix          # (R, R)
    G.graph["coords"] = built.residue_centroids        # (R, 3)
    G.graph["water_positions"]   = built.water_centroids          # (W, 3) ou None
    G.graph["raw_pdb_df"] = built.raw_pdb_df
    G.graph["pdb_df"]     = built.pdb_df
    G.graph["rgroup_df"]  = built.rgroup_df
    return G


# Exemplo rápido de uso programático
if __name__ == "__main__":
    from core.config import make_default_config
    cfg = make_default_config(centroid_threshold=8.5)
    G = build_graph_with_config("/home/elementare/GithubProjects/pMHC_graph/Analysis/selected_strs_renumber/6zkx.trunc.fit_renum.pdb", cfg)

    # 1) coordenadas e distâncias
    coords = G.graph["coords"]                # (N, 3)
    labels = G.graph["residue_labels"]        # ordem alinhada
    D      = G.graph["distance_matrix"]       # (N, N)

    # 2) DataFrames
    raw_pdb_df = G.graph["raw_pdb_df"]        # todos os átomos
    pdb_df     = G.graph["pdb_df"]            # proteína, heavy atoms
    rgroup_df  = G.graph["rgroup_df"]         # grupo R (side chain)

    for l in labels:
        if "A:LYS:146" in l:
            print(l)
    # print(labels)

